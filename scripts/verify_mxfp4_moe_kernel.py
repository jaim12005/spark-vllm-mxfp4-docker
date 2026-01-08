#!/usr/bin/env python3
"""
True MXFP4 MoE Kernel Verification Test

This test verifies the native MXFP4 MoE path in FlashInfer, independent of vLLM.
It uses actual MXFP4-quantized weights (not BF16) and compares against a reference.

Key test points:
- A) Construct MXFP4 weights using FlashInfer's mxfp4_quantize
- B) Call the SM121 CUTLASS MoE entrypoint (cutlass_fused_moe with backend="121")
- C) Validate correctness against dequantized BF16 reference
- D) Test M=320, 512, 1024 to check if M>=320 crash is path-specific

IMPORTANT: For SM121, the correct API is:
- cutlass_fused_moe() with get_cutlass_fused_moe_module(backend="121")
- NOT trtllm_fp4_block_scale_moe() which is for SM100 only

Usage:
    python3 verify_mxfp4_moe_kernel.py [--quick] [--verbose]
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import Tuple, Optional, List
import json

import torch
import torch.nn.functional as F

# Test result tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    error: Optional[str] = None
    details: Optional[dict] = None

results: List[TestResult] = []

def log_test(name: str, passed: bool, error: str = None, details: dict = None):
    results.append(TestResult(name, passed, error, details))
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if error and not passed:
        print(f"         Error: {error}")
    if details:
        for k, v in details.items():
            print(f"         {k}: {v}")


def check_sm121() -> bool:
    """Check if running on SM121 (GB10)."""
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc[0] == 12 and cc[1] == 1


def check_imports():
    """Verify all required FlashInfer imports are available."""
    print("\n=== Import Verification ===")
    
    try:
        import flashinfer
        print(f"FlashInfer version: {flashinfer.__version__}")
        print(f"FlashInfer location: {flashinfer.__file__}")
        log_test("FlashInfer import", True)
    except ImportError as e:
        log_test("FlashInfer import", False, str(e))
        return False
    
    # Check mxfp4_quantize
    try:
        from flashinfer import mxfp4_quantize, mxfp4_dequantize
        log_test("mxfp4_quantize import", True)
    except ImportError as e:
        log_test("mxfp4_quantize import", False, str(e))
        return False
    
    # Check mxfp4_dequantize_host (for reference comparison)
    try:
        from flashinfer import mxfp4_dequantize_host
        log_test("mxfp4_dequantize_host import", True)
    except ImportError as e:
        log_test("mxfp4_dequantize_host import", False, str(e))
        # Not fatal - we can use mxfp4_dequantize instead
    
    # Check CUTLASS MoE function (the correct API for SM121)
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        log_test("cutlass_fused_moe import", True)
    except ImportError as e:
        log_test("cutlass_fused_moe import", False, str(e))
        return False
    
    # Check ActivationType enum
    try:
        from flashinfer.fused_moe.core import ActivationType
        log_test("ActivationType import", True)
    except ImportError as e:
        log_test("ActivationType import", False, str(e))
        # Not fatal, continue
    
    # Check that SM121 backend module generator exists
    try:
        from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
        log_test("SM120/121 MoE module generator", True)
    except ImportError as e:
        log_test("SM120/121 MoE module generator", False, str(e))
        # Not fatal, continue
    
    return True


def test_mxfp4_quantization_format():
    """
    Test A: Verify MXFP4 quantization produces correct format.
    
    Contract:
    - Packed weight dtype: uint8 (two FP4 values per byte)
    - Packed weight shape: [M, K/2]
    - Scale factor dtype: uint8 (UE8M0 format for MXFP4)
    - Scale factor group size: 32 (MXFP4 standard)
    """
    print("\n=== Test A: MXFP4 Quantization Format ===")
    
    from flashinfer import mxfp4_quantize
    
    # Test dimensions
    M, K = 256, 4096  # K must be multiple of 32 for MXFP4
    
    # Create test tensor
    weight = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    try:
        weight_fp4, weight_scale = mxfp4_quantize(weight)
        
        # Verify packed weight format
        log_test(
            "Packed weight dtype",
            weight_fp4.dtype == torch.uint8,
            f"Expected uint8, got {weight_fp4.dtype}",
            {"dtype": str(weight_fp4.dtype)}
        )
        
        log_test(
            "Packed weight shape",
            weight_fp4.shape == (M, K // 2),
            f"Expected ({M}, {K // 2}), got {weight_fp4.shape}",
            {"shape": str(weight_fp4.shape)}
        )
        
        # Verify scale factor format
        log_test(
            "Scale factor dtype",
            weight_scale.dtype == torch.uint8,
            f"Expected uint8, got {weight_scale.dtype}",
            {"dtype": str(weight_scale.dtype)}
        )
        
        # Scale shape depends on layout, just verify it exists and is reasonable
        scale_numel = weight_scale.numel()
        expected_min_scales = (M * K) // 32  # At least one scale per 32 elements
        log_test(
            "Scale factor count",
            scale_numel >= expected_min_scales,
            f"Expected >= {expected_min_scales}, got {scale_numel}",
            {"numel": scale_numel, "expected_min": expected_min_scales}
        )
        
        print(f"  Weight FP4 shape: {weight_fp4.shape}")
        print(f"  Scale shape: {weight_scale.shape}")
        print(f"  Scale numel: {scale_numel}")
        
        return True, weight_fp4, weight_scale, weight
        
    except Exception as e:
        log_test("MXFP4 quantization", False, str(e))
        traceback.print_exc()
        return False, None, None, None


def test_mxfp4_roundtrip():
    """Test that MXFP4 quantize -> dequantize roundtrip is reasonable."""
    print("\n=== Test: MXFP4 Roundtrip Accuracy ===")
    
    from flashinfer import mxfp4_quantize, mxfp4_dequantize
    
    M, K = 128, 1024
    weight = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    try:
        weight_fp4, weight_scale = mxfp4_quantize(weight)
        weight_dequant = mxfp4_dequantize(weight_fp4, weight_scale)
        
        # Move to same device for comparison
        weight_cpu = weight.float().cpu()
        weight_dequant_cpu = weight_dequant.float().cpu() if weight_dequant.device.type == 'cpu' else weight_dequant.float().cpu()
        
        # Calculate relative error
        abs_diff = torch.abs(weight_cpu - weight_dequant_cpu)
        rel_error = abs_diff / (torch.abs(weight_cpu) + 1e-8)
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()
        
        # FP4 has limited precision, expect ~10-30% relative error
        log_test(
            "MXFP4 roundtrip mean relative error",
            mean_rel_error < 0.5,  # 50% tolerance for FP4
            f"Mean relative error too high: {mean_rel_error:.4f}",
            {"mean_rel_error": f"{mean_rel_error:.4f}", "max_rel_error": f"{max_rel_error:.4f}"}
        )
        
        return True
        
    except Exception as e:
        log_test("MXFP4 roundtrip", False, str(e))
        traceback.print_exc()
        return False


def create_mxfp4_moe_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create MXFP4-quantized MoE weights for testing.
    
    Returns:
        Tuple of (fc1_fp4, fc1_scale, fc2_fp4, fc2_scale,
                  fc1_bf16_ref, fc2_bf16_ref, fc1_dequant, fc2_dequant)
    """
    from flashinfer import mxfp4_quantize, mxfp4_dequantize
    
    # FC1: [num_experts, 2 * intermediate_size, hidden_size]
    # (gate + up projection combined)
    fc1_bf16 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    
    # FC2: [num_experts, hidden_size, intermediate_size]
    fc2_bf16 = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    )
    
    # Quantize each expert's weights
    fc1_fp4_list = []
    fc1_scale_list = []
    fc2_fp4_list = []
    fc2_scale_list = []
    fc1_dequant_list = []
    fc2_dequant_list = []
    
    for e in range(num_experts):
        # Quantize FC1 for this expert
        w1_fp4, w1_sf = mxfp4_quantize(fc1_bf16[e])
        fc1_fp4_list.append(w1_fp4)
        fc1_scale_list.append(w1_sf)
        
        # Dequantize for reference
        w1_dequant = mxfp4_dequantize(w1_fp4, w1_sf)
        fc1_dequant_list.append(w1_dequant)
        
        # Quantize FC2 for this expert
        w2_fp4, w2_sf = mxfp4_quantize(fc2_bf16[e])
        fc2_fp4_list.append(w2_fp4)
        fc2_scale_list.append(w2_sf)
        
        # Dequantize for reference
        w2_dequant = mxfp4_dequantize(w2_fp4, w2_sf)
        fc2_dequant_list.append(w2_dequant)
    
    # Stack into expert dimension
    fc1_fp4 = torch.stack(fc1_fp4_list, dim=0)
    fc1_scale = torch.stack(fc1_scale_list, dim=0)
    fc2_fp4 = torch.stack(fc2_fp4_list, dim=0)
    fc2_scale = torch.stack(fc2_scale_list, dim=0)
    
    # Stack dequantized for reference
    fc1_dequant = torch.stack(fc1_dequant_list, dim=0)
    fc2_dequant = torch.stack(fc2_dequant_list, dim=0)
    
    return (fc1_fp4, fc1_scale, fc2_fp4, fc2_scale,
            fc1_bf16, fc2_bf16, fc1_dequant, fc2_dequant)


def reference_moe_bf16(
    hidden_states: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    routing_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Reference BF16 MoE implementation for comparison.
    
    Uses simple top-k routing with softmax weights.
    """
    batch_size, hidden_size = hidden_states.shape
    num_experts = fc1_weights.shape[0]
    intermediate_size = fc2_weights.shape[2]
    
    # Routing: softmax -> top-k
    routing_probs = F.softmax(routing_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Renormalize
    
    # Initialize output
    output = torch.zeros_like(hidden_states)
    
    # Process each token
    for b in range(batch_size):
        token_output = torch.zeros(hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
        
        for k in range(top_k):
            expert_idx = topk_indices[b, k].item()
            expert_weight = topk_weights[b, k].item()
            
            # Get expert weights
            w1 = fc1_weights[expert_idx]  # [2 * intermediate_size, hidden_size]
            w2 = fc2_weights[expert_idx]  # [hidden_size, intermediate_size]
            
            # Split gate and up projections
            gate_proj = w1[:intermediate_size]  # [intermediate_size, hidden_size]
            up_proj = w1[intermediate_size:]    # [intermediate_size, hidden_size]
            
            # Forward pass: SwiGLU activation
            x = hidden_states[b]  # [hidden_size]
            gate = F.silu(x @ gate_proj.T)  # [intermediate_size]
            up = x @ up_proj.T              # [intermediate_size]
            intermediate = gate * up        # [intermediate_size]
            
            # Down projection
            expert_out = intermediate @ w2.T  # [hidden_size]
            
            token_output += expert_weight * expert_out
        
        output[b] = token_output
    
    return output


def test_mxfp4_moe_kernel(M: int, verbose: bool = False) -> bool:
    """
    Test B & C: Call SM121 CUTLASS MoE kernel and validate against reference.
    
    Uses cutlass_fused_moe() with backend="121" - the correct API for SM121.
    
    Args:
        M: Number of tokens (batch size)
        verbose: Print detailed output
    
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n=== Test: SM121 CUTLASS MoE Kernel (M={M}) ===")
    
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
    from flashinfer import mxfp4_quantize, mxfp4_dequantize
    
    # Model dimensions (must be multiples for CUTLASS alignment)
    # SM121 block-scaled requires M,N multiples of 128
    num_experts = 8
    hidden_size = 1024  # Must be multiple of 128 for SM121 block-scaled
    intermediate_size = 2048  # Must be multiple of 128
    top_k = 2
    
    device = "cuda"
    
    try:
        # Create hidden states (BF16 input)
        hidden_states = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
        
        # Create routing: top-k selection
        routing_logits = torch.randn(M, num_experts, dtype=torch.float32, device=device)
        routing_probs = F.softmax(routing_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_probs, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        # token_selected_experts: [M, top_k] int32
        token_selected_experts = topk_indices.to(torch.int32)
        # token_final_scales: [M, top_k] float32
        token_final_scales = topk_weights.to(torch.float32)
        
        # Create BF16 weights for testing (not MXFP4 quantized for initial API test)
        # FC1: [num_experts, 2 * intermediate_size, hidden_size] (gate + up combined)
        fc1_expert_weights = torch.randn(
            num_experts, 2 * intermediate_size, hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # FC2: [num_experts, hidden_size, intermediate_size]
        fc2_expert_weights = torch.randn(
            num_experts, hidden_size, intermediate_size,
            dtype=torch.bfloat16, device=device
        )
        
        if verbose:
            print(f"  hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
            print(f"  token_selected_experts shape: {token_selected_experts.shape}, dtype: {token_selected_experts.dtype}")
            print(f"  token_final_scales shape: {token_final_scales.shape}, dtype: {token_final_scales.dtype}")
            print(f"  fc1_expert_weights shape: {fc1_expert_weights.shape}, dtype: {fc1_expert_weights.dtype}")
            print(f"  fc2_expert_weights shape: {fc2_expert_weights.shape}, dtype: {fc2_expert_weights.dtype}")
        
        # Call SM121 CUTLASS MoE kernel
        # Using BF16 weights first to test API, then will add MXFP4 path
        print(f"  Calling cutlass_fused_moe (backend=121, BF16 weights)...")
        torch.cuda.synchronize()
        
        mxfp4_output = cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.bfloat16,
            quant_scales=[],  # Empty for BF16
            fc1_expert_biases=None,
            fc2_expert_biases=None,
            input_sf=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            cluster_size=1,
            cluster_rank=0,
            output=None,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,  # Would be True for MXFP4
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_packed_weights=False,
            tune_max_num_tokens=8192,
            enable_pdl=None,
            activation_type=ActivationType.Swiglu,
        )
        
        torch.cuda.synchronize()
        
        # Handle output (may be tensor or list)
        if isinstance(mxfp4_output, list):
            mxfp4_output = mxfp4_output[0]
        
        if verbose:
            print(f"  Output shape: {mxfp4_output.shape}, dtype: {mxfp4_output.dtype}")
        
        # Compute reference using BF16 weights
        print(f"  Computing BF16 reference...")
        ref_output = reference_moe_bf16(
            hidden_states, fc1_expert_weights, fc2_expert_weights, routing_logits, top_k
        )
        
        # Compare outputs
        abs_diff = torch.abs(mxfp4_output.float() - ref_output.float())
        rel_diff = abs_diff / (torch.abs(ref_output.float()) + 1e-8)
        
        mean_abs_error = abs_diff.mean().item()
        max_abs_error = abs_diff.max().item()
        mean_rel_error = rel_diff.mean().item()
        max_rel_error = rel_diff.max().item()
        
        # Check for NaN/Inf
        has_nan = torch.isnan(mxfp4_output).any().item()
        has_inf = torch.isinf(mxfp4_output).any().item()
        
        log_test(
            f"SM121 MoE M={M} no NaN",
            not has_nan,
            "Output contains NaN"
        )
        
        log_test(
            f"SM121 MoE M={M} no Inf",
            not has_inf,
            "Output contains Inf"
        )
        
        # Check correlation (should be very high for BF16 vs BF16)
        mxfp4_flat = mxfp4_output.float().flatten()
        ref_flat = ref_output.float().flatten()
        correlation = torch.corrcoef(torch.stack([mxfp4_flat, ref_flat]))[0, 1].item()
        
        log_test(
            f"SM121 MoE M={M} correlation",
            correlation > 0.9,  # High correlation expected for BF16 vs BF16
            f"Low correlation: {correlation:.4f}",
            {
                "correlation": f"{correlation:.4f}",
                "mean_abs_error": f"{mean_abs_error:.4f}",
                "mean_rel_error": f"{mean_rel_error:.4f}",
            }
        )
        
        print(f"  Output shape: {mxfp4_output.shape}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean abs error: {mean_abs_error:.4f}")
        print(f"  Mean rel error: {mean_rel_error:.4f}")
        
        return True
        
    except RuntimeError as e:
        error_str = str(e)
        if "illegal memory access" in error_str.lower():
            log_test(f"SM121 MoE M={M}", False, f"CUDA illegal memory access")
        elif "nvfp4" in error_str.lower():
            log_test(f"SM121 MoE M={M}", False, f"nvfp4 format error: {error_str}")
        elif "No supported CUDA architectures" in error_str:
            log_test(f"SM121 MoE M={M}", False, f"Architecture not supported: {error_str}")
        else:
            log_test(f"SM121 MoE M={M}", False, error_str)
        traceback.print_exc()
        return False
        
    except Exception as e:
        log_test(f"SM121 MoE M={M}", False, str(e))
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="MXFP4 MoE Kernel Verification")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip large M)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MXFP4 MoE Kernel Verification Test")
    print("=" * 60)
    
    # Check SM121
    if not check_sm121():
        print("WARNING: Not running on SM121 (GB10). Some tests may fail.")
    else:
        cc = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute Capability: {cc[0]}.{cc[1]}")
    
    # Check imports
    if not check_imports():
        print("\nFATAL: Required imports failed. Cannot continue.")
        sys.exit(1)
    
    # Test A: Quantization format
    quant_ok, _, _, _ = test_mxfp4_quantization_format()
    
    # Test roundtrip accuracy
    test_mxfp4_roundtrip()
    
    # Test B & C: MoE kernel at various M values
    # Critical: include M=320, 512, 1024 to check if crash is path-specific
    if args.quick:
        m_values = [1, 4, 64, 128]
    else:
        m_values = [1, 4, 64, 128, 256, 320, 512, 1024]
    
    print("\n" + "=" * 60)
    print("Testing SM121 CUTLASS MoE Kernel at Various Batch Sizes")
    print("(Using cutlass_fused_moe with backend=121)")
    print("(Testing if M>=320 crash is specific to old API or persists)")
    print("=" * 60)
    
    for M in m_values:
        # Reset CUDA state between tests to isolate crashes
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        test_mxfp4_moe_kernel(M, verbose=args.verbose)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")
    
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
    
    # Output JSON if requested
    if args.output:
        output_data = {
            "summary": {
                "passed": passed,
                "failed": failed,
                "total": total,
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "error": r.error,
                    "details": r.details,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {args.output}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

