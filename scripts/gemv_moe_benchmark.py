#!/usr/bin/env python3
"""
GEMV MoE Fallback Benchmark

Compares CUTLASS grouped GEMM vs Python GEMV fallback for small batch sizes.
This tests the hypothesis that GEMV is faster for M=1 decode.

Usage:
    docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
        python3 /workspace/scripts/gemv_moe_benchmark.py
"""
import os
import sys
import time
from typing import List, Optional

# Setup paths
os.environ.setdefault("PYTHONPATH", "/workspace/flashinfer:/workspace/vllm")
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch
import torch.nn.functional as F

# Check GPU
device = torch.device("cuda:0")
torch.cuda.set_device(device)

print("=" * 70)
print("GEMV MoE Fallback Benchmark")
print("=" * 70)
print(f"Device: {torch.cuda.get_device_name()}")
print()


# ============================================================================
# GEMV Fallback Implementation
# ============================================================================

def moe_gemv_fallback(
    input_bf16: torch.Tensor,               # [M, hidden_dim] BF16
    token_selected_experts: torch.Tensor,   # [M, topk] int
    token_final_scales: torch.Tensor,       # [M, topk] float32
    fc1_weights: torch.Tensor,              # [num_experts, intermediate*2, hidden_dim] BF16
    fc2_weights: torch.Tensor,              # [num_experts, hidden_dim, intermediate] BF16
) -> torch.Tensor:
    """
    GEMV-based MoE for small batch sizes.
    
    This is a simple Python implementation using torch.mv for matrix-vector multiply.
    For M=1, this avoids the massive tile waste of grouped GEMM.
    """
    M, hidden_dim = input_bf16.shape
    topk = token_selected_experts.shape[1]
    
    output = torch.zeros(M, hidden_dim, dtype=torch.bfloat16, device=input_bf16.device)
    
    for token_idx in range(M):
        token_input = input_bf16[token_idx]  # [hidden_dim]
        
        for k in range(topk):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()
            
            # FC1: [intermediate*2, hidden_dim] @ [hidden_dim] → [intermediate*2]
            fc1_out = torch.mv(fc1_weights[expert_idx], token_input)
            
            # SwiGLU activation
            intermediate_dim = fc1_out.shape[0] // 2
            gate = fc1_out[:intermediate_dim]
            up = fc1_out[intermediate_dim:]
            activated = F.silu(gate) * up  # [intermediate_dim]
            
            # FC2: [hidden_dim, intermediate] @ [intermediate] → [hidden_dim]
            fc2_out = torch.mv(fc2_weights[expert_idx], activated)
            
            output[token_idx] += scale * fc2_out
    
    return output


def moe_gemv_batched(
    input_bf16: torch.Tensor,               # [M, hidden_dim] BF16
    token_selected_experts: torch.Tensor,   # [M, topk] int
    token_final_scales: torch.Tensor,       # [M, topk] float32
    fc1_weights: torch.Tensor,              # [num_experts, intermediate*2, hidden_dim] BF16
    fc2_weights: torch.Tensor,              # [num_experts, hidden_dim, intermediate] BF16
) -> torch.Tensor:
    """
    Batched GEMV using torch.bmm - more efficient than loop.
    """
    M, hidden_dim = input_bf16.shape
    topk = token_selected_experts.shape[1]
    num_experts = fc1_weights.shape[0]
    intermediate_dim = fc1_weights.shape[1] // 2
    
    output = torch.zeros(M, hidden_dim, dtype=torch.bfloat16, device=input_bf16.device)
    
    # Flatten expert selection for batched processing
    # Each token selects topk experts, so we have M*topk expert calls
    flat_experts = token_selected_experts.view(-1)  # [M*topk]
    flat_scales = token_final_scales.view(-1)       # [M*topk]
    
    # Expand input for each expert selection
    # [M, hidden_dim] → [M, topk, hidden_dim] → [M*topk, hidden_dim]
    expanded_input = input_bf16.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_dim)
    
    # Gather FC1 weights for selected experts: [M*topk, intermediate*2, hidden_dim]
    fc1_selected = fc1_weights[flat_experts]
    
    # FC1: batched matrix-vector multiply
    # [M*topk, intermediate*2, hidden_dim] @ [M*topk, hidden_dim, 1] → [M*topk, intermediate*2, 1]
    fc1_out = torch.bmm(fc1_selected, expanded_input.unsqueeze(-1)).squeeze(-1)  # [M*topk, intermediate*2]
    
    # SwiGLU
    gate = fc1_out[:, :intermediate_dim]
    up = fc1_out[:, intermediate_dim:]
    activated = F.silu(gate) * up  # [M*topk, intermediate]
    
    # Gather FC2 weights for selected experts: [M*topk, hidden_dim, intermediate]
    fc2_selected = fc2_weights[flat_experts]
    
    # FC2: batched matrix-vector multiply
    fc2_out = torch.bmm(fc2_selected, activated.unsqueeze(-1)).squeeze(-1)  # [M*topk, hidden_dim]
    
    # Apply scales and sum across experts
    fc2_out = fc2_out * flat_scales.unsqueeze(-1).to(fc2_out.dtype)
    
    # Reshape and sum: [M*topk, hidden_dim] → [M, topk, hidden_dim] → [M, hidden_dim]
    output = fc2_out.view(M, topk, hidden_dim).sum(dim=1)
    
    return output


# ============================================================================
# Benchmark Setup
# ============================================================================

def create_test_tensors(M: int, num_experts: int = 128, hidden_dim: int = 2944, 
                        intermediate_dim: int = 5888, topk: int = 8):
    """Create test tensors matching gpt-oss dimensions."""
    
    # Input activation
    input_bf16 = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device=device)
    
    # Expert selection (random experts)
    token_selected_experts = torch.randint(0, num_experts, (M, topk), device=device)
    
    # Routing weights (sum to 1 per token)
    token_final_scales = torch.softmax(
        torch.randn(M, topk, device=device), dim=-1
    ).float()
    
    # FC1 weights: [num_experts, intermediate*2, hidden_dim] for SwiGLU
    fc1_weights = torch.randn(
        num_experts, intermediate_dim * 2, hidden_dim, 
        dtype=torch.bfloat16, device=device
    ) * 0.02
    
    # FC2 weights: [num_experts, hidden_dim, intermediate]
    fc2_weights = torch.randn(
        num_experts, hidden_dim, intermediate_dim,
        dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return input_bf16, token_selected_experts, token_final_scales, fc1_weights, fc2_weights


def benchmark_gemv(M: int, num_iters: int = 20, warmup: int = 3):
    """Benchmark GEMV implementations."""
    
    print(f"\n--- Benchmarking M={M} ---")
    
    # Create tensors
    input_bf16, experts, scales, fc1, fc2 = create_test_tensors(M)
    
    # Memory usage
    weight_mem_gb = (fc1.numel() + fc2.numel()) * 2 / 1e9  # BF16 = 2 bytes
    print(f"Weight memory: {weight_mem_gb:.2f} GB")
    
    results = {}
    
    # Benchmark 1: Simple GEMV loop
    print("  Testing: GEMV loop...", end=" ", flush=True)
    try:
        # Warmup
        for _ in range(warmup):
            out = moe_gemv_fallback(input_bf16, experts, scales, fc1, fc2)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            out = moe_gemv_fallback(input_bf16, experts, scales, fc1, fc2)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        time_per_call_ms = (elapsed / num_iters) * 1000
        results["gemv_loop"] = time_per_call_ms
        print(f"{time_per_call_ms:.3f} ms")
    except Exception as e:
        print(f"FAILED: {e}")
        results["gemv_loop"] = None
    
    # Benchmark 2: Batched GEMV (torch.bmm)
    print("  Testing: GEMV batched (bmm)...", end=" ", flush=True)
    try:
        # Warmup
        for _ in range(warmup):
            out = moe_gemv_batched(input_bf16, experts, scales, fc1, fc2)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            out = moe_gemv_batched(input_bf16, experts, scales, fc1, fc2)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        time_per_call_ms = (elapsed / num_iters) * 1000
        results["gemv_batched"] = time_per_call_ms
        print(f"{time_per_call_ms:.3f} ms")
    except Exception as e:
        print(f"FAILED: {e}")
        results["gemv_batched"] = None
    
    return results


def benchmark_cutlass(M: int, num_iters: int = 20, warmup: int = 3):
    """Benchmark CUTLASS via vLLM wrapper."""
    
    print(f"  Testing: CUTLASS (vLLM wrapper)...", end=" ", flush=True)
    
    try:
        from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
        from flashinfer import mxfp8_quantize
        
        # Create tensors matching vLLM's expected format
        num_experts = 128
        hidden_dim = 2944
        intermediate_dim = 5888
        topk = 8
        
        # BF16 activation
        input_bf16 = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device=device)
        
        # Quantize to MXFP8
        input_fp8, input_scale = mxfp8_quantize(input_bf16, True, 32)
        
        # Expert selection
        token_selected_experts = torch.randint(0, num_experts, (M, topk), dtype=torch.int, device=device)
        token_final_scales = torch.softmax(torch.randn(M, topk, device=device), dim=-1).float()
        
        # MXFP4 weights (packed uint8)
        # FC1: [experts, intermediate*2, hidden/2]
        fc1_weights = torch.randint(0, 256, (num_experts, intermediate_dim * 2, hidden_dim // 2), 
                                    dtype=torch.uint8, device=device)
        # FC2: [experts, hidden, intermediate/2]
        fc2_weights = torch.randint(0, 256, (num_experts, hidden_dim, intermediate_dim // 2),
                                    dtype=torch.uint8, device=device)
        
        # Scales
        fc1_scales = torch.randint(0, 256, (num_experts, intermediate_dim * 2, hidden_dim // 32),
                                   dtype=torch.uint8, device=device)
        fc2_scales = torch.randint(0, 256, (num_experts, hidden_dim, intermediate_dim // 32),
                                   dtype=torch.uint8, device=device)
        
        fake_input_scale = torch.ones(num_experts, device=device)
        quant_scales = [
            fc1_scales.contiguous().view(torch.int32),
            fake_input_scale,
            fc2_scales.contiguous().view(torch.int32),
            fake_input_scale,
        ]
        
        output = torch.empty(M, hidden_dim, dtype=torch.bfloat16, device=device)
        
        # Warmup
        for _ in range(warmup):
            flashinfer_cutlass_fused_moe(
                input=input_fp8,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                fc1_expert_weights=fc1_weights.contiguous().view(torch.long),
                fc2_expert_weights=fc2_weights.contiguous().view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=quant_scales,
                use_mxfp8_act_scaling=True,
                input_sf=input_scale,
                output=output,
                activation_type=1,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            flashinfer_cutlass_fused_moe(
                input=input_fp8,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                fc1_expert_weights=fc1_weights.contiguous().view(torch.long),
                fc2_expert_weights=fc2_weights.contiguous().view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=quant_scales,
                use_mxfp8_act_scaling=True,
                input_sf=input_scale,
                output=output,
                activation_type=1,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        time_per_call_ms = (elapsed / num_iters) * 1000
        print(f"{time_per_call_ms:.3f} ms")
        return time_per_call_ms
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    print("\nBenchmarking MoE kernels at different batch sizes (M)")
    print("gpt-oss config: 128 experts, 2944 hidden, 5888 intermediate, topk=8")
    print()
    
    # Test sizes
    test_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    all_results = {}
    
    for M in test_sizes:
        results = benchmark_gemv(M)
        cutlass_time = benchmark_cutlass(M)
        results["cutlass"] = cutlass_time
        all_results[M] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Time per MoE call (ms)")
    print("=" * 70)
    print(f"{'M':>6} | {'GEMV Loop':>12} | {'GEMV Batched':>12} | {'CUTLASS':>12} | {'Best':>12}")
    print("-" * 70)
    
    for M, results in all_results.items():
        gemv_loop = results.get("gemv_loop")
        gemv_batch = results.get("gemv_batched")
        cutlass = results.get("cutlass")
        
        times = [(t, n) for t, n in [
            (gemv_loop, "GEMV Loop"), 
            (gemv_batch, "GEMV Batch"), 
            (cutlass, "CUTLASS")
        ] if t is not None]
        
        if times:
            best_time, best_name = min(times, key=lambda x: x[0])
        else:
            best_name = "N/A"
        
        gemv_loop_str = f"{gemv_loop:.3f}" if gemv_loop else "N/A"
        gemv_batch_str = f"{gemv_batch:.3f}" if gemv_batch else "N/A"
        cutlass_str = f"{cutlass:.3f}" if cutlass else "N/A"
        
        print(f"{M:>6} | {gemv_loop_str:>12} | {gemv_batch_str:>12} | {cutlass_str:>12} | {best_name:>12}")
    
    # Calculate tok/s for M=1 decode
    print("\n" + "=" * 70)
    print("Decode Performance Projection (M=1, 60 layers × 2 MoE = 120 calls/token)")
    print("=" * 70)
    
    if 1 in all_results:
        results = all_results[1]
        for name, time_ms in results.items():
            if time_ms:
                per_token_ms = time_ms * 120
                tok_per_sec = 1000 / per_token_ms
                print(f"{name:>15}: {time_ms:.3f} ms/call → {per_token_ms:.1f} ms/token → {tok_per_sec:.1f} tok/s")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

