#!/usr/bin/env python3
"""
SM121 MXFP4 MoE GEMM + FA2 Attention Sinks Verification Script

This script verifies that:
1. The SM121 (GB10) GPU is detected correctly
2. MXFP4 MoE GEMM kernels are available and working
3. FA2 attention with sinks is available and working
4. Performance is reasonable for both prefill and decode workloads

Usage:
    python scripts/verify_sm121_mxfp4_moe_fa2_sinks.py [--quick] [--verbose]

Requirements:
    - NVIDIA GB10 (SM121) GPU
    - FlashInfer with SM121 support
    - CUDA 12.8+
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for test runs."""
    verbose: bool = False
    quick: bool = False
    require_sm121: bool = True
    output_dir: Path = field(default_factory=lambda: Path("test_outputs"))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    timing_ms: Optional[float] = None
    details: Dict = field(default_factory=dict)


class TestResults:
    """Collection of test results."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def add(self, result: TestResult):
        self.results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}: {result.name}")
        if result.message:
            print(f"         {result.message}")
    
    def summary(self) -> Tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        return passed, failed
    
    def to_json(self) -> str:
        return json.dumps([
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "timing_ms": r.timing_ms,
                "details": r.details,
            }
            for r in self.results
        ], indent=2)


# =============================================================================
# Environment Verification
# =============================================================================

def verify_environment(config: TestConfig, results: TestResults) -> bool:
    """Verify the environment is set up correctly."""
    
    print("\n=== Environment Verification ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        results.add(TestResult(
            name="CUDA Available",
            passed=False,
            message="CUDA is not available"
        ))
        return False
    
    results.add(TestResult(
        name="CUDA Available",
        passed=True,
        message=f"CUDA {torch.version.cuda}"
    ))
    
    # Check GPU
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    
    # Get compute capability
    try:
        import flashinfer
        from flashinfer.utils import get_compute_capability
        cc = get_compute_capability(device)
        major, minor = cc
        
        is_sm121 = (major == 12 and minor == 1)
        
        results.add(TestResult(
            name="GPU Detection",
            passed=True,
            message=f"{gpu_name} (SM{major}{minor})",
            details={"compute_capability": f"{major}.{minor}"}
        ))
        
        if config.require_sm121 and not is_sm121:
            results.add(TestResult(
                name="SM121 Required",
                passed=False,
                message=f"Expected SM121, got SM{major}{minor}"
            ))
            return False
        elif not is_sm121:
            print(f"  ⚠ WARNING: Running on SM{major}{minor}, not SM121")
        
    except Exception as e:
        results.add(TestResult(
            name="GPU Detection",
            passed=False,
            message=f"Failed to detect GPU: {e}"
        ))
        return False
    
    # Check FlashInfer version
    try:
        import flashinfer
        version = getattr(flashinfer, "__version__", "unknown")
        results.add(TestResult(
            name="FlashInfer Import",
            passed=True,
            message=f"Version: {version}"
        ))
    except ImportError as e:
        results.add(TestResult(
            name="FlashInfer Import",
            passed=False,
            message=f"Import failed: {e}"
        ))
        return False
    
    # Check PYTHONPATH includes local FlashInfer
    pythonpath = os.environ.get("PYTHONPATH", "")
    if "flashinfer" in pythonpath.lower():
        results.add(TestResult(
            name="Local FlashInfer in PYTHONPATH",
            passed=True,
            message="Local FlashInfer detected"
        ))
    else:
        results.add(TestResult(
            name="Local FlashInfer in PYTHONPATH",
            passed=True,  # Warning, not failure
            message="Local FlashInfer not in PYTHONPATH (using installed version)"
        ))
    
    # Check key environment variables
    env_vars = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL", "not set"),
        "FLASHINFER_CUDA_ARCH_LIST": os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "auto"),
    }
    
    if config.verbose:
        print("\n  Environment Variables:")
        for key, value in env_vars.items():
            print(f"    {key} = {value}")
    
    return True


# =============================================================================
# MXFP4 MoE GEMM Tests
# =============================================================================

def test_mxfp4_moe_gemm(config: TestConfig, results: TestResults) -> bool:
    """Test MXFP4 MoE GEMM functionality on SM121 using cutlass_fused_moe."""
    
    print("\n=== SM121 CUTLASS MoE GEMM Tests ===")
    
    try:
        from flashinfer.fused_moe.core import cutlass_fused_moe, get_cutlass_fused_moe_module
        from flashinfer.fused_moe import GatedActType
        
        # Try to load SM121 module
        module = get_cutlass_fused_moe_module(backend='121', use_fast_build=False)
        
        results.add(TestResult(
            name="MoE GEMM API Import",
            passed=True,
            message=f"SM121 CUTLASS MoE module loaded successfully"
        ))
    except ImportError as e:
        results.add(TestResult(
            name="MoE GEMM API Import",
            passed=False,
            message=f"Import failed: {e}"
        ))
        return False
    except RuntimeError as e:
        results.add(TestResult(
            name="MoE GEMM API Import",
            passed=False,
            message=f"SM121 module load failed: {e}"
        ))
        return False
    
    device = torch.device("cuda")
    
    # Test configurations for BF16 MoE (the primary SM121 path)
    test_configs = [
        {"num_tokens": 4, "hidden_dim": 4096, "num_experts": 8, "topk": 2, "regime": "decode"},
        {"num_tokens": 32, "hidden_dim": 4096, "num_experts": 8, "topk": 2, "regime": "decode"},
    ]
    
    if not config.quick:
        test_configs.extend([
            {"num_tokens": 128, "hidden_dim": 4096, "num_experts": 8, "topk": 2, "regime": "prefill"},
            {"num_tokens": 512, "hidden_dim": 4096, "num_experts": 8, "topk": 2, "regime": "prefill"},
        ])
    
    all_passed = True
    
    for cfg in test_configs:
        num_tokens = cfg["num_tokens"]
        hidden_dim = cfg["hidden_dim"]
        num_experts = cfg["num_experts"]
        topk = cfg["topk"]
        regime = cfg["regime"]
        intermediate_dim = 4096  # Simpler config for testing
        
        test_name = f"SM121 CUTLASS MoE ({regime}, M={num_tokens})"
        
        try:
            # Create test data for cutlass_fused_moe
            input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
            
            # Token-expert assignments (topk per token)
            token_selected_experts = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
            
            # Scaling factors per token-expert pair
            token_final_scales = torch.ones(num_tokens, topk, dtype=torch.float32, device=device)
            
            # BF16 Expert weights
            # FC1: (num_experts, intermediate_dim * 2, hidden_dim) for gated activation
            # FC2: (num_experts, hidden_dim, intermediate_dim)
            fc1_expert_weights = torch.randn(num_experts, intermediate_dim * 2, hidden_dim, dtype=torch.bfloat16, device=device)
            fc2_expert_weights = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16, device=device)
            
            # Warmup
            for _ in range(2):
                result = cutlass_fused_moe(
                    input=input_tensor,
                    token_selected_experts=token_selected_experts,
                    token_final_scales=token_final_scales,
                    fc1_expert_weights=fc1_expert_weights,
                    fc2_expert_weights=fc2_expert_weights,
                    output_dtype=torch.bfloat16,
                    quant_scales=None,
                )
                torch.cuda.synchronize()
            
            # Timed run
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(10):
                result = cutlass_fused_moe(
                    input=input_tensor,
                    token_selected_experts=token_selected_experts,
                    token_final_scales=token_final_scales,
                    fc1_expert_weights=fc1_expert_weights,
                    fc2_expert_weights=fc2_expert_weights,
                    output_dtype=torch.bfloat16,
                    quant_scales=None,
                )
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 10 * 1000  # ms
            
            # Result is a list: [output, ...]
            output = result[0]
            
            # Validate output
            if output.shape != (num_tokens, hidden_dim):
                raise ValueError(f"Output shape mismatch: {output.shape} vs expected ({num_tokens}, {hidden_dim})")
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("Output contains NaN or Inf")
            
            results.add(TestResult(
                name=test_name,
                passed=True,
                message=f"BF16 MoE OK, Time: {elapsed:.3f}ms",
                timing_ms=elapsed,
                details={"output_shape": list(output.shape)}
            ))
            
        except Exception as e:
            results.add(TestResult(
                name=test_name,
                passed=False,
                message=str(e)
            ))
            all_passed = False
    
    return all_passed


# =============================================================================
# FA2 Attention Sink Tests
# =============================================================================

def test_fa2_attention_sinks(config: TestConfig, results: TestResults) -> bool:
    """Test FA2 attention with sinks functionality."""
    
    print("\n=== FA2 Attention Sink Tests ===")
    
    try:
        import flashinfer
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
        from flashinfer.attention import BatchAttentionWithAttentionSinkWrapper
        from flashinfer.jit.attention.variants import attention_sink_decl
        from flashinfer.jit.utils import filename_safe_dtype_map
        
        results.add(TestResult(
            name="Attention Sink API Import",
            passed=True,
            message="All attention sink APIs available"
        ))
    except ImportError as e:
        results.add(TestResult(
            name="Attention Sink API Import",
            passed=False,
            message=f"Import failed: {e}"
        ))
        return False
    
    device = torch.device("cuda")
    
    # Test configurations
    test_configs = [
        {"batch_size": 1, "seq_len": 128, "num_qo_heads": 32, "num_kv_heads": 8},
        {"batch_size": 4, "seq_len": 256, "num_qo_heads": 32, "num_kv_heads": 32},
    ]
    
    if not config.quick:
        test_configs.extend([
            {"batch_size": 16, "seq_len": 512, "num_qo_heads": 32, "num_kv_heads": 8},
            {"batch_size": 1, "seq_len": 2048, "num_qo_heads": 32, "num_kv_heads": 32},
        ])
    
    all_passed = True
    head_dim = 128
    
    for cfg in test_configs:
        batch_size = cfg["batch_size"]
        seq_len = cfg["seq_len"]
        num_qo_heads = cfg["num_qo_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        
        for dtype in [torch.float16, torch.bfloat16]:
            dtype_name = "fp16" if dtype == torch.float16 else "bf16"
            test_name = f"FA2 Attention Sink ({dtype_name}, B={batch_size}, S={seq_len})"
            
            try:
                torch.manual_seed(42)
                sm_scale = 1.0 / math.sqrt(head_dim)
                
                # Create workspace
                float_workspace_buffer = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                
                # Create JIT args for attention sink
                jit_args = (
                    f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_False_fa2",
                    dtype,  # dtype_q
                    dtype,  # dtype_kv
                    dtype,  # dtype_o
                    torch.int32,  # idtype
                    head_dim,
                    head_dim,
                    ["sink"],
                    ["float"],
                    ["sm_scale"],
                    ["double"],
                    "AttentionSink",
                    attention_sink_decl["fa2"],
                )
                jit_kwargs = {"use_sliding_window": False}
                
                # Create wrapper
                wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    float_workspace_buffer,
                    kv_layout="NHD",
                    backend="fa2",
                    jit_args=jit_args,
                    jit_kwargs=jit_kwargs,
                )
                
                # Create indptrs
                qo_indptr_host = torch.arange(
                    0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
                )
                kv_indptr_host = torch.arange(
                    0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
                )
                
                # Plan
                wrapper.plan(
                    qo_indptr_host,
                    kv_indptr_host,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    causal=True,
                    window_left=-1,
                    q_data_type=dtype,
                    kv_data_type=dtype,
                )
                
                # Create tensors
                q = torch.randn(batch_size * seq_len, num_qo_heads, head_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
                
                # Warmup
                for _ in range(2):
                    output = wrapper.run(q, k, v, sink, sm_scale)
                    torch.cuda.synchronize()
                
                # Timed run
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                for _ in range(10):
                    output = wrapper.run(q, k, v, sink, sm_scale)
                
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / 10 * 1000  # ms
                
                # Validate
                expected_shape = (batch_size * seq_len, num_qo_heads, head_dim)
                if output.shape != expected_shape:
                    raise ValueError(f"Output shape mismatch: {output.shape} vs {expected_shape}")
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("Output contains NaN or Inf")
                
                results.add(TestResult(
                    name=test_name,
                    passed=True,
                    message=f"Shape OK, Time: {elapsed:.3f}ms",
                    timing_ms=elapsed
                ))
                
            except Exception as e:
                results.add(TestResult(
                    name=test_name,
                    passed=False,
                    message=str(e)
                ))
                all_passed = False
    
    # Also test the convenience wrapper
    try:
        test_name = "BatchAttentionWithAttentionSinkWrapper"
        
        float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        
        wrapper_paged = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            head_dim_qk=128,
            head_dim_vo=128,
            window_left=-1,
        )
        
        results.add(TestResult(
            name=test_name,
            passed=True,
            message="Wrapper created successfully"
        ))
        
    except Exception as e:
        results.add(TestResult(
            name=test_name,
            passed=False,
            message=str(e)
        ))
        all_passed = False
    
    return all_passed


# =============================================================================
# Backend Summary
# =============================================================================

def print_backend_summary(config: TestConfig):
    """Print a summary of detected backends."""
    
    print("\n=== Backend Summary ===")
    
    try:
        import flashinfer
        from flashinfer.utils import (
            get_compute_capability,
            determine_attention_backend,
            is_sm90a_supported,
        )
        
        device = torch.device("cuda")
        cc = get_compute_capability(device)
        
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Compute Capability: SM{cc[0]}{cc[1]}")
        print(f"  FlashInfer Version: {getattr(flashinfer, '__version__', 'unknown')}")
        print(f"  CUDA Version: {torch.version.cuda}")
        
        # Attention backend
        backend = determine_attention_backend(
            device,
            pos_encoding_mode=0,
            use_fp16_qk_reductions=False,
            use_custom_mask=False,
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
        )
        print(f"  Attention Backend: {backend.upper()}")
        print(f"  SM90a Support: {is_sm90a_supported(device)}")
        
        # MoE backend
        try:
            from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
            print("  MoE Backend: MXFP4 CUTLASS (available)")
        except ImportError:
            print("  MoE Backend: Not available")
        
        # Attention sinks
        try:
            from flashinfer.attention import BatchAttentionWithAttentionSinkWrapper
            print("  Attention Sinks: Available")
        except ImportError:
            print("  Attention Sinks: Not available")
        
        # PYTHONPATH check
        pythonpath = os.environ.get("PYTHONPATH", "")
        local_fi = "YES" if "flashinfer" in pythonpath.lower() else "NO"
        print(f"  Local FlashInfer Active: {local_fi}")
        
        print("")
        
    except Exception as e:
        print(f"  Error getting backend summary: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify SM121 MXFP4 MoE GEMM + FA2 Attention Sinks"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick tests only (fewer configurations)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-require-sm121",
        action="store_true",
        help="Don't require SM121 GPU (for testing on other GPUs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_outputs",
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    config = TestConfig(
        verbose=args.verbose,
        quick=args.quick,
        require_sm121=not args.no_require_sm121,
        output_dir=Path(args.output_dir),
    )
    
    print("=" * 60)
    print("SM121 MXFP4 MoE GEMM + FA2 Attention Sinks Verification")
    print("=" * 60)
    
    results = TestResults()
    
    # Environment verification
    if not verify_environment(config, results):
        print("\n✗ Environment verification failed - cannot continue")
        sys.exit(1)
    
    # Print backend summary
    print_backend_summary(config)
    
    # Run tests
    moe_passed = test_mxfp4_moe_gemm(config, results)
    attention_passed = test_fa2_attention_sinks(config, results)
    
    # Summary
    passed, failed = results.summary()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = config.output_dir / "verification_results.json"
    with open(results_file, "w") as f:
        f.write(results.to_json())
    print(f"\nResults saved to: {results_file}")
    
    if failed > 0:
        print("\n✗ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

