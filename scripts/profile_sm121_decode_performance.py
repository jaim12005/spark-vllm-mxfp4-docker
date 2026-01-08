#!/usr/bin/env python3
"""
SM121 Decode Performance Profiler

This script profiles decode performance on SM121 (GB10) to identify:
1. Top kernels by time during decode
2. FlashInfer decode kernel usage
3. MoE GEMM contribution to decode latency
4. CPU overhead (scheduling, IPC, Python)

Usage:
    python scripts/profile_sm121_decode_performance.py [--model MODEL] [--output-dir DIR]

For nsys profiling:
    nsys profile -o decode_profile python scripts/profile_sm121_decode_performance.py

Requirements:
    - NVIDIA GB10 (SM121) GPU
    - FlashInfer with SM121 support
    - cupti-python (optional, for CUPTI timing)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProfileConfig:
    """Configuration for profiling."""
    output_dir: Path = field(default_factory=lambda: Path("profile_outputs"))
    num_warmup: int = 5
    num_iterations: int = 50
    use_cupti: bool = True
    use_torch_profiler: bool = True
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    seq_lens: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048, 4096])
    hidden_dim: int = 4096
    num_experts: int = 8
    topk: int = 2
    num_qo_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128


@dataclass 
class ProfileResult:
    """Result of a profiling run."""
    name: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    kernel_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Timing Utilities
# =============================================================================

def bench_gpu_time_cuda_events(
    fn: Callable,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> Tuple[float, float]:
    """Benchmark GPU time using CUDA events."""
    
    # Warmup
    for _ in range(num_warmup):
        fn()
    
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn()
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    import numpy as np
    return np.median(times), np.std(times)


def bench_gpu_time_cupti(
    fn: Callable,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> Tuple[float, float]:
    """Benchmark GPU time using CUPTI (more accurate)."""
    
    try:
        from flashinfer.testing import bench_gpu_time
        return bench_gpu_time(fn, enable_cupti=True, num_iters=num_iterations, num_warmup=num_warmup)
    except ImportError:
        print("  Warning: CUPTI not available, falling back to CUDA events")
        return bench_gpu_time_cuda_events(fn, num_warmup, num_iterations)


@contextmanager
def torch_profile_context(output_dir: Path, name: str):
    """Context manager for torch profiler."""
    
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    
    with prof:
        yield prof
    
    # Export traces
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{name}_trace.json"
    prof.export_chrome_trace(str(trace_file))
    
    # Print summary
    print(f"\nKernel Summary for {name}:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# =============================================================================
# Benchmark Functions
# =============================================================================

def profile_moe_gemm_decode(config: ProfileConfig) -> List[ProfileResult]:
    """Profile MoE GEMM performance in decode-like scenarios."""
    
    print("\n=== Profiling MoE GEMM (Decode) ===")
    
    try:
        from flashinfer.fused_moe import (
            trtllm_fp4_block_scale_moe,
            trtllm_bf16_moe,
            RoutingMethodType,
            GatedActType,
        )
        has_moe = True
    except ImportError:
        print("  MoE GEMM not available")
        return []
    
    device = torch.device("cuda")
    results = []
    
    for batch_size in config.batch_sizes:
        num_tokens = batch_size  # In decode, 1 token per request
        hidden_dim = config.hidden_dim
        intermediate_dim = 4 * hidden_dim
        num_experts = config.num_experts
        topk = config.topk
        
        print(f"\n  Batch size: {batch_size}")
        
        # Create test data
        routing_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
        hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        
        # FP4 weights
        gemm1_weights = torch.randint(
            0, 256, 
            (num_experts, 2 * intermediate_dim, hidden_dim // 2),
            dtype=torch.uint8, device=device
        )
        gemm2_weights = torch.randint(
            0, 256,
            (num_experts, hidden_dim, intermediate_dim // 2),
            dtype=torch.uint8, device=device
        )
        
        gemm1_scales = torch.ones(
            num_experts, 2 * intermediate_dim, hidden_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
        gemm2_scales = torch.ones(
            num_experts, hidden_dim, intermediate_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
        
        hidden_states_scale = torch.ones(
            num_tokens, hidden_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
        
        def run_moe():
            return trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_scales,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_scales,
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=num_experts,
                top_k=topk,
                n_group=None,
                topk_group=None,
                intermediate_size=intermediate_dim,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                routing_method_type=int(RoutingMethodType.Default),
                do_finalize=True,
                gated_act_type=int(GatedActType.Silu),
            )[0]
        
        try:
            if config.use_cupti:
                median_ms, std_ms = bench_gpu_time_cupti(run_moe, config.num_warmup, config.num_iterations)
            else:
                median_ms, std_ms = bench_gpu_time_cuda_events(run_moe, config.num_warmup, config.num_iterations)
            
            print(f"    MXFP4 MoE GEMM: {median_ms:.3f} ± {std_ms:.3f} ms")
            
            results.append(ProfileResult(
                name=f"moe_gemm_decode_bs{batch_size}",
                avg_time_ms=median_ms,
                std_time_ms=std_ms,
                min_time_ms=median_ms - 2*std_ms,
                max_time_ms=median_ms + 2*std_ms,
                metadata={
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "num_experts": num_experts,
                    "topk": topk,
                }
            ))
            
        except Exception as e:
            print(f"    Error: {e}")
    
    return results


def profile_attention_decode(config: ProfileConfig) -> List[ProfileResult]:
    """Profile attention performance in decode scenarios."""
    
    print("\n=== Profiling Attention (Decode) ===")
    
    try:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        from flashinfer.utils import get_compute_capability, determine_attention_backend
        
        device = torch.device("cuda")
        backend = determine_attention_backend(
            device, 0, False, False, torch.bfloat16, torch.bfloat16
        )
        print(f"  Using backend: {backend}")
        
    except ImportError as e:
        print(f"  Attention not available: {e}")
        return []
    
    results = []
    
    for batch_size in config.batch_sizes:
        for kv_len in [512, 1024, 2048, 4096]:
            print(f"\n  Batch: {batch_size}, KV len: {kv_len}")
            
            num_qo_heads = config.num_qo_heads
            num_kv_heads = config.num_kv_heads
            head_dim = config.head_dim
            page_size = 16
            num_pages_per_seq = (kv_len + page_size - 1) // page_size
            
            try:
                # Create workspace
                workspace_buffer = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    workspace_buffer,
                    kv_layout="NHD",
                )
                
                # Create paged KV cache
                total_pages = batch_size * num_pages_per_seq
                k_cache = torch.randn(
                    total_pages, page_size, num_kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device
                )
                v_cache = torch.randn(
                    total_pages, page_size, num_kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device
                )
                
                # Query (one per request)
                q = torch.randn(
                    batch_size, num_qo_heads, head_dim,
                    dtype=torch.bfloat16, device=device
                )
                
                # Page table
                kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages_per_seq
                kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
                kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
                
                # Plan
                wrapper.plan(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    data_type=torch.bfloat16,
                )
                
                def run_decode():
                    return wrapper.run(q, (k_cache, v_cache))
                
                if config.use_cupti:
                    median_ms, std_ms = bench_gpu_time_cupti(run_decode, config.num_warmup, config.num_iterations)
                else:
                    median_ms, std_ms = bench_gpu_time_cuda_events(run_decode, config.num_warmup, config.num_iterations)
                
                print(f"    Decode attention: {median_ms:.3f} ± {std_ms:.3f} ms")
                
                results.append(ProfileResult(
                    name=f"attention_decode_bs{batch_size}_kv{kv_len}",
                    avg_time_ms=median_ms,
                    std_time_ms=std_ms,
                    min_time_ms=median_ms - 2*std_ms,
                    max_time_ms=median_ms + 2*std_ms,
                    metadata={
                        "batch_size": batch_size,
                        "kv_len": kv_len,
                        "num_qo_heads": num_qo_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": head_dim,
                        "backend": backend,
                    }
                ))
                
            except Exception as e:
                print(f"    Error: {e}")
    
    return results


def profile_with_torch_profiler(config: ProfileConfig) -> Dict[str, Any]:
    """Run detailed profiling with torch profiler."""
    
    print("\n=== Torch Profiler Analysis ===")
    
    device = torch.device("cuda")
    results = {}
    
    # Profile MoE GEMM
    try:
        from flashinfer.fused_moe import (
            trtllm_fp4_block_scale_moe,
            RoutingMethodType,
            GatedActType,
        )
        
        batch_size = 8
        hidden_dim = config.hidden_dim
        intermediate_dim = 4 * hidden_dim
        num_experts = config.num_experts
        topk = config.topk
        
        routing_logits = torch.randn(batch_size, num_experts, dtype=torch.bfloat16, device=device)
        hidden_states = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)
        gemm1_weights = torch.randint(0, 256, (num_experts, 2*intermediate_dim, hidden_dim//2), dtype=torch.uint8, device=device)
        gemm2_weights = torch.randint(0, 256, (num_experts, hidden_dim, intermediate_dim//2), dtype=torch.uint8, device=device)
        gemm1_scales = torch.ones(num_experts, 2*intermediate_dim, hidden_dim//32, dtype=torch.float8_e4m3fn, device=device)
        gemm2_scales = torch.ones(num_experts, hidden_dim, intermediate_dim//32, dtype=torch.float8_e4m3fn, device=device)
        hidden_states_scale = torch.ones(batch_size, hidden_dim//32, dtype=torch.float8_e4m3fn, device=device)
        
        def run_moe():
            return trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_scales,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_scales,
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=num_experts,
                top_k=topk,
                n_group=None,
                topk_group=None,
                intermediate_size=intermediate_dim,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                routing_method_type=int(RoutingMethodType.Default),
                do_finalize=True,
                gated_act_type=int(GatedActType.Silu),
            )[0]
        
        # Warmup
        for _ in range(5):
            run_moe()
            torch.cuda.synchronize()
        
        # Profile
        with torch_profile_context(config.output_dir, "moe_decode") as prof:
            for _ in range(20):
                run_moe()
                torch.cuda.synchronize()
        
        # Extract top kernels
        kernel_times = {}
        for evt in prof.key_averages():
            if evt.key and evt.cuda_time_total > 0:
                kernel_times[evt.key] = evt.cuda_time_total / 1000  # to ms
        
        results["moe_kernels"] = dict(sorted(kernel_times.items(), key=lambda x: -x[1])[:10])
        
    except Exception as e:
        print(f"  MoE profiling error: {e}")
    
    return results


# =============================================================================
# Performance Analysis
# =============================================================================

def analyze_results(
    moe_results: List[ProfileResult],
    attention_results: List[ProfileResult],
    config: ProfileConfig,
) -> Dict[str, Any]:
    """Analyze profiling results and provide recommendations."""
    
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)
    
    analysis = {
        "bottlenecks": [],
        "recommendations": [],
        "summary": {},
    }
    
    # Analyze MoE performance
    if moe_results:
        print("\n--- MoE GEMM Analysis ---")
        
        # Find decode-specific issues
        small_batch_results = [r for r in moe_results if r.metadata.get("batch_size", 0) <= 4]
        large_batch_results = [r for r in moe_results if r.metadata.get("batch_size", 0) >= 16]
        
        if small_batch_results:
            avg_small = sum(r.avg_time_ms for r in small_batch_results) / len(small_batch_results)
            print(f"  Small batch (1-4) avg: {avg_small:.3f} ms")
            
            # For decode, MoE should be very fast (< 1ms for batch 1)
            bs1_results = [r for r in moe_results if r.metadata.get("batch_size") == 1]
            if bs1_results and bs1_results[0].avg_time_ms > 1.0:
                analysis["bottlenecks"].append({
                    "component": "MoE GEMM",
                    "issue": f"Batch-1 latency ({bs1_results[0].avg_time_ms:.2f}ms) > 1ms threshold",
                    "severity": "high" if bs1_results[0].avg_time_ms > 2.0 else "medium",
                })
                analysis["recommendations"].append(
                    "Check MoE tile selection for small M - may need decode-optimized tiles"
                )
    
    # Analyze attention performance
    if attention_results:
        print("\n--- Attention Analysis ---")
        
        for result in attention_results:
            kv_len = result.metadata.get("kv_len", 0)
            batch_size = result.metadata.get("batch_size", 0)
            print(f"  BS={batch_size}, KV={kv_len}: {result.avg_time_ms:.3f} ms")
            
            # Check for attention scaling issues
            if kv_len >= 4096 and result.avg_time_ms > 5.0:
                analysis["bottlenecks"].append({
                    "component": "Attention",
                    "issue": f"Long context ({kv_len}) decode slow ({result.avg_time_ms:.2f}ms)",
                    "severity": "medium",
                })
    
    # Generate recommendations
    print("\n--- Recommendations ---")
    
    recommendations = [
        {
            "category": "vLLM Runtime",
            "items": [
                "Ensure --enforce-eager=False for CUDA graph support",
                "Tune --max-num-batched-tokens for your workload",
                "Verify prefix caching is enabled if prompts share prefixes",
                "Check --gpu-memory-utilization (higher = more KV cache)",
            ],
        },
        {
            "category": "FlashInfer",
            "items": [
                "Verify FA2 decode path is engaged (not just prefill)",
                "Check KV layout matches expectations (HND vs NHD)",
                "Ensure JIT cache is warmed up (first run is slow)",
            ],
        },
        {
            "category": "CPU/IPC Overhead",
            "items": [
                "If decode is latency-bound, test local client bypassing HTTP",
                "Check ZMQ polling overhead in vLLM engine",
                "Profile Python orchestration time between tokens",
            ],
        },
    ]
    
    for rec in recommendations:
        print(f"\n  {rec['category']}:")
        for item in rec["items"]:
            print(f"    • {item}")
    
    analysis["recommendations"] = recommendations
    
    return analysis


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SM121 Decode Performance Profiler")
    parser.add_argument("--output-dir", type=str, default="profile_outputs")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--no-cupti", action="store_true", help="Disable CUPTI timing")
    parser.add_argument("--no-torch-profiler", action="store_true", help="Skip torch profiler")
    parser.add_argument("--quick", action="store_true", help="Quick profile with fewer configs")
    
    args = parser.parse_args()
    
    config = ProfileConfig(
        output_dir=Path(args.output_dir),
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        use_cupti=not args.no_cupti,
        use_torch_profiler=not args.no_torch_profiler,
    )
    
    if args.quick:
        config.batch_sizes = [1, 8]
        config.seq_lens = [512, 2048]
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SM121 Decode Performance Profiler")
    print("=" * 60)
    
    # Check device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Output: {config.output_dir}")
    
    # Run profiling
    moe_results = profile_moe_gemm_decode(config)
    attention_results = profile_attention_decode(config)
    
    # Torch profiler analysis
    if config.use_torch_profiler:
        torch_results = profile_with_torch_profiler(config)
    else:
        torch_results = {}
    
    # Analyze
    analysis = analyze_results(moe_results, attention_results, config)
    
    # Save results
    all_results = {
        "moe_results": [
            {
                "name": r.name,
                "avg_time_ms": r.avg_time_ms,
                "std_time_ms": r.std_time_ms,
                "metadata": r.metadata,
            }
            for r in moe_results
        ],
        "attention_results": [
            {
                "name": r.name,
                "avg_time_ms": r.avg_time_ms,
                "std_time_ms": r.std_time_ms,
                "metadata": r.metadata,
            }
            for r in attention_results
        ],
        "torch_profiler": torch_results,
        "analysis": analysis,
    }
    
    output_file = config.output_dir / "profile_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print CSV summary for easy comparison
    print("\n--- CSV Summary (for benchmarking) ---")
    print("component,config,avg_ms,std_ms")
    for r in moe_results:
        print(f"moe_gemm,bs{r.metadata.get('batch_size', '?')},{r.avg_time_ms:.3f},{r.std_time_ms:.3f}")
    for r in attention_results:
        print(f"attention,bs{r.metadata.get('batch_size', '?')}_kv{r.metadata.get('kv_len', '?')},{r.avg_time_ms:.3f},{r.std_time_ms:.3f}")


if __name__ == "__main__":
    main()

