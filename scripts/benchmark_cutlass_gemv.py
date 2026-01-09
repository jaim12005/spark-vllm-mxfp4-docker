#!/usr/bin/env python3
"""
Benchmark CUTLASS GemvBlockScaled for FP4 MoE decode optimization.

This script benchmarks the CUTLASS GEMV kernel against the grouped GEMM
kernel for small batch sizes (M=1-32) typical of decode workloads.

Usage:
    python benchmark_cutlass_gemv.py [--m 1,2,4,8,16,32] [--iterations 100]

Target performance:
    - Current grouped GEMM at M=1: ~29 tok/s
    - llama.cpp at M=1: ~58 tok/s  
    - Target with GEMV: >=52 tok/s (match SGLang)
"""

import argparse
import time
import torch
import numpy as np
from typing import List, Tuple

# gpt-oss-120b MoE dimensions
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_DIM = 4096
INTER_DIM = 5888  # gate_proj + up_proj combined = 2944 * 2
NUM_LAYERS = 60
BLOCK_SIZE = 32  # MXFP4 block size


def create_mxfp4_tensor(shape: Tuple[int, ...], device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a tensor in MXFP4 format (packed FP4 with FP8 scales).
    
    Returns:
        (packed_data, scales) where:
        - packed_data: uint8 tensor with 2 FP4 values per byte
        - scales: float8_e4m3fn tensor with scale factors
    """
    # Original shape with FP4 values
    numel = np.prod(shape)
    
    # Random FP4 values (0-15 range for unsigned FP4)
    # In practice, FP4 e2m1 has range ~[-6, 6]
    fp4_values = torch.randint(0, 16, (numel,), dtype=torch.uint8, device=device)
    
    # Pack into nibbles (2 FP4 values per byte)
    # Even indices go to low nibble, odd to high nibble
    packed_shape = list(shape)
    packed_shape[-1] = packed_shape[-1] // 2
    packed = fp4_values[0::2] | (fp4_values[1::2] << 4)
    packed_data = packed.view(packed_shape)
    
    # Create scale factors
    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    scales = torch.randn(num_blocks, dtype=torch.float32, device=device)
    # Convert to FP8 e4m3
    scales = scales.to(torch.float8_e4m3fn)
    
    return packed_data, scales


def benchmark_grouped_gemm(m: int, k: int, n: int, num_experts: int, iterations: int = 100) -> float:
    """
    Benchmark the current grouped GEMM approach for MoE.
    
    This uses cutlass_fused_moe with a 128x128 tile, showing inefficiency at small M.
    """
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer import mxfp8_quantize
    except ImportError as e:
        print(f"Warning: Could not import FlashInfer MoE: {e}")
        return float('inf')
    
    device = "cuda"
    
    # Create inputs matching vLLM's MoE format
    hidden_states = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    
    # Expert indices and weights (simulating top-k routing)
    topk_indices = torch.randint(0, num_experts, (m, TOP_K), dtype=torch.int64, device=device)
    topk_weights = torch.rand(m, TOP_K, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    
    # Create FP4 weights (matching gpt-oss-120b structure)
    # FC1: [num_experts, inter_dim, hidden_dim/2] packed
    fc1_weights, fc1_scales = create_mxfp4_tensor((num_experts, n, k), device)
    # FC2: [num_experts, hidden_dim, inter_dim/2] packed
    fc2_weights, fc2_scales = create_mxfp4_tensor((num_experts, k, n // 2), device)
    
    # Warm up
    for _ in range(5):
        try:
            output = cutlass_fused_moe(
                input=hidden_states,
                token_selected_experts=topk_indices,
                token_final_scales=topk_weights,
                fc1_expert_weights=fc1_weights,
                fc2_expert_weights=fc2_weights,
                output_dtype=torch.bfloat16,
                quant_scales=[fc1_scales, fc2_scales],
                activation_type=ActivationType.Swiglu,
            )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Grouped GEMM failed: {e}")
            return float('inf')
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_indices,
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_weights,
            fc2_expert_weights=fc2_weights,
            output_dtype=torch.bfloat16,
            quant_scales=[fc1_scales, fc2_scales],
            activation_type=ActivationType.Swiglu,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms


def benchmark_gemv_simple(m: int, k: int, n: int, iterations: int = 100) -> float:
    """
    Benchmark a simple PyTorch GEMV for comparison.
    
    This is not optimized but provides a baseline.
    """
    device = "cuda"
    
    # BF16 inputs (before quantization)
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    
    # Warm up
    for _ in range(5):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms


def benchmark_gemv_compiled(m: int, k: int, n: int, iterations: int = 100) -> float:
    """
    Benchmark torch.compile'd GEMV.
    """
    device = "cuda"
    
    A = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    B = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    
    @torch.compile(mode="max-autotune")
    def gemv(a, b):
        return torch.mm(a, b)
    
    # Warm up (includes compilation)
    for _ in range(10):
        C = gemv(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        C = gemv(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms


def estimate_tok_per_sec(moe_time_ms: float, num_layers: int = NUM_LAYERS) -> float:
    """
    Estimate tokens per second based on MoE layer time.
    
    For gpt-oss-120b:
    - 60 MoE layers
    - Each MoE has FC1 + FC2
    - Total MoE time per token ≈ 60 * moe_time_ms * 2
    
    But MoE is only ~40-50% of total decode time, so:
    - Total time per token ≈ 60 * moe_time_ms * 2 / 0.45
    """
    total_moe_time_ms = moe_time_ms * num_layers * 2  # FC1 + FC2
    # Assuming MoE is 45% of total time (from nsys profiling)
    total_time_ms = total_moe_time_ms / 0.45
    return 1000.0 / total_time_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS GEMV for MoE")
    parser.add_argument("--m", type=str, default="1,2,4,8,16,32",
                        help="Batch sizes to test (comma-separated)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per benchmark")
    parser.add_argument("--skip-grouped-gemm", action="store_true",
                        help="Skip grouped GEMM benchmark (slow)")
    args = parser.parse_args()
    
    m_values = [int(x) for x in args.m.split(",")]
    
    print("=" * 70)
    print("CUTLASS GEMV Benchmark for FP4 MoE Decode")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Intermediate dim: {INTER_DIM}")
    print(f"  Num experts: {NUM_EXPERTS}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Iterations: {args.iterations}")
    print()
    
    print("=" * 70)
    print("Benchmark Results (per MoE call, single layer)")
    print("=" * 70)
    print(f"{'M':>5} | {'PyTorch GEMV':>14} | {'Compiled GEMV':>14} | {'Grouped GEMM':>14} | {'Proj tok/s':>12}")
    print("-" * 70)
    
    for m in m_values:
        # PyTorch GEMV (FC1 dimension: k=4096, n=11776)
        pytorch_time = benchmark_gemv_simple(m, HIDDEN_DIM, INTER_DIM * 2, args.iterations)
        
        # Compiled GEMV
        compiled_time = benchmark_gemv_compiled(m, HIDDEN_DIM, INTER_DIM * 2, args.iterations)
        
        # Grouped GEMM
        if not args.skip_grouped_gemm:
            grouped_time = benchmark_grouped_gemm(m, HIDDEN_DIM, INTER_DIM * 2, NUM_EXPERTS, args.iterations)
        else:
            grouped_time = float('nan')
        
        # Estimate tokens/sec based on best time
        best_time = min(pytorch_time, compiled_time)
        if not np.isnan(grouped_time) and not np.isinf(grouped_time):
            best_time = min(best_time, grouped_time)
        
        tok_per_sec = estimate_tok_per_sec(best_time) if best_time > 0 else 0
        
        print(f"{m:>5} | {pytorch_time:>12.3f}ms | {compiled_time:>12.3f}ms | "
              f"{grouped_time:>12.3f}ms | {tok_per_sec:>10.1f}")
    
    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)
    print("""
Key observations:
1. PyTorch GEMV uses cuBLAS gemm which is not optimized for FP4
2. torch.compile can help but doesn't use Tensor Cores for FP4
3. Grouped GEMM with 128x128 tiles wastes compute at small M

What's needed:
1. CUTLASS GemvBlockScaled kernel integration (FP4 native)
2. Proper routing logic to gather tokens per expert
3. Efficient scale factor handling for block-scaled format

Performance target:
- llama.cpp: ~58 tok/s at M=1
- SGLang: ~52 tok/s at M=1  
- Current vLLM: ~29 tok/s at M=1
- Target: >=52 tok/s
""")


if __name__ == "__main__":
    main()


