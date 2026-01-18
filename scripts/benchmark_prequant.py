#!/usr/bin/env python3
"""Benchmark pre-quantized GEMV vs original."""

import torch
import time
from flashinfer.gemv import gemv_mxfp4_dp4a, quantize_activations_q8, gemv_mxfp4_dp4a_prequant

def benchmark_original(M, K, N, warmup=20, iters=200):
    inp = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(1, 128, (N, K // 32), dtype=torch.uint8, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    
    for _ in range(warmup):
        gemv_mxfp4_dp4a(inp, w, s, out)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        gemv_mxfp4_dp4a(inp, w, s, out)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6

def benchmark_prequant(M, K, N, warmup=20, iters=200):
    inp = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(1, 128, (N, K // 32), dtype=torch.uint8, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    
    # Pre-allocate Q8 buffer
    q8 = quantize_activations_q8(inp)
    
    for _ in range(warmup):
        quantize_activations_q8(inp, q8)
        gemv_mxfp4_dp4a_prequant(q8, w, s, K, out)
    torch.cuda.synchronize()
    
    # Time quantization + GEMV together
    start = time.perf_counter()
    for _ in range(iters):
        quantize_activations_q8(inp, q8)
        gemv_mxfp4_dp4a_prequant(q8, w, s, K, out)
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / iters * 1e6
    
    # Time GEMV only (quantization done once)
    q8 = quantize_activations_q8(inp)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        gemv_mxfp4_dp4a_prequant(q8, w, s, K, out)
    torch.cuda.synchronize()
    gemv_only = (time.perf_counter() - start) / iters * 1e6
    
    return total_time, gemv_only

def main():
    print("=" * 70)
    print("GEMV Benchmark: Original vs Pre-Quantized")
    print("=" * 70)
    print(f"{'Config':<25} {'Original':>12} {'Prequant+Q':>12} {'GEMV only':>12} {'Speedup':>10}")
    print("-" * 70)

    for M, K, N in [(1, 8192, 8192), (1, 8192, 24576), (4, 8192, 8192)]:
        orig = benchmark_original(M, K, N)
        total, gemv_only = benchmark_prequant(M, K, N)
        speedup = orig / gemv_only
        
        config = f"M={M}, K={K}, N={N}"
        print(f"{config:<25} {orig:>10.1f}us {total:>10.1f}us {gemv_only:>10.1f}us {speedup:>9.2f}x")

    print()
    print("Note: 'GEMV only' = time when activations are pre-quantized and reused")
    print("      (e.g., for qkv_proj and o_proj in same layer)")

if __name__ == "__main__":
    main()
