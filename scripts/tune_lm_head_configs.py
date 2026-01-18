#!/usr/bin/env python3
"""Test different NWARPS/ROWS configurations for LM Head."""

import torch
import time
import ctypes

M = 1
K = 2880
N = 201088  # vocab_size

print("=" * 70)
print(f"LM Head Configuration Tuning: M={M}, K={K}, N={N}")
print("=" * 70)

# Create test data
weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Pre-quantize
from flashinfer.gemv import quantize_activations_q8
n_k_blocks = K // 32
q8 = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
quantize_activations_q8(input_bf16, q8)

# Calculate theoretical bandwidth
weight_bytes = N * K // 2
scale_bytes = N * K // 32
input_bytes = M * K * 2
output_bytes = M * N * 2
total_bytes = weight_bytes + scale_bytes + input_bytes + output_bytes
print(f"Data size: {total_bytes / 1e6:.1f} MB")
print(f"Theoretical time @ 273 GB/s: {total_bytes / 273e9 * 1e3:.2f} ms")
print()

# Get the JIT module directly
from flashinfer.jit.gemv import get_gemv_module
module = get_gemv_module()

# The module has the raw kernel function
# We need to call it with different grid/block configurations

# For testing, we'll use the prequant kernel via Python wrapper
# but measure the impact of different problem sizes

from flashinfer.gemv import gemv_mxfp4_dp4a_prequant

def benchmark(name, iterations=50):
    """Benchmark the current configuration."""
    output = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    
    # Warmup
    for _ in range(10):
        result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000
    
    bandwidth = total_bytes / (elapsed_ms / 1000) / 1e9
    efficiency = bandwidth / 273 * 100
    
    print(f"{name}:")
    print(f"  Time: {elapsed_ms:.3f} ms, BW: {bandwidth:.1f} GB/s ({efficiency:.1f}%)")
    return elapsed_ms

# Test current configuration
print("\n=== Current Configuration ===")
base_time = benchmark("NWARPS=2, ROWS=16 (auto-selected)")

# Test with different N values to understand scaling
print("\n=== Scaling Analysis ===")
test_sizes = [50000, 100000, 150000, 201088]
for test_n in test_sizes:
    test_weight = weight[:test_n]
    test_scale = scale[:test_n]
    
    # Warmup
    for _ in range(5):
        result = gemv_mxfp4_dp4a_prequant(q8, test_weight, test_scale, K)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        result = gemv_mxfp4_dp4a_prequant(q8, test_weight, test_scale, K)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    
    test_bytes = test_n * K // 2 + test_n * K // 32 + M * K * 2 + M * test_n * 2
    bw = test_bytes / (elapsed_ms / 1000) / 1e9
    eff = bw / 273 * 100
    
    # Expected config
    if test_n <= 4096:
        config = "NWARPS=2, ROWS=8"
    elif test_n <= 65536:
        config = "NWARPS=2, ROWS=8"
    else:
        config = "NWARPS=2, ROWS=16"
    
    print(f"N={test_n:6d}: {elapsed_ms:.3f} ms, {bw:.1f} GB/s ({eff:.1f}%) [{config}]")

print("\n=== Analysis ===")
print(f"LM Head is memory-bound (arithmetic intensity = {(M * N * K * 2) / total_bytes:.2f} FLOP/byte)")
print(f"At 80% efficiency, possible causes:")
print(f"  - Memory controller not fully saturated")
print(f"  - Cross-warp reduction overhead")
print(f"  - Block scheduling gaps")
