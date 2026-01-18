#!/usr/bin/env python3
"""Verify which GEMV configuration is being selected."""

import torch
import time

M = 1
K = 2880
N = 201088

print(f"Testing N={N}, K={K}")
print()

# Expected config based on new heuristics:
# N > 100000 → NWARPS=4
# N > 65536 → ROWS=16
print("Expected config for N=201088, K=2880:")
print("  - N > 100000 → NWARPS=4")
print("  - N > 65536  → ROWS=16")
print("  → Should use (NWARPS=4, ROWS=16)")
print()

# Create test data
weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Pre-quantize
from flashinfer.gemv import quantize_activations_q8, gemv_mxfp4_dp4a_prequant
n_k_blocks = K // 32
q8 = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
quantize_activations_q8(input_bf16, q8)

# Calculate data
weight_bytes = N * K // 2
scale_bytes = N * K // 32
input_bytes = M * K * 2
output_bytes = M * N * 2
total_bytes = weight_bytes + scale_bytes + input_bytes + output_bytes

# Warmup
for _ in range(10):
    result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) / 100 * 1000

bandwidth = total_bytes / (elapsed_ms / 1000) / 1e9
efficiency = bandwidth / 273 * 100

print(f"Performance:")
print(f"  Time: {elapsed_ms:.3f} ms")
print(f"  Bandwidth: {bandwidth:.1f} GB/s ({efficiency:.1f}% efficiency)")

# Compare with smaller N to see config difference
print()
print("Comparison with smaller N (to trigger different configs):")

test_sizes = [2880, 50000, 100000, 150000, 201088]
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
    
    # Determine expected config
    if test_n > 100000:
        nwarps = 4
    elif K <= 3072:
        nwarps = 2
    else:
        nwarps = 4
    
    if test_n <= 256:
        rows = 2
    elif test_n <= 512:
        rows = 4
    elif test_n <= 4096:
        rows = 8
    elif test_n <= 65536:
        rows = 8
    else:
        rows = 16
    
    print(f"  N={test_n:6d}: {elapsed_ms:.3f} ms, {bw:.1f} GB/s ({eff:.1f}%) [NWARPS={nwarps}, ROWS={rows}]")
