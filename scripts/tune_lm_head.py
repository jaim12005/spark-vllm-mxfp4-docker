#!/usr/bin/env python3
"""Tune LM Head GEMV configuration for optimal performance."""

import torch
import time

# Test different configurations for LM Head
M = 1
K = 2880
N = 201088  # vocab_size

print("=" * 70)
print(f"LM Head Tuning: M={M}, K={K}, N={N}")
print("=" * 70)

# Create test data
weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Pre-quantize
from flashinfer.gemv import quantize_activations_q8, gemv_mxfp4_dp4a_prequant

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
print(f"Theoretical max (273 GB/s): {total_bytes / 273e9 * 1e6:.2f} ms")
print()

# Warmup
for _ in range(10):
    result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
iterations = 50
for _ in range(iterations):
    result = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) / iterations * 1000

bandwidth = total_bytes / (elapsed_ms / 1000) / 1e9
efficiency = bandwidth / 273 * 100

print(f"Current (NWARPS=2, ROWS=16):")
print(f"  Time: {elapsed_ms:.3f} ms")
print(f"  Bandwidth: {bandwidth:.1f} GB/s ({efficiency:.1f}% efficiency)")
print()

# Roofline analysis
flops = M * N * K * 2  # multiply-accumulate
tflops = flops / (elapsed_ms / 1000) / 1e12
print(f"  Compute: {tflops:.2f} TFLOPS")
print(f"  Arithmetic intensity: {flops / total_bytes:.2f} FLOP/byte")
