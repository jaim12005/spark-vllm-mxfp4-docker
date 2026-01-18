#!/usr/bin/env python3
"""Test transposed weight GEMV kernel."""

import torch
import time

print("Testing transposed weight GEMV kernel...")
print()

# LM Head dimensions
M = 1
K = 2880
N = 201088

# Create test data
weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

print(f"LM Head: M={M}, K={K}, N={N}")
print(f"Weight shape: {weight.shape}")
print(f"Scale shape: {scale.shape}")
print()

# Import functions
from flashinfer.gemv import (
    quantize_activations_q8,
    gemv_mxfp4_dp4a_prequant,
    transpose_weights_fp4,
    transpose_scales_fp4,
    gemv_mxfp4_transposed,
)

# Pre-quantize activations
n_k_blocks = K // 32
q8 = torch.empty((M, n_k_blocks, 36), dtype=torch.uint8, device="cuda")
quantize_activations_q8(input_bf16, q8)
print("Activations quantized")

# Transpose weights (one-time)
print("Transposing weights...")
start = time.perf_counter()
weight_t = transpose_weights_fp4(weight, K)
scale_t = transpose_scales_fp4(scale, K)
torch.cuda.synchronize()
transpose_time = (time.perf_counter() - start) * 1000
print(f"Transpose time: {transpose_time:.2f} ms (one-time cost)")
print(f"weight_t shape: {weight_t.shape}")
print(f"scale_t shape: {scale_t.shape}")
print()

# Verify output shapes are correct
assert weight_t.shape == (n_k_blocks, N, 16), f"Expected {(n_k_blocks, N, 16)}, got {weight_t.shape}"
assert scale_t.shape == (n_k_blocks, N), f"Expected {(n_k_blocks, N)}, got {scale_t.shape}"

# Benchmark original kernel
print("Benchmarking original prequant kernel...")
for _ in range(10):
    result_orig = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    result_orig = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
torch.cuda.synchronize()
orig_time = (time.perf_counter() - start) / 50 * 1000

# Benchmark transposed kernel
print("Benchmarking transposed kernel...")
for _ in range(10):
    result_trans = gemv_mxfp4_transposed(q8, weight_t, scale_t, K)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    result_trans = gemv_mxfp4_transposed(q8, weight_t, scale_t, K)
torch.cuda.synchronize()
trans_time = (time.perf_counter() - start) / 50 * 1000

# Calculate bandwidth
weight_bytes = N * K // 2
scale_bytes = N * K // 32
input_bytes = M * K * 2
output_bytes = M * N * 2
total_bytes = weight_bytes + scale_bytes + input_bytes + output_bytes

orig_bw = total_bytes / (orig_time / 1000) / 1e9
trans_bw = total_bytes / (trans_time / 1000) / 1e9

print()
print("=" * 60)
print("Results:")
print("=" * 60)
print(f"Original (prequant):  {orig_time:.3f} ms, {orig_bw:.1f} GB/s ({orig_bw/273*100:.1f}%)")
print(f"Transposed:           {trans_time:.3f} ms, {trans_bw:.1f} GB/s ({trans_bw/273*100:.1f}%)")
print(f"Speedup:              {orig_time/trans_time:.2f}x")
print()

# Check outputs
print(f"Original output: min={result_orig.min().item():.4f}, max={result_orig.max().item():.4f}")
print(f"Transposed output: min={result_trans.min().item():.4f}, max={result_trans.max().item():.4f}")
