#!/usr/bin/env python3
"""Test transposed kernel with controlled data to avoid NaN."""

import torch
import time

print("Testing transposed weight GEMV kernel with controlled data...")
print()

# LM Head dimensions
M = 1
K = 2880
N = 201088

# Create controlled test data (avoid edge cases)
weight = torch.randint(0, 128, (N, K // 2), dtype=torch.uint8, device="cuda")  # Avoid high nibbles
scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127  # E8M0 = 1.0
input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Clamp input to avoid extreme values
input_bf16 = input_bf16.clamp(-2.0, 2.0)

print(f"LM Head: M={M}, K={K}, N={N}")

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

# Transpose weights
weight_t = transpose_weights_fp4(weight, K)
scale_t = transpose_scales_fp4(scale, K)

# Warmup and benchmark original
for _ in range(10):
    result_orig = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    result_orig = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
torch.cuda.synchronize()
orig_time = (time.perf_counter() - start) / 50 * 1000

# Warmup and benchmark transposed
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

# Verify outputs
orig_nan = torch.isnan(result_orig).sum().item()
trans_nan = torch.isnan(result_trans).sum().item()
print(f"Original NaN count: {orig_nan}")
print(f"Transposed NaN count: {trans_nan}")
print(f"Original: min={result_orig.min().item():.2f}, max={result_orig.max().item():.2f}, mean={result_orig.mean().item():.2f}")
print(f"Transposed: min={result_trans.min().item():.2f}, max={result_trans.max().item():.2f}, mean={result_trans.mean().item():.2f}")

# Compare values
diff = (result_orig - result_trans).abs()
print(f"Max difference: {diff.max().item():.4f}")
print(f"Mean difference: {diff.mean().item():.4f}")
