#!/usr/bin/env python3
"""Debug transposed weight GEMV kernel with small inputs."""

import torch

# Small test case
M = 1
K = 64  # 2 K-blocks
N = 4

print(f"Debug test: M={M}, K={K}, N={N}")

# Create simple test data
weight = torch.zeros((N, K // 2), dtype=torch.uint8, device="cuda")
scale = torch.ones((N, K // 32), dtype=torch.uint8, device="cuda") * 127  # E8M0 = 1.0
input_bf16 = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")

# Set a simple weight pattern
# FP4 value 2 = index 2 → doubled value = 2
# So setting 0x22 means two elements with value 2
weight.fill_(0x22)  # All elements = 2

print(f"Weight[0, :8]: {weight[0, :8].tolist()}")
print(f"Scale[0, :]: {scale[0, :].tolist()}")
print(f"Input[0, :8]: {input_bf16[0, :8].tolist()}")

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

# Check Q8 values
print(f"\nQ8 block 0 (first 8 bytes): {q8[0, 0, :8].tolist()}")

# Run original kernel
result_orig = gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)
print(f"\nOriginal result: {result_orig[0].tolist()}")

# Transpose weights
weight_t = transpose_weights_fp4(weight, K)
scale_t = transpose_scales_fp4(scale, K)

print(f"\nweight_t shape: {weight_t.shape}")
print(f"weight_t[0, 0, :8]: {weight_t[0, 0, :8].tolist()}")
print(f"weight_t[0, 1, :8]: {weight_t[0, 1, :8].tolist()}")

print(f"\nscale_t shape: {scale_t.shape}")
print(f"scale_t[0, :]: {scale_t[0, :].tolist()}")

# Run transposed kernel
result_trans = gemv_mxfp4_transposed(q8, weight_t, scale_t, K)
print(f"\nTransposed result: {result_trans[0].tolist()}")

# Check for NaN
if torch.isnan(result_orig).any():
    print("\nWARNING: Original result has NaN!")
if torch.isnan(result_trans).any():
    print("\nWARNING: Transposed result has NaN!")
