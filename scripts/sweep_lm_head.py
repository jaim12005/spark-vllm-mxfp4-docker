#!/usr/bin/env python3
"""Sweep different NWARPS/ROWS configurations for LM Head."""

import torch
import time

M = 1
K = 2880
N = 201088  # vocab_size

print("=" * 70)
print(f"LM Head Config Sweep: M={M}, K={K}, N={N}")
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
print(f"Theoretical time @ 273 GB/s: {total_bytes / 273e9 * 1e3:.3f} ms")
print()

# Get the module to call kernels directly
from flashinfer.jit.gemv import get_gemv_module
module = get_gemv_module()

# Test configurations using the prequant wrapper
# We'll modify the heuristic temporarily by testing subsets of N
# that would trigger different configs

configs_to_test = [
    ("NWARPS=2, ROWS=8", 2880),      # Triggers ROWS=8 (N <= 4096)
    ("NWARPS=2, ROWS=16", 201088),   # Triggers ROWS=16 (N > 65536)
]

from flashinfer.gemv import gemv_mxfp4_dp4a_prequant

print("Testing auto-selected configurations via N size:")
print("-" * 70)

for config_name, test_n in configs_to_test:
    if test_n > N:
        continue
        
    test_weight = weight[:test_n]
    test_scale = scale[:test_n]
    
    test_bytes = test_n * K // 2 + test_n * K // 32 + M * K * 2 + M * test_n * 2
    
    # Warmup
    for _ in range(10):
        result = gemv_mxfp4_dp4a_prequant(q8, test_weight, test_scale, K)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        result = gemv_mxfp4_dp4a_prequant(q8, test_weight, test_scale, K)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    
    bw = test_bytes / (elapsed_ms / 1000) / 1e9
    eff = bw / 273 * 100
    
    # Calculate blocks and threads
    n_k_blocks = K // 32
    if test_n <= 4096:
        rows = 8
    elif test_n <= 65536:
        rows = 8
    else:
        rows = 16
    
    blocks = (test_n + rows - 1) // rows
    
    print(f"N={test_n:6d} ({config_name}): {elapsed_ms:.3f} ms, {bw:.1f} GB/s ({eff:.1f}%), {blocks} blocks")

# Now test with full N=201088 at different ROWS by using smaller weight slices
# and extrapolating
print()
print("Bandwidth per output element analysis:")
print("-" * 70)

test_sizes = [10000, 25000, 50000, 100000, 150000, 201088]
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
    
    # Time per output element
    time_per_elem_ns = elapsed_ms * 1e6 / test_n
    
    # Bytes per output element
    bytes_per_elem = K // 2 + K // 32 + 2  # weight + scale + output
    
    # Effective bandwidth per element
    bw_per_elem = bytes_per_elem / (time_per_elem_ns * 1e-9) / 1e9
    
    print(f"N={test_n:6d}: {elapsed_ms:.3f} ms, {time_per_elem_ns:.2f} ns/elem, {bw_per_elem:.1f} GB/s per elem")

print()
print("The per-element bandwidth shows if we have constant overhead or scaling issues.")
