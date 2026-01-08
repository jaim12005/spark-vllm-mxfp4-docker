#!/usr/bin/env python3
"""
Analyze DP4A GEMV benchmark results for MoE decode optimization.
"""

# DP4A GEMV benchmark results (from test_dp4a_kernel.py)
dp4a_results = {
    "FC1": {"ms": 0.3096, "gflops": 311.56},  # hidden -> inter (4096 -> 11776)
    "FC2": {"ms": 0.2537, "gflops": 380.28},  # inter -> hidden (11776 -> 4096)
}

# PyTorch BF16 baseline
pytorch_results = {
    "FC1": {"ms": 0.4080, "gflops": 236.46},
    "FC2": {"ms": 0.3870, "gflops": 249.24},
}

# gpt-oss-120b MoE architecture
NUM_EXPERTS = 256        # Total experts
TOPK = 8                 # Experts used per token
NUM_LAYERS = 60          # Number of MoE layers
HIDDEN_DIM = 4096        # Hidden dimension (model dim)
INTER_DIM = 11776        # Intermediate dimension (FFN dim)

print("=" * 70)
print("DP4A GEMV Analysis for gpt-oss-120b MoE Decode (M=1)")
print("=" * 70)

print("\n1. Single GEMV Performance")
print("-" * 50)

for name, dp4a in dp4a_results.items():
    pt = pytorch_results[name]
    speedup = pt["ms"] / dp4a["ms"]
    print(f"{name}: DP4A={dp4a['ms']:.4f}ms vs PyTorch={pt['ms']:.4f}ms ({speedup:.2f}x faster)")

print("\n2. Per-Expert MoE GEMV (FC1 + FC2)")
print("-" * 50)

# MoE expert: FC1 (gate + up projection) + FC2 (down projection)
# Actually: FC1 is gate_proj + up_proj, but for simplicity:
# MoE call = 2 GEMV (FC1 for SwiGLU has 2 projections, FC2 for down)
# For SwiGLU: gate and up are separate, so 3 GEMVs per expert

# Simplified: assume 2 GEMVs per expert (FC1 + FC2)
dp4a_per_expert = dp4a_results["FC1"]["ms"] + dp4a_results["FC2"]["ms"]
pytorch_per_expert = pytorch_results["FC1"]["ms"] + pytorch_results["FC2"]["ms"]

print(f"Per expert (FC1 + FC2):")
print(f"  DP4A:    {dp4a_per_expert:.4f} ms")
print(f"  PyTorch: {pytorch_per_expert:.4f} ms")
print(f"  Speedup: {pytorch_per_expert/dp4a_per_expert:.2f}x")

print("\n3. Per-Token MoE (TopK=8 experts)")
print("-" * 50)

dp4a_topk = dp4a_per_expert * TOPK
pytorch_topk = pytorch_per_expert * TOPK

print(f"TopK={TOPK} experts per token:")
print(f"  DP4A:    {dp4a_topk:.4f} ms")
print(f"  PyTorch: {pytorch_topk:.4f} ms")
print(f"  Speedup: {pytorch_topk/dp4a_topk:.2f}x")

print("\n4. Full Forward Pass (60 layers)")
print("-" * 50)

# Each layer has one MoE block
dp4a_full = dp4a_topk * NUM_LAYERS
pytorch_full = pytorch_topk * NUM_LAYERS

# Current CUTLASS MoE benchmark (from nsys profiling):
# MoE FC1: 266 μs per call = 0.266 ms
# MoE FC2: 115 μs per call = 0.115 ms
# Per expert: ~0.381 ms
cutlass_per_expert = 0.266 + 0.115
cutlass_topk = cutlass_per_expert * TOPK
cutlass_full = cutlass_topk * NUM_LAYERS

print(f"MoE time for 60 layers (TopK={TOPK}):")
print(f"  DP4A:     {dp4a_full:.2f} ms")
print(f"  PyTorch:  {pytorch_full:.2f} ms")
print(f"  CUTLASS:  {cutlass_full:.2f} ms (current)")

print("\n5. Decode Throughput Projection")
print("-" * 50)

# Current vLLM decode: ~29 tok/s = ~34.5 ms per token
# nsys profiling showed:
# - MoE: 40.5% of GPU time
# - GEMV (attention): 51.2% of GPU time
# - Other: 8.3%

current_decode_ms = 1000 / 29  # ~34.5 ms
moe_portion = 0.405            # 40.5% from nsys

# If we replace MoE with DP4A
moe_current = current_decode_ms * moe_portion  # ~14 ms
moe_dp4a = dp4a_full                           # New MoE time

# Estimated new decode time
other_time = current_decode_ms * (1 - moe_portion)
new_decode_ms = other_time + moe_dp4a
new_throughput = 1000 / new_decode_ms

print(f"Current decode: {current_decode_ms:.1f} ms ({29} tok/s)")
print(f"  MoE portion:  {moe_current:.1f} ms ({moe_portion*100:.1f}%)")
print(f"  Other:        {other_time:.1f} ms")
print()
print(f"With DP4A MoE:  {new_decode_ms:.1f} ms ({new_throughput:.1f} tok/s)")
print(f"  MoE (DP4A):   {moe_dp4a:.1f} ms")
print(f"  Other:        {other_time:.1f} ms")
print()
print(f"Potential speedup: {current_decode_ms/new_decode_ms:.2f}x")
print(f"Target (llama.cpp): 58 tok/s")

print("\n6. Reality Check")
print("-" * 50)

# The DP4A GEMV times seem too fast - let's verify
# Memory bandwidth check

# FC1: N=11776, K=4096
fc1_bytes = 11776 * 4096 / 2  # FP4 packed
fc1_bytes += 11776 * 4096 / 32  # Scales
fc1_bytes += 4096 * 2  # BF16 activation
fc1_bytes += 11776 * 2  # BF16 output

# FC2: N=4096, K=11776
fc2_bytes = 4096 * 11776 / 2
fc2_bytes += 4096 * 11776 / 32
fc2_bytes += 11776 * 2
fc2_bytes += 4096 * 2

# GB10 bandwidth: ~800 GB/s
bandwidth_gbs = 800

fc1_mem_bound_ms = (fc1_bytes / 1e9) / bandwidth_gbs * 1000
fc2_mem_bound_ms = (fc2_bytes / 1e9) / bandwidth_gbs * 1000

print("Memory bandwidth check:")
print(f"  FC1: {fc1_bytes/1e6:.2f} MB, memory-bound: {fc1_mem_bound_ms:.4f} ms")
print(f"       Achieved: {dp4a_results['FC1']['ms']:.4f} ms ({fc1_mem_bound_ms/dp4a_results['FC1']['ms']*100:.1f}% of peak)")
print(f"  FC2: {fc2_bytes/1e6:.2f} MB, memory-bound: {fc2_mem_bound_ms:.4f} ms")
print(f"       Achieved: {dp4a_results['FC2']['ms']:.4f} ms ({fc2_mem_bound_ms/dp4a_results['FC2']['ms']*100:.1f}% of peak)")

print("\n7. Summary")
print("-" * 50)
print("The DP4A kernel achieves ~10% of memory bandwidth peak,")
print("suggesting significant optimization opportunity.")
print()
print("Key findings:")
print(f"  1. DP4A is {pytorch_per_expert/dp4a_per_expert:.1f}x faster than PyTorch BF16 GEMV")
print(f"  2. Projected decode: {new_throughput:.1f} tok/s vs current {29} tok/s")
print(f"  3. Still far from llama.cpp's 58 tok/s")
print()
print("The gap to llama.cpp suggests they use:")
print("  - Better memory access patterns")
print("  - Warp-level optimization")
print("  - Different thread blocking")

