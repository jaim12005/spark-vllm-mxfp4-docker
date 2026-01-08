#!/usr/bin/env python3
"""
Optimized GEMV benchmark - using torch.compile and efficient indexing.

Usage:
    docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
        python3 /workspace/benchmark_gemv_optimized.py
"""
import os
import sys
import time

os.environ.setdefault("PYTHONPATH", "/workspace/flashinfer:/workspace/vllm")
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.cuda.set_device(device)

print("=" * 70)
print("Optimized GEMV Benchmark for MoE")
print("=" * 70)

# gpt-oss dimensions
NUM_EXPERTS = 128
HIDDEN_DIM = 2944
INTERMEDIATE_DIM = 5888
TOPK = 8
NUM_LAYERS = 60


# ============================================================================
# Optimized GEMV - No Python loops, uses einsum
# ============================================================================
def moe_gemv_einsum(
    input: torch.Tensor,                    # [M, hidden_dim]
    token_selected_experts: torch.Tensor,   # [M, topk]
    token_final_scales: torch.Tensor,       # [M, topk]
    fc1_weights: torch.Tensor,              # [num_experts, intermediate*2, hidden_dim]
    fc2_weights: torch.Tensor,              # [num_experts, hidden_dim, intermediate]
) -> torch.Tensor:
    """
    GEMV using efficient gather + einsum.
    No Python loops - fully vectorized.
    """
    M, hidden_dim = input.shape
    topk = token_selected_experts.shape[1]
    intermediate_dim = fc1_weights.shape[1] // 2
    
    # Gather FC1 weights for selected experts: [M, topk, inter*2, hidden]
    # Use advanced indexing
    fc1_selected = fc1_weights[token_selected_experts]  # [M, topk, inter*2, hidden]
    
    # FC1: einsum for batched matrix-vector
    # [M, topk, inter*2, hidden] @ [M, 1, hidden, 1] -> [M, topk, inter*2]
    fc1_out = torch.einsum('mtih,mh->mti', fc1_selected, input)
    
    # SwiGLU
    gate = fc1_out[:, :, :intermediate_dim]
    up = fc1_out[:, :, intermediate_dim:]
    activated = F.silu(gate) * up  # [M, topk, intermediate]
    
    # Gather FC2 weights: [M, topk, hidden, intermediate]
    fc2_selected = fc2_weights[token_selected_experts]
    
    # FC2: [M, topk, hidden, inter] @ [M, topk, inter] -> [M, topk, hidden]
    fc2_out = torch.einsum('mthi,mti->mth', fc2_selected, activated)
    
    # Scale and reduce: [M, topk, hidden] * [M, topk, 1] -> sum -> [M, hidden]
    output = (fc2_out * token_final_scales.unsqueeze(-1)).sum(dim=1)
    
    return output


# Compile the function
moe_gemv_compiled = torch.compile(moe_gemv_einsum, mode="max-autotune")


# ============================================================================
# Single expert GEMV (for comparison)
# ============================================================================
def single_expert_gemv(
    input: torch.Tensor,          # [M, hidden_dim]
    fc1_weight: torch.Tensor,     # [intermediate*2, hidden_dim]
    fc2_weight: torch.Tensor,     # [hidden_dim, intermediate]
) -> torch.Tensor:
    """Single expert MoE - simpler case."""
    # FC1
    fc1_out = F.linear(input, fc1_weight)  # [M, intermediate*2]
    
    # SwiGLU
    intermediate_dim = fc1_out.shape[1] // 2
    gate = fc1_out[:, :intermediate_dim]
    up = fc1_out[:, intermediate_dim:]
    activated = F.silu(gate) * up
    
    # FC2
    output = F.linear(activated, fc2_weight)  # [M, hidden_dim]
    
    return output


single_expert_compiled = torch.compile(single_expert_gemv, mode="max-autotune")


# ============================================================================
# Benchmark
# ============================================================================
def benchmark_fn(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    
    return (time.time() - start) / iters * 1000  # ms


def main():
    print(f"\nCreating test tensors...")
    
    # Input
    x = torch.randn(1, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    
    # Expert routing
    topk_indices = torch.randint(0, NUM_EXPERTS, (1, TOPK), device=device)
    topk_weights = torch.softmax(torch.randn(1, TOPK, device=device), dim=-1).float()
    
    # Weights (BF16 for fair comparison with dequantized MXFP4)
    fc1_weights = torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM * 2, HIDDEN_DIM, 
                              dtype=torch.bfloat16, device=device) * 0.01
    fc2_weights = torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                              dtype=torch.bfloat16, device=device) * 0.01
    
    # Single expert weights
    fc1_single = fc1_weights[0]
    fc2_single = fc2_weights[0]
    
    print("\nWarming up torch.compile...")
    
    # Warmup compiled versions (triggers compilation)
    for _ in range(3):
        _ = moe_gemv_compiled(x, topk_indices, topk_weights, fc1_weights, fc2_weights)
        _ = single_expert_compiled(x, fc1_single, fc2_single)
    torch.cuda.synchronize()
    
    print("Benchmarking...\n")
    
    # Benchmark single expert (baseline)
    time_single = benchmark_fn(
        lambda: single_expert_compiled(x, fc1_single, fc2_single)
    )
    print(f"Single Expert GEMV (compiled): {time_single:.3f} ms")
    print(f"  Per-token (60 layers × 2 × 1 expert): {time_single * 120:.1f} ms")
    print(f"  Projected: {1000 / (time_single * 120):.1f} tok/s")
    
    print()
    
    # Benchmark full MoE with topk=8
    time_moe = benchmark_fn(
        lambda: moe_gemv_compiled(x, topk_indices, topk_weights, fc1_weights, fc2_weights)
    )
    print(f"Full MoE (8 experts, compiled): {time_moe:.3f} ms")
    print(f"  Per-token (60 layers × 2): {time_moe * 120:.1f} ms")
    print(f"  Projected: {1000 / (time_moe * 120):.1f} tok/s")
    
    print()
    
    # Non-compiled for comparison
    time_moe_eager = benchmark_fn(
        lambda: moe_gemv_einsum(x, topk_indices, topk_weights, fc1_weights, fc2_weights)
    )
    print(f"Full MoE (8 experts, eager): {time_moe_eager:.3f} ms")
    print(f"  Per-token (60 layers × 2): {time_moe_eager * 120:.1f} ms")
    print(f"  Projected: {1000 / (time_moe_eager * 120):.1f} tok/s")
    
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"\nCurrent vLLM CUTLASS (MXFP4): ~29 tok/s")
    print(f"Target (llama.cpp): ~58 tok/s")
    print()
    
    # Memory analysis
    print("Memory Analysis:")
    print(f"  BF16 weights: {(fc1_weights.numel() + fc2_weights.numel()) * 2 / 1e9:.2f} GB")
    print(f"  MXFP4 weights: {(fc1_weights.numel() + fc2_weights.numel()) / 2 / 1e9:.2f} GB (4x smaller)")
    print()
    
    # Problem analysis
    print("Why Python GEMV is slow:")
    print("  1. Weight gather is expensive (reading from 8 random experts)")
    print("  2. einsum dispatches to generic GEMM, not optimized GEMV")
    print("  3. BF16 → 4x more memory bandwidth than MXFP4")
    print("  4. No fusion between FC1, activation, FC2")
    print()
    print("What CUTLASS does better:")
    print("  1. Grouped GEMM batches all experts efficiently")
    print("  2. Uses tensor cores for MMA")
    print("  3. MXFP4 reduces memory bandwidth")
    print("  4. Fuses activation into kernel")


if __name__ == "__main__":
    main()

