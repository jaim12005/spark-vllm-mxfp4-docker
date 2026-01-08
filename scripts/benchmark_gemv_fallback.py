#!/usr/bin/env python3
"""
Benchmark GEMV fallback vs CUTLASS grouped GEMM for MoE at small M.

This script compares:
1. Current CUTLASS grouped GEMM (128x128 tiles)
2. Python GEMV fallback using torch.mv

Usage:
    docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
        python3 /workspace/scripts/benchmark_gemv_fallback.py
"""
import os
import sys
import time
from typing import Optional, List

# Setup paths
os.environ.setdefault("PYTHONPATH", "/workspace/flashinfer:/workspace/vllm")
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch
import torch.nn.functional as F

# Check CUDA
assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

print("=" * 70)
print("GEMV Fallback Benchmark for MoE (gpt-oss-120b dimensions)")
print("=" * 70)


# ============================================================================
# gpt-oss-120b MoE dimensions
# ============================================================================
NUM_EXPERTS = 128
HIDDEN_DIM = 2944
INTERMEDIATE_DIM = 5888  # For SwiGLU: gate + up
TOPK = 8
NUM_LAYERS = 60


# ============================================================================
# GEMV Fallback Implementation
# ============================================================================
def moe_gemv_fallback(
    input: torch.Tensor,                    # [M, hidden_dim] BF16
    token_selected_experts: torch.Tensor,   # [M, topk] int
    token_final_scales: torch.Tensor,       # [M, topk] float32
    fc1_weights: torch.Tensor,              # [num_experts, intermediate*2, hidden_dim] BF16
    fc2_weights: torch.Tensor,              # [num_experts, hidden_dim, intermediate] BF16
) -> torch.Tensor:
    """
    GEMV-based MoE for small batch sizes.
    
    Uses torch.mv (matrix-vector multiply) which dispatches to cuBLAS GEMV.
    More efficient than grouped GEMM with 128x128 tiles when M is small.
    """
    M, hidden_dim = input.shape
    topk = token_selected_experts.shape[1]
    
    output = torch.zeros(M, hidden_dim, dtype=input.dtype, device=input.device)
    
    for token_idx in range(M):
        token_input = input[token_idx]  # [hidden_dim]
        
        for k in range(topk):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()
            
            # FC1: [intermediate*2, hidden_dim] @ [hidden_dim] → [intermediate*2]
            fc1_out = torch.mv(fc1_weights[expert_idx], token_input)
            
            # SwiGLU activation: split into gate and up, apply silu to gate
            intermediate_dim = fc1_out.shape[0] // 2
            gate = fc1_out[:intermediate_dim]
            up = fc1_out[intermediate_dim:]
            activated = F.silu(gate) * up  # [intermediate_dim]
            
            # FC2: [hidden_dim, intermediate] @ [intermediate] → [hidden_dim]
            fc2_out = torch.mv(fc2_weights[expert_idx], activated)
            
            output[token_idx] += scale * fc2_out
    
    return output


def moe_gemv_batched(
    input: torch.Tensor,                    # [M, hidden_dim] BF16
    token_selected_experts: torch.Tensor,   # [M, topk] int
    token_final_scales: torch.Tensor,       # [M, topk] float32
    fc1_weights: torch.Tensor,              # [num_experts, intermediate*2, hidden_dim] BF16
    fc2_weights: torch.Tensor,              # [num_experts, hidden_dim, intermediate] BF16
) -> torch.Tensor:
    """
    Optimized GEMV using batched matrix multiply.
    
    Gathers expert weights and uses torch.bmm for better parallelism.
    """
    M, hidden_dim = input.shape
    topk = token_selected_experts.shape[1]
    intermediate_dim = fc1_weights.shape[1] // 2
    
    # Flatten to process all (token, expert) pairs at once
    # Total pairs: M * topk
    flat_experts = token_selected_experts.flatten()  # [M * topk]
    flat_scales = token_final_scales.flatten()       # [M * topk]
    
    # Expand input for each expert selection
    # [M, hidden_dim] -> [M * topk, hidden_dim]
    expanded_input = input.repeat_interleave(topk, dim=0)
    
    # Gather FC1 weights for selected experts
    # [M * topk, intermediate*2, hidden_dim]
    fc1_selected = fc1_weights[flat_experts]
    
    # Batched FC1: [M*topk, intermediate*2, hidden_dim] @ [M*topk, hidden_dim, 1]
    fc1_out = torch.bmm(fc1_selected, expanded_input.unsqueeze(-1)).squeeze(-1)
    
    # SwiGLU
    gate = fc1_out[:, :intermediate_dim]
    up = fc1_out[:, intermediate_dim:]
    activated = F.silu(gate) * up  # [M * topk, intermediate]
    
    # Gather FC2 weights
    fc2_selected = fc2_weights[flat_experts]  # [M * topk, hidden_dim, intermediate]
    
    # Batched FC2
    fc2_out = torch.bmm(fc2_selected, activated.unsqueeze(-1)).squeeze(-1)
    
    # Scale and reduce
    fc2_out = fc2_out * flat_scales.unsqueeze(-1)
    
    # Reshape and sum across topk
    fc2_out = fc2_out.view(M, topk, hidden_dim)
    output = fc2_out.sum(dim=1)
    
    return output


# ============================================================================
# CUTLASS Path (via FlashInfer)
# ============================================================================
def get_cutlass_moe():
    """Get FlashInfer's CUTLASS MoE function if available."""
    try:
        from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
        from flashinfer import mxfp8_quantize
        return flashinfer_cutlass_fused_moe, mxfp8_quantize
    except ImportError as e:
        print(f"Warning: Could not import FlashInfer CUTLASS MoE: {e}")
        return None, None


# ============================================================================
# Benchmark Functions
# ============================================================================
def benchmark_fn(fn, warmup=3, iters=20):
    """Benchmark a function and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return (elapsed / iters) * 1000  # ms


def create_test_tensors(M: int, dtype=torch.bfloat16):
    """Create test tensors for benchmarking."""
    # Input activation
    x = torch.randn(M, HIDDEN_DIM, dtype=dtype, device=device)
    
    # Expert routing
    topk_indices = torch.randint(0, NUM_EXPERTS, (M, TOPK), device=device)
    topk_weights = torch.softmax(torch.randn(M, TOPK, device=device), dim=-1).float()
    
    # FC1 weights: [num_experts, intermediate*2, hidden_dim] for SwiGLU
    fc1_weights = torch.randn(
        NUM_EXPERTS, INTERMEDIATE_DIM * 2, HIDDEN_DIM, 
        dtype=dtype, device=device
    ) * 0.01
    
    # FC2 weights: [num_experts, hidden_dim, intermediate]
    fc2_weights = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
        dtype=dtype, device=device
    ) * 0.01
    
    return x, topk_indices, topk_weights, fc1_weights, fc2_weights


def main():
    print(f"\nConfiguration:")
    print(f"  Experts: {NUM_EXPERTS}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Intermediate dim: {INTERMEDIATE_DIM}")
    print(f"  TopK: {TOPK}")
    print(f"  Layers: {NUM_LAYERS}")
    print()
    
    # Test batch sizes
    M_values = [1, 2, 4, 8, 16, 32, 64]
    
    # Create weights once (large tensors)
    print("Creating test weights...")
    _, _, _, fc1_weights, fc2_weights = create_test_tensors(1)
    
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"{'M':>4} | {'GEMV Loop':>12} | {'GEMV Batched':>12} | {'Per-Token':>12} | {'Proj tok/s':>12}")
    print("-" * 70)
    
    for M in M_values:
        # Create inputs for this batch size
        x, topk_indices, topk_weights, _, _ = create_test_tensors(M)
        
        # Benchmark GEMV loop
        def run_gemv_loop():
            return moe_gemv_fallback(x, topk_indices, topk_weights, fc1_weights, fc2_weights)
        
        # Benchmark GEMV batched
        def run_gemv_batched():
            return moe_gemv_batched(x, topk_indices, topk_weights, fc1_weights, fc2_weights)
        
        # Verify correctness
        out_loop = run_gemv_loop()
        out_batched = run_gemv_batched()
        max_diff = (out_loop - out_batched).abs().max().item()
        assert max_diff < 0.1, f"Outputs differ by {max_diff}"
        
        # Benchmark
        time_loop = benchmark_fn(run_gemv_loop)
        time_batched = benchmark_fn(run_gemv_batched)
        
        # Calculate per-token time (assuming M=1 decode)
        # Full decode: NUM_LAYERS layers * 2 MoE calls per layer
        moe_calls_per_token = NUM_LAYERS * 2
        per_token_ms = time_batched * moe_calls_per_token / M
        projected_tok_per_sec = 1000 / per_token_ms if per_token_ms > 0 else 0
        
        print(f"{M:>4} | {time_loop:>10.3f}ms | {time_batched:>10.3f}ms | {per_token_ms:>10.1f}ms | {projected_tok_per_sec:>10.1f}")
    
    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)
    
    # Single comparison at M=1
    x1, topk1, weights1, _, _ = create_test_tensors(1)
    
    time_gemv = benchmark_fn(
        lambda: moe_gemv_batched(x1, topk1, weights1, fc1_weights, fc2_weights),
        warmup=5, iters=50
    )
    
    print(f"\nM=1 GEMV Batched: {time_gemv:.3f} ms per MoE call")
    print(f"Per-token (60 layers × 2): {time_gemv * 120:.1f} ms")
    print(f"Projected throughput: {1000 / (time_gemv * 120):.1f} tok/s")
    
    # Compare with current performance
    print(f"\nCurrent vLLM CUTLASS: ~29 tok/s")
    print(f"Target (llama.cpp): ~58 tok/s")
    
    # Memory usage
    fc1_mem = fc1_weights.numel() * 2 / 1e9  # BF16 = 2 bytes
    fc2_mem = fc2_weights.numel() * 2 / 1e9
    print(f"\nWeight memory (BF16): FC1={fc1_mem:.2f}GB, FC2={fc2_mem:.2f}GB")
    print("Note: Actual model uses MXFP4 (4x smaller)")


if __name__ == "__main__":
    main()

