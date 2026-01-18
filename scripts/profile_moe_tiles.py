#!/usr/bin/env python3
"""Profile MoE kernel with different batch sizes to see tile selection and performance."""

import torch
import time
import os

# Ensure we use local FlashInfer
os.environ.setdefault("PYTHONPATH", "/workspace/flashinfer:/workspace/vllm")

def profile_moe_kernel():
    """Profile MoE kernel across different batch sizes."""
    
    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: {major}.{minor}")
    print()
    
    if major != 12:
        print("This script is for SM12x (Blackwell) GPUs")
        return
    
    # Import FlashInfer
    try:
        from flashinfer.fused_moe.core import select_tile_mn_for_sm120, SM120_SUPPORTED_TILE_MN
        from flashinfer import mxfp4_quantize, mxfp8_quantize
    except ImportError as e:
        print(f"Failed to import FlashInfer: {e}")
        print("Make sure PYTHONPATH includes /workspace/flashinfer")
        return
    
    print("=" * 70)
    print("SUPPORTED TILES:")
    print("=" * 70)
    for tile in sorted(SM120_SUPPORTED_TILE_MN):
        print(f"  {tile[0]:3d} x {tile[1]:3d}")
    print()
    
    # Model dimensions (GPT-OSS-120B MoE)
    hidden_size = 7168
    intermediate_size = 3584  # Per expert
    num_experts = 128
    top_k = 8
    
    print("=" * 70)
    print("MODEL DIMENSIONS (GPT-OSS-120B MoE):")
    print("=" * 70)
    print(f"  Hidden size:       {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Num experts:       {num_experts}")
    print(f"  Top-K:             {top_k}")
    print()
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    print("=" * 70)
    print("TILE SELECTION BY BATCH SIZE:")
    print("=" * 70)
    print(f"{'Batch':<8} {'Tokens':<10} {'Tile M×N':<12} {'Schedule (expected)':<20}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        # In decode, tokens = batch_size * top_k (each token routes to top_k experts)
        num_tokens = batch_size * top_k
        tile_m, tile_n = select_tile_mn_for_sm120(num_tokens)
        
        # Expected schedule based on tile size
        schedule = "PingPong" if tile_m == 64 else "Cooperative"
        
        print(f"{batch_size:<8} {num_tokens:<10} {tile_m:3d}×{tile_n:<3d}       {schedule:<20}")
    
    print()
    
    # Benchmark actual kernel execution
    print("=" * 70)
    print("KERNEL PROFILING (warm-up + 10 iterations):")
    print("=" * 70)
    
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
    except ImportError as e:
        print(f"Cannot import cutlass_fused_moe: {e}")
        return
    
    # Create test weights (FP4 quantized)
    print("\nPreparing quantized weights...")
    
    # FC1: hidden -> intermediate*2 (for SwiGLU)
    fc1_bf16 = torch.randn(num_experts, intermediate_size * 2, hidden_size, 
                           dtype=torch.bfloat16, device=device)
    fc1_fp4, fc1_scale = mxfp4_quantize(fc1_bf16.view(num_experts, -1))
    fc1_fp4 = fc1_fp4.view(num_experts, intermediate_size * 2, -1)
    fc1_scale = fc1_scale.view(num_experts, intermediate_size * 2, -1)
    
    # FC2: intermediate -> hidden
    fc2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                           dtype=torch.bfloat16, device=device)
    fc2_fp4, fc2_scale = mxfp4_quantize(fc2_bf16.view(num_experts, -1))
    fc2_fp4 = fc2_fp4.view(num_experts, hidden_size, -1)
    fc2_scale = fc2_scale.view(num_experts, hidden_size, -1)
    
    # quant_scales format: [fc1_weight_scale, fc2_weight_scale, fc1_act_scale, fc2_act_scale]
    # For MXFP4, we use identity activation scales (all 1s encoded as 0x7F in E8M0 format)
    print("Weights quantized to MXFP4")
    print()
    
    results = []
    
    for batch_size in [1, 4, 16, 64, 256]:
        num_tokens = batch_size
        
        # Create input activations in BF16, then quantize to FP8 for MXFP4 kernel
        hidden_states_bf16 = torch.randn(num_tokens, hidden_size, 
                                         dtype=torch.bfloat16, device=device)
        
        # Quantize activations to FP8 (the kernel expects FP8×FP4)
        hidden_states_fp8, input_scale = mxfp8_quantize(hidden_states_bf16)
        
        # Create routing (random expert assignment)
        topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), 
                                     dtype=torch.int32, device=device)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
        
        # Get expected tile
        tile_m, tile_n = select_tile_mn_for_sm120(num_tokens * top_k)
        
        # quant_scales: [fc1_w_scale, fc2_w_scale, fc1_act_scale, fc2_act_scale]
        quant_scales = [fc1_scale, fc2_scale, input_scale, None]
        
        # Warm-up
        torch.cuda.synchronize()
        for _ in range(3):
            try:
                output = cutlass_fused_moe(
                    input=hidden_states_fp8,
                    token_selected_experts=topk_indices,
                    token_final_scales=topk_weights,
                    fc1_expert_weights=fc1_fp4,
                    fc2_expert_weights=fc2_fp4,
                    output_dtype=torch.bfloat16,
                    quant_scales=quant_scales,
                    activation_type=ActivationType.Swiglu,
                    use_w4_group_scaling=True,  # Enable MXFP4 (FP8xFP4) mode
                    use_mxfp8_act_scaling=True,  # Enable MXFP8 activation scaling
                )
            except Exception as e:
                print(f"Batch {batch_size}: FAILED - {e}")
                break
        else:
            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            iterations = 10
            for _ in range(iterations):
                output = cutlass_fused_moe(
                    input=hidden_states_fp8,
                    token_selected_experts=topk_indices,
                    token_final_scales=topk_weights,
                    fc1_expert_weights=fc1_fp4,
                    fc2_expert_weights=fc2_fp4,
                    output_dtype=torch.bfloat16,
                    quant_scales=quant_scales,
                    activation_type=ActivationType.Swiglu,
                    use_w4_group_scaling=True,
                    use_mxfp8_act_scaling=True,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            avg_ms = (elapsed / iterations) * 1000
            
            results.append({
                'batch': batch_size,
                'tokens': num_tokens * top_k,
                'tile': f"{tile_m}×{tile_n}",
                'time_ms': avg_ms,
            })
            
            print(f"Batch {batch_size:3d}: tile={tile_m}×{tile_n}, "
                  f"time={avg_ms:.3f}ms, "
                  f"throughput={num_tokens/avg_ms*1000:.0f} tok/s")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        print(f"{'Batch':<8} {'Tokens':<8} {'Tile':<10} {'Time (ms)':<12} {'Throughput':<12}")
        print("-" * 70)
        for r in results:
            throughput = r['batch'] / r['time_ms'] * 1000
            print(f"{r['batch']:<8} {r['tokens']:<8} {r['tile']:<10} {r['time_ms']:<12.3f} {throughput:<12.0f}")


if __name__ == "__main__":
    profile_moe_kernel()
