#!/usr/bin/env python3
"""Profile MoE GEMM kernel fraction of total decode time.

This script measures what percentage of decode time is spent in MoE GEMM kernels
to determine if tile optimization is worthwhile (Section 8.0 of the plan).
"""
import torch
from torch.profiler import profile, ProfilerActivity
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

# gpt-oss-120b dimensions
hidden_size = 2944
intermediate_size = 7680
num_experts = 128
top_k = 8  # gpt-oss-120b uses top_k=8

def create_moe_inputs(num_tokens: int):
    """Create properly quantized MoE inputs."""
    # Create activations
    x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
    
    # Create and quantize weights
    w13_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                           dtype=torch.bfloat16, device=device) * 0.01
    w2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                          dtype=torch.bfloat16, device=device) * 0.01
    
    # Quantize weights
    w13_flat = w13_bf16.reshape(-1, hidden_size)
    w2_flat = w2_bf16.reshape(-1, intermediate_size)
    w13_fp4, w13_scale = mxfp4_quantize(w13_flat)
    w2_fp4, w2_scale = mxfp4_quantize(w2_flat)
    
    # Reshape
    w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
    w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)
    w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
    w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)
    
    # View as expected types
    fc1_weights = w13_fp4.contiguous().view(torch.long)
    fc2_weights = w2_fp4.contiguous().view(torch.long)
    fc1_scale = w13_scale.contiguous().view(torch.int32)
    fc2_scale = w2_scale.contiguous().view(torch.int32)
    
    # Routing - use torch.topk to avoid duplicate experts
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).float()
    topk_ids = topk_ids.to(torch.int32)
    
    # Scales
    fake_input_scale = torch.ones(num_experts, device=device)
    quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]
    
    return x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales

def run_moe(x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales):
    """Run MoE forward pass."""
    return cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True,
        input_sf=x_scale,
        quant_scales=quant_scales,
    )

def profile_moe(num_tokens: int, num_warmup: int = 3, num_runs: int = 10):
    """Profile MoE kernel and return timing breakdown."""
    print(f"\n=== Profiling with {num_tokens} tokens ===")
    
    # Create inputs
    x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales = create_moe_inputs(num_tokens)
    
    # Warmup
    for _ in range(num_warmup):
        _ = run_moe(x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales)
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_runs):
            _ = run_moe(x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales)
        torch.cuda.synchronize()
    
    # Analyze results
    events = prof.key_averages()
    
    total_cuda_time = 0
    moe_gemm_time = 0
    moe_kernels = []
    
    for evt in events:
        # Use self_cuda_time_total for CUDA kernel time
        cuda_time = getattr(evt, 'self_cuda_time_total', 0) or getattr(evt, 'cuda_time_total', 0)
        if cuda_time > 0:
            total_cuda_time += cuda_time
            name_lower = evt.key.lower()
            # Look for CUTLASS/MoE kernels
            if any(x in name_lower for x in ['cutlass', 'gemm', 'moe', 'blockscaled', 'warpspecialized', 'sm120', 'sm121']):
                moe_gemm_time += cuda_time
                moe_kernels.append((evt.key[:60], cuda_time / 1000))  # Convert to ms
    
    # Print top kernels
    print(f"\nTop CUDA kernels (total time across {num_runs} runs):")
    sorted_events = sorted(events, key=lambda x: getattr(x, 'self_cuda_time_total', 0) or 0, reverse=True)
    for evt in sorted_events[:10]:
        cuda_time = getattr(evt, 'self_cuda_time_total', 0) or 0
        if cuda_time > 0:
            print(f"  {cuda_time/1000:.2f}ms: {evt.key[:70]}")
    
    # Calculate fraction
    if total_cuda_time > 0:
        moe_fraction = moe_gemm_time / total_cuda_time * 100
        print(f"\nMoE GEMM fraction: {moe_fraction:.1f}%")
        print(f"Total CUDA time: {total_cuda_time/1000:.2f}ms over {num_runs} runs")
        print(f"MoE GEMM time: {moe_gemm_time/1000:.2f}ms")
        return moe_fraction, total_cuda_time / 1000 / num_runs
    return 0, 0

def main():
    print("=" * 70)
    print("MoE GEMM Fraction Profiling")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    
    # Profile different batch sizes
    results = {}
    for num_tokens in [1, 4, 16, 64]:
        fraction, avg_time = profile_moe(num_tokens)
        results[num_tokens] = (fraction, avg_time)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("| Tokens | MoE GEMM % | Avg Time (ms) |")
    print("|--------|------------|---------------|")
    for num_tokens, (fraction, avg_time) in results.items():
        print(f"| {num_tokens:6d} | {fraction:10.1f}% | {avg_time:13.2f} |")
    
    # Decision gate (from Section 8.0)
    avg_fraction = sum(f for f, _ in results.values()) / len(results)
    print(f"\nAverage MoE GEMM fraction: {avg_fraction:.1f}%")
    if avg_fraction < 10:
        print("❌ < 10%: Tile optimization NOT worthwhile")
    elif avg_fraction < 30:
        print("⚠️ 10-30%: Proceed cautiously, expect modest gains")
    else:
        print("✅ > 30%: Tile optimization IS worthwhile")

if __name__ == "__main__":
    main()
