#!/usr/bin/env python3
"""Simple MoE GEMM timing without profiler.

This gives us accurate timing for the MoE kernel calls.
"""
import torch
import time
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
top_k = 8

def create_moe_inputs(num_tokens: int):
    """Create properly quantized MoE inputs."""
    x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
    
    w13_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                           dtype=torch.bfloat16, device=device) * 0.01
    w2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                          dtype=torch.bfloat16, device=device) * 0.01
    
    w13_flat = w13_bf16.reshape(-1, hidden_size)
    w2_flat = w2_bf16.reshape(-1, intermediate_size)
    w13_fp4, w13_scale = mxfp4_quantize(w13_flat)
    w2_fp4, w2_scale = mxfp4_quantize(w2_flat)
    
    w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
    w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)
    w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
    w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)
    
    fc1_weights = w13_fp4.contiguous().view(torch.long)
    fc2_weights = w2_fp4.contiguous().view(torch.long)
    fc1_scale = w13_scale.contiguous().view(torch.int32)
    fc2_scale = w2_scale.contiguous().view(torch.int32)
    
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).float()
    topk_ids = topk_ids.to(torch.int32)
    
    fake_input_scale = torch.ones(num_experts, device=device)
    quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]
    
    return x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales

def benchmark_moe(num_tokens: int, num_warmup: int = 5, num_runs: int = 20):
    """Benchmark MoE kernel."""
    x_quant, x_scale, fc1_weights, fc2_weights, topk_ids, topk_weights, quant_scales = create_moe_inputs(num_tokens)
    
    # Warmup
    for _ in range(num_warmup):
        _ = cutlass_fused_moe(
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
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    for i in range(num_runs):
        start_events[i].record()
        _ = cutlass_fused_moe(
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
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return avg_time, min_time, max_time

def main():
    print("=" * 70)
    print("MoE GEMM Kernel Timing (gpt-oss-120b dimensions)")
    print("=" * 70)
    print(f"Config: hidden={hidden_size}, inter={intermediate_size}, experts={num_experts}, top_k={top_k}")
    print()
    
    results = {}
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128]:
        avg, min_t, max_t = benchmark_moe(num_tokens)
        results[num_tokens] = (avg, min_t, max_t)
        tok_per_sec = num_tokens / (avg / 1000)  # tokens per second
        print(f"tokens={num_tokens:3d}: avg={avg:.3f}ms, min={min_t:.3f}ms, max={max_t:.3f}ms, {tok_per_sec:.0f} tok/s")
    
    print()
    print("=" * 70)
    print("Analysis for Tile Selection")
    print("=" * 70)
    
    # The tile selection uses 64x128 for < 64 tokens, 128x128 for >= 64
    # Compare performance at the threshold
    if 64 in results and 32 in results:
        time_32 = results[32][0]
        time_64 = results[64][0]
        print(f"At 32 tokens (uses 64x128): {time_32:.3f}ms")
        print(f"At 64 tokens (uses 128x128): {time_64:.3f}ms")
        print(f"Ratio: {time_64/time_32:.2f}x (expected ~2x for double tokens)")
    
    # Estimate MoE contribution to decode
    # A typical decode step includes attention + MoE + other ops
    # MoE typically dominates because it's compute-heavy
    print()
    print("=" * 70)
    print("MoE Decode Contribution Estimate")
    print("=" * 70)
    
    # For single-token decode (most common)
    if 1 in results:
        moe_time = results[1][0]
        # Typical vLLM decode step for gpt-oss-120b is ~30-50ms
        # MoE is called once per layer (40 layers in gpt-oss-120b)
        # But this is fused MoE (both up/gate and down projections)
        # So ~40 MoE calls per decode step
        num_layers = 40
        total_moe = moe_time * num_layers
        print(f"Single MoE call: {moe_time:.3f}ms")
        print(f"Total MoE time ({num_layers} layers): {total_moe:.1f}ms")
        print(f"If decode step is ~30ms: MoE = {total_moe/30*100:.0f}%")
        print(f"If decode step is ~50ms: MoE = {total_moe/50*100:.0f}%")

if __name__ == "__main__":
    main()
