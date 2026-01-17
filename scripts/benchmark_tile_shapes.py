#!/usr/bin/env python3
"""Benchmark different tile shapes for SM120 MoE GEMM.

This script directly calls the CUTLASS kernel with different tile configurations
to measure performance impact of tile shape selection.

Usage:
    python scripts/benchmark_tile_shapes.py [--tiles 128x128,64x128,128x64]
"""

import argparse
import time
import torch
from flashinfer import mxfp8_quantize, mxfp4_quantize


def get_module_for_tile(tile_m: int, tile_n: int):
    """Get JIT-compiled module for specific tile shape."""
    # Import here to avoid caching issues
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    # Clear the cache for this specific tile
    # get_cutlass_fused_moe_module.cache_clear()
    
    module = get_cutlass_fused_moe_module(
        backend="120",
        use_fast_build=False,
        tile_mn=(tile_m, tile_n),
    )
    return module


def benchmark_tile(tile_m: int, tile_n: int, num_tokens: int, hidden_size: int, 
                   inter_size: int, num_experts: int, num_warmup: int = 3, 
                   num_runs: int = 10):
    """Benchmark a specific tile shape."""
    from flashinfer.fused_moe.core import ActivationType
    
    mxfp4_block = 32
    device = "cuda"
    
    # Create weights
    weight_bf16 = torch.randn(num_experts, inter_size * 2, hidden_size, 
                               dtype=torch.bfloat16, device=device) * 0.01
    fc1_w_fp4, fc1_w_sf = mxfp4_quantize(weight_bf16)
    fc1_w_packed = fc1_w_fp4.view(torch.long)
    
    weight2_bf16 = torch.randn(num_experts, hidden_size, inter_size, 
                                dtype=torch.bfloat16, device=device) * 0.01
    fc2_w_fp4, fc2_w_sf = mxfp4_quantize(weight2_bf16)
    fc2_w_packed = fc2_w_fp4.view(torch.long)
    
    # Reshape scale factors
    fc1_sf_reshaped = fc1_w_sf.view(num_experts, inter_size * 2, hidden_size // mxfp4_block)
    fc2_sf_reshaped = fc2_w_sf.view(num_experts, hidden_size, inter_size // mxfp4_block)
    fc1_scale_int32 = fc1_sf_reshaped.contiguous().view(torch.int32)
    fc2_scale_int32 = fc2_sf_reshaped.contiguous().view(torch.int32)
    
    # Create activations
    act_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    act_fp8, act_sf = mxfp8_quantize(act_bf16, True, 32)
    
    # Routing
    topk = 2
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(num_tokens, topk, device=device), dim=-1).float()
    
    # Scale factors
    fake_global = torch.ones(num_experts, dtype=torch.float32, device=device)
    quant_scales = [fc1_scale_int32, fake_global, fc2_scale_int32, fake_global]
    
    # Get module for this tile
    try:
        module = get_module_for_tile(tile_m, tile_n)
    except Exception as e:
        return None, str(e)
    
    # Import the actual kernel call
    from flashinfer.fused_moe.core import cutlass_fused_moe
    
    # Warmup
    for _ in range(num_warmup):
        try:
            output = cutlass_fused_moe(
                input=act_fp8,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=fc1_w_packed,
                fc2_expert_weights=fc2_w_packed,
                output_dtype=torch.bfloat16,
                activation_type=ActivationType.Swiglu,
                use_mxfp8_act_scaling=True,
                input_sf=act_sf,
                quant_scales=quant_scales,
            )
            torch.cuda.synchronize()
        except Exception as e:
            return None, str(e)
    
    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = cutlass_fused_moe(
            input=act_fp8,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_w_packed,
            fc2_expert_weights=fc2_w_packed,
            output_dtype=torch.bfloat16,
            activation_type=ActivationType.Swiglu,
            use_mxfp8_act_scaling=True,
            input_sf=act_sf,
            quant_scales=quant_scales,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    return avg_time, None


def main():
    parser = argparse.ArgumentParser(description="Benchmark tile shapes")
    parser.add_argument("--tiles", type=str, default="128x128,64x128,128x64,64x64",
                        help="Comma-separated tile shapes (MxN)")
    parser.add_argument("--tokens", type=str, default="1,4,16,64,128",
                        help="Comma-separated token counts to test")
    parser.add_argument("--hidden-size", type=int, default=2944,
                        help="Hidden size (default: 2944 for gpt-oss-120b)")
    parser.add_argument("--inter-size", type=int, default=7680,
                        help="Intermediate size (default: 7680)")
    parser.add_argument("--num-experts", type=int, default=128,
                        help="Number of experts (default: 128)")
    args = parser.parse_args()
    
    tiles = [tuple(map(int, t.split("x"))) for t in args.tiles.split(",")]
    token_counts = [int(t) for t in args.tokens.split(",")]
    
    print("=" * 70)
    print("SM120 MoE Tile Shape Benchmark")
    print("=" * 70)
    print(f"Hidden size: {args.hidden_size}")
    print(f"Intermediate size: {args.inter_size}")
    print(f"Num experts: {args.num_experts}")
    print(f"Tiles to test: {tiles}")
    print(f"Token counts: {token_counts}")
    print()
    
    # Results table
    results = {}
    
    for tile_m, tile_n in tiles:
        print(f"\n--- Testing tile ({tile_m}, {tile_n}) ---")
        results[(tile_m, tile_n)] = {}
        
        for num_tokens in token_counts:
            avg_time, error = benchmark_tile(
                tile_m, tile_n, num_tokens, 
                args.hidden_size, args.inter_size, args.num_experts
            )
            
            if error:
                print(f"  tokens={num_tokens:4d}: ERROR - {error[:50]}")
                results[(tile_m, tile_n)][num_tokens] = None
            else:
                throughput = num_tokens / avg_time if avg_time > 0 else 0
                print(f"  tokens={num_tokens:4d}: {avg_time*1000:7.2f}ms ({throughput:8.1f} tok/s)")
                results[(tile_m, tile_n)][num_tokens] = avg_time
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Time in ms (lower is better)")
    print("=" * 70)
    
    # Header
    header = "Tile      |" + "|".join(f" {t:>6} tok" for t in token_counts)
    print(header)
    print("-" * len(header))
    
    for tile_m, tile_n in tiles:
        row = f"({tile_m:3d},{tile_n:3d}) |"
        for num_tokens in token_counts:
            time_val = results[(tile_m, tile_n)].get(num_tokens)
            if time_val is None:
                row += "    ERROR |"
            else:
                row += f" {time_val*1000:8.2f} |"
        print(row)
    
    print()
    print("Best tile per token count:")
    for num_tokens in token_counts:
        best_tile = None
        best_time = float('inf')
        for tile_m, tile_n in tiles:
            time_val = results[(tile_m, tile_n)].get(num_tokens)
            if time_val is not None and time_val < best_time:
                best_time = time_val
                best_tile = (tile_m, tile_n)
        if best_tile:
            print(f"  tokens={num_tokens:4d}: {best_tile} ({best_time*1000:.2f}ms)")


if __name__ == "__main__":
    main()
