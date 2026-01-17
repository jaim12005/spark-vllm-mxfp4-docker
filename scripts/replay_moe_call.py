#!/usr/bin/env python3
"""
Replay a captured vLLM MoE call for debugging and profiling.

Usage:
    python3 replay_moe_call.py /tmp/moe_call_*.pt

To capture a call, start vLLM with:
    VLLM_CAPTURE_MOE_CALL=1 vllm serve ...

Then send a request. The first MoE call will be saved to /tmp/moe_call_*.pt
"""
import sys
import torch
import argparse

sys.path.insert(0, '/workspace/flashinfer')

from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType


def replay_call(capture_path: str, weights_path: str = None, verbose: bool = True):
    """Replay a captured MoE call.
    
    Args:
        capture_path: Path to input capture (.pt file with fi_input, topk_ids, etc.)
                     OR legacy full capture with all data
        weights_path: Path to weights file (optional, for split captures)
    """
    print(f"Loading capture from: {capture_path}")
    data = torch.load(capture_path, map_location="cpu")
    
    # Check if this is a split capture (no fc1_expert_weights in data)
    if "fc1_expert_weights" not in data:
        if weights_path is None:
            weights_path = "/tmp/moe_weights.pt"
        print(f"Loading weights from: {weights_path}")
        weights = torch.load(weights_path, map_location="cpu")
        data.update(weights)
    
    device = "cuda"
    
    # Print captured shapes
    if verbose:
        print("\n=== Captured tensors ===")
        print(f"fi_input: shape={data['fi_input'].shape}, dtype={data['fi_input'].dtype}")
        print(f"topk_ids: shape={data['topk_ids'].shape}, dtype={data['topk_ids'].dtype}")
        print(f"topk_weights: shape={data['topk_weights'].shape}, dtype={data['topk_weights'].dtype}")
        print(f"fc1_expert_weights: shape={data['fc1_expert_weights'].shape}, dtype={data['fc1_expert_weights'].dtype}")
        print(f"fc2_expert_weights: shape={data['fc2_expert_weights'].shape}, dtype={data['fc2_expert_weights'].dtype}")
        if data['input_sf'] is not None:
            print(f"input_sf: shape={data['input_sf'].shape}, dtype={data['input_sf'].dtype}")
        print(f"\nquant_scales: {len(data['quant_scales'])} tensors")
        for i, qs in enumerate(data['quant_scales']):
            print(f"  [{i}]: shape={qs.shape}, dtype={qs.dtype}")
        print(f"\nMetadata:")
        print(f"  hidden_size: {data['hidden_size']}")
        print(f"  original_hidden_size: {data['original_hidden_size']}")
        print(f"  intermediate_size: {data['intermediate_size']}")
        print(f"  num_experts: {data['num_experts']}")
        print(f"  use_mxfp8_act_scaling: {data['use_mxfp8_act_scaling']}")
    
    # Move tensors to GPU
    fi_input = data['fi_input'].to(device)
    topk_ids = data['topk_ids'].to(device)
    topk_weights = data['topk_weights'].to(device)
    quant_scales = [qs.to(device) for qs in data['quant_scales']]
    fc1_expert_weights = data['fc1_expert_weights'].to(device)
    fc2_expert_weights = data['fc2_expert_weights'].to(device)
    input_sf = data['input_sf'].to(device) if data['input_sf'] is not None else None
    fc1_expert_biases = data['fc1_expert_biases'].to(device) if data['fc1_expert_biases'] is not None else None
    fc2_expert_biases = data['fc2_expert_biases'].to(device) if data['fc2_expert_biases'] is not None else None
    
    hidden_size = data['hidden_size']
    use_mxfp8_act_scaling = data['use_mxfp8_act_scaling']
    
    # Create output tensor
    output_shape = list(fi_input.shape)
    output_shape[-1] = hidden_size
    output = torch.empty(output_shape, device=device, dtype=torch.bfloat16)
    
    print(f"\n=== Calling cutlass_fused_moe ===")
    
    # Build kwargs
    extra_kwargs = {}
    if use_mxfp8_act_scaling:
        extra_kwargs["use_mxfp8_act_scaling"] = True
        extra_kwargs["input_sf"] = input_sf
    
    try:
        result = cutlass_fused_moe(
            input=fi_input,
            token_selected_experts=topk_ids.to(torch.int).contiguous(),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=quant_scales,
            fc1_expert_biases=fc1_expert_biases,
            fc2_expert_biases=fc2_expert_biases,
            activation_type=ActivationType.Swiglu,
            **extra_kwargs,
        )
        
        print(f"\n=== SUCCESS ===")
        print(f"Output shape: {result[0].shape}")
        print(f"Output sample (first 8): {result[0][0, :8]}")
        print(f"Output sample (last 8): {result[0][0, -8:]}")
        print(f"Has NaN: {torch.isnan(result[0]).any()}")
        print(f"Has Inf: {torch.isinf(result[0]).any()}")
        print(f"Output min: {result[0].min().item():.6f}, max: {result[0].max().item():.6f}")
        
        # Check if values are in reasonable range
        max_abs = result[0].abs().max().item()
        if max_abs < 1000:
            print(f"Values in reasonable range (max_abs={max_abs:.2f}) - LOOKS GOOD")
        else:
            print(f"Values EXTREME (max_abs={max_abs:.2e}) - potential issue")
            
        return result[0]
        
    except Exception as e:
        print(f"\n=== FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_call(capture_path: str, warmup: int = 5, iterations: int = 20):
    """Benchmark the replayed call."""
    import time
    
    print(f"\n=== Benchmarking ===")
    data = torch.load(capture_path, map_location="cpu")
    device = "cuda"
    
    # Move tensors to GPU
    fi_input = data['fi_input'].to(device)
    topk_ids = data['topk_ids'].to(device)
    topk_weights = data['topk_weights'].to(device)
    quant_scales = [qs.to(device) for qs in data['quant_scales']]
    fc1_expert_weights = data['fc1_expert_weights'].to(device)
    fc2_expert_weights = data['fc2_expert_weights'].to(device)
    input_sf = data['input_sf'].to(device) if data['input_sf'] is not None else None
    fc1_expert_biases = data['fc1_expert_biases'].to(device) if data['fc1_expert_biases'] is not None else None
    fc2_expert_biases = data['fc2_expert_biases'].to(device) if data['fc2_expert_biases'] is not None else None
    
    hidden_size = data['hidden_size']
    use_mxfp8_act_scaling = data['use_mxfp8_act_scaling']
    
    output_shape = list(fi_input.shape)
    output_shape[-1] = hidden_size
    output = torch.empty(output_shape, device=device, dtype=torch.bfloat16)
    
    extra_kwargs = {}
    if use_mxfp8_act_scaling:
        extra_kwargs["use_mxfp8_act_scaling"] = True
        extra_kwargs["input_sf"] = input_sf
    
    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        cutlass_fused_moe(
            input=fi_input,
            token_selected_experts=topk_ids.to(torch.int).contiguous(),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=quant_scales,
            fc1_expert_biases=fc1_expert_biases,
            fc2_expert_biases=fc2_expert_biases,
            activation_type=ActivationType.Swiglu,
            **extra_kwargs,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        cutlass_fused_moe(
            input=fi_input,
            token_selected_experts=topk_ids.to(torch.int).contiguous(),
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_expert_weights,
            fc2_expert_weights=fc2_expert_weights,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=quant_scales,
            fc1_expert_biases=fc1_expert_biases,
            fc2_expert_biases=fc2_expert_biases,
            activation_type=ActivationType.Swiglu,
            **extra_kwargs,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / iterations) * 1000
    num_tokens = fi_input.shape[0]
    print(f"\nResults:")
    print(f"  Total time: {elapsed*1000:.2f} ms for {iterations} iterations")
    print(f"  Average: {avg_ms:.3f} ms per call")
    print(f"  Tokens: {num_tokens}")
    print(f"  Throughput: {num_tokens / (avg_ms / 1000):.1f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay captured vLLM MoE call")
    parser.add_argument("capture_path", help="Path to captured .pt file (input or full)")
    parser.add_argument("--weights", help="Path to weights .pt file (for split captures)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()
    
    result = replay_call(args.capture_path, weights_path=args.weights)
    
    if args.benchmark and result is not None:
        benchmark_call(args.capture_path, iterations=args.iterations)
