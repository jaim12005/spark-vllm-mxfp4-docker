#!/usr/bin/env python3
"""Profile decode-only workload for kernel analysis."""

import argparse
import time
import torch
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Prompt tokens: {args.prompt_tokens}, Output tokens: {args.output_tokens}")
    
    llm = LLM(
        model=args.model,
        quantization="mxfp4",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=131072,
        max_num_seqs=2,
        max_num_batched_tokens=8192,
        enforce_eager=True,  # Disable CUDA graphs for accurate profiling
        enable_prefix_caching=True,
    )
    
    # Create a prompt of approximately the right length
    # Using repeated text to reach target token count
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode(base_prompt)
    
    # Adjust prompt to exact token count
    if len(tokens) < args.prompt_tokens:
        # Repeat until we have enough
        multiplier = (args.prompt_tokens // len(tokens)) + 1
        base_prompt = base_prompt * multiplier
        tokens = tokenizer.encode(base_prompt)
    
    # Truncate to exact token count
    prompt = tokenizer.decode(tokens[:args.prompt_tokens])
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"Actual prompt tokens: {actual_tokens}")
    
    sampling_params = SamplingParams(
        max_tokens=args.output_tokens,
        temperature=0.0,  # Greedy for determinism
    )
    
    # Warmup runs
    print(f"\nWarmup: {args.warmup} runs")
    for i in range(args.warmup):
        outputs = llm.generate([prompt], sampling_params)
        print(f"  Warmup {i+1}: {len(outputs[0].outputs[0].token_ids)} tokens generated")
    
    # Sync before profiling
    torch.cuda.synchronize()
    
    # Profiled runs
    print(f"\nProfiled runs: {args.runs}")
    total_time = 0
    total_tokens = 0
    
    for i in range(args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = llm.generate([prompt], sampling_params)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        gen_tokens = len(outputs[0].outputs[0].token_ids)
        tps = gen_tokens / elapsed
        total_time += elapsed
        total_tokens += gen_tokens
        
        print(f"  Run {i+1}: {gen_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    
    avg_tps = total_tokens / total_time
    print(f"\nAverage: {avg_tps:.1f} tok/s")
    print(f"Output sample: {outputs[0].outputs[0].text[:100]}...")

if __name__ == "__main__":
    main()
