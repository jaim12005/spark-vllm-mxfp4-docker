#!/usr/bin/env python3
"""
Baseline benchmark for vLLM with MXFP4 on SM121.

Configuration:
- CUTLASS grouped GEMM (our kernel)
- No speculative decoding
- No CUDA graphs (--enforce-eager)
- Standard attention (FA2)

This establishes our baseline before any optimizations.
"""

import argparse
import time
import torch
from openai import OpenAI


def benchmark_decode(
    client: OpenAI,
    model: str,
    max_tokens: int,
    prompt: str,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> dict:
    """Benchmark decode throughput."""
    
    # Warmup
    for _ in range(warmup_runs):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    
    # Benchmark runs
    times = []
    tokens = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start
        
        output_tokens = response.usage.completion_tokens
        times.append(elapsed)
        tokens.append(output_tokens)
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens) / len(tokens)
    throughput = avg_tokens / avg_time
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "throughput": throughput,
        "times": times,
        "tokens": tokens,
    }


def benchmark_prefill(
    client: OpenAI,
    model: str,
    prompt_tokens: int,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> dict:
    """Benchmark prefill throughput."""
    
    # Generate a long prompt
    base_text = "The quick brown fox jumps over the lazy dog. "
    prompt = base_text * (prompt_tokens // 10)  # ~10 tokens per sentence
    
    # Warmup
    for _ in range(warmup_runs):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,  # Just prefill
            temperature=0.0,
        )
    
    # Benchmark runs
    times = []
    input_tokens = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start
        
        prompt_tokens_actual = response.usage.prompt_tokens
        times.append(elapsed)
        input_tokens.append(prompt_tokens_actual)
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(input_tokens) / len(input_tokens)
    throughput = avg_tokens / avg_time
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "throughput": throughput,
        "times": times,
        "input_tokens": input_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline vLLM benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    args = parser.parse_args()
    
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    
    print("=" * 60)
    print("vLLM Baseline Benchmark (SM121 + MXFP4)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: CUTLASS grouped GEMM, no spec decode, eager mode")
    print()
    
    # Decode benchmark
    print(f"[1/2] Decode benchmark (max_tokens={args.max_tokens})...")
    prompt = "Write a detailed essay about the history of artificial intelligence."
    decode_results = benchmark_decode(
        client, args.model, args.max_tokens, prompt,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    print(f"  Throughput: {decode_results['throughput']:.2f} tok/s")
    print(f"  Avg tokens: {decode_results['avg_tokens']:.0f}")
    print(f"  Avg time:   {decode_results['avg_time']:.3f}s")
    print()
    
    # Prefill benchmark
    print(f"[2/2] Prefill benchmark (prompt_tokens={args.prefill_tokens})...")
    prefill_results = benchmark_prefill(
        client, args.model, args.prefill_tokens,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    print(f"  Throughput: {prefill_results['throughput']:.2f} tok/s")
    print(f"  Avg tokens: {prefill_results['avg_tokens']:.0f}")
    print(f"  Avg time:   {prefill_results['avg_time']:.3f}s")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Decode:  {decode_results['throughput']:.2f} tok/s (target: 58 tok/s llama.cpp)")
    print(f"Prefill: {prefill_results['throughput']:.2f} tok/s (target: 2450 tok/s llama.cpp)")
    print()
    
    # Gap analysis
    decode_gap = 58 / decode_results['throughput'] if decode_results['throughput'] > 0 else float('inf')
    prefill_gap = prefill_results['throughput'] / 2450 if prefill_results['throughput'] > 0 else 0
    print(f"Decode gap:  {decode_gap:.2f}x slower than llama.cpp")
    print(f"Prefill:     {prefill_gap:.2f}x faster than llama.cpp")


if __name__ == "__main__":
    main()


