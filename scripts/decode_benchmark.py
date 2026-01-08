#!/usr/bin/env python3
"""
Decode-focused benchmark for vLLM MXFP4 performance testing.

Usage:
    docker exec vllm-dev python3 /workspace/decode_benchmark.py

This script tests decode (token generation) throughput via the OpenAI API.
"""
import time
import requests
import json
import argparse

BASE_URL = "http://localhost:8000/v1"


def run_decode_test(model: str, num_tokens: int = 64, prompt: str = "Count from 1 to 100"):
    """Run a decode-heavy test (short prompt, many output tokens)."""
    start = time.time()
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": num_tokens,
            "stream": False
        },
        timeout=120
    )
    
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        output_tokens = data.get("usage", {}).get("completion_tokens", num_tokens)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        tok_per_sec = output_tokens / elapsed
        return {
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tok_per_sec": tok_per_sec,
            "success": True
        }
    else:
        return {
            "error": f"{response.status_code} - {response.text[:200]}",
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(description="Decode benchmark for vLLM")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate per run")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup tokens")
    args = parser.parse_args()

    print("Decode-focused benchmark")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Tokens per run: {args.tokens}")
    print(f"Runs: {args.runs}")
    print()

    # Warmup
    print("Warmup...")
    result = run_decode_test(args.model, args.warmup)
    if result["success"]:
        print(f"  Generated {result['output_tokens']} tokens in {result['elapsed']:.2f}s = {result['tok_per_sec']:.1f} tok/s")
    else:
        print(f"  Error: {result['error']}")
        return

    # Benchmark runs
    print(f"\nRunning {args.runs} decode tests...")
    results = []
    for i in range(args.runs):
        result = run_decode_test(args.model, args.tokens)
        if result["success"]:
            results.append(result["tok_per_sec"])
            print(f"Run {i+1}: Generated {result['output_tokens']} tokens in {result['elapsed']:.2f}s = {result['tok_per_sec']:.1f} tok/s")
        else:
            print(f"Run {i+1}: Error - {result['error']}")

    if results:
        avg = sum(results) / len(results)
        print(f"\nAverage: {avg:.1f} tok/s")
        print(f"Min: {min(results):.1f} tok/s")
        print(f"Max: {max(results):.1f} tok/s")


if __name__ == "__main__":
    main()

