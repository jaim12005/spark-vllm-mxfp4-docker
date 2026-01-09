#!/usr/bin/env python3
"""
Profile vLLM decode workload for kernel-level analysis.

Usage:
    # With server already running:
    python scripts/profile_decode.py --endpoint http://localhost:8000
    
    # Or run standalone (will load model):
    python scripts/profile_decode.py --standalone
"""

import argparse
import json
import time
import requests
from typing import Optional


def make_request(endpoint: str, prompt: str, max_tokens: int = 32) -> dict:
    """Send a completion request to vLLM server."""
    url = f"{endpoint}/v1/completions"
    payload = {
        "model": "gpt-oss-120b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    response = requests.post(url, json=payload, timeout=300)
    return response.json()


def warmup(endpoint: str, num_warmup: int = 3):
    """Warmup the model with a few requests."""
    print(f"Warming up with {num_warmup} requests...")
    prompt = "The quick brown fox jumps over the lazy dog. " * 50  # ~500 tokens
    for i in range(num_warmup):
        result = make_request(endpoint, prompt, max_tokens=16)
        print(f"  Warmup {i+1}/{num_warmup} done")
    print("Warmup complete.\n")


def profile_decode(endpoint: str, prompt_tokens: int = 2048, output_tokens: int = 32):
    """
    Profile decode phase.
    
    Uses a long prompt to get through prefill, then measures decode.
    """
    # Create prompt of approximate length
    base_prompt = "Hello world. " * 10  # ~30 tokens per repetition
    repetitions = prompt_tokens // 30
    prompt = base_prompt * repetitions
    
    print(f"Profiling decode: ~{prompt_tokens} prompt tokens, {output_tokens} output tokens")
    
    start = time.perf_counter()
    result = make_request(endpoint, prompt, max_tokens=output_tokens)
    elapsed = time.perf_counter() - start
    
    if "choices" in result:
        output_text = result["choices"][0]["text"]
        # Rough token count (actual would need tokenizer)
        approx_output_tokens = len(output_text.split())
        print(f"  Generated ~{approx_output_tokens} tokens in {elapsed:.2f}s")
        print(f"  Approx decode rate: {approx_output_tokens/elapsed:.1f} tok/s")
    else:
        print(f"  Error: {result}")
    
    return result


def standalone_profile():
    """Profile without a running server (loads model inline)."""
    from vllm import LLM, SamplingParams
    import torch
    
    print("Loading model for standalone profiling...")
    llm = LLM(
        model="openai/gpt-oss-120b",
        quantization="mxfp4",
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.70,
        max_model_len=8192,  # Shorter for profiling
    )
    
    # Warmup
    print("Warming up...")
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    _ = llm.generate(["Hello world"] * 2, sampling_params)
    
    # Profile decode
    print("\nProfiled generation:")
    prompt = "The quick brown fox " * 500  # ~2000 tokens
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    output_text = outputs[0].outputs[0].text
    print(f"  Output: {output_text[:100]}...")
    print(f"  Elapsed: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Profile vLLM decode")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="vLLM server endpoint")
    parser.add_argument("--standalone", action="store_true", help="Run standalone (load model)")
    parser.add_argument("--prompt-tokens", type=int, default=2048, help="Approximate prompt length")
    parser.add_argument("--output-tokens", type=int, default=32, help="Number of output tokens")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup requests")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup")
    args = parser.parse_args()
    
    if args.standalone:
        standalone_profile()
    else:
        if not args.skip_warmup:
            warmup(args.endpoint, args.warmup)
        profile_decode(args.endpoint, args.prompt_tokens, args.output_tokens)


if __name__ == "__main__":
    main()
