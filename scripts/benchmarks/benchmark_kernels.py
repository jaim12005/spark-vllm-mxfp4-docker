#!/usr/bin/env python3
"""
Benchmark different MXFP4 MoE kernels for gpt-oss-120b.

Usage:
    # From host (runs inside container)
    ./scripts/benchmark_kernels.py --kernel marlin
    ./scripts/benchmark_kernels.py --kernel gemm
    ./scripts/benchmark_kernels.py --kernel gemv
    ./scripts/benchmark_kernels.py --all

    # Or set env var directly:
    VLLM_MXFP4_MOE_KERNEL=marlin vllm serve ...

Available kernels:
    - marlin: Original Marlin backend
    - gemm:   CUTLASS grouped GEMM (SM12x native)
    - gemv:   DP4A GEMV (experimental, falls back to GEMM)
    - triton: Triton backend
"""

import argparse
import subprocess
import sys
import time
import os


KERNELS = ["marlin", "gemm", "gemv", "triton"]


def run_benchmark(kernel: str, max_tokens: int = 128, num_runs: int = 5) -> dict:
    """Run benchmark with specified kernel."""
    
    # Set environment variable for kernel selection
    env = os.environ.copy()
    env["VLLM_MXFP4_MOE_KERNEL"] = kernel
    
    # Python code to run inside container
    benchmark_code = f'''
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Warmup
for _ in range(2):
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{{"role": "user", "content": "Hello"}}],
        max_tokens=32,
        temperature=0
    )

# Benchmark
prompt = "Write a detailed essay about the history of artificial intelligence."
times = []
tokens = []
for i in range({num_runs}):
    start = time.perf_counter()
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{{"role": "user", "content": prompt}}],
        max_tokens={max_tokens},
        temperature=0
    )
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    tokens.append(r.usage.completion_tokens)

avg_time = sum(times)/len(times)
avg_tokens = sum(tokens)/len(tokens)
tps = avg_tokens / avg_time
print(f"RESULT:{{tps:.2f}}")
'''
    
    cmd = [
        "docker", "exec",
        "-e", f"PYTHONPATH=/workspace/flashinfer:/workspace/vllm",
        "vllm-dev",
        "python3", "-c", benchmark_code
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Parse result
        for line in output.split("\n"):
            if line.startswith("RESULT:"):
                tps = float(line.split(":")[1])
                return {"kernel": kernel, "tok_s": tps, "success": True}
        
        return {"kernel": kernel, "error": output[-500:], "success": False}
    except subprocess.TimeoutExpired:
        return {"kernel": kernel, "error": "Timeout", "success": False}
    except Exception as e:
        return {"kernel": kernel, "error": str(e), "success": False}


def start_server(kernel: str) -> subprocess.Popen:
    """Start vLLM server with specified kernel."""
    
    cmd = [
        "docker", "exec", "-d",
        "-e", f"PYTHONPATH=/workspace/flashinfer:/workspace/vllm",
        "-e", f"VLLM_MXFP4_MOE_KERNEL={kernel}",
        "vllm-dev",
        "vllm", "serve", "openai/gpt-oss-120b",
        "--quantization", "mxfp4",
        "--enforce-eager",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--port", "8000"
    ]
    
    subprocess.run(cmd, check=True)
    return None


def wait_for_server(timeout: int = 600) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["docker", "exec", "vllm-dev", "curl", "-s", "http://localhost:8000/health"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout:
            return True
        time.sleep(5)
    return False


def stop_server():
    """Stop vLLM server."""
    subprocess.run(
        ["docker", "exec", "vllm-dev", "pkill", "-9", "-f", "vllm"],
        capture_output=True
    )
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MXFP4 MoE kernels")
    parser.add_argument("--kernel", choices=KERNELS, help="Kernel to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all kernels")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--skip-server", action="store_true", 
                        help="Skip starting server (assume already running)")
    args = parser.parse_args()
    
    if not args.kernel and not args.all:
        parser.print_help()
        sys.exit(1)
    
    kernels = KERNELS if args.all else [args.kernel]
    results = []
    
    print("=" * 60)
    print("MXFP4 MoE Kernel Benchmark")
    print("=" * 60)
    print(f"Model: openai/gpt-oss-120b")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs per kernel: {args.num_runs}")
    print()
    
    for kernel in kernels:
        print(f"[{kernel}] ", end="", flush=True)
        
        if not args.skip_server:
            # Stop any existing server
            stop_server()
            
            # Start server with this kernel
            print("Starting server... ", end="", flush=True)
            start_server(kernel)
            
            if not wait_for_server():
                print("FAILED (timeout)")
                results.append({"kernel": kernel, "error": "Server timeout", "success": False})
                continue
            
            print("Ready. ", end="", flush=True)
        
        # Run benchmark
        print("Benchmarking... ", end="", flush=True)
        result = run_benchmark(kernel, args.max_tokens, args.num_runs)
        results.append(result)
        
        if result["success"]:
            print(f"{result['tok_s']:.2f} tok/s")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')[:50]}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Kernel':<10} {'Throughput':>12} {'vs llama.cpp (58)':>20}")
    print("-" * 42)
    
    for r in results:
        if r["success"]:
            tps = r["tok_s"]
            gap = f"{58/tps:.2f}x slower" if tps < 58 else f"{tps/58:.2f}x faster"
            print(f"{r['kernel']:<10} {tps:>10.2f} tok/s {gap:>18}")
        else:
            print(f"{r['kernel']:<10} {'FAILED':>12}")


if __name__ == "__main__":
    main()


