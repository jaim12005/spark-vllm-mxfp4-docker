#!/usr/bin/env python3
"""
Benchmark DP4A-based GEMV kernel for MXFP4 MoE decode optimization.

This script tests the llama.cpp-inspired DP4A kernel approach where:
1. FP4 weights are converted to INT8 via lookup table
2. BF16 activations are quantized to INT8 (Q8_1 format)
3. DP4A instruction is used for efficient dot products
4. Scale factors are applied at the end

Expected performance improvement over CUTLASS grouped GEMM for M=1:
- CUTLASS: 128x128 tile = 0.78% compute efficiency
- DP4A: No tile waste = ~100% compute efficiency
"""

import torch
import time
import math
import subprocess
import os

def check_dp4a_support():
    """Check if DP4A is available (SM 6.1+)."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 6 or (capability[0] == 6 and capability[1] >= 1)

def create_mxfp4_weights(N: int, K: int, device: torch.device):
    """
    Create MXFP4-format weights for testing.
    
    MXFP4 format (llama.cpp block_mxfp4):
    - 32 elements per block
    - 1 byte E8M0 scale + 16 bytes packed nibbles = 17 bytes per block
    
    For FlashInfer format:
    - weights: [N, K/2] uint8 packed nibbles
    - scales: [N, K/32] uint8 E8M0
    """
    n_blocks = K // 32
    
    # Create random FP4 values (0-15)
    weights_int4 = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=device)
    
    # Pack into nibbles (2 values per byte)
    weights_packed = (weights_int4[:, 0::2] & 0x0F) | ((weights_int4[:, 1::2] & 0x0F) << 4)
    weights_packed = weights_packed.contiguous()
    
    # Create random E8M0 scales (exponents from 120 to 135 for reasonable range)
    scales = torch.randint(120, 136, (N, n_blocks), dtype=torch.uint8, device=device)
    
    return weights_packed, scales

def create_bf16_activation(K: int, device: torch.device):
    """Create random BF16 activation vector."""
    return torch.randn(K, dtype=torch.bfloat16, device=device)

def reference_mxfp4_gemv(weights_packed, scales, activation):
    """
    Reference implementation of MXFP4 GEMV.
    
    This is a slow Python implementation for correctness verification.
    """
    N, K_half = weights_packed.shape
    K = K_half * 2
    n_blocks = K // 32
    
    # E2M1 dequantization table (doubled values)
    kvalues = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], 
                           dtype=torch.float32, device=weights_packed.device)
    
    # Unpack weights
    weights_low = (weights_packed & 0x0F).to(torch.int64)
    weights_high = (weights_packed >> 4).to(torch.int64)
    
    # Interleave to get original order
    weights_unpacked = torch.stack([weights_low, weights_high], dim=-1).reshape(N, K)
    
    # Dequantize using lookup table
    weights_dequant = kvalues[weights_unpacked] * 0.5  # Halve because table is doubled
    
    # Apply scales (E8M0 -> float: 2^(e - 127))
    scales_float = torch.pow(2.0, scales.float() - 127).unsqueeze(-1)
    weights_dequant = weights_dequant.reshape(N, n_blocks, 32) * scales_float
    weights_dequant = weights_dequant.reshape(N, K)
    
    # Matrix-vector multiply
    output = weights_dequant @ activation.float()
    
    return output.to(torch.bfloat16)

def benchmark_torch_bf16_gemv(N: int, K: int, num_iters: int = 100):
    """Benchmark PyTorch's BF16 matrix-vector multiply."""
    device = torch.device("cuda")
    
    weights = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    activation = torch.randn(K, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = weights @ activation
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        _ = weights @ activation
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * N * K
    gflops = (flops / 1e9) / (avg_ms / 1000)
    
    return avg_ms, gflops

def benchmark_reference_mxfp4(N: int, K: int, num_iters: int = 10):
    """Benchmark reference Python MXFP4 implementation."""
    device = torch.device("cuda")
    
    weights_packed, scales = create_mxfp4_weights(N, K, device)
    activation = create_bf16_activation(K, device)
    
    # Warmup
    for _ in range(3):
        _ = reference_mxfp4_gemv(weights_packed, scales, activation)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        _ = reference_mxfp4_gemv(weights_packed, scales, activation)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * N * K  
    gflops = (flops / 1e9) / (avg_ms / 1000)
    
    return avg_ms, gflops

def compile_dp4a_kernel():
    """Compile the DP4A GEMV kernel if not already compiled."""
    kernel_path = "/workspace/flashinfer/csrc/gemv/gemv_fp4_dp4a_llama.cu"
    so_path = "/tmp/gemv_dp4a.so"
    
    if not os.path.exists(kernel_path):
        print(f"Kernel source not found at {kernel_path}")
        return None
    
    if os.path.exists(so_path):
        # Check if source is newer
        if os.path.getmtime(kernel_path) <= os.path.getmtime(so_path):
            return so_path
    
    print("Compiling DP4A GEMV kernel...")
    cmd = [
        "nvcc",
        "-O3",
        "--shared",
        "-Xcompiler", "-fPIC",
        "-arch=sm_121",  # SM121 for GB10
        "-o", so_path,
        kernel_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return None
        print(f"Compiled to {so_path}")
        return so_path
    except Exception as e:
        print(f"Compilation error: {e}")
        return None

def main():
    print("=" * 70)
    print("DP4A-based MXFP4 GEMV Benchmark")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability()
    print(f"Device: {device_name}")
    print(f"Compute Capability: SM{capability[0]}{capability[1]}")
    print(f"DP4A Support: {'Yes' if check_dp4a_support() else 'No'}")
    print()
    
    # gpt-oss-120b dimensions
    # Hidden: 4096, Intermediate: 11776
    test_configs = [
        # (N, K, description)
        (11776, 4096, "FC1 gate/up (hidden -> inter)"),
        (4096, 11776, "FC2 down (inter -> hidden)"),
        (4096, 4096, "Square matrix"),
        (8192, 4096, "2x hidden -> inter"),
    ]
    
    print("Benchmark: PyTorch BF16 GEMV (baseline)")
    print("-" * 70)
    print(f"{'Config':<35} {'Avg (ms)':<12} {'GFLOPS':<12}")
    print("-" * 70)
    
    for N, K, desc in test_configs:
        avg_ms, gflops = benchmark_torch_bf16_gemv(N, K)
        print(f"{desc:<35} {avg_ms:>10.4f}   {gflops:>10.2f}")
    
    print()
    print("Benchmark: Reference Python MXFP4 GEMV (correctness check)")
    print("-" * 70)
    print(f"{'Config':<35} {'Avg (ms)':<12} {'GFLOPS':<12}")
    print("-" * 70)
    
    for N, K, desc in test_configs[:2]:  # Only first two (slow)
        avg_ms, gflops = benchmark_reference_mxfp4(N, K, num_iters=5)
        print(f"{desc:<35} {avg_ms:>10.4f}   {gflops:>10.2f}")
    
    print()
    print("=" * 70)
    print("Analysis: Expected Performance")
    print("=" * 70)
    
    # Calculate expected performance
    K = 4096
    N = 11776
    
    print(f"\nFor MXFP4 GEMV with K={K}, N={N}:")
    print()
    
    # DP4A theoretical: each DP4A does 8 MACs (4 per instruction x 2 nibbles)
    # SM121 has ~192 CUDA cores per SM
    # At ~1.5 GHz, each core can do many DP4A ops/cycle
    
    # Memory bound calculation
    weight_bytes = N * K / 2  # FP4 packed
    scale_bytes = N * K / 32  # E8M0 per 32 elements
    activation_bytes = K * 2  # BF16
    output_bytes = N * 2      # BF16
    
    total_bytes = weight_bytes + scale_bytes + activation_bytes + output_bytes
    
    # GB10 bandwidth: ~800 GB/s
    bandwidth_gbs = 800
    mem_bound_ms = (total_bytes / 1e9) / bandwidth_gbs * 1000
    
    print(f"Memory traffic: {total_bytes/1e6:.2f} MB")
    print(f"Memory-bound lower limit: {mem_bound_ms:.4f} ms")
    print()
    
    # llama.cpp achieves ~58 tok/s with full model
    # Our CUTLASS achieves ~29 tok/s
    # Target: 2x speedup on MoE decode
    
    print("Performance targets:")
    print(f"  Current CUTLASS MoE: ~22 ms per token (29 tok/s)")
    print(f"  llama.cpp reference: ~10 ms per token (58 tok/s)")
    print(f"  Required speedup: 2x on MoE kernel")
    print()
    
    # Check if kernel was compiled
    so_path = compile_dp4a_kernel()
    if so_path:
        print(f"\nKernel compiled at: {so_path}")
        print("TODO: Load and benchmark the actual DP4A kernel")
    else:
        print("\nKernel compilation not available in this environment.")
        print("Run inside the vllm-dev container to compile and test.")

if __name__ == "__main__":
    main()


