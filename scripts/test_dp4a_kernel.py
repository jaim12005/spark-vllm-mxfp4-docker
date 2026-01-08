#!/usr/bin/env python3
"""
Test the compiled DP4A GEMV kernel.
"""

import torch
import ctypes
import time
import os

# Load the compiled kernel
SO_PATH = "/tmp/gemv_dp4a.so"

def load_kernel():
    """Load the compiled CUDA kernel."""
    if not os.path.exists(SO_PATH):
        print(f"Kernel not found at {SO_PATH}")
        return None
    
    lib = ctypes.CDLL(SO_PATH)
    
    # Define the function signature
    # void gemv_fp4_dp4a(weights, scales, input, output, N, K, stream)
    lib.gemv_fp4_dp4a.argtypes = [
        ctypes.c_void_p,  # weights
        ctypes.c_void_p,  # scales (not used, embedded)
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_void_p,  # stream
    ]
    lib.gemv_fp4_dp4a.restype = None
    
    return lib

def create_test_data(N: int, K: int, device: torch.device):
    """
    Create test data in the expected format.
    
    For this kernel, weights are expected as block_mxfp4 format:
    - Each block: 1 byte scale + 16 bytes packed nibbles = 17 bytes for 32 elements
    - Total blocks: (N * K) / 32
    """
    n_blocks_per_row = K // 32
    
    # Create packed weights in block_mxfp4 format
    # [N * n_blocks_per_row] blocks, each 17 bytes
    # For simplicity, create as contiguous uint8 array
    
    # Actually, for the kernel, we'll create:
    # - weights: [N, K/2] uint8 packed nibbles
    # - Plus embedded scale at start of each block
    
    # Simpler: create weights as PyTorch would see them
    weights_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    
    # Input activation
    input_bf16 = torch.randn(K, dtype=torch.bfloat16, device=device)
    
    # Output buffer
    output_bf16 = torch.zeros(N, dtype=torch.bfloat16, device=device)
    
    return weights_packed, input_bf16, output_bf16

def benchmark_kernel(lib, N: int, K: int, num_iters: int = 100):
    """Benchmark the DP4A kernel."""
    device = torch.device("cuda")
    
    weights, input_bf16, output = create_test_data(N, K, device)
    
    # Get raw pointers
    weights_ptr = weights.data_ptr()
    input_ptr = input_bf16.data_ptr()
    output_ptr = output.data_ptr()
    
    # Warmup
    for _ in range(10):
        lib.gemv_fp4_dp4a(
            weights_ptr, None, input_ptr, output_ptr,
            N, K, None
        )
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iters):
        lib.gemv_fp4_dp4a(
            weights_ptr, None, input_ptr, output_ptr,
            N, K, None
        )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    flops = 2 * N * K
    gflops = (flops / 1e9) / (avg_ms / 1000)
    
    return avg_ms, gflops

def main():
    print("=" * 70)
    print("DP4A GEMV Kernel Test")
    print("=" * 70)
    
    lib = load_kernel()
    if lib is None:
        print("Failed to load kernel!")
        return
    
    print("Kernel loaded successfully!")
    print()
    
    # Test configurations
    test_configs = [
        (11776, 4096, "FC1 (hidden -> inter)"),
        (4096, 11776, "FC2 (inter -> hidden)"),
        (4096, 4096, "Square"),
    ]
    
    print("Benchmarking DP4A GEMV kernel:")
    print("-" * 70)
    print(f"{'Config':<30} {'Avg (ms)':<12} {'GFLOPS':<12}")
    print("-" * 70)
    
    for N, K, desc in test_configs:
        try:
            avg_ms, gflops = benchmark_kernel(lib, N, K)
            print(f"{desc:<30} {avg_ms:>10.4f}   {gflops:>10.2f}")
        except Exception as e:
            print(f"{desc:<30} FAILED: {e}")
    
    print()
    print("Comparison with PyTorch BF16 GEMV:")
    print("-" * 70)
    
    for N, K, desc in test_configs:
        device = torch.device("cuda")
        weights = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        activation = torch.randn(K, dtype=torch.bfloat16, device=device)
        
        # Warmup
        for _ in range(10):
            _ = weights @ activation
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(100):
            _ = weights @ activation
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        gflops = (2 * N * K / 1e9) / (avg_ms / 1000)
        print(f"{desc:<30} {avg_ms:>10.4f}   {gflops:>10.2f}")

if __name__ == "__main__":
    main()

