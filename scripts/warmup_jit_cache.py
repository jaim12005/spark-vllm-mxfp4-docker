#!/usr/bin/env python3
"""
Pre-compile FlashInfer MoE CUTLASS kernels for SM121.

Run this once after installing FlashInfer to avoid the ~3-5 minute
JIT compilation delay on first vLLM server start.

Usage:
    docker exec vllm-dev python3 /workspace/scripts/warmup_jit_cache.py
"""

import os
import sys
import time

# Ensure we're using the mounted FlashInfer
os.environ.setdefault("PYTHONPATH", "/workspace/flashinfer:/workspace/vllm")
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

# Set architecture for JIT
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.1a"

import torch

def warmup_moe_kernels():
    """Trigger JIT compilation of MoE CUTLASS kernels."""
    print("=" * 60)
    print("FlashInfer MoE Kernel Warmup for SM121")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: SM{cap[0]}{cap[1]}")
    print()
    
    if cap[0] != 12:
        print(f"WARNING: This script is for SM12x, but found SM{cap[0]}{cap[1]}")
    
    # Import FlashInfer MoE (triggers JIT if not cached)
    print("Importing FlashInfer fused_moe module...")
    start = time.time()
    
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer import mxfp8_quantize
        print(f"  Import took {time.time() - start:.1f}s")
    except Exception as e:
        print(f"ERROR: Failed to import: {e}")
        return False
    
    # Create small test tensors to trigger kernel compilation
    print()
    print("Compiling MoE CUTLASS kernels (this may take 3-5 minutes)...")
    
    # Typical gpt-oss dimensions
    num_experts = 8
    hidden_dim = 2944  # Must be divisible by 128 for MXFP4
    intermediate_dim = 5888
    
    # Test multiple M sizes to compile different tile configs
    test_m_sizes = [1, 16, 64, 256, 1024, 2048]
    
    for m in test_m_sizes:
        print(f"  M={m:4d}...", end=" ", flush=True)
        start = time.time()
        
        try:
            # Create BF16 input
            x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
            
            # Create random routing
            topk = 2
            topk_indices = torch.randint(0, num_experts, (m, topk), device=device)
            topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).float()
            
            # Create fake FP4 weights (packed uint8)
            fc1_weights = torch.randint(0, 256, (num_experts, intermediate_dim * 2, hidden_dim // 2), 
                                        dtype=torch.uint8, device=device)
            fc2_weights = torch.randint(0, 256, (num_experts, hidden_dim, intermediate_dim // 2),
                                        dtype=torch.uint8, device=device)
            
            # Create scales (UE8M0 format)
            fc1_scales = torch.randint(0, 256, (num_experts, intermediate_dim * 2, hidden_dim // 32),
                                       dtype=torch.uint8, device=device)
            fc2_scales = torch.randint(0, 256, (num_experts, hidden_dim, intermediate_dim // 32),
                                       dtype=torch.uint8, device=device)
            
            # Quantize activations
            x_fp8, x_scale = mxfp8_quantize(x)
            
            # Call the kernel (this triggers JIT if not cached)
            from flashinfer.fused_moe.core import ActivationType
            output = cutlass_fused_moe(
                input=x_fp8,
                token_selected_experts=topk_indices,
                token_final_scales=topk_weights,
                fc1_expert_weights=fc1_weights,
                fc2_expert_weights=fc2_weights,
                output_dtype=torch.bfloat16,
                quant_scales=[x_scale, fc1_scales, fc2_scales],
                activation_type=ActivationType.Swiglu,
            )
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    print()
    print("=" * 60)
    print("Warmup complete! JIT cache is now populated.")
    print("Subsequent vLLM starts will be much faster.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = warmup_moe_kernels()
    sys.exit(0 if success else 1)


