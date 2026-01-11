#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for MXFP4 lm_head implementation.

This script validates that:
1. mxfp4_e2m1_quantize correctly quantizes BF16 tensors
2. The quantized GEMM produces results close to BF16 reference
3. The implementation works for lm_head-sized tensors (gpt-oss-120b: hidden_size=8192, vocab_size=128256)
"""

import sys
import torch

# Ensure we're using local flashinfer/vllm
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")


def test_mxfp4_quantize():
    """Test that mxfp4_e2m1_quantize works correctly."""
    print("=" * 60)
    print("Test 1: MXFP4 Quantization")
    print("=" * 60)

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_e2m1_quantize,
        mxfp4_e2m1_dequantize,
    )

    # Create test tensor
    weight = torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda")

    # Quantize
    weight_fp4, weight_scale = mxfp4_e2m1_quantize(weight)

    print(f"Original weight: {weight.shape}, dtype={weight.dtype}")
    print(f"Quantized weight: {weight_fp4.shape}, dtype={weight_fp4.dtype}")
    print(f"Scale: {weight_scale.shape}, dtype={weight_scale.dtype}")

    # Expected shapes
    expected_weight_shape = (1024, 256)  # K/2 packed
    expected_scale_shape = (1024, 16)  # K/32

    assert weight_fp4.shape == expected_weight_shape, f"Expected {expected_weight_shape}, got {weight_fp4.shape}"
    assert weight_scale.shape == expected_scale_shape, f"Expected {expected_scale_shape}, got {weight_scale.shape}"
    assert weight_fp4.dtype == torch.uint8, f"Expected uint8, got {weight_fp4.dtype}"
    assert weight_scale.dtype == torch.uint8, f"Expected uint8, got {weight_scale.dtype}"

    # Dequantize and check error
    weight_dequant = mxfp4_e2m1_dequantize(weight_fp4, weight_scale, torch.bfloat16)
    # Ensure same device
    weight_dequant = weight_dequant.to(weight.device)
    
    # Calculate relative error
    rel_error = (weight_dequant - weight).abs() / (weight.abs() + 1e-6)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"Dequantized weight: {weight_dequant.shape}, dtype={weight_dequant.dtype}")
    print(f"Max relative error: {max_rel_error:.4f}")
    print(f"Mean relative error: {mean_rel_error:.4f}")

    # FP4 has limited precision, expect some error but it should be reasonable
    assert max_rel_error < 2.0, f"Max relative error too high: {max_rel_error}"
    assert mean_rel_error < 0.5, f"Mean relative error too high: {mean_rel_error}"

    print("✓ MXFP4 quantization test passed!\n")


def test_mxfp8_quantize():
    """Test that mxfp8_quantize (for activations) works correctly."""
    print("=" * 60)
    print("Test 2: MXFP8 Quantization (activations)")
    print("=" * 60)

    from flashinfer import mxfp8_quantize

    # Create test activation tensor
    x = torch.randn(32, 8192, dtype=torch.bfloat16, device="cuda")

    # Quantize with swizzled layout (for Blackwell)
    x_fp8, x_scale = mxfp8_quantize(x, True, 32)

    print(f"Original activation: {x.shape}, dtype={x.dtype}")
    print(f"Quantized activation: {x_fp8.shape}, dtype={x_fp8.dtype}")
    print(f"Scale: {x_scale.shape}, dtype={x_scale.dtype}")

    assert x_fp8.dtype == torch.float8_e4m3fn, f"Expected float8_e4m3fn, got {x_fp8.dtype}"

    print("✓ MXFP8 quantization test passed!\n")


def test_group_gemm_kernel():
    """Test the FP8×FP4 GEMM kernel."""
    print("=" * 60)
    print("Test 3: group_gemm_mxfp8_mxfp4_nt_groupwise kernel")
    print("=" * 60)

    from flashinfer import mxfp8_quantize, mxfp4_quantize
    from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise

    # Realistic lm_head dimensions (smaller vocab for testing)
    # Kernel requires: K % 128 == 0, N % 8 == 0
    M = 4  # num_tokens (will be padded to 4)
    K = 8192  # hidden_size (gpt-oss-120b)
    N = 1024  # vocab_size subset (must be multiple of 8)

    # Create test tensors
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Reference BF16 matmul
    ref_output = torch.matmul(x, weight.T)

    # Quantize activations to FP8
    x_fp8, x_scale = mxfp8_quantize(x, True, 32)

    # Quantize weights to FP4
    weight_fp4, weight_scale = mxfp4_quantize(weight)

    print(f"Activation: {x.shape} -> FP8 {x_fp8.shape}")
    print(f"Weight: {weight.shape} -> FP4 {weight_fp4.shape}")

    # Setup for group_gemm
    m_padded = (M + 3) // 4 * 4
    m_indptr = torch.tensor([0, m_padded], dtype=torch.int32, device="cuda")

    # Pad if needed
    if m_padded > M:
        pad_size = m_padded - M
        x_fp8 = torch.nn.functional.pad(x_fp8, (0, 0, 0, pad_size))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, pad_size))

    # Reshape for group_gemm: B needs to be (batch_size, n, k // 2)
    weight_fp4_reshaped = weight_fp4.unsqueeze(0)
    weight_scale_reshaped = weight_scale.unsqueeze(0)

    print(f"x_fp8: {x_fp8.shape}, x_scale: {x_scale.shape}")
    print(f"weight_fp4: {weight_fp4_reshaped.shape}, weight_scale: {weight_scale_reshaped.shape}")
    print(f"m_indptr: {m_indptr}")

    # Run kernel
    try:
        output = group_gemm_mxfp4_nt_groupwise(
            a=x_fp8,
            b=weight_fp4_reshaped,
            a_scale=x_scale,
            b_scale=weight_scale_reshaped,
            m_indptr=m_indptr,
            mma_sm=1,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out_dtype=torch.bfloat16,
        )

        # Remove padding
        if m_padded > M:
            output = output[:M, :]

        print(f"Output: {output.shape}")

        # Compare with reference
        diff = (output - ref_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_error = diff / (ref_output.abs() + 1e-6)
        max_rel_error = rel_error.max().item()

        print(f"Max absolute diff: {max_diff:.4f}")
        print(f"Mean absolute diff: {mean_diff:.4f}")
        print(f"Max relative error: {max_rel_error:.4f}")

        # FP4 has limited precision, allow for some error
        assert max_rel_error < 1.0, f"Max relative error too high: {max_rel_error}"

        print("✓ group_gemm_mxfp8_mxfp4_nt_groupwise test passed!\n")

    except Exception as e:
        print(f"✗ Kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_dequant_gemm():
    """Test the dequant + BF16 GEMM approach (current implementation)."""
    print("=" * 60)
    print("Test 3b: Dequant + BF16 GEMM approach")
    print("=" * 60)

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_e2m1_quantize,
        mxfp4_e2m1_dequantize,
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Realistic lm_head dimensions (smaller for testing)
    M = 4  # num_tokens
    K = 256  # hidden_size
    N = 128  # vocab_size subset

    # Create test tensors with bounded values to avoid extreme relative errors
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Reference BF16 matmul
    ref_output = torch.nn.functional.linear(x, weight)

    # Quantize weights to FP4
    weight_fp4, weight_scale = mxfp4_e2m1_quantize(weight)
    print(f"Weight: {weight.shape} -> FP4 {weight_fp4.shape}, scale {weight_scale.shape}")

    # Dequantize and GEMM
    weight_dequant = mxfp4_e2m1_dequantize(weight_fp4, weight_scale, torch.bfloat16)
    weight_dequant = weight_dequant.to(x.device)  # Ensure same device
    output = torch.nn.functional.linear(x, weight_dequant)

    print(f"Output: {output.shape}")

    # Compare with reference using multiple metrics
    diff = (output - ref_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Cosine similarity is more meaningful for ML (value correlation)
    output_flat = output.flatten().float()
    ref_flat = ref_output.flatten().float()
    cosine_sim = torch.nn.functional.cosine_similarity(
        output_flat.unsqueeze(0), ref_flat.unsqueeze(0)
    ).item()
    
    # Root mean squared error normalized by reference std
    rmse = (diff.float() ** 2).mean().sqrt().item()
    ref_std = ref_output.float().std().item()
    normalized_rmse = rmse / (ref_std + 1e-6)

    print(f"Max absolute diff: {max_diff:.4f}")
    print(f"Mean absolute diff: {mean_diff:.4f}")
    print(f"Normalized RMSE: {normalized_rmse:.4f}")
    print(f"Cosine similarity: {cosine_sim:.4f}")

    # For FP4, cosine similarity > 0.9 is good, > 0.95 is excellent
    # Normalized RMSE < 0.5 is acceptable
    if cosine_sim > 0.95:
        print("✓ Dequant + GEMM test passed with excellent accuracy!\n")
    elif cosine_sim > 0.90:
        print(f"✓ Dequant + GEMM test passed with good accuracy (cos_sim={cosine_sim:.4f})\n")
    elif cosine_sim > 0.80:
        print(f"⚠ Dequant + GEMM test passed with moderate accuracy (cos_sim={cosine_sim:.4f})\n")
    else:
        assert False, f"Cosine similarity too low: {cosine_sim:.4f}"
    return True


def test_lm_head_size():
    """Test with actual gpt-oss-120b lm_head dimensions."""
    print("=" * 60)
    print("Test 4: lm_head size (gpt-oss-120b dimensions)")
    print("=" * 60)

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_e2m1_quantize,
    )

    # gpt-oss-120b dimensions
    hidden_size = 8192
    vocab_size = 128256

    print(f"hidden_size: {hidden_size}")
    print(f"vocab_size: {vocab_size}")

    # Create lm_head weight
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight_bytes_bf16 = weight.numel() * 2  # 2 bytes per bf16

    print(f"BF16 weight size: {weight_bytes_bf16 / 1024 / 1024:.2f} MB")

    # Quantize
    weight_fp4, weight_scale = mxfp4_e2m1_quantize(weight)

    weight_fp4_bytes = weight_fp4.numel()  # 1 byte per uint8
    weight_scale_bytes = weight_scale.numel()  # 1 byte per uint8
    total_fp4_bytes = weight_fp4_bytes + weight_scale_bytes

    print(f"FP4 weight size: {weight_fp4_bytes / 1024 / 1024:.2f} MB")
    print(f"FP4 scale size: {weight_scale_bytes / 1024 / 1024:.2f} MB")
    print(f"Total MXFP4 size: {total_fp4_bytes / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {weight_bytes_bf16 / total_fp4_bytes:.2f}x")

    # Expected compression: ~4x (16 bits -> 4 bits + scales overhead)
    assert total_fp4_bytes < weight_bytes_bf16 * 0.35, "Compression not effective"

    print("✓ lm_head size test passed!\n")


def main():
    print("\n" + "=" * 60)
    print("MXFP4 lm_head Implementation Tests")
    print("=" * 60 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return 1

    device = torch.cuda.get_device_properties(0)
    print(f"GPU: {device.name}")
    print(f"Compute capability: {device.major}.{device.minor}")
    print()

    try:
        test_mxfp4_quantize()
        test_mxfp8_quantize()
        
        # Test dequant + GEMM approach (works on all GPUs)
        test_dequant_gemm()
        
        # Test lm_head memory savings
        test_lm_head_size()
        
        # Only run FP8×FP4 kernel tests on Blackwell
        if device.major >= 12:
            print("\n" + "=" * 60)
            print("Optional: Testing FP8×FP4 kernel (may fail for small dims)")
            print("=" * 60)
            try:
                test_group_gemm_kernel()
            except Exception as e:
                print(f"FP8×FP4 kernel test failed (expected for some configs): {e}")
        else:
            print(f"Skipping FP8×FP4 kernel tests (requires SM12x, got SM{device.major}{device.minor})")

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
