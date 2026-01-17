#!/usr/bin/env python3
"""
Smoke test using properly quantized weights via mxfp4_quantize.
"""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

# Small dimensions for quick test
hidden_size = 256  # Must be divisible by 128
intermediate_size = 512
num_experts = 1  # Single expert for simplicity
num_tokens = 4
top_k = 1

print(f"=== Properly quantized weights smoke test ===")
print(f"hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
print(f"num_experts: {num_experts}, num_tokens: {num_tokens}, top_k: {top_k}")

# Create BF16 input
x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
print(f"\n1. Input BF16: {x_bf16[0, :4]}")

# Quantize input to FP8 (as vLLM does)
x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
print(f"2. x_quant dtype: {x_quant.dtype}, x_scale dtype: {x_scale.dtype}")

# Create small BF16 weights and quantize them properly
# FC1: [num_experts, 2*intermediate, hidden] -> [num_experts, 2*intermediate, hidden//2] packed
# FC2: [num_experts, hidden, intermediate] -> [num_experts, hidden, intermediate//2] packed

# Create small BF16 weights for testing
w13_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                       dtype=torch.bfloat16, device=device) * 0.1  # Small values
w2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                      dtype=torch.bfloat16, device=device) * 0.1

print(f"3. BF16 weights created:")
print(f"   w13_bf16: shape={w13_bf16.shape}")
print(f"   w2_bf16: shape={w2_bf16.shape}")

# Quantize weights using mxfp4_quantize
# The function returns (weight_fp4, weight_scale)
print(f"\n4. Quantizing weights with mxfp4_quantize...")

# Reshape for quantization (quantize along last dim)
w13_flat = w13_bf16.reshape(-1, hidden_size)
w2_flat = w2_bf16.reshape(-1, intermediate_size)

w13_fp4, w13_scale = mxfp4_quantize(w13_flat)
w2_fp4, w2_scale = mxfp4_quantize(w2_flat)

print(f"   w13_fp4: shape={w13_fp4.shape}, dtype={w13_fp4.dtype}")
print(f"   w13_scale: shape={w13_scale.shape}, dtype={w13_scale.dtype}")
print(f"   w2_fp4: shape={w2_fp4.shape}, dtype={w2_fp4.dtype}")
print(f"   w2_scale: shape={w2_scale.shape}, dtype={w2_scale.dtype}")

# Reshape back to expert format
# FP4 is packed: 2 values per byte, so last dim is hidden//2
w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)

# Scale has group_size=32, so last dim is hidden//32
w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)

print(f"\n5. Reshaped to expert format:")
print(f"   w13_fp4: shape={w13_fp4.shape}")
print(f"   w13_scale: shape={w13_scale.shape}")
print(f"   w2_fp4: shape={w2_fp4.shape}")
print(f"   w2_scale: shape={w2_scale.shape}")

# View as long (int64) as vLLM does
fc1_weights = w13_fp4.contiguous().view(torch.long)
fc2_weights = w2_fp4.contiguous().view(torch.long)

# View scales as int32 as vLLM does  
fc1_scale = w13_scale.contiguous().view(torch.int32)
fc2_scale = w2_scale.contiguous().view(torch.int32)

print(f"\n6. Viewed as vLLM types:")
print(f"   fc1_weights: dtype={fc1_weights.dtype}")
print(f"   fc1_scale: dtype={fc1_scale.dtype}")

# Fake input scale
fake_input_scale = torch.ones(num_experts, device=device)
quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]

# Routing - all tokens to expert 0
topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

# Output tensor
output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

# Call kernel
print(f"\n7. Calling cutlass_fused_moe...")
try:
    result = cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids.to(torch.int).contiguous(),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=torch.bfloat16,
        output=output,
        quant_scales=quant_scales,
        input_sf=x_scale,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
    )
    
    print(f"\n=== SUCCESS ===")
    print(f"Output shape: {result[0].shape}")
    print(f"Output sample (first 8): {result[0][0, :8]}")
    print(f"Has NaN: {torch.isnan(result[0]).any()}")
    print(f"Has Inf: {torch.isinf(result[0]).any()}")
    print(f"Output min: {result[0].min().item():.4f}, max: {result[0].max().item():.4f}")
    
    # Check if values are in reasonable range
    if result[0].abs().max() < 100:
        print("Values in reasonable range (< 100) - LOOKS GOOD")
    else:
        print("Values EXTREME - potential issue")
    
except Exception as e:
    print(f"\n=== FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
