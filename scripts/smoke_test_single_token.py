#!/usr/bin/env python3
"""
Test CUTLASS kernel with single token (M=1) - the inference case.
"""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

# Match model dimensions
hidden_size = 2944
intermediate_size = 2944  # same as model
num_experts = 128
num_tokens = 1  # SINGLE TOKEN - the inference case
top_k = 4

print(f"=== Single token test (M={num_tokens}) ===")
print(f"hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
print(f"num_experts: {num_experts}, top_k: {top_k}")

# Create BF16 input (single token)
x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
print(f"\n1. Input BF16: shape={x_bf16.shape}, range=[{x_bf16.min():.4f}, {x_bf16.max():.4f}]")

# Quantize to FP8
x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
print(f"2. x_quant dtype: {x_quant.dtype}, x_scale shape: {x_scale.shape}")

# Create simple weights with mxfp4_quantize
# FC1: [num_experts, 2*inter, hidden] for gate+up
# FC2: [num_experts, hidden, inter] for down
print("\n3. Creating and quantizing weights...")

w13_bf16 = torch.randn(num_experts * 2 * intermediate_size, hidden_size, 
                       dtype=torch.bfloat16, device=device) * 0.02
w2_bf16 = torch.randn(num_experts * hidden_size, intermediate_size,
                      dtype=torch.bfloat16, device=device) * 0.02

w13_fp4, w13_scale = mxfp4_quantize(w13_bf16)
w2_fp4, w2_scale = mxfp4_quantize(w2_bf16)

# Reshape to expert format
w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)
w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)

# View as vLLM does
fc1_weights = w13_fp4.contiguous().view(torch.long)
fc2_weights = w2_fp4.contiguous().view(torch.long)
fc1_scale = w13_scale.contiguous().view(torch.int32)
fc2_scale = w2_scale.contiguous().view(torch.int32)

print(f"   fc1_weights: {fc1_weights.shape}")
print(f"   fc2_weights: {fc2_weights.shape}")

# Fake input scale
fake_input_scale = torch.ones(num_experts, device=device)
quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]

# Routing - pick specific experts
topk_ids = torch.tensor([[11, 23, 54, 58]], dtype=torch.int32, device=device)
topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

# Output tensor
output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

print(f"\n4. Calling cutlass_fused_moe with M={num_tokens}...")
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
    
    print(f"\n=== RESULT ===")
    print(f"Output shape: {result[0].shape}")
    print(f"Output range: [{result[0].min().item():.4f}, {result[0].max().item():.4f}]")
    print(f"Output mean: {result[0].float().mean().item():.4f}")
    print(f"Has NaN: {torch.isnan(result[0]).any()}")
    print(f"Has Inf: {torch.isinf(result[0]).any()}")
    
    max_abs = result[0].abs().max().item()
    if max_abs < 100:
        print(f"✓ Values REASONABLE (max_abs={max_abs:.2f})")
    else:
        print(f"✗ Values EXTREME (max_abs={max_abs:.2f}) - BUG!")
        
except Exception as e:
    print(f"\n=== FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
