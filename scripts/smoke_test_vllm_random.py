#!/usr/bin/env python3
"""
Smoke test with RANDOM weights to test numerical correctness.
"""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

# Smaller dimensions for quick test
hidden_size = 256  # Must be divisible by 128
intermediate_size = 512
num_experts = 2
num_tokens = 4
top_k = 1

print(f"=== Random weights smoke test ===")
print(f"hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
print(f"num_experts: {num_experts}, num_tokens: {num_tokens}, top_k: {top_k}")

# Create BF16 input
x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
print(f"\n1. Input BF16: {x_bf16[0, :4]}")

# Quantize to FP8
x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
print(f"2. x_quant dtype: {x_quant.dtype}, x_scale dtype: {x_scale.dtype}")

# Create RANDOM weights (non-zero)
w13_weight = torch.randint(0, 256, (num_experts, 2 * intermediate_size, hidden_size // 2),
                           dtype=torch.uint8, device=device)
w2_weight = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2),
                          dtype=torch.uint8, device=device)

fc1_weights = w13_weight.contiguous().view(torch.long)
fc2_weights = w2_weight.contiguous().view(torch.long)
print(f"3. Random weights created")

# Create weight scales (random but reasonable)
w13_weight_scale = torch.randint(100, 150, (num_experts, 2 * intermediate_size, hidden_size // 32),
                                 dtype=torch.uint8, device=device)
w2_weight_scale = torch.randint(100, 150, (num_experts, hidden_size, intermediate_size // 32),
                                dtype=torch.uint8, device=device)

fc1_scale = w13_weight_scale.contiguous().view(torch.int32)
fc2_scale = w2_weight_scale.contiguous().view(torch.int32)
print(f"4. Random scales created")

# Fake input scale
fake_input_scale = torch.ones(num_experts, device=device)

quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]

# Routing - all tokens to expert 0
topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

# Output tensor
output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

# Call kernel
print(f"\n5. Calling cutlass_fused_moe...")
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
    print(f"Output sample (last 8): {result[0][0, -8:]}")
    print(f"Has NaN: {torch.isnan(result[0]).any()}")
    print(f"Has Inf: {torch.isinf(result[0]).any()}")
    print(f"All zeros: {(result[0] == 0).all()}")
    print(f"Output min: {result[0].min()}, max: {result[0].max()}")
    
except Exception as e:
    print(f"\n=== FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
