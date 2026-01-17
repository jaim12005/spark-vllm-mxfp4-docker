#!/usr/bin/env python3
"""Basic smoke test for CUTLASS MoE kernel."""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType
from flashinfer import mxfp8_quantize

torch.manual_seed(42)
device = "cuda"

# Simple test: ones through identity-ish operation
num_tokens = 4
num_experts = 1
hidden = 256
inter = 512
top_k = 1

# Create activations of all 1.0
x_bf16 = torch.ones(num_tokens, hidden, dtype=torch.bfloat16, device=device)
print(f"Input (BF16): {x_bf16[0, :4]}")

# Quantize to FP8 (this is what vLLM does)
x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
print(f"x_quant shape: {x_quant.shape}, dtype: {x_quant.dtype}")
print(f"x_scale shape: {x_scale.shape}, dtype: {x_scale.dtype}")

# Zero weights = zero output (simple verification)
fc1_fp4 = torch.zeros(num_experts, 2 * inter, hidden // 2, dtype=torch.uint8, device=device)
fc1_scale = torch.ones(num_experts, 2 * inter, hidden // 32, dtype=torch.uint8, device=device) * 127

fc2_fp4 = torch.zeros(num_experts, hidden, inter // 2, dtype=torch.uint8, device=device)
fc2_scale = torch.ones(num_experts, hidden, inter // 32, dtype=torch.uint8, device=device) * 127

# Routing
token_experts = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
token_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device)

# Fake input scale (1.0)
fake_input_scale = torch.ones(1, dtype=torch.float32, device=device)

print("\nCalling cutlass_fused_moe...")
output = cutlass_fused_moe(
    input=x_quant,
    token_selected_experts=token_experts,
    token_final_scales=token_weights,
    fc1_expert_weights=fc1_fp4,
    fc2_expert_weights=fc2_fp4,
    output_dtype=torch.bfloat16,
    activation_type=ActivationType.Swiglu,
    quant_scales=[fc1_scale, fake_input_scale, fc2_scale, fake_input_scale],
    input_sf=x_scale,
)

print(f"\nZero weights test:")
print(f"Output shape: {output[0].shape}")
print(f"Output[0,:8]: {output[0][0, :8]}")
print(f"Expected: all zeros (zero weights)")
print(f"All zeros? {(output[0] == 0).all()}")
print("\nSUCCESS")
