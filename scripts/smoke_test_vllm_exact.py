#!/usr/bin/env python3
"""
Smoke test that EXACTLY mimics how vLLM calls FlashInfer for MXFP4.

Based on vllm/model_executor/layers/quantization/mxfp4.py lines 1706-1850
"""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

# Match model dimensions (gpt-oss-120b-like)
# Original hidden_size=2880 gets padded to 2944 (divisible by 128)
original_hidden_size = 2880
hidden_size = 2944  # Padded
intermediate_size = 8064  # From model config (also padded if needed)
num_experts = 8
num_tokens = 4
top_k = 2

print(f"=== vLLM-exact smoke test ===")
print(f"hidden_size: {hidden_size} (original: {original_hidden_size})")
print(f"intermediate_size: {intermediate_size}")
print(f"num_experts: {num_experts}, num_tokens: {num_tokens}, top_k: {top_k}")

# Step 1: Create BF16 input (what vLLM gets from previous layer)
x_bf16 = torch.randn(num_tokens, original_hidden_size, dtype=torch.bfloat16, device=device)
print(f"\n1. Input BF16: shape={x_bf16.shape}, dtype={x_bf16.dtype}")
print(f"   Sample values: {x_bf16[0, :4]}")

# Step 2: Pad input to match weight dimensions (as vLLM does)
pad_size = hidden_size - original_hidden_size
x_padded = torch.nn.functional.pad(x_bf16, (0, pad_size)) if pad_size > 0 else x_bf16
print(f"\n2. Padded input: shape={x_padded.shape}")

# Step 3: Quantize to FP8 (as vLLM does)
# mxfp8_quantize(x_padded, is_sf_swizzled_layout=True, sf_vec_size=32)
x_quant, x_scale = mxfp8_quantize(x_padded, True, 32)
print(f"\n3. Quantized input:")
print(f"   x_quant: shape={x_quant.shape}, dtype={x_quant.dtype}")
print(f"   x_scale: shape={x_scale.shape}, dtype={x_scale.dtype}")

# Step 4: Create weights - viewed as torch.long (as vLLM does)
# Weights are [num_experts, 2*intermediate_size, hidden_size//2] for FC1
# and [num_experts, hidden_size, intermediate_size//2] for FC2
# The //2 is because FP4 is packed (2 values per byte)
w13_weight = torch.zeros(num_experts, 2 * intermediate_size, hidden_size // 2, 
                         dtype=torch.uint8, device=device)
w2_weight = torch.zeros(num_experts, hidden_size, intermediate_size // 2,
                        dtype=torch.uint8, device=device)

# View as long (int64) as vLLM does
fc1_weights = w13_weight.contiguous().view(torch.long)
fc2_weights = w2_weight.contiguous().view(torch.long)
print(f"\n4. Weights (viewed as long):")
print(f"   fc1: shape={fc1_weights.shape}, dtype={fc1_weights.dtype}")
print(f"   fc2: shape={fc2_weights.shape}, dtype={fc2_weights.dtype}")

# Step 5: Create weight scales
# Scale factors are [num_experts, 2*intermediate_size, hidden_size//32] for FC1
# Group size is 32 for MXFP4
w13_weight_scale = torch.ones(num_experts, 2 * intermediate_size, hidden_size // 32,
                              dtype=torch.uint8, device=device) * 127  # ~1.0 scale
w2_weight_scale = torch.ones(num_experts, hidden_size, intermediate_size // 32,
                             dtype=torch.uint8, device=device) * 127

# View as int32 as vLLM does
fc1_scale = w13_weight_scale.contiguous().view(torch.int32)
fc2_scale = w2_weight_scale.contiguous().view(torch.int32)
print(f"\n5. Weight scales (viewed as int32):")
print(f"   fc1_scale: shape={fc1_scale.shape}, dtype={fc1_scale.dtype}")
print(f"   fc2_scale: shape={fc2_scale.shape}, dtype={fc2_scale.dtype}")

# Step 6: Create fake input scale (as vLLM does)
fake_input_scale = torch.ones(num_experts, device=device)
print(f"\n6. Fake input scale: shape={fake_input_scale.shape}")

# Step 7: Build quant_scales list (as vLLM does)
quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]
print(f"\n7. quant_scales: {len(quant_scales)} tensors")
for i, qs in enumerate(quant_scales):
    print(f"   [{i}]: shape={qs.shape}, dtype={qs.dtype}")

# Step 8: Create routing tensors
topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
print(f"\n8. Routing:")
print(f"   topk_ids: shape={topk_ids.shape}, dtype={topk_ids.dtype}")
print(f"   topk_weights: shape={topk_weights.shape}, dtype={topk_weights.dtype}")

# Step 9: Create output tensor (as vLLM does)
output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
print(f"\n9. Output buffer: shape={output.shape}, dtype={output.dtype}")

# Step 10: Call cutlass_fused_moe EXACTLY as vLLM does
print(f"\n10. Calling cutlass_fused_moe...")
try:
    result = cutlass_fused_moe(
        input=x_quant,  # FP8 quantized input
        token_selected_experts=topk_ids.to(torch.int).contiguous(),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,  # viewed as long
        fc2_expert_weights=fc2_weights,  # viewed as long
        output_dtype=torch.bfloat16,
        output=output,
        quant_scales=quant_scales,
        input_sf=x_scale,  # activation scale factors
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
    )
    
    print(f"\n=== SUCCESS ===")
    print(f"Output shape: {result[0].shape}")
    print(f"Output sample: {result[0][0, :8]}")
    print(f"Has NaN: {torch.isnan(result[0]).any()}")
    print(f"Has Inf: {torch.isinf(result[0]).any()}")
    print(f"All zeros (expected with zero weights): {(result[0] == 0).all()}")
    
except Exception as e:
    print(f"\n=== FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
