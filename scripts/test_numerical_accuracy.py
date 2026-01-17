#!/usr/bin/env python3
"""Test numerical accuracy of CUTLASS MoE kernel against BF16 reference.

This verifies the kernel produces numerically correct results, not just
"not NaN" but actually close to the expected output.
"""
import torch
import sys
sys.path.insert(0, '/workspace/flashinfer')

from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fused_moe.core import cutlass_fused_moe, ActivationType

torch.manual_seed(42)
device = "cuda"

def silu(x):
    return x * torch.sigmoid(x)

def bf16_reference_moe(x_bf16, w13_bf16, w2_bf16, topk_ids, topk_weights):
    """Reference implementation in BF16."""
    num_tokens, hidden_size = x_bf16.shape
    num_experts = w13_bf16.shape[0]
    intermediate_size = w2_bf16.shape[2]
    top_k = topk_ids.shape[1]
    
    output = torch.zeros_like(x_bf16)
    
    for token_idx in range(num_tokens):
        token_output = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
        for k in range(top_k):
            expert_id = topk_ids[token_idx, k].item()
            weight = topk_weights[token_idx, k].item()
            
            # FC1: x @ W13^T -> (2*intermediate,)
            # W13 is [2*intermediate, hidden]
            fc1_out = x_bf16[token_idx] @ w13_bf16[expert_id].T
            
            # SwiGLU: gate = first half, up = second half
            gate = fc1_out[:intermediate_size]
            up = fc1_out[intermediate_size:]
            hidden = silu(gate) * up
            
            # FC2: hidden @ W2^T -> (hidden,)
            # W2 is [hidden, intermediate]
            fc2_out = hidden @ w2_bf16[expert_id].T
            
            token_output += weight * fc2_out
        
        output[token_idx] = token_output
    
    return output

def main():
    # Small dimensions for quick test
    hidden_size = 256
    intermediate_size = 512
    num_experts = 4
    num_tokens = 8
    top_k = 2
    
    print("=" * 60)
    print("Numerical Accuracy Test: CUTLASS vs BF16 Reference")
    print("=" * 60)
    print(f"hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
    print(f"num_experts: {num_experts}, num_tokens: {num_tokens}, top_k: {top_k}")
    
    # Create inputs
    x_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    
    # Create weights with small values
    w13_bf16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                           dtype=torch.bfloat16, device=device) * 0.05
    w2_bf16 = torch.randn(num_experts, hidden_size, intermediate_size,
                          dtype=torch.bfloat16, device=device) * 0.05
    
    # Routing - ensure no duplicate experts per token
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).float()
    topk_ids = topk_ids.to(torch.int32)
    
    print(f"\nRouting sample (token 0): experts={topk_ids[0].tolist()}, weights={topk_weights[0].tolist()}")
    
    # BF16 Reference
    print("\nComputing BF16 reference...")
    ref_output = bf16_reference_moe(x_bf16, w13_bf16, w2_bf16, topk_ids, topk_weights)
    print(f"Reference output sample: {ref_output[0, :4]}")
    
    # CUTLASS kernel
    print("\nComputing CUTLASS FP8xFP4 kernel...")
    
    # Quantize input to FP8
    x_quant, x_scale = mxfp8_quantize(x_bf16, True, 32)
    
    # Quantize weights to FP4
    w13_flat = w13_bf16.reshape(-1, hidden_size)
    w2_flat = w2_bf16.reshape(-1, intermediate_size)
    w13_fp4, w13_scale = mxfp4_quantize(w13_flat)
    w2_fp4, w2_scale = mxfp4_quantize(w2_flat)
    
    w13_fp4 = w13_fp4.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
    w2_fp4 = w2_fp4.reshape(num_experts, hidden_size, intermediate_size // 2)
    w13_scale = w13_scale.reshape(num_experts, 2 * intermediate_size, hidden_size // 32)
    w2_scale = w2_scale.reshape(num_experts, hidden_size, intermediate_size // 32)
    
    fc1_weights = w13_fp4.contiguous().view(torch.long)
    fc2_weights = w2_fp4.contiguous().view(torch.long)
    fc1_scale = w13_scale.contiguous().view(torch.int32)
    fc2_scale = w2_scale.contiguous().view(torch.int32)
    
    fake_input_scale = torch.ones(num_experts, device=device)
    quant_scales = [fc1_scale, fake_input_scale, fc2_scale, fake_input_scale]
    
    result = cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True,
        input_sf=x_scale,
        quant_scales=quant_scales,
    )
    
    cutlass_output = result[0]
    print(f"CUTLASS output sample: {cutlass_output[0, :4]}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    diff = (cutlass_output - ref_output).abs()
    rel_diff = diff / (ref_output.abs() + 1e-6)
    
    print(f"Absolute error: max={diff.max():.4f}, mean={diff.mean():.4f}")
    print(f"Relative error: max={rel_diff.max():.2%}, mean={rel_diff.mean():.2%}")
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        cutlass_output.flatten().unsqueeze(0).float(),
        ref_output.flatten().unsqueeze(0).float()
    ).item()
    print(f"Cosine similarity: {cos_sim:.4f}")
    
    # Pearson correlation
    cutlass_flat = cutlass_output.flatten().float()
    ref_flat = ref_output.flatten().float()
    pearson = torch.corrcoef(torch.stack([cutlass_flat, ref_flat]))[0, 1].item()
    print(f"Pearson correlation: {pearson:.4f}")
    
    # Judgment
    print("\n" + "=" * 60)
    if cos_sim > 0.95 and pearson > 0.95:
        print("✅ PASS: High correlation, kernel output is coherent with reference")
    elif cos_sim > 0.80 and pearson > 0.80:
        print("⚠️ MARGINAL: Moderate correlation, some quantization error expected")
    else:
        print("❌ FAIL: Low correlation, potential numerical issues")
    print("=" * 60)
    
    # Note about expected error
    print("\nNote: FP8×FP4 has significant quantization error vs BF16.")
    print("Cosine sim > 0.90 is good, > 0.95 is excellent for this precision.")

if __name__ == "__main__":
    main()
