# SM120 Decode Optimization Analysis

## The Problem

vLLM decode performance on SM120 (GB10) is **2x slower** than llama.cpp:
- vLLM: ~29 tok/s
- llama.cpp: ~58 tok/s

The bottleneck is the **MoE grouped GEMM** kernel which is locked to 128×128 tiles,
resulting in ~0.78% compute efficiency for M=1 (single token decode).

## Two Solutions Investigated

### Option 1: CUTLASS GemvBlockScaled Fallback

**What it is:**
CUTLASS provides `GemvBlockScaled` - a specialized GEMV kernel for block-scaled FP4
that's optimized for M=1 (matrix-vector multiply instead of matrix-matrix multiply).

**Key characteristics:**
- File: `cutlass/gemm/kernel/gemv_blockscaled.h`
- Supports FP4×FP4 with block scaling (SFVecSize=16)
- Uses `cp.async` for efficient memory access
- No tile size constraints (no 128×128 limitation)
- Has batched variant for multiple vectors

**Implementation effort: HIGH**
1. Need to integrate into FlashInfer's `fused_moe/core.py`
2. Need to handle expert routing for GEMV (each expert = separate GEMV)
3. Need to manage separate kernel dispatch for M<threshold vs M>=threshold
4. Need to handle activation quantization (BF16→FP8) before GEMV

**Code path:**
```
cutlass/examples/91_fp4_gemv/91_fp4_gemv.cu  - Example
cutlass/gemm/device/gemv_blockscaled.h       - Device wrapper
cutlass/gemm/kernel/gemv_blockscaled.h       - Kernel implementation
```

**Estimated speedup:** 
- Theoretical: ~10-20x for M=1 (remove tile waste)
- Practical: ~2-4x (memory bandwidth limited)

---

### Option 2: Speculative Decoding (N-gram)

**What it is:**
Speculative decoding generates multiple candidate tokens per step, increasing the
effective batch size (M) for MoE operations.

**vLLM support:**
- `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 5}'`
- N-gram method requires NO draft model (unlike EAGLE/Medusa)
- Works by matching input context with previous patterns

**How it helps MoE:**
- With `num_speculative_tokens=4`: M goes from 1 to 5 (1 + 4 candidates)
- Tile efficiency: 5/128 = 3.9% (5x better than M=1)
- With `num_speculative_tokens=8`: M=9, efficiency = 7.0%

**Implementation effort: LOW**
1. Just add CLI flags to vLLM startup
2. Already integrated with MoE models
3. No kernel changes needed

**Estimated speedup:**
- N-gram with 4 candidates: 20-40% (limited by rejection rate)
- N-gram with 8 candidates: 30-60% 
- Highly dependent on text pattern predictability

---

## Recommendation

### Short-term (hours): Try Speculative Decoding

```bash
# Test with n-gram speculative decoding (no draft model)
vllm serve openai/gpt-oss-120b \
  --quantization mxfp4 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 5}' \
  ...other flags...
```

**Expected outcome:**
- Improved batch sizes for MoE (M=1 → M=5)
- ~20-40% decode throughput improvement
- May have higher acceptance rate on repetitive text

### Medium-term (days): Implement GEMV Fallback

1. Add dispatch logic in `flashinfer/fused_moe/core.py`:
   ```python
   def cutlass_fused_moe(...):
       if num_tokens < GEMV_THRESHOLD:
           return gemv_moe_fallback(...)
       else:
           return grouped_gemm_moe(...)
   ```

2. Implement `gemv_moe_fallback()` using CUTLASS GemvBlockScaled
3. Handle expert routing loop (one GEMV per active expert per token)

### Long-term (weeks): Hybrid Approach

Combine both:
- Speculative decoding to batch tokens when possible
- GEMV fallback for remaining single-token decode steps
- Target: match or exceed llama.cpp's 58 tok/s

---

## Technical Details

### CUTLASS GemvBlockScaled API

```cpp
template <
  typename ElementA_,      // FP4
  typename LayoutA_,       // RowMajor
  typename ElementB_,      // FP4
  typename ElementC_,      // BF16/FP16
  typename ElementAccumulator_,
  typename EpilogueOutputOp_,
  int kElementsPerAccess_ = 32,   // Fixed for FP4
  int kThreadCount_ = 128,
  int kThreadsPerRow_ = 16,
  typename ElementSFA_ = float_e4m3_t,  // Scale factor A
  typename ElementSFB_ = float_e4m3_t,  // Scale factor B
  int kSFVecSize_ = 16
>
struct GemvBlockScaled;
```

### vLLM Speculative Decoding Config

```python
speculative_config = {
    "method": "ngram",           # No draft model needed
    "num_speculative_tokens": 4, # Generate 4 candidates
    "prompt_lookup_max": 5,      # Look back 5 tokens for patterns
    "prompt_lookup_min": 1,      # Minimum pattern size
}
```

---

## Next Steps

1. [ ] Test n-gram speculative decoding with gpt-oss-120b
2. [ ] Measure acceptance rate and throughput impact
3. [ ] If insufficient, prototype GEMV fallback
4. [ ] Benchmark against llama.cpp target (58 tok/s)
