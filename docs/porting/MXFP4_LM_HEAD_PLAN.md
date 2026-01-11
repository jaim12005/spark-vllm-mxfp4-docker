---
name: MXFP4 lm_head Plan
overview: Fix MXFP4 infrastructure in vLLM (currently simulation-only), then implement MXFP4 quantization for lm_head to reduce memory bandwidth from 1.16GB to 0.29GB per token, achieving an estimated 30-45% decode speedup to match llama.cpp's 58 tok/s.
todos:
  - id: fix-mxfp4-quantize
    content: "Phase 0: Fix _mxfp4_quantize to return real quantized tensors like MXFP8"
    status: pending
  - id: add-mxfp4-utils
    content: "Phase 0: Create mxfp4_utils.py wrapper for flashinfer.mxfp4_quantize"
    status: pending
  - id: mxfp4-linear-method
    content: "Phase 1: Implement Mxfp4LinearMethod class in mxfp4.py"
    status: pending
  - id: weight-quantization
    content: "Phase 2: Load BF16, quantize to MXFP4 via process_weights_after_loading (Pattern B)"
    status: pending
  - id: fp4-gemm-apply
    content: "Phase 3: Implement apply() using group_gemm_mxfp8_mxfp4_nt_groupwise (FP8×FP4)"
    status: pending
  - id: get-quant-method
    content: "Phase 1: Update get_quant_method() to return Mxfp4LinearMethod for LinearBase"
    status: pending
  - id: accuracy-test
    content: "Phase 4: Create accuracy validation test comparing BF16 vs MXFP4 lm_head"
    status: pending
  - id: benchmark
    content: "Phase 4: Benchmark tg32 performance with MXFP4 lm_head"
    status: pending
  - id: small-m-kernel
    content: "Phase 5 (if needed): Implement small-M specialized GEMV kernel for decode"
    status: pending
---

# MXFP4 lm_head Optimization Plan

## Problem Summary

The lm_head layer is a dense GEMM that converts hidden states to logits over the vocabulary. In gpt-oss-120b:

| Property | Value |

|----------|-------|

| vocab_size | 201,088 |

| hidden_size | 2,880 |

| lm_head shape | [201088, 2880] |

| BF16 size | 1.16 GB |

| MXFP4 size | 0.29 GB |

**Current state**: vLLM uses BF16 for lm_head because:

1. Model config explicitly excludes it: `"modules_to_not_convert": ["lm_head", ...]`
2. `Mxfp4Config.get_quant_method()` returns `UnquantizedLinearMethod()` for `LinearBase`
3. Native checkpoint stores lm_head as BF16

**llama.cpp**: Uses MXFP4 for lm_head via GGUF conversion, achieving 4x memory reduction.

---

## Memory Bandwidth Impact

```
GB10 bandwidth: 273 GB/s

BF16: 1.16 GB / 273 GB/s = 4.24 ms per token
MXFP4: 0.29 GB / 273 GB/s = 1.06 ms per token
Delta: 3.18 ms = ~10 tok/s penalty at 29 tok/s baseline
```

---

## Implementation Strategy

### Phase 0: Fix MXFP4 Infrastructure in vLLM

**Problem**: vLLM's `_mxfp4_quantize()` currently simulates quantization by doing quant+dequant round-trip, returning BF16 data with no memory savings. MXFP8 is properly implemented.

**Current (broken):**

```python
# vllm/model_executor/layers/fused_moe/utils.py:177
def _mxfp4_quantize(...):
    A = quant_dequant_mxfp4(A)  # Simulates, returns BF16
    return A, None  # No scales!
```

**Target (like MXFP8):**

```python
def _mxfp4_quantize(...):
    return mxfp4_e2m1_quantize(A)  # Returns (FP4 tensor, E8M0 scales)
```

**Files to modify:**

1. **Create `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`**

                        - Wrapper for `flashinfer.mxfp4_quantize()` (mirrors `mxfp8_utils.py`)

2. **Fix `vllm/model_executor/layers/fused_moe/utils.py`**

                        - Update `_mxfp4_quantize()` to return real quantized tensors
                        - Add platform capability check: `current_platform.supports_mx()`

---

### Phase 1: MXFP4 Linear Method for lm_head

**Files to modify:**

1. **[vllm/model_executor/layers/quantization/mxfp4.py](~/projects/vllm/vllm/model_executor/layers/quantization/mxfp4.py)**

                        - Implement `Mxfp4LinearMethod` class (parallel to `Mxfp4MoEMethod`)
                        - Handle FP4 weight loading with E8M0 block scales
                        - Use FlashInfer `mm_fp4()` or cuDNN FP4 GEMM for compute

2. **[vllm/model_executor/layers/quantization/mxfp4.py](~/projects/vllm/vllm/model_executor/layers/quantization/mxfp4.py)** - `get_quant_method()`

                        - Change from returning `UnquantizedLinearMethod()` to `Mxfp4LinearMethod()`
                        - Still respect `ignored_layers` config

3. **[vllm/model_executor/layers/vocab_parallel_embedding.py](~/projects/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py)**

                        - Ensure `ParallelLMHead` can accept quantized weights

### Phase 2: Weight Conversion at Load Time

Since the HuggingFace checkpoint stores lm_head as BF16, we need runtime quantization:

1. **Load BF16 weights** from safetensors/fastsafetensors
2. **Quantize to MXFP4** using `flashinfer.mxfp4_quantize()`
3. **Replace weight parameter** with packed FP4 + E8M0 scales

**Location**: `Mxfp4LinearMethod.process_weights_after_loading()`

**Critical: Checkpoint Loading Compatibility**

We must use **Pattern B** (load BF16 then quantize) to support both:

- `--load-format safetensors` (standard HuggingFace loader)
- `--load-format fastsafetensors` (optimized parallel loader)

Pattern B means:

1. `create_weights()` allocates BF16 weight with original checkpoint shape
2. Standard weight_loader loads BF16 data from checkpoint  
3. `process_weights_after_loading()` quantizes BF16 → MXFP4 and replaces the weight

This is the same pattern used by FP8's non-serialized checkpoint handling (see `fp8.py` lines 497-538).

### Phase 3: Kernel Selection

**Critical insight**: FlashInfer's `mm_fp4()` expects **both** A and B to be FP4. For FP8 activations × MXFP4 weights, we need a different kernel.

**Available FlashInfer Kernels:**

| Kernel | A dtype | B dtype | Notes |

|--------|---------|---------|-------|

| `mm_fp4()` | FP4 | FP4 | Both inputs FP4 |

| `bmm_mxfp8()` | FP8 | FP8 | Both inputs FP8 |

| `group_gemm_mxfp8_mxfp4_nt_groupwise()` | **FP8** | **FP4** | FP8 acts × FP4 weights |

**Recommended approach for lm_head (FP8×FP4):**

Use `group_gemm_mxfp8_mxfp4_nt_groupwise()` with batch_size=1:

```python
from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise
from flashinfer import mxfp8_quantize, mxfp4_quantize

# Weights: MXFP4 (prepared at load time)
# weight_fp4: [n, k // 2], uint8 (packed FP4)
# weight_scale: [n_padded, k // 32], uint8 (E8M0)

# Activations: quantize BF16 → FP8 at runtime
x_fp8, x_scale = mxfp8_quantize(hidden_states)  # [m, k], fp8_e4m3fn

# Create indptr for single "group" (dense GEMM as batch_size=1 grouped GEMM)
m_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=x.device)

# FP8 × FP4 GEMM
output = group_gemm_mxfp8_mxfp4_nt_groupwise(
    a=x_fp8,                    # [m, k], FP8
    b=weight_fp4.unsqueeze(0),  # [1, n, k // 2], packed FP4
    a_scale=x_scale,            # [m_padded, k // 32], uint8
    b_scale=weight_scale.unsqueeze(0),  # [1, n_padded, k // 32], uint8
    m_indptr=m_indptr,          # [2], int32
    out_dtype=torch.bfloat16,
)
```

**Alternative: FP4×FP4 with `mm_fp4()`**

If FP8 quantization overhead is too high for decode, can use FP4 activations:

```python
from flashinfer import mm_fp4, mxfp4_quantize

# Quantize activations to FP4 (less accurate but simpler)
x_fp4, x_scale = mxfp4_quantize(hidden_states)

output = mm_fp4(
    x_fp4, weight_fp4.T,
    x_scale, weight_scale.T,
    out_dtype=torch.bfloat16,
    block_size=32,
    use_nvfp4=False
)
```

**Weight layout requirements for `group_gemm_mxfp8_mxfp4_nt_groupwise()`:**

- B: column-major, shape `[batch_size, n, k // 2]`, uint8
- B_scale: row-major, shape `[batch_size, n_padded, k // 32]`, uint8
- Scale granularity: K//32 (MXFP4 block size)

---

## Implementation Details

### Mxfp4LinearMethod Class

**Critical: Weight loading strategy**

The checkpoint stores lm_head as BF16. We must:

1. Load BF16 weights from checkpoint (not uint8)
2. Quantize to MXFP4 in `process_weights_after_loading()`
3. Use `replace_parameter()` to swap weights with quantized versions

This follows the same pattern as FP8's non-serialized checkpoint handling.

```python
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import replace_parameter

class Mxfp4LinearMethod(LinearMethodBase):
    """MXFP4 linear layer for dense projections (lm_head, etc.)."""
    
    def __init__(self):
        self.mxfp4_block_size = 32  # Group size for MXFP4
    
    def create_weights(self, layer, input_size_per_partition, 
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        
        # Store dimensions for process_weights_after_loading
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        
        # CRITICAL: Load as BF16 first (checkpoint is BF16)
        # Will be quantized to MXFP4 in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,  # BF16 from checkpoint
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        
        # Scales will be created after quantization
        layer.register_parameter("weight_scale", None)
    
    def process_weights_after_loading(self, layer):
        """Quantize BF16 weights to MXFP4 after checkpoint loading."""
        from flashinfer import mxfp4_quantize
        
        # Quantize BF16 → MXFP4 (packed FP4 + E8M0 scales)
        weight_fp4, weight_scale = mxfp4_quantize(layer.weight.data)
        
        # Replace BF16 weight with packed FP4 (frees BF16 memory)
        replace_parameter(layer, "weight", weight_fp4)
        
        # Add scale parameter
        replace_parameter(layer, "weight_scale", weight_scale)
    
    def apply(self, layer, x, bias=None):
        from flashinfer import mxfp8_quantize
        from flashinfer.gemm import group_gemm_mxfp8_mxfp4_nt_groupwise
        
        num_tokens = x.shape[0]
        
        # Quantize activations BF16 → FP8
        x_fp8, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True, sf_vec_size=32)
        
        # Pad num_tokens to multiple of 4 (kernel requirement)
        m_padded = (num_tokens + 3) // 4 * 4
        m_indptr = torch.tensor([0, m_padded], dtype=torch.int32, device=x.device)
        
        # FP8 × FP4 GEMM (using grouped GEMM with batch_size=1)
        output = group_gemm_mxfp8_mxfp4_nt_groupwise(
            a=x_fp8,                              # [m, k], FP8
            b=layer.weight.unsqueeze(0),          # [1, n, k // 2], packed FP4
            a_scale=x_scale,                      # [m_padded, k // 32], uint8
            b_scale=layer.weight_scale.unsqueeze(0),  # [1, n_padded, k // 32], uint8
            m_indptr=m_indptr,
            out_dtype=torch.bfloat16,
        )
        
        # Trim padding if added
        if num_tokens < m_padded:
            output = output[:num_tokens]
        
        if bias is not None:
            output = output + bias
        return output
```

### LogitsProcessor Compatibility

The [logits_processor.py](~/projects/vllm/vllm/model_executor/layers/logits_processor.py) calls:

```python
logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
```

This already uses the quant_method abstraction, so `Mxfp4LinearMethod.apply()` will be called automatically.

---

## Testing Plan

1. **Unit test**: Verify MXFP4 weight quantization matches expected output
2. **Accuracy test**: Compare logits between BF16 and MXFP4 lm_head (expect <0.5% difference)
3. **Performance test**: Measure tok/s improvement on gpt-oss-120b

---

## Expected Results

| Metric | Before | After | Change |

|--------|--------|-------|--------|

| lm_head memory | 1.16 GB | 0.29 GB | -75% |

| lm_head latency | 4.2 ms | ~1.5-2.5 ms* | -40-65% |

| tg32 | 29 tok/s | 35-40 tok/s | +20-38% |

*Note: Theoretical lower bound is 1.1 ms based on pure bandwidth. Actual will be higher due to small-M inefficiency (see Risk #3).

---

## Risks and Mitigations

| Risk | Severity | Mitigation |

|------|----------|------------|

| Accuracy degradation | Medium | Use FP8 activations to maintain precision; validate with perplexity tests |

| cuDNN FP4 not available | Low | Fallback to CUTLASS or DP4A GEMV kernel |

| Weight loading compatibility | Low | Pattern B (load BF16, quantize at runtime) supports all loaders |

| **Small-M decode inefficiency** | **High** | See detailed analysis below |

---

## Risk #3: Small-M Decode Inefficiency (Critical)

### Problem

The `group_gemm_mxfp8_mxfp4_nt_groupwise()` kernel has constraints that hurt small-M (decode) performance:

| Constraint | Impact for M=1-2 |

|------------|------------------|

| `m_indptr` must be multiple of 4 | Must pad M=1 → M=4 (75% wasted compute) |

| `tile_m` fixed at 128 | One tile processes 128 rows, but only 1-2 used |

| Kernel launch overhead | Fixed cost dominates small problem sizes |

### Performance Reality

```
Theoretical (bandwidth-limited):
  MXFP4 lm_head: 0.29 GB / 273 GB/s = 1.06 ms

Actual (small-M inefficiency):
 - Padding overhead: M=1 → M=4 = 4x wasted work
 - Tile underutilization: 1/128 = 0.78% efficiency  
 - Launch overhead: ~0.1-0.2 ms fixed cost
  
  Realistic estimate: 1.5-2.5 ms (vs theoretical 1.1 ms)
```

BF16 lm_head also suffers from small-M inefficiency, so the **relative speedup is still meaningful** (4.2ms → 1.5-2.5ms = 40-65% reduction).

### Potential Solutions (Future Work)

1. **DP4A-based GEMV kernel** for M≤4

                        - llama.cpp uses this approach
                        - Per-element granularity, no tile waste
                        - Would need custom kernel in FlashInfer

2. **cuBLAS/cuBLASLt FP4 GEMV**

                        - Check if cuBLAS has small-M optimized FP4 path
                        - May require CUDA 13.0+

3. **Batching multiple decode steps**

                        - Speculative decoding increases effective M
                        - Eagle3 with 3-4 spec tokens → M=4-8

4. **Hybrid approach**

                        - Use grouped GEMM for prefill (large M)
                        - Use GEMV kernel for decode (M≤4)

### Recommendation

Implement Phase 1-3 with `group_gemm_mxfp8_mxfp4_nt_groupwise()` first. Benchmark to measure actual improvement. If decode is still bottlenecked, investigate small-M specialized kernel as Phase 5

---

## Files to Create/Modify

| Phase | File | Action |

|-------|------|--------|

| 0 | `vllm/.../quantization/utils/mxfp4_utils.py` | **Create**: Wrapper for `flashinfer.mxfp4_quantize()` |

| 0 | `vllm/.../fused_moe/utils.py` | **Fix**: `_mxfp4_quantize()` to return real tensors |

| 1 | `vllm/.../quantization/mxfp4.py` | **Add**: `Mxfp4LinearMethod` class |

| 1 | `vllm/.../quantization/mxfp4.py` | **Modify**: `get_quant_method()` to return MXFP4 for LinearBase |

| 4 | `docs/porting/MXFP4_LM_HEAD_PLAN.md` | This plan document |

| 4 | `scripts/test_mxfp4_lm_head.py` | Validation script |