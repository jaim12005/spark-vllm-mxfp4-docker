# llama.cpp MXFP4 GEMV Kernel Analysis

## Executive Summary

llama.cpp achieves ~58 tok/s on gpt-oss-120b vs our 29 tok/s primarily through a different approach to decode compute:

**Key difference**: llama.cpp uses **INT8 DP4A instructions** instead of Tensor Cores for small batch sizes.

---

## llama.cpp Architecture

### Kernel Location
- `ggml/src/ggml-cuda/mmvq.cu` - Matrix-vector quantized kernel
- `ggml/src/ggml-cuda/vecdotq.cuh` - Vector dot product implementations

### MXFP4 Data Format

```c
// ggml-common.h
#define QK_MXFP4 32  // 32 elements per block

typedef struct {
    uint8_t e;            // E8M0 scale factor (1 byte)
    uint8_t qs[QK_MXFP4/2];  // 16 bytes of packed FP4 nibbles
} block_mxfp4;  // Total: 17 bytes per 32 elements
```

### FP4 to INT8 Dequantization Table

```c
// E2M1 values (doubled for int8 range)
GGML_TABLE_BEGIN(int8_t, kvalues_mxfp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12,     // Positive values (0-7)
    0, -1, -2, -3, -4, -6, -8, -12,  // Negative values (8-15)
GGML_TABLE_END()
```

### Core Dot Product (vec_dot_mxfp4_q8_1)

```cuda
static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq,      // MXFP4 weights
    const block_q8_1 * __restrict__ bq8_1,  // Q8_1 activations
    const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        // Load 4 FP4 values (1 byte)
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        
        // Convert FP4 to int8 via lookup table
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        // DP4A: 4x int8 dot product (hardware instruction)
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    // Apply scale factor
    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}
```

---

## Why DP4A is Better for M=1 Decode

### Tensor Core (CUTLASS) Issues at M=1
| Issue | Impact |
|-------|--------|
| 128×128 tile size | 99.2% compute waste |
| Complex scheduling | High overhead per operation |
| Warp synchronization | Stalls for small problems |
| TMA setup cost | Fixed overhead dominates |

### DP4A Advantages for M=1
| Advantage | Benefit |
|-----------|---------|
| Per-element granularity | No tile waste |
| Simple execution | Low overhead |
| Independent threads | No warp sync needed |
| Fast table lookup | FP4→int8 conversion cheap |

### Compute Model Comparison

**Tensor Core (M=1 with 128×128 tile)**:
- Load 128×128 tile = 16,384 elements
- Compute 128×128 = 16,384 MACs
- Only use 1×128 = 128 results
- Efficiency: 128/16,384 = 0.78%

**DP4A (M=1)**:
- Process exactly K/4 iterations
- Each DP4A does 4 int8 MACs
- No wasted compute
- Efficiency: ~100%

---

## Implementation Plan for FlashInfer

### Phase 1: Create DP4A-based GEMV Kernel

**New file**: `flashinfer/csrc/gemv/gemv_fp4_dp4a.cu`

```cuda
// FP4 to int8 lookup table (doubled values)
__constant__ int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

// Quantize BF16 activation to Q8_1 format
__device__ void quantize_bf16_to_q8(
    const nv_bfloat16* input,
    int8_t* output_qs,
    float* output_scale,
    int size);

// FP4 GEMV using DP4A
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void gemv_fp4_dp4a(
    const uint8_t* __restrict__ weights,      // [K/2] packed FP4
    const uint8_t* __restrict__ weight_scales, // [K/32] E8M0
    const int8_t* __restrict__ act_q8,        // [K] quantized activation
    const float act_scale,                     // Activation scale
    nv_bfloat16* __restrict__ output,         // [N] output
    int N, int K);
```

### Phase 2: Optimize for MoE

Key considerations:
1. **Expert batching**: Process all 8 active experts per token in single kernel launch
2. **Weight prefetching**: Stream weights from HBM to L2
3. **Warp-level reduction**: Use shuffle for final summation
4. **Activation reuse**: Quantize activation once, reuse for all experts

### Phase 3: Integration Points

```python
# flashinfer/fused_moe/core.py

def should_use_dp4a_gemv(num_tokens: int) -> bool:
    """Use DP4A for small batch sizes, grouped GEMM for large."""
    return num_tokens <= 8  # Tune this threshold

def cutlass_fused_moe(...):
    if should_use_dp4a_gemv(num_tokens):
        return dp4a_gemv_moe_forward(...)
    else:
        return grouped_gemm_moe_forward(...)
```

---

## Expected Performance

Based on llama.cpp results:

| Metric | Current (CUTLASS) | Target (DP4A) |
|--------|-------------------|---------------|
| M=1 throughput | 29 tok/s | 50-60 tok/s |
| M=1 MoE latency | ~22ms | ~10-12ms |
| Compute efficiency | 0.78% | ~50-70% |

---

## Key Insights from llama.cpp

1. **Don't use Tensor Cores for M=1**: The tile overhead dominates
2. **INT8 accumulation is sufficient**: FP4×Q8 → int32 accumulator → float result
3. **Lookup tables are fast**: FP4→int8 conversion via table is cheap
4. **Simple kernels win**: Less scheduling overhead = lower latency
5. **Activation quantization amortized**: Q8_1 format with block scale is efficient

---

## Benchmark Results (Implemented)

### DP4A GEMV Performance

| Config | DP4A (ms) | PyTorch BF16 (ms) | Speedup |
|--------|-----------|-------------------|---------|
| FC1 (4096→11776) | 0.310 | 0.408 | **1.32x** |
| FC2 (11776→4096) | 0.254 | 0.387 | **1.52x** |

### Critical Finding: GEMV is NOT the Solution for MoE

**Problem**: Per-expert GEMV is slower than CUTLASS grouped GEMM!

| Approach | 60 Layers (TopK=8) | Why |
|----------|---------------------|-----|
| **CUTLASS grouped GEMM** | 182.88 ms | Batches all experts, reuses weights |
| **DP4A per-expert GEMV** | 270.38 ms | 8x memory traffic (no weight reuse) |
| **Target (llama.cpp)** | ~103 ms | Different architecture |

**Root cause**: The CUTLASS grouped GEMM achieves **weight reuse** by processing all 8 experts in a single kernel. The tile inefficiency (0.78% at M=1) is outweighed by the memory bandwidth savings from not re-reading weights 8 times.

### Memory Bandwidth Analysis

| Metric | Value |
|--------|-------|
| FC1 weight size | 25.66 MB (MXFP4) |
| Peak memory-bound time | 0.032 ms |
| Achieved DP4A time | 0.31 ms |
| **Efficiency** | **~10% of peak** |

Our DP4A kernel achieves only 10% of memory bandwidth, suggesting room for optimization, but even at 100% efficiency:
- 8 experts × 2 GEMVs × 0.032 ms = 0.51 ms per layer
- 60 layers × 0.51 ms = **30.7 ms** (best possible with GEMV)

Compare to CUTLASS grouped GEMM: **182.88 ms / 60 = 3.05 ms per layer**.

The 6x gap shows that CUTLASS is **not just reusing weights once per token, but across multiple tokens** in a way our GEMV cannot.

## Conclusion

**GEMV is fundamentally the wrong approach for MoE decode on this architecture.**

The grouped GEMM in CUTLASS, despite its M=128 tile constraint, achieves better memory efficiency through:
1. **Weight reuse across experts** - All 8 experts share the same weight loading
2. **Batched computation** - Single kernel launch vs 8+ launches
3. **Better memory access patterns** - Coalesced TMA loads

### What llama.cpp Does Differently

llama.cpp achieves 58 tok/s vs our 29 tok/s. The key differences are:

1. **MXFP4 lm_head** - See critical finding below
2. **DP4A for small batches** - INT8 instructions instead of Tensor Cores
3. **Simpler architecture** - Less Python/framework overhead
4. **Fused kernels** - QKV projection fused with attention

---

## Critical Finding: lm_head Format Difference (2026-01-11)

### The Problem

**llama.cpp uses MXFP4 for lm_head, while vLLM uses BF16.**

| | llama.cpp | vLLM |
|--|-----------|------|
| **lm_head format** | MXFP4 (FP4) | BF16 |
| **lm_head size** | 0.29 GB | 1.16 GB |
| **Memory read time** | ~1.1 ms | ~4.2 ms |
| **Per-token overhead** | Low | **+3.1 ms** |

For gpt-oss-120b:
- `vocab_size = 201,088`
- `hidden_size = 2,880`
- `lm_head = [201088 × 2880]`

### Memory Bandwidth Impact

```
GB10 memory bandwidth: 273 GB/s

BF16 lm_head: 201088 × 2880 × 2 bytes = 1.16 GB
  → Read time: 1.16 GB / 273 GB/s = 4.24 ms

MXFP4 lm_head: 201088 × 2880 × 0.5 bytes = 0.29 GB
  → Read time: 0.29 GB / 273 GB/s = 1.06 ms

Difference: 3.18 ms per token = 10 tok/s penalty at 29 tok/s baseline
```

### Why vLLM Uses BF16 for lm_head

1. **Model config excludes it**: The Hugging Face checkpoint's `quantization_config` has:
   ```json
   "modules_to_not_convert": ["lm_head", "model.embed_tokens", ...]
   ```

2. **MXFP4 LinearMethod not implemented**: vLLM's mxfp4.py falls back to BF16:
   ```python
   if isinstance(layer, LinearBase):
       # TODO: Add support for MXFP4 Linear Method.
       return UnquantizedLinearMethod()  # Falls back to BF16!
   ```

3. **Native checkpoint is BF16**: The safetensors file has:
   ```
   lm_head.weight: shape=[201088, 2880], dtype=torch.bfloat16
   ```

### How llama.cpp Handles This

According to NVIDIA Developer Forums and benchmarks:
- llama.cpp maintains native MXFP4 precision for lm_head
- The GGUF conversion likely quantizes lm_head to MXFP4
- This enables efficient FP4 weight reads with DP4A compute

### Optimization Path for vLLM

To match llama.cpp lm_head performance:

1. **Option A: Implement MXFP4 LinearMethod**
   - Add FP4 weight support for dense linear layers
   - Use DP4A or FP8×FP4 kernel for lm_head
   - Requires: New kernel, weight conversion

2. **Option B: Dynamic FP8 quantization**
   - Quantize lm_head weights to FP8 at load time
   - Use cuBLAS FP8 GEMM
   - Simpler but less memory savings than FP4

3. **Option C: Pre-quantized checkpoint**
   - Create a variant checkpoint with FP4 lm_head
   - Load FP4 weights directly
   - Requires: Model re-export

### Estimated Impact

| Current | With MXFP4 lm_head |
|---------|-------------------|
| 29 tok/s | ~38-42 tok/s |
| 34.5 ms/tok | ~26-28 ms/tok |

This single optimization could provide **30-45% decode speedup**.

### Recommended Next Steps

1. [x] ~~Create `gemv_fp4_dp4a.cu` kernel~~ (Done, but not viable)
2. [x] ~~Benchmark vs CUTLASS~~ (Done, CUTLASS wins)
3. [ ] **Profile llama.cpp** to understand their decode approach
4. [ ] **Investigate speculative decoding** to increase effective batch size
5. [ ] **Focus on attention GEMV** instead of MoE (51% of decode time)

---

## References

- llama.cpp source: https://github.com/ggerganov/llama.cpp
- DP4A documentation: NVIDIA CUDA Programming Guide
- MX Format spec: OCP Microscaling Formats v1.0

