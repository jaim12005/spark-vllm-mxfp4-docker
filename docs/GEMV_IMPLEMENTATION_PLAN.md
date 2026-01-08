# GEMV Implementation Plan for SM120 MoE Decode Optimization

## Executive Summary

**Goal**: Improve decode throughput from ~29 tok/s to ~58 tok/s (match llama.cpp)

**Root Cause**: The CUTLASS grouped GEMM kernel uses 128×128 tiles, resulting in 0.78% compute efficiency for M=1 decode.

**Solution**: Implement a GEMV (matrix-vector multiply) fallback path using CUTLASS `GemvBlockScaled` for small batch sizes.

---

## CUTLASS GEMV Primitives Analysis

### Available Options

| Primitive | Location | Use Case |
|-----------|----------|----------|
| `GemvBlockScaled` | `gemm/kernel/gemv_blockscaled.h` | **Best fit** - FP4×FP4 with block scaling |
| `GemvBatchedStrided` | `gemm/kernel/gemv_batched_strided.h` | Batched GEMV with regular strides |
| `Gemv` (legacy) | `gemm/kernel/gemv.h` | Basic GEMV, no block scaling |

### Recommended: `GemvBlockScaled`

**Why this is the best choice:**

1. **Native FP4 support**: Designed for `__nv_fp4_e2m1` with block scaling
2. **Scale factor handling**: Built-in support for `kSFVecSize=16` (MXFP4 compatible)
3. **Batch support**: Has `batch_count` and `batch_stride` parameters
4. **Proven example**: `examples/91_fp4_gemv/91_fp4_gemv.cu` demonstrates usage

**Key constraints:**
```cpp
static_assert(kSFVecSize == 16, "Only SFVecSize = 16 is supported");
static_assert(kElementsPerAccess == 32, "for fp4 kernel, 32 elt per access");
```

---

## Architecture Design

### Current MoE Flow (Grouped GEMM)
```
tokens (M×H) → TopK routing → Grouped GEMM (all experts) → output
                                    ↑
                            128×128 tiles (0.78% efficiency at M=1)
```

### Proposed Flow (Hybrid GEMV/GEMM)
```
tokens (M×H) → TopK routing → [dispatch based on M]
                                    ↓
                    M < threshold: GEMV per expert (no tile waste)
                    M ≥ threshold: Grouped GEMM (efficient for large M)
```

### Key Design Decisions

1. **Threshold**: Switch from GEMV to GEMM at M=16 or M=32 (to be benchmarked)
2. **Expert handling**: Loop over active experts, one GEMV call per expert
3. **Activation quantization**: BF16 → FP8 before GEMV (reuse existing path)
4. **Weight reuse**: Same MXFP4 weights, no requantization needed

---

## Implementation Plan

### Phase 1: Standalone GEMV Kernel Wrapper (Week 1) ✅ IN PROGRESS

**Goal**: Create a working FP4 GEMV kernel callable from Python

**Files created:**
```
flashinfer/
├── csrc/
│   └── gemv/
│       └── gemv_fp4_blockscaled.cu  # CUDA kernel wrapper ✅
├── flashinfer/
│   └── gemv/
│       ├── __init__.py              # Module exports ✅
│       └── core.py                  # Python bindings ✅

mxfp4/scripts/
└── benchmark_cutlass_gemv.py        # Benchmark script ✅
```

**Tasks:**
1. [x] Create CUDA kernel wrapper using `GemvBlockScaled`
2. [x] Define Python interface matching MoE weight format
3. [x] Add heuristic for GEMV vs GEMM selection (`should_use_gemv_for_moe()`)
4. [ ] Create JIT module for compilation
5. [ ] Handle MXFP4 weight layout (uint8 packed, scale factors)
6. [ ] Write unit tests with known inputs

**Kernel signature (implemented):**
```python
def gemv_fp4_blockscaled(
    activations: torch.Tensor,       # [M, K] uint8 (packed FP4)
    weights: torch.Tensor,           # [K, N] uint8 (packed FP4)
    activation_scales: torch.Tensor, # FP8 scale factors
    weight_scales: torch.Tensor,     # FP8 scale factors
    output: Optional[torch.Tensor],  # [M, N] BF16
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```

### Phase 2: MoE Integration (Week 2)

**Goal**: Integrate GEMV into FlashInfer's MoE pipeline

**Files to modify:**
```
flashinfer/fused_moe/core.py         # Add dispatch logic
flashinfer/fused_moe/gemv_moe.py     # New: GEMV-based MoE implementation
```

**Tasks:**
1. [ ] Add `GEMV_THRESHOLD` configuration (default: 16)
2. [ ] Implement `gemv_moe_forward()` function
3. [ ] Handle expert routing for GEMV path
4. [ ] Fuse activation (SwiGLU) into GEMV epilogue if possible
5. [ ] Add dispatch in `cutlass_fused_moe()`:
   ```python
   if num_tokens < GEMV_THRESHOLD:
       return gemv_moe_forward(...)
   else:
       return grouped_gemm_moe_forward(...)
   ```

### Phase 3: Optimization & Benchmarking (Week 3)

**Goal**: Tune for maximum performance

**Tasks:**
1. [ ] Benchmark GEMV vs GEMM at various M values (1, 2, 4, 8, 16, 32)
2. [ ] Find optimal `GEMV_THRESHOLD`
3. [ ] Profile kernel time breakdown
4. [ ] Optimize expert loop (parallel launches, streams)
5. [ ] Compare against llama.cpp baseline

**Benchmarking script:**
```python
# Sweep M values and measure per-token latency
for m in [1, 2, 4, 8, 16, 32, 64, 128]:
    gemv_time = benchmark_gemv_moe(m, ...)
    gemm_time = benchmark_gemm_moe(m, ...)
    print(f"M={m}: GEMV={gemv_time:.3f}ms, GEMM={gemm_time:.3f}ms")
```

---

## Technical Details

### Weight Layout Compatibility

Current MXFP4 weights in vLLM:
```python
fc1_weight: torch.uint8  # [num_experts, inter_dim, hidden_dim/2] - packed FP4
fc1_scale:  torch.float8_e4m3fn  # [num_experts, inter_dim, hidden_dim/32]
```

GEMV expects:
```cpp
ElementA = __nv_fp4_e2m1  // Activation (quantized from BF16)
ElementB = __nv_fp4_e2m1  // Weight (packed as uint8)
ElementSFA = float_e4m3_t // Activation scale
ElementSFB = float_e4m3_t // Weight scale
```

**Conversion**: Weights are already in correct format. Activations need BF16→FP8→FP4 quantization (or use FP8×FP4 variant).

### Expert Loop Strategy

For M=1 decode with TopK=8 experts:
```python
for expert_idx in active_experts:  # 8 experts
    # FC1: [1, H] × [inter, H]^T → [1, inter]
    gemv(input, fc1_weight[expert_idx], fc1_scale[expert_idx], fc1_out)
    
    # Activation (SwiGLU)
    activated = silu(fc1_out[:, :inter//2]) * fc1_out[:, inter//2:]
    
    # FC2: [1, inter//2] × [H, inter//2]^T → [1, H]
    gemv(activated, fc2_weight[expert_idx], fc2_scale[expert_idx], fc2_out)
    
    output += routing_weight[expert_idx] * fc2_out
```

**Optimization**: Launch all expert GEMVs in parallel using CUDA streams or batch them.

### Activation Fusion

The `GemvBlockScaled` epilogue can fuse:
- Alpha/beta scaling
- Bias addition
- Output scale factor computation

For SwiGLU, we may need a custom epilogue or keep activation as separate kernel.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GEMV memory-bound, not faster | Medium | High | Early benchmarking in Phase 1 |
| Expert loop overhead dominates | Medium | Medium | Parallel expert execution |
| Weight layout mismatch | Low | High | Careful layout analysis |
| JIT compilation complexity | Medium | Medium | Follow existing FlashInfer patterns |

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Decode throughput (M=1) | 29 tok/s | 45 tok/s | 58 tok/s |
| Per-token MoE latency | ~22ms | ~12ms | ~8ms |
| Kernel efficiency (M=1) | 0.78% | 20%+ | 50%+ |

---

## Alternative Approaches (If GEMV Insufficient)

1. **Speculative decoding**: Batch multiple tokens naturally
2. **Persistent kernels**: Keep MoE kernel resident, reduce launch overhead
3. **Custom CUDA kernel**: Hand-tuned GEMV without CUTLASS overhead
4. **INT8 GEMV**: Higher efficiency but needs requantization

---

## Appendix: CUTLASS GemvBlockScaled API Reference

```cpp
template <
  typename ElementA_,              // Input type (__nv_fp4_e2m1)
  typename LayoutA_,               // cutlass::layout::RowMajor
  typename ElementB_,              // Weight type (__nv_fp4_e2m1)
  typename ElementC_,              // Output type (__nv_bfloat16)
  typename ElementAccumulator_,    // Accumulator (float)
  typename EpilogueOutputOp_,      // Epilogue operation
  int kElementsPerAccess_ = 32,    // Fixed for FP4
  int kThreadCount_ = 128,
  int kThreadsPerRow_ = 16,
  typename ElementSFA_ = float_e4m3_t,  // Activation scale
  typename ElementSFB_ = float_e4m3_t,  // Weight scale
  int kSFVecSize_ = 16
>
struct GemvBlockScaled;

// Arguments structure
struct Arguments {
  MatrixCoord problem_size;        // (M, K) - M rows, K columns
  int32_t batch_count;             // Number of batches (experts)
  typename EpilogueOutputOp::Params epilogue;
  
  TensorRefA ref_A;                // Input tensor
  ElementB const *ptr_B;           // Weight pointer
  ElementC const *ptr_C;           // Bias (optional)
  ElementC *ptr_D;                 // Output pointer
  
  ElementSFA const *ptr_SFA;       // Input scale factors
  ElementSFB const *ptr_SFB;       // Weight scale factors
  
  int64_t stride_A;
  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
  int64_t batch_stride_SFA;
  int64_t batch_stride_SFB;
};
```

---

## Progress Summary

### Completed (2026-01-08)

**Phase 1: Standalone GEMV Kernel Wrapper ✅**

Created the following files in FlashInfer:
- `flashinfer/gemv/__init__.py` - Module exports
- `flashinfer/gemv/core.py` - Python interface with `should_use_gemv_for_moe()` heuristic
- `flashinfer/jit/gemv.py` - JIT compilation spec
- `csrc/gemv/gemv_fp4_blockscaled.cu` - CUDA kernel with software dequant fallback
- `csrc/gemv/gemv_epilogue_bf16.h` - Custom epilogue for BF16 output

Created in mxfp4:
- `scripts/benchmark_cutlass_gemv.py` - Benchmark script

**Benchmark Results (Software Dequant Fallback):**
```
GEMV (FP4 fallback): 0.462 ms per call (M=1, K=4096, N=11776)
PyTorch BF16 matmul: 0.427 ms per call

Estimated performance: ~7.2 tok/s (vs. target ~58 tok/s)
```

The software dequantization fallback is slow (as expected) because it:
1. Dequantizes FP4 values in software (no Tensor Core usage)
2. Has suboptimal memory access patterns
3. Doesn't fuse operations

**CUTLASS GemvBlockScaled Integration Attempt ❌**

Attempted to integrate the native CUTLASS `GemvBlockScaled` kernel but encountered:
1. Namespace conflicts between `cute::Tensor` and other Tensor types
2. Complex epilogue interface requirements (expects FP4 output by default)
3. Template parameter matching challenges

The CUTLASS GEMV integration requires more invasive changes to the CUTLASS includes
and is deferred.

**Speculative Decoding Test (2026-01-08)**

Tested ngram speculative decoding as an alternative approach:

```bash
vllm serve ... --speculative-config.method=ngram \
    --speculative-config.num_speculative_tokens=4 \
    --speculative-config.prompt_lookup_max=5
```

**Results:**
- Open-ended generation: ~17-20 tok/s (WORSE than baseline ~29 tok/s)
- Reason: Ngram speculation requires repetitive patterns in output that match the prompt

**Conclusion:** Ngram speculation is NOT suitable for gpt-oss-120b's use case (open-ended generation).
Other speculative methods (suffix, eagle, draft model) may work better but require additional setup.

### Key Findings

1. **CUTLASS GemvBlockScaled** is the right primitive but integration is complex
2. **Ngram speculation** doesn't help for open-ended generation
3. **Current bottleneck** remains the 128×128 tile inefficiency at M=1

### Next: Phase 2 - Alternative Approaches

**Option A: Continue CUTLASS GEMV Integration**
- Fix namespace conflicts by isolating CUTLASS includes
- Create proper custom epilogue matching kernel expectations
- Time estimate: 1-2 weeks

**Option B: Custom CUDA GEMV Kernel**
- Write a hand-tuned GEMV without CUTLASS complexity
- More control but more work
- Time estimate: 2-3 weeks

**Option C: Draft Model Speculation**
- Use a small draft model for speculative decoding
- Better for open-ended generation than ngram
- Time estimate: 1 week (if draft model available)

**Recommendation:** Try Option C (draft model speculation) first as it's lowest effort.
If insufficient, pursue Option A (CUTLASS GEMV integration).

### Remaining Work

1. **Try draft model speculation** with a small compatible model
2. **If insufficient**: Continue CUTLASS GemvBlockScaled integration
3. **Phase 3**: Add dispatch logic in FlashInfer MoE pipeline
4. **Phase 4**: Benchmark and optimize to match llama.cpp (~58 tok/s)

