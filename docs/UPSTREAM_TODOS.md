# FlashInfer Upstream Improvement Opportunities

This document tracks improvement opportunities identified in FlashInfer that could benefit MXFP4 performance on SM121.

**Reference format**: Durable references use commit SHA + file + grep-able snippet. Line numbers are approximate and may drift.

---

## High Priority

### 1. Reduce Block Scale Granularity

**Location**: `csrc/fused_moe/sm12x_activation_quantizer.cuh`
**Snippet**: `kSm12xBlockScaleGranularity = 128`
**Reference commit**: Check current upstream/main

**Issue**: The 128-element scale granularity prevents using M=64 tiles for better decode efficiency.

**Potential fix**:
- Reduce `kSm12xBlockScaleGranularity` from 128 to 64
- Update scale factor packing in quantizer
- May require changes to weight quantization format

**Impact**: Could enable 2x better compute utilization for M=1 decode.

**Effort**: High - significant architectural change

---

### 2. Runner/Setup Caching for MoE

**Location**: `flashinfer/fused_moe/core.py`
**Snippet**: `def cutlass_fused_moe`
**Reference commit**: Check current upstream/main

**Issue**: Each MoE call re-creates the CUTLASS runner, adding overhead especially for CUDA graphs.

**Potential fix**:
- Cache runner objects by configuration signature
- Pre-allocate workspace buffers
- Lazy initialization with caching

**Impact**: Reduce per-call overhead, improve TTFT.

**Effort**: Medium

---

### 3. Pre-compiled Kernel Cache

**Location**: JIT compilation system
**Snippet**: `@jit_compile` decorators

**Issue**: First-run JIT compilation takes 3-5 minutes on SM121.

**Potential fix**:
- Ship pre-compiled .so/.cubin for common SM architectures
- Include SM121 in the pre-compiled set
- Add fallback to JIT only for unsupported configs

**Impact**: Eliminate cold-start delay.

**Effort**: Medium - build system changes

---

## Medium Priority

### 4. Activation Persistence (FP8 Between Layers)

**Location**: `flashinfer/fused_moe/core.py`
**Snippet**: `mxfp8_quantize` calls

**Issue**: Currently quantize BF16→FP8 at every MoE layer. Could persist FP8 between layers.

**Potential fix**:
- Modify output dtype to FP8 when next layer expects FP8 input
- Skip input quantization when already FP8
- Requires model-level changes in vLLM

**Impact**: Reduce quantization overhead (currently ~2% of decode time).

**Effort**: High - requires vLLM model changes

---

### 5. Smaller M Tiles for Decode

**Location**: `csrc/fused_moe/sm12x_grouped_gemm.cuh`
**Snippet**: Tile configuration selection

**Issue**: Only M=128 tiles available, wastes compute for M=1 decode.

**Potential fix**:
- Add M=32 or M=64 tile configurations
- Requires solving TMA layout constraints with scale granularity
- May need custom epilogue for smaller tiles

**Impact**: Up to 4x better utilization for small M.

**Effort**: High - blocked by scale granularity issue (#1)

---

### 6. GEMV Fallback for Very Small M

**Location**: Dispatch logic in `cutlass_fused_moe`
**Snippet**: N/A - not yet implemented

**Issue**: For M=1, a software GEMV might be faster than grouped GEMM.

**Finding from investigation**: DP4A GEMV was tested but found SLOWER than grouped GEMM due to lack of weight reuse across experts. See `docs/investigations/GEMV_IMPLEMENTATION_PLAN.md`.

**Status**: NOT RECOMMENDED - grouped GEMM wins for MoE even at M=1.

---

## Low Priority

### 7. Fused LayerNorm + Quantization

**Location**: Separate kernels for LayerNorm and quantization

**Issue**: Two kernel launches where one could suffice.

**Potential fix**:
- Fuse LayerNorm output with FP8 quantization
- Write FP8 output directly from LayerNorm kernel

**Impact**: Reduce kernel launch overhead, one less memory round-trip.

**Effort**: Medium

---

### 8. Async Scale Factor Computation

**Location**: `mxfp8_quantize` implementation

**Issue**: Scale factor computation may be on critical path.

**Potential fix**:
- Compute scales asynchronously/overlapped with other work
- Use separate CUDA stream for scale computation

**Impact**: Hide scale computation latency.

**Effort**: Low-Medium

---

## Completed / Not Pursuing

### ~~GEMV for MoE Decode~~

**Status**: Investigated and rejected.

**Finding**: DP4A GEMV is slower than CUTLASS grouped GEMM for MoE because:
- MoE requires 8 expert GEMVs per token
- No weight reuse across experts in GEMV path
- Grouped GEMM batches all experts, reusing weights

**Benchmark results**:
- CUTLASS grouped GEMM (60 layers, TopK=8): 182.88 ms
- DP4A per-expert GEMV (60 layers, TopK=8): 270.38 ms

See `docs/investigations/GEMV_IMPLEMENTATION_PLAN.md` for details.

---

## How to Use This Document

1. When porting features, check if any of these improvements apply
2. Reference the snippet to find current location in code
3. Update commit references when rebasing to new upstream
4. Move items to "Completed" when done or "Not Pursuing" with rationale
