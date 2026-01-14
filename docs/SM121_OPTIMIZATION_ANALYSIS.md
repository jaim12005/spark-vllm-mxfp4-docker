# SM121 Optimization Analysis: vLLM/FlashInfer vs TensorRT-LLM, llama.cpp, and SGLang

## Executive Summary

This document compares our vLLM/FlashInfer SM121 implementation against **llama.cpp**, **TensorRT-LLM**, and **SGLang** implementations for gpt-oss-120b with MXFP4 quantization. The goal is to achieve ≥52 tok/s decode.

**Current Status (Plain Decode):**
| Engine | Backend | pp2048 (t/s) | tg32 (t/s) |
|--------|---------|--------------|------------|
| llama.cpp | MXFP4 | 2449 | **58** |
| SGLang | - | - | **52** |
| vLLM (baseline) | Marlin MoE, BF16 dense | 4341 | 31.6 |
| vLLM + Marlin + FlashInfer | Marlin MoE, BF16 dense | 4341 | 32.6 |
| **vLLM + MXFP4 all** | **CUTLASS MoE + Marlin QKV/O/lm_head** | **4580** | **48.6** ✓ |
| vLLM + MXFP4 all + FP8 KV | CUTLASS MoE + FP8 KV cache | 4520 | 49.0 |
| **vLLM + CUDA graphs** | **CUTLASS MoE + FP8 KV + graphs** | **4540** | **50.5** ✓ |

**Key findings:**
1. **50.5 tok/s achieved** with full MXFP4 + FP8 KV cache + CUDA graphs
2. QKV/O quantization gave +32% decode improvement (29.1 → 38.5 tok/s)
3. Adding lm_head gave +27% additional (38.5 → 48.9 tok/s)
4. FP8 KV cache adds only +0.3 t/s — attention is NOT the bottleneck at batch_size=2
5. **CUDA graphs add +1.5 t/s** (48.96 → 50.50) — kernel launch overhead reduction

**Remaining gap:** ~1.5-7.5 tok/s to match SGLang (52) / llama.cpp (58) — **only 3-15% improvement needed!**

---

# PART 1: PLAIN DECODE OPTIMIZATIONS

This section covers optimizations that apply to **standard autoregressive decode** without speculative decoding.

---

## 1.1 What llama.cpp is doing on SM121 (MXFP4) that we are not

The last-mile decode gap is plausibly explained by **two llama.cpp-specific choices** that reduce both math and memory pressure:

1. **Native FP4×FP4 block-scaled MMA on Blackwell (no FP8 activations)**  
   llama.cpp emits the Blackwell MMA instruction `mma.sync.aligned.kind::mxf4.block_scale...e2m1.e2m1...` (FP4×FP4) in `ggml/src/ggml-cuda/mma.cuh` (~903–921).

2. **Runtime FP4 activation quantization (E2M1) rather than FP8**  
   llama.cpp quantizes activations to FP4 (E2M1) at runtime in `ggml/src/ggml-cuda/quantize.cu` (`quantize_mmq_mxfp4`, ~75–172), using `__nv_fp4x4_e2m1` when CUDA ≥ 12.8.

**Why this matters:** compared to our current FlashInfer CUTLASS path (**FP8 activations × FP4 weights**), llama.cpp's approach:
- **Halves activation bandwidth** (4b vs 8b per element).
- Potentially improves **tensor-core throughput** by using the native FP4×FP4 block-scale MMA.

### Additional llama.cpp decode optimizations worth copying

- **Fused MoE routing kernel (softmax → top-k → write ids/weights)**: `ggml/src/ggml-cuda/topk-moe.cu` explicitly targets 1..512 experts and has a dedicated `case 128` path (~53–167, ~184–225). This is *exactly* gpt-oss-120b's expert count.
- **Dedicated low-batch GEMV kernels + per-batch warp tuning**: `ggml/src/ggml-cuda/mmvq.cu` uses `calc_nwarps()` to increase warps for very small `ncols_dst` (~86–101).
- **Fast 4-bit LUT dequantization**: `ggml/src/ggml-cuda/vecdotq.cuh` uses `__byte_perm` to accelerate 4-bit table lookups (~31–80) and provides an MXFP4×Q8_1 dot path (`vec_dot_mxfp4_q8_1`, ~292–314).

---

## 1.2 SGLang Plain Decode Optimizations

SGLang achieves 52 tok/s on gpt-oss-120b. The following SGLang optimizations apply to **plain decode** (not speculative decode):

### Fused TopK+Softmax Kernel (`moe_topk_softmax_kernels.cu`)

SGLang has a warp-level fused kernel for softmax + top-k + renormalization that applies to power-of-2 expert counts (gpt-oss has 128 experts = power of 2).

```cpp
// SGLang: Fused softmax+topk for 128 experts in single warp-level kernel
topkGatingSoftmax<T, VPT, 128, WARPS_PER_TB, BYTES_PER_LDG><<<...>>>(
    input, finished, output, num_rows, indices, k, start_expert, end_expert,
    renormalize, moe_softcapping, correction_bias);
```

**Key features:**
- Template-specialized for 1, 2, 4, 8, 16, 32, 64, 128, 256 experts
- Warp-level butterfly reduction (no shared memory for small expert counts)
- Fuses softmax → max/argmax → top-k selection → renormalization
- Eliminates separate kernel launches per operation

**Current vLLM/FlashInfer:** Separate softmax and top-k operations

**Expected Impact:** +1-3 tok/s (eliminates per-layer kernel launch overhead × 36 layers)
**Effort:** Medium (port kernel from SGLang's `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu`)

### SGLang feature applicability matrix (Plain Decode)

| SGLang Feature | Applies? | Why |
|----------------|----------|-----|
| `moe_fused_gate` (biased grouped topk) | ❌ | gpt-oss uses softmax, not sigmoid + bias |
| `topkGatingSoftmax` (fused softmax+topk) | ✅ | 128 experts = power of 2 |
| Two-Batch Overlap (TBO) | ❌ | No expert parallel on single GPU |
| Single-Batch Overlap (SBO) | ❌ | No shared experts in gpt-oss |
| DeepGEMM SM partitioning | ❌ | SM90/SM100 only |
| Fused router (CUDA/TC) | ⚠️ | Marginal for bs=1 decode |

---

## 1.3 Feature Matrix: Plain Decode Impact Rankings

**gpt-oss-120b architecture:** hidden_size=2880, 36 layers, 128 experts (4 selected/token), vocab=201088, head_dim=64 (GQA: 64 Q heads / 8 KV heads)

### MoE & Dense Layer Optimizations

| Rank | Feature | TRT-LLM Has? | SGLang Has? | vLLM/FlashInfer Has? | Decode Impact | Effort |
|------|---------|--------------|-------------|----------------------|---------------|--------|
| **1** | **MoE GEMM FP4×FP4 block-scale MMA (Blackwell `mxf4.block_scale`)** | ? | ❌ | ❌ | **+5-15%** | High |
| **2** | MoE GEMM tile+stage tuning | ✅ (many variants) | ⚠️ | ⚠️ (128×128 only) | **+5-10%** | Medium |
| **3** | Activation quant fusion in MoE GEMM | ✅ (SM100 only) | ❌ | ❌ (separate kernel) | **+5-10%** | High |
| **4** | MoE GEMM decode tile configs | ✅ | ⚠️ | ❌ (not for Blackwell) | **+3-5%** | Medium |
| **5** | **Fused MoE routing (softmax→topk)** | ✅ | ✅ (`topkGatingSoftmax`) | ❌ | **+2-4%** | Medium |
| **6** | Low-M CUDA core dispatch for dense | ✅ (M≤4, N≤128k) | ⚠️ | ❌ | **+2-3%** | Medium |
| **7** | Native FP4 dense GEMM (FP8×FP4) | ✅ | ❌ | ❌ (Marlin dequant) | +2-5% | High |
| **8** | Fused QKV projection | ✅ | ✅ | ✅ | +1-2% | Done |
| **9** | Grouped GEMM expert scheduling | ✅ | ✅ | ✅ | +1-2% | Done |

**Legend:** ✅ = Implemented, ⚠️ = Partial/Unverified, ❌ = Missing

---

## 1.4 Attention & KV Cache ✅ TESTED

1. **FP8 KV cache tested — minimal impact at batch_size=2**  
   ✅ **Implemented:** FlashInfer `decode.py` and `prefill.py` updated to allow FP8 E4M3 KV cache with
   attention sinks (commit `c5ad4e14` on `flashinfer/mxfp4_v2`).
   
   **Results (2026-01-14):**
   | KV Cache | pp2048 (t/s) | tg32 (t/s) | tg128 (t/s) |
   |----------|--------------|------------|-------------|
   | BF16 | 4580 | 48.63 | 48.09 |
   | FP8 E4M3 | 4520 | 48.96 | 48.24 |
   | **Delta** | -1.3% | **+0.7%** | **+0.3%** |
   
   **Conclusion:** FP8 KV cache provides minimal improvement (+0.3-0.7%) at batch_size=2.
   This confirms attention is **not the bottleneck** for MoE models at low batch sizes — MoE GEMM dominates.
   Larger batch sizes may show more benefit. The FP8 KV cache is working correctly (verified via
   JIT module URI containing `dtype_kv_e4m3`).

2. **Attention backend selection matters and is coupled with sinks**  
   vLLM has explicit logic to choose TRTLLM attention vs FlashInfer attention for decode based on
   runtime conditions. This should be treated as a first-class tuning surface.

3. **GPT-OSS uses attention sinks; sinks constrain which fast paths are available**  
   FlashInfer's decode wrapper documents that sinks require tensor-core decode and are limited to
   specific backends (FA2 for sinks). ✅ FP8 KV cache now works with sinks after our fix.

---

## 1.5 MoE Routing/Permutation Knobs and SM12x Kernel Configuration

1. **MoE routing tile (`tileTokensDim`) is a separate knob from GEMM tile/stages**  
   TRT-LLM varies routing/permutation tiling (`tileTokensDim`) heavily; this can affect decode
   latency independently of GEMM tile selection. Worth tuning for gpt-oss-120b shapes.

2. **Our SM12x CUTLASS MoE path is currently constrained**  
   The SM12x path is effectively "128×128 only" today, and some features like `min_latency_mode` 
   are explicitly not implemented for Blackwell. Need to expand the validated SM12x tactic/config set.

3. **Consider A/B operand swap as a tactical workaround**  
   SM12x has asymmetric TMA and scale-layout constraints for FP4 operands. Worth investigating if
   we hit dead-ends expanding the tile config space.

4. **Padding/alignment overhead is real at hidden_size=2880**  
   Some MXFP4 backends round hidden/intermediate dims to alignment boundaries. Even ~1–3% overhead
   can be meaningful when the remaining gap is only ~3-15%.

---

## 1.6 Deep Dive: Low-M CUDA Core Dispatch for Dense Layers

### The Problem

During decode (batch_size=1), dense layer operations become matrix-vector multiplies:

```
Input:  [1, hidden_size]     # M=1
Weight: [hidden_size, N]     # K × N
Output: [1, N]               # M=1

Examples for gpt-oss-120b (hidden_size=2880, vocab=201088):
- QKV projection: [1, 2880] × [2880, 5120] = [1, 5120]  (GQA: 64 Q + 8 KV heads)
- O projection:   [1, 4096] × [4096, 2880] = [1, 2880]
- LM head:        [1, 2880] × [2880, 201088] = [1, 201088]

Note: MoE layers (Gate/Up/Down) are handled by FlashInfer grouped GEMM, not dense GEMV.
```

For M=1, Tensor Core GEMM is inefficient—CUDA cores with optimized memory access patterns are faster.

### TRT-LLM's Implementation

From [`gemmPlugin.cpp:394-407`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/plugins/gemmPlugin/gemmPlugin.cpp#L394-L407):

```cpp
// Threshold-based dispatch to CUDA core kernel
if (!isArch90or100 && M <= 4 && N <= 128000 && mUseFp8 && noPadDim && cudaKernelSupportType)
{
    cudaKernelFinished = cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}
```

**Thresholds:**
| Datatype | SM80 | SM90 | SM100+ |
|----------|------|------|--------|
| FP8 | M ≤ 4 | N/A | M ≤ 4 |
| BF16/FP16 | M ≤ 6 | M ≤ 2 | — |

### What vLLM/FlashInfer Should Do

**Option A: Integrate TRT-LLM's cuda_core_gemm (Recommended)**
- Port `cuda_core_gemm` kernels from TRT-LLM
- Add M-threshold dispatch in `LinearBase.forward()`
- Support FP8, BF16, FP16 for dense layers

**Affected layers (gpt-oss-120b has 36 layers, all MoE):**
| Layer | Count per Forward | Shape (M=1) | N Qualifies? | Benefit |
|-------|-------------------|-------------|--------------|---------|
| QKV projection | 36 | [1,2880]×[2880,5120] | ✅ (N=5120) | Medium |
| O projection | 36 | [1,4096]×[4096,2880] | ✅ (N=2880) | Medium |
| LM head | 1 | [1,2880]×[2880,201088] | ❌ (N>128k) | N/A |

**Important:** TRT-LLM's CUDA core path has a hard threshold `N <= 128000`. The LM head
(vocab=201,088) exceeds this and falls back to standard cuBLAS/CUTLASS GEMM.

**Estimated Impact:** +1-2% decode TPS for QKV/O; LM head needs separate solution

---

## 1.7 Deep Dive: Activation Quantization Fusion

### The Situation

We use **MXFP4** (group size 32), not NVFP4 (group size 16). Per FlashInfer code:
- `isWMxfp4AMxfp8Quant()` — MXFP4 weights × MXFP8 activations ← **Our path**

### Current Flow

```
1. BF16 activations from model
2. Call mxfp8_quantize() → FP8 activations + scales  (separate kernel)
3. Call cutlass_fused_moe(FP8 × FP4)                  (GEMM kernel)
```

### Current workaround: identity scales

The SM12x CUTLASS kernels currently use **identity scales (all 1s)** for activations rather than
computing proper block-wise absmax values. This is a known limitation.

### Proper block-scale implementation (LayoutSFA)

A proper implementation requires:

1. **`float_ue8m0_t` scale format**: SM12x block-scaled MMA expects unsigned E8M0 scales
2. **LayoutSFA (Scale-Factor Array)**: CUTLASS's layout for block scales
3. **Fused absmax computation**: Compute per-group maximum during BF16→FP8 conversion

**Effort:** High (requires modifying both quantization kernel and GEMM prologue)
**Impact:** +2-5% accuracy improvement; may also improve throughput

### TRT-LLM Comparison

TRT-LLM's `trtllmGen` MXFP4 MoE kernels:
- `Bf16MxE2m1BlockScaleMoeRunner` — BF16 acts × MXFP4 weights (with prologue fusion)
- `MxE4m3MxE2m1BlockScaleMoeRunner` — MXFP8 acts × MXFP4 weights
- **SM100-only**: "Only SM100f is supported by MXFP4 block scale MOE"

**Estimated Impact:** +5-10% decode TPS
**Effort:** High (requires CUTLASS kernel modification)

---

## 1.8 Deep Dive: MoE GEMM Tile & Pipeline Stage Tuning

### Current State

FlashInfer SM120 MoE uses a **single fixed tile configuration**:
- Tile: 128×128×128
- Stages: ~2-4 (limited tuning)
- No per-problem-size optimization

### TRT-LLM's Approach: Fine-Grained Tile + Stage Selection

TRT-LLM ships many precompiled kernel variants:

| Template | Format | tileN | tileK | numStages | Use Case |
|----------|--------|-------|-------|-----------|----------|
| BatchedGemmMxE2m1E4m3LowLatency | MXFP4×FP8 | 8 | 512 | varies | Decode |
| BatchedGemmMxE2m1MxE4m3Base | MXFP4×MXFP8 | varies | 512 | varies | General |

### The Gap

| Aspect | TRT-LLM SM100 | FlashInfer SM120 |
|--------|---------------|------------------|
| Tile N variants | Multiple (8, etc.) | 128 only |
| Stage variants | Multiple per template | Fixed |
| Per-problem tuning | ✅ Auto-selected | ❌ Single config |

### What FlashInfer Should Do

1. **Add small-N tiles** for decode (M=1-4):
   - 128×8×256 with 6-9 stages
   - 128×16×256 with 5-6 stages
   - 128×32×256 with 5 stages

2. **Profile optimal stages** for SM121

3. **Add tile/stage autotuning** per problem size

4. **Consider split-K** for low-M cases

**Estimated Impact:** +5-10% decode TPS

---

## 1.9 Deep Dive: MoE GEMM Decode-Oriented Tile Configs

FlashInfer has a `min_latency_mode` parameter, but per the [FlashInfer docs](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cutlass_fused_moe.html):

> "Currently, some advanced features like FP8 block scaling and **minimum latency mode are not implemented for Blackwell architecture**."

This means we cannot simply "wire" min_latency_mode — it needs to be **implemented first** for SM120/SM121.

### What min_latency_mode Should Do

- Select smaller tile sizes (e.g., 128×8, 128×16, 128×32) for small M
- Adjust pipeline stages for decode latency
- Prioritize latency over throughput

### What FlashInfer Needs

Implement decode-oriented tile configurations for SM120/SM121 in:
- `flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl`
- Add small-N tile namespaces (currently only 128×128)
- Wire `min_latency_mode` to select these configs

**Estimated Impact:** +3-5% decode TPS
**Effort:** Medium (requires FlashInfer kernel changes)

---

## 1.10 Native FP4 for Dense Layers

### Current State

Dense layers (QKV, O, LM head) use **Marlin backend** which dequantizes FP4→BF16:

```
Current: Load FP4 weights → Dequant to BF16 → BF16 GEMM
```

### Trade-off Analysis

| Approach | Pros | Cons |
|----------|------|------|
| Marlin (dequant) | Proven, stable, works | Extra memory for BF16 weights |
| FP8×FP4 native | Consistent with MoE, lower memory | Requires activation quantization |

### Current Assessment

Marlin is working well (+32% decode improvement for QKV/O). Native FP4 may provide marginal
additional gains but requires more engineering.

**Estimated Impact:** +2-5% decode TPS (if implemented)
**Effort:** High
**Priority:** Low (Marlin already works well)

---

## 1.11 Additional Findings (Plain Decode)

### GEMV vs GEMM for MoE Layers

**Q: Should we use GEMV for MoE GEMM?**

**A: No.** Even for batch=1 decode:
- Each token activates 4 experts (gpt-oss-120b: `experts_per_token=4`)
- With 128 total experts, tokens are distributed sparsely
- Grouped GEMM with low-M tuning is more appropriate than per-expert GEMV
- Memory bandwidth bound regardless (loading expert weights dominates)

### vLLM Fused QKV Status

✅ **Confirmed**: vLLM has fused QKV via `QKVParallelLinear` and `MergedColumnParallelLinear`.
This is **not a gap** compared to TRT-LLM.

### TRT-LLM Dual-Path (FP4 + BF16) Clarification

**Q: Does TRT-LLM use FP4 and BF16 paths simultaneously?**

**A: No.** MXFP4 models use `MxE4m3MxE2m1BlockScaleMoeRunner` (FP8×FP4). BF16 models use separate runners.

---

## 1.12 Plain Decode Action Plan

**Starting point:** 50.5 tok/s (vLLM + MXFP4 all + FP8 KV + CUDA graphs)

### Completed ✅

| Task | Result | Date |
|------|--------|------|
| FP8 KV cache with attention sinks | +0.3 t/s (attention not bottleneck) | 2026-01-14 |
| CUDA graphs on SM121 | +1.5 t/s | 2026-01-14 |

### Next Steps (Priority Order)

| Rank | Task | Expected Gain | Effort | Source |
|------|------|---------------|--------|--------|
| **1** | **Port `topkGatingSoftmax` fused kernel for 128 experts** | +1-3 tok/s | Medium | SGLang |
| **2** | Expand SM12x MoE kernel configs (add 128×8, 128×16 tiles) | +2-4 tok/s | Medium | TRT-LLM |
| **3** | Implement min_latency_mode for Blackwell | +2-3 tok/s | Medium | TRT-LLM |
| **4** | Low-M CUDA core for QKV/O (N<128k) | +1-2 tok/s | Medium | TRT-LLM |
| **5** | Investigate FP4×FP4 `mxf4.block_scale` MoE GEMM | +2-6 tok/s | High | llama.cpp |
| **6** | Activation quant fusion in MoE GEMM | +5-10% | High | TRT-LLM |

**Expected result:** 50.5 → **56-62 tok/s** (exceeds llama.cpp!)

---

# PART 2: SPECULATIVE DECODE OPTIMIZATIONS (Eagle3)

This section covers optimizations specific to **Eagle3 speculative decoding**.

**Note:** These optimizations only apply when running with Eagle3. For plain decode benchmarking,
skip this section.

---

## 2.1 Eagle3 Overview

vLLM supports Eagle3 speculative decoding with the gpt-oss-120b-Eagle3-long-context draft model.
Current acceptance rate is ~42%.

### SGLang Eagle3 Optimizations

#### Native CUDA Tree Verification Kernel (`eagle_utils.cu`)

SGLang has a pure CUDA kernel for Eagle3 tree verification:

```cpp
// SGLang: Single GPU kernel for tree verification (no Python loop overhead)
VerifyTreeGreedy<<<grid, block, 0, stream>>>(
    predicts, accept_index, accept_token_num, candidates,
    retrive_index, retrive_next_token, retrive_next_sibling, target_predict,
    batch_size, num_speculative_tokens, num_draft_tokens);
```

**Key features:**
- Entire tree traversal runs on GPU without CPU sync
- Uses `retrive_next_token` and `retrive_next_sibling` for efficient tree navigation
- Single kernel handles batch verification

**Current vLLM:** Tree verification has Python loop overhead and CPU-GPU synchronization

**Expected Impact:** +1-2 tok/s
**Effort:** Low-Medium (kernel is self-contained in `sgl-kernel/csrc/speculative/eagle_utils.cu`)

#### Bitpacked Tree Masks

SGLang uses bitpacking for Eagle attention masks:

```cpp
// SGLang: 8x memory reduction for tree masks
uint8_t* tree_mask;  // 1 bit per element instead of 1 byte
tree_mask[byte_idx] |= (1 << bit_idx);
```

**Current vLLM:** Full bool masks for tree attention

**Expected Impact:** +0.5-1 tok/s (reduces memory bandwidth for mask loading)
**Effort:** Low (small kernel modification)

#### `fast_topk` Kernel for Draft Sampling

SGLang uses a dedicated `fast_topk` kernel for Eagle draft token selection:

```python
# SGLang spec_utils.py
from sgl_kernel import fast_topk
topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```

**Expected Impact:** +0.5-1 tok/s (faster draft iteration)
**Effort:** Low

---

## 2.2 Feature Matrix: Eagle3 Speculative Decoding

| Rank | Feature | TRT-LLM Has? | SGLang Has? | vLLM Has? | Decode Impact | Effort |
|------|---------|--------------|-------------|-----------|---------------|--------|
| **1** | Tree-based drafting (SpecTreeManager) | ✅ | ✅ | ⚠️ (different impl) | **+10-20%** | High |
| **2** | Greedy draft sampling (argmax) | ✅ | ✅ (`is_all_greedy`) | ❌ (stochastic) | **+5-10%** | Low |
| **3** | Full CUDA graph for drafting loop | ✅ | ✅ | ⚠️ (needs verification) | **+2-5%** | Low |
| **4** | **Native CUDA tree verify kernel** | ⚠️ | ✅ (`VerifyTreeGreedy`) | ❌ (Python) | **+2-4%** | Low-Med |
| **5** | Pre-computed tree masks | ✅ | ✅ | ⚠️ (partial) | +1-2% | Low |
| **6** | **Bitpacked tree masks** | ⚠️ | ✅ (8x compression) | ❌ | +1-2% | Low |
| **7** | Pre-allocated Eagle3 buffers | ✅ | ✅ | ⚠️ (persistent buffers) | +1-2% | Low |
| **8** | **`fast_topk` for draft sampling** | ⚠️ | ✅ (`sgl_kernel`) | ❌ (`torch.topk`) | +1-2% | Low |
| **9** | Quantization-aware draft training | ✅ | ❌ | ❌ (~42% accept) | +5-10% | N/A |

---

## 2.3 CUDA Graphs for Eagle3

### Current State

**vLLM DOES have CUDA graph support for Eagle3** via `EagleCudaGraphManager`:

```python
# vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py
class EagleCudaGraphManager:
    def __init__(self, vllm_config, device):
        if cudagraph_mode == CUDAGraphMode.FULL:
            # For Eagle, only use CUDA graphs for decode
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        self.cudagraph_mode = cudagraph_mode
```

### Key Considerations

- **CUDA graphs are not just "enabled/disabled" — hit-rate matters**
- vLLM's Eagle3 code has a CUDA-graph path *and* a slow/unsafe eager fallback
- Ensure capture sizes cover the real `num_reqs`/`num_tokens` you benchmark

### TRT-LLM's Approach

TRT-LLM captures the entire drafting loop in a single graph:

```python
# tensorrt_llm/_torch/speculative/drafting_loops.py
class LinearDraftingLoopWrapper:
    def capture_graph(self):
        with torch.cuda.graph(self.graph):
            for i in range(self.max_tokens):
                self.tokens[i] = self.draft_one_token(...)
```

---

## 2.4 Greedy Draft Sampling

### The Problem

vLLM's Eagle3 uses **stochastic draft sampling** with temperature, top-k/p, and Gumbel noise:

```python
# vllm/v1/spec_decode/eagle.py
draft_tokens = gumbel_sample(
    logits, 
    self.temperature[:num_reqs], 
    self.seeds[:num_reqs], 
    pos + 1, 
    apply_temperature=True
)
```

TRT-LLM uses **greedy (argmax) draft sampling**:

```python
# tensorrt_llm/_torch/speculative/drafting_loops.py
def sample(self, logits: torch.Tensor) -> torch.Tensor:
    tokens = torch.argmax(logits, dim=-1)
    return tokens
```

### Why Greedy is Faster

1. **No Gumbel noise generation** — Avoids RNG overhead
2. **No temperature scaling** — Fewer FLOPs
3. **Simple argmax** — Single reduction operation
4. **Deterministic** — Better for CUDA graph capture

### What vLLM Should Do

Add a `greedy_draft_sampling` config option:

```python
if config.greedy_draft_sampling or sampling_params.temperature == 0:
    draft_tokens = torch.argmax(logits, dim=-1)
else:
    draft_tokens = gumbel_sample(logits, ...)
```

**Estimated Impact:** +2-3 tok/s (reduces drafting overhead)

---

## 2.5 Tree-Based Drafting

### Current State

vLLM **does have tree structure support** via `speculative_token_tree` config, but the implementation
differs from TRT-LLM's `SpecTreeManager`.

### TRT-LLM's Implementation

```python
# tensorrt_llm/_torch/speculative/spec_tree_manager.py
class SpecTreeManager:
    def __init__(self, tree_config):
        self.eagle_paths = self._compute_paths()
        self.spec_dec_mask_matrix = self._compute_mask_matrix()
        self.spec_dec_packed_mask = self._pack_masks()
```

### Why Tree Drafting is Better

1. **Higher acceptance probability** — Multiple paths increase chance of match
2. **Better utilization** — Verifier processes tree in single forward pass
3. **Parallel branch generation** — All branches at same depth computed together

### Key Components Missing in vLLM

| Component | Purpose | TRT-LLM Location |
|-----------|---------|------------------|
| `eagle_paths` | Pre-computed token paths through tree | `SpecTreeManager` |
| `spec_dec_mask_matrix` | Attention mask for tree structure | `SpecTreeManager` |
| `spec_dec_packed_mask` | Packed masks for efficient memory | `SpecTreeManager` |
| `TreeDraftingLoopWrapper` | CUDA-graphable tree generation | `drafting_loops.py` |

**Estimated Impact:** +10-20% decode TPS (higher acceptance, better parallelism)

---

## 2.6 Pre-Allocated Eagle3 Buffers

### Current State

vLLM **does have persistent buffer support** for CUDA graph capture. Verify:
1. Are these buffers being used on SM121?
2. Is the allocation pattern optimal?

### TRT-LLM's approach

```python
class LinearDraftingLoopWrapper:
    def __init__(self):
        self.hidden_states = torch.empty(max_tokens, hidden_size)
        self.logits = torch.empty(max_tokens, vocab_size)
        self.tokens = torch.empty(max_tokens, dtype=torch.long)
```

### Why Pre-Allocation Matters

- Reduces CUDA allocator pressure
- Enables CUDA graph capture (graphs require fixed memory)
- Avoids synchronization on allocation

**Estimated Impact:** +0.5-1 tok/s (likely already partially implemented)

---

## 2.7 Why Our Eagle3 Acceptance Rate Is Low (~42%)

Factors that affect acceptance rate:

1. **Draft/verifier numerical mismatch**: Draft model trained on BF16, verifier uses MXFP4
2. **Tree structure**: Different tree depths/widths affect acceptance probability
3. **Sampling alignment**: Draft and endpoint sampling parameters should match

**Investigation needed:** Profile acceptance rate by:
- Quantization mismatch (draft trained on BF16, verifier uses MXFP4)
- Sampling parameter mismatch
- Model-specific issues with gpt-oss-120b + Eagle3

---

## 2.8 Eagle3 Speculative Decode Action Plan

| Rank | Task | Expected Gain | Effort | Source |
|------|------|---------------|--------|--------|
| **1** | Add greedy draft sampling option | +2-3 tok/s | Low | TRT-LLM |
| **2** | Port `VerifyTreeGreedy` CUDA kernel | +1-2 tok/s | Low-Med | SGLang |
| **3** | Use bitpacked tree masks | +0.5-1 tok/s | Low | SGLang |
| **4** | Verify Eagle3 buffer allocation | +0.5-1 tok/s | Low | TRT-LLM |
| **5** | Optimize tree drafting (SpecTreeManager) | +3-5 tok/s | Medium | TRT-LLM |
| **6** | Pre-computed tree masks (eagle_paths) | +1-2 tok/s | Medium | TRT-LLM |

**Expected result with Eagle3:** 50.5 → **60-70 tok/s** (with speculative decode)

---

# APPENDIX: Code References

## TRT-LLM Low-M Dispatch

Source: [`gemmPlugin.cpp:394-407`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/plugins/gemmPlugin/gemmPlugin.cpp#L394-L407)

```cpp
if (!isArch90or100 && M <= 4 && N <= 128000 && mUseFp8 && noPadDim && cudaKernelSupportType)
{
    cudaKernelFinished = cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}
```

## vLLM Eagle3 CUDA Graph Support

```python
# vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py
class EagleCudaGraphManager:
    def capture_graph(self, num_tokens, generate_fn, ...):
        capture_graphs(num_tokens, generate_fn, ...)
```

## FlashInfer SM120 MoE GEMM

```cpp
// flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl
// Uses: ElementInputA = float_e4m3_t (FP8), ElementInputB = float_e2m1_t (FP4)
```

## SGLang Fused TopK+Softmax Kernel

Source: `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu`

```cpp
template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(
    const T* input, const bool* finished, float* output, const int num_rows,
    int* indices, const int k, const int start_expert, const int end_expert,
    const bool renormalize, const float moe_softcapping, const float* correction_bias);
// Template specializations for 1, 2, 4, 8, 16, 32, 64, 128, 256 experts
```

## SGLang Tree Verification Kernel

Source: `sgl-kernel/csrc/speculative/eagle_utils.cu`

```cpp
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts, IdType* accept_index, IdType* accept_token_num,
    IdType2* candidates, IdType2* retrive_index, IdType2* retrive_next_token,
    IdType2* retrive_next_sibling, IdType2* target_predict,
    uint32_t batch_size, uint32_t num_speculative_tokens, uint32_t num_draft_tokens);
```

## SGLang Bitpacked Tree Masks

Source: `sgl-kernel/csrc/speculative/eagle_utils.cu`

```cpp
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list, int64_t* selected_index, int64_t* verified_seq_len,
    uint8_t* tree_mask,  // Packed: 1 bit per element instead of 1 byte
    int64_t* positions, int64_t* retrive_index, ...);
```

---

# Summary

## Plain Decode

**Current best:** 50.5 tok/s (vLLM + MXFP4 all + FP8 KV + CUDA graphs) — **only 3% below SGLang!**

**Already implemented:**
- ✅ MXFP4 MoE with CUTLASS FP8×FP4
- ✅ MXFP4 QKV/O with Marlin (+32% decode improvement)
- ✅ MXFP4 lm_head with Marlin (+27% additional improvement)
- ✅ Fused QKV projection
- ✅ FP8 E4M3 KV cache with attention sinks (+0.3 t/s)
- ✅ CUDA graphs enabled (+1.5 t/s)

**Next priority (Plain Decode):**
1. **Port `topkGatingSoftmax`** — Fuse softmax+topk for 128 experts (+1-3 tok/s)
2. **Expand SM12x MoE kernel configs** — Add small-N tiles (+2-4 tok/s)
3. **Implement min_latency_mode** — Decode-oriented tile selection (+2-3 tok/s)

**Projected trajectory (Plain Decode):**
- Current: **50.5 tok/s**
- After MoE routing fusion: **52-53 tok/s** (matches SGLang!)
- After MoE tile tuning: **56-62 tok/s** (exceeds llama.cpp!)

## Speculative Decode

**Next priority (Eagle3):**
1. Greedy draft sampling
2. Port VerifyTreeGreedy CUDA kernel
3. Bitpacked tree masks

**Projected trajectory (Eagle3):**
- After Eagle3 optimizations: **60-70 tok/s**

---

## Key Learnings

1. **Attention is NOT the bottleneck** at batch_size=2 for MoE models (FP8 KV gave only +0.3 t/s)
2. **CUDA graphs work on SM121** and give +1.5 t/s (kernel launch overhead reduction)
3. **MoE GEMM dominates decode time** (~61%). Focus optimization on MoE kernel tuning.
4. **Plain decode and spec decode have separate optimization paths** — focus on one at a time.
