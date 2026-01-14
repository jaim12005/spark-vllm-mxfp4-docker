# SM121 Optimization Analysis: vLLM/FlashInfer vs TensorRT-LLM, llama.cpp, and SGLang

## Executive Summary

This document compares our vLLM/FlashInfer SM121 implementation against **llama.cpp**, **TensorRT-LLM**, and **SGLang** implementations for gpt-oss-120b with MXFP4 quantization. The goal is to identify high-impact optimizations to achieve ≥52 tok/s decode.

**Current Status:**
| Engine | Backend | pp2048 (t/s) | tg32 (t/s) |
|--------|---------|--------------|------------|
| llama.cpp | MXFP4 | 2449 | **58** |
| SGLang | - | - | **52** |
| vLLM (baseline) | Marlin MoE, BF16 dense | 4341 | 31.6 |
| vLLM + Marlin + FlashInfer | Marlin MoE, BF16 dense | 4341 | 32.6 |
| **vLLM + MXFP4 all** | **CUTLASS MoE + Marlin QKV/O/lm_head** | **4580** | **48.6** ✓ |
| vLLM + MXFP4 all + FP8 KV | CUTLASS MoE + FP8 KV cache | 4520 | 49.0 |

**Key findings:**
1. **49.0 tok/s achieved** with full MXFP4 quantization + FP8 KV cache
2. QKV/O quantization gave +32% decode improvement (29.1 → 38.5 tok/s)
3. Adding lm_head gave +27% additional (38.5 → 48.9 tok/s)
4. **FP8 KV cache adds only +0.3 t/s** — attention is NOT the bottleneck at batch_size=2

**Remaining gap:** ~3-9 tok/s to match SGLang (52) / llama.cpp (58) — **only 6-16% improvement needed!**

---

## New: What llama.cpp is doing on SM121 (MXFP4) that we are not

The last-mile decode gap is plausibly explained by **two llama.cpp-specific choices** that reduce both math and memory pressure:

1. **Native FP4×FP4 block-scaled MMA on Blackwell (no FP8 activations)**  
   llama.cpp emits the Blackwell MMA instruction `mma.sync.aligned.kind::mxf4.block_scale...e2m1.e2m1...` (FP4×FP4) in `ggml/src/ggml-cuda/mma.cuh` (~903–921).

2. **Runtime FP4 activation quantization (E2M1) rather than FP8**  
   llama.cpp quantizes activations to FP4 (E2M1) at runtime in `ggml/src/ggml-cuda/quantize.cu` (`quantize_mmq_mxfp4`, ~75–172), using `__nv_fp4x4_e2m1` when CUDA ≥ 12.8.

**Why this matters:** compared to our current FlashInfer CUTLASS path (**FP8 activations × FP4 weights**), llama.cpp’s approach:
- **Halves activation bandwidth** (4b vs 8b per element).
- Potentially improves **tensor-core throughput** by using the native FP4×FP4 block-scale MMA.

### Additional llama.cpp decode optimizations worth copying

- **Fused MoE routing kernel (softmax → top-k → write ids/weights)**: `ggml/src/ggml-cuda/topk-moe.cu` explicitly targets 1..512 experts and has a dedicated `case 128` path (~53–167, ~184–225). This is *exactly* gpt-oss-120b’s expert count.
- **Dedicated low-batch GEMV kernels + per-batch warp tuning**: `ggml/src/ggml-cuda/mmvq.cu` uses `calc_nwarps()` to increase warps for very small `ncols_dst` (~86–101).
- **Fast 4-bit LUT dequantization**: `ggml/src/ggml-cuda/vecdotq.cuh` uses `__byte_perm` to accelerate 4-bit table lookups (~31–80) and provides an MXFP4×Q8_1 dot path (`vec_dot_mxfp4_q8_1`, ~292–314).

---

## New: What SGLang is doing that we could use (gpt-oss-120b + SM121 specific)

SGLang achieves 52 tok/s on gpt-oss-120b. After reviewing `~/projects/sglang-spark`, we identified
optimizations that apply to our specific configuration (128 experts, standard softmax routing, single
SM121 GPU with no expert parallel).

**Note:** Many SGLang optimizations target DeepSeek V3 (256 experts, biased grouped top-k, shared experts)
or expert-parallel scenarios. These are **not applicable** to gpt-oss-120b on SM121:
- ❌ `moe_fused_gate` — requires biased sigmoid + grouped top-k (DeepSeek-style), not softmax
- ❌ Two-Batch Overlap (TBO) — requires expert parallel, we run TP=1
- ❌ Single-Batch Overlap (SBO) — requires shared experts, gpt-oss has none
- ❌ DeepGEMM SM partitioning — SM90/SM100 only, not SM121

### Applicable SGLang optimizations for gpt-oss-120b + SM121

#### 1. Fused TopK+Softmax Kernel (`moe_topk_softmax_kernels.cu`)

SGLang has a warp-level fused kernel for softmax + top-k + renormalization that applies to power-of-2
expert counts (gpt-oss has 128 experts = power of 2).

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

#### 2. Native CUDA Tree Verification Kernel (`eagle_utils.cu`)

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

#### 3. Bitpacked Tree Masks

SGLang uses bitpacking for Eagle attention masks:

```cpp
// SGLang: 8x memory reduction for tree masks
// Instead of bool* tree_mask (1 byte per element)
uint8_t* tree_mask;  // 1 bit per element
tree_mask[byte_idx] |= (1 << bit_idx);
```

**Current vLLM:** Full bool masks for tree attention

**Expected Impact:** +0.5-1 tok/s (reduces memory bandwidth for mask loading)
**Effort:** Low (small kernel modification)

#### 4. `fast_topk` Kernel for Draft Sampling

SGLang uses a dedicated `fast_topk` kernel for Eagle draft token selection:

```python
# SGLang spec_utils.py
from sgl_kernel import fast_topk
topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
```

This is more efficient than `torch.topk` for the specific shapes in speculative decoding
(small batch × vocab-sized tensors).

**Expected Impact:** +0.5-1 tok/s (faster draft iteration)
**Effort:** Low (can use `torch.topk` with compile or port kernel)

### SGLang feature applicability matrix for gpt-oss-120b + SM121

| SGLang Feature | Applies? | Why |
|----------------|----------|-----|
| `moe_fused_gate` (biased grouped topk) | ❌ | gpt-oss uses softmax, not sigmoid + bias |
| `topkGatingSoftmax` (fused softmax+topk) | ✅ | 128 experts = power of 2 |
| Two-Batch Overlap (TBO) | ❌ | No expert parallel on single GPU |
| Single-Batch Overlap (SBO) | ❌ | No shared experts in gpt-oss |
| `VerifyTreeGreedy` CUDA kernel | ✅ | Eagle3 spec decode is used |
| Bitpacked tree masks | ✅ | Eagle3 tree attention |
| `fast_topk` kernel | ✅ | Draft token selection |
| DeepGEMM SM partitioning | ❌ | SM90/SM100 only |
| Fused router (CUDA/TC) | ⚠️ | Marginal for bs=1 decode |

---

## Feature Matrix: Impact Rankings

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

### Eagle3 Speculative Decoding Optimizations

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

**Ranking rationale:** gpt-oss-120b is MoE-dominated (128 experts). Dense layers (QKV/O/LM head) have
smaller hidden_size=2880, so GEMV optimization has less impact than MoE tile/stage tuning.

**Legend:** ✅ = Implemented, ⚠️ = Partial/Unverified, ❌ = Missing, N/A = Requires model retraining

**Source notes:**
- **TRT-LLM**: SM100/SM90 focused, some features need porting to SM121
- **SGLang**: Applicable features are primarily Eagle3-related and fused routing for 128 experts
- **llama.cpp**: Native FP4×FP4 MMA path is the key differentiator for SM121

---

## New Findings from Code Review (Overlooked in the Original Draft)

This section summarizes additional optimization levers discovered by reviewing our local repos:
`~/projects/flashinfer`, `~/projects/vllm`, and `~/projects/TensorRT-LLM`. These are *in addition to*
the MoE and Eagle-focused items below.

### A. Attention & KV Cache ✅ TESTED

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

2. **Attention backend selection matters and is coupled with sinks/spec-decode**  
   vLLM has explicit logic to choose TRTLLM attention vs FlashInfer attention for decode based on
   runtime conditions (e.g., speculative decoding, token counts, and cache dtype). This should be
   treated as a first-class tuning surface (not assumed "done" once QKV/O weights are quantized).

3. **GPT-OSS uses attention sinks; sinks constrain which fast paths are available**  
   FlashInfer's decode wrapper documents that sinks require tensor-core decode and are limited to
   specific backends (FA2 for sinks). ✅ FP8 KV cache now works with sinks after our fix.

### B. Eagle3 Speculative Decode: Avoid the Slow Fallback Path

4. **CUDA graphs are not just “enabled/disabled” — hit-rate matters**  
   vLLM’s Eagle3 code has a CUDA-graph path *and* a slow/unsafe eager fallback when no captured
   graph matches the live batch shape. The optimization is therefore:
   - Enable graphs (remove `--enforce-eager`)
   - Ensure capture sizes cover the real `num_reqs`/`num_tokens` you benchmark so you don’t fall back

5. **Greedy draft sampling (argmax) remains a high-signal, low-effort win**  
   vLLM Eagle currently uses Gumbel sampling (`gumbel_sample`) for drafting; TRT-LLM uses argmax.
   Adding an argmax option is still a good Phase 1 candidate.

### C. MoE: Routing/Permutation Knobs and SM12x Kernel Configuration Limits

6. **MoE routing tile (`tileTokensDim`) is a separate knob from GEMM tile/stages**  
   TRT-LLM varies routing/permutation tiling (`tileTokensDim`) heavily; this can affect decode
   latency independently of GEMM tile selection. It’s worth tuning for gpt-oss-120b shapes.

7. **Our SM12x CUTLASS MoE path is currently constrained beyond what "autotune" suggests**  
   Even though the surrounding infrastructure can enumerate many tactics, our current SM12x launcher
   support is deliberately limited (e.g., the SM12x path is effectively "128×128 only" today, and
   some features like `min_latency_mode` are explicitly not implemented for Blackwell). The doc's
   MoE tuning plan should explicitly include: "expand the validated SM12x tactic/config set" as a prerequisite.

8. **Consider A/B operand swap as a tactical workaround**  
   SM12x has asymmetric TMA and scale-layout constraints for FP4 operands. If certain tile/stage
   configurations fail due to these constraints, an A/B operand swap variant (swapping which operand
   is FP4 vs FP8) may bypass the limitation. This is only viable if it doesn't require runtime
   repacking of FP4 weights (weights are pre-packed at load time). Worth investigating if we hit
   dead-ends expanding the tile config space.

9. **Padding/alignment overhead is real at hidden_size=2880**  
   Some MXFP4 backends round hidden/intermediate dims to alignment boundaries. Even a ~1–3% overhead
   can be meaningful when the remaining gap is only ~6–16%.

## Deep Dive: Top 4 Optimizations

### 1. Low-M CUDA Core Dispatch for Dense Layers (Highest Impact)

#### The Problem

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

#### TRT-LLM's Implementation

From [`gemmPlugin.cpp:394-407`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/plugins/gemmPlugin/gemmPlugin.cpp#L394-L407):

```cpp
// Threshold-based dispatch to CUDA core kernel
if (!isArch90or100 && M <= 4 && N <= 128000 && mUseFp8 && noPadDim && cudaKernelSupportType)
{
    cudaKernelFinished = cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}
else if (!isArch90or100 && ((mArch < 90 && M <= 6) || (isArch90or100 && M <= 2)) && N <= 128000 && !mUseFp8 ...)
{
    cudaKernelFinished = cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}
```

**Thresholds (from code above):**
| Datatype | SM80 | SM90 | SM100+ |
|----------|------|------|--------|
| FP8 | M ≤ 4 | N/A | M ≤ 4 |
| BF16/FP16 | M ≤ 6 | M ≤ 2 | — |

Note: SM120+ thresholds not explicitly shown in this code path.

#### What vLLM/FlashInfer Should Do

**Option A: Integrate TRT-LLM's cuda_core_gemm (Recommended)**
- Port `cuda_core_gemm` kernels from TRT-LLM
- Add M-threshold dispatch in `LinearBase.forward()`
- Support FP8, BF16, FP16 for dense layers

**Option B: cuBLASLt GEMV**
- Use `cublasLtMatmul` with GEMV algo hints for M=1
- Less work but potentially less optimal

**Affected layers (gpt-oss-120b has 36 layers, all MoE):**
| Layer | Count per Forward | Shape (M=1) | N Qualifies? | Benefit |
|-------|-------------------|-------------|--------------|---------|
| QKV projection | 36 | [1,2880]×[2880,5120] | ✅ (N=5120) | Medium |
| O projection | 36 | [1,4096]×[4096,2880] | ✅ (N=2880) | Medium |
| LM head | 1 | [1,2880]×[2880,201088] | ❌ (N>128k) | N/A |

**Important:** TRT-LLM's CUDA core path has a hard threshold `N <= 128000`. The LM head
(vocab=201,088) exceeds this and falls back to standard cuBLAS/CUTLASS GEMM.

Note: MoE expert layers (128 experts, 4 selected per token) use grouped GEMM via FlashInfer,
not dense GEMV. The low-M dispatch applies to attention projections only (not LM head).

**LM head optimization (alternative approach):** For very large N (vocab=201k), consider:
- Streaming/tiled GEMV that processes N in chunks of ~64k
- cuBLASLt with GEMV algorithm hints
- TopK-fused LM head that avoids materializing full [1, 201088] logits

**Estimated Impact:** +3-5% decode TPS for QKV/O; LM head needs separate solution

---

### 2. Activation Quantization Fusion

#### The Situation

We use **MXFP4** (group size 32), not NVFP4 (group size 16). Per FlashInfer code:
- `isWMxfp4AMxfp8Quant()` — MXFP4 weights × MXFP8 activations ← **Our path**
- `isNvfp4Quant()` — NVFP4 format (different, not what we use)

The [FlashInfer API docs](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cutlass_fused_moe.html)
mention "For NVFP4, both quantized and non-quantized inputs are supported" — but this applies to
NVFP4, **not our MXFP4 path**.

For our MXFP4 path (FP8 activations), the docs state: "For FP8, the input must be quantized."

#### Current Flow

```
1. BF16 activations from model
2. Call mxfp8_quantize() → FP8 activations + scales  (separate kernel)
3. Call cutlass_fused_moe(FP8 × FP4)                  (GEMM kernel)
```

#### Current workaround: identity scales

The SM12x CUTLASS kernels currently use **identity scales (all 1s)** for activations rather than
computing proper block-wise absmax values. This is a known limitation documented in FlashInfer's
SM120 launcher code. The quantization step (`mxfp8_quantize()`) computes per-group scales, but the
GEMM kernel ignores them and assumes unit scaling.

**Why this matters:**
- Numerical accuracy is degraded (no dynamic range adjustment per block)
- We're paying the cost of a separate quantization kernel but not using its output fully
- This may partially explain accuracy gaps vs other implementations

#### Proper block-scale implementation (LayoutSFA)

A proper implementation requires:

1. **`float_ue8m0_t` scale format**: SM12x block-scaled MMA expects unsigned E8M0 scales
   (8-bit exponent, no mantissa) in a specific memory layout
2. **LayoutSFA (Scale-Factor Array)**: CUTLASS's layout for block scales that pairs with the
   FP4/FP8 data tiles. Each 32-element group needs its corresponding scale factor.
3. **Fused absmax computation**: Compute per-group maximum absolute value during BF16→FP8
   conversion to derive the scale, then pack it into LayoutSFA format

```
Proper flow:
1. BF16 activations from model
2. Fused BF16→FP8 with absmax:
   - Compute max(|x|) per 32-element group
   - Derive scale = max / FP8_MAX
   - Quantize: x_fp8 = x / scale
   - Pack scale into LayoutSFA (float_ue8m0_t)
3. GEMM kernel reads scales from LayoutSFA, applies during MMA
```

**Effort:** High (requires modifying both quantization kernel and GEMM prologue)
**Impact:** +2-5% accuracy improvement; may also improve throughput by enabling better hardware paths

#### Finalize fusion opportunity

If we currently run a separate finalize/epilogue kernel after the MoE GEMM (e.g., for dequantization,
bias addition, or activation function), this could be fused into the GEMM epilogue to reduce kernel
launch count. Each layer has 36 MoE blocks × 2 GEMMs (FC1, FC2), so even a small per-launch overhead
compounds.

**Investigation needed:** Profile the MoE forward pass to identify if there are separate post-GEMM
kernels that could be fused. Look for:
- Separate dequant/rescale kernels after GEMM
- Activation functions (SiLU/Swiglu) not fused into FC1 epilogue
- Bias addition as separate kernel

#### TRT-LLM Comparison

TRT-LLM's `trtllmGen` MXFP4 MoE kernels (from [`mxFp4BlockScaleMoe.cpp`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp)):
- `Bf16MxE2m1BlockScaleMoeRunner` — BF16 acts × MXFP4 weights (with prologue fusion)
- `MxE4m3MxE2m1BlockScaleMoeRunner` — MXFP8 acts × MXFP4 weights
- **SM100-only**: Line 55 states "Only SM100f is supported by MXFP4 block scale MOE"

#### What FlashInfer Could Do

Add CUTLASS custom prologue for BF16→FP8 fusion in SM120 path:
- Modify `moe_gemm_sm120_mixed_input_launcher.inl`
- Add `PrologueTransform` that converts BF16→FP8 during tile load

**Estimated Impact:** +5-10% decode TPS
**Effort:** High (requires CUTLASS kernel modification)

---

### 3. CUDA Graph for Eagle3 Drafting Loops

#### Current State

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

#### The Question: Is It Actually Enabled on SM121?

Our benchmarks use `--enforce-eager`, which **disables CUDA graphs**. The question is:
1. Do CUDA graphs work on SM121 without `--enforce-eager`?
2. Are there SM121-specific issues (illegal instruction, etc.) that forced us to use eager mode?
3. If graphs work, do we actually *hit* the captured graphs during benchmarks, or do we fall back to eager due to shape mismatch?

#### TRT-LLM's Approach (for comparison)

TRT-LLM captures the entire drafting loop in a single graph:

```python
# tensorrt_llm/_torch/speculative/drafting_loops.py
class LinearDraftingLoopWrapper:
    def capture_graph(self):
        with torch.cuda.graph(self.graph):
            for i in range(self.max_tokens):
                self.tokens[i] = self.draft_one_token(...)
```

#### Action Items

1. **Verify CUDA graph status on SM121** — Remove `--enforce-eager` and test
2. **If graphs fail on SM121** — Debug the specific failure (illegal instruction, etc.)
3. **If graphs work** — Benchmark with graphs enabled *and* verify graph hit-rate (avoid falling back to eager due to capture-size mismatch)

**Affected code:** `vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py`

**Estimated Impact:** +2-5% decode TPS (if not already enabled)

---

### 4. MoE GEMM Tile & Pipeline Stage Tuning

#### Current State

FlashInfer SM120 MoE uses a **single fixed tile configuration**:
- Tile: 128×128×128
- Stages: ~2-4 (limited tuning)
- No per-problem-size optimization

#### TRT-LLM's Approach: Fine-Grained Tile + Stage Selection

TRT-LLM ships many precompiled kernel variants with different configurations.
From [`trtllmGen_bmm_export/config.json`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/config.json):

| Template | Format | tileN | tileK | numStages | Use Case |
|----------|--------|-------|-------|-----------|----------|
| BatchedGemmMxE2m1E4m3LowLatency | MXFP4×FP8 | 8 | 512 | varies | Decode |
| BatchedGemmMxE2m1MxE4m3Base | MXFP4×MXFP8 | varies | 512 | varies | General |
| Bmm_Bfloat16_MxE2m1Bfloat16 | MXFP4×BF16 | 16-32 | 128-256 | 3-5 | BF16 path |

Note: `MxE2m1` = MXFP4 format (E2M1 = 2 exponent, 1 mantissa, group size 32)

**What are stages?** Software pipelining stages that overlap memory loads with computation:
- More stages = more tiles "in flight" = better latency hiding
- Tradeoff: more stages = more shared memory usage
- Optimal stages depend on tile size and memory bandwidth

```
Stages = 3:
  [Load tile 2] [Compute tile 0]  ← Loading while computing
  [Load tile 3] [Compute tile 1]
  [Load tile 4] [Compute tile 2]
```

#### The Gap

| Aspect | TRT-LLM SM100 | FlashInfer SM120 |
|--------|---------------|------------------|
| Tile N variants | Multiple (8, etc. per config.json) | 128 only |
| Stage variants | Multiple per template | Fixed |
| Per-problem tuning | ✅ Auto-selected | ❌ Single config |

#### What FlashInfer Should Do

1. **Add small-N tiles** for decode (M=1-4):
   - 128×8×256 with 6-9 stages
   - 128×16×256 with 5-6 stages
   - 128×32×256 with 5 stages

2. **Profile optimal stages** for SM121:
   - SM121 has different memory bandwidth than SM100
   - May need different stage counts

3. **Add tile/stage autotuning** per problem size:
   - Expert FC1 GEMM: M=1-32, N=2048, K=7168
   - Expert FC2 GEMM: M=1-32, N=7168, K=2048

4. **Consider split-K** for low-M cases:
   - `clusterDimZ > 1` to parallelize K reduction
   - TRT-LLM uses split-K=2,3,4 for small tiles

5. **Configure the CUTLASS autotuner (gated on config expansion)**  
   FlashInfer has autotuning infrastructure that can benchmark multiple kernel configurations and
   select the best one per problem size. However, this is currently ineffective for SM12x because:
   - The SM12x launcher only supports a single 128×128 tile config
   - Autotuning can't pick from options that don't exist
   
   **Dependency order:**
   1. First: Expand the validated SM12x config set (add 128×8, 128×16, 128×32 tiles, vary stages)
   2. Then: Enable autotuning/selection logic to pick optimal config per problem shape
   3. Finally: Cache winning configs for gpt-oss-120b's specific M/N/K combinations
   
   Without step 1, autotuner work is wasted. This is why tile expansion is Phase 2 priority.

**Estimated Impact:** +5-10% decode TPS

---

### 5. Greedy Draft Sampling

#### The Problem

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
    # TODO: inject the sampler here so we can support non-greedy
    tokens = torch.argmax(logits, dim=-1)
    return tokens
```

#### Why Greedy is Faster

1. **No Gumbel noise generation** — Avoids RNG overhead
2. **No temperature scaling** — Fewer FLOPs
3. **Simple argmax** — Single reduction operation
4. **Deterministic** — Better for CUDA graph capture

#### Trade-off

- Greedy drafting works best when endpoint also uses greedy sampling
- For stochastic endpoints, draft-endpoint mismatch may reduce acceptance
- TRT-LLM's approach: prioritize speed, accept lower acceptance for stochastic cases

#### What vLLM Should Do

Add a `greedy_draft_sampling` config option:

```python
# Proposed change
if config.greedy_draft_sampling or sampling_params.temperature == 0:
    draft_tokens = torch.argmax(logits, dim=-1)
else:
    draft_tokens = gumbel_sample(logits, ...)
```

**Estimated Impact:** +5-10% decode TPS (reduces drafting overhead)

---

### 6. Tree-Based Drafting Comparison

#### Current State

vLLM **does have tree structure support** via `speculative_token_tree` config, but the implementation
differs from TRT-LLM's `SpecTreeManager`. The key question is whether vLLM's tree drafting is:
1. Fully functional on SM121
2. As optimized as TRT-LLM's approach

TRT-LLM uses **tree-based drafting** with pre-computed structures:

```
Position 0: A
           / \
Position 1: B   C
           /|   |\
Position 2: D E F G
```

#### TRT-LLM's Implementation

```python
# tensorrt_llm/_torch/speculative/spec_tree_manager.py
class SpecTreeManager:
    def __init__(self, tree_config):
        # Pre-compute all tree structures at init
        self.eagle_paths = self._compute_paths()
        self.spec_dec_mask_matrix = self._compute_mask_matrix()
        self.spec_dec_packed_mask = self._pack_masks()
        self.tokens_gather_idx_for_drafter_model = self._compute_gather_indices()
```

```python
# tensorrt_llm/_torch/speculative/drafting_loops.py
class TreeDraftingLoopWrapper:
    def capture_graph(self):
        with torch.cuda.graph(self.graph):
            for depth in range(self.max_depth):
                # Generate all branches at this depth in parallel
                self.tokens[depth] = self.draft_tree_level(...)
```

#### Why Tree Drafting is Better

1. **Higher acceptance probability** — Multiple paths increase chance of match
2. **Better utilization** — Verifier processes tree in single forward pass
3. **Parallel branch generation** — All branches at same depth computed together

#### Key Components Missing in vLLM

| Component | Purpose | TRT-LLM Location |
|-----------|---------|------------------|
| `eagle_paths` | Pre-computed token paths through tree | `SpecTreeManager` |
| `spec_dec_mask_matrix` | Attention mask for tree structure | `SpecTreeManager` |
| `spec_dec_packed_mask` | Packed masks for efficient memory | `SpecTreeManager` |
| `TreeDraftingLoopWrapper` | CUDA-graphable tree generation | `drafting_loops.py` |

#### What vLLM Should Do

1. **Add SpecTreeManager equivalent** — Pre-compute tree structures at init
2. **Implement TreeDraftingLoopWrapper** — Parallel tree generation
3. **Modify verification** — Handle tree-structured draft tokens

**Estimated Impact:** +10-20% decode TPS (higher acceptance, better parallelism)

---

### 7. MoE GEMM Decode-Oriented Tile Configs

#### The Problem

FlashInfer has a `min_latency_mode` parameter, but per the [FlashInfer docs](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cutlass_fused_moe.html):

> "Currently, some advanced features like FP8 block scaling and **minimum latency mode are not implemented for Blackwell architecture**."

This means we cannot simply "wire" min_latency_mode — it needs to be **implemented first** for SM120/SM121.

#### What min_latency_mode Should Do

- Select smaller tile sizes (e.g., 128×8, 128×16, 128×32) for small M
- Adjust pipeline stages for decode latency
- Prioritize latency over throughput

#### What FlashInfer Needs

Implement decode-oriented tile configurations for SM120/SM121 in:
- `flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl`
- Add small-N tile namespaces (currently only 128×128)
- Wire `min_latency_mode` to select these configs

**Note on autotuning:** Once multiple tile configs exist, the autotuner (§4.5) can benchmark and
select the best one. But the autotuner is useless until these decode-oriented tiles are added.
The sequence is: add tiles → enable autotuner → cache optimal configs for gpt-oss-120b shapes.

**Estimated Impact:** +3-5% decode TPS
**Effort:** Medium (requires FlashInfer kernel changes, not just vLLM wiring)

---

### 8. Pre-Allocated Eagle3 Buffers

#### Current State

vLLM **does have persistent buffer support** for CUDA graph capture, but we should verify:
1. Are these buffers being used on SM121 (especially with `--enforce-eager`)?
2. Is the allocation pattern optimal?

TRT-LLM's approach for comparison:

```python
# TRT-LLM: Pre-allocate once
class LinearDraftingLoopWrapper:
    def __init__(self):
        self.hidden_states = torch.empty(max_tokens, hidden_size)
        self.logits = torch.empty(max_tokens, vocab_size)
        self.tokens = torch.empty(max_tokens, dtype=torch.long)
```

#### Why Pre-Allocation Matters

- Reduces CUDA allocator pressure
- Enables CUDA graph capture (graphs require fixed memory)
- Avoids synchronization on allocation

#### Action Items

1. Verify vLLM's persistent buffer usage with CUDA graphs enabled
2. Profile allocation patterns with `--enforce-eager` removed
3. Compare buffer management between vLLM and TRT-LLM

**Estimated Impact:** +0.5-1 tok/s (likely already partially implemented)

---

### 9. Native FP4 for Dense Layers

#### Current State

Dense layers (QKV, O, LM head) use **Marlin backend** which dequantizes FP4→BF16:

```
Current: Load FP4 weights → Dequant to BF16 → BF16 GEMM
```

For consistency with MoE, could use **FP8×FP4 GEMM**:

```
Proposed: Load FP4 weights → Quantize BF16 act to FP8 → FP8×FP4 GEMM
```

#### Trade-off Analysis

| Approach | Pros | Cons |
|----------|------|------|
| Marlin (dequant) | Proven, stable, works | Extra memory for BF16 weights |
| FP8×FP4 native | Consistent with MoE, lower memory | Requires activation quantization |

#### Current Assessment

Marlin is working well (+32% decode improvement for QKV/O). Native FP4 may provide marginal
additional gains but requires more engineering.

**Estimated Impact:** +2-5% decode TPS (if implemented)
**Effort:** High
**Priority:** Low (Marlin already works well)

---

## Additional Findings

### Why Our Eagle3 Acceptance Rate Is Low (~42%)

Factors that affect acceptance rate:

1. **Draft/verifier numerical mismatch**: If the draft model was trained on BF16 outputs but the
   verifier uses MXFP4, the distributions may not align well.

2. **Tree structure**: Different tree depths/widths affect acceptance probability.

3. **Sampling alignment**: Draft and endpoint sampling parameters should match.
   - **Draft sampling**: How the draft model selects candidate tokens to propose
   - **Endpoint sampling**: How the verifier selects the final token (temperature, top-k, etc.)

**Investigation needed:** Profile why our acceptance rate is ~42% and whether it's due to:
- Quantization mismatch (draft trained on BF16, verifier uses MXFP4)
- Sampling parameter mismatch
- Model-specific issues with gpt-oss-120b + Eagle3

### GEMV vs GEMM for MoE Layers

**Q: Should we use GEMV for MoE GEMM?**

**A: No.** Even for batch=1 decode:
- Each token activates 4 experts (gpt-oss-120b: `experts_per_token=4`)
- With 128 total experts, tokens are distributed sparsely
- Grouped GEMM with low-M tuning is more appropriate than per-expert GEMV
- Memory bandwidth bound regardless (loading expert weights dominates)

### vLLM Fused QKV Status

✅ **Confirmed**: vLLM has fused QKV via `QKVParallelLinear` and `MergedColumnParallelLinear`.

```python
# vllm/model_executor/layers/linear.py
class QKVParallelLinear(ColumnParallelLinear):
    # Fuses Q, K, V into single [3 * hidden_size, hidden_size] weight matrix
```

This is **not a gap** compared to TRT-LLM.

### TRT-LLM Dual-Path (FP4 + BF16) Clarification

**Q: Does TRT-LLM use FP4 and BF16 paths simultaneously?**

**A: No.** The architecture supports multiple paths, but at runtime:
- MXFP4 models use `MxE4m3MxE2m1BlockScaleMoeRunner` (FP8×FP4)
- BF16 models use separate runners
- The `trtllmGen` config shows fused prologue transforms (e.g., `patchF2fp` for FP4→BF16 dequant), not simultaneous dual-path

### Purpose of Dequant Path

The dequantization path (FP4→BF16) is used for:
1. **Marlin-style backends**: Full dequant before GEMM on older hardware
2. **Fallback path**: When native FP4 MMA is unavailable
3. **Debugging/validation**: Compare quantized vs dequantized accuracy

For SM121 with native FP8×FP4 MMA, dequant path is suboptimal but works.

---

## Recommended Action Plan

**Starting point:** 48.9 tok/s (vLLM + MXFP4 all layers)

### Phase 1: Quick Wins (1 week)

| Task | Expected Gain | Effort | Status | Source |
|------|---------------|--------|--------|--------|
| ~~Verify attention backend + sinks path; evaluate FP8 KV cache~~ | ~~+1-3 tok/s~~ | Low | ✅ **Done** (+0.3 t/s) | §A |
| Add greedy draft sampling option | +2-3 tok/s | Low | Pending | TRT-LLM §5 |
| Verify/enable CUDA graphs on SM121 *and ensure graph hit-rate* | +1-2 tok/s | Low | Pending | §3 + §B |
| **Port `VerifyTreeGreedy` CUDA kernel** | +1-2 tok/s | Low-Med | Pending | **SGLang** |
| **Use bitpacked tree masks** | +0.5-1 tok/s | Low | Pending | **SGLang** |
| Verify Eagle3 buffer allocation | +0.5-1 tok/s | Low | Pending | §8 |

**Note:** FP8 KV cache gave only +0.3 t/s (not +1-3 as estimated). Attention is not the bottleneck at batch_size=2.

**Expected Phase 1 result:** 48.6 → **52-55 tok/s**

### Phase 2: MoE Kernel Optimization (2-3 weeks)

| Task | Expected Gain | Effort | Source |
|------|---------------|--------|--------|
| **Investigate FP4 activations + FP4×FP4 `mxf4.block_scale` MoE GEMM on SM121** | +2-6 tok/s | High | llama.cpp |
| Implement min_latency_mode for Blackwell | +2-3 tok/s | Medium | TRT-LLM §7 |
| Add small-N MoE tiles (128×8, 128×16, 128×32) | +2-4 tok/s | Medium | TRT-LLM §4 |
| Tune pipeline stages (5-9) for SM121 | +1-2 tok/s | Medium | TRT-LLM §4 |
| **Port `topkGatingSoftmax` fused kernel for 128 experts** | +1-3 tok/s | Medium | **SGLang** |
| Tune MoE routing/permutation tile (`tileTokensDim`) for decode | +0.5-2 tok/s | Medium | TRT-LLM §C |
| Expand SM12x validated tactic/config set (remove "128×128 only" ceiling) | +1-3 tok/s | Medium | §C |
| Low-M CUDA core for QKV/O (N<128k; LM head N=201k excluded) | +1-2 tok/s | Medium | TRT-LLM §1 |

**Expected Phase 2 result:** 52-55 → **58-64 tok/s** (exceeds llama.cpp!)

### Phase 3: Advanced Eagle3 (3-4 weeks)

| Task | Expected Gain | Effort | Source |
|------|---------------|--------|--------|
| Optimize tree drafting (compare vLLM vs TRT-LLM vs SGLang approach) | +3-5 tok/s | Medium | TRT-LLM/SGLang §6 |
| Pre-computed tree masks (eagle_paths, etc.) | +1-2 tok/s | Medium | TRT-LLM §6 |
| CUTLASS prologue fusion (if needed after Phase 2) | +2-3 tok/s | High | TRT-LLM §2 |

**Expected Phase 3 result:** 58-64 → **65-72 tok/s** (with speculative decode)

### Phase 4: Long-term Excellence

| Task | Impact | Effort |
|------|--------|--------|
| Train quantization-aware Eagle3 draft model | +10-20 tok/s with spec decode | Very High |
| Port trtllmGen-style kernel generator to SM121 | +5-10% overall | Very High |
| Native FP4 for dense layers (FP8×FP4) | +2-5% | High |

---

## Appendix: Code References

### TRT-LLM Low-M Dispatch

Source: [`gemmPlugin.cpp:394-407`](https://github.com/NVIDIA/TensorRT-LLM/blob/bf16fbd86/cpp/tensorrt_llm/plugins/gemmPlugin/gemmPlugin.cpp#L394-L407) (commit bf16fbd86)

```cpp
if (!isArch90or100 && M <= 4 && N <= 128000 && mUseFp8 && noPadDim && cudaKernelSupportType)
{
    cudaKernelFinished = cuda_core_gemm::cudaCoreGemmDispatcher(params, stream);
}
```

### vLLM Eagle3 CUDA Graph Support

```python
# vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py
class EagleCudaGraphManager:
    # vLLM HAS CUDA graph support for Eagle3
    # Question: Is it enabled on SM121? (check --enforce-eager)
    def capture_graph(self, num_tokens, generate_fn, ...):
        capture_graphs(num_tokens, generate_fn, ...)
```

### FlashInfer SM120 MoE GEMM

```cpp
// flashinfer/csrc/nv_internal/.../moe_gemm_sm120_mixed_input_launcher.inl
// Uses: ElementInputA = float_e4m3_t (FP8), ElementInputB = float_e2m1_t (FP4)
// Requires explicit pre-quantization of BF16→FP8
```

### SGLang Fused TopK+Softmax Kernel

Source: `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu`

```cpp
// Warp-level fused softmax + top-k for power-of-2 expert counts
template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(
    const T* input, const bool* finished, float* output, const int num_rows,
    int* indices, const int k, const int start_expert, const int end_expert,
    const bool renormalize, const float moe_softcapping, const float* correction_bias);

// Template specializations for 1, 2, 4, 8, 16, 32, 64, 128, 256 experts
// gpt-oss-120b uses 128 experts → uses case 128
```

### SGLang Tree Verification Kernel

Source: `sgl-kernel/csrc/speculative/eagle_utils.cu`

```cpp
// Single GPU kernel for Eagle3 tree verification
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts, IdType* accept_index, IdType* accept_token_num,
    IdType2* candidates, IdType2* retrive_index, IdType2* retrive_next_token,
    IdType2* retrive_next_sibling, IdType2* target_predict,
    uint32_t batch_size, uint32_t num_speculative_tokens, uint32_t num_draft_tokens);
```

### SGLang Bitpacked Tree Masks

Source: `sgl-kernel/csrc/speculative/eagle_utils.cu`

```cpp
// 8x memory reduction: pack bool masks into uint8_t
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list, int64_t* selected_index, int64_t* verified_seq_len,
    uint8_t* tree_mask,  // Packed: 1 bit per element instead of 1 byte
    int64_t* positions, int64_t* retrive_index, ...);
```

---

## Summary

**Current best:** 49.0 tok/s (vLLM + MXFP4 all layers + FP8 KV) — **only 6% below SGLang target!**

**Model:** gpt-oss-120b (hidden=2880, 36 layers, 128 experts, 4/token, vocab=201088)

**Already implemented:**
- ✅ MXFP4 MoE with CUTLASS FP8×FP4
- ✅ MXFP4 QKV/O with Marlin (+32% decode improvement)
- ✅ MXFP4 lm_head with Marlin (+27% additional improvement)
- ✅ Fused QKV projection
- ✅ FP8 E4M3 KV cache with attention sinks (+0.3 t/s, minimal — attention not bottleneck)

**Quick wins (Phase 1) — from TRT-LLM + SGLang:**
0. ~~**Attention sanity & knobs**~~ — ✅ Done. FP8 KV cache tested, +0.3 t/s (not bottleneck at bs=2)
1. **Greedy draft sampling** — Add argmax option to Eagle3 (+2-3 tok/s) — Low effort [TRT-LLM]
2. **Verify CUDA graphs on SM121** — Remove `--enforce-eager`, test (+1-2 tok/s) — Low effort
3. **Port `VerifyTreeGreedy` CUDA kernel** — Eliminate Python tree verification (+1-2 tok/s) — Low-Med effort [SGLang]
4. **Use bitpacked tree masks** — 8x memory reduction (+0.5-1 tok/s) — Low effort [SGLang]
5. **Verify Eagle3 buffer allocation** — Check persistent buffer usage (+0.5-1 tok/s) — Low effort

**MoE optimization (Phase 2) — from TRT-LLM + llama.cpp + SGLang:**
6. **Implement min_latency_mode for Blackwell** — [Not yet implemented](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.cutlass_fused_moe.html) (+2-3 tok/s) — Medium effort [TRT-LLM]
7. **MoE tile+stage tuning** — Add small-N tiles (128×8, 128×16) (+2-4 tok/s) — Medium effort [TRT-LLM]
8. **Port `topkGatingSoftmax` fused kernel** — Fuse softmax+topk for 128 experts (+1-3 tok/s) — Medium effort [SGLang]
9. **Expand SM12x MoE kernel configs** — Remove "128×128 only" ceiling (+1-3 tok/s) — Medium effort
10. **Low-M CUDA core for QKV/O** — N<128k qualifies; LM head (vocab=201k) excluded (+1-2 tok/s) — Medium effort [TRT-LLM]
11. **Investigate FP4×FP4 `mxf4.block_scale`** — Native Blackwell MMA for MoE (+2-6 tok/s) — High effort [llama.cpp]

**Advanced Eagle3 (Phase 3):**
12. **Optimize tree drafting** — Compare vLLM vs TRT-LLM vs SGLang approach (+3-5 tok/s) — Medium effort

**Projected trajectory:**
- Phase 1: 49.0 → **52-55 tok/s** (matches/exceeds SGLang!)
- Phase 2: 52-55 → **58-64 tok/s** (exceeds llama.cpp!)
- Phase 3: 58-64 → **65-72 tok/s** (with speculative decode)

**Key learning from FP8 KV cache testing:** Attention is NOT the bottleneck at batch_size=2 for MoE models.
MoE GEMM dominates decode time. Focus remaining optimization effort on MoE kernel tuning (Phase 2).

**Key insight:** SGLang's applicable optimizations for gpt-oss-120b + SM121 are primarily in the
**Eagle3 speculative decoding path** (tree verification, bitpacking), not MoE kernels. Most SGLang
MoE optimizations target DeepSeek V3 (biased grouped topk, shared experts) or expert parallel,
which don't apply to our single-GPU gpt-oss scenario.

Note: gpt-oss-120b is MoE-dominated (128 experts). Dense layer optimizations have less impact
due to smaller hidden_size=2880. MoE tile/stage tuning and Eagle3 improvements are highest priority.
