# vLLM Performance Gap Analysis vs. llama.cpp and SGLang

## Current State (2026-01-09)

| Engine | pp2048 (tok/s) | tg32 (tok/s) | Gap |
|--------|----------------|--------------|-----|
| **llama.cpp** | 2,450 | **57.85** | Baseline |
| **SGLang spark** | - | **~52** | - |
| **Our vLLM** | **4,808** ✓ | 29.26 | **2x slower decode** |

**Prefill is excellent** (2x better than llama.cpp). **Decode is the blocker**.

---

## Root Cause Analysis: Why is vLLM 2x Slower on Decode?

### 1. MoE Kernel Tile Efficiency (40% of GPU time)

| Aspect | vLLM/FlashInfer | llama.cpp |
|--------|-----------------|-----------|
| **Kernel** | `cutlass_fused_moe` (Grouped GEMM) | `mul_mat_vec_q` (GEMV per expert) |
| **Tile Shape** | 128×128×128 (fixed) | Optimized per K dimension |
| **M=1 Efficiency** | 0.78% (1/128) | Near 100% |
| **Weight Reuse** | ✅ Batches all 8 experts | ❌ Per-expert (8× memory traffic) |
| **Net Result** | Weight reuse wins at M=1 | Better tile efficiency, but 8× traffic |

**Why CUTLASS grouped GEMM is still faster:**
- At M=1, loading weights dominates compute
- Grouped GEMM loads each expert's weights **once** for all tokens routed to it
- GEMV loads weights separately for each of 8 experts per token
- Memory bandwidth: 8× difference overcomes tile inefficiency

### 2. Activation Persistence (Key Insight from llama.cpp)

| Aspect | vLLM/FlashInfer | llama.cpp |
|--------|-----------------|-----------|
| **Activation Format** | BF16 between layers | **Q8_1 persisted** |
| **Per-Layer Overhead** | BF16 → FP8 → FP32 → BF16 (each layer) | Q8 → FP32 → Q8 (reuse scale) |
| **Memory Traffic** | 2 bytes/element × 2 (read+write) | 1 byte/element + small overhead |

**Critical Difference:**
llama.cpp keeps activations in quantized format (Q8_1) between layers, only converting at:
- Input embedding → first layer
- Last layer → output logits

vLLM converts BF16→FP8 on **every single MoE call** (60 layers × 2 GEMMs = 120 quantizations per token).

### 3. Framework Overhead

| Aspect | vLLM | SGLang | llama.cpp |
|--------|------|--------|-----------|
| **Scheduler** | Python async | "Zero-overhead C++" | Native C++ |
| **IPC** | ZMQ between processes | Minimal | None |
| **Token Dispatch** | Per-iteration overhead | Batched | Inline |
| **Memory Alloc** | Dynamic | Pre-allocated pools | Stack-allocated |

SGLang's "Zero-Overhead Batch Scheduler" (from [v0.4 blog](https://docs.sglang.ai/)) explicitly targets this.

### 4. Attention (51% of GPU time)

| Aspect | vLLM | llama.cpp |
|--------|------|-----------|
| **Kernel** | FlashInfer FA2 decode | Custom FA2-like |
| **KV Cache** | BF16 | Often Q8 or FP8 |
| **Layout** | HND (forced) | Optimized per arch |

Attention takes **51%** of decode time (from nsys profiling). Optimizing this is equally important as MoE.

---

## Actionable Optimization Paths

### Path A: Reduce Activation Quantization Overhead (High Impact)

**Goal:** Persist activations in FP8/Q8 format between layers, avoiding BF16 roundtrip.

**Implementation:**
1. Modify vLLM model executor to keep hidden states in FP8 between MoE layers
2. Only convert FP8→BF16 for:
   - LayerNorm (requires FP32)
   - Attention (depends on kernel support)
   - Final output logits

**Estimated Impact:** 20-30% decode speedup (eliminates 60× BF16→FP8 per token)

**Complexity:** Medium - requires model executor changes

### Path B: FP4 KV Cache (SGLang has this)

**Goal:** Reduce attention memory bandwidth by 4× with FP4 KV cache.

**Implementation:**
- SGLang already has `KVFP4QuantizeUtil` in their spark branch
- FlashInfer may need FP4 KV attention kernel

**Code Reference:**
```python
# From SGLang spark branch
from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil
cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
```

**Estimated Impact:** 10-20% decode speedup (attention is 51% of time, but mostly compute-bound)

**Complexity:** Medium - need FP4 attention kernel

### Path C: Zero-Overhead Scheduler (SGLang approach)

**Goal:** Eliminate Python scheduler overhead between decode iterations.

**Implementation:**
1. Move scheduler hot path to C++/Rust
2. Pre-allocate memory pools
3. Eliminate ZMQ IPC for single-GPU case

**Estimated Impact:** 10-20% decode speedup (framework overhead is ~10% of latency)

**Complexity:** High - major vLLM architectural change

### Path D: Speculative Decoding (Increase Effective Batch Size)

**Goal:** Process multiple candidate tokens per forward pass, amortizing fixed costs.

**Implementation:**
- Use draft model (small transformer) for speculation
- N-gram speculation doesn't help for open-ended generation (tested)

**Estimated Impact:** 20-50% speedup if acceptance rate > 60%

**Complexity:** Low - vLLM has built-in support, just need compatible draft model

### Path E: Persistent CUDA Kernels

**Goal:** Keep kernels resident between decode iterations to eliminate launch overhead.

**Implementation:**
- Use CUDA persistent kernel pattern
- Keep thread blocks alive, waiting for new tokens
- Communicate via CUDA streams/events

**Estimated Impact:** 5-10% decode speedup

**Complexity:** Very High - requires rewriting MoE/attention kernels

---

## Recommended Priority Order

| Priority | Path | Impact | Effort | Dependencies |
|----------|------|--------|--------|--------------|
| **1** | D: Draft Model Speculation | High | Low | Find/train draft model |
| **2** | A: FP8 Activation Persistence | High | Medium | Model executor changes |
| **3** | B: FP4 KV Cache | Medium | Medium | FlashInfer kernel |
| **4** | C: Zero-Overhead Scheduler | Medium | High | Major refactor |
| **5** | E: Persistent Kernels | Low | Very High | Kernel rewrites |

---

## Concrete Next Steps

### Step 1: Test Eagle3 Speculative Decoding (BLOCKED)

NVIDIA released official Eagle3 heads for gpt-oss-120b:
- [nvidia/gpt-oss-120b-Eagle3-throughput](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-throughput) - For high-concurrency
- [nvidia/gpt-oss-120b-Eagle3-short-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-short-context) - For <8k context
- [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context) - For >8k context

**Current Status: BLOCKED**

vLLM + FlashInfer combination fails with:
```
AssertionError: FlashInfer backend currently only supports models in which 
all layers share the same values for the following hyperparameters: 
`window_left`, `logits_soft_cap`, `sm_scale`.
```

The Eagle3 drafter model has different attention hyperparameters than the main
model, and FlashInfer's attention backend doesn't support heterogeneous configs.

**Potential Fixes:**
1. Modify FlashInfer to support per-layer hyperparameters
2. Use a different attention backend (but FA2 is required for SM121)
3. Wait for vLLM/FlashInfer upstream fix

**Workaround (Not Yet Tested):**
Try TensorRT-LLM as the model card suggests:
```bash
trtllm-serve <gpt-oss-120b checkpoint> --backend pytorch \
    --speculative_config.decoding_type=Eagle \
    --speculative_config.speculative_model_dir=<eagle3 checkpoint>
```

### Step 2: Profile Activation Overhead (1 day)

1. Instrument `mxfp8_quantize` calls in FlashInfer
2. Measure time spent on BF16→FP8 conversion per token
3. Quantify potential savings from FP8 persistence

### Step 3: Evaluate FP4 KV Cache (2-3 days)

1. Review SGLang's `KVFP4QuantizeUtil` implementation
2. Check if FlashInfer supports FP4 KV attention
3. Prototype integration if kernel exists

---

## Why llama.cpp is Faster (Summary)

1. **Q8 Activation Persistence**: Activations stay quantized between layers
2. **Optimized GEMV**: Custom DP4A kernels tuned for small batch sizes
3. **No Framework Overhead**: Native C++ with no Python/IPC
4. **Inline Processing**: No token batching/dispatch overhead

## Why SGLang is Faster (Likely)

1. **Zero-Overhead Scheduler**: C++/Rust scheduler core
2. **FP4 KV Cache**: Reduces attention memory bandwidth
3. **Better Memory Management**: Pre-allocated pools
4. **Same FlashInfer MoE**: Uses identical `cutlass_fused_moe`

The MoE kernel itself is likely the same. The difference is everything around it.

