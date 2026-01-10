# vLLM Baseline Kernel Analysis: Marlin + FlashInfer FA2

**Date**: 2026-01-10  
**Configuration**: Marlin MoE kernel + FlashInfer CUTLASS FA2 attention  
**Model**: gpt-oss-120b (MXFP4 quantized)  
**GPU**: NVIDIA GB10 (SM121)  
**Performance**: 30.3 tok/s decode (64 tokens, 2048 prompt)

---

## Executive Summary

**Attention is NOT the bottleneck.** FlashInfer attention accounts for only **~1.5%** of GPU time.

The decode path is dominated by:
1. **MoE GEMM (Marlin)**: ~34%
2. **Dense GEMV/GEMM**: ~38%
3. **Memory operations**: ~12%
4. **Activations/Norms**: ~5%
5. **Attention**: ~1.5%

---

## Kernel Breakdown by Category

### Top Kernels by GPU Time

| Rank | Time % | Category | Kernel | Notes |
|------|--------|----------|--------|-------|
| 1 | 22.3% | Dense GEMV | `internal::gemvx::kernel<bf16>` | embed_tokens/lm_head |
| 2 | 20.8% | MoE GEMM | `marlin_moe_wna16::Marlin` | Expert GEMM (decode) |
| 3 | 13.3% | Dense GEMV | `internal::gemvx::kernel<bf16>` | More dense layers |
| 4 | 12.8% | MoE GEMM | `marlin_moe_wna16::Marlin` | Expert GEMM (prefill) |
| 5 | 5.6% | Memory | `elementwise_kernel` (bf16→fp8 copy) | Activation conversion |
| 6 | 4.0% | Memory | `elementwise_kernel` | PyTorch ops |
| 7 | 3.4% | Memory | `CatArrayBatchedCopy` | Tensor concatenation |
| 8 | 2.8% | Dense GEMM | `cutlass_80_tensorop_bf16_gemm` | Non-MoE linear layers |
| 9 | 2.3% | Memory | `FillFunctor<unsigned char>` | Buffer initialization |
| 10 | 2.1% | MoE | `marlin_repack_kernel` | Weight repacking |
| 11 | 1.4% | Activation | `swigluoai_and_mul_kernel` | SwiGLU activation |
| **12** | **1.2%** | **Attention** | `BatchPrefillWithPagedKVCacheKernel` | **FlashInfer FA2** |
| 13 | 1.2% | Norm | `fused_add_rms_norm_kernel` | RMSNorm + residual |
| 14 | 1.1% | Memory | `elementwise_kernel` | Various ops |
| 15 | 0.9% | Memory | `reduce_kernel` | Reductions |
| 16 | 0.5% | Embedding | `rotary_embedding_kernel` | RoPE |
| 17 | 0.3% | Attention | `PersistentVariableLengthMergeStatesKernel` | State merging |
| 18 | 0.2% | MoE | `moe_align_block_size_kernel` | Expert routing |
| 19 | 0.2% | KV Cache | `reshape_and_cache_flash_kernel` | KV cache write |
| 20 | 0.2% | MoE | `topkGatingSoftmax` | Router softmax |

---

## Aggregated Categories

| Category | Total Time % | Primary Kernels |
|----------|--------------|-----------------|
| **MoE GEMM** | **33.6%** | `marlin_moe_wna16::Marlin` |
| **Dense GEMV/GEMM** | **38.4%** | `gemvx::kernel`, `cutlass_80_tensorop` |
| **Memory/Data Movement** | **12.4%** | `elementwise_kernel`, `CatArrayBatchedCopy`, `FillFunctor` |
| **MoE Overhead** | **2.7%** | `marlin_repack`, `moe_align`, `topkGatingSoftmax`, `count_and_sort` |
| **Activations/Norms** | **3.1%** | `swigluoai`, `fused_add_rms_norm`, `rotary_embedding` |
| **Attention** | **1.5%** | `BatchPrefillWithPagedKVCacheKernel`, `MergeStatesKernel` |
| **KV Cache** | **0.4%** | `reshape_and_cache_flash_kernel`, `copy_page_indices` |
| **Other** | **7.9%** | Various PyTorch ops |

---

## Key Findings

### 1. Attention is NOT the Bottleneck (1.5% of time)

FlashInfer attention kernels only consume **1.5%** of GPU time:
- `BatchPrefillWithPagedKVCacheKernel`: 1.2%
- `PersistentVariableLengthMergeStatesKernel`: 0.3%

**Conclusion**: Optimizing attention further (e.g., trying different backends) will have minimal impact on overall throughput.

### 2. MoE GEMM is Expensive (33.6% of time)

Marlin MoE kernels are the second-largest category:
- Decode experts: 20.8%
- Prefill experts: 12.8%

**Optimization opportunities**:
- CUTLASS grouped GEMM might be faster than Marlin for MXFP4
- Runner caching to reduce setup overhead
- Tile size tuning (64x128 vs 128x128)

### 3. Dense GEMV is the Largest Category (38.4%)

`gemvx::kernel` calls dominate:
- 22.3% + 13.3% = 35.6% just from GEMV
- Additional 2.8% from CUTLASS GEMM

These are likely:
- `embed_tokens` (input embedding)
- `lm_head` (output projection)
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projections)

**Note**: Dense layers use GEMV (M=1 decode) while MoE uses GEMM. This is expected for single-token decode.

### 4. Memory Overhead is Significant (12.4%)

Data movement operations consume 12.4%:
- BF16↔FP8 conversions for MoE
- Tensor concatenation
- Buffer initialization

**Optimization opportunities**:
- Fused kernels to reduce memory round-trips
- MXFP8 persistent activations (avoid bf16→fp8 conversion)

### 5. MoE Routing Overhead is Small (2.7%)

Expert routing ops are relatively cheap:
- `marlin_repack`: 2.1%
- `topkGatingSoftmax`: 0.2%
- `moe_align_block_size`: 0.2%
- `count_and_sort_expert_tokens`: 0.1%

---

## Implications for Optimization

### Per Plan Phase 4 Decision Gate:

| Condition | Result | Action |
|-----------|--------|--------|
| "MoE is >50% of decode time" | **34%** - Close | Focus on MoE kernel optimization |
| "Attention is >40% of decode time" | **1.5%** - No | **Do NOT focus on attention** |
| "Quantization overhead >10%" | **~5%** - No | Memory ops are the overhead, not quant |

### Recommended Priority Order (Updated)

1. **MoE Kernel Optimization** (34% of time)
   - Test CUTLASS grouped GEMM vs Marlin
   - Tile size tuning
   - Runner caching

2. **Dense GEMV Optimization** (38% of time)
   - Consider INT8 quantization for lm_head
   - Fused embed+RoPE kernels

3. **Memory Reduction** (12% of time)
   - MXFP8 persistent activations
   - Fused activation+quantization

4. **Speculative Decoding** (multiplier on all above)
   - Eagle3 acceptance rate improvements
   - Numerical alignment with draft model

5. ~~Attention Optimization~~ **LOW PRIORITY** (1.5% of time)
   - Current FlashInfer FA2 is efficient
   - Attention sinks work correctly

---

## Comparison with Targets

| Engine | tg32 (tok/s) | Notes |
|--------|--------------|-------|
| llama.cpp | 57.85 | Uses DP4A GEMV, persistent activations |
| SGLang | ~52 | Unknown internals |
| **vLLM (this profile)** | **30.3** | Marlin + FlashInfer FA2 |

**Gap**: 27.5 tok/s (48% slower than llama.cpp)

---

## Raw nsys Data

Profile captured at: `/tmp/marlin_flashinfer_profile.nsys-rep`

### Command Used
```bash
VLLM_MXFP4_MOE_KERNEL=marlin \
VLLM_ATTENTION_BACKEND=FLASHINFER \
nsys profile -o /tmp/marlin_flashinfer_profile \
  python3 profile_decode.py --prompt-tokens 2048 --output-tokens 64 --runs 3
```

### Summary
- Total profiled runs: 3
- Tokens per run: 64
- Average throughput: 30.3 tok/s
- KV cache layout: HND
- Attention backend: FlashInfer CUTLASS FA2
- MoE kernel: Marlin
