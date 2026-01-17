# MXFP4 Feature Matrix

Status and configuration for each feature in the optimization work.

> **See also:** [SM121_OPTIMIZATION_ANALYSIS.md](SM121_OPTIMIZATION_ANALYSIS.md) for detailed TRT-LLM comparison and impact estimates.

---

## Performance-Ranked Feature Overview

Features ranked by **decode TPS impact** based on TRT-LLM analysis:

| Rank | Feature | Status | Decode Impact | Effort | Notes |
|------|---------|--------|---------------|--------|-------|
| **1** | Low-M CUDA core dispatch (dense layers) | ❌ Missing | **+15-20%** | Medium | TRT-LLM uses M≤4 threshold |
| **2** | Fused BF16→FP8 prologue (MoE GEMM) | ❌ Missing | **+10-15%** | High | Requires CUTLASS prologue mod |
| **3** | CUDA graph for Eagle3 drafting | ❌ Missing | **+10-15%** | Medium | TRT-LLM captures full loop |
| **4** | MoE GEMM min_latency tuning (SM121) | ⚠️ Partial | **+5-10%** | Low | Need SM121-specific tactics |
| **5** | CUTLASS Grouped GEMM | ✅ Works | Baseline | - | MARLIN better acceptance |
| **6** | Fused QKV projection | ✅ Complete | +2-5% | Done | vLLM has `QKVParallelLinear` |
| **7** | Pre-computed Eagle3 tree masks | ⚠️ Partial | +2-5% | Low | Some allocation per call |
| **8** | QKV/O MXFP4 | ✅ Complete | +1-3% | Done | `--mxfp4-layers moe,qkv,o` |
| **9** | lm_head MXFP4 | ✅ Complete | <1% | Done | Only 6% of decode time |

**Legend:** ✅ Complete | ⚠️ Partial | ❌ Missing | 🚫 Not Pursuing

---

## Legacy Feature Status

| Feature | Status | Priority | Env Var / CLI | Default |
|---------|--------|----------|---------------|---------|
| CUTLASS Grouped GEMM | ✅ Works | P1 | `VLLM_MXFP4_MOE_KERNEL=gemm` | `auto` |
| **QKV/O MXFP4** | ✅ Complete | P1.5 | `--mxfp4-layers moe,qkv,o` | `moe` |
| Runner/Setup Caching | ⏳ Pending | P2 | - | - |
| Activation MXFP8 | ⏳ Pending | P3 | `VLLM_MXFP4_ACTIVATION=mxfp8` | `bf16` |
| Tile Variants (64x128) | 🚫 Blocked | P4 | `FLASHINFER_MOE_TILE=64x128` | `auto` |
| Speculative Decoding | ⚠️ Works | P5 | vLLM flags | off |
| Attention Sinks | ⏳ Pending | P6 | `VLLM_USE_ATTENTION_SINKS=1` | `0` |
| **lm_head MXFP4** | ✅ Complete | - | `--mxfp4-layers lm_head` | off |

---

## Key Optimization: Low-M Dense Layer Dispatch

**The biggest gap vs TRT-LLM**: vLLM always uses GEMM for dense layers, even for M=1 (single token decode).

### TRT-LLM's Approach

```cpp
// TRT-LLM dispatches to CUDA core kernels for small M:
if (M <= 4 && mUseFp8)  cudaCoreGemmDispatcher(...)  // FP8 path
if (M <= 6 && !mUseFp8) cudaCoreGemmDispatcher(...)  // BF16/FP16 path
else                    runGemm(...)                  // CUTLASS/cuBLAS
```

### Why CUDA Cores Win at M=1

| Metric | Tensor Core GEMM | CUDA Core GEMV-like |
|--------|------------------|---------------------|
| Occupancy at M=1 | Poor (wasted lanes) | Good |
| Memory pattern | Tiled (overhead) | Column-streaming |
| Setup overhead | High (TMA, warp sync) | Low |

### Affected Layers (gpt-oss-120b)

| Layer | Per Forward | M at Decode | Impact |
|-------|-------------|-------------|--------|
| QKV projection | 61 | 1 | High |
| O projection | 61 | 1 | High |
| LM head | 1 | 1 | High (N=256k) |
| Gate/Up (non-MoE) | 3 | 1 | Medium |

### Action Item

Port TRT-LLM's `cuda_core_gemm` kernels to vLLM/FlashInfer. Expected impact: **+15-20% decode TPS**.

---

**Note on lm_head MXFP4**: Currently uses Marlin kernel (weight-only FP4 compression → dequant to BF16 → BF16 GEMM). This is a pragmatic intermediate that reduces memory bandwidth but does NOT use native FP8×FP4 MMA. Since lm_head is only ~6% of decode time, this is not the bottleneck. The path to 52+ tok/s requires optimizing MoE (34% of decode) and attention (1.5% but could be higher with different configs), not lm_head.

**lm_head MXFP4 Compatibility Gates** (automatically disabled when):
- **LoRA enabled**: FP4-packed weights incompatible with LoRA's additive updates
  - Note: This is a broad gate - ANY LoRA config disables lm_head MXFP4, even if LoRA doesn't target lm_head specifically
- **Tied embeddings**: Would corrupt shared embedding table
  - Config check: `hf_config.tie_word_embeddings`
  - Structural check: `data_ptr()` comparison detects actual storage aliasing
- Falls back to BF16 lm_head when either condition is detected

**Legend**: ✅ Complete | 🔄 In Progress | ⏳ Pending | ❌ Blocked | 🚫 Not Pursuing

---

## CLI Arguments

### MXFP4 Layer Selection (`--mxfp4-layers`)

Control which layer types are quantized with MXFP4 (Marlin kernel for dense, hardware-specific for MoE):

| Token | Layers Matched | Description |
|-------|----------------|-------------|
| `moe` | `*.experts.*` | MoE expert weights (default, always included) |
| `qkv` | `*.qkv_proj` | Fused QKV projection |
| `o` | `*.o_proj` | Attention output projection |
| `lm_head` | `lm_head` | Output logits projection |
| `all` | All above | Shorthand for full quantization |

**Default**: `--mxfp4-layers moe` (backwards compatible)

**Usage Examples**:

```bash
# Default: only MoE experts (current behavior)
vllm serve openai/gpt-oss-120b --quantization mxfp4

# Add QKV/O projections for decode speedup
vllm serve openai/gpt-oss-120b --quantization mxfp4 --mxfp4-layers moe,qkv,o

# Full quantization (all supported layers)
vllm serve openai/gpt-oss-120b --quantization mxfp4 --mxfp4-layers all

# Equivalent to "all"
vllm serve openai/gpt-oss-120b --quantization mxfp4 --mxfp4-layers moe,qkv,o,lm_head
```

**Compatibility Notes**:
- **LoRA**: When LoRA is enabled, QKV/O/lm_head fall back to BF16 (LoRA incompatible with FP4-packed weights)
- **Tied embeddings**: lm_head falls back to BF16 when `tie_word_embeddings=True`
- **Blackwell-only**: lm_head MXFP4 currently requires SM12x (GB10/Thor)

---

## Environment Variables

### MoE Kernel Selection

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_MXFP4_MOE_KERNEL` | `auto`, `marlin`, `gemm`, `gemv`, `triton` | Override MoE kernel selection |

```bash
# Use Marlin (baseline)
export VLLM_MXFP4_MOE_KERNEL=marlin

# Use CUTLASS grouped GEMM
export VLLM_MXFP4_MOE_KERNEL=gemm

# Use DP4A GEMV (experimental, currently slower)
export VLLM_MXFP4_MOE_KERNEL=gemv

# Use Triton
export VLLM_MXFP4_MOE_KERNEL=triton
```

### Activation Quantization

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_MXFP4_ACTIVATION` | `bf16`, `mxfp8`, `mxfp4`, `fp6` | Activation format for MoE |

```bash
# BF16 activations (default, most compatible)
export VLLM_MXFP4_ACTIVATION=bf16

# MXFP8 activations (reduced conversion overhead)
export VLLM_MXFP4_ACTIVATION=mxfp8
```

### FlashInfer Configuration

| Variable | Options | Description |
|----------|---------|-------------|
| `FLASHINFER_MOE_TILE` | `auto`, `128x128`, `64x128` | Tile configuration for MoE GEMM |
| `FLASHINFER_LOGLEVEL` | `0`-`5` | Logging verbosity |
| `FLASHINFER_JIT_VERBOSE` | `0`, `1` | Show JIT compilation output |
| `FLASHINFER_NVCC_THREADS` | number | Parallel JIT compilation threads |

### Attention Configuration

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER`, `FLASH_ATTN` | Attention backend |
| `VLLM_USE_ATTENTION_SINKS` | `0`, `1` | Enable attention sinks |

### CUDA Graphs

| Flag/Variable | Description |
|---------------|-------------|
| `--enforce-eager` | Disable CUDA graphs |
| `VLLM_USE_CUDA_GRAPH=0` | Disable CUDA graphs via env |

---

## Speculative Decoding Options

### Eagle3 Models

| Model | Use Case | Flag |
|-------|----------|------|
| `nvidia/gpt-oss-120b-Eagle3-short-context` | Short prompts (<4K) | `--speculative-model` |
| `nvidia/gpt-oss-120b-Eagle3-long-context` | Long prompts (>4K) | `--speculative-model` |
| `nvidia/gpt-oss-120b-Eagle3-throughput` | Batch throughput | `--speculative-model` |

### vLLM Speculative Decoding Config

```bash
vllm serve openai/gpt-oss-120b \
    --speculative-config '{"method": "eagle3", "model": "nvidia/gpt-oss-120b-Eagle3-short-context", "num_speculative_tokens": 3}' \
    --quantization mxfp4 \
    --enforce-eager  # Required on SM121 currently
```

---

## Feature Dependencies

```
CUTLASS GEMM (P1)
    │
    ▼
Runner Caching (P2) ──────────────────┐
    │                                 │
    ▼                                 │
Activation MXFP8 (P3)                 │
    │                                 │
    ▼                                 │
Tile Variants (P4)                    │
    │                                 │
    ▼                                 ▼
Speculative Decoding (P5) ◄─── Gate: tg32 ≥40 tok/s
    │
    ▼
Attention Sinks (P6)
```

---

## Configuration Combinations Tested

| ID | Kernel | Activation | Spec Decode | Sinks | Graphs | Status |
|----|--------|------------|-------------|-------|--------|--------|
| B1 | marlin | bf16 | off | off | on | ⏳ Baseline |
| C1 | gemm | bf16 | off | off | on | ⏳ P1 |
| C2 | gemm | mxfp8 | off | off | on | ⏳ P3 |
| C3 | gemm | mxfp8 | eagle3-short | off | off | ⏳ P5 |
| C4 | gemm | mxfp8 | eagle3-short | on | off | ⏳ P6 |

---

## Eagle3 Speculative Decoding Status

### Current Performance (vLLM + MXFP4)

| Config | Backend | K | Accept Rate | Decode TPS |
|--------|---------|---|-------------|------------|
| eagle3_wip | CUTLASS | 1 | 42.8% | 28.5 |
| eagle3_wip | CUTLASS | 4 | 20.5% | 20.1 |
| eagle3_wip | MARLIN | 1 | **48.1%** | **33.8** |
| eagle3_wip | MARLIN | 4 | 24.2% | 21.3 |

**Finding**: MARLIN (dequant→BF16) outperforms CUTLASS (FP8×FP4) due to better draft-verifier alignment.

### TRT-LLM Comparison

| Engine | Model | Accept Rate | Reason |
|--------|-------|-------------|--------|
| TRT-LLM | DeepSeek-R1-FP4 | 86% | Draft trained on quantized outputs |
| vLLM | gpt-oss-120b | 42-48% | Draft trained on BF16 outputs |

**Implication**: Acceptance gap is due to draft model training, not implementation.

### Implementation Gaps vs TRT-LLM

| Feature | TRT-LLM | vLLM | Impact |
|---------|---------|------|--------|
| CUDA graph drafting loop | ✅ | ❌ | +10-15% TPS |
| Pre-computed tree masks | ✅ | ⚠️ | +2-5% TPS |
| Greedy draft sampling | ✅ | ✅ | - |

---

## Known Issues by Configuration

### CUDA Graphs + Eagle3

- **Issue**: `cudaErrorIllegalInstruction` during replay
- **Workaround**: Use `--enforce-eager`
- **Status**: Open

### Attention Sinks + SM121

- **Issue**: `cudaErrorIllegalInstruction` in sink kernel
- **Workaround**: Disable sinks
- **Status**: Open

### M=64 Tiles

- **Issue**: TMA layout constraint with 128-element scale granularity
- **Workaround**: Use M=128 tiles only
- **Status**: Requires FlashInfer architectural change

### Tied Embeddings + MXFP4 lm_head

- **Issue**: When `tie_word_embeddings=True`, `lm_head.weight` aliases `embed_tokens.weight`. Quantizing lm_head would also quantize the embedding table, breaking `F.embedding()` (FP4-packed weights are incompatible with embedding lookup).
- **Behavior**: MXFP4 quantization is automatically skipped for `lm_head` when `tie_word_embeddings=True`.
- **Detection**: Checked in `Mxfp4Config.get_quant_method()` via `hf_config.tie_word_embeddings`.
- **Log Message**: `"[MXFP4] Skipping MXFP4 quantization for lm_head... tie_word_embeddings=True"`
- **Status**: By design (not a bug)

---

## Quick Reference: Full Config Example

```bash
# Optimized configuration (when all features work)
export VLLM_MXFP4_MOE_KERNEL=gemm
export VLLM_MXFP4_ACTIVATION=mxfp8
export FLASHINFER_MOE_TILE=auto
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDA_GRAPH=1
export FLASHINFER_LOGLEVEL=0

vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --speculative-config '{"method": "eagle3", "model": "nvidia/gpt-oss-120b-Eagle3-short-context", "num_speculative_tokens": 3}'
```
