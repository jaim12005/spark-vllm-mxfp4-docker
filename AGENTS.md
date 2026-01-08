# AGENTS.md - Project Context for AI Assistants

## Mission

Make **vLLM the fastest inference engine for gpt-oss-120b** on NVIDIA GB10 (SM121), outperforming SGLang and llama.cpp by leveraging:

- **Native FP4 hardware features** (SM121 block-scaled MMA)
- **MXFP4 quantized weights** (4-bit weights, group size 32)
- **FlashAttention-2 with attention sinks** for long context stability

---

## Critical: Implementation Stack

### WE ARE USING

| Component | Implementation | Location |
|-----------|----------------|----------|
| **MoE GEMM** | CUTLASS SM120/121 grouped GEMM | `flashinfer/fused_moe/` |
| **Attention** | FlashAttention-2 (FA2) | `flashinfer/prefill.py`, `flashinfer/decode.py` |
| **Attention Sinks** | FA2 variant | `flashinfer/jit/attention/variants.py` |
| **FP4 Quantization** | FlashInfer `mxfp4_quantize()` | `flashinfer/fp4_quantization.py` |
| **Framework** | vLLM with FlashInfer backend | Local repos mounted in Docker |

### WE ARE NOT USING

| ❌ Component | Why Not |
|-------------|---------|
| **TensorRT-LLM (trtllm)** | We use CUTLASS directly, not TRT-LLM wrappers |
| **cuDNN** | Not for MoE or attention in this project |
| **trtllm_fp4_block_scale_moe** | This is SM100-only, crashes on SM121 |
| **FA3** | SM90 only; we're on SM121 which uses FA2 |
| **nvfp4** | We use MXFP4 (group size 32), not NVFP4 (group size 16) |

---

## Correct API Usage

### MoE GEMM on SM121

```python
# ✓ CORRECT: Use cutlass_fused_moe
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

output = cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_indices,
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=torch.bfloat16,
    quant_scales=[],  # Empty for BF16, populated for FP4/FP8
    activation_type=ActivationType.Swiglu,
)
```

```python
# ❌ WRONG: Do NOT use trtllm_fp4_block_scale_moe on SM121
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe  # SM100 ONLY!
# This will fail with: "No supported CUDA architectures found for major versions [10]"
```

### MXFP4 Quantization

```python
# ✓ CORRECT: Use mxfp4_quantize for MXFP4 (group size 32)
from flashinfer import mxfp4_quantize, mxfp4_dequantize

weight_fp4, weight_scale = mxfp4_quantize(weight_bf16)
# weight_fp4: [M, K/2] uint8 (packed nibbles)
# weight_scale: uint8 (UE8M0 format, group size 32)
```

```python
# ❌ WRONG: nvfp4_quantize is for NVFP4 (group size 16), different format
from flashinfer import nvfp4_quantize  # Different format, don't mix!
```

### Attention with Sinks (FA2)

```python
# ✓ CORRECT: FA2 attention with sinks
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

wrapper = BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer,
    backend="fa2",  # Explicitly FA2
)
wrapper.plan(..., use_sinks=True)  # Enable attention sinks
```

---

## Architecture: SM121 (GB10)

- **Compute Capability**: 12.1
- **Memory**: 120GB (not "limited" - this is desktop Blackwell with full memory)
- **Block-scaled MMA**: Supports FP4×FP4, FP8×FP4, but NOT BF16×FP4 directly
- **MXFP4 Path**: BF16 activations → quantize to FP8 → FP8×FP4 kernel

### MXFP4 on SM121 (How It Actually Works)

SM121 block-scaled MMA only accepts FP8/FP6/FP4 inputs. For MXFP4 (W4A16) with BF16 activations:

1. **FlashInfer's higher-level API** accepts BF16 activations
2. **Internally quantizes** BF16 → FP8 (e4m3)
3. **Dispatches to FP8×FP4 CUTLASS kernel**

This is why you don't see "MXFP4" tile configs at the kernel level—they go through the FP8×FP4 path.

---

## Repository Layout

```
~/projects/
├── flashinfer/          # Local FlashInfer development (CUTLASS kernels)
│   ├── flashinfer/      # Python package
│   │   ├── fused_moe/   # MoE implementation (cutlass_fused_moe)
│   │   ├── prefill.py   # Prefill attention (FA2 + sinks)
│   │   ├── decode.py    # Decode attention
│   │   └── fp4_quantization.py  # mxfp4_quantize, nvfp4_quantize
│   ├── include/         # CUDA headers (framework-agnostic)
│   └── csrc/            # CUDA sources (TVM-FFI bindings)
│
├── vllm/                # Local vLLM (uses FlashInfer as backend)
│
└── ai/mxfp4/            # THIS REPO - Docker config + test harness
    ├── docker-compose.dev.yml   # Dev container config
    ├── docker-compose.yml       # Production config
    └── scripts/
        ├── sm121_vllm_test_harness.sh      # Crash-proof testing
        ├── verify_mxfp4_moe_kernel.py      # MXFP4 MoE verification
        ├── verify_sm121_mxfp4_moe_fa2_sinks.py  # Integration tests
        └── profile_sm121_decode_performance.py  # Profiling
```

---

## Docker Environment

The `vllm-dev` container mounts local repos:

```yaml
volumes:
  - ~/projects/flashinfer:/workspace/flashinfer
  - ~/projects/vllm:/workspace/vllm

environment:
  - PYTHONPATH=/workspace/flashinfer:/workspace/vllm${PYTHONPATH:+:$PYTHONPATH}
  - VLLM_ATTENTION_BACKEND=FLASHINFER
  - VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
  - VLLM_USE_CUDA_GRAPH=1
```

**Always verify you're using local FlashInfer:**
```bash
docker exec vllm-dev python3 -c "import flashinfer; print(flashinfer.__file__)"
# Should print: /workspace/flashinfer/flashinfer/__init__.py
# NOT: /usr/local/lib/python3.x/site-packages/flashinfer/...
```

---

## Known Issues / Gotchas

### 1. Wrong MoE API causes SM121 crash

**Symptom**: `RuntimeError: No supported CUDA architectures found for major versions [10]`

**Cause**: Using `trtllm_fp4_block_scale_moe` which targets SM100

**Fix**: Use `cutlass_fused_moe` instead

### 2. Previous M>=320 crash was API-specific

The old test (`verify_sm121_mxfp4_moe_fa2_sinks.py`) crashed at M>=320 tokens. This was due to using a different internal code path, NOT a CUTLASS kernel bug. The production `cutlass_fused_moe` API works for all M values.

### 3. KV cache layout is HND, not NHD

vLLM forces `HND` layout for KV cache on SM121. Ensure profiling/testing uses matching layout:

```python
wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    kv_layout="HND",  # Match vLLM's layout
)
```

### 4. PYTHONPATH required for editable install in Docker

FlashInfer's namespace package structure requires explicit PYTHONPATH. Without it, Python may import from site-packages instead of the mounted local repo.

---

## Performance Targets (Scores to Beat)

### The Problem: vLLM wins prefill but loses decode

| Depth | llama.cpp pp | vLLM pp | llama.cpp tg | vLLM tg | SGLang tg |
|-------|--------------|---------|--------------|---------|-----------|
| 0 | 2449.83 | **4663.70** ✓ | **57.85** | 33.55 ❌ | ~52 |
| 4096 | 2293.59 | **3172.51** ✓ | **54.81** | 32.63 ❌ | - |
| 8192 | 2147.98 | **2687.84** ✓ | **52.14** | 31.50 ❌ | - |
| 16384 | 1845.71 | **2044.47** ✓ | **48.53** | 29.55 ❌ | - |
| 32768 | 1404.70 | 1398.80 | **41.72** | 26.65 ❌ | - |

**Key insight**: vLLM is ~2x better on prefill, but ~40% slower on decode.

### Targets

| Metric | Current (vLLM) | Target | Gap |
|--------|----------------|--------|-----|
| **tg32 @ d0** | 33.55 t/s | ≥52 t/s | +55% needed |
| **tg32 @ d4096** | 32.63 t/s | ≥47 t/s | +44% needed |
| **tg32 @ d8192** | 31.50 t/s | ≥41 t/s | +30% needed |
| **tg32 @ d16384** | 29.55 t/s | ≥34 t/s | +15% needed |
| **tg32 @ d32768** | 26.65 t/s | ≥26 t/s | ✓ (parity) |

At longer contexts (32k), vLLM catches up. The decode gap narrows with depth.

### SGLang Reference (single prompt)

```
Output token throughput: 52.37 tok/s
TTFT: 49.87 ms  
TPOT: 18.83 ms (time per output token)
```

### Full Benchmark Data

<details>
<summary>llama.cpp (llama-bench, build f5acfb2ff)</summary>

```
model                      test              t/s
gpt-oss 120B MXFP4 MoE    pp2048            2449.83 ± 10.27
gpt-oss 120B MXFP4 MoE    tg32              57.85 ± 0.44
gpt-oss 120B MXFP4 MoE    pp2048 @ d4096    2293.59 ± 8.99
gpt-oss 120B MXFP4 MoE    tg32 @ d4096      54.81 ± 0.30
gpt-oss 120B MXFP4 MoE    pp2048 @ d8192    2147.98 ± 10.64
gpt-oss 120B MXFP4 MoE    tg32 @ d8192      52.14 ± 0.50
gpt-oss 120B MXFP4 MoE    pp2048 @ d16384   1845.71 ± 7.11
gpt-oss 120B MXFP4 MoE    tg32 @ d16384     48.53 ± 0.36
gpt-oss 120B MXFP4 MoE    pp2048 @ d32768   1404.70 ± 7.36
gpt-oss 120B MXFP4 MoE    tg32 @ d32768     41.72 ± 0.18
```

</details>

<details>
<summary>vLLM (llama-benchy, single Spark)</summary>

```
model                 test               t/s           e2e_ttft (ms)
openai/gpt-oss-120b   pp2048             4663.70±42    614.72±3
openai/gpt-oss-120b   tg32               33.55±0.05    
openai/gpt-oss-120b   pp2048 @ d4096     3172.51±16    821.97±4
openai/gpt-oss-120b   tg32 @ d4096       32.63±0.02    
openai/gpt-oss-120b   pp2048 @ d8192     2687.84±9     941.50±3
openai/gpt-oss-120b   tg32 @ d8192       31.50±0.10    
openai/gpt-oss-120b   pp2048 @ d16384    2044.47±8     1186.10±5
openai/gpt-oss-120b   tg32 @ d16384      29.55±0.01    
openai/gpt-oss-120b   pp2048 @ d32768    1398.80±4     1659.47±5
openai/gpt-oss-120b   tg32 @ d32768      26.65±0.01    
```

</details>

### Decode Bottleneck Analysis

The decode gap (33 vs 58 tok/s) suggests:

1. **MoE routing overhead?** - Per-token expert selection has fixed cost
2. **Attention decode kernel?** - FA2 decode path efficiency on SM121
3. **Python/IPC overhead?** - vLLM's scheduler, ZMQ, async engine
4. **CUDA graph effectiveness?** - Are graphs actually being used for decode?
5. **KV cache access pattern?** - Memory bandwidth during decode

**Priority**: Profile decode-heavy workload to identify top kernels and CPU time.

---

## Testing Commands

```bash
# Verify FlashInfer import location
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 -c "import flashinfer; print(flashinfer.__file__)"

# Run MXFP4 MoE kernel verification
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 /workspace/verify_mxfp4_moe_kernel.py --quick

# Run full test harness (inside container)
cd /workspace && ./sm121_vllm_test_harness.sh --mode verify --skip-server

# Profile decode performance
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 /workspace/profile_sm121_decode_performance.py --kv-layout HND
```

---

## Summary for AI Agents

When working on this project:

1. **We use CUTLASS via FlashInfer**, not TensorRT-LLM or cuDNN
2. **The correct MoE API is `cutlass_fused_moe`**, not `trtllm_*` functions
3. **SM121 uses FA2** (not FA3), with attention sinks wired up
4. **MXFP4 uses group size 32**, goes through FP8×FP4 kernel internally
5. **Always verify PYTHONPATH** points to local FlashInfer repo
6. **KV cache is HND layout** on SM121/vLLM

If you're debugging crashes or wrong backends, start by verifying which FlashInfer you're actually importing.

