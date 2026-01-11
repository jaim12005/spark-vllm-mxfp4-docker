# AGENTS.md - Project Context for AI Assistants

## Mission

Make **vLLM the fastest inference engine for gpt-oss-120b** on NVIDIA GB10 (SM121), outperforming SGLang and llama.cpp by leveraging:

- **Native FP4 hardware features** (SM121 block-scaled MMA)
- **MXFP4 quantized weights** (4-bit weights, group size 32)
- **FlashAttention-2** for attention

### Targets to Beat

| Engine | pp2048 (t/s) | tg32 (t/s) |
|--------|--------------|------------|
| **llama.cpp** | 2449 | **58** |
| **SGLang** | - | **52** |
| **vLLM (baseline)** | 4808 ✓ | 29 ❌ |

**Goal**: Achieve ≥52 tok/s decode while maintaining prefill performance.

---

## Implementation Stack

### WE ARE USING

| Component | Implementation | Location |
|-----------|----------------|----------|
| **MoE GEMM** | CUTLASS SM120/121 grouped GEMM | `flashinfer/fused_moe/` |
| **Attention** | FlashAttention-2 (FA2) | `flashinfer/prefill.py`, `flashinfer/decode.py` |
| **FP4 Quantization** | FlashInfer `mxfp4_quantize()` | `flashinfer/fp4_quantization.py` |
| **Framework** | vLLM with FlashInfer backend | Local repos mounted in Docker |

### WE ARE NOT USING

| ❌ Component | Why Not |
|-------------|---------|
| **TensorRT-LLM** | We use CUTLASS directly |
| **trtllm_fp4_block_scale_moe** | SM100-only, crashes on SM121 |
| **FA3** | SM90 only; SM121 uses FA2 |
| **nvfp4** | We use MXFP4 (group size 32), not NVFP4 (group size 16) |

---

## Architecture: SM121 (GB10)

- **Compute Capability**: 12.1
- **Memory**: 120GB
- **Block-scaled MMA**: FP8×FP4 (not BF16×FP4 directly)
- **MXFP4 Path**: BF16 activations → quantize to FP8 → FP8×FP4 kernel

---

## Repository Layout

```
~/projects/
├── flashinfer/          # Local FlashInfer (CUTLASS kernels)
├── vllm/                # Local vLLM (uses FlashInfer backend)
└── ai/mxfp4/            # THIS REPO - Docker config + benchmarking
    ├── AGENTS.md        # This file
    ├── docker-compose.dev.yml
    ├── docs/
    │   ├── MXFP4_V2_PLAN.md      # Comprehensive optimization plan
    │   ├── BENCHMARK_RESULTS.md   # Live benchmark tracking
    │   ├── FEATURE_MATRIX.md      # Feature status and env vars
    │   ├── UPSTREAM_TODOS.md      # FlashInfer improvement opportunities
    │   ├── analysis/              # Engine comparisons
    │   ├── investigations/        # Historical analysis
    │   └── porting/               # Feature porting docs
    └── scripts/
        ├── setup_mxfp4_v2.sh      # Branch setup
        ├── benchmark_matrix.py    # Systematic testing
        └── test_level*.sh         # Testing protocol
```

---

## Docker Environment

### Starting the Dev Container

```bash
cd ~/projects/ai/mxfp4
docker compose -f docker-compose.dev.yml up -d
docker exec -it vllm-dev bash
```

### Volume Mounts

Local repos are mounted into the container:
- `~/projects/vllm` → `/workspace/vllm`
- `~/projects/flashinfer` → `/workspace/flashinfer`

### Critical: PYTHONPATH

FlashInfer requires explicit PYTHONPATH to use local repo over installed wheel:

```bash
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
```

### Verify Correct Imports

```bash
docker exec vllm-dev python3 -c "import flashinfer; print(flashinfer.__file__)"
# ✓ Should print: /workspace/flashinfer/flashinfer/__init__.py
# ❌ NOT: /usr/local/lib/python3.x/site-packages/flashinfer/...
```

---

## vLLM Server Configuration

### Current Dev Command

```bash
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --enable-prefix-caching \
    --load-format fastsafetensors
```

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `VLLM_MXFP4_BACKEND` | **Unified backend selector** (recommended) |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` or `FLASH_ATTN` |
| `FLASHINFER_LOGLEVEL` | 0-5 for debug logging |

### MXFP4 Backend Options

**CLI argument (recommended):**
```bash
vllm serve ... --quantization mxfp4 --mxfp4-backend CUTLASS
```

**Environment variable:**
```bash
export VLLM_MXFP4_BACKEND=CUTLASS
```

| Backend | Description | SM Support |
|---------|-------------|------------|
| `auto` | Hardware-based auto-selection | - |
| `MARLIN` | Marlin dequant→BF16 | All GPUs |
| `CUTLASS` | FlashInfer CUTLASS FP8×FP4 | SM100, SM12x |
| `TRITON` | OpenAI Triton | SM90-SM100 |
| `TRTLLM` | TRT-LLM BF16×FP4 | SM100 only |
| `TRTLLM_MXFP8` | TRT-LLM FP8×FP4 | SM100 only |

### SM12x Configuration (GB10)

For SM12x native CUTLASS MXFP4 (once Phase 2 is complete):
```bash
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --mxfp4-backend CUTLASS
```

**Data Flow**:
1. vLLM receives BF16 activations from model
2. vLLM calls `mxfp8_quantize()` to convert BF16 → FP8
3. FlashInfer CUTLASS kernel executes FP8×FP4 GEMM on SM12x tensor cores

**Current Workaround** (until Phase 2 is complete):
```bash
export VLLM_MXFP4_BACKEND=MARLIN
vllm serve openai/gpt-oss-120b --quantization mxfp4
```

### Deprecated Variables

The following are deprecated and will show warnings:
- `VLLM_MXFP4_MOE_KERNEL` → Use `VLLM_MXFP4_BACKEND`
- `VLLM_MXFP4_ACTIVATION` → Use `VLLM_MXFP4_BACKEND`
- `VLLM_USE_FLASHINFER_MOE_MXFP4_*` → Use `VLLM_MXFP4_BACKEND`

See `docs/FEATURE_MATRIX.md` for full list.

---

## Correct API Usage

### MoE GEMM on SM121

```python
# ✓ CORRECT: Use cutlass_fused_moe
from flashinfer.fused_moe import cutlass_fused_moe

output = cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_indices,
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=torch.bfloat16,
    activation_type=ActivationType.Swiglu,
)
```

```python
# ❌ WRONG: Do NOT use on SM121
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe  # SM100 ONLY!
```

### MXFP4 Quantization

```python
# ✓ CORRECT: MXFP4 (group size 32)
from flashinfer import mxfp4_quantize
weight_fp4, weight_scale = mxfp4_quantize(weight_bf16)
```

```python
# ❌ WRONG: NVFP4 (group size 16) - different format
from flashinfer import nvfp4_quantize
```

---

## Development Workflow

### Editing Kernels

1. Edit on host: `vim ~/projects/flashinfer/include/flashinfer/moe/...`
2. Clear JIT cache (see below for selective clearing)
3. Run test - FlashInfer JIT recompiles automatically

### Editing Python

Changes are immediate - no cache clearing needed.

### FlashInfer JIT Cache Structure

The JIT cache is organized by architecture and operation type:

```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120/           # SM120 MoE GEMM kernels (~10min to rebuild all)
│   ├── moe_gemm_kernels_bf16_fp4.cuda.o   # MXFP4 (BF16 act)
│   ├── moe_gemm_kernels_fp16_fp4.cuda.o   # MXFP4 (FP16 act)
│   ├── moe_gemm_kernels_fp4_fp4.cuda.o    # NVFP4
│   └── fused_moe_120.so                   # Final library
├── fp4_quantization_121/    # FP4 quantization ops
└── ...                      # Attention, other ops
```

### Selective Cache Clearing (Faster Testing)

**Full clear** (slow - rebuilds everything ~10+ min):
```bash
rm -rf ~/.cache/flashinfer/0.6.0/121a/cached_ops/*
```

**MoE-only clear** (fast - rebuilds only MoE kernels ~2-5 min):
```bash
# Clear ONLY SM120 MoE GEMM kernels
rm -rf ~/.cache/flashinfer/0.6.0/121a/cached_ops/fused_moe_120/
```

**Inside Docker**:
```bash
docker exec vllm-dev rm -rf /root/.cache/flashinfer/0.6.0/121a/cached_ops/fused_moe_120/
```

Use selective clearing when iterating on MoE launcher code to save significant build time.

### After Container Recreation

```bash
cd /workspace/flashinfer && uv pip install --no-build-isolation -e .
cd /workspace/vllm && python3 use_existing_torch.py && \
  uv pip install -r requirements/build.txt && \
  uv pip install --no-build-isolation -e .
```

---

## Testing Protocol

Run in order after each feature port:

| Level | Test | Command |
|-------|------|---------|
| 1 | Smoke | `scripts/test_level1_smoke.sh` |
| 1.5 | Kernel Validation | `scripts/test_level1.5_kernel_validation.sh` |
| 2 | Correctness | `scripts/test_level2_correctness.sh` |
| 3 | Stress | `scripts/test_level3_stress.sh` |
| 4 | Benchmark | `scripts/test_level4_benchmark.sh` |
| 5 | Regression | `scripts/test_level5_regression.sh` |
| 6 | Combinatorial | `scripts/test_level6_matrix.sh` |

---

## Quick Reference

### Verify Kernel Path

```bash
# Check which MoE kernel is being used
FLASHINFER_LOGLEVEL=3 vllm serve ... 2>&1 | grep -i "moe\|kernel\|cutlass"
```

### Clear All Caches

```bash
# On host (outside Docker):
rm -rf ~/projects/ai/mxfp4/.cache/flashinfer/*
rm -rf ~/projects/ai/mxfp4/.cache/vllm/*

# Inside Docker container:
rm -rf ~/.cache/flashinfer/*
rm -rf ~/.cache/vllm/*
```

**Note:** The cache directories are Docker mounts. Delete the *contents* (`/*`), not the directories themselves, or Docker bind mounts will break.

### Run Baseline Benchmark

```bash
llama-benchy \
  --model gpt-oss-120b \
  --endpoint http://localhost:8000 \
  --prompt-length 2048 \
  --output-lengths 32,128 \
  --num-requests 10
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/MXFP4_V2_PLAN.md` | Full optimization plan and methodology |
| `docs/BENCHMARK_RESULTS.md` | Live benchmark tracking |
| `docs/FEATURE_MATRIX.md` | Feature status and configuration |
| `docs/UPSTREAM_TODOS.md` | FlashInfer improvement opportunities |
| `docs/analysis/` | llama.cpp, SGLang, vLLM analysis |
| `docs/porting/` | Per-feature porting documentation |

---

## Summary for AI Agents

1. **We use CUTLASS via FlashInfer**, not TensorRT-LLM
2. **Correct MoE API is `cutlass_fused_moe`**
3. **SM121 uses FA2** (not FA3)
4. **MXFP4 uses group size 32**, goes through FP8×FP4 kernel
5. **Always verify PYTHONPATH** points to local repos
6. **KV cache is HND layout** on SM121

If debugging crashes: verify which FlashInfer you're importing first.
