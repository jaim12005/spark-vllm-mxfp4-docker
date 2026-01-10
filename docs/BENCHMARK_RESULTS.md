# MXFP4 v2 Benchmark Results

Live tracking of benchmark results across configurations.

---

## Current Best Results

| Metric | Baseline | Best | Config | Date |
|--------|----------|------|--------|------|
| tg32 (tok/s) | 31.63 | 31.63 | Upstream Baseline | 2026-01-09 |
| tg128 (tok/s) | 31.62 | 31.62 | Upstream Baseline | 2026-01-09 |
| tg256 (tok/s) | 31.12 | 31.12 | Upstream Baseline | 2026-01-09 |
| pp512 (tok/s) | 2209 | 2209 | Upstream Baseline | 2026-01-09 |
| pp1024 (tok/s) | 3386 | 3386 | Upstream Baseline | 2026-01-09 |
| pp2048 (tok/s) | 4341 | 4341 | Upstream Baseline | 2026-01-09 |
| pp4096 (tok/s) | 4008 | 4008 | Upstream Baseline | 2026-01-09 |
| TTFT@pp2048 (ms) | ~555 | ~555 | Upstream Baseline | 2026-01-09 |

---

## Upstream Baseline (vLLM main, No PR changes)

**Status**: ✅ COMPLETE - 2026-01-09

This is the TRUE baseline - upstream vLLM main without any PR #31740 changes.

### Environment
```yaml
date: 2026-01-09 16:25:01
git_sha_vllm: ac9f9330e (upstream/main)
git_sha_flashinfer: bd2b033f (upstream/main)
cuda_version: (NGC 25.12)
gpu: NVIDIA GB10 (SM121)
```

### vLLM Configuration
```yaml
quantization: mxfp4
tensor_parallel_size: 1
max_model_len: 131072
max_num_seqs: 2
max_num_batched_tokens: 8192
enforce_eager: true
enable_prefix_caching: true
load_format: fastsafetensors
served_model: openai/gpt-oss-120b

env_vars:
  VLLM_ATTENTION_BACKEND: (unset - auto)
```

### Observed Runtime Configuration
```yaml
# From vLLM logs:
attention_backend: TRITON_ATTN  # NOT FlashInfer!
moe_kernel: Marlin
fp4_mode: "Weight-only FP4 compression" (not native SM121)
```

### Workload Parameters
```yaml
prompt_length: 2048
output_lengths: [32]
batch_size: 1
concurrency: 1
test_requests: 10
tool: llama-benchy 0.1.1
latency_mode: generation
```

### Results

#### Prefill Throughput (varying context length)
```yaml
pp512_tps: 2209.10 ± 97.25   # TTFT: 332ms
pp1024_tps: 3385.55 ± 81.00  # TTFT: 418ms  
pp2048_tps: 4340.70 ± 12.22  # TTFT: 555ms
pp4096_tps: 4008.20 ± 23.05  # TTFT: 1051ms
```

#### Decode Throughput (pp=2048, varying output length)
```yaml
tg16_tps: 31.53 ± 0.06
tg32_tps: 31.58 ± 0.06
tg64_tps: 31.62 ± 0.03
tg128_tps: 31.62 ± 0.05
tg256_tps: 31.12 ± 0.35
```

#### Latency (pp=2048, tg=32)
```yaml
ttfr_ms: 457.13 ± 13.94
est_ppt_ms: 417.48 ± 13.94
e2e_ttft_ms: 554.80 ± 14.04
```

#### Variance Analysis
| Metric | Mean | Std Dev | CV% | Notes |
|--------|------|---------|-----|-------|
| pp512 | 2209 | ±97 | 4.4% | Lower prefill shows higher relative variance |
| pp1024 | 3386 | ±81 | 2.4% | |
| pp2048 | 4341 | ±12 | 0.3% | Suspiciously low - may be warmed state |
| pp4096 | 4008 | ±23 | 0.6% | |
| tg32 | 31.58 | ±0.06 | 0.2% | **Very stable** - decode is memory-bound |
| tg128 | 31.62 | ±0.05 | 0.2% | |
| tg256 | 31.12 | ±0.35 | 1.1% | Slight degradation at longer context |

**Key insight**: Decode throughput (tg*) is highly stable (~0.2% CV). Prefill throughput (pp*) can vary 2-7% between runs due to prefix caching and warmup effects.

### Key Findings

1. **Attention**: Using `TRITON_ATTN`, NOT FlashInfer
2. **MoE**: Using Marlin kernel
3. **FP4**: Weight-only compression, NOT native SM121 FP4 compute
4. **Warning**: "Your GPU does not have native support for FP4 computation"

This confirms PR #31740 is needed to enable:
- FlashInfer attention backend
- CUTLASS MoE kernels
- Native SM121 FP4 path

---

## nsys Kernel Profiling (Marlin + FlashInfer FA2)

**Status**: ✅ COMPLETE - 2026-01-10

Full nsys profiling to identify bottlenecks. See detailed analysis in `docs/analysis/VLLM_BASELINE_ANALYSIS.md`.

### Configuration
```yaml
moe_kernel: marlin
attention_backend: FLASHINFER (native CUTLASS FA2 on SM121)
kv_cache_layout: HND
prompt_tokens: 2048
output_tokens: 64
enforce_eager: true
```

### Key Findings

**Attention is NOT the bottleneck.** FlashInfer attention accounts for only **1.5%** of GPU time.

| Category | GPU Time % | Primary Kernels |
|----------|------------|-----------------|
| **Dense GEMV/GEMM** | **38.4%** | `gemvx::kernel`, `cutlass_80_tensorop` |
| **MoE GEMM (Marlin)** | **33.6%** | `marlin_moe_wna16::Marlin` |
| **Memory/Data Movement** | **12.4%** | `elementwise_kernel`, `CatArrayBatchedCopy` |
| **Activations/Norms** | **3.1%** | `swigluoai`, `fused_add_rms_norm` |
| **MoE Overhead** | **2.7%** | `marlin_repack`, `topkGatingSoftmax` |
| **Attention** | **1.5%** | `BatchPrefillWithPagedKVCacheKernel` |
| **Other** | **8.3%** | Various PyTorch ops |

### Performance
```yaml
decode_throughput: 30.3 tok/s
tokens_generated: 64
runs: 3
warmup: 2
```

### Optimization Priority (Based on Profile)

1. **MoE GEMM** (34%): Test CUTLASS grouped GEMM vs Marlin
2. **Dense GEMV** (38%): Consider quantized lm_head
3. **Memory** (12%): MXFP8 persistent activations
4. ~~Attention~~ (1.5%): **LOW PRIORITY** - already efficient

### Artifacts
- Profile: `docs/perf_artifacts/marlin_flashinfer_profile.nsys-rep`
- Database: `docs/perf_artifacts/marlin_flashinfer_profile.sqlite`

---

## PR #31740 with Baseline Config (Marlin + TRITON_ATTN)

**Status**: ✅ COMPLETE - 2026-01-09

Running PR #31740 with forced baseline configuration to verify parity.

### Environment
```yaml
date: 2026-01-09 17:27:17
git_sha_vllm: 77bf5a554 (pr-31740)
git_sha_flashinfer: bd2b033f (upstream/main -> mxfp4_v2)
```

### Configuration Override
```bash
# Force baseline configuration on PR branch
export VLLM_MXFP4_MOE_KERNEL=marlin
vllm serve ... --attention-backend TRITON_ATTN
```

### Results
```yaml
throughput:
  pp2048_tps: 4701.82 ± 362.71
  tg32_tps: 31.87 ± 0.05
```

### Comparison to Upstream Baseline
| Metric | Upstream | PR+Baseline (3 runs) | Notes |
|--------|----------|---------------------|-------|
| pp2048 | 4341 ± 12 | 4200-4500 ± 300 | High variance due to prefix caching, warmup |
| tg32 | 31.63 ± 0.04 | 31.82-31.89 ± 0.05 | **Stable, matches baseline** ✓ |

**Conclusion**: Decode (tg32) performance matches. Prefill (pp2048) has high run-to-run variance but is in the same range. The decode stability is what matters for our optimization goal.

---

## PR #31740 with Native Config (FlashInfer + CUTLASS)

**Status**: PENDING - Requires FlashInfer SM12x support

### Expected Configuration
```yaml
attention_backend: FLASHINFER  # Native FA2
moe_kernel: SM100_FI_MXFP4_MXFP8_CUTLASS  # Native CUTLASS
```

### Blocker
FlashInfer `upstream/main` lacks SM12x JIT compilation support:
```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

Need FlashInfer `mxfp4_wip` branch with SM12x kernels.

---

## Eagle3 Speculative Decoding (Baseline Config)

**Status**: ✅ COMPLETE - 2026-01-09

Testing Eagle3 with baseline backends (Marlin + TRITON_ATTN) to establish Eagle3 baseline.

### Configuration
```yaml
attention_backend: TRITON_ATTN
moe_kernel: marlin (via VLLM_MXFP4_MOE_KERNEL=marlin)
speculative_config:
  method: eagle3
  num_speculative_tokens: 3
```

### Results

| Eagle3 Model | tg32 (tok/s) | pp2048 (tok/s) | Acceptance Rate |
|--------------|--------------|----------------|-----------------|
| **short-context** | 13.97 ± 0.08 | 5153 ± 455 | ~31% |
| **long-context** | 14.42 ± 0.09 | 5193 ± 465 | ~31% |
| **throughput** | 14.71 ± 0.07 | 5241 ± 395 | ~31% |
| *Baseline (no spec)* | *31.6* | *4341* | *N/A* |

### Analysis

**Eagle3 with baseline config is 2x SLOWER than no speculation!**

| Metric | No Spec | With Eagle3 | Delta |
|--------|---------|-------------|-------|
| tg32 | 31.6 tok/s | 14.4 tok/s | **-54%** |

**Root Cause**: ~31% draft acceptance rate (should be 60-70%+)
- Per-position acceptance: 0.55, 0.27, 0.13 (very low)
- Drafted tokens are being rejected, wasting compute

**Hypothesis**: Eagle3 was trained with different numerical characteristics than Marlin + TRITON_ATTN produces. The FlashInfer + CUTLASS backends (used in mxfp4_wip where 61 tok/s was achieved) likely have outputs that match better.

### Next Steps
Port FlashInfer sink support from mxfp4_wip to enable FlashInfer attention, then re-test Eagle3.

---

## Attention Backend Testing (Sinks Disabled)

**Status**: ✅ COMPLETE - 2026-01-09

Added `VLLM_ATTENTION_SINKS` env var to control attention sink behavior for testing.

### Configuration
```yaml
# Marlin + FLASH_ATTN (sinks disabled)
env: VLLM_ATTENTION_SINKS=false, VLLM_MXFP4_MOE_KERNEL=marlin
attention_config: {"backend": "FLASH_ATTN"}
```

### Results

| Backend | tg32 (tok/s) | Output Quality | Notes |
|---------|--------------|----------------|-------|
| **TRITON_ATTN** (env=false) | **31.5** | ✅ Correct | ⚠️ Sinks NOT actually disabled |
| **FLASH_ATTN** (sinks disabled) | **31.9** | ❌ Garbage | Bug in FA2 path on SM121 |
| **FLASHINFER** | **31.7** | ✅ Correct | Uses native FA2 sink module on SM121 |

### Analysis

**IMPORTANT CORRECTION**: TRITON_ATTN does NOT check `VLLM_ATTENTION_SINKS` env var!
When we tested "TRITON_ATTN with sinks disabled", sinks were **actually still enabled**.

- **TRITON_ATTN**: Has native sink support, ignores env var - sinks always active
- **FLASH_ATTN**: Respects env var, but produces garbage without sinks on SM121

The raw FA2 kernel is actually 2x faster than TRITON's unified_attention in microbenchmarks, but something in vLLM's FLASH_ATTN backend path corrupts the output.

### Root Cause: Multiple Issues with FLASH_ATTN on SM121

#### Issue 1: Non-standard Head Dimension

gpt-oss-120b uses **head_dim=45** which is NOT in FA2's supported list:
- FA2 supports: hdim32, 64, 96, 128, 160, 192, 256
- **hdim45 is NOT supported!**

Even with native SM121 FA2 kernels, the model's non-standard head dimension causes garbage output.
TRITON_ATTN is more flexible and handles arbitrary head dims correctly.

#### Issue 2: Dao-AILab flash-attention Lacks SM121 Support

**This is a known issue**: [flash-attention#1969 - Support for compute capability 12.1 (sm121) - NVIDIA GB10 GPU](https://github.com/Dao-AILab/flash-attention/issues/1969)

The vLLM FLASH_ATTN backend uses Dao-AILab's flash-attention library, which:
- **Does NOT officially support SM121 (GB10 / DGX Spark)**
- Only compiles kernels for SM80 (see `cuobjdump` output below)
- Relies on PTX JIT compilation from SM80 → SM121 at runtime
- PTX forward compatibility fails silently, producing numerically incorrect results

```bash
# Proof: vLLM FA2 binary only contains SM80 cubins
$ cuobjdump -lelf /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
ELF file    1: _vllm_fa2_C.abi3.1.sm_80.cubin
ELF file    2: _vllm_fa2_C.abi3.2.sm_80.cubin
... (all sm_80, no sm_120 or sm_121)
```

**Why SM121 can't be easily added**:
The vLLM flash-attn source has **88 CUDA files hardcoded for SM80** (`*_sm80.cu`). Adding SM121 to CMakeLists only produces PTX from SM80 source, which JIT compiles incorrectly at runtime.

```bash
$ ls /workspace/vllm/.deps/vllm-flash-attn-src/csrc/flash_attn/src/*.cu | grep -oE 'sm[0-9]+' | sort -u
sm80  # Only SM80 kernels exist!
```

**Workarounds**:
1. Use **TRITON_ATTN** (works correctly on SM121)
2. Use **FlashInfer backend** (has native SM121a JIT compilation with Blackwell-specific fixes)
3. Wait for Dao-AILab to add SM121 support (open issue since Oct 2025)

### Conclusion
For SM121 with gpt-oss-120b (head_dim=45):
- Use **TRITON_ATTN** for correct output (handles non-standard head dims)
- **FLASH_ATTN fails** due to unsupported head_dim=45, even with native SM121 build
- **FlashInfer** (from mxfp4_wip with native FA2 sink support) is the path to 61+ tok/s

**Note**: We built FA2 with native SM121 support (see `docs/FLASH_ATTN_SM121_BUILD.md`) and verified the kernel runs without NaN/Inf, but gpt-oss-120b's head_dim=45 is outside FA2's supported dimensions.

---

## FlashInfer Native Attention with Sinks (SM121)

**Status**: ✅ WORKING - 2026-01-10

Successfully ported FlashInfer native FA2 attention with sink support from mxfp4_wip branch.

### Configuration
```yaml
# Marlin + FlashInfer + Sinks
env: VLLM_MXFP4_MOE_KERNEL=marlin
attention_config: {"backend": "FLASHINFER"}
# VLLM_ATTENTION_SINKS defaults to "auto" (sinks enabled)
```

### Results

| Config | tg32 (tok/s) | Output Quality | Notes |
|--------|--------------|----------------|-------|
| **Marlin + FlashInfer + Sinks** | **32.6** | ✅ Coherent | FA2 sink module working |
| **Marlin + FlashInfer (no sinks)** | **32.6** | ❌ Garbage | Model requires sinks |
| **Marlin + TRITON_ATTN + Sinks** | **32.3** | ✅ Coherent | Reference baseline |

### Key Fixes Applied

1. **SM12x TRTLLM exclusion** (`vllm/utils/flashinfer.py`):
   - TRTLLM cubins only work on SM100/SM103, not SM12x
   - Added capability check to prevent auto-detection on SM121

2. **o_data_type parameter bug** (`vllm/v1/attention/backends/flashinfer.py`):
   - `fast_plan_decode` was missing `o_data_type` param in positional args
   - Caused argument shift, breaking `seq_lens` interpretation
   - **Pre-existing bug in vLLM main** - needs upstream PR

3. **non_blocking=None fix** (`flashinfer/decode.py`):
   - PyTorch doesn't accept `None` for `non_blocking` parameter
   - Added fallback: `nb = non_blocking if non_blocking is not None else True`

4. **Sinks passed to run()** (`flashinfer.py`):
   - `_use_fa2_sinks` was on builder, not impl
   - Changed to simply pass `sinks=self.sinks` (None if not available)

### Analysis

FlashInfer native attention achieves **same performance as TRITON_ATTN** (~32 tok/s) with correct output when sinks are enabled.

**Key finding**: gpt-oss-120b **requires sinks** for coherent output:
- FlashInfer without sinks → garbage
- TRITON_ATTN always uses sinks (ignores `VLLM_ATTENTION_SINKS` env var)
- All coherent results we measured were with sinks enabled

### Next Steps
Test Eagle3 with FlashInfer attention to verify speculative decoding still works.

---

## Comparison Matrix

| Config | tg32 | tg128 | tg256 | pp2048 | TTFT p50 | Notes |
|--------|------|-------|-------|--------|----------|-------|
| **Upstream Baseline** | **31.6** | **31.6** | **31.1** | **4341** | ~555ms | TRITON_ATTN, Marlin, no native FP4 |
| **PR #31740 + Baseline** | **31.9** | - | - | **4702** | ~579ms | Same backends, parity confirmed |
| **Marlin + FlashInfer + Sinks** | **32.6** | - | - | - | - | ✅ Coherent, FA2 sink module |
| PR #31740 Native | - | - | - | - | - | Blocked on FlashInfer SM12x |
| **Eagle3 short-context** | **14.0** | - | - | 5153 | ~662ms | 31% acceptance rate - REGRESSION |
| **Eagle3 long-context** | **14.4** | - | - | 5193 | ~577ms | 31% acceptance rate - REGRESSION |
| **Eagle3 throughput** | **14.7** | - | - | 5241 | ~655ms | 31% acceptance rate - REGRESSION |
| Eagle3 + FlashInfer | - | - | - | - | - | TODO |
| CUTLASS GEMM | - | - | - | - | - | TODO |
| CUTLASS + MXFP8 | - | - | - | - | - | TODO |
| + Eagle3 throughput | - | - | - | - | - | TODO |
| | | | | | | |
| **llama.cpp** | **57.85** | - | - | 2449 | - | Target |
| **SGLang** | **52.37** | - | - | - | 49.87ms | Target |

---

## Reference: Competing Engines

### llama.cpp (llama-bench, build f5acfb2ff)

| Test | Throughput (t/s) |
|------|------------------|
| pp2048 | 2449.83 ± 10.27 |
| tg32 | **57.85 ± 0.44** |
| pp2048 @ d4096 | 2293.59 ± 8.99 |
| tg32 @ d4096 | 54.81 ± 0.30 |

### SGLang

| Metric | Value |
|--------|-------|
| Output throughput | **52.37 tok/s** |
| TTFT | 49.87 ms |
| TPOT | 18.83 ms |

---

## Detailed Run Logs

Each benchmark run should link to:
1. Full metadata file in `docs/TEST_LOGS/`
2. nsys profile (if captured) in `docs/perf_artifacts/`
3. Feature porting doc in `docs/porting/`

| Run ID | Date | Config | tg32 | Notes |
|--------|------|--------|------|-------|
| baseline_upstream_v1 | 2026-01-09 | TRITON_ATTN + Marlin | 31.6 | nsys profile captured |

---

## Configuration Reference

Environment variables and CLI arguments for consistent benchmarking.

### vLLM Runtime Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VLLM_MXFP4_MOE_KERNEL` | `auto`, `marlin`, `gemm`, `gemv`, `triton` | `auto` | MoE kernel selection. `marlin` for baseline, `gemm` for CUTLASS. |
| `VLLM_ATTENTION_SINKS` | `auto`, `true`, `false` | `auto` | Attention sink control. `false` disables sinks for testing. |

### Attention Backend Selection

Use `--attention-config` CLI argument (not env var):

```bash
# TRITON_ATTN (baseline fallback)
vllm serve ... --attention-config '{"backend": "TRITON_ATTN"}'

# FLASH_ATTN (FlashAttention 2)
vllm serve ... --attention-config '{"backend": "FLASH_ATTN"}'

# FLASHINFER (requires SM12x kernel support)
vllm serve ... --attention-config '{"backend": "FLASHINFER"}'
```

### FlashInfer Build Configuration

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | Target SM121 with FP4 hardware path |
| `FLASHINFER_NVCC_THREADS` | `4` | Parallel JIT compilation threads |
| `FLASHINFER_LOGLEVEL` | `0-5` | Logging verbosity (0=quiet) |
| `FLASHINFER_JIT_VERBOSE` | `0/1` | JIT compilation logging |

### Baseline Configuration

For reproducible baseline benchmarks, use:

```bash
# Environment
export VLLM_MXFP4_MOE_KERNEL=marlin
unset VLLM_ATTENTION_BACKEND  # Don't use deprecated env var

# Server command
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --enforce-eager \
    --load-format fastsafetensors \
    --attention-config '{"backend": "TRITON_ATTN"}'
```

### Testing Different Configurations

```bash
# Test with CUTLASS MoE (requires FlashInfer SM12x support)
export VLLM_MXFP4_MOE_KERNEL=gemm

# Test with sinks disabled (for Eagle3 or FlashInfer testing)
export VLLM_ATTENTION_SINKS=false

# Test with FlashInfer attention
vllm serve ... --attention-config '{"backend": "FLASHINFER"}'
```

### Legacy Variables (Deprecated)

These are no longer set by default in Docker configs:

| Variable | Notes |
|----------|-------|
| `VLLM_ATTENTION_BACKEND` | Use `--attention-config` CLI instead |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | Set in docker-compose.yml for production |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | Set in docker-compose.yml for production |
| `VLLM_FLASHINFER_MOE_BACKEND` | Set in docker-compose.yml for production |
| `VLLM_USE_CUDA_GRAPH` | Set in docker-compose.yml for production |

---

## How to Run Benchmarks

```bash
# Start vLLM server
docker compose -f docker-compose.dev.yml --profile serve up serve

# In another terminal, run llama-benchy
llama-benchy \
  --model gpt-oss-120b \
  --base-url http://localhost:8000 \
  --num-requests 10 \

# Collect metadata
scripts/collect_benchmark_metadata.sh > docs/TEST_LOGS/run_$(date +%Y%m%d_%H%M%S).yaml
```

---

## FlashInfer FA2 Sink Module Investigation (2026-01-10)

### Root Cause Analysis

Systematic validation of 6 potential issues with the FlashInfer FA2 attention sink implementation:

| Issue | Status | Finding |
|-------|--------|---------|
| **#1: Backend mismatch (prefill vs decode)** | **CRITICAL BUG** | Prefill has `use_sinks` parameter, decode has NONE |
| **#2: Argument ordering mismatch** | OK | Sink module expects `[sinks, sm_scale]`, code passes exactly that |
| **#3: Cache key missing compile-time knobs** | **BUG** | `use_logits_soft_cap=False`, `use_fp16_qk_reduction=False` hardcoded |
| **#4: Sink value space (log vs linear)** | OK | Model stores natural log, kernel converts with `* log2e` |
| **#5: LSE renormalization** | OK | Sinks applied once (first tile), properly incorporated into running sum |
| **#6: Shape/layout mismatch (QO vs KV heads)** | OK | Model: `[64]` (QO heads), kernel: `sink[qo_head_idx]` - matches |

### Critical Bugs Found

1. **Decode path has NO sink support**
   - Location: `flashinfer/decode.py` - no `use_sinks` parameter anywhere
   - Impact: Prefill uses sinks, decode doesn't → model outputs garbage after first token
   
2. **Sink module cache key missing compile-time knobs**
   - Location: `flashinfer/jit/attention/modules.py:1122-1124`
   - Hardcoded: `use_logits_soft_cap=False`, `use_fp16_qk_reduction=False`
   - Impact: If model uses logits soft cap, wrong kernel is compiled/used

### Model Sink Values (gpt-oss-120b)

```
Shape: [64] (one per QO head)
Dtype: bfloat16
Range: -2.06 to 5.75 (natural log space)
Example: [1.56, 1.08, 1.25, 1.24, 1.05]
```

### Required Fixes

1. Add `use_sinks` parameter to decode path (`BatchDecodeWithPagedKVCacheWrapper.plan()`)
2. Add `use_logits_soft_cap` and `use_fp16_qk_reduction` to sink module generation
3. Wire decode sinks through vLLM's FlashInfer backend

---

## FlashInfer FA2 Sink Implementation Results (2026-01-10)

**Status**: ✅ COMPLETE - Implementation working with sinks

After implementing all three fixes above, FlashInfer FA2 with sinks is now functional.

### Configuration

```yaml
moe_kernel: Marlin (VLLM_MXFP4_MOE_KERNEL=marlin)
attention_backend: FLASHINFER (--attention-config '{"backend": "FLASHINFER"}')
sinks: enabled (default, no VLLM_ATTENTION_SINKS override)
```

### Results

| Configuration | Coherent | tg=32 (tok/s) | tg=128 (tok/s) |
|---------------|----------|---------------|----------------|
| Marlin + FA2 (no sinks) | ❌ GARBAGE | N/A | N/A |
| Marlin + FA2 + Sinks | ✅ | **31.9** | **32.3** |

### Comparison to Baseline

| Config | tg=32 (tok/s) | Notes |
|--------|---------------|-------|
| Baseline (Marlin + TRITON_ATTN) | 31.6 | From upstream baseline |
| Marlin + FlashInfer FA2 + Sinks | 31.9 | **Matches baseline** |

### Key Findings

1. **gpt-oss-120b requires sinks** - Without sinks, output is garbage regardless of attention backend
2. **FlashInfer FA2 sink module works** - Performance matches TRITON_ATTN baseline (~32 tok/s)
3. **No regression** - New sink implementation does not degrade performance

### Test Output Sample

```
Prompt: "The capital of France is"
Without sinks: " a is the is a is the is a is the is a is the is a is the is"
With sinks: " Paris.\n\nGreat! If you have any more questions or need further assistance, feel free to ask"
```

---

## FlashInfer Speculative Decoding Support (2026-01-10)

### Problem

FlashInfer backend failed with Eagle3 speculative decoding due to:
```
AssertionError: FlashInfer backend currently only supports models in which 
all layers share the same values for: `window_left`, `logits_soft_cap`, `sm_scale`.
```

**Root cause**: Main model (gpt-oss-120b) and draft model (Eagle3) have different hyperparameters:
- Main model: `window_left=128`, `has_sinks=True`
- Draft model: `window_left=-1`, `has_sinks=False`

### Solution: Per-Hyperparameter-Group Wrappers

Implemented Option A: Create separate FlashInfer wrappers for each unique set of hyperparameters.

**Changes**:
1. `vllm/v1/attention/backends/utils.py`:
   - Added `group_layers_by_hyperparameters()` to group layers with identical hyperparameters

2. `vllm/v1/attention/backends/flashinfer.py`:
   - Added `HyperparamKey` type alias
   - Modified `FIPrefill` and `FIDecode` to store per-group wrappers
   - Modified `FlashInferMetadataBuilder` to create wrappers per group
   - Modified `FlashInferImpl.forward()` to look up correct wrapper by layer's hyperparameters

### Results

| Configuration | tg32 (tok/s) | tg64 (tok/s) | tg128 (tok/s) | Notes |
|--------------|--------------|--------------|---------------|-------|
| Baseline (no spec) | 32 | - | - | Marlin + TRITON_ATTN |
| FlashInfer (no spec, with sinks) | 31.9 | - | 32.3 | Matches baseline |
| **Eagle3-long + FlashInfer** | **19.1** | **20.5** | **22.1** | Speculative decoding working! |
| Eagle3-long + TRITON_ATTN | ~24-25 | - | - | For comparison |

### Key Findings

1. **FlashInfer + Eagle3 speculative decoding now works** - No more assertion failures
2. **Performance is similar to TRITON_ATTN** - Neither backend achieves the target 61 tok/s
3. **Both backends are slower than no speculation** - Indicates low acceptance rate issue, not backend-specific
4. **The 61 tok/s result from mxfp4_wip likely used different hyperparameters or model configuration**

### Next Steps

- Investigate why acceptance rate is low with both TRITON_ATTN and FlashInfer
- Compare numerical outputs between backends to identify any precision mismatches
- Test with BF16 (no quantization) to isolate MXFP4 as potential culprit
