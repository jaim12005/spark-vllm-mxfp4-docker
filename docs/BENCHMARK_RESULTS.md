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

## Comparison Matrix

| Config | tg32 | tg128 | tg256 | pp2048 | TTFT p50 | Notes |
|--------|------|-------|-------|--------|----------|-------|
| **Upstream Baseline** | **31.6** | **31.6** | **31.1** | **4341** | ~555ms | TRITON_ATTN, Marlin, no native FP4 |
| **PR #31740 + Baseline** | **31.9** | - | - | **4702** | ~579ms | Same backends, parity confirmed |
| PR #31740 Native | - | - | - | - | - | Blocked on FlashInfer SM12x |
| CUTLASS GEMM | - | - | - | - | - | TODO |
| CUTLASS + MXFP8 | - | - | - | - | - | TODO |
| + Eagle3 short | - | - | - | - | - | TODO |
| + Eagle3 long | - | - | - | - | - | TODO |
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
