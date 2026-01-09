# MXFP4 v2 Benchmark Results

Live tracking of benchmark results across configurations.

---

## Current Best Results

| Metric | Baseline | Best | Config | Date |
|--------|----------|------|--------|------|
| tg32 (tok/s) | 32.14 | 32.14 | Upstream Baseline | 2026-01-09 |
| tg128 (tok/s) | - | - | - | - |
| pp2048 (tok/s) | 4340 | 4340 | Upstream Baseline | 2026-01-09 |
| TTFT p50 (ms) | ~589 | ~589 | Upstream Baseline | 2026-01-09 |

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
```yaml
throughput:
  pp2048_tps: 4340.18 ± 818.53
  tg32_tps: 32.14 ± 0.06

latency:
  ttfr_ms: 493.88 ± 166.12
  est_ppt_ms: 432.84 ± 166.12
  e2e_ttft_ms: 588.74 ± 161.70
```

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

## Baseline with PR #31740 (Marlin, No Spec Decode)

**Status**: TODO - Run next

### Environment
```yaml
date: <pending>
git_sha_vllm: 77bf5a554 (pr-31740 / mxfp4_v2)
git_sha_flashinfer: bd2b033f (upstream/main)
```

### Expected Changes from Upstream
- FlashInfer attention enabled
- CUTLASS MoE available
- Native SM121 FP4 path

### Results
```yaml
throughput:
  pp2048_tps: <pending>
  tg32_tps: <pending>
  tg128_tps: <pending>
```

### Kernel Validation
```yaml
expected_kernels:
  - marlin_* OR cutlass_*
observed_kernels: <pending>
validation: PENDING
```

---

## Comparison Matrix

| Config | tg32 | tg128 | pp2048 | TTFT p50 | Notes |
|--------|------|-------|--------|----------|-------|
| **Upstream Baseline** | **32.14** | - | **4340** | ~589ms | TRITON_ATTN, Marlin, no native FP4 |
| PR #31740 Baseline | - | - | - | - | TODO |
| CUTLASS GEMM | - | - | - | - | TODO |
| CUTLASS + MXFP8 | - | - | - | - | TODO |
| + Eagle3 short | - | - | - | - | TODO |
| + Eagle3 long | - | - | - | - | TODO |
| + Eagle3 throughput | - | - | - | - | TODO |
| | | | | | |
| **llama.cpp** | **57.85** | - | 2449 | - | Target |
| **SGLang** | **52.37** | - | - | 49.87ms | Target |

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
| - | - | - | - | - |

---

## How to Run Benchmarks

```bash
# Start vLLM server
docker compose -f docker-compose.dev.yml --profile serve up serve

# In another terminal, run llama-benchy
llama-benchy \
  --model gpt-oss-120b \
  --endpoint http://localhost:8000 \
  --prompt-length 2048 \
  --output-lengths 32,128 \
  --num-requests 10 \
  --warmup 3

# Collect metadata
scripts/collect_benchmark_metadata.sh > docs/TEST_LOGS/run_$(date +%Y%m%d_%H%M%S).yaml
```
