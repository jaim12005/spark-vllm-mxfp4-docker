# MXFP4 v2 Benchmark Results

Live tracking of benchmark results across configurations.

---

## Current Best Results

| Metric | Baseline | Best | Config | Date |
|--------|----------|------|--------|------|
| tg32 (tok/s) | - | - | - | - |
| tg128 (tok/s) | - | - | - | - |
| pp2048 (tok/s) | - | - | - | - |
| TTFT p50 (ms) | - | - | - | - |

---

## Baseline (Marlin, No Spec Decode)

**Status**: TODO - Run after branch setup

### Environment
```yaml
date: YYYY-MM-DD HH:MM:SS
git_sha_vllm: <pending>
git_sha_flashinfer: <pending>
docker_image_hash: <pending>
cuda_version: <pending>
driver_version: <pending>
cudnn_version: <pending>
gpu_clocks: default
gpu_power_mode: default
```

### vLLM Configuration
```yaml
quantization: mxfp4
tensor_parallel_size: 1
max_model_len: 131072
max_num_seqs: 2
max_num_batched_tokens: 8192
enforce_eager: false  # CUDA graphs enabled
enable_prefix_caching: true
load_format: fastsafetensors
served_model: openai/gpt-oss-120b

env_vars:
  VLLM_MXFP4_MOE_KERNEL: marlin
  VLLM_ATTENTION_BACKEND: FLASHINFER
  VLLM_USE_CUDA_GRAPH: 1
```

### Workload Parameters
```yaml
prompt_length: 2048  # tokens
output_lengths: [32, 128]
batch_size: 1
concurrency: 1
temperature: 1.0
top_p: 1.0
top_k: 0
seed: 42
tokenizer: openai/gpt-oss-120b
warmup_requests: 3
test_requests: 10
```

### Results
```yaml
startup_time_s: <pending>

throughput:
  pp2048_tps: <pending>
  tg32_tps: <pending>
  tg128_tps: <pending>

latency:
  ttft_p50_ms: <pending>
  ttft_p99_ms: <pending>
  tpot_p50_ms: <pending>
  tpot_p99_ms: <pending>
  itl_p50_ms: <pending>
  itl_p99_ms: <pending>
  e2e_p50_ms: <pending>
  e2e_p99_ms: <pending>

resource:
  gpu_memory_peak_gb: <pending>
  cpu_usage_percent: <pending>
```

### Kernel Validation
```yaml
expected_kernels:
  - marlin_*
observed_kernels: <pending>
validation: PENDING
```

---

## Comparison Matrix

| Config | tg32 | tg128 | pp2048 | TTFT p50 | TTFT p99 | Memory |
|--------|------|-------|--------|----------|----------|--------|
| Baseline (Marlin) | - | - | - | - | - | - |
| CUTLASS GEMM | - | - | - | - | - | - |
| CUTLASS + MXFP8 | - | - | - | - | - | - |
| + Eagle3 short | - | - | - | - | - | - |
| + Eagle3 long | - | - | - | - | - | - |
| + Eagle3 throughput | - | - | - | - | - | - |

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
