# vLLM Baseline Analysis (SM121 / gpt-oss-120b)

**Status**: TODO - Profile upstream vLLM decode path

## Overview

This document analyzes the baseline vLLM performance for gpt-oss-120b on SM121 (NVIDIA GB10 / DGX Spark). The goal is to understand where time is spent during decode to identify optimization opportunities.

## Configuration

```yaml
model: openai/gpt-oss-120b
quantization: mxfp4
gpu: NVIDIA GB10 (SM121)
vllm_version: <TBD - from upstream/main>
flashinfer_version: <TBD - from upstream/main>
```

---

## Decode Path Profiling

### Kernel Breakdown

| Kernel Category | Time % | Avg (μs) | Instances/Token | Notes |
|-----------------|--------|----------|-----------------|-------|
| MoE FC1 | ? | ? | 60 | |
| MoE FC2 | ? | ? | 60 | |
| Attention GEMV | ? | ? | ? | |
| LayerNorm | ? | ? | ? | |
| Other | ? | ? | ? | |

### Per-Layer Timing

| Layer Type | Avg Time (μs) | % of Total |
|------------|---------------|------------|
| Attention | ? | ? |
| MoE | ? | ? |
| LayerNorm | ? | ? |
| Embedding | ? | ? |

### Memory Bandwidth Utilization

- Theoretical peak: ~XX GB/s
- Measured during decode: ? GB/s
- Utilization: ?%

---

## CPU Thread Analysis

### What Threads Are Busy While GPU Idle

Profile methodology:
```bash
# During inference
top -H -p $(pgrep -f "vllm serve")
```

| Thread | CPU % | Description |
|--------|-------|-------------|
| Main | ? | |
| Scheduler | ? | |
| ZMQ | ? | |
| Other | ? | |

### Poll/Busy Loop Detection

vLLM V1 uses a busy-loop scheduler (`run_busy_loop()` in `engine/core.py`).

**Questions to answer:**
- [ ] Is CPU usage intentional or a bug?
- [ ] What's the CPU usage during active decode vs idle?
- [ ] Does CPU usage affect decode throughput?

### Async Scheduling Mode Effects

| Mode | CPU Usage | Decode t/s | TTFT (ms) |
|------|-----------|------------|-----------|
| sync | ? | ? | ? |
| async | ? | ? | ? |

---

## Scheduler Analysis

### Token Scheduling Overhead

- Time between GPU kernel completion and next kernel launch: ? μs
- Scheduler decision time: ? μs
- Queue processing time: ? μs

### IPC/ZMQ Overhead

- Message serialization: ? μs
- Network round-trip: ? μs
- Deserialization: ? μs

### CUDA Launch Patterns

- Kernel launches per token: ?
- Average launch overhead: ? μs
- CUDA graph capture status: ?

---

## Quantization Overhead

### BF16→FP8 Conversion Timing

For each MoE layer, activations are quantized:

| Operation | Time (μs) | Calls/Token |
|-----------|-----------|-------------|
| mxfp8_quantize | ? | 60 |
| Scale computation | ? | 60 |
| Total | ? | |

### Memory Traffic from Conversions

- Additional reads: ? GB/token
- Additional writes: ? GB/token
- Impact on bandwidth utilization: ?%

---

## Comparison with Other Engines

| Metric | vLLM | llama.cpp | SGLang |
|--------|------|-----------|--------|
| tg32 (tok/s) | 29 | 58 | 52 |
| MoE time % | ? | ? | ? |
| Attn time % | ? | ? | ? |
| CPU usage % | ? | ? | ? |

---

## Key Findings

### Bottlenecks Identified

1. **TBD**: Description
2. **TBD**: Description

### Optimization Opportunities

1. **TBD**: Description
2. **TBD**: Description

---

## Profiling Commands

```bash
# nsys profile for kernel breakdown
nsys profile --stats=true -o vllm_decode \
  python -c "import requests; requests.post('http://localhost:8000/v1/completions', json={'model': 'gpt-oss-120b', 'prompt': 'Hello', 'max_tokens': 32})"

# CPU thread analysis
perf top -p $(pgrep -f "vllm serve") -t

# Memory bandwidth
nvidia-smi dmon -s m -d 1
```

---

## Next Steps

- [ ] Run nsys profile on baseline
- [ ] Document kernel breakdown
- [ ] Profile CPU thread usage
- [ ] Compare with llama.cpp profiling data
- [ ] Identify top 3 optimization targets
