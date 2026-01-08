# SM121 MXFP4 MoE GEMM + FA2 Sinks Testing Guide

This document provides comprehensive instructions for verifying and optimizing MXFP4 MoE GEMM + FA2 attention sinks on NVIDIA GB10 (SM121) using FlashInfer and vLLM.

## Location

This testing harness lives in `~/projects/ai/mxfp4/` alongside the Docker configuration.

## Directory Layout
```
~/projects/
├── flashinfer/     # FlashInfer repo → mounted at /workspace/flashinfer
├── vllm/           # vLLM repo → mounted at /workspace/vllm
└── ai/mxfp4/       # Docker config + testing harness
    ├── docker-compose.dev.yml   # Development container
    ├── docker-compose.yml       # Production container
    ├── Dockerfile.dev
    ├── SM121_TESTING_README.md  # This file
    └── scripts/
        ├── sm121_vllm_test_harness.sh           # Main test harness
        ├── verify_sm121_mxfp4_moe_fa2_sinks.py  # FlashInfer verification
        └── profile_sm121_decode_performance.py  # Performance profiler
```

### Container Status
```bash
# Check container is running
docker ps --filter "name=vllm-dev"

# Enter container
docker exec -it vllm-dev bash
```

## Status Summary

### Backend Configuration (from docker-compose.dev.yml)
| Component | Environment Variable | Value |
|-----------|---------------------|-------|
| Attention | `VLLM_ATTENTION_BACKEND` | `FLASHINFER` |
| MoE GEMM | `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | `1` |
| MoE Backend | `VLLM_FLASHINFER_MOE_BACKEND` | `throughput` |
| FP4 MoE | `VLLM_USE_FLASHINFER_MOE_FP4` | `1` |
| CUDA Graph | `VLLM_USE_CUDA_GRAPH` | `1` |

### Key Files
| File | Purpose |
|------|---------|
| `scripts/sm121_vllm_test_harness.sh` | Crash-proof vLLM testing harness |
| `scripts/verify_sm121_mxfp4_moe_fa2_sinks.py` | FlashInfer-specific verification |
| `scripts/profile_sm121_decode_performance.py` | Decode performance profiling |
| `benchmarks/sm121_mxfp4_moe_gemm_bench.py` | MoE GEMM microbenchmark |

## Quick Start

### 1. Enter Docker Container

```bash
# Enter the running vllm-dev container
docker exec -it vllm-dev bash

# Or start it if not running
cd ~/projects/ai/mxfp4
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml exec dev bash
```

### 2. Verify FlashInfer Installation (inside container)

```bash
# Check FlashInfer config
python -m flashinfer show-config

# Run verification tests (from host mxfp4 dir)
cd ~/projects/ai/mxfp4
python scripts/verify_sm121_mxfp4_moe_fa2_sinks.py --verbose
```

### 3. Run Full Test Harness (from host)

```bash
cd ~/projects/ai/mxfp4

# Run all tests
./scripts/sm121_vllm_test_harness.sh --mode all

# Run specific modes
./scripts/sm121_vllm_test_harness.sh --mode verify    # Backend verification only
./scripts/sm121_vllm_test_harness.sh --mode stress    # Stress tests
./scripts/sm121_vllm_test_harness.sh --mode benchmark # Performance benchmarks

# Use existing server (skip starting)
./scripts/sm121_vllm_test_harness.sh --mode verify --skip-server
```

### 4. Start vLLM Server Separately (optional)

```bash
# Use the serve profile
cd ~/projects/ai/mxfp4
docker compose -f docker-compose.dev.yml --profile serve up serve
```

## Test Harness Features

### A. Crash-Proof Testing (Requirements A1-A3)

The harness implements robust crash handling:

1. **Never pipe through grep** - Logs are written directly to files, filtering done afterwards
2. **Line-buffered output** - Uses `stdbuf -oL -eL` to prevent log loss
3. **Process group cleanup** - Uses `setsid` and `kill -- -$PGID` to clean up all child processes
4. **Zombie detection** - Automatically checks for defunct processes after each run
5. **Timeout handling** - Configurable timeouts with graceful shutdown

### Crash Diagnostics Collection

When a crash occurs, the harness automatically collects:

```
crash_diagnostics_HHMMSS/
├── server_log.txt          # Full server log
├── dmesg.log               # Kernel messages (GPU Xid errors)
├── nvidia_smi.log          # GPU memory/utilization state
├── processes.log           # Process tree snapshot
└── flashinfer_api.log      # FlashInfer API calls (if logging enabled)
```

### Debug Mode

For catching illegal memory access:

```bash
DEBUG_MODE=1 ./scripts/sm121_vllm_test_harness.sh --mode verify

# This enables:
# CUDA_LAUNCH_BLOCKING=1
# TORCH_SHOW_CPP_STACKTRACES=1
# NCCL_ASYNC_ERROR_HANDLING=1
# NCCL_DEBUG=INFO
```

## B. Functional Verification

### Backend Detection

The harness creates a `backend_summary.log` with:

```
=== Backend Summary ===
--- Attention Backend ---
Attention: FLASHINFER detected
FA2: Enabled
Sinks: Enabled (if configured)

--- MoE Backend ---
MoE: MXFP4 CUTLASS detected
Architecture: SM120/SM121 path

--- PYTHONPATH Check ---
Local FlashInfer: YES (in PYTHONPATH)
```

### Smoke Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Short prompt | Basic request | Valid non-empty response |
| NaN/Inf check | Numerical stability | No NaN/Inf in logits |
| Concurrency | 4 parallel requests | All succeed |
| Long context | 4k token prefill | No timeout/error |
| Memory stability | 5 repeated runs | Memory growth < 1GB |

### Stress Tests

```bash
# Configure stress test parameters
STRESS_REQUESTS=200 \
STRESS_CONCURRENCY=8 \
./scripts/sm121_vllm_test_harness.sh --mode stress
```

Pass criteria:
- 0 failed requests
- 0 5xx HTTP errors
- No zombie processes
- Memory growth < 5GB

## C. Performance Profiling

### Baseline Measurement

```bash
# Run MoE GEMM microbenchmark
python benchmarks/sm121_mxfp4_moe_gemm_bench.py --regime both --require-fp4

# Run decode profiler
python scripts/profile_sm121_decode_performance.py --output-dir profile_outputs
```

### nsys Profiling

```bash
# Profile decode path
nsys profile -o decode_profile \
    python scripts/profile_sm121_decode_performance.py --no-torch-profiler

# View in NVIDIA Nsight Systems
nsys-ui decode_profile.nsys-rep
```

### Key Metrics to Measure

| Metric | Target | Description |
|--------|--------|-------------|
| Prefill throughput (pp2048) | - | Tokens/second for 2048 token prefill |
| Decode throughput (tg32) | - | Tokens/second for 32 token generation |
| TTFT | < 100ms | Time to first token |
| Decode latency | < 20ms/token | Per-token decode time |
| MoE GEMM (batch 1) | < 1ms | Single-token MoE latency |

## D. Environment Variable Reference

### Docker Compose Environment (from docker-compose.dev.yml)

```yaml
environment:
  # MXFP4 configuration
  - VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
  - VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0
  - VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=0
  - VLLM_FLASHINFER_MOE_BACKEND=throughput
  - VLLM_USE_FLASHINFER_MOE_FP4=1
  
  # Performance
  - VLLM_ATTENTION_BACKEND=FLASHINFER
  - VLLM_USE_CUDA_GRAPH=1
  - FLASHINFER_NVCC_THREADS=4
  
  # Development
  - FLASHINFER_LOGLEVEL=0
  - FLASHINFER_JIT_VERBOSE=0
```

### FlashInfer Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASHINFER_LOGLEVEL` | 0 | API logging (0=off, 1=basic, 3=detailed, 5=with stats) |
| `FLASHINFER_LOGDEST` | stdout | Log destination (file path or stdout/stderr) |
| `FLASHINFER_JIT_VERBOSE` | 0 | JIT compilation logging |
| `FLASHINFER_JIT_DEBUG` | 0 | Debug build (slower, with symbols) |
| `FLASHINFER_CUDA_ARCH_LIST` | auto | Target architectures (e.g., "12.1") |
| `FLASHINFER_NVCC_THREADS` | 1 | Parallel compilation threads |

### vLLM MXFP4 MoE Variables

| Variable | docker-compose Value | Description |
|----------|---------------------|-------------|
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | `1` | Enable MXFP4 with BF16 activations |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `0` | Enable MXFP4 with MXFP8 activations |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS` | `0` | Use CUTLASS for MXFP8 |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | Enable FP4 MoE |
| `VLLM_FLASHINFER_MOE_BACKEND` | `throughput` | Backend selection (throughput/latency) |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | Attention backend (FLASHINFER/FLASH_ATTN) |
| `VLLM_USE_CUDA_GRAPH` | `1` | Enable CUDA graphs for decode |

### vLLM Serve Command Line Parameters (from docker-compose.yml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--quantization` | `mxfp4` | Quantization method |
| `--tensor-parallel-size` | `1` | TP size (single GPU) |
| `--gpu-memory-utilization` | `0.70` | GPU memory for KV cache |
| `--max-model-len` | `131072` | Max sequence length |
| `--max-num-seqs` | `2` | Max concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Max tokens per batch |
| `--async-scheduling` | enabled | Async request scheduling |
| `--enable-prefix-caching` | enabled | Prefix caching for shared prompts |
| `--enable-auto-tool-choice` | enabled | Auto tool selection |
| `--tool-call-parser` | `openai` | Tool call parser format |
| `--reasoning-parser` | `openai_gptoss` | Reasoning parser for GPT-OSS |
| `--load-format` | `fastsafetensors` | Fast model loading |

## Performance Optimization Recommendations

### 1. vLLM Runtime Settings (High Impact)

Current docker-compose settings:
```yaml
command: >
  vllm serve openai/gpt-oss-120b
    --quantization mxfp4
    --gpu-memory-utilization 0.70
    --max-model-len 131072
    --max-num-seqs 2
    --max-num-batched-tokens 8192
    --async-scheduling
    --enable-prefix-caching
    --load-format fastsafetensors
```

**Tuning suggestions**:
- Increase `--max-num-seqs` for higher concurrency (if memory allows)
- Increase `--gpu-memory-utilization` to 0.85-0.95 for more KV cache
- `--async-scheduling` is already enabled (good)
- CUDA graphs are enabled via `VLLM_USE_CUDA_GRAPH=1`

### 2. FlashInfer Configuration (Medium Impact)

Already configured in docker-compose:
```yaml
environment:
  - VLLM_ATTENTION_BACKEND=FLASHINFER
  - VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
  - VLLM_FLASHINFER_MOE_BACKEND=throughput
  - FLASHINFER_NVCC_THREADS=4
```

**Additional tuning**:
```bash
# Enable JIT verbose logging to see kernel selection
export FLASHINFER_JIT_VERBOSE=1

# Warm up JIT cache (first run compiles kernels)
python -c "import flashinfer; flashinfer.jit.warmup_jit_cache()"
```

### 3. CPU/IPC Optimization (Common Decode Bottleneck)

If decode is CPU-bound:

1. **Bypass HTTP for benchmarks**: Use vLLM Python API directly
2. **Check tokenization overhead**: Pre-tokenize prompts
3. **Reduce ZMQ polling**: Check async engine settings
4. **Use dedicated CPU cores**: Set CPU affinity for engine workers

### 4. MoE GEMM Tile Selection

For decode (small M), ensure correct tiles are selected:

```
SM120 Decode-Optimized Tiles:
- 64x128x128 for batch 1-8
- 64x192x128 for models with hidden_dim=2880 (gpt-oss-120b)
```

If wrong tiles are selected, decode latency can be 10x worse.

### 5. GB10-Specific Considerations

GB10 is a desktop GPU with:
- Single GPU (no tensor parallelism needed)
- Limited memory compared to data center GPUs
- SM121 compute capability

Recommendations:
- Keep `--tensor-parallel-size 1`
- Use `--gpu-memory-utilization 0.70` to leave headroom
- Enable prefix caching to reuse KV cache across requests

## Troubleshooting

### Server Crashes on Startup

1. Check GPU memory availability: `nvidia-smi`
2. Reduce `--gpu-memory-utilization` to 0.8
3. Check CUDA version compatibility: `nvcc --version`
4. Enable debug mode and check dmesg for Xid errors

### FlashInfer Not Detected

```bash
# Verify installation
python -c "import flashinfer; print(flashinfer.__version__)"

# Check PYTHONPATH
echo $PYTHONPATH | tr ':' '\n' | grep flashinfer
```

### Slow Decode Performance

1. Run profiler to identify bottleneck:
   ```bash
   python scripts/profile_sm121_decode_performance.py
   ```

2. Check for CPU bottlenecks (look for high CPU time in trace)

3. Verify CUDA graphs are active (check logs for "CUDA graph capture")

4. Test with local client to eliminate HTTP overhead

### Zombie Processes

```bash
# Find lingering processes
ps auxwf | grep -E 'vllm|EngineCore|python' | grep -v grep

# Kill process group
kill -9 -$(ps -o pgid= -p PID | tr -d ' ')
```

## Acceptance Criteria

A successful test run must satisfy:

- [ ] 0 zombie processes after cleanup
- [ ] 0 silent backend fallbacks (backend summary matches expectations)
- [ ] Stress test: 200 requests, 8 concurrency, 0 failures
- [ ] Decode performance within 10-20% of llama.cpp/SGLang (or documented explanation)

## Related Files

- `CLAUDE.md` - FlashInfer development guide
- `benchmarks/flashinfer_benchmark.py` - Unified benchmarking framework
- `tests/attention/test_attention_sink.py` - Attention sink unit tests
- `tests/moe/test_fused_moe.py` - MoE GEMM unit tests

## Contact

For issues with SM121/GB10 support, check:
1. FlashInfer GitHub issues
2. vLLM documentation on FlashInfer integration
3. NVIDIA CUDA 12.8+ release notes for SM121 support

