# MXFP4 v2 Benchmarking & Optimization Plan

## Mission

Make **vLLM the fastest inference engine for gpt-oss-120b** on NVIDIA GB10 (SM121), outperforming SGLang and llama.cpp by leveraging native FP4 hardware features.

## Targets to Beat

| Engine | pp2048 (t/s) | tg32 (t/s) | Source |
|--------|--------------|------------|--------|
| **llama.cpp** | 2449.83 | **57.85** | llama-bench |
| **SGLang** | - | **~52** | benchy |
| **vLLM (current)** | **4808** ✓ | 29.26 ❌ | benchy |

**Prefill is excellent** (2x llama.cpp). **Decode is the bottleneck** (2x slower than targets).

---

## Strategy: Clean Start with Feature Gating

### Core Approach

1. **Fresh branches** (`mxfp4_v2`) based on upstream/main
2. **Port features one at a time** from `mxfp4_wip` branches
3. **Gate each feature** with environment variables for independent testing
4. **Benchmark each configuration** with full reproducibility metadata
5. **Critical review** each port with hypothesis and success criteria

### Why This Approach

- Previous `mxfp4_wip` accumulated too many changes without isolation
- Can't tell which changes help vs hurt
- Need systematic A/B testing with proper gating

---

## Phase 1: Backup Current Work

Tag existing branches before starting fresh:

```bash
# FlashInfer
cd ~/projects/flashinfer
git tag mxfp4_wip_backup_$(date +%Y%m%d) mxfp4_wip

# vLLM  
cd ~/projects/vllm
git tag mxfp4_wip_backup_$(date +%Y%m%d) mxfp4_wip

# mxfp4 repo
cd ~/projects/ai/mxfp4
git tag mxfp4_backup_$(date +%Y%m%d)
```

---

## Phase 2: Create Fresh Branches

```bash
# FlashInfer - fresh from upstream
cd ~/projects/flashinfer
git fetch upstream
git checkout -b mxfp4_v2 upstream/main

# vLLM - from PR with MXFP4 gating work
cd ~/projects/vllm
git fetch upstream
git fetch upstream pull/31740/head:pr-31740
git checkout -b mxfp4_v2 pr-31740
```

---

## Phase 3: Baseline Definition

The **baseline** configuration for all comparisons:

| Setting | Value |
|---------|-------|
| MoE Kernel | **Marlin** (vLLM default) |
| Speculative Decoding | **Off** |
| CUDA Graphs | **Enabled** |
| Attention Backend | FlashInfer FA2 |
| Attention Sinks | Off (upstream default) |
| Quantization | MXFP4 weights |

### Baseline vLLM Command

```bash
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --load-format fastsafetensors
    # NOTE: No --enforce-eager (CUDA graphs enabled by default)
```

---

## Phase 4: Configuration Matrix

### Primary Variables

| Variable | Options | Env Var |
|----------|---------|---------|
| MoE Kernel | marlin, gemm (CUTLASS), gemv (DP4A), triton | `VLLM_MXFP4_MOE_KERNEL` |
| Activation Quant | mxfp4, mxfp8, bf16, fp6 | `VLLM_MXFP4_ACTIVATION` |
| Spec Decode | off, eagle3-short, eagle3-long, eagle3-throughput | vLLM flags |
| Attention Sinks | off, on | `VLLM_USE_ATTENTION_SINKS` |
| CUDA Graphs | off, on | `--enforce-eager` flag |
| Tile Config | auto, 128x128, 64x128 | `FLASHINFER_MOE_TILE` |

### Eagle3 Speculative Decoding Models

| Model | Use Case |
|-------|----------|
| `nvidia/gpt-oss-120b-Eagle3-short-context` | Short prompts |
| `nvidia/gpt-oss-120b-Eagle3-long-context` | Long prompts |
| `nvidia/gpt-oss-120b-Eagle3-throughput` | Batch throughput |

---

## Phase 5: Features to Port (Priority Order)

| Priority | Feature | Hypothesis | Success Criteria |
|----------|---------|------------|------------------|
| 1 | **CUTLASS Grouped GEMM** | Faster than Marlin for MXFP4 on SM121 | ≥10% tg32 improvement |
| 2 | **Runner/Setup Caching** | Reduce JIT and TTFT overhead | TTFT ≤ 500ms, startup ≤ 60s |
| 3 | **Activation Quant (MXFP8)** | Reduce BF16→FP8 conversion overhead | ≥5% tg32 improvement |
| 4 | **Tile Variants (64x128)** | Better M=1 efficiency | ≥5% tg32 improvement |
| 5 | **Speculative Decoding** | Increase effective batch size | ≥50% tg32 improvement |
| 6 | **Attention Sinks** | Long-context stability | No crashes, ≤3% perf regression |

### Dependency Graph

```
CUTLASS GEMM (P1)
    ↓
Runner Caching (P2)
    ↓
Activation MXFP8 (P3) ──→ Tile Variants (P4)
                              ↓
                        Spec Decode (P5)
                              ↓
                        Attention Sinks (P6)
```

### Gate Criteria Between Phases

- **P1→P2**: CUTLASS path verified via profiler, no Marlin kernels
- **P2→P3**: TTFT < 1s, startup < 90s
- **P3→P4**: No correctness regressions, perf stable
- **P4→P5**: Tile config works without crashes
- **P5→P6**: Spec decode provides ≥30% speedup before adding sinks

---

## Phase 6: Structured Testing Protocol

After each feature implementation, run these levels:

### Level 1: Smoke Test
```bash
# Boot and basic inference
scripts/test_level1_smoke.sh
```
- Server starts without crash
- Single inference request succeeds
- Expected kernel logs appear

### Level 1.5: Kernel Path Validation (GATE)
```bash
# Verify correct kernels are engaged
scripts/test_level1.5_kernel_validation.sh
```

**Kernel checks by configuration:**

| Config | Must See | Must NOT See |
|--------|----------|--------------|
| kernel=marlin | `marlin_*` | `cutlass_*_moe` |
| kernel=gemm | `cutlass_*_grouped_gemm` | `marlin_*` |
| kernel=gemv | `gemv_fp4_*` or GEMM fallback logged | - |
| activation=mxfp8 | `mxfp8_quantize` | - |

**If validation fails, benchmark results are INVALID.**

### Level 2: Correctness
```bash
scripts/test_level2_correctness.sh
```
- Output matches reference (perplexity within tolerance)
- No NaN/Inf in outputs
- Token-by-token comparison on fixed seed

### Level 3: Stress Test
```bash
scripts/test_level3_stress.sh
```
- 100 sequential requests
- Concurrent requests (2-4)
- Long context (32K+ tokens)
- No OOM or crashes

### Level 4: Performance Benchmark
```bash
scripts/test_level4_benchmark.sh
```
- Run llama-benchy with full metadata collection
- Record pp2048, tg32, tg128
- Capture p50/p99 TTFT, TPOT, ITL

### Level 5: Regression Check
```bash
scripts/test_level5_regression.sh
```
- Compare against baseline
- Flag if perf drops >3%
- Flag if memory increases >1GB

### Level 6: Combinatorial Matrix
```bash
scripts/test_level6_matrix.sh
```
- Test feature combinations
- Ensure orthogonal gating works
- No feature interference

---

## Phase 7: Benchmark Reproducibility Metadata

Every benchmark run MUST record:

### Environment
```yaml
git_shas:
  vllm: <sha>
  flashinfer: <sha>
docker_image: <hash>
pip_freeze: <file>
cuda_version: <version>
driver_version: <version>
cudnn_version: <version>
gpu_clocks: <locked/default>
gpu_power_mode: <mode>
```

### vLLM Configuration
```yaml
vllm_flags:
  quantization: mxfp4
  tensor_parallel_size: 1
  max_model_len: 131072
  max_num_seqs: 2
  max_num_batched_tokens: 8192
  enforce_eager: false
  enable_prefix_caching: true
  load_format: fastsafetensors
  
env_vars:
  VLLM_MXFP4_MOE_KERNEL: <value>
  VLLM_ATTENTION_BACKEND: FLASHINFER
  # ... all relevant vars
```

### Workload Parameters
```yaml
prompt_length: 2048  # tokens
output_length: 32    # or 128
batch_size: 1
concurrency: 1
temperature: 1.0
top_p: 1.0
seed: 42
tokenizer: openai/gpt-oss-120b
```

### Results Format
```yaml
startup_time_s: <seconds>
warmup_requests: 3

throughput:
  pp2048_tps: <value>
  tg32_tps: <value>
  tg128_tps: <value>

latency:
  ttft_p50_ms: <value>
  ttft_p99_ms: <value>
  tpot_p50_ms: <value>
  tpot_p99_ms: <value>
  itl_p50_ms: <value>
  itl_p99_ms: <value>
  e2e_p50_ms: <value>
  e2e_p99_ms: <value>

resource:
  gpu_memory_peak_gb: <value>
  cpu_usage_percent: <value>
```

---

## Phase 8: FlashInfer Upstream TODOs

Improvement opportunities identified in FlashInfer (durable references):

| Commit | File | Snippet | Improvement |
|--------|------|---------|-------------|
| `abc123` | `csrc/fused_moe/sm12x_*.cuh` | `kSm12xBlockScaleGranularity = 128` | Reduce to 64 for better M=1 |
| `abc123` | `flashinfer/fused_moe/core.py` | `def cutlass_fused_moe` | Add runner caching |
| `abc123` | `csrc/fused_moe/*.cu` | JIT compilation | Pre-compile common configs |

*Note: Line numbers are approximate. Use grep with snippet to find current location.*

---

## Phase 9: Critical Review Template

For each ported feature, create `docs/porting/FEATURE_<name>.md`:

```markdown
# Feature: <Name>

## REQUIRED: Hypothesis (1 sentence)
"This should improve <metric> by <X%> because <reason>."

## REQUIRED: Success Criteria (hard numbers)
| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| tg32 (tok/s) | 29 | ≥32 | ? |
| TTFT (ms) | 500 | ≤515 | ? |
| Memory (GB) | 80 | ≤82 | ? |

## Source Files Changed
- `flashinfer/path/to/file.py` - Description
- `vllm/path/to/file.py` - Description

## Environment Variables Added
- `VAR_NAME` - Description, default, options

## Implementation Notes
- Key decisions made
- Tradeoffs considered

## REQUIRED: Areas for Improvement
1. <Specific improvement opportunity>
2. <Specific improvement opportunity>

## Success Criteria Evaluation
| Criterion | Pass/Fail | Notes |
|-----------|-----------|-------|
| tg32 improved ≥5% | ? | |
| TTFT regression ≤3% | ? | |
| Memory increase ≤2GB | ? | |
| No crashes in stress test | ? | |

## Test Results
- Level 1 (Smoke): PASS/FAIL
- Level 1.5 (Kernel Validation): PASS/FAIL  
- Level 2 (Correctness): PASS/FAIL
- Level 3 (Stress): PASS/FAIL
- Level 4 (Benchmark): See results below
- Level 5 (Regression): PASS/FAIL

## Benchmark Results
<link to full results in docs/BENCHMARK_RESULTS.md>
```

---

## Documentation Structure

```
~/projects/ai/mxfp4/
├── AGENTS.md                 # Mission, architecture, tools (slim)
├── docs/
│   ├── MXFP4_V2_PLAN.md     # This document
│   ├── BENCHMARK_RESULTS.md  # Live benchmark tracking
│   ├── FEATURE_MATRIX.md     # Feature status and env vars
│   ├── UPSTREAM_TODOS.md     # FlashInfer improvements (durable refs)
│   ├── analysis/
│   │   ├── LLAMA_CPP_ANALYSIS.md
│   │   ├── SGLANG_ANALYSIS.md
│   │   └── VLLM_BASELINE_ANALYSIS.md
│   ├── investigations/       # Historical analysis (moved here)
│   │   ├── GEMV_IMPLEMENTATION_PLAN.md
│   │   ├── PERFORMANCE_GAP_ANALYSIS.md
│   │   └── TYPE_TRANSFORMATION_ANALYSIS.md
│   ├── porting/              # One file per feature
│   │   ├── FEATURE_CUTLASS_GEMM.md
│   │   ├── FEATURE_RUNNER_CACHING.md
│   │   └── ...
│   ├── perf_artifacts/       # nsys/ncu run IDs
│   └── TEST_LOGS/            # Test run logs
└── scripts/
    ├── setup_mxfp4_v2.sh     # Branch setup automation
    ├── benchmark_matrix.py    # Systematic testing
    ├── collect_benchmark_metadata.sh
    ├── validate_kernel_path.py
    ├── test_level1_smoke.sh
    ├── test_level1.5_kernel_validation.sh
    ├── test_level2_correctness.sh
    ├── test_level3_stress.sh
    ├── test_level4_benchmark.sh
    ├── test_level5_regression.sh
    └── test_level6_matrix.sh
```

---

## Current Docker Configuration

### Development Environment (`docker-compose.dev.yml`)

The dev container mounts local FlashInfer and vLLM repos for live editing:

```yaml
volumes:
  - ~/projects/vllm:/workspace/vllm
  - ~/projects/flashinfer:/workspace/flashinfer
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ./.cache/flashinfer:/root/.cache/flashinfer
  - ./.cache/vllm:/root/.cache/vllm
```

### vLLM Serve Command (Current)

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
    --enforce-eager \                    # CUDA graphs disabled (for stability)
    --enable-prefix-caching \
    --load-format fastsafetensors
```

### Key Environment Variables

| Variable | Current Value | Purpose |
|----------|---------------|---------|
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | 1 | Enable MXFP4 with BF16 activations |
| `VLLM_ATTENTION_BACKEND` | FLASHINFER | Use FlashInfer for attention |
| `VLLM_USE_CUDA_GRAPH` | 1 | Enable CUDA graphs (overridden by --enforce-eager) |
| `FLASHINFER_NVCC_THREADS` | 4 | Parallel JIT compilation |
| `FLASHINFER_CUDA_ARCH_LIST` | 12.1a | Target SM121 architecture |
| `PYTHONPATH` | /workspace/flashinfer:/workspace/vllm | Use local repos |

### Production vs Dev Differences

| Setting | Production | Dev |
|---------|------------|-----|
| Source mounts | No | Yes |
| `--enforce-eager` | No | Yes |
| `--async-scheduling` | Yes | No |
| Tool choice/reasoning | Yes | No |

---

## VLLM Baseline Analysis Scope

Create `docs/analysis/VLLM_BASELINE_ANALYSIS.md` covering:

### Decode Path Profiling
- Kernel breakdown (MoE, attention, other)
- Per-layer timing
- Memory bandwidth utilization

### CPU Thread Analysis
- What threads are busy while GPU idle
- Poll/busy loop detection
- Async scheduling mode effects

### Scheduler Analysis
- Token scheduling overhead
- IPC/ZMQ overhead
- CUDA launch patterns

### Quantization Overhead
- BF16→FP8 conversion timing
- Scale factor computation
- Memory traffic from conversions

### Comparison Format
Use same format as llama.cpp and SGLang analyses for easy comparison.

---

## Next Steps (In Order)

1. [ ] Run `scripts/setup_mxfp4_v2.sh` to create branches
2. [ ] Document baseline with `scripts/collect_benchmark_metadata.sh`
3. [ ] Run baseline benchmark with llama-benchy
4. [ ] Create `docs/analysis/VLLM_BASELINE_ANALYSIS.md`
5. [ ] Port first feature (CUTLASS GEMM) with critical review
6. [ ] Repeat test protocol for each feature

---

## Quick Reference: Environment Variable Overrides

```bash
# Force specific MoE kernel
export VLLM_MXFP4_MOE_KERNEL=gemm  # marlin, gemm, gemv, triton

# Force specific activation format  
export VLLM_MXFP4_ACTIVATION=mxfp8  # mxfp4, mxfp8, bf16

# Enable attention sinks
export VLLM_USE_ATTENTION_SINKS=1

# FlashInfer tile config
export FLASHINFER_MOE_TILE=64x128  # auto, 128x128, 64x128

# Debug logging
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_JIT_VERBOSE=1
```
