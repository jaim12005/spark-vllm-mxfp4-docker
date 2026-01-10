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

## Success Criteria (Define "Done")

| Level | tg32 Target | Status | What It Means |
|-------|-------------|--------|---------------|
| **Minimum** | ≥52 tok/s | ⏳ | Beat SGLang |
| **Target** | ≥58 tok/s | ⏳ | Match llama.cpp |
| **Stretch** | ≥61 tok/s | ⏳ | Confirm Eagle3 result from WIP |

**Hard Constraints:**
- Prefill must stay ≥4500 tok/s (no regression)
- TTFT p99 must stay ≤1000ms
- No crashes in 100-request stress test
- Must work with CUDA graphs OR have documented reason why not

**Known Win to Preserve:**
Previous `mxfp4_wip` showed Eagle3 + `--enforce-eager` achieved **61 tok/s** on tg128. This MUST be verified on fresh branches before deep optimization work.

---

## Phase 0: Baseline Profiling (DO THIS FIRST)

**Before porting ANY features**, understand where time actually goes.

### Step 0.1: Verify Current Kernel Path

```bash
# Start vLLM with upstream defaults
docker compose -f docker-compose.dev.yml up -d
docker exec -it vllm-dev bash

# Check which MoE kernel is actually used
FLASHINFER_LOGLEVEL=3 python3 -c "
import torch
from vllm import LLM, SamplingParams

llm = LLM(model='openai/gpt-oss-120b', quantization='mxfp4', 
          tensor_parallel_size=1, enforce_eager=True)
output = llm.generate(['Hello'], SamplingParams(max_tokens=10))
" 2>&1 | grep -i "moe\|kernel\|marlin\|cutlass"
```

**Record:**
- [ ] Which MoE kernel fires? (Marlin / CUTLASS / Triton)
- [ ] Which attention backend? (FlashInfer FA2 / Flash Attention)
- [ ] Any fallback warnings?

### Step 0.2: Profile Decode Path with nsys

```bash
# Profile a decode workload (not prefill)
nsys profile --stats=true -o baseline_decode \
  python3 scripts/profile_decode_only.py \
    --prompt-tokens 2048 \
    --output-tokens 64 \
    --model openai/gpt-oss-120b

# Generate report
nsys stats baseline_decode.nsys-rep --report gputrace > baseline_kernels.txt
```

**Record in `docs/analysis/VLLM_BASELINE_ANALYSIS.md`:**

| Kernel Category | Time % | Avg (μs) | Calls/Token | Notes |
|-----------------|--------|----------|-------------|-------|
| MoE FC1 | ? | ? | 60 | |
| MoE FC2 | ? | ? | 60 | |
| Attention | ? | ? | ? | |
| LayerNorm | ? | ? | ? | |
| Quantization | ? | ? | ? | |
| Other | ? | ? | ? | |

### Step 0.3: Verify Eagle3 Still Works

```bash
# Test Eagle3 speculative decoding (known to work in WIP)
vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --speculative-config '{"method": "eagle3", "model": "nvidia/gpt-oss-120b-Eagle3-short-context", "num_speculative_tokens": 3}' \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# Benchmark
docker exec vllm-dev llama-benchy \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --pp 2048 \
  --tg 32 128 \
  --runs 5 \
  --no-cache
```

**Record:**
- [ ] Eagle3 works on fresh branches? YES / NO
- [ ] tg32 with Eagle3: ___ tok/s
- [ ] tg128 with Eagle3: ___ tok/s
- [ ] Any crashes or errors?

### Step 0.4: Decision Gate

Based on profiling results, decide optimization priority:

| If Profiling Shows... | Then Priority Is... |
|-----------------------|---------------------|
| MoE is >50% of decode time | Focus on MoE kernel (CUTLASS, tiles) |
| Attention is >40% of decode time | Focus on attention optimization |
| Quantization overhead >10% | Focus on activation persistence (MXFP8) |
| Eagle3 already hits target | Focus on stability (CUDA graphs, sinks) |
| Marlin is being used (not CUTLASS) | Investigate why, may need explicit override |

### Phase 0 Deliverables

Before proceeding to Phase 1:

- [ ] `docs/analysis/VLLM_BASELINE_ANALYSIS.md` filled with actual data
- [ ] `docs/perf_artifacts/baseline_decode.nsys-rep` captured
- [ ] Kernel path verified and documented
- [ ] Eagle3 status confirmed
- [ ] Priority order validated or adjusted based on data

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

## Phase 5: Feature Porting Process

### CRITICAL REVIEW REQUIREMENT

Every feature porting attempt MUST include a critical review of the implementation:

1. **Question assumptions** - Why was it done this way? Is there a better approach?
2. **Identify inefficiencies** - Memory access patterns, kernel launches, data copies
3. **Note alternative approaches** - What else could be tried?
4. **Document upstream TODOs** - Are there related improvements in FlashInfer?
5. **Suggest experiments** - What benchmarks would validate improvements?

### Porting Process (Step by Step)

For each feature:

```bash
# 1. CREATE feature branch
git checkout -b feature/X mxfp4_v2

# 2. REVIEW the WIP implementation
# Read the code carefully, understand what it does and why

# 3. CRITICAL REVIEW (MANDATORY)
# Document in docs/porting/FEATURE_X.md:
# - What does this implementation do?
# - What are its limitations?
# - What assumptions does it make?
# - Are there better approaches?
# - What upstream TODOs relate to this?
# - What experiments could improve it?

# 4. CHERRY-PICK minimal changes
# - Add proper env var gating
# - Ensure feature can be disabled
# - MUST NOT break existing feature combinations

# 5. BENCHMARK
# - Test with feature ON and OFF
# - Test in combination with previously merged features
# - Verify baseline still works unchanged
# - Record in BENCHMARK_RESULTS.md

# 6. STRUCTURED TESTING (see Testing Protocol)
# - Run full test suite for this feature
# - Document results in TEST_LOGS/

# 7. DECISION
# - If all tests pass: Merge to mxfp4_v2
# - If any test fails: Keep on branch, document failure
# - Either way: Keep critical review notes and test results
```

### Features to Port (Priority Order)

| Priority | Feature | Hypothesis | Success Criteria | Key Questions |
|----------|---------|------------|------------------|---------------|
| 1 | **CUTLASS Grouped GEMM** | Faster than Marlin for MXFP4 on SM121 | ≥10% tg32 improvement | Is runner caching implemented? Can setup be cached? |
| 2 | **Runner/Setup Caching** | Reduce JIT and TTFT overhead | TTFT ≤ 500ms, startup ≤ 60s | How much JIT overhead? |
| 3 | **Activation Quant (MXFP8)** | Reduce BF16→FP8 conversion overhead | ≥5% tg32 improvement | Is quantization fused or separate kernel? |
| 4 | **Tile Variants (64x128)** | Better M=1 efficiency | ≥5% tg32 improvement | Is PingPong dispatch implemented? Scale granularity? |
| 5 | **Speculative Decoding** | Increase effective batch size | ≥50% tg32 improvement | Test all 3: short-context, long-context, throughput |
| 6 | **Attention Sinks** | Long-context stability | No crashes, ≤3% perf regression | Why does it crash on SM121? Kernel issue? |

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

### Level 1: Smoke Test (Boot Check)
```bash
scripts/test_level1_smoke.sh
```
Tests:
- [ ] Server starts with feature enabled
- [ ] Server starts with feature disabled (baseline still works)
- [ ] No Python import errors
- [ ] No CUDA initialization errors
- [ ] JIT compilation completes (if applicable)

**Pass criteria**: Server reaches "ready" state within 5 minutes

### Level 1.5: Kernel Path Validation (GATE)
```bash
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

### Level 2: Correctness Test (Output Validation)
```bash
scripts/test_level2_correctness.sh
```
Tests:
- [ ] Single prompt returns coherent text
- [ ] Token count matches expectation (no truncation bugs)
- [ ] No NaN/Inf in outputs
- [ ] Logprobs are valid (if enabled)
- [ ] Deterministic with same seed

**Pass criteria**: All outputs pass sanity checks

### Level 3: Stress Test (Stability)
```bash
scripts/test_level3_stress.sh
```
Tests:
- [ ] 100 sequential requests without crash
- [ ] 10 concurrent requests without crash
- [ ] Long generation (1024+ tokens) completes
- [ ] Memory usage stable (no leaks over 100 requests)
- [ ] No zombie processes after completion

**Pass criteria**: All requests complete, memory delta < 10%

### Level 4: Performance Benchmark
```bash
scripts/test_level4_benchmark.sh
```
Tests:
- [ ] pp2048 throughput (prefill)
- [ ] tg32 throughput (decode)
- [ ] tg128 throughput (decode, longer)
- [ ] TTFT (time to first token)
- [ ] Compare to baseline configuration

**Pass criteria**: Results recorded in BENCHMARK_RESULTS.md

### Level 5: Regression Test (Feature Combinations)
```bash
scripts/test_level5_regression.sh
```
Tests:
- [ ] Baseline configuration still works
- [ ] All previously merged features still work
- [ ] New feature + each prior feature individually
- [ ] Performance within 5% of previous measurements

**Pass criteria**: No regressions detected

### Level 6: Combinatorial Test (Full Matrix)
```bash
scripts/test_level6_matrix.sh
```
Tests:
- [ ] New feature × all kernel options
- [ ] New feature × all activation options
- [ ] New feature × speculation on/off
- [ ] New feature × CUDA graphs on/off
- [ ] Record any incompatible combinations

**Pass criteria**: Document which combinations work/fail

### Test Log Template

Create `docs/TEST_LOGS/FEATURE_<name>_<date>.md`:

```markdown
# Test Log: [Feature Name]

Date: YYYY-MM-DD
Branch: feature/X

## Level 1: Smoke Test
- [ ] PASS / FAIL
- Duration: Xm Ys
- Notes:

## Level 1.5: Kernel Path Validation
- [ ] PASS / FAIL / INVALID
- Observed kernels:
- Notes:

## Level 2: Correctness Test
- [ ] PASS / FAIL
- Notes:

## Level 3: Stress Test
- [ ] PASS / FAIL
- Memory before: X GB
- Memory after: Y GB
- Notes:

## Level 4: Performance Benchmark

| Metric | Baseline | This Feature | Delta |
|--------|----------|--------------|-------|
| pp2048 | | | |
| tg32 | | | |
| tg128 | | | |
| TTFT p50 | | | |
| TTFT p99 | | | |

## Level 5: Regression Test
- [ ] PASS / FAIL
- Regressions found:

## Level 6: Combinatorial Test

| Combination | Status | Notes |
|-------------|--------|-------|
| + marlin | | |
| + cutlass | | |
| + eagle3-short | | |
| + cuda graphs | | |

## Overall Result
- [ ] READY TO MERGE
- [ ] NEEDS FIXES (list issues)
- [ ] BLOCKED (reason)
```

---

## Phase 7: Orthogonal Feature Gating

### CRITICAL REQUIREMENT: Feature Independence

Each feature MUST be independently toggleable without affecting other features. This enables:

- **Testing any combination** of features (2^N configurations)
- **Isolating performance/stability impact** of each feature
- **Regression testing** when adding new features
- **A/B comparisons** between specific implementations

### Design Principles

1. **No Feature Coupling**: Enabling CUTLASS GEMM must not force a specific activation quantization
2. **Graceful Fallback**: If a feature is disabled, the system falls back to baseline behavior
3. **Combinatorial Testing**: `benchmark_matrix.py` will test the full feature matrix
4. **Additive Changes Only**: New features add code paths, never remove existing ones

### Environment Variables (Orthogonal Axes)

```bash
# Axis 1: MoE Kernel Selection
VLLM_MXFP4_MOE_KERNEL: str = "auto"  # auto, marlin, cutlass, gemv, triton

# Axis 2: Activation Quantization (independent of kernel)
VLLM_MXFP4_ACTIVATION: str = "auto"  # auto, mxfp4, mxfp8, fp8, bf16

# Axis 3: Tile Configuration (only affects CUTLASS path)
FLASHINFER_MOE_TILE: str = "auto"  # auto, 128x128, 64x128

# Axis 4: Attention Sinks (independent of MoE)
VLLM_USE_ATTENTION_SINKS: str = "0"  # 0, 1

# Axis 5: Speculative Decoding (independent of kernel/quant)
# Eagle3 models: short-context, long-context, throughput
# Configured via --speculative-config flag

# Axis 6: CUDA Graphs (independent of everything)
# Configured via --enforce-eager flag (graphs enabled by default)
```

### Feature Matrix Example

Any combination should work:

| Test | Kernel | Activation | Sinks | Spec | Graphs |
|------|--------|------------|-------|------|--------|
| Baseline | marlin | bf16 | off | off | on |
| Test A | cutlass | mxfp8 | off | off | on |
| Test B | cutlass | mxfp8 | on | off | on |
| Test C | cutlass | mxfp8 | on | eagle3-short | off |
| Test D | cutlass | mxfp8 | off | eagle3-long | on |
| Test E | cutlass | mxfp8 | off | eagle3-throughput | on |
| Test F | gemv | bf16 | off | off | on |

### Implementation Pattern

```python
# Each feature checks its own env var, independent of others
def get_moe_kernel():
    kernel = os.getenv("VLLM_MXFP4_MOE_KERNEL", "auto")
    if kernel == "auto":
        return detect_best_kernel()  # Baseline-compatible default
    return kernel

def get_activation_quant():
    quant = os.getenv("VLLM_MXFP4_ACTIVATION", "auto")
    if quant == "auto":
        return "bf16"  # Baseline-compatible default
    return quant

# Features compose independently
kernel = get_moe_kernel()   # marlin, cutlass, gemv, triton
quant = get_activation_quant()  # mxfp4, mxfp8, fp8, bf16
# Both can be any valid combination
```

---

## Phase 8: Engine Analysis Documents

These analysis documents are critical references for optimization decisions:

| Document | Purpose | Location |
|----------|---------|----------|
| `LLAMA_CPP_ANALYSIS.md` | How llama.cpp achieves 58 tok/s decode (DP4A GEMV, activation persistence) | `docs/analysis/` |
| `SGLANG_ANALYSIS.md` | How SGLang achieves ~52 tok/s (scheduler, kernel choices) | `docs/analysis/` |
| `VLLM_BASELINE_ANALYSIS.md` | Analyze what upstream vLLM does before our changes | `docs/analysis/` |

### vLLM Baseline Analysis (To Create)

Perform the same depth of analysis on upstream vLLM as we did for llama.cpp and SGLang:

1. **Kernel Profiling** - nsys/ncu trace of decode path
   - Which kernels are called?
   - Time breakdown per kernel category (attention, MoE, LayerNorm, etc.)
   - Memory bandwidth utilization

2. **Data Flow Analysis**
   - What dtype are activations at each layer?
   - Where does quantization/dequantization happen?
   - How many kernel launches per decode step?

3. **Scheduler Analysis**
   - How does the V1 scheduler work?
   - What's the per-token Python overhead?
   - How does batching affect decode?

4. **Attention Analysis**
   - Which attention backend is used by default?
   - KV cache layout (HND vs NHD)?
   - Page size and memory access patterns?

5. **MoE Analysis** (for gpt-oss-120b)
   - Which MoE kernel is used (Marlin, Triton, CUTLASS)?
   - Expert routing overhead?
   - Weight loading patterns?

This analysis establishes our true baseline before any optimization work.

---

## Phase 9: Benchmark Reproducibility Metadata

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

## Phase 10: FlashInfer Upstream TODOs

Improvement opportunities identified in FlashInfer (durable references):

| Commit | File | Snippet | Improvement |
|--------|------|---------|-------------|
| `abc123` | `csrc/fused_moe/sm12x_*.cuh` | `kSm12xBlockScaleGranularity = 128` | Reduce to 64 for better M=1 |
| `abc123` | `flashinfer/fused_moe/core.py` | `def cutlass_fused_moe` | Add runner caching |
| `abc123` | `csrc/fused_moe/*.cu` | JIT compilation | Pre-compile common configs |

*Note: Line numbers are approximate. Use grep with snippet to find current location.*

---

## Phase 11: Critical Review Template

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

### Immediate (Phase 0 - Baseline Profiling)
1. [x] **Verify kernel path** - Marlin MoE, FlashInfer FA2 attention (2026-01-10)
2. [x] **Profile decode with nsys** - Captured `marlin_flashinfer_profile.nsys-rep` (2026-01-10)
3. [x] **Fill in VLLM_BASELINE_ANALYSIS.md** - Full kernel breakdown, attention is 1.5% (2026-01-10)
4. [ ] **Verify Eagle3 works** - Tested but low acceptance rate (~5%), needs investigation
5. [x] **Decision gate** - MoE (34%) and Dense GEMV (38%) are bottlenecks, NOT attention (1.5%)

### Setup (Phase 1-2)
6. [ ] Run `scripts/setup_mxfp4_v2.sh` to create branches (if not already done)
7. [ ] Document baseline with `scripts/collect_benchmark_metadata.sh`

### Feature Porting (Phase 5+)
8. [ ] Port first feature based on profiling priority
9. [ ] Run 6-level test protocol
10. [ ] Critical review with hypothesis/success criteria
11. [ ] Repeat for each feature

### Success Checkpoints
- [ ] **Checkpoint 1**: tg32 ≥ 40 tok/s (baseline + one optimization)
- [ ] **Checkpoint 2**: tg32 ≥ 52 tok/s (beat SGLang - MINIMUM SUCCESS)
- [ ] **Checkpoint 3**: tg32 ≥ 58 tok/s (match llama.cpp - TARGET)
- [ ] **Checkpoint 4**: Stable with CUDA graphs (production-ready)

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

---

## Reference Repositories

| Repo | Location | Branch | Purpose |
|------|----------|--------|---------|
| vLLM | `~/projects/vllm` | `mxfp4_v2` | Main vLLM development |
| FlashInfer | `~/projects/flashinfer` | `mxfp4_v2` | CUTLASS kernels |
| WIP Reference | `mxfp4_wip` tags | - | Feature source for porting |
| llama.cpp | `~/projects/llama.cpp` | `main` | Reference implementation |
| SGLang | `~/projects/sglang` | `main` | Competing engine |
| mxfp4 | `~/projects/ai/mxfp4` | `main` | This repo - Docker + benchmarking |

---

## Files to Create/Modify Summary

| File | Status | Description |
|------|--------|-------------|
| `AGENTS.md` | ✅ Done | Mission, architecture, tools only |
| `docs/MXFP4_V2_PLAN.md` | ✅ Done | This document |
| `docs/BENCHMARK_RESULTS.md` | ✅ Done | Live benchmark tracking |
| `docs/FEATURE_MATRIX.md` | ✅ Done | Feature status and env vars |
| `docs/UPSTREAM_TODOS.md` | ✅ Done | FlashInfer improvement opportunities |
| `docs/porting/FEATURE_TEMPLATE.md` | ✅ Done | Critical review template |
| `docs/analysis/LLAMA_CPP_ANALYSIS.md` | ✅ Moved | llama.cpp kernel analysis |
| `docs/analysis/SGLANG_ANALYSIS.md` | ✅ Done | SGLang analysis |
| `docs/analysis/VLLM_BASELINE_ANALYSIS.md` | ⏳ TODO | Profile upstream vLLM decode path |
| `docs/investigations/` | ✅ Done | Historical attempts (reference only) |
| `docs/TEST_LOGS/` | ✅ Created | Directory for test log files |
| `docs/perf_artifacts/` | ✅ Created | Directory for nsys/ncu artifacts |
| `scripts/setup_mxfp4_v2.sh` | ✅ Done | Branch setup automation |
| `scripts/benchmark_matrix.py` | ⏳ TODO | Systematic performance testing |
| `scripts/test_level1_smoke.sh` | ⏳ TODO | Boot/startup verification |
| `scripts/test_level1.5_kernel_validation.sh` | ⏳ TODO | Kernel path validation |
| `scripts/test_level2_correctness.sh` | ⏳ TODO | Output validation |
| `scripts/test_level3_stress.sh` | ⏳ TODO | Stability under load |
| `scripts/test_level4_benchmark.sh` | ⏳ TODO | Performance measurement |
| `scripts/test_level5_regression.sh` | ⏳ TODO | Regression detection |
| `scripts/test_level6_matrix.sh` | ⏳ TODO | Combinatorial feature testing |
| `scripts/collect_benchmark_metadata.sh` | ⏳ TODO | Reproducibility metadata |
| `scripts/validate_kernel_path.py` | ⏳ TODO | Kernel path checker |
