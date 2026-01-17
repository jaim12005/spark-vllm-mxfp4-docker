# Feature: <NAME>

**Status**: Not Started | In Progress | Completed | Blocked
**Priority**: P<N>
**Branch**: mxfp4_v2

---

## REQUIRED: Hypothesis (1 sentence)

> "This should improve <metric> by <X%> because <reason>."

---

## REQUIRED: Success Criteria (hard numbers)

| Metric | Baseline | Target | Actual | Pass? |
|--------|----------|--------|--------|-------|
| tg32 (tok/s) | 29 | ≥? | - | - |
| pp2048 (tok/s) | 4808 | ≥? | - | - |
| TTFT p50 (ms) | ? | ≤? | - | - |
| TTFT p99 (ms) | ? | ≤? | - | - |
| Memory (GB) | ? | ≤? | - | - |
| Startup (s) | ? | ≤? | - | - |

**Hard constraints:**
- No TTFT regression > 3%
- No memory increase > 2GB
- No crashes in stress test

---

## Source Files Changed

### FlashInfer
| File | Change Description |
|------|-------------------|
| `path/to/file.py` | Description |

### vLLM
| File | Change Description |
|------|-------------------|
| `path/to/file.py` | Description |

---

## Environment Variables Added

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `VAR_NAME` | `default` | `opt1`, `opt2` | What it controls |

---

## Implementation Notes

### Key Decisions
1. Decision and rationale

### Tradeoffs Considered
1. Option A vs Option B: Chose A because...

### Code Review Findings
1. Potential issue or improvement

---

## REQUIRED: Areas for Improvement

After implementing, identify opportunities for further optimization:

1. **<Improvement 1>**: Description
   - Current state: X
   - Better approach: Y
   - Expected gain: Z%
   - Effort: Low/Medium/High

2. **<Improvement 2>**: Description
   - Current state: X
   - Better approach: Y
   - Expected gain: Z%
   - Effort: Low/Medium/High

---

## Test Results

### Level 1: Smoke Test
- [ ] Server starts without crash
- [ ] Single inference succeeds
- [ ] Expected kernel logs appear
- **Result**: PASS / FAIL

### Level 1.5: Kernel Path Validation (GATE)
- [ ] Correct kernels observed in profiler
- [ ] No fallback kernels detected
- **Result**: PASS / FAIL / INVALID

### Level 2: Correctness
- [ ] Output matches reference (perplexity within tolerance)
- [ ] No NaN/Inf in outputs
- **Result**: PASS / FAIL

### Level 3: Stress Test
- [ ] 100 sequential requests: OK
- [ ] Concurrent requests (2-4): OK
- [ ] Long context (32K+): OK
- **Result**: PASS / FAIL

### Level 4: Performance Benchmark
See detailed results below.

### Level 5: Regression Check
- [ ] Perf within 3% of baseline
- [ ] Memory within 1GB of baseline
- **Result**: PASS / FAIL

### Level 6: Combinatorial
- [ ] Works with other features enabled
- [ ] No feature interference
- **Result**: PASS / FAIL

---

## Success Criteria Evaluation

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| tg32 improved ≥X% | ≥X% | ?% | ? |
| TTFT regression ≤3% | ≤3% | ?% | ? |
| Memory increase ≤2GB | ≤2GB | ?GB | ? |
| No crashes in stress | 0 | ? | ? |

**Overall**: PASS / FAIL

---

## Benchmark Results

### Environment
```yaml
date: YYYY-MM-DD
git_sha_vllm: <sha>
git_sha_flashinfer: <sha>
docker_image: <hash>
cuda_version: <version>
```

### Configuration
```yaml
quantization: mxfp4
kernel: <marlin|gemm|gemv>
cuda_graphs: <true|false>
# ... other relevant settings
```

### Results
```yaml
throughput:
  pp2048_tps: ?
  tg32_tps: ?
  tg128_tps: ?

latency:
  ttft_p50_ms: ?
  ttft_p99_ms: ?
  tpot_p50_ms: ?
  tpot_p99_ms: ?
```

### Profiling Artifacts
- nsys: `docs/perf_artifacts/<run_id>.nsys-rep`
- ncu: `docs/perf_artifacts/<run_id>.ncu-rep`

---

## Rollback Plan

If feature causes issues:

```bash
# Disable via environment variable
export VAR_NAME=off

# Or revert to baseline branch
git checkout mxfp4_v2_baseline
```

---

## References

- Related PR: 
- Upstream issue: 
- Related investigation: `docs/investigations/<file>.md`
