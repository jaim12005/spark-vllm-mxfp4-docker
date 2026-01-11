# SM121 Plan: Fuse BF16→FP8 Activation Quantization into MoE Expand (CUTLASS FP8×FP4)

## Context

On SM121, our native MoE path uses **CUTLASS block-scaled FP8×FP4** (FP8 activations × MXFP4 weights).
This is great for **prefill**, but decode regressed because we currently pay **BF16→FP8 activation quantization** as a separate step (via `mxfp8_quantize()`), and on decode (\(M \approx 1\)) that overhead dominates.

The goal of this plan is to **remove the separate BF16→FP8 quantization pass** by fusing it into the existing MoE **expand/permutation** kernel that already touches activations on the way into GEMM1.

## Goal

- **Reduce decode overhead** by eliminating the standalone `mxfp8_quantize()` step.
- Keep the **CUTLASS FP8×FP4 MoE GEMM** unchanged (still consumes FP8 + scale-factor layout).
- Preserve correctness (numerics within expected tolerance for FP8).

Non-goal: Fusing BF16→FP8 inside the CUTLASS GEMM mainloop itself (that would require a different kernel family because the SM12x TMA path expects pre-laid-out FP8 + SFA tiles).

## Current Dataflow (today)

1. Model produces BF16 hidden states.
2. vLLM/FlashInfer quantizes BF16 → FP8 (and produces per-block scale factors) via `mxfp8_quantize()`.
3. MoE routing expands/permutates activations for experts.
4. CUTLASS FP8×FP4 MoE GEMM runs.

## Target Dataflow (after this plan)

1. Model produces BF16 hidden states.
2. MoE routing expand/permutation **outputs FP8 activations** (and writes the required scale-factor buffer layout).
3. CUTLASS FP8×FP4 MoE GEMM consumes those FP8 activations + scale factors.

## Where to Fuse (why here)

The expand/permutation step already:
- reads the unpermuted activations,
- writes the permuted/expanded activations buffer,
- and is on the critical path immediately before GEMM1.

So it’s the right place to fuse a format conversion (BF16 → FP8) while we’re already moving the data.

## Implementation Plan

### Phase 0 — Establish baselines (decode-focused)

- Record decode benchmarks for:
  - **Marlin baseline**
  - **CUTLASS FP8×FP4 (current, separate BF16→FP8 quant)**
- Capture per-token timing (nsys or lightweight counters) for:
  - BF16→FP8 quantization
  - expand/permutation
  - GEMM1/GEMM2

Success criteria: we can attribute the decode delta to the standalone quant step + related overheads.

### Phase 1 — Move BF16→FP8 into expand/permutation (FlashInfer CUDA)

Primary target: `flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`

Work items:
- Extend/enable the existing expand kernel path so it supports:
  - **Input activations**: `__nv_bfloat16` (BF16)
  - **Expanded activations**: `__nv_fp8_e4m3` (FP8)
  - **Scaling type**: `MXFPX` (vec=32), matching the SM12x MoE kernel expectations.
- Ensure scale-factor buffers (`ElementSF*`) are produced in the expected layout:
  - swizzled / tiled layout required by the SM12x block-scaled TMA path
  - alignment requirements (128-granularity rules)
- Ensure the post-expand GEMM path consumes the FP8 expanded buffer (no extra conversion).

Notes:
- This should reuse the kernel’s existing “output a different format during expansion” pattern.
- We must keep `shape_info.problem_shapes` truthful (including \(M=0\) experts), and only treat scale-factor layout as a separate concern.

### Phase 2 — vLLM plumbing: stop doing standalone quant for MoE

Primary target: vLLM MXFP4 quant method (where `mxfp8_quantize()` is invoked before MoE).

Work items:
- Add a mode/flag to the FlashInfer MoE call indicating:
  - “input activations are BF16, please quantize-to-FP8 during expand”
- Pass any needed quantization parameters (e.g. global scale / calibration knobs) through the existing `quant_scales` plumbing.
- Ensure the MoE CUTLASS runner is instantiated for FP8 activations × FP4 weights (unchanged).

### Phase 3 — Correctness validation

- Numerical parity tests:
  - Compare outputs (BF16) before/after on representative prompts for gpt-oss-120b.
  - Focus on decode step-by-step stability (no drift accumulation).
- Stress tests:
  - varying batch sizes / `max_num_seqs`
  - varying prompt lengths
  - k=1 and k>1 routing cases

### Phase 4 — Performance validation

Measure decode again:
- tg32/tg128 with the same harness used for baselines
- Confirm reductions in:
  - kernel count
  - memory traffic for intermediate FP8 activations
  - end-to-end per-token latency

Target outcome: recover the CUTLASS decode regression (29 → ~31+) and ideally exceed Marlin once lm_head work lands.

## Risks / Gotchas

- **Scale-factor layout correctness**: SM12x block-scaled TMA loads are picky; must match expected layout/stride.
- **Quantization calibration**: if standalone `mxfp8_quantize()` used a different scaling policy than the fused path, outputs may drift.
- **Small-M corner cases**: decode \(M=1\) is the primary motivation; must be explicitly optimized (avoid per-token overhead).
- **Autotuner configs**: keep SM12x tactic filtering aligned with what’s supported to avoid excessive “skipping tactics” overhead/log spam.

## Deliverables

- Fused expand/permutation kernel supports BF16→FP8 for SM12x MoE.
- vLLM no longer calls `mxfp8_quantize()` for the MoE activations in the CUTLASS path.
- Benchmarks updated in `docs/BENCHMARK_RESULTS.md` with before/after decode numbers.

