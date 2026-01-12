# MXFP4 Quantization for QKV/O Projections

**Status**: Proposed  
**Date**: 2026-01-12  
**Author**: AI Assistant  

## Overview

Add MXFP4 weight-only quantization (Marlin kernel) to QKV and O projections for gpt-oss-120b on SM121, with feature gating for safe rollback. Defer embed_tokens pending profiling.

## Context

**Profiling data** (`docs/analysis/VLLM_BASELINE_ANALYSIS.md`):

- Dense GEMV/GEMM: **38%** of decode time (includes QKV, O, embed_tokens, lm_head)
- MoE: **34%** of decode time
- Attention: only **1.5%**

The lm_head MXFP4 implementation already exists in `vllm/model_executor/layers/quantization/mxfp4.py` using `Mxfp4LMHeadMethod` with Marlin kernel.

## Proposed Changes

### 1. Add `Mxfp4LinearMethod` for Dense Projections

The existing `Mxfp4LinearMethod` (mxfp4.py:448-555) is already implemented but **never returned** from `get_quant_method()`. Currently line 383 returns `UnquantizedLinearMethod()` for all `LinearBase` layers.

**Change**: Make `Mxfp4Config.get_quant_method()` return `Mxfp4LinearMethod()` for QKV/O projections when enabled.

### 2. Feature Gating with Environment Variables

Add layer-specific quantization controls:

| Env Var | Default | Description |
|---------|---------|-------------|
| `VLLM_MXFP4_QUANTIZE_QKV` | `0` | Enable MXFP4 for qkv_proj layers |
| `VLLM_MXFP4_QUANTIZE_O` | `0` | Enable MXFP4 for o_proj layers |
| `VLLM_MXFP4_QUANTIZE_LM_HEAD` | `auto` | Existing (auto-enables on Blackwell) |

**Rationale**: Off by default for safety; opt-in enables A/B testing.

### 3. Layer Matching Logic

In `Mxfp4Config.get_quant_method()`, add prefix matching:

```python
elif isinstance(layer, LinearBase):
    is_qkv = prefix.endswith(".qkv_proj")
    is_o = prefix.endswith(".o_proj")
    
    if is_qkv and os.getenv("VLLM_MXFP4_QUANTIZE_QKV", "0") == "1":
        return Mxfp4LinearMethod()
    if is_o and os.getenv("VLLM_MXFP4_QUANTIZE_O", "0") == "1":
        return Mxfp4LinearMethod()
    
    # Default: unquantized
    return UnquantizedLinearMethod()
```

### 4. LoRA Compatibility Gate

The existing `Mxfp4LMHeadMethod` has LoRA guards. Apply the same pattern:

```python
if vllm_config.lora_config is not None:
    logger.warning_once(
        f"[MXFP4] Skipping MXFP4 for {prefix} because LoRA is enabled."
    )
    return UnquantizedLinearMethod()
```

### 5. Quantization Flow

The existing `Mxfp4LinearMethod` already implements:

1. `create_weights()`: Creates BF16 weight parameters
2. `process_weights_after_loading()`: Quantizes BF16 -> MXFP4 using `mxfp4_e2m1_quantize()`, then calls `prepare_fp4_layer_for_marlin()`
3. `apply()`: Uses `apply_fp4_marlin_linear()` for fused dequant+GEMM

This is the same Marlin path as lm_head - no kernel work needed.

## Files to Modify

| File | Change |
|------|--------|
| `vllm/model_executor/layers/quantization/mxfp4.py` | Update `get_quant_method()` to return `Mxfp4LinearMethod` for QKV/O based on env vars |
| `vllm/envs.py` | Add `VLLM_MXFP4_QUANTIZE_QKV` and `VLLM_MXFP4_QUANTIZE_O` |
| `docs/FEATURE_MATRIX.md` | Document new env vars |

## Testing Plan

### Level 1: Smoke Test

```bash
VLLM_MXFP4_QUANTIZE_QKV=1 VLLM_MXFP4_QUANTIZE_O=1 \
  vllm serve openai/gpt-oss-120b --quantization mxfp4 ...
```

- Verify server starts and generates coherent text

### Level 2: Kernel Validation

```bash
nsys profile ... 2>&1 | grep -i "marlin\|gemvx"
```

- With flags ON: must see `marlin_*` kernels, reduced `gemvx::kernel<bf16>`
- With flags OFF: baseline behavior (only BF16 GEMV)

### Level 3: Correctness

- Golden prompts (geography, math, code)
- Logit diff vs baseline < threshold

### Level 4: Performance

- tg32 benchmark with flags ON vs OFF
- Target: measurable improvement in dense GEMV time

## Failure Modes to Guard Against

| Risk | Mitigation |
|------|------------|
| Quality regression | Feature flags default OFF; A/B comparison required |
| LoRA incompatibility | Explicit LoRA check with fallback to BF16 |
| Kernel mismatch | Marlin is same kernel as lm_head (proven) |
| Weight shape issues | QKV is [3*q_size, hidden], Marlin supports rectangular |

## embed_tokens Decision

**Deferred** pending profiling. Need nsys breakdown to isolate embed_tokens contribution:

- If embed_tokens is small % of the 38% Dense GEMV bucket, low priority
- If significant, will require specialized quantized embedding kernel (not a GEMM)

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| tg32 (tok/s) | ~30 | >= 35 (with QKV/O MXFP4) |
| Dense GEMV % | 38% | < 30% |
| Quality | Baseline | No regression on golden prompts |

## Implementation Todos

- [ ] Add `VLLM_MXFP4_QUANTIZE_QKV` and `VLLM_MXFP4_QUANTIZE_O` to `vllm/envs.py`
- [ ] Update `Mxfp4Config.get_quant_method()` to return `Mxfp4LinearMethod` for QKV/O
- [ ] Add LoRA compatibility check for linear layers (fallback to BF16)
- [ ] Update `docs/FEATURE_MATRIX.md` with new env vars
- [ ] Run smoke test: server startup and basic generation with flags ON
- [ ] nsys profile to verify Marlin kernels are used for QKV/O
- [ ] Run tg32 benchmark comparing QKV/O MXFP4 ON vs OFF
- [ ] Profile embed_tokens contribution to Dense GEMV bucket (future decision)
