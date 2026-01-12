# MXFP4 Quantization for QKV/O Projections

**Status**: Proposed  
**Date**: 2026-01-12  

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

### 2. Feature Gating with `--mxfp4-layers` CLI Argument

Add a new CLI argument following the `--mxfp4-backend` pattern:

```bash
# Default: only MoE experts (current behavior)
vllm serve ... --quantization mxfp4

# Add QKV/O projections
vllm serve ... --quantization mxfp4 --mxfp4-layers moe,qkv,o,lm_head

# Shorthand for all supported layers
vllm serve ... --quantization mxfp4 --mxfp4-layers all
```

**Supported layer tokens:**

| Token | Layers Matched | Description |
|-------|----------------|-------------|
| `moe` | `*.experts.*` | MoE expert weights (default, always included) |
| `qkv` | `*.qkv_proj` | Fused QKV projection |
| `o` | `*.o_proj` | Attention output projection |
| `lm_head` | `lm_head` | Output logits projection |
| `all` | All above | Shorthand for full quantization |

**Default behavior**: `--mxfp4-layers moe` (backwards compatible)

**Rationale**: 
- Follows existing `--mxfp4-backend` pattern
- Explicit inclusion list is clearer than exclusion
- Easy to A/B test different layer combinations

### 3. Layer Matching Logic

In `Mxfp4Config.get_quant_method()`, check the configured layers:

```python
elif isinstance(layer, LinearBase):
    # Get configured layers from vllm_config
    mxfp4_layers = vllm_config.model_config.mxfp4_layers  # e.g., {"qkv", "o", "lm_head"}
    
    is_qkv = prefix.endswith(".qkv_proj")
    is_o = prefix.endswith(".o_proj")
    
    if is_qkv and "qkv" in mxfp4_layers:
        return Mxfp4LinearMethod()
    if is_o and "o" in mxfp4_layers:
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
| `vllm/config/model.py` | Add `mxfp4_layers: str = "moe"` field to ModelConfig |
| `vllm/engine/arg_utils.py` | Add `--mxfp4-layers` CLI argument |
| `vllm/model_executor/layers/quantization/mxfp4.py` | Update `get_quant_method()` to check `mxfp4_layers` config |
| `docs/FEATURE_MATRIX.md` | Document new `--mxfp4-layers` option |

## Testing Plan

### Level 1: Smoke Test

```bash
vllm serve openai/gpt-oss-120b \
  --quantization mxfp4 \
  --mxfp4-layers moe,qkv,o,lm_head \
  --enforce-eager
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

- [ ] Add `mxfp4_layers: str = "moe"` to `vllm/config/model.py` ModelConfig
- [ ] Add `--mxfp4-layers` argument to `vllm/engine/arg_utils.py`
- [ ] Update `Mxfp4Config.get_quant_method()` to check `mxfp4_layers` and return `Mxfp4LinearMethod` for QKV/O
- [ ] Add LoRA compatibility check for linear layers (fallback to BF16)
- [ ] Update `docs/FEATURE_MATRIX.md` with new `--mxfp4-layers` option
- [ ] Run smoke test: `--mxfp4-layers moe,qkv,o,lm_head`
- [ ] nsys profile to verify Marlin kernels are used for QKV/O
- [ ] Run tg32 benchmark comparing `--mxfp4-layers moe` vs `--mxfp4-layers all`
- [ ] Profile embed_tokens contribution to Dense GEMV bucket (future decision)
