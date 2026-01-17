# SM121 CUTLASS MoE GEMM Porting Plan

**Source**: FlashInfer `mxfp4_wip` branch  
**Target**: FlashInfer `mxfp4_v2` branch  
**Goal**: Enable native FP4×FP8 CUTLASS GEMM on SM121 (GB10)

---

## Problem Statement

Current SM12x CUTLASS MoE kernels in FlashInfer main fail on SM121:
```
Unsupported tile (128, 128, 64) and cluster (1, 1, 1) shape combination for arch 120
```

The `mxfp4_wip` branch contains a working implementation that:
1. Correctly handles SM120/SM121 architecture detection
2. Fixes tile shape configuration (K in elements vs bytes)
3. Adds identity scale factor support for MXFP4 (W4A16) path
4. Provides proper LayoutSFA utilities

---

## Files to Port (Minimal Set)

### Phase 1: Core Architecture Support (Required) ✅ COMPLETED

| File | Lines | Action | Status |
|------|-------|--------|--------|
| `sm12x_arch_config.h` | 97 | **NEW** | ✅ Done |
| `moe_tma_warp_specialized_traits.h` | +61 | **MODIFY** | ✅ Done |
| `moe_gemm_template_dispatch_tma_ws.h` | +59 | **MODIFY** | ✅ Done (incl. K fix) |
| `moe_gemm_tma_ws_launcher.inl` | +45 | **MODIFY** | ✅ Done (incl. K fix + redefinition fix) |

**Total Phase 1**: ~262 lines of changes (done)

**Key Fixes (Polish)**:

1. **K bytes→elements: Single Source of Truth** (`sm12x_arch_config.h`):
   ```cpp
   Sm12xKBytesToElements<T, KBytes>::value  // T = activation type
   ```
   - Used by BOTH dispatch and launcher
   - FP4 activations: 128 bytes → 256 elements
   - FP8 activations: 128 bytes → 128 elements (identity)
   - Parameterized on activation type T to match CUTLASS conventions

2. **Removed false MXFP4 W4A16 claims**:
   - Kernel only supports NVFP4 (FP4×FP4) and FP8×FP4
   - BF16/FP16 activations are quantized to FP8 by vLLM layer first
   - Removed `IsMXFP4_W4A16`, `IsWFP4ABF16`, `IsWFP4AFP16` from kernel code

3. **Fixed macro redefinition errors**:
   - Constexpr variables `IsSM12x_`, `IsSM103_`, `TileK_` were at namespace scope
   - When macro expanded multiple times → redefinition errors
   - Fixed by inlining computation directly into template arguments

**Current Status**: Phase 1 passes redefinition check but fails with:
```
static assertion failed: "Specialization requires Stages set to value 2 or more."
```
This requires Phase 2 (SM120 block-scaled MMA needs ≥2 pipeline stages).

**vLLM Configuration Changes (Phase 1.5)** ✅ COMPLETED:
- Added unified `VLLM_MXFP4_BACKEND` env var
- Deprecated legacy flags with warnings
- Changed Blackwell default from BF16→MXFP8 CUTLASS (SM12x-compatible)
- Data flow: BF16 → `mxfp8_quantize()` → FP8×FP4 CUTLASS kernel

**Backend options (case-insensitive):**
| Backend | Description | SM Support |
|---------|-------------|------------|
| `MARLIN` | Marlin dequant→BF16 | All GPUs |
| `CUTLASS` | FlashInfer CUTLASS FP8×FP4 | SM100, SM12x |
| `TRITON` | OpenAI Triton | SM90-SM100 |
| `TRTLLM` | TRT-LLM BF16×FP4 | SM100 only |
| `TRTLLM_MXFP8` | TRT-LLM FP8×FP4 | SM100 only |

```bash
# For SM12x (once Phase 2 is complete):
vllm serve openai/gpt-oss-120b --quantization mxfp4 --mxfp4-backend CUTLASS

# Current workaround:
vllm serve openai/gpt-oss-120b --quantization mxfp4 --mxfp4-backend MARLIN
```

### Phase 2: Dedicated SM120 Launcher (Required) ✅ COMPLETED

The SM120 block-scaled MMA (`sm120_blockscaled_mma_array_tma.hpp`) requires at least 2 pipeline stages:
```cpp
static_assert(Stages >= 2, "Specialization requires Stages set to value 2 or more.");
```

**Solution**: Instead of hacking the generic launcher, port the dedicated SM120 launcher from `mxfp4_wip`:

| File | Lines | Action | Status |
|------|-------|--------|--------|
| `moe_gemm_sm120_mixed_input_launcher.h` | 163 | **NEW** | ✅ Done |
| `moe_gemm_sm120_mixed_input_launcher.inl` | 600 | **NEW** | ✅ Done |
| `sm12x_layout_sfa_utils.h` | 281 | **NEW** | ✅ Done |
| `sm12x_activation_quantizer.cuh` | 1234 | **NEW** | ✅ Done |
| `moe_gemm_tma_ws_launcher.inl` | -27 | **REVERT** | ✅ Cleaned up |
| `moe_gemm_template_dispatch_tma_ws.h` | +15 | **MODIFY** | ✅ Done |

**Key Changes**:

1. **SM120-specific kernel schedule**: Uses `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`
   instead of the generic `KernelPtrArrayTmaWarpSpecializedCooperative`. This schedule is designed
   for SM120/SM121 block-scaled tensor ops and ensures proper pipeline stage configuration.

2. **Clean dispatch separation**: SM12x dispatch now routes directly to `sm120_mixed_input_moe_gemm_kernelLauncher`
   instead of going through the generic TMA WS launcher.

3. **Reverted generic launcher**: All SM12x-specific hacks (ternary K conversion, etc.) removed.
   The generic launcher is now clean and only handles SM90/SM100/SM103.

4. **Identity scale buffer management**: The `sm12x_activation_quantizer.cuh` provides cached identity
   scale buffers for MXFP4 workloads (all 0x7F = scale factor 1.0).

**Benefits**:
- Cleaner code separation (SM12x logic in dedicated files)
- Proper pipeline stage handling (no more "Stages < 2" errors)
- Easier maintenance (changes don't affect other architectures)
- Better error messages (SM12x-specific assertions and logging)

---

## Detailed Changes

### Phase 1a: `sm12x_arch_config.h` (NEW - 97 lines)

**Purpose**: Unified SM120/SM121 detection

```cpp
// Key additions:
#define CUTLASS_ARCH_MMA_SM12x_SUPPORTED  // Covers SM120 OR SM121

constexpr bool kIsSm12xSupported = ...;
constexpr bool kIsSm12xFp4Supported = kIsSm12xSupported && kIsFp4Enabled;

constexpr uint8_t kSm12xIdentityScaleRaw = 0x7F;  // Identity scale (1.0)
constexpr int kSm12xBlockScaleGranularity = 128;
```

**Isolation**: New file, no impact on other architectures.

### Phase 1b: `moe_tma_warp_specialized_traits.h` (MODIFY - +66 lines)

**Purpose**: Add SM12x specialization validation

Key changes:
1. Include `sm12x_arch_config.h`
2. Rename `isValidSM120MOESpecialisation` → `isValidSM12xMOESpecialisation`
3. Add MXFP4 (BF16/FP16 × FP4) support check
4. Add backward-compat aliases (deprecated)

**Isolation**: Changes are additive; old code paths unchanged.

### Phase 1c: `moe_gemm_template_dispatch_tma_ws.h` (MODIFY - +86 lines)

**Purpose**: Fix K dimension handling for FP4

Key changes:
1. Add `dispatch_sizeof_bits<T>()` helper (returns 4 for FP4)
2. Fix `are_tile_shapes_supported_sm120()` - only allow (128,128,128B)
3. Fix K tile conversion: `KtileElems = (K * 8) / bits_per_element`
4. Use `isValidSM12xMOESpecialisation` instead of SM120

**Critical Fix**: The original code used `cutlass::sizeof_bits<T>` which doesn't work correctly for FP4 due to include-order issues.

### Phase 1d: `moe_gemm_tma_ws_launcher.inl` (MODIFY - +73 lines)

**Purpose**: Enable mixed-input and correct K conversion

Key changes:
1. Add `KBytesToElements` helper template
2. Fix `KTileElements` computation for SM120 block-scaled
3. Add `IsMXFP4_W4A16` detection (BF16/FP16 × FP4)
4. Update `static_assert` to allow mixed-input on SM120

---

## Porting Strategy

### Step 1: Create `sm12x_arch_config.h`
- Copy from mxfp4_wip (clean, Apache 2.0 license)
- No modifications needed

### Step 2: Update `moe_tma_warp_specialized_traits.h`
- Cherry-pick relevant hunks from diff
- Keep backward-compat aliases
- Test: Compile for SM121

### Step 3: Update `moe_gemm_template_dispatch_tma_ws.h`
- Add `dispatch_sizeof_bits<T>()` helper
- Fix `are_tile_shapes_supported_sm120()`
- Fix SHAPE_CASE macro
- Test: Compile for SM121

### Step 4: Update `moe_gemm_tma_ws_launcher.inl`
- Add K conversion helper
- Fix MmaTileShape for SM120
- Add MXFP4 detection
- Test: Compile for SM121

### Step 5: Test Minimal Port
```bash
# Restart server with native CUTLASS
export VLLM_MXFP4_BACKEND=CUTLASS
vllm serve openai/gpt-oss-120b --quantization mxfp4 ...

# Check autotuner no longer skips all tactics
grep "Skipping tactic" server.log  # Should be fewer/none
```

### Step 6 (If needed): Add SM120 Mixed-Input Launcher
If Phase 1 is insufficient (MXFP4 path fails), port:
- `moe_gemm_sm120_mixed_input_launcher.h`
- `moe_gemm_sm120_mixed_input_launcher.inl`
- `sm12x_layout_sfa_utils.h`

### Step 7 (If needed): Add Identity Scale Manager
If SFA buffer allocation fails, port:
- `sm12x_activation_quantizer.cuh` (minimal subset for identity scales)

---

## Testing Plan

| Test | Command | Expected |
|------|---------|----------|
| Compile | `ninja -C build` in flashinfer | No errors |
| Autotuner | Check vLLM logs | No "Skipping tactic" for SM120 |
| Coherence | Curl `/v1/completions` | Correct answers |
| Benchmark | `llama-benchy` | tg32 > 29 t/s (improvement over Marlin) |

---

## What NOT to Port

1. **Test files** (`tests/sm12*.py`, `tests/sm12*.cu`) - validation only
2. **Benchmark files** (`benchmarks/sm121*.py`) - not needed for functionality
3. **GEMV files** (`csrc/gemv/*`) - different feature, not MoE
4. **Scripts** (`scripts/verify_*.py`, `scripts/*.sh`) - debugging tools

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FP4 type resolution fails | Medium | High | Add `dispatch_sizeof_bits<T>()` (already in plan) |
| TMA layout errors | Medium | High | Use exact LayoutSFA from mxfp4_wip |
| Cluster shape unsupported | Low | High | Only use 1x1x1 cluster (already validated) |
| Identity scale incorrect | Low | Medium | Use 0x7F (tested in mxfp4_wip) |

---

## Estimated Effort

| Phase | Files | Lines | Time |
|-------|-------|-------|------|
| Phase 1 (Core) | 4 | ~320 | 1-2 hours |
| Phase 2 (Launcher) | 3 | ~1044 | 2-3 hours |
| Phase 3 (Scales) | 1 | ~1234 | 1-2 hours |
| Testing | - | - | 2-3 hours |
| **Total** | 8 | ~2600 | 6-10 hours |

**Recommendation**: Start with Phase 1 only. It may be sufficient to fix the autotuner failures. Add Phase 2/3 only if needed.
