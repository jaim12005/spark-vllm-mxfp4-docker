## Overview

This change set is focused on **making CUTLASS fused-MoE usable and iterable on SM120/SM121 (Blackwell/GB10) for the MXFP4 path**, with an emphasis on:

- **Configurable logical GEMM tiles (M,N)** for SM120/121 MoE GEMMs (including “small-M” decode tiles via a swap/transpose trick).
- **Dramatically reduced JIT compile surface area** via a *minimal* build profile that only instantiates the MXFP4 runner path.
- A **new SM120-specific launcher** that avoids a known CUTLASS grouped-GEMM `initialize()` “Error Internal” failure mode caused by templated kernel type instantiation.
- A correctness/robustness fix to **block scale-factor layout construction** to match padded/aligned SF buffer sizing.
- A suite of **SM121 capability probing scripts** to validate which PTX instructions actually compile for `sm_121a`, and to explain why some CUTLASS assumptions are wrong for SM121.

---

## Dependency bump

### `3rdparty/cutlass`
- Updates CUTLASS submodule from `f3fde58...` → `0b40bb7...`.
- This is likely required to pick up SM12x-related changes (TMA grouped GEMM plumbing, block-scaled support, etc.), and the rest of the diff builds on the new CUTLASS behavior.

---

## New benchmarking utility

### `benchmark_sm120_tiles.py` (new)
- Standalone benchmark harness for **SM120 tile experiments** using FlashInfer’s CUTLASS fused MoE module.
- Generates dummy MoE inputs, performs routing (`softmax + topk`), quantizes:
  - **Weights to MXFP4** via `mxfp4_quantize`.
  - **Activations to MXFP8** via `mxfp8_quantize`.
- Builds and runs a CUTLASS fused MoE runner via:
  - `get_cutlass_fused_moe_module(backend="120", tile_mn=tile_mn)`
- Constructs `quant_scales` in the expected positional format (notably viewing block scales as `int32` and weights as `int64`) and benchmarks kernel time.

Net: provides a reproducible microbench entrypoint for tile selection work.

---

## Minimal MXFP4-only instantiation mode (compile-time reduction)

### `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu`
- Adds a compile flag gate: `FLASHINFER_FUSED_MOE_MXFP4_MINIMAL`.
- Under minimal mode, **only instantiates the MXFP4 runner variants**:
  - FP8 e4m3 activations × FP4 e2m1 weights → half and (if enabled) bf16 outputs.
- Otherwise retains the existing broad instantiation matrix.

### `csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu`
- Enforces minimal-mode restriction at runtime:
  - `FLASHINFER_FUSED_MOE_MXFP4_MINIMAL` only supports FP8×FP4 fused MoE (MXFP4).
- Skips construction branches for other quantization modes (NVFP4, INT4, fp16/bf16, etc.) when minimal mode is enabled.

**Intent:** reduce template instantiation count and JIT compile time/failure surface while iterating on SM120/121 tile shapes.

---

## Block scale-factor layout fix (padded dims vs logical dims)

### `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
- Fixes `setupFP4BlockScalingFactors()` to build `LayoutSF` using **padded/aligned GEMM dimensions**, not the raw `(gemm_m, gemm_n, gemm_k)`.

Key logic added:
- Determines whether config is NVFP4 vs MXFPX and uses the corresponding:
  - `MinNDimAlignment{NVFP4|MXFPX}` (typically 128)
  - `MinKDimAlignment{NVFP4|MXFPX}` (typically block_size*4 / often 128)
- Computes:
  - `padded_gemm_m/n/k = alignToSfDim(gemm_*, min_align)`
- Uses padded shapes in:
  - `tile_atom_to_shape_SFA/SFB(...)`
  - the swap/transpose assert that expects symmetry between SFA/SFB layouts under transpose.

**Why it matters:** the SF buffers and per-expert offsets are sized/offset using aligned dimensions; constructing `LayoutSF` from unpadded dims causes kernel indexing to disagree with allocation/offset math (OOB or incorrect scale reads).

---

## SM120/121 tile support in Python API + auto-selection

### `flashinfer/fused_moe/core.py`
Adds a full SM120 tile configuration and selection layer:

- `SM120_SUPPORTED_TILE_MN`: enumerates supported logical `(M,N)` tiles.
  - “Native” tiles where `M >= 64`.
  - “Swapped” tiles for `M < 64`, using transpose/swap trick.
- `select_tile_mn_for_sm120(num_tokens)`: a heuristic:
  - `<64 tokens`: `(64,128)`
  - `>=256 tokens`: `(256,64)`
  - else: `(128,128)`

`get_cutlass_fused_moe_module(...)` changes:
- Signature now accepts `tile_mn=(128,128)` and is cached.
- For backend `"120"` / `"121"`:
  - Validates constraints:
    - Native: `M % 64 == 0`
    - Swapped: `M in {8,16,32}` and `N >= 64` (because physical M becomes logical N)
    - `N in {8,16,32,64,128,256}`
    - `tile_mn` must be in the enumerated supported set (smem capacity / divisibility).
  - Calls `gen_cutlass_fused_moe_sm120_module(..., tile_mn=tile_mn)` so each tile becomes a distinct JIT artifact.

`cutlass_fused_moe(...)` changes:
- Adds `auto_tile_select: bool = True`.
- If enabled, selects tile via `select_tile_mn_for_sm120(num_rows)` and passes to `get_cutlass_fused_moe_module`.

Also adds:
- `prewarm_moe_tiles()` helper to precompile known-good tiles at startup (currently only `(128,128)`).

Net: makes tile selection a first-class knob for SM120/121 and supports prewarming to avoid runtime JIT latency.

---

## SM12x arch flag handling (“a” vs “f” suffix)

### `flashinfer/compilation_context.py`
- Adjusts CUDA arch selection:
  - For **SM12x**, appends `"f"` suffix (e.g., `12.1f`) with a comment:
    - `"a"` suffix rejects block-scaled MMA and FP4 ldmatrix instructions.
  - For **SM9x**, continues appending `"a"`.

This is an important build-enablement change for FP4/FP8 and block-scaled features on SM12x.

---

## SM120: dedicated mixed-input MoE GEMM launcher (namespace-scope kernel types)

### New files
- `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.h`
- `.../moe_gemm_sm120_mixed_input_launcher.inl`

This is a major chunk. It introduces a **SM120-only launcher** for block-scaled mixed-input MoE GEMM with two core goals:

1. **Avoid CUTLASS “Error Internal” on `initialize()`**  
   The file documents a reproducible failure where instantiating the CUTLASS kernel type graph inside a function template can lead to:
   - `can_implement() == Success`, but `initialize() == Error Internal`.
   The workaround implemented:
   - Keep the **CUTLASS kernel types at namespace scope** in a non-templated context.
   - Only template the outer launcher (or otherwise keep dispatch templating from forcing type graph instantiation inside templated function bodies).

2. **Enable logical tile experiments with compile-time flags**  
   The launcher is configured via nvcc defines set by Python JIT:
   - `-DLOGICAL_TILE_M=<M>`
   - `-DLOGICAL_TILE_N=<N>`
   - `-DSWAP_AB=0|1`

It defines two kernel-type configurations using macros:

- **Standard mode (SWAP_AB=0)**:  
  A = FP8 activations, B = FP4 weights, output row-major BF16.

- **Transposed/swap mode (SWAP_AB=1)** for small logical M:  
  Swaps operands and uses a transposed formulation:
  - A = FP4 weights, B = FP8 activations,
  - output written as column-major to represent Dᵀ,
  - uses “virtual M=128” to satisfy tcgen05 minimum M constraints by pushing the small dimension into N.

The launcher:
- Forces `tma_inputs.swap_ab` to match compile-time `SWAP_AB`.
- Wires operand pointers/strides and SF pointers based on swap mode.
- Uses `cutlass::gemm::device::GemmUniversalAdapter` with grouped problem shapes.
- Checks workspace sizing, `can_implement`, and returns workspace size if requested.
- Rejects finalize fusion for now.

---

## Dispatch integration: route SM120 through the new launcher + loosen tile pruning

### `.../launchers/moe_gemm_tma_ws_launcher.inl`
- Includes the new SM120 launcher `.inl`.
- Adds an early arch check:
  - If `ArchTag == Sm120`, dispatches directly to `sm120_mixed_input_moe_gemm_kernelLauncher(...)` and returns.

Also cleans up helper templates:
- `construct_if_true` rewritten in a more nvcc-friendly constexpr style.
- `deduce_layout_sf()` now returns typed null pointers (`LayoutSFA*` / `LayoutSFB*`) when block-scaled, else `void*`.

### `.../moe_gemm_template_dispatch_tma_ws.h`
- Includes the SM120 launcher.
- Updates dispatch logic:
  - For `Arch::kMinComputeCapability >= 120`, uses the SM120-specific path and **explicitly restricts to MXFP4**.
  - Keeps SM90 path as before.

Tile filtering change:
- `are_tile_shapes_supported_sm120()` is changed from a hardcoded allowlist (128/256 combos) to **always return true**.
- Rationale: tile validation now happens at a “higher level” (Python checks), and this prevents template pruning from killing experimental tiles.

---

## JIT generator changes: per-tile modules + build profiles

### `flashinfer/jit/fused_moe.py`
Key additions:

- Environment variable:
  - `FLASHINFER_FUSED_MOE_BUILD_PROFILE`
  - Profiles:
    - `"full"`: existing broad build.
    - `"mxfp4_minimal"`: build only what’s needed for SM120/121 MXFP4 iteration.

`gen_cutlass_fused_moe_sm120_module(...)` now:
- Accepts `tile_mn`.
- Computes `swap_ab = (logical_m < 64)` and passes:
  - `-DLOGICAL_TILE_M=...`
  - `-DLOGICAL_TILE_N=...`
  - `-DSWAP_AB=...`
- Modifies the module name suffix to include tile dimensions (and optional `_mxfp4min`) so caching doesn’t collide across tiles.
- Defaults SM120/121 to minimal profile unless explicitly overridden via env var presence.

`gen_cutlass_fused_moe_module(...)` restructuring:
- Avoids generating/compiling the huge matrix of generated kernels for SM120/121:
  - For SM120 family, **skips CUTLASS kernel generation entirely** and uses a curated source list.
- Establishes `base_sources` necessary for MXFP4 MoE + binding + heuristic.
- Uses `generated_sources` only when appropriate (non-SM120 full builds, etc.).

Net: per-tile JIT artifacts and much faster compile iteration for SM120/121 experiments.

---

## New SM121 hardware/capability probing + smoke tests

New scripts (all under `scripts/`), aimed at answering: *what does SM121 actually support vs what CUTLASS enables?*

- `check_sm121_capabilities.py`: compiles specific PTX snippets and reports mismatches; includes a narrative about CUTLASS `cute/arch/config.hpp` incorrectly enabling SM100A copy atoms for SM121A, leading to illegal `ldmatrix.m8n16...b4x16_p64`.
- `sm121_complete_capability_matrix.py`: broader matrix across ldmatrix/stmatrix/TMA/MMA/tcgen05/wgmma, concluding many FP4/“f8f6f4 kind” instructions are not supported for `sm_121a` in the tested toolchain.
- `test_all_ldmatrix_sm121.py`: enumerates all ldmatrix variants used by CUTLASS and checks compilation support.
- `test_sm121_all_instructions.py`: comprehensive check across categories, with analysis focused on SM120 block-scaled MXFP4 kernel prerequisites.
- `flashinfer_cutlass_fused_moe_mxfp4_smoke.py`: minimal runtime smoke test that builds the SM12x module (fast build) and calls `cutlass_fused_moe` with correctly-shaped dummy buffers/scales.
- `test_fp4_smem_correctness.py`: a small suite to sanity-check quantize/dequantize paths and scale interleaving behavior; includes conditional SM12x checks.
- `test_tile_expansion.py`: enumerates the supported tile set, estimates smem pipeline stages, and attempts JIT compile of each tile configuration under `FLASHINFER_FUSED_MOE_BUILD_PROFILE=mxfp4_minimal`.

These scripts collectively support rapid diagnosis of “why this kernel can’t run/compile on SM121” and validate tile compilation coverage.

---

## What an expert reviewer should pay attention to

1. **Correctness of SF padding change**  
   The switch to padded `(M,N,K)` in `LayoutSF` construction is foundational. Verify that:
   - The alignments used match the allocation/offset math everywhere else.
   - Swap/transposed mode still indexes SF correctly (especially the assert comparing SFA/SFB).

2. **SM120 launcher operand/SF wiring under swap_ab**  
   The launcher rewires:
   - A/B pointers, strides, and SF pointers depending on `swap_ab`.
   Review carefully that:
   - The logical tile maps correctly to the physical tile implied by the swapped namespace.
   - Output layout choices (RowMajor vs ColumnMajor) correctly “untranspose” in memory.

3. **JIT caching + module naming**  
   SM120 now encodes tile dimensions in the module name. Ensure:
   - No cache collisions across tiles or build profiles.
   - `functools.cache` keying includes `tile_mn` (it does via function args).

4. **Minimal build profile behavior**  
   Ensure the minimal profile:
   - Doesn’t accidentally remove necessary translation units for MXFP4 MoE runner initialization.
   - Doesn’t break non-SM120 builds (profile defaults differ by arch).

5. **SM12x arch suffix selection (`f` vs `a`)**  
   This can make or break availability of block-scaled/FP4 features. Ensure the logic matches the intended nvcc/ptxas behavior in your environment.
