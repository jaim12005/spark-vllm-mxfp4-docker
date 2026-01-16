## Overview

This diff is a focused set of CUTLASS SM120 fixes to **make small-N (and generally non-128×128) block-scaled MXF8F6F4 kernels practical and correct**, while improving epilogue performance under SM120’s tighter shared-memory budget.

The main themes are:

- **Epilogue tile selection tuned for SM120** (99KB SMEM) to favor `stmatrix`-based stores when possible.
- **Epilogue R2S store atom selection parameterized by EpiN**, so small epilogue tiles don’t accidentally pick incompatible `stmatrix` variants.
- **Relaxed MMA tile-N constraints** and new **TileN-aware permutations/copy atoms**, enabling valid tiles down to N=8 (in multiples of 8).
- **Correctness fixes for block-scale-factor (SF) layouts under TMA**, by padding SF tiles to at least 128 in M/N (matching TMA requirements and existing config logic).
- **MXF8F6F4 FP4 shared-memory storage modeled as `uint8_t`** (layout + copy) to satisfy the byte-addressing/padding contract required by `ldmatrix.b4x16_p64`.

---

## Epilogue: SM120 tile auto-selection and store atom selection

### `include/cutlass/epilogue/collective/builders/sm120_builder.inl`

#### 1) Default epilogue tile changed from `(64,32)` → dynamic `(EpiM,EpiN)` preferring `EpiN=16`
Previously, the fallback epilogue tile shape defaulted to:

- `Shape<_64, _32>{}`

Now it computes:

- `EpiM = min(64, CTA_M)` (to respect MMA tile M constraints)
- `EpiN = 16 if CTA_N % 16 == 0 else 8 if CTA_N % 8 == 0 else CTA_N`

…and returns `Shape<Int<EpiM>, Int<EpiN>>{}`.

The embedded rationale is strong and practical:

- SM120 has **99KB SMEM** vs SM100’s ~227KB, so epilogue SMEM footprint matters more.
- For a 128×128 CTA:
  - `EpiN=8` saves SMEM (~2KB) but forces slow `AutoVectorizingCopy` and yields ~5.70 pipeline stages.
  - `EpiN=16` uses `stmatrix.x1` (fast) with ~4KB SMEM and ~5.58 stages.
  - Both round to the same effective stage count, so **prefer `EpiN=16` for speed**.

#### 2) R2S store op now depends on `EpiN`
The builder now extracts `EpiN` from `EpilogueTile_MN` and passes it into the store-op selector:

- `sm120_get_smem_store_op_for_accumulator<..., EpiN>()`

This is a key mechanical change: it allows CUTLASS to pick `stmatrix.x2`, `stmatrix.x1`, or fall back to auto-vectorizing based on the *actual* epilogue tile width.

---

## Epilogue common: Tile-aware `stmatrix` selection

### `include/cutlass/epilogue/collective/builders/sm120_common.inl`

#### `sm120_get_smem_store_op_for_accumulator` now takes `EpiN`
Signature change:

- from: `template <class GmemStrideTypeD, class ElementD>`
- to:   `template <class GmemStrideTypeD, class ElementD, int EpiN = 64>`

Behavior changes:

- For **row-major D (N-major)** stores (where `size<1>(stride) == 1`):
  - If `EpiN >= 32`: use `SM90_U32x2_STSM_N` (stmatrix.x2)
  - Else if `EpiN >= 16`: use `SM90_U32x1_STSM_N` (stmatrix.x1)
  - Else: fall back to `AutoVectorizingCopyWithAssumedAlignment`

Also explicitly calls out a real gotcha:

- `EpiN=8` leads to “ambiguous scatter” due to SMEM swizzle/layout conflicts; even x1 can’t save it.

Net effect: small epilogue tiles stop accidentally selecting a `stmatrix` variant that is incompatible with the swizzle/layout constraints.

---

## MMA builder: allow TileN down to 8, and make permutation/copy selection TileN-aware

### `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`

#### 1) Static assert on Tile N relaxed and made precise
Old:

- `static_assert(size<1>(TileShape_MNK{}) >= 32, "Invalid tile shape N.");`

New:

- `static_assert(TileN >= 8 && TileN % 8 == 0, "Tile N must be a multiple of 8 (MMA atom N dimension).");`

This enables tiles like N=8/16/24, which is essential for “small-N” experimentation.

#### 2) TileN passed into the N-permute selector
Old:

- `sm120_tile_n_permute_selector<SFVectorSize>()`

New:

- `sm120_tile_n_permute_selector<SFVectorSize, TileN>()`

This prevents a mismatch where the permute layout assumes 32-wide N (4 atoms) even when the tile is smaller.

#### 3) Separate “layout type” vs “copy type” for B under MXF8F6F4 (FP4)
This section is important and subtle:

- `SmemAllocTypeB` (allocation sizing / element count)
- `SmemLayoutTypeB` (layout math)
- `SmemCopyTypeB` (copy atom ValType / register interpretation)

For MXF8F6F4 with FP4 weights:
- Use `uint8_t` for **copy and register compatibility** (because `ldmatrix.b4x16*` uses byte addressing and expects padding).
- Use “actual FP4 type” semantics for **layout/stage sizing correctness**.

Concretely:
- Layout selector for B uses `SmemLayoutTypeB`
- Copy atom for B uses `SmemCopyTypeB` and now also takes `TileN`

This is directly aimed at two historical failure modes:
1) misaddressed ldmatrix due to sub-byte pointer arithmetic, and
2) incorrect stage count / SMEM budgeting due to treating packed FP4 as fully packed bytes.

#### 4) Scale-factor layout padding fixes in the builder (SFA/SFB)
Two key changes:

- SFA M-dimension padded up to at least 128 (`Blk_MN`) for TMA:
  - `TileM_SFA = ceil_div(TileM, 128) * 128`
- SFB N-dimension padded up to at least 128 for TMA and UTCCP:
  - `TileN_SFB = ceil_div(TileN, 128) * 128`

Then the stride math is updated to use `TileM_SFA` / `TileN_SFB` instead of raw tile dims.

Finally:
- Stage count computation uses `SmemLayoutTypeB` (actual FP4 logical byte count) rather than the `uint8_t` copy type, preventing stage underestimation/overestimation.

---

## SM120 common: TileN-aware shared-memory copy atom selection + TileN-aware permute layouts

### `include/cutlass/gemm/collective/builders/sm120_common.inl`

#### 1) `sm120_rr_smem_copy_selector_B` gains a `TileN` parameter
New signature includes:

- `int TileN = 128`

Selection logic now picks x4/x2/x1 ldmatrix variants based on divisibility:

- `TileN % 32 == 0` → x4
- `TileN % 16 == 0` → x2
- else → x1

This is applied consistently across:
- FP6 (`SM100_SU6_DU8x16_*_LDSM_N`)
- FP4 (`SM100_SU4_DU8x16_*_LDSM_N`)
- FP8 / non-F8F6F4 (`SM75_U32x*_*`)

This is a concrete “make small tiles legal” improvement: the previous fixed x4 selection would be invalid (or wasteful) for smaller N.

#### 2) `sm120_tile_n_permute_selector` now depends on TileN
Old behavior always returned a layout sized for N=32 (4 atoms).

Now it returns layouts matched to the tile:
- TileN >= 32: `Shape<_8,_2,_2>` (32)
- TileN == 24: `Shape<_8,_3>` (24)
- TileN == 16: `Shape<_8,_2>` (16)
- TileN == 8:  `Shape<_8>`     (8)

And the static assert message is updated to reflect TileN validity (“multiple of 8”).

---

## Collective MMA: SF TMA shapes padded to 128 and small-tile asserts relaxed

### `include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`

#### 1) Scale-factor (SF) TMA tile shapes now explicitly padded
Adds:
- `TileShape_SFA`: M padded to ≥128
- `TileShape_SFB`: N padded to ≥128

…and uses them when constructing `make_tma_copy` for SFA/SFB.

This aligns the TMA descriptor shape with:
- the actual SMEM layout sizing logic (Sm1xxBlockScaledConfig),
- and the hardware alignment requirement that motivated the padding in the first place.

#### 2) Skips strict accumulator-vs-SF size asserts when padded
Previously, there were hard asserts that the SF register view matched the accumulator M/N:

- `size<1>(tCrSFA) == size<1>(accum)`
- `size<1>(tCrSFB) == size<2>(accum)`

Now these are conditionally enforced only when the CTA dimension is not “small”:

- `IsCtaMSmall = TileM < 128`
- `IsCtaNSmall = TileN < 128`

When small, SF layouts are padded, so the strict equality would be wrong; skipping these avoids compile-time failure while keeping correctness possible.

#### 3) Clarifies FP4 SMEM padding is a hardware contract
Adds a detailed comment explaining why `uint8_t` SMEM allocation for FP4 in MXF8F6F4 mode is required:

- `ldmatrix.b4x16_p64` requires padded format
- PTX spec: treat sub-byte operands as byte operands for SMEM allocation
- `_p64` implies 64 bits of padding per 16×4-bit chunk

This is an important reviewer-facing clarification: the “extra” bytes aren’t inefficiency; they’re part of the instruction’s addressing contract.

### `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`
Carries the same explanatory comment for the `uint8_t` SMEM allocation under MXF8F6F4 FP4 weights.

---

## What an expert reviewer should pay attention to

1) **Epilogue tile heuristic correctness**
- Verify `(64,16)` is always safe for the CTA shapes you expect, especially for non-128×128 CTAs.
- Confirm the fallback chooses `EpiN=8` only when `CTA_N % 16 != 0` but `%8==0` and that it doesn’t reintroduce the “ambiguous scatter” case unexpectedly.

2) **`stmatrix` compatibility conditions**
- The comments mention swizzle compatibility thresholds (`EpiN >= 16` for x1, `>=32` for x2). Double-check these assumptions against the actual swizzle layout atom used by the epilogue builder on SM120.

3) **TileN-aware ldmatrix variant selection**
- Ensure the x1/x2/x4 selection matches the actual semantics of those copy atoms for FP4/FP6.
- Confirm TileN=24 paths are valid for the chosen copy atom + layout combination (this is a common “looks fine but fails ptxas” trap).

4) **Scale-factor padding correctness and downstream indexing**
- Padding SF tiles to 128 for TMA is correct, but you must ensure:
  - the kernel uses the same padded interpretation when computing SF pointers/strides,
  - and that no code assumes “SF layout == accumulator layout” for small tiles (the new conditional asserts are a start).

5) **Separation of copy type vs layout type for FP4**
- This is the highest-risk, highest-reward part.
- Verify that using `uint8_t` for Copy_Atom ValType produces correct register formatting for the MMA path, while using FP4-typed layout sizing still matches how SMEM is actually allocated and addressed.

If you want, I can also give you a compact “expected behavior matrix” (TileN ∈ {8,16,24,32,64,128} × copy-atom chosen × permute-layout chosen × whether stmatrix is used) to sanity-check the design quickly.
