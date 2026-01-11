This folder contains the **standalone reproducers** we built while debugging an SM121 (Blackwell) failure:

> `cutlass::gemm::device::GemmUniversalAdapter::initialize()` returning **`Error Internal`** for an MXFP4 grouped GEMM, even when `can_implement()` returns **`Success`**.

It is organized as:

- `repro_sm12x_group_gemm_fp8_groupwise_sm120.cu`: **known-working** grouped GEMM baseline (FP8 groupwise).
- `repro_sm12x_mxf8_mxf4_grouped_init.cu`: the **minimal failing** CUTLASS grouped GEMM (MXF8×MXF4).
- `steps/`: a **one-change-per-step bisection** from “works” → “fails” and back.
- `repro_flashinfer_sm120_mixed_input_launcher_init.cu`: an *attempted* harness to call the FlashInfer launcher directly (kept for reference; it requires linking FlashInfer/TensorRT-LLM objects, so it’s not a pure single-file compile).

## What we found (high level)

- `can_implement()` can be **misleadingly green**: it checks the *static* constraints, while `initialize()` constructs runtime state (notably **TMA descriptors** / internal schedules) and can fail with `Error Internal`.
- The failure was **not** “FP4 is impossible on SM121” and **not** a simple stride/alignment mistake (we validated layouts/strides and even byte-dumped them).
- The strongest reproducer signal is a **code-structure sensitivity**:
  - The *same* kernel configuration + byte-identical arguments can succeed when kernel types are defined in a normal function, but fail when those CUTLASS kernel types are defined inside a **function template**.
  - Moving CUTLASS kernel type definitions to **namespace scope** (and keeping the launcher function templated only for dispatch) restores `initialize()` success in the minimal repro (`steps/step22...`).

## Suggested explanation (why this can happen)

This looks like a **CUTLASS/NVCC compiler/linkage bug** around SM12x TMA + block-scaled grouped GEMM when the kernel type graph is instantiated inside a function template:

- `initialize()` builds internal runtime metadata (including TMA descriptors). Even with identical user-facing args, the compiler can produce different instantiations / linkage / inlining decisions for the CUTLASS internals.
- In the “bad” structure (kernel types nested in a function template), those internal decisions appear to lead to an invalid internal configuration that surfaces only as `Error Internal` at `initialize()`.

We do **not** have a definitive root-cause in CUDA/CUTLASS source yet, but we now have a crisp bisection and a practical workaround that matches FlashInfer’s integration needs.

## How to run

- **Inside `vllm-dev` container** (recommended): CUTLASS headers are available there via FlashInfer.
- Use `steps/build_and_run_step.sh` to compile/run individual steps.
- The older `build_and_run.sh` scripts are kept for convenience, but the “golden” workflow is the step harness.

## Test summaries (each step)

See `steps/README.md` for the authoritative per-step intent list. The outcomes we observed in `vllm-dev` were:

- **step00**: ✅ success (known-working FP8 groupwise baseline)
- **step01**: ❌ `initialize=Error Internal` (MXF8×MXF4 failing baseline)
- **step02**: ⚠️ compilation/static-assert bisection step (MXF8×MXF8 constraints)
- **step03**: ⚠️ compilation/name-fix bisection step (schedule tag naming)
- **step04**: ⚠️ compilation/static-assert bisection step (`SFVectorSize` constraints)
- **step05**: ⚠️ schedule-tag validation step (deducing `SfVectorSize`)
- **step06**: ❌ `initialize=Error Internal`
- **step07**: ❌ `initialize=Error Internal`
- **step08**: ❌ `initialize=Error Internal`
- **step09**: ❌ `initialize=Error Internal`
- **step10**: ❌ `initialize=Error Internal`
- **step11**: ❌ `initialize=Error Internal`
- **step12**: ❌ `initialize=Error Internal` (even with corrected ColumnMajor B stride)
- **step13**: ✅ `initialize=Success` (int32 grouped shapes, container environment)
- **step14**: ✅ `initialize=Success` (int64 grouped shapes can also succeed; this disproved the earlier “int64 is the cause” hypothesis)
- **step15**: ❌ `initialize=Error Internal` (template `<SFVectorSize>` wrapper reintroduces failure)
- **step16**: ❌ `initialize=Error Internal` (providing `host_problem_shapes` does not fix it)
- **step18**: ❌ `initialize=Error Internal` (local `constexpr` bridge does not fix it)
- **step20**: ❌ `initialize=Error Internal` (template version, with byte dumps)
- **step21**: ✅ `initialize=Success` (non-template version, with byte dumps)
- **step22**: ✅ `initialize=Success` (**kernel types moved out of the function template**; key workaround)

