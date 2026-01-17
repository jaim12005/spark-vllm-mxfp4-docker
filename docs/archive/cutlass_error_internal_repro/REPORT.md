## Story: vLLM → FlashInfer → CUTLASS (SM121 MXFP4 grouped GEMM `initialize(): Error Internal`)

### Context / symptom

When running **vLLM** with **MXFP4 quantization** on **SM121 (Blackwell)**, we hit a failure inside the FlashInfer/CUTLASS MoE path:

- `cutlass::gemm::device::GemmUniversalAdapter::initialize()` returns **`Error Internal`**
- while `can_implement()` returns **`Success`**

That mismatch is important:
- **`can_implement()`**: validates static constraints (types, alignments, layout compatibility).
- **`initialize()`**: constructs runtime metadata (e.g., TMA descriptors / scheduling state / workspace use) and can fail even if constraints look OK.

### Phase 1: rule out vLLM plumbing

We started by validating vLLM/FlashInfer integration assumptions:

- **Activations must be FP8** for SM12x block-scaled MMA (the kernel expects pre-quantized FP8, not BF16).
- **Weights must be MXFP4 FP4** (FlashInfer represents them as packed FP4, surfaced as an `int64` tensor on the Python side / DLPack mapping).
- **Alignments/schedules**: aligned `AlignmentB=128` and used epilogue schedule auto; experimented with kernel schedule tags.

Even after these “common errors” were ruled out (or fixed), `initialize()` still failed in the problematic path.

### Phase 2: isolate to standalone CUTLASS repro

We then built **standalone reproducers** (this folder) to eliminate:
- vLLM graph execution
- FlashInfer Python/C++ binding plumbing
- any routing/packing complexity beyond what CUTLASS needs

The key result:
- `repro_sm12x_mxf8_mxf4_grouped_init.cu` reproduces **`initialize(): Error Internal`** in a minimal CUTLASS context.

This proved it’s not “vLLM did something weird”; it’s either:
- a subtle CUTLASS API misuse (stride/layout/shape), or
- a CUTLASS/CUDA compiler/runtime bug.

### Phase 3: establish a known-good baseline

We added a green baseline:
- `repro_sm12x_group_gemm_fp8_groupwise_sm120.cu` (known-working FP8 groupwise grouped GEMM).

This told us:
- grouped GEMM on SM121 is **not** broadly broken,
- the failure is specific to the **block-scaled / MXF8×MXF4** configuration (or how we construct it).

### Phase 4: bisection via step repros

We created `steps/stepXX_*.cu` to change **one thing at a time**.

We tried (among others):
- different **kernel schedules** (pingpong/cooperative, blockscaled ptr-array variants)
- corrected **ColumnMajor B stride** construction
- `int64_t` vs `int` problem shapes

The big turning point (in container reality) was NOT “int64 shapes”:
- In the actual `vllm-dev` environment, both `int` and `int64_t` problem shapes can succeed.

Instead, the crisp discriminant became:

> Defining the CUTLASS kernel type graph *inside a function template* can trigger `initialize(): Error Internal`, even when the produced layouts/strides are byte-identical.

Evidence:
- **step21** (non-template): ✅ success
- **step20** (template): ❌ error internal  
  and `step20` prints **byte-identical** layouts/strides to `step21`.

Workaround evidence:
- **step22**: keep the runner function templated, but move CUTLASS kernel type definitions (`CollectiveMainloop`, `GemmKernel`, `Gemm`) to **namespace scope** → ✅ success.

### What we think is happening

This behavior is consistent with a **compiler / linkage / inlining sensitivity** in the CUTLASS SM12x TMA + block-scaled grouped GEMM initialization path:

- `initialize()` is doing “real work” (constructing TMA descriptors and internal state).
- Even with identical user-visible args, moving the kernel type graph into a function template changes compilation units and instantiation patterns enough to provoke a failure that CUTLASS reports only as `Error Internal`.

We do not have a deeper vendor-level explanation yet (i.e., the exact internal invariant being violated), but the repro now isolates it to a *very small* structural change.

### Practical mitigation for FlashInfer/vLLM

Use the **step22 pattern**:
- Keep the launcher API templated for dispatch convenience.
- But define the CUTLASS kernel types at **namespace scope** (or otherwise avoid nesting that type graph inside a function template).

This matches the minimal repro and is a low-risk refactor (no arithmetic changes, only C++ type placement).

