This folder contains **incremental “step” repros**. Each file is a copy of the previous step with a
single focused change, so we can bisect what flips SM12x behavior from “works” to “fails”.

## Steps

- `step00_fp8_groupwise_working.cu`
  - Baseline: calls FlashInfer’s known-working SM12x grouped GEMM:
    `flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`

- `step01_mxf8_mxf4_grouped_init_fail.cu`
  - Baseline failing case: CUTLASS `initialize()` for MXF8×MXF4 grouped GEMM returns
    `Error Internal` on SM121 (even for tiny shapes).

- `step02_mxf8_mxf8_grouped_init.cu`
  - Like step01 but uses **MXF8×MXF8** (no FP4 weights) to isolate whether FP4 is the trigger.

- `step03_mxf8_mxf4_grouped_init_blockwise_schedule.cu`
  - Like step01 but swaps the **mainloop kernel schedule** to `KernelScheduleSm120Blockwise`
    to test schedule sensitivity.

- `step04_mxf8_mxf8_grouped_init_sf16.cu`
  - Like step02 but uses `SFVectorSize=16` to satisfy CUTLASS’ blockscaled builder constraints
    for MXF8×MXF8.

- `step05_mxf8_mxf8_grouped_init_mxf8f6f4_schedule.cu`
  - MXF8×MXF8 again, but swaps the builder schedule tag to
    `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120` to test whether the schedule tag is what
    CUTLASS uses to correctly deduce `SfVectorSize`.

- `step06_mxf8_mxf4_grouped_init_pingpong_blockscaled_sm120.cu`
  - Same as step01, but uses CUTLASS’ **SM120 blockscaled ptr-array schedule tag**:
    `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120`.

- `step07_mxf8_mxf4_grouped_init_coop_blockscaled_sm120.cu`
  - Same as step06 but uses the cooperative variant:
    `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`.

- `step08_mxf8_mxf4_grouped_init_pingpong_blockscaled_sm120_s2.cu`
  - Same as step06 but instantiates the schedule template with `SchedulerPipelineStageCount=2`:
    `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120<2>`.

- `step09_mxf8_mxf4_grouped_init_coop_blockscaled_sm120_s2.cu`
  - Same as step07 but uses `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<2>`.

- `step10_mxf8_mxf4_grouped_init_pingpong_mxf8f6f4_sm120_s2.cu`
  - MXF8×MXF4 grouped init using the **newly-added CUTLASS ptr-array schedule tag**:
    `KernelPtrArrayTmaWarpSpecializedPingpongMxf8f6f4Sm120<2>`.

- `step11_mxf8_mxf4_grouped_init_coop_mxf8f6f4_sm120_s2.cu`
  - Cooperative variant using `KernelPtrArrayTmaWarpSpecializedCooperativeMxf8f6f4Sm120<2>`.

- `step12_mxf8_mxf4_grouped_init_coop_mxf8f6f4_sm120_s2_strideBfix.cu`
  - Same as step11, but fixes `StrideB` construction to treat B as **K×N ColumnMajor**
    (stride = `(1, K)`), rather than using `(N, K)` shape.

- `step13_mxf8_mxf4_grouped_init_int32_problemshape.cu`
  - Like step01, but uses CUTLASS’ **int32** grouped problem shape (`Shape<int,int,int>`)
    to match the upstream SM120 unit test, isolating whether `int64_t` shapes trigger the
    SM121 `initialize(): Error Internal`.

- `step14_mxf8_mxf4_grouped_init_int64_problemshape_regress.cu`
  - Copy of step13, but flips only the grouped problem shape back to
    `Shape<int64_t,int64_t,int64_t>` to confirm this is the minimal regression that causes
    `initialize(): Error Internal` on SM121.

- `step15_mxf8_mxf4_grouped_init_template_sfvector_intargs.cu`
  - Starts from step14 (which uses `KernelPtrArrayTmaWarpSpecializedPingpong` and succeeds) and
    re-introduces **only** the `template <int SFVectorSize>` structure used by step01/10/11/12, while
    keeping runtime `M/N/K` as `int`, to test whether template instantiation itself is triggering
    the `initialize(): Error Internal`.

- `step16_mxf8_mxf4_grouped_init_host_problem_shapes_ptr.cu`
  - Copy of step15, but passes a valid `host_problem_shapes` pointer in CUTLASS’
    `cutlass::gemm::GroupProblemShape` (3rd arg in `{num_groups, problem_sizes, host_problem_shapes}`),
    to test whether failing schedules require host-visible shapes during `initialize()`.

- `step18_mxf8_mxf4_grouped_init_template_sfvector_local_constexpr_bridge.cu`
  - Copy of step15, but introduces `constexpr int kSF = SFVectorSize;` and then uses `kSF`
    everywhere (including `Sm1xxBlockScaledConfig<kSF>`), to test whether “template param used
    directly” vs “local constexpr used” affects `initialize()`.

- `step20_mxf8_mxf4_grouped_init_template_debug_layout_bytes.cu`
  - Like step15 (FAIL), but prints sizes/byte-dumps for the computed scale-factor layouts and packed
    strides, and adds `static_assert(std::is_trivially_copyable_v<...>)` checks. Used to detect UB or
    subtle layout mismatch when the code lives inside a function template.

- `step21_mxf8_mxf4_grouped_init_nontemplate_debug_layout_bytes.cu`
  - Like step14 (SUCCESS), but with the same debug dumps and trivial-copy asserts as step20, so we
    can compare byte-identical inputs between the template vs non-template variants.

- `step22_mxf8_mxf4_grouped_init_template_but_kernel_types_out_of_template.cu`
  - Keeps `run_once<32>(...)` as a function template, but moves all CUTLASS kernel type definitions
    (`CollectiveMainloop`, `GemmKernel`, `Gemm`, etc.) to namespace scope. This tests whether simply
    *nesting CUTLASS kernel types inside a function template* is what triggers `initialize(): Error Internal`.

## Running in the container

These files live in the `ai/mxfp4` repo; the `vllm-dev` container’s `/workspace/flashinfer` is **not**
a bind mount in this environment, so you need to `docker cp` the chosen step into the container and
compile it there.

