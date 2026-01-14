# SM120 MoE Tile Expansion Plan

## Problem

The FlashInfer SM120 MoE GEMM launcher currently has a **single fixed tile configuration**:

```cpp
namespace sm120_mxfp4_bf16_128x128x128 {
    using TileShape_MNK = Shape<_128, _128, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}
```

This is suboptimal for decode (batch_size=1-2) where M is very small. Larger tiles have:
- More wasted compute (padding M up to 128 when M=1-4)
- More shared memory usage than needed
- Higher latency (larger tiles take longer to complete)

## Current Performance

| M | Current Tile | Utilization | Latency |
|---|--------------|-------------|---------|
| 1 | 128×128×128 | 0.8% | High |
| 2 | 128×128×128 | 1.6% | High |
| 4 | 128×128×128 | 3.1% | High |
| 8 | 128×128×128 | 6.3% | High |

## Proposed Solution

Add decode-oriented tile configurations following TRT-LLM's approach:

### New Tile Configurations

```cpp
// Small-N tiles for decode (M=1-4)
namespace sm120_mxfp4_bf16_128x8x128 {
    using TileShape_MNK = Shape<_128, _8, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

namespace sm120_mxfp4_bf16_128x16x128 {
    using TileShape_MNK = Shape<_128, _16, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

namespace sm120_mxfp4_bf16_128x32x128 {
    using TileShape_MNK = Shape<_128, _32, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

namespace sm120_mxfp4_bf16_128x64x128 {
    using TileShape_MNK = Shape<_128, _64, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}
```

### Tile Selection Logic

```cpp
// Select tile based on M (number of tokens)
TileConfig selectTileForDecode(int M) {
    if (M <= 8) {
        return TileConfig::_128x8x128;
    } else if (M <= 16) {
        return TileConfig::_128x16x128;
    } else if (M <= 32) {
        return TileConfig::_128x32x128;
    } else if (M <= 64) {
        return TileConfig::_128x64x128;
    } else {
        return TileConfig::_128x128x128;  // Default for prefill
    }
}
```

## Expected Impact

| M | New Tile | Utilization | Est. Speedup |
|---|----------|-------------|--------------|
| 1 | 128×8×128 | 12.5% | +15× util |
| 2 | 128×8×128 | 25% | +15× util |
| 4 | 128×8×128 | 50% | +16× util |
| 8 | 128×8×128 | 100% | +16× util |

Projected decode improvement: **+2-4 tok/s** (3-8% improvement)

## Implementation Steps

### Step 1: Add New Tile Namespaces

File: `flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl`

1. Duplicate the `sm120_mxfp4_bf16_128x128x128` namespace for each new tile size
2. Update `TileShape_MNK` in each namespace

### Step 2: Add Tile Dispatch Function

```cpp
template <typename T, typename WeightType, typename OutputType, typename EpilogueTag>
void sm120_mixed_input_moe_gemm_dispatch(
    TmaWarpSpecializedGroupedGemmInput tma_inputs, 
    int num_experts,
    int multi_processor_count, 
    cudaStream_t stream, 
    int* occupancy,
    size_t* workspace_size,
    bool min_latency_mode) {
    
    // Get M from problem shape
    int M = tma_inputs.get_m();
    
    if (min_latency_mode && M <= 8) {
        // Use 128×8 tile for decode
        sm120_mixed_input_moe_gemm_kernelLauncher<
            T, WeightType, OutputType, EpilogueTag,
            Shape<_128, _8, _128>, Shape<_1, _1, _1>, true>(...);
    } else if (min_latency_mode && M <= 16) {
        // Use 128×16 tile
        sm120_mixed_input_moe_gemm_kernelLauncher<...>();
    } else {
        // Default: 128×128 for prefill
        sm120_mixed_input_moe_gemm_kernelLauncher<
            T, WeightType, OutputType, EpilogueTag,
            Shape<_128, _128, _128>, Shape<_1, _1, _1>, true>(...);
    }
}
```

### Step 3: Expose min_latency_mode in Python API

File: `flashinfer/fused_moe/core.py`

Add parameter to `cutlass_fused_moe()`:
```python
def cutlass_fused_moe(
    input: torch.Tensor,
    ...
    min_latency_mode: bool = False,  # NEW: Use decode-optimized tiles
) -> torch.Tensor:
```

### Step 4: Wire Through vLLM

File: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`

Set `min_latency_mode=True` during decode when M is small.

## Validation Plan

1. **Compile test**: Verify all new tile namespaces compile without "Error Internal"
2. **Correctness test**: Compare output against 128×128 baseline
3. **Performance test**: Benchmark all tiles across M=1,2,4,8,16,32,64,128
4. **Integration test**: Run full vLLM decode benchmark

## CUTLASS Validation

CUTLASS Blackwell MoE examples **already use small-N tiles** for decode:

```cpp
// From cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu
using MmaTileMNK = Shape<_128,_16,Int<128 / sizeof(ElementA)>>;
// Comment: "use tile size of N=16 to match real use cases 
// (N is typically very small in decoding stage)"
```

This confirms CUTLASS supports 128×16 tiles for SM120 block-scaled MMA.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compiler "Error Internal" on new tiles | Follow same namespace pattern as 128×128 |
| ~~CUTLASS doesn't support small-N tiles~~ | ✅ CUTLASS examples use 128×16 for decode |
| Shared memory constraints | Reduce stage count for smaller tiles |
| JIT cache explosion (multiple tile variants) | Cache key includes tile config |

## Alternative: Stage Tuning Only

If small tiles aren't supported by CUTLASS block-scaled MMA, we can alternatively tune **stages** for the existing 128×128 tile:

```cpp
// More stages = better latency hiding for small M
namespace sm120_mxfp4_bf16_128x128x128_stages6 {
    using StageCount = Int<6>;  // Instead of Auto
}
```

This is less impactful but may still provide +1-2 tok/s.

## References

- TRT-LLM tile configs: `trtllmGen_bmm_export/config.json`
- CUTLASS SM120 tests: `cutlass/test/unit/gemm/device/sm120_tensorop_gemm/`
- FlashInfer current launcher: `moe_gemm_sm120_mixed_input_launcher.inl`
