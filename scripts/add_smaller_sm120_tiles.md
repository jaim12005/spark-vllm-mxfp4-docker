# Adding Smaller CUTLASS Tiles for SM120 Decode Optimization

## Summary

The 128×128 tile constraint for SM120 block-scaled GEMM is a **software limitation**,
not a hardware constraint. The hardware MMA instruction is 16×8, and CUTLASS's own
examples show M=64 tiles working for SM120 block-scaled operations.

## Evidence

### Hardware MMA is 16×8
```cpp
// cute/atom/mma_traits_sm120.hpp
using Shape_MNK = Shape<_16,_8,_64>;  // Actual hardware MMA instruction
```

### CUTLASS Example Uses M=64
```cpp
// examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu
using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;
constexpr int ScaleGranularityM = 1;  // Not 128!
```

## Files to Modify

### 1. FlashInfer: `are_tile_shapes_supported_sm120()` in `moe_gemm_template_dispatch_tma_ws.h`

**Current code (line ~353):**
```cpp
return (TileM == 128 && TileN == 128 && (TileK == 128 || TileK == 256));
```

**Proposed change:**
```cpp
// Allow M=64 and M=128 with various N sizes
if constexpr (TileM != 64 && TileM != 128) {
    return false;
}
if constexpr (TileN != 64 && TileN != 128 && TileN != 256) {
    return false;
}
return (TileK == 128 || TileK == 256);
```

### 2. FlashInfer: Add SHAPE_CASE entries in `moe_gemm_template_dispatch_tma_ws.h`

**Add after `SHAPE_CASE(120, 128, 128, 128)`:**
```cpp
SHAPE_CASE(120, 64, 128, 128)   // M=64 for decode
SHAPE_CASE(120, 128, 64, 128)   // N=64 variant
SHAPE_CASE(120, 64, 64, 128)    // Both smaller
```

### 3. FlashInfer: Update `CutlassTileConfigSM120` enum in `gemm_configs.h`

**Add new tile configs:**
```cpp
enum class CutlassTileConfigSM120 : int {
  Undefined = 0,
  ChooseWithHeuristic = 1,
  // Existing
  CtaShape128x128x128B = shape_tuple_to_enum(128, 128, 128),
  // ... other existing ...
  
  // NEW: Smaller tiles for decode optimization
  CtaShape64x128x128B = shape_tuple_to_enum(64, 128, 128),
  CtaShape128x64x128B = shape_tuple_to_enum(128, 64, 128),
  CtaShape64x64x128B = shape_tuple_to_enum(64, 64, 128),
};
```

### 4. FlashInfer: Update heuristic in `cutlass_heuristic.cpp`

**Add smaller tiles to `get_candidate_configs_sm120()`:**
```cpp
if ((config & CutlassGemmConfig::FP4_ONLY) != 0) {
    candidate_configs.push_back(CutlassGemmConfig{
        CutlassTileConfigSM120::CtaShape64x128x128B, MainloopScheduleType::AUTO,
        EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
    candidate_configs.push_back(CutlassGemmConfig{
        CutlassTileConfigSM120::CtaShape128x128x128B, MainloopScheduleType::AUTO,
        EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
    return candidate_configs;
}
```

## Testing Plan

1. **Compile test**: Build FlashInfer with the changes
2. **Unit test**: Run SM120 grouped GEMM tests with new tiles
3. **Benchmark**: Compare M=64 vs M=128 tiles for M=1 decode
4. **Integration test**: Run vLLM decode benchmark

## Expected Results

For M=1 decode:
- M=128 tile: 0.78% efficiency
- M=64 tile: 1.56% efficiency (2x improvement)
- M=32 tile (if possible): 3.12% efficiency (4x improvement)

While still not ideal, M=64 tiles should roughly halve the decode overhead.

## Risk Assessment

1. **Pipeline stages**: Smaller tiles may reduce pipeline depth. Need to verify Stages >= 2.
2. **Shared memory**: Smaller tiles may not hit shared memory limits.
3. **Occupancy**: May change kernel occupancy characteristics.

## Alternative: ScaleGranularityM=1 Path

The CUTLASS example uses `ScaleGranularityM = 1` which decouples scale factor layout
from tile size. This might be a cleaner approach than adding new tile configs.

## References

- CUTLASS example: `examples/87_blackwell_geforce_gemm_blockwise/87b_*.cu`
- SM120 MMA traits: `include/cute/atom/mma_traits_sm120.hpp`
- Block scale layout: `include/cutlass/detail/sm100_blockscaled_layout.hpp`

