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

## Reference: TRT-LLM MXFP4 Configs (SM100)

From `trtllmGen_bmm_export/config.json`:

| Template | mmaM | tileN | tileK | mmaK | Cluster | Use Case |
|----------|------|-------|-------|------|---------|----------|
| `BatchedGemmMxE2m1E4m3LowLatency` | 128 | **8** | 512 | 32 | 1 | **Decode** |
| `BatchedGemmMxE2m1MxE4m3LowLatency` | 128 | **8** | 512 | 32 | 1 | **Decode** |
| `BatchedGemmMxE2m1MxE4m3HighThroughput` | **256** | 8 | 512 | 32 | **2** | Prefill |
| `BatchedGemmMxE2m1Bf16LowLatency` | 128 | 8 | 256 | 16 | 1 | BF16 path |

**Key observations:**
- TRT-LLM uses **tileN=8** for decode (smaller than CUTLASS example's 16)
- tileK=512 for FP4/FP8 paths
- Scale factor layout: "8x4" (sfLayoutB/sfLayoutC)

## Reference: llama.cpp Dynamic Tile Selection

llama.cpp uses **runtime tile selection** instead of hardcoded tiles:

```cpp
// Precompile 16 variants: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128
for (int mmq_x = 8; mmq_x <= mmq_x_max; mmq_x += 8) {
    if (mmq_x % granularity != 0 || shared_mem > smpbo) continue;
    const int ntiles_x = (ncols_max + mmq_x - 1) / mmq_x;
    if (ntiles_x < ntiles_x_best) {
        mmq_x_best = mmq_x;  // Pick tile minimizing waste
    }
}
switch (mmq_x_best) { ... }  // Dispatch to precompiled kernel
```

**Plus dynamic warp count** (`calc_nwarps`):
```cpp
switch (ncols_dst) {
    case 1..4: return 4;  // More warps for small problems
    case 5..8: return 2;
    default:   return 1;
}
```

## Current Performance

| M | Current Tile | Utilization | Latency |
|---|--------------|-------------|---------|
| 1 | 128×128×128 | 0.8% | High |
| 2 | 128×128×128 | 1.6% | High |
| 4 | 128×128×128 | 3.1% | High |
| 8 | 128×128×128 | 6.3% | High |

## Proposed Solution

Add decode-oriented tile configurations following TRT-LLM/llama.cpp approaches:

### New Tile Configurations

Following TRT-LLM's configs (tileN=8 for decode, tileK=512):

```cpp
// Decode tile: 128×8×512 (matches TRT-LLM BatchedGemmMxE2m1E4m3LowLatency)
namespace sm120_mxfp4_bf16_128x8x512 {
    using TileShape_MNK = Shape<_128, _8, _512>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
    // sfLayoutB = "8x4" for block scales
}

// Intermediate tiles for gradual scaling
namespace sm120_mxfp4_bf16_128x16x256 {
    using TileShape_MNK = Shape<_128, _16, _256>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

namespace sm120_mxfp4_bf16_128x32x256 {
    using TileShape_MNK = Shape<_128, _32, _256>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

namespace sm120_mxfp4_bf16_128x64x128 {
    using TileShape_MNK = Shape<_128, _64, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}

// High-throughput prefill tile (matches TRT-LLM HighThroughput)
namespace sm120_mxfp4_bf16_256x8x512 {
    using TileShape_MNK = Shape<_256, _8, _512>;
    using ClusterShape_MNK = Shape<_2, _1, _1>;  // 2x cluster for throughput
}
```

### Option A: Static Threshold Dispatch (Simple)

```cpp
// Select tile based on M (number of tokens)
TileConfig selectTileForDecode(int M, bool min_latency_mode) {
    if (min_latency_mode && M <= 8) {
        return TileConfig::_128x8x512;   // Decode
    } else if (min_latency_mode && M <= 32) {
        return TileConfig::_128x32x256;
    } else if (M > 256) {
        return TileConfig::_256x8x512;   // Prefill (high throughput)
    } else {
        return TileConfig::_128x128x128; // Default
    }
}
```

### Option B: Dynamic Selection (llama.cpp-style)

Precompile multiple tiles, select best at runtime:

```cpp
// Precompile: 8, 16, 32, 64, 128 tiles
template <int TileN>
void launch_moe_gemm(...);

void dispatch_moe_gemm(int M, int N, int K, ...) {
    const int tile_options[] = {8, 16, 32, 64, 128};
    int best_tile = 128;
    int best_waste = INT_MAX;
    
    for (int tile : tile_options) {
        if (smem_for_tile(tile) > max_smem) continue;
        int waste = (tile - (M % tile)) % tile;  // Padding waste
        if (waste < best_waste) {
            best_waste = waste;
            best_tile = tile;
        }
    }
    
    switch (best_tile) {
        case   8: launch_moe_gemm<  8>(...); break;
        case  16: launch_moe_gemm< 16>(...); break;
        case  32: launch_moe_gemm< 32>(...); break;
        case  64: launch_moe_gemm< 64>(...); break;
        case 128: launch_moe_gemm<128>(...); break;
    }
}
```

**Pros of Option B:**
- No hardcoded thresholds
- Automatically adapts to different batch sizes
- Same approach proven in llama.cpp (58 tok/s!)

**Cons:**
- More kernels to compile (JIT cache larger)
- Slightly more complex launcher

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
