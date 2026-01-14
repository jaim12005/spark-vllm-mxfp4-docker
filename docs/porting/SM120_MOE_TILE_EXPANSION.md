# SM120 MoE Tile Expansion Plan

## Approach: Dynamic Tile Selection (Option B)

Following llama.cpp's proven approach: precompile multiple tile variants, select best at runtime.

---

## 1. Problem Statement

The FlashInfer SM120 MoE GEMM launcher currently has a **single fixed tile configuration**:

```cpp
namespace sm120_mxfp4_bf16_128x128x128 {
    using TileShape_MNK = Shape<_128, _128, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}
```

### Why This Is Suboptimal for Decode

| M | Current Tile | Utilization | Padding Waste |
|---|--------------|-------------|---------------|
| 1 | 128×128×128 | 0.8% | 127 rows wasted |
| 2 | 128×128×128 | 1.6% | 126 rows wasted |
| 4 | 128×128×128 | 3.1% | 124 rows wasted |
| 8 | 128×128×128 | 6.3% | 120 rows wasted |

---

## 2. Reference Implementations

### 2.1 TRT-LLM MXFP4 Configs (SM100)

From `trtllmGen_bmm_export/config.json`:

| Template | mmaM | tileN | tileK | Cluster | Use Case |
|----------|------|-------|-------|---------|----------|
| `BatchedGemmMxE2m1E4m3LowLatency` | 128 | **8** | 512 | 1 | **Decode** |
| `BatchedGemmMxE2m1MxE4m3HighThroughput` | **256** | 8 | 512 | **2** | Prefill |

**Key insight:** TRT-LLM uses **tileN=8** for decode, not 16 or 128.

### 2.2 llama.cpp Dynamic Selection

llama.cpp's approach that achieves 58 tok/s:

```cpp
// Precompile 16 variants: 8, 16, 24, 32, ..., 128
for (int mmq_x = 8; mmq_x <= mmq_x_max; mmq_x += 8) {
    if (mmq_x % granularity != 0 || shared_mem > smpbo) continue;
    const int ntiles_x = (ncols_max + mmq_x - 1) / mmq_x;
    if (ntiles_x < ntiles_x_best) {
        mmq_x_best = mmq_x;  // Minimize padding waste
    }
}
switch (mmq_x_best) { /* dispatch to precompiled kernel */ }
```

**Plus dynamic warp tuning:**
```cpp
switch (ncols_dst) {
    case 1..4: return 4;  // More warps for tiny problems
    case 5..8: return 2;
    default:   return 1;
}
```

---

## 3. FlashInfer Autotuner Evaluation

### 3.1 Current Autotuner Architecture

FlashInfer has a sophisticated autotuner (`flashinfer/autotuner.py`):

```python
class AutoTuner:
    def choose_one(self, custom_op, runners, tuning_config, inputs):
        """Profile all tactics, return best (runner, tactic) pair."""
        for tac in valid_tactics:
            time = self._profile_single_kernel(runner, inputs, tac)
            if time < min_time:
                best_tactic = tac
        # Cache result
        self.profiling_cache[cache_key] = (runner_id, tactic)
```

**Tuning config supports:**
- Dynamic tensor dimensions (DynamicTensorSpec)
- Bucket-based profiling (powers of 2)
- Constraint specifications
- Cached results per GPU model

### 3.2 What's Disabled/Missing for SM120

| Feature | SM100 (TRT-LLM) | SM120 (Current) | Benefit if Enabled |
|---------|-----------------|-----------------|-------------------|
| **Multiple tile configs** | ✅ Many tiles | ❌ Only 128×128×128 | **+2-4 tok/s** |
| **Mainloop schedule tuning** | ✅ AUTO/PINGPONG/COOP | ❌ Fixed | +1-2 tok/s |
| **Cluster shape tuning** | ✅ 1×1, 2×1, etc. | ❌ Fixed 1×1×1 | +5-10% prefill |
| **min_latency_mode** | ✅ Decode optimization | ❌ Not wired | +2-4 tok/s |
| **Stage count tuning** | ✅ AUTO picks stages | ❌ Fixed AUTO | +5-15% |

### 3.3 Tile Configs Defined But Not Implemented

The `CutlassTileConfigSM120` enum has 6 configs, but only 1 is implemented:

```cpp
enum class CutlassTileConfigSM120 {
    Undefined,
    ChooseWithHeuristic,
    CtaShape128x128x128B,  // ✅ IMPLEMENTED
    CtaShape128x128x64B,   // ❌ Not implemented
    CtaShape256x128x64B,   // ❌ Not implemented
    CtaShape128x256x64B,   // ❌ Not implemented
    CtaShape128x128x256B,  // ❌ Not implemented
    CtaShape256x128x128B,  // ❌ Not implemented
};
```

The dispatcher falls back to 128×128×128:
```cpp
switch (gemmConfig.tile_config_sm120) {
    case CtaShape128x128x128B: /* dispatch */ break;
    default:
        // Falls back to 128×128×128 for anything else!
}
```

### 3.4 Recommendation: Enable Autotuner for SM120

**Phase 1: Tile Expansion (this document)**
- Implement multiple tile variants (8, 16, 32, 64, 128)
- Add dynamic dispatch based on M

**Phase 2: Integrate with Autotuner**
- Expose tile configs as tactics
- Let autotuner profile and cache best configs per (M, K, N, expert_count)

---

## 4. Implementation Plan (Dynamic Dispatch)

### 4.1 New Tile Configurations

Following TRT-LLM's decode configs:

```cpp
// Priority tiles for decode
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

// Existing (for prefill/fallback)
namespace sm120_mxfp4_bf16_128x128x128 {
    using TileShape_MNK = Shape<_128, _128, _128>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}
```

### 4.2 Dynamic Dispatch Logic

```cpp
// Runtime tile selection (llama.cpp-style)
template <typename T, typename WeightType, typename OutputType>
void sm120_moe_gemm_dispatch(
    TmaWarpSpecializedGroupedGemmInput tma_inputs,
    int num_experts,
    int M,  // Number of tokens (extracted from problem shape)
    cudaStream_t stream)
{
    // Select tile that minimizes padding waste
    constexpr int tile_options[] = {8, 16, 32, 64, 128};
    int best_tile = 128;
    int best_waste = INT_MAX;
    
    for (int tile : tile_options) {
        // Check shared memory fits
        size_t smem = smem_for_tile(tile);
        if (smem > max_smem_per_sm) continue;
        
        // Calculate padding waste
        int waste = (tile - (M % tile)) % tile;
        if (waste < best_waste || (waste == best_waste && tile < best_tile)) {
            best_waste = waste;
            best_tile = tile;
        }
    }
    
    // Dispatch to precompiled kernel
    switch (best_tile) {
        case   8: launch_sm120_moe<8>(...);   break;
        case  16: launch_sm120_moe<16>(...);  break;
        case  32: launch_sm120_moe<32>(...);  break;
        case  64: launch_sm120_moe<64>(...);  break;
        case 128: launch_sm120_moe<128>(...); break;
    }
}
```

### 4.3 Python API Changes

```python
# flashinfer/fused_moe/core.py
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    # NEW: Enable decode-optimized tile selection
    auto_tile_select: bool = True,  # Dynamic dispatch (Option B)
    activation_type: ActivationType = ActivationType.Swiglu,
) -> torch.Tensor:
```

### 4.4 vLLM Integration

```python
# vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
_ = flashinfer_cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_ids.to(torch.int),
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=self.out_dtype,
    auto_tile_select=True,  # Enable dynamic tile selection
    activation_type=activation_type,
)
```

---

## 5. Expected Performance Impact

### 5.1 Tile Utilization Improvement

| M | Old Tile | New Tile | Utilization | Improvement |
|---|----------|----------|-------------|-------------|
| 1 | 128×128 | 128×8 | 0.8% → 12.5% | **+15× util** |
| 2 | 128×128 | 128×8 | 1.6% → 25% | **+15× util** |
| 4 | 128×128 | 128×8 | 3.1% → 50% | **+16× util** |
| 8 | 128×128 | 128×8 | 6.3% → 100% | **+16× util** |

### 5.2 Projected Decode Throughput

| Configuration | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| Current (128×128 only) | ~50 | Baseline |
| After tile expansion | **52-56** | +4-12% |
| + mainloop tuning | **54-58** | +8-16% |

---

## 6. Implementation Steps

### Step 1: Add Tile Namespaces
**File:** `moe_gemm_sm120_mixed_input_launcher.inl`
- Duplicate the existing namespace for each new tile size
- Follow the exact pattern to avoid "Error Internal" compiler bug

### Step 2: Add Dispatch Function
**File:** `moe_gemm_sm120_mixed_input_launcher.h`
- Add `sm120_moe_gemm_dispatch()` with tile selection logic
- Wire through to Python bindings

### Step 3: Update Python API
**File:** `flashinfer/fused_moe/core.py`
- Add `auto_tile_select` parameter
- Pass to C++ launcher

### Step 4: Integrate with Autotuner (Future)
**File:** `flashinfer/autotuner.py`
- Expose tiles as tactics
- Let autotuner profile and cache

---

## 7. Validation Plan

1. **Compile test**: All 5 tile variants build without errors
2. **Correctness test**: Output matches reference for each tile
3. **Performance test**: Profile each tile for M=1,2,4,8,16,32,64,128
4. **Integration test**: Full vLLM decode benchmark
5. **Regression test**: Ensure prefill performance maintained

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compiler "Error Internal" | Follow existing namespace pattern exactly |
| JIT cache explosion | 5 tiles × 2 dtypes = 10 kernels (acceptable) |
| Shared memory overflow | Check SMEM before dispatch, fallback to smaller tile |
| Performance regression | Benchmark each tile variant before merge |

---

## 9. References

- **TRT-LLM tile configs:** `trtllmGen_bmm_export/config.json`
- **llama.cpp dynamic dispatch:** `ggml-cuda/mmq.cuh:3980-4048`
- **CUTLASS SM120 examples:** `cutlass/examples/92_blackwell_moe_gemm/`
- **FlashInfer autotuner:** `flashinfer/autotuner.py`
- **Current launcher:** `moe_gemm_sm120_mixed_input_launcher.inl`
