# SM120 MoE Tile Expansion Plan

## Approach: JIT-Based Dynamic Tile Selection

FlashInfer uses **JIT compilation** with disk caching, so we don't precompile variants.
Instead, we parameterize the tile shape and let JIT compile on first use.

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

## 2. JIT vs Precompile

| Approach | How It Works | First-call Latency | Binary Size |
|----------|--------------|-------------------|-------------|
| **Precompile** (llama.cpp) | All variants compiled at build time | None | Large |
| **JIT** (FlashInfer) | Compile on first use, cache to disk | ~10-30s per tile | Small |

### FlashInfer's JIT Caching

FlashInfer already caches compiled kernels:
```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120/
│   ├── moe_gemm_128x128x128.so   # Current (only one)
│   └── (future: moe_gemm_128x8x128.so, etc.)
```

**Advantage:** We compile only the tiles actually used, not all possible variants.

---

## 3. Reference Implementations

### 3.1 TRT-LLM MXFP4 Configs (SM100)

From `trtllmGen_bmm_export/config.json`:

| Template | mmaM | tileN | tileK | Cluster | Use Case |
|----------|------|-------|-------|---------|----------|
| `BatchedGemmMxE2m1E4m3LowLatency` | 128 | **8** | 512 | 1 | **Decode** |
| `BatchedGemmMxE2m1MxE4m3HighThroughput` | **256** | 8 | 512 | **2** | Prefill |

**Key insight:** TRT-LLM uses **tileN=8** for decode, not 16 or 128.

### 3.2 llama.cpp Dynamic Selection

llama.cpp's tile selection logic (adapted for our JIT approach):

```cpp
// Select tile that minimizes padding waste
for (int tile : {8, 16, 32, 64, 128}) {
    int waste = (tile - (M % tile)) % tile;
    if (waste < best_waste) {
        best_tile = tile;
    }
}
// In llama.cpp: switch to precompiled kernel
// In FlashInfer: JIT compile if not cached, then run
```

---

## 4. FlashInfer Autotuner Evaluation

### 4.1 Current Autotuner Architecture

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

### 4.2 What's Disabled/Missing for SM120

| Feature | SM100 (TRT-LLM) | SM120 (Current) | Benefit if Enabled |
|---------|-----------------|-----------------|-------------------|
| **Multiple tile configs** | ✅ Many tiles | ❌ Only 128×128×128 | **+2-4 tok/s** |
| **Mainloop schedule tuning** | ✅ AUTO/PINGPONG/COOP | ❌ Fixed | +1-2 tok/s |
| **Cluster shape tuning** | ✅ 1×1, 2×1, etc. | ❌ Fixed 1×1×1 | +5-10% prefill |
| **min_latency_mode** | ✅ Decode optimization | ❌ Not wired | +2-4 tok/s |
| **Stage count tuning** | ✅ AUTO picks stages | ❌ Fixed AUTO | +5-15% |

### 4.3 Tile Configs Defined But Not Implemented

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

---

## 5. Implementation Plan (JIT-Based)

### 5.1 Parameterize Tile Shape in JIT Template

**File:** `flashinfer/jit/batch_moe_gen.py`

```python
def gen_cutlass_fused_moe_sm120_module(
    tile_m: int = 128,
    tile_n: int = 128,  # NEW: Parameterized (was fixed)
    tile_k: int = 128,
    use_fast_build: bool = False,
):
    """Generate SM120 MoE GEMM module with configurable tile shape."""
    
    # Include tile shape in cache key for separate caching
    module_name = f"fused_moe_120_{tile_m}x{tile_n}x{tile_k}"
    
    return JitModule(
        name=module_name,
        sources=[...],
        extra_cuda_cflags=[
            f"-DTILE_M={tile_m}",
            f"-DTILE_N={tile_n}",
            f"-DTILE_K={tile_k}",
        ],
        ...
    )
```

### 5.2 Add Tile Selection Logic in Python

**File:** `flashinfer/fused_moe/core.py`

```python
def select_tile_for_m(M: int) -> int:
    """Select optimal tile_n based on number of tokens M.
    
    Minimizes padding waste while staying within valid tile options.
    """
    VALID_TILES = [8, 16, 32, 64, 128]
    
    best_tile = 128
    best_waste = float('inf')
    
    for tile in VALID_TILES:
        waste = (tile - (M % tile)) % tile
        if waste < best_waste or (waste == best_waste and tile < best_tile):
            best_waste = waste
            best_tile = tile
    
    return best_tile


@functools.cache
def get_cutlass_fused_moe_module(
    backend: str = "120",
    tile_n: int = 128,  # NEW: Parameterized
    use_fast_build: bool = False,
):
    """Get JIT-compiled module for specific tile configuration.
    
    Results are cached by (backend, tile_n), so each tile variant
    is compiled only once and reused.
    """
    if backend in ("120", "121"):
        module = gen_cutlass_fused_moe_sm120_module(
            tile_m=128,
            tile_n=tile_n,  # Pass tile selection
            tile_k=128,
            use_fast_build=use_fast_build,
        ).build_and_load()
    ...
    return module
```

### 5.3 Wire Through to Kernel Launch

**File:** `flashinfer/fused_moe/core.py`

```python
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    auto_tile_select: bool = True,  # NEW: Enable dynamic tile selection
    activation_type: ActivationType = ActivationType.Swiglu,
) -> torch.Tensor:
    
    M = input.shape[0]  # Number of tokens
    
    # Select optimal tile based on M
    if auto_tile_select:
        tile_n = select_tile_for_m(M)
    else:
        tile_n = 128  # Default (prefill-optimized)
    
    # Get JIT-compiled module for this tile config
    # Cached after first call - subsequent calls are instant
    module = get_cutlass_fused_moe_module(
        backend="120",
        tile_n=tile_n,
    )
    
    # Launch kernel
    return module.run(input, ...)
```

### 5.4 Update C++ Template to Accept Tile Parameters

**File:** `moe_gemm_sm120_mixed_input_launcher.inl`

```cpp
// Template parameterized by tile shape (JIT provides values)
template <int TILE_M, int TILE_N, int TILE_K>
namespace sm120_mxfp4_tile {

using TileShape_MNK = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// ... rest of CUTLASS setup ...

}  // namespace sm120_mxfp4_tile
```

---

## 6. JIT Caching Behavior

### 6.1 Cache Structure

After running with different M values, cache will contain:

```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120_128x8x128/     # Decode (M=1-8)
│   └── fused_moe_120_128x8x128.so
├── fused_moe_120_128x16x128/    # Decode (M=9-16)
│   └── fused_moe_120_128x16x128.so
├── fused_moe_120_128x32x128/    # Small batch (M=17-32)
│   └── fused_moe_120_128x32x128.so
├── fused_moe_120_128x64x128/    # Medium batch (M=33-64)
│   └── fused_moe_120_128x64x128.so
└── fused_moe_120_128x128x128/   # Prefill (M>64)
    └── fused_moe_120_128x128x128.so
```

### 6.2 First-Call vs Cached Performance

| Scenario | First Call | Cached Calls |
|----------|------------|--------------|
| Decode (M=1) | ~20s JIT compile | <1ms |
| Prefill (M=2048) | ~20s JIT compile | <1ms |

### 6.3 Pre-warming Strategy (Optional)

To avoid JIT latency during inference:

```python
def prewarm_moe_tiles():
    """Pre-compile all expected tile variants during server startup."""
    for tile_n in [8, 16, 32, 64, 128]:
        _ = get_cutlass_fused_moe_module(backend="120", tile_n=tile_n)
    print("All MoE tile variants compiled and cached")
```

---

## 7. Expected Performance Impact

### 7.1 Tile Utilization Improvement

| M | Old Tile | New Tile | Utilization | Improvement |
|---|----------|----------|-------------|-------------|
| 1 | 128×128 | 128×8 | 0.8% → 12.5% | **+15× util** |
| 2 | 128×128 | 128×8 | 1.6% → 25% | **+15× util** |
| 4 | 128×128 | 128×8 | 3.1% → 50% | **+16× util** |
| 8 | 128×128 | 128×8 | 6.3% → 100% | **+16× util** |

### 7.2 Projected Decode Throughput

| Configuration | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| Current (128×128 only) | ~50 | Baseline |
| After JIT tile selection | **52-56** | +4-12% |
| + mainloop tuning | **54-58** | +8-16% |

---

## 8. Implementation Steps

### Step 1: Parameterize JIT Template
**File:** `flashinfer/jit/batch_moe_gen.py`
- Add `tile_m`, `tile_n`, `tile_k` parameters
- Include in cache key (module name)

### Step 2: Add Tile Selection Function
**File:** `flashinfer/fused_moe/core.py`
- Add `select_tile_for_m(M)` function
- Returns optimal tile from [8, 16, 32, 64, 128]

### Step 3: Wire Through get_cutlass_fused_moe_module
**File:** `flashinfer/fused_moe/core.py`
- Add `tile_n` parameter
- Pass to JIT generator

### Step 4: Update C++ Launcher Template
**File:** `moe_gemm_sm120_mixed_input_launcher.inl`
- Use preprocessor defines for tile dimensions
- Ensure CUTLASS types work with variable tile shapes

### Step 5: (Optional) Add Pre-warming
**File:** `flashinfer/fused_moe/core.py`
- Add `prewarm_moe_tiles()` function
- Call during vLLM server startup

---

## 9. Validation Plan

1. **JIT compile test**: Each tile variant compiles without errors
2. **Cache test**: Verify separate .so files per tile config
3. **Correctness test**: Output matches reference for each tile
4. **Performance test**: Profile each tile for M=1,2,4,8,16,32,64,128
5. **Integration test**: Full vLLM decode benchmark
6. **Regression test**: Ensure prefill performance maintained

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compiler "Error Internal" | Follow existing namespace pattern exactly |
| First-call JIT latency | Pre-warm during server startup |
| Cache key collision | Include full tile shape in module name |
| CUTLASS tile incompatibility | Test each tile shape individually |

---

## 11. References

- **TRT-LLM tile configs:** `trtllmGen_bmm_export/config.json`
- **llama.cpp tile selection:** `ggml-cuda/mmq.cuh:3980-4048`
- **CUTLASS SM120 examples:** `cutlass/examples/92_blackwell_moe_gemm/`
- **FlashInfer JIT system:** `flashinfer/jit/`
- **FlashInfer autotuner:** `flashinfer/autotuner.py`
- **Current launcher:** `moe_gemm_sm120_mixed_input_launcher.inl`
