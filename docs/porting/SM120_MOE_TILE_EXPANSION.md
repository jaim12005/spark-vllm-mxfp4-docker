# SM120 MoE Tile Expansion Plan

## Approach: JIT-Based Dynamic Tile Selection

FlashInfer uses **JIT compilation** with disk caching, so we don't precompile variants.
Instead, we parameterize the tile shape and let JIT compile on first use.

---

## 1. GEMM Layout Definition

**CRITICAL: Understand the dimension mapping before implementing.**

### 1.1 FlashInfer MoE GEMM Layout

From `moe_gemm_kernels.h`:

```cpp
// GroupedGemmInput members:
int64_t num_rows = 0;  // M = tokens routed to each expert
int64_t n = 0;         // N = output dimension (intermediate_size or hidden_size)
int64_t k = 0;         // K = reduction dimension (hidden_size or intermediate_size)

// Default layout (swap_ab = false):
// A = activations: (M, K) RowMajor    → (tokens, hidden_dim)
// B = weights:     (K, N) ColumnMajor → (hidden_dim, intermediate_dim)  
// C = output:      (M, N) RowMajor    → (tokens, intermediate_dim)
```

### 1.2 Dimension Summary

| CUTLASS Dim | Maps To | Typical Size (gpt-oss-120b) | Variable? |
|-------------|---------|----------------------------|-----------|
| **M** | Tokens per expert after routing | 1 (decode) to 2048+ (prefill) | **YES** |
| **N** | intermediate_size or hidden_size | 14336 or 5120 | Fixed |
| **K** | hidden_size or intermediate_size | 5120 or 14336 | Fixed |

### 1.3 TileShape_MNK Convention

```cpp
using TileShape_MNK = Shape<TILE_M, TILE_N, TILE_K>;
//                          ^^^^^^
//                          Token dimension - THIS IS WHAT WE TUNE
```

**Conclusion: Tokens = M. To optimize decode, we tune TILE_M.**

---

## 2. Problem Statement

The FlashInfer SM120 MoE GEMM launcher currently has a **single fixed tile configuration**:

```cpp
namespace sm120_mxfp4_bf16_128x128x128 {
    using TileShape_MNK = Shape<_128, _128, _128>;
    //                          ^^^^
    //                          TILE_M = 128 (token dimension)
    using ClusterShape_MNK = Shape<_1, _1, _1>;
}
```

### Why TILE_M=128 Is Suboptimal for Decode

| M (tokens) | Current TILE_M | M-Dimension Utilization | Padding Waste |
|------------|----------------|-------------------------|---------------|
| 1 | 128 | 0.8% | 127 token slots wasted |
| 2 | 128 | 1.6% | 126 token slots wasted |
| 4 | 128 | 3.1% | 124 token slots wasted |
| 8 | 128 | 6.3% | 120 token slots wasted |

For decode (M=1), we compute a 128×N output tile but only need 1×N.
**99.2% of the tile computation is wasted padding.**

---

## 3. JIT vs Precompile

| Approach | How It Works | First-call Latency | Binary Size |
|----------|--------------|-------------------|-------------|
| **Precompile** (llama.cpp) | All variants compiled at build time | None | Large |
| **JIT** (FlashInfer) | Compile on first use, cache to disk | ~10-30s per tile | Small |

### FlashInfer's JIT Caching

FlashInfer already caches compiled kernels:
```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120/
│   ├── moe_gemm_M128_N128_K128.so   # Current (only one)
│   └── (future: moe_gemm_M8_N128_K128.so, etc.)
```

**Advantage:** We compile only the tiles actually used, not all possible variants.

---

## 4. Reference Implementations

### 4.1 TRT-LLM MXFP4 Configs (SM100)

From `trtllmGen_bmm_export/config.json`:

| Template | Description | Tile Config | Use Case |
|----------|-------------|-------------|----------|
| `BatchedGemmMxE2m1E4m3LowLatency` | Decode-optimized | Small M-tile | **Decode** |
| `BatchedGemmMxE2m1MxE4m3HighThroughput` | Prefill-optimized | Large M-tile, cluster | Prefill |

**Note:** TRT-LLM's config format uses different naming. The key insight is they have
separate configs for decode vs prefill, optimizing the token dimension differently.

### 4.2 Tile Selection Strategies

**Option A: Thresholding (TRT-LLM style, recommended)**
```cpp
// Pick smallest tile that fits all tokens in one CTA
for (int tile_m : {8, 16, 32, 64, 128}) {
    if (M <= tile_m) return tile_m;
}
return 128;
```
- Simple, predictable mapping
- Guarantees single-tile coverage for decode
- No ambiguity in bucket assignment

**Option B: Min-waste with large-tile tie-break (llama.cpp style)**
```cpp
// Minimize padding waste, prefer larger tile on ties
for (int tile_m : {8, 16, 32, 64, 128}) {
    int waste = (tile_m - (M % tile_m)) % tile_m;
    if (waste < best_waste || (waste == best_waste && tile_m > best_tile_m)) {
        best_waste = waste;
        best_tile_m = tile_m;
    }
}
```
- More complex, handles multi-tile scenarios
- Larger tiles have better occupancy characteristics

**We use Option A** for simplicity and decode optimization.

---

## 5. FlashInfer Autotuner Evaluation

### 5.1 Current Autotuner Architecture

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

### 5.2 What's Disabled/Missing for SM120

| Feature | SM100 (TRT-LLM) | SM120 (Current) | Benefit if Enabled |
|---------|-----------------|-----------------|-------------------|
| **Multiple M-tile configs** | ✅ Many tiles | ❌ Only TILE_M=128 | **+2-4 tok/s** |
| **Mainloop schedule tuning** | ✅ AUTO/PINGPONG/COOP | ❌ Fixed | +1-2 tok/s |
| **Cluster shape tuning** | ✅ 1×1, 2×1, etc. | ❌ Fixed 1×1×1 | +5-10% prefill |
| **min_latency_mode** | ✅ Decode optimization | ❌ Not wired | +2-4 tok/s |
| **Stage count tuning** | ✅ AUTO picks stages | ❌ Fixed AUTO | +5-15% |

### 5.3 Tile Configs Defined But Not Implemented

The `CutlassTileConfigSM120` enum has 6 configs, but only 1 is implemented:

```cpp
enum class CutlassTileConfigSM120 {
    Undefined,
    ChooseWithHeuristic,
    CtaShape128x128x128B,  // ✅ IMPLEMENTED (M=128, N=128, K in bytes)
    CtaShape128x128x64B,   // ❌ Not implemented
    CtaShape256x128x64B,   // ❌ Not implemented
    CtaShape128x256x64B,   // ❌ Not implemented
    CtaShape128x128x256B,  // ❌ Not implemented
    CtaShape256x128x128B,  // ❌ Not implemented
};
```

**Note:** These configs vary M and N but don't address the decode problem (small M).
We need to add small-M tiles (8, 16, 32) not covered by this enum.

---

## 6. Implementation Plan (JIT-Based)

### 6.1 Parameterize Tile Shape in JIT Template

**File:** `flashinfer/jit/batch_moe_gen.py`

```python
def gen_cutlass_fused_moe_sm120_module(
    tile_m: int = 128,  # Token dimension - PARAMETERIZED FOR DECODE
    tile_n: int = 128,  # Output dimension
    tile_k: int = 128,  # Reduction dimension
    use_fast_build: bool = False,
):
    """Generate SM120 MoE GEMM module with configurable tile shape."""
    
    # Include tile shape in cache key for separate caching
    module_name = f"fused_moe_120_M{tile_m}_N{tile_n}_K{tile_k}"
    
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

### 6.2 Add Tile Selection Logic in Python

**File:** `flashinfer/fused_moe/core.py`

```python
def select_tile_m_for_tokens(num_tokens: int) -> int:
    """Select optimal TILE_M based on number of tokens (M dimension).
    
    Uses thresholding: pick smallest tile that covers all tokens in one tile.
    This minimizes both padding waste AND number of tile launches.
    
    Mapping:
      M=1-8   → TILE_M=8   (1 tile, ≤7 slots wasted)
      M=9-16  → TILE_M=16  (1 tile, ≤7 slots wasted)
      M=17-32 → TILE_M=32  (1 tile, ≤15 slots wasted)
      M=33-64 → TILE_M=64  (1 tile, ≤31 slots wasted)
      M=65+   → TILE_M=128 (prefill-optimized)
    """
    VALID_TILE_M = [8, 16, 32, 64, 128]
    
    for tile_m in VALID_TILE_M:
        if num_tokens <= tile_m:
            return tile_m
    
    return 128  # Fallback for large batches


@functools.cache
def get_cutlass_fused_moe_module(
    backend: str = "120",
    tile_m: int = 128,  # Parameterized for decode optimization
    use_fast_build: bool = False,
):
    """Get JIT-compiled module for specific tile configuration.
    
    Results are cached by (backend, tile_m), so each tile variant
    is compiled only once and reused.
    """
    if backend in ("120", "121"):
        module = gen_cutlass_fused_moe_sm120_module(
            tile_m=tile_m,
            tile_n=128,  # Fixed for now
            tile_k=128,  # Fixed for now
            use_fast_build=use_fast_build,
        ).build_and_load()
    ...
    return module
```

### 6.3 Wire Through to Kernel Launch

**File:** `flashinfer/fused_moe/core.py`

```python
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    auto_tile_select: bool = True,  # Enable dynamic M-tile selection
    activation_type: ActivationType = ActivationType.Swiglu,
) -> torch.Tensor:
    
    num_tokens = input.shape[0]  # This becomes M in the GEMM
    
    # Select optimal TILE_M based on token count
    if auto_tile_select:
        tile_m = select_tile_m_for_tokens(num_tokens)
    else:
        tile_m = 128  # Default (prefill-optimized)
    
    # Get JIT-compiled module for this tile config
    # Cached after first call - subsequent calls are instant
    module = get_cutlass_fused_moe_module(
        backend="120",
        tile_m=tile_m,
    )
    
    # Launch kernel
    return module.run(input, ...)
```

### 6.4 Update C++ Template to Accept Tile Parameters

**File:** `moe_gemm_sm120_mixed_input_launcher.inl`

```cpp
// Template parameterized by tile shape (JIT provides values via -D flags)
#ifndef TILE_M
#define TILE_M 128  // Default
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif

namespace sm120_mxfp4_tile {

using TileShape_MNK = cute::Shape<
    cute::Int<TILE_M>,  // Token dimension (variable for decode)
    cute::Int<TILE_N>,  // Output dimension
    cute::Int<TILE_K>   // Reduction dimension
>;
using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

// ... rest of CUTLASS setup ...

}  // namespace sm120_mxfp4_tile
```

---

## 7. JIT Caching Behavior

### 7.1 Cache Structure

After running with different token counts, cache will contain:

```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120_M8_N128_K128/      # M ∈ [1, 8]   - decode
│   └── fused_moe_120_M8_N128_K128.so
├── fused_moe_120_M16_N128_K128/     # M ∈ [9, 16]  - decode
│   └── fused_moe_120_M16_N128_K128.so
├── fused_moe_120_M32_N128_K128/     # M ∈ [17, 32] - small batch
│   └── fused_moe_120_M32_N128_K128.so
├── fused_moe_120_M64_N128_K128/     # M ∈ [33, 64] - medium batch
│   └── fused_moe_120_M64_N128_K128.so
└── fused_moe_120_M128_N128_K128/    # M ∈ [65, ∞)  - prefill
    └── fused_moe_120_M128_N128_K128.so
```

**Thresholding guarantees:** Each M value maps to exactly one tile, and that tile
covers all tokens in a single CTA (no multi-tile overhead for decode).

### 7.2 First-Call vs Cached Performance

| Scenario | First Call | Cached Calls |
|----------|------------|--------------|
| Decode (M=1) | ~20s JIT compile | <1ms |
| Prefill (M=2048) | ~20s JIT compile | <1ms |

### 7.3 Pre-warming Strategy (Optional)

To avoid JIT latency during inference:

```python
def prewarm_moe_tiles():
    """Pre-compile all expected tile variants during server startup."""
    for tile_m in [8, 16, 32, 64, 128]:
        _ = get_cutlass_fused_moe_module(backend="120", tile_m=tile_m)
    print("All MoE TILE_M variants compiled and cached")
```

---

## 8. Expected Performance Impact

### 8.1 M-Dimension Utilization Improvement

| M (tokens) | Old TILE_M | New TILE_M | Utilization | Improvement |
|------------|------------|------------|-------------|-------------|
| 1 | 128 | 8 | 0.8% → 12.5% | **+15× util** |
| 2 | 128 | 8 | 1.6% → 25% | **+15× util** |
| 4 | 128 | 8 | 3.1% → 50% | **+16× util** |
| 8 | 128 | 8 | 6.3% → 100% | **+16× util** |

### 8.2 Projected Decode Throughput

| Configuration | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| Current (TILE_M=128 only) | ~50 | Baseline |
| After JIT M-tile selection | **52-56** | +4-12% |
| + mainloop tuning | **54-58** | +8-16% |

---

## 9. Implementation Steps

### Step 1: Parameterize JIT Template
**File:** `flashinfer/jit/batch_moe_gen.py`
- Add `tile_m`, `tile_n`, `tile_k` parameters
- Include in cache key (module name): `fused_moe_120_M{tile_m}_N{tile_n}_K{tile_k}`

### Step 2: Add Tile Selection Function
**File:** `flashinfer/fused_moe/core.py`
- Add `select_tile_m_for_tokens(num_tokens)` function
- Returns optimal TILE_M from [8, 16, 32, 64, 128]

### Step 3: Wire Through get_cutlass_fused_moe_module
**File:** `flashinfer/fused_moe/core.py`
- Add `tile_m` parameter
- Pass to JIT generator

### Step 4: Update C++ Launcher Template
**File:** `moe_gemm_sm120_mixed_input_launcher.inl`
- Use preprocessor defines for tile dimensions: `-DTILE_M=...`
- Ensure CUTLASS types work with variable tile shapes

### Step 5: (Optional) Add Pre-warming
**File:** `flashinfer/fused_moe/core.py`
- Add `prewarm_moe_tiles()` function
- Call during vLLM server startup

---

## 10. Validation Plan

1. **JIT compile test**: Each TILE_M variant compiles without errors
2. **Cache test**: Verify separate .so files per tile config
3. **Correctness test**: Output matches reference for each TILE_M
4. **Performance test**: Profile each tile for M=1,2,4,8,16,32,64,128
5. **Integration test**: Full vLLM decode benchmark
6. **Regression test**: Ensure prefill performance maintained

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Compiler "Error Internal" | Follow existing namespace pattern exactly |
| First-call JIT latency | Pre-warm during server startup |
| Cache key collision | Include full tile shape in module name |
| CUTLASS tile incompatibility | Test each TILE_M value individually |
| CUTLASS doesn't support small M-tiles | Verify SM120 MMA supports 8×N×K shape |

---

## 12. References

- **FlashInfer GEMM layout:** `csrc/nv_internal/.../moe_gemm_kernels.h` (line 53-55)
- **TRT-LLM tile configs:** `trtllmGen_bmm_export/config.json`
- **llama.cpp tile selection:** `ggml-cuda/mmq.cuh:3980-4048`
- **CUTLASS SM120 examples:** `cutlass/examples/92_blackwell_moe_gemm/`
- **FlashInfer JIT system:** `flashinfer/jit/`
- **FlashInfer autotuner:** `flashinfer/autotuner.py`
- **Current launcher:** `moe_gemm_sm120_mixed_input_launcher.inl`
