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

**Key insight:** In MoE grouped GEMM, M is **tokens per expert after routing**,
not total input tokens. For decode with 1 token and top_k=8:
- 8 experts each get M=1
- 120 experts get M=0 (skipped)
- Each active expert runs a GEMM with M=1

| Tokens/Expert | Current TILE_M | M-Dimension Utilization | Padding Waste |
|---------------|----------------|-------------------------|---------------|
| 1 | 128 | 0.8% | 127 token slots wasted |
| 2 | 128 | 1.6% | 126 token slots wasted |
| 4 | 128 | 3.1% | 124 token slots wasted |
| 8 | 128 | 6.3% | 120 token slots wasted |

For decode (1 token per expert), we compute a 128×N output tile but only need 1×N.
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

### 4.1 TRT-LLM MXFP4 Configs (SM100) - Reference Only

From `trtllmGen_bmm_export/config.json`:

| Template | Description | Tile Config | Use Case |
|----------|-------------|-------------|----------|
| `BatchedGemmMxE2m1E4m3LowLatency` | Decode-optimized | Small M-tile | **Decode** |
| `BatchedGemmMxE2m1MxE4m3HighThroughput` | Prefill-optimized | Large M-tile, cluster | Prefill |

**⚠️ CAUTION:** These are SM100 configs, not SM120. Key differences:
- SM100 and SM120 have different TMA/vectorization requirements
- Some tile sizes may not compile or perform poorly on SM120
- Treat as "worth investigating," not as proven solutions

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

**CRITICAL:** For MoE grouped GEMM, the relevant M is **tokens per expert after routing**,
not total input tokens. Each expert runs its own GEMM with its own M value.

### Understanding the Row Expansion

```
Input tokens:        num_tokens (e.g., 1 for decode)
Top-k experts:       top_k (e.g., 8 for gpt-oss-120b)
Expanded rows:       num_tokens × top_k = expanded_num_rows (e.g., 1 × 8 = 8)

After routing, these 8 expert-token pairs are distributed across experts:
- 8 active experts each get 1 row
- 120 inactive experts get 0 rows
```

The GEMM sees the **per-expert row counts**, not total tokens or expanded rows.

### Tile Selection Function

**⚠️ CRITICAL: Avoid GPU→CPU sync in hot path**

The naive approach (`rows_per_expert.max().item()`) forces a GPU sync every token,
which is catastrophic for decode latency. Use host-side heuristics instead.

```python
def select_tile_m_for_moe(num_tokens: int) -> int:
    """Select TILE_M using simple decode vs prefill heuristic.
    
    NO GPU TENSORS, NO .item() CALLS - these sync and kill decode perf.
    
    Just two cases:
    - Decode/small batch: TILE_M=16
    - Prefill/large batch: TILE_M=128
    """
    DECODE_THRESHOLD = 64  # Tune based on workload mix
    
    if num_tokens <= DECODE_THRESHOLD:
        return 16   # Decode-optimized
    else:
        return 128  # Prefill-optimized
```

**Why this simple approach works:**

1. **Correctness doesn't require tile ≥ max_rows_per_expert**
   - If max_rows > TILE_M, CUTLASS iterates multiple CTAs along M
   - Smaller tiles are still correct, just potentially slower

2. **Estimating max from skew can backfire**
   - If one expert gets a big blob but most are tiny, using max pessimizes the common case
   - p90 is better but requires GPU sync (which we can't do)

3. **Decode vs prefill covers 99% of cases**
   - Decode: always small M per expert → TILE_M=16
   - Prefill: large M per expert → TILE_M=128
   - Middle ground is rare; add 32/64 only if profiling shows need

**What NOT to do:**
```python
# ❌ BAD: Forces GPU→CPU sync every call
max_rows = rows_per_expert.max().item()

# ❌ BAD: Also syncs  
p90_rows = torch.quantile(rows_per_expert.float(), 0.9).item()

# ❌ BAD: Over-engineering - just use decode vs prefill threshold
estimated_max = (num_tokens * top_k / num_experts) * skew_factor
```

**Future optimization (if needed):**
If profiling shows the simple threshold isn't optimal, compute max_m on host
in the C++ GEMM setup (where per-expert sizes are already materialized).


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
    auto_tile_select: bool = True,
    activation_type: ActivationType = ActivationType.Swiglu,
) -> torch.Tensor:
    
    num_tokens = input.shape[0]  # Host-side value, no sync
    
    # Select TILE_M: just decode vs prefill, nothing fancy
    if auto_tile_select:
        tile_m = select_tile_m_for_moe(num_tokens)
    else:
        tile_m = 128  # Default
    
    # Get JIT-compiled module for this tile config
    module = get_cutlass_fused_moe_module(backend="120", tile_m=tile_m)
    
    # Launch kernel
    return module.run(input, ...)
```

**Key points:**
- Just `num_tokens` (host-side int) → no GPU sync
- Simple threshold: ≤64 tokens → decode tile, else prefill tile
- No estimation of max_rows_per_expert needed

### 6.4 Update C++ Template to Accept Tile Parameters

**File:** `moe_gemm_sm120_mixed_input_launcher.inl`

```cpp
// Tile config injected via -D flags from JIT
#ifndef TILE_M
#define TILE_M 128  // Default
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif

// Struct template for tile configuration (namespaces can't be templated)
template <int TM, int TN, int TK>
struct Sm120MxFP4TileConfig {
    using TileShape_MNK = cute::Shape<
        cute::Int<TM>,  // Token dimension (variable for decode)
        cute::Int<TN>,  // Output dimension
        cute::Int<TK>   // Reduction dimension
    >;
    using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
};

// Instantiate with JIT-provided tile sizes
using CurrentTileConfig = Sm120MxFP4TileConfig<TILE_M, TILE_N, TILE_K>;
using TileShape_MNK = CurrentTileConfig::TileShape_MNK;
using ClusterShape_MNK = CurrentTileConfig::ClusterShape_MNK;

// ... rest of CUTLASS kernel setup uses TileShape_MNK, ClusterShape_MNK ...
```

---

## 7. JIT Caching Behavior

### 7.1 Cache Structure

After running with the minimal tile set, cache will contain:

```
~/.cache/flashinfer/0.6.0/121a/cached_ops/
├── fused_moe_120_M16_N128_K128/     # decode / small batch
│   └── fused_moe_120_M16_N128_K128.so
└── fused_moe_120_M128_N128_K128/    # prefill / large batch
    └── fused_moe_120_M128_N128_K128.so
```

**Typical workload → tile mapping (gpt-oss-120b: 128 experts, top_k=8):**

| Workload | Input Tokens | Estimated Max/Expert | TILE_M |
|----------|--------------|---------------------|--------|
| Decode (1-8 req) | 1-8 | 1-2 | **16** |
| Small batch | 32 | ~4 | **16** |
| Medium batch | 128 | ~12 | **16** |
| Large batch | 512 | ~40 | **128** |
| Prefill | 2048 | ~200 | **128** |

**Start with 2 tiles.** Add 32/64 only if measurements show:
1. Significant time in 32-64 range workloads
2. Measurable latency improvement with finer tiles

### 7.2 First-Call vs Cached Performance

| Scenario | First Call | Cached Calls |
|----------|------------|--------------|
| Decode (M=1) | ~20s JIT compile | <1ms |
| Prefill (M=2048) | ~20s JIT compile | <1ms |

### 7.3 Multi-Worker Locking (REQUIRED for vLLM)

vLLM runs multiple workers per GPU. Without locking, all workers race to compile:

```
Worker 0: compiling fused_moe_120_M16...
Worker 1: compiling fused_moe_120_M16...  ← WASTED WORK
Worker 2: compiling fused_moe_120_M16...  ← WASTED WORK
...
```

**Solution: File lock per module in cache directory**

```python
import fcntl
from pathlib import Path

@functools.cache
def get_cutlass_fused_moe_module(backend: str, tile_m: int, ...):
    cache_dir = Path.home() / ".cache/flashinfer/0.6.0/121a/cached_ops"
    module_name = f"fused_moe_120_M{tile_m}_N128_K128"
    lock_file = cache_dir / f"{module_name}.lock"
    
    # Ensure cache dir exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Acquire exclusive lock before build
    with open(lock_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Blocks until lock acquired
        try:
            # Check if another worker already compiled while we waited
            if module_already_cached(module_name):
                return load_cached_module(module_name)
            
            # We're the first - compile it
            module = gen_cutlass_fused_moe_sm120_module(
                tile_m=tile_m, ...
            ).build_and_load()
            return module
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

**Note:** FlashInfer's existing JIT may already have locking. Check before implementing.

### 7.4 Pre-warming Strategy (Recommended)

To avoid JIT latency during inference:

```python
def prewarm_moe_tiles():
    """Pre-compile tile variants during server startup.
    
    Start with just 2 tiles to minimize startup time:
    - 16: decode and small batches
    - 128: prefill and large batches
    
    At ~20s per tile, this is ~40s startup cost.
    Add more tiles (32, 64) only after measuring real gains.
    """
    TILES_TO_PREWARM = [16, 128]  # Minimal set
    for tile_m in TILES_TO_PREWARM:
        _ = get_cutlass_fused_moe_module(backend="120", tile_m=tile_m)
    print(f"Prewarmed {len(TILES_TO_PREWARM)} MoE tile variants")
```

**Startup cost:**
| Tiles | Compile Time | Use Case |
|-------|--------------|----------|
| 2 (16, 128) | ~40s | **Recommended starting point** |
| 3 (16, 64, 128) | ~60s | Add 64 if medium batches are common |
| 5 (8-128) | ~100s | Only if all sizes show measured gains |

---

## 8. Expected Performance Impact

### 8.0 REQUIRED: Measure MoE GEMM Fraction First

**Before implementing ANY tile variants, measure the baseline:**

```bash
# Option 1: Nsight Systems (most accurate)
nsys profile -o decode_profile \
  python -c "import vllm; ... # run decode workload"
nsys stats decode_profile.nsys-rep --report cuda_gpu_kern_sum

# Option 2: NVTX ranges (if instrumented)
# Look for moe_gemm or cutlass kernel names

# Option 3: torch.profiler with CUDA activities
with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
    # run decode
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Decision gate:**

| MoE GEMM % of Decode | Action |
|---------------------|--------|
| **< 10%** | ❌ STOP - not worth optimizing tiles |
| **10-30%** | ⚠️ Proceed cautiously, expect modest gains |
| **> 30%** | ✅ Tile optimization is worthwhile |

**From prior analysis (Section 3.2 of SM121_OPTIMIZATION_ANALYSIS.md):**
- MoE GEMM estimated at ~61% of decode time
- This justifies tile optimization work

**Re-measure after each change** to validate actual gains.

---

### 8.1 M-Dimension Utilization Improvement

**Per-expert M (tokens routed to each expert):**

| Max Tokens/Expert | Old TILE_M | New TILE_M | Utilization | Improvement |
|-------------------|------------|------------|-------------|-------------|
| 1-16 (decode) | 128 | 16* | 0.8-12.5% → 6.3-100% | **+8× util** |
| 17-32 | 128 | 32* | 13-25% → 53-100% | **+4× util** |
| 33-64 | 128 | 64* | 26-50% → 52-100% | **+2× util** |
| 65+ | 128 | 128 | 51-100% | (no change) |

*Conservative starting point. Smaller tiles (8) may work but require validation.

**⚠️ Utilization improvement ≠ throughput improvement**
- Higher utilization reduces wasted compute
- But kernel launch overhead, memory bandwidth, and other factors also matter
- Actual tok/s gain will be lower than utilization gain suggests

**Note:** These are per-expert M values after routing, not total input tokens.

### 8.2 Projected Decode Throughput (Conditional)

**Only valid if MoE GEMM is >30% of decode time:**

| Configuration | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| Current (TILE_M=128 only) | ~50 | Baseline |
| After JIT M-tile selection | **52-56** | +4-12% (if MoE is ~60% of decode) |
| + mainloop tuning | **54-58** | +8-16% |

**Scaling formula:**
```
Expected gain = (utilization improvement) × (MoE GEMM fraction) × (efficiency factor)
              ≈ 8× × 0.6 × 0.1  (very rough)
              ≈ 50% theoretical → 5-10% realistic
```

---

## 9. Implementation Steps

### Step 1: Validate Tile Sizes on SM120 (BEFORE coding)
**Critical:** Not all tile sizes work on SM120. Before implementing:
1. Check CUTLASS SM120 examples for supported tile shapes
2. Verify TMA alignment requirements (often 16-element minimum)
3. Check vectorized epilogue constraints
4. Compile a test kernel with each proposed tile size

**Recommended validation order (minimal approach):**
- ✅ 128×128×128 - known working (prefill)
- 🔬 16×128×128 - validate first (decode) - may need alignment adjustments
- ⏸️ 32, 64 - defer until 16+128 are working and measured
- ⏸️ 8 - defer; may not work with TMA/vectorized epilogues

### Step 2: Parameterize JIT Template
**File:** `flashinfer/jit/batch_moe_gen.py`
- Add `tile_m` parameter (keep tile_n, tile_k fixed initially)
- Include in cache key: `fused_moe_120_M{tile_m}_N128_K128`

### Step 3: Add Tile Selection Function
**File:** `flashinfer/fused_moe/core.py`
- Add `select_tile_m_for_moe()` function using host-side heuristics
- Start with minimal set: **[16, 128]** (2 tiles = ~40s compile)
- Add 32/64 only after measuring real workload gains

### Step 4: Wire Through get_cutlass_fused_moe_module
**File:** `flashinfer/fused_moe/core.py`
- Add `tile_m` parameter
- Pass to JIT generator

### Step 5: Update C++ Launcher Template
**File:** `moe_gemm_sm120_mixed_input_launcher.inl`
- Use struct template with -DTILE_M preprocessor define
- Ensure CUTLASS types work with variable tile shapes

### Step 6: Benchmark Each Tile Size
- Profile decode latency for each validated tile
- Smaller tiles may have higher overhead that offsets utilization gains
- Pick based on measured performance, not theory

### Step 7: (Optional) Add Pre-warming
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
| **GPU sync in tile selection** | Use host-side heuristics ONLY (Section 6.2) |
| Compiler "Error Internal" | Follow existing namespace pattern exactly |
| First-call JIT latency | Pre-warm during server startup |
| Cache key collision | Include full tile shape in module name |
| **Multi-worker JIT race** | File lock per module in cache dir (Section 7.3) |
| **Small M-tiles don't compile** | Start with 32/64, add smaller tiles incrementally |
| **TMA alignment violations** | Check SM120 TMA requirements (often 16-element min) |
| **Vectorized epilogue fails** | May need N≥16; test before assuming N=8 works |
| **Overhead exceeds savings** | Benchmark each tile; smaller isn't always faster |
| SM100 configs don't transfer | Don't assume TRT-LLM SM100 tiles work on SM120 |

---

## 12. References

- **FlashInfer GEMM layout:** `csrc/nv_internal/.../moe_gemm_kernels.h` (line 53-55)
- **TRT-LLM tile configs:** `trtllmGen_bmm_export/config.json`
- **llama.cpp tile selection:** `ggml-cuda/mmq.cuh:3980-4048`
- **CUTLASS SM120 examples:** `cutlass/examples/92_blackwell_moe_gemm/`
- **FlashInfer JIT system:** `flashinfer/jit/`
- **FlashInfer autotuner:** `flashinfer/autotuner.py`
- **Current launcher:** `moe_gemm_sm120_mixed_input_launcher.inl`
