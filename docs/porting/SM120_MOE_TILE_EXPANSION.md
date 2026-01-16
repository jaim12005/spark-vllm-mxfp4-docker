# SM120 MoE Tile Expansion Plan

> **⚠️ KEY FINDING (2026-01-14):** The tcgen05 tensor core has a **hardware minimum of M=64**. 
> Tiles smaller than 64 in the M dimension are impossible. The solution is to **swap M and N**
> dimensions, leveraging N's minimum of 8. See [Section 15](#15-hardware-constraint-tcgen05-minimum-tile-sizes).

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

> **Hardware Constraint:** We cannot simply reduce TILE_M below 64 because tcgen05 tensor 
> cores have a **minimum M tile size of 64**. See [Section 15](#15-hardware-constraint-tcgen05-minimum-tile-sizes)
> for the M/N swap solution that works around this limitation.

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

**File:** `flashinfer/jit/fused_moe.py` (NOT `batch_moe_gen.py`)

**Critical refactoring needed:** The existing `gen_cutlass_fused_moe_module()` helper
doesn't accept a custom module name. You must modify it to prevent cache collisions.

```python
# Current signature (simplified):
def gen_cutlass_fused_moe_module(backend: str, ...):
    # Module name is auto-generated, doesn't include tile shape
    ...

# REQUIRED modification:
def gen_cutlass_fused_moe_module(
    backend: str,
    tile_m: int = 128,  # NEW: Token dimension
    tile_n: int = 128,  # Output dimension  
    tile_k: int = 128,  # Reduction dimension
    use_fast_build: bool = False,
):
    """Generate MoE GEMM module with configurable tile shape.
    
    CRITICAL: Include tile shape in module_name for cache separation.
    Without this, all tile variants collide in the cache!
    """
    
    # Include tile shape in cache key for separate caching
    # e.g., "fused_moe_120_M64_N128_K128" vs "fused_moe_120_M128_N128_K128"
    module_name = f"fused_moe_{backend}_M{tile_m}_N{tile_n}_K{tile_k}"
    
    return JitModule(
        name=module_name,  # ← This is the cache key
        sources=[...],
        extra_cuda_cflags=[
            f"-DTILE_M={tile_m}",
            f"-DTILE_N={tile_n}",
            f"-DTILE_K={tile_k}",
        ],
        ...
    )
```

**Why this matters:**
- JIT cache is keyed by `module_name`
- Without tile shape in name: `fused_moe_120` caches once, all tiles collide
- With tile shape in name: `fused_moe_120_M64_...` and `fused_moe_120_M128_...` are separate

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
    - Decode/small batch: TILE_M=64 (safest small tile)
    - Prefill/large batch: TILE_M=128
    
    Start with 64 (likely compiles cleanly), then try 32/16 if gains justify.
    """
    DECODE_THRESHOLD = 64  # Tune based on workload mix
    
    if num_tokens <= DECODE_THRESHOLD:
        return 64   # Decode-optimized (start safe, reduce later if validated)
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

After running with all tile variants, cache will contain:

```
$(flashinfer_cache_dir)/cached_ops/   # Use FlashInfer's cache dir, not hardcoded path
├── fused_moe_120_M8/                 # decode (1-8 tokens), transpose mode
├── fused_moe_120_M16/                # decode (9-16 tokens), transpose mode
├── fused_moe_120_M32/                # decode (17-32 tokens), transpose mode
├── fused_moe_120/                    # default (128), standard mode
└── fused_moe_120_M256/               # large batch prefill, standard mode
```

**Note:** The actual path varies by FlashInfer version and GPU architecture.
Use FlashInfer's cache dir helper (e.g., `flashinfer.jit.get_cache_dir()`) rather
than hardcoding `~/.cache/flashinfer/0.6.0/121a/...`.

**Typical workload → tile mapping (gpt-oss-120b: 128 experts, top_k=8):**

| Workload | Input Tokens | TILE_M | Mode |
|----------|--------------|--------|------|
| Single decode | 1-8 | **8** | Transpose |
| Small batch decode | 9-16 | **16** | Transpose |
| Medium batch | 17-32 | **32** | Transpose |
| Prefill / Large batch | 33-128 | **128** | Standard |
| Very large batch | 129+ | **256** | Standard |

### 7.2 First-Call vs Cached Performance

| Scenario | First Call | Cached Calls |
|----------|------------|--------------|
| Decode (M=1) | ~20s JIT compile | <1ms |
| Prefill (M=2048) | ~20s JIT compile | <1ms |

### 7.3 Multi-Worker Locking (Check Before Implementing)

vLLM runs multiple workers per GPU. Without locking, all workers race to compile:

```
Worker 0: compiling fused_moe_120_M64...
Worker 1: compiling fused_moe_120_M64...  ← WASTED WORK
Worker 2: compiling fused_moe_120_M64...  ← WASTED WORK
...
```

**⚠️ BEFORE IMPLEMENTING: Check if FlashInfer already has locking**

```python
# Check FlashInfer's JIT implementation for existing locks:
# - flashinfer/jit/core.py or similar
# - Look for fcntl, filelock, or threading.Lock usage
# - Double-locking can cause deadlocks!

# If FlashInfer already locks: DO NOTHING, just use their system
# If FlashInfer doesn't lock: Add locking as shown below
```

**Solution (only if FlashInfer doesn't already lock):**

```python
import fcntl
from flashinfer.jit import get_cache_dir  # Use FlashInfer's helper, NOT hardcoded path

@functools.cache
def get_cutlass_fused_moe_module(backend: str, tile_m: int, ...):
    # ❌ DON'T hardcode: Path.home() / ".cache/flashinfer/0.6.0/121a/..."
    # ✅ DO use FlashInfer's cache dir helper (version/arch agnostic)
    cache_dir = get_cache_dir()  # Or however FlashInfer exposes this
    
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

**Implementation checklist:**
- [ ] Check if FlashInfer JIT already locks builds
- [ ] If yes: skip adding our own lock
- [ ] If no: use FlashInfer's cache dir helper, not hardcoded path
- [ ] Test with multiple workers to verify no deadlock

### 7.4 Pre-warming Strategy (Recommended)

To avoid JIT latency during inference:

```python
def prewarm_moe_tiles():
    """Pre-compile tile variants during server startup.
    
    Pre-warm commonly used tiles:
    - 8, 16, 32: Decode-optimized (transpose mode)
    - 128: Prefill-optimized (standard mode)
    - 256: Large batch (optional, only if needed)
    
    At ~20s per tile, full set is ~80-100s startup cost.
    """
    TILES_TO_PREWARM = [8, 16, 32, 128]  # Core set
    # Add 256 if you expect very large batches
    for tile_m in TILES_TO_PREWARM:
        _ = get_cutlass_fused_moe_module(backend="120", tile_m=tile_m)
    print(f"Prewarmed {len(TILES_TO_PREWARM)} MoE tile variants")
```

**Startup cost:**
| Tiles | Compile Time | Use Case |
|-------|--------------|----------|
| 2 (32, 128) | ~40s | Minimal decode + prefill |
| 4 (8, 16, 32, 128) | ~80s | **Recommended for full decode optimization** |
| 5 (8, 16, 32, 128, 256) | ~100s | Full set including large batch |

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

**Recommended validation order:**

Small TILE_M can break TMA transaction shapes, epilogue vectorization, and 
warp-specialized scheduling. Validate from safest to riskiest:

| Order | TILE_M | Risk | Notes |
|-------|--------|------|-------|
| 1 | 128 | ✅ None | Known working (current) |
| 2 | 64 | 🟢 Low | Likely compiles cleanly |
| 3 | 32 | 🟢 Low | Likely compiles cleanly |
| 4 | 16 | 🟡 Medium | May need alignment adjustments |
| 5 | 8 | 🔴 High | Try only if 16 works; TMA/vectorization issues likely |

**Implementation strategy:**
1. Implement 64 first (safest small tile)
2. Benchmark 64 vs 128 on real decode workload
3. If 64 shows gains, try 32
4. If 32 works, try 16
5. Only try 8 if 16 is clean AND you need more granularity

### Step 2: Parameterize JIT Template
**File:** `flashinfer/jit/fused_moe.py`
- Modify `gen_cutlass_fused_moe_module()` to accept `tile_m` parameter
- **Critical:** Include tile_m in `module_name` for cache separation
- Cache key format: `fused_moe_120_M{tile_m}_N128_K128`

### Step 3: Add Tile Selection Function
**File:** `flashinfer/fused_moe/core.py`
- Add `select_tile_m_for_moe()` function using host-side threshold
- Start with minimal safe set: **[64, 128]** (2 tiles = ~40s compile)
- Add 32, then 16, then 8 only after each is validated

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
7. **Skewed routing stress test**: See below

### 10.1 Skewed Routing Stress Test

The performance mapping assumes roughly uniform routing. Reality can be skewed.
This test validates whether decode/prefill gating is sufficient.

**Test case: One hot expert**
```python
def test_skewed_routing():
    """Stress test: one expert gets most tokens, others get few."""
    num_tokens = 64
    num_experts = 128
    top_k = 8
    
    # Construct skewed routing: expert 0 gets 80% of assignments
    # This creates max_rows_per_expert >> avg_rows_per_expert
    token_selected_experts = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    for i in range(num_tokens):
        if i < int(num_tokens * 0.8):  # 80% to expert 0
            token_selected_experts[i] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        else:  # 20% distributed
            token_selected_experts[i] = torch.randint(0, num_experts, (top_k,))
    
    # Expected distribution:
    # - Expert 0: ~51 rows (80% of 64 tokens)
    # - Other experts: ~0-2 rows each
    # - avg_rows_per_expert ≈ 4
    # - max_rows_per_expert ≈ 51
    
    # With TILE_M=64 (our decode tile):
    # - Expert 0: 51 rows fits in one 64-tile (OK)
    # - With TILE_M=16: would need 4 CTAs for expert 0
    
    # Run both tiles
    output_64 = run_moe_with_tile(input, token_selected_experts, tile_m=64)
    output_128 = run_moe_with_tile(input, token_selected_experts, tile_m=128)
    
    # Verify correctness
    assert torch.allclose(output_64, output_128, rtol=1e-3)
    
    # Measure performance
    time_64 = benchmark(lambda: run_moe_with_tile(..., tile_m=64))
    time_128 = benchmark(lambda: run_moe_with_tile(..., tile_m=128))
    
    print(f"Skewed routing: TILE_M=64: {time_64:.2f}ms, TILE_M=128: {time_128:.2f}ms")
    # If 64 is slower than 128, may need max-based selection for skewed cases
```

**What this test reveals:**
| Result | Implication |
|--------|-------------|
| 64 ≈ 128 (within 10%) | Decode/prefill gating is sufficient |
| 64 slower than 128 | May need max-based selection for skewed routing |
| 64 much faster | Tile expansion is working as intended |
| Correctness failure | CUTLASS multi-CTA iteration has a bug |

**Extreme skew test (optional):**
```python
# All tokens to one expert
token_selected_experts[:, :] = 0  # Expert 0 gets everything
# max_rows_per_expert = num_tokens * top_k = 512
# With TILE_M=64: needs 8 CTAs along M
# Verify this still produces correct output
```

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

## 13. Status Update (2026-01-14)

### Implemented (v2 - Macro-based with M/N Swap)

**Supported Tile Sizes**: `[8, 16, 32, 128, 256]`

| TILE_M | Mode | Namespace | Description |
|--------|------|-----------|-------------|
| 8 | Transpose | `sm120_mxfp4_bf16_transposed_8` | Decode-optimized (N=8) |
| 16 | Transpose | `sm120_mxfp4_bf16_transposed_16` | Decode-optimized (N=16) |
| 32 | Transpose | `sm120_mxfp4_bf16_transposed_32` | Decode-optimized (N=32) |
| 128 | Standard | `sm120_mxfp4_bf16_128x128x128` | Prefill-optimized |
| 256 | Standard | `sm120_mxfp4_bf16_256x256x128` | Large batch prefill |

**Key Changes:**
- **Macro-based namespaces**: `DEFINE_SM120_MXFP4_STANDARD_NAMESPACE` and `DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE` reduce duplication
- **M/N Swap for small tiles**: Tiles < 128 use transpose mode to work around tcgen05 M≥64 constraint
- **Python threshold selection**: `select_tile_m_for_sm120()` picks smallest tile that fits token count
- **JIT cache separation**: Each tile size compiles to separate `.so` file

### Implementation Details

**C++ Launcher** (`moe_gemm_sm120_mixed_input_launcher.inl`):
- Two macros generate namespace contents (standard vs transposed)
- `#if TILE_M < 128` selects transpose mode, swapping A/B pointers
- Transposed mode: ElementInputA=FP4, ElementInputB=FP8, LayoutC=ColumnMajor

**Python Selection** (`flashinfer/fused_moe/core.py`):
```python
SM120_SUPPORTED_TILE_M = (8, 16, 32, 128, 256)
SM120_TILE_THRESHOLDS = [
    (8, 8),      # 1-8 tokens -> TILE_M=8
    (16, 16),    # 9-16 tokens -> TILE_M=16
    (32, 32),    # 17-32 tokens -> TILE_M=32
    (128, 128),  # 33-128 tokens -> TILE_M=128
]
SM120_DEFAULT_TILE_M = 256  # > 128 tokens
```

### Resolved Issues
- **M=64 removed**: The tcgen05 M≥64 constraint made TILE_M=64 problematic. Replaced with M/N swap.
- **TILE_M=64 references removed**: No longer used; smallest standard tile is 128.

---

## 15. Hardware Constraint: tcgen05 Minimum Tile Sizes

### 15.1 The Fundamental Constraint

On SM120/SM121 (Blackwell), the `tcgen05` tensor core instructions have **asymmetric minimum tile requirements**:

| Dimension | Minimum Tile Size | Notes |
|-----------|-------------------|-------|
| **M** | **64** | Token dimension - this is our bottleneck |
| **N** | **8** | Output dimension - much more flexible |
| **K** | Varies | Reduction dimension |

**This explains the tile size limits**: The hardware cannot execute an MMA with M < 64 for tcgen05. Tiles smaller than 64 (32, 16, 8) are impossible. TILE_M=64 should work as it's at the minimum, but our compilation failure (Section 13) may be due to other CUTLASS/TMA constraints, not this hardware minimum. The M/N swap approach sidesteps this entirely by making the small dimension N instead.

### 15.2 The M/N Swap Solution (FlashInfer PR 2327)

**Key insight**: Since M minimum is 64 but N minimum is only 8, we can **swap the matrix operands** to put the small dimension in N instead of M.

**Standard GEMM layout:**
```
C[M,N] = A[M,K] × B[K,N]

For decode (1 token per expert):
  M = 1 (tokens) ← PROBLEM: minimum is 64!
  N = 14336 (intermediate_size)
  K = 5120 (hidden_size)
```

**Swapped GEMM layout:**
```
C^T[N,M] = B^T[N,K] × A^T[K,M]

After transpose/swap:
  M' = 14336 (was N) ← Now the large dimension, tile M = 128
  N' = 1 (was M) ← Now the small dimension, tile N = 8/16/32
  K' = 5120 (unchanged)
```

**⚠️ CONSTRAINT: After swapping, the new M dimension (original N) must be divisible by 128.**

This means the model's `intermediate_size` and `hidden_size` must be multiples of 128:

| Model | intermediate_size | hidden_size | Divisible by 128? |
|-------|------------------|-------------|-------------------|
| gpt-oss-120b | 14336 | 5120 | ✅ 14336/128=112, 5120/128=40 |
| Llama-3-70B | 28672 | 8192 | ✅ 28672/128=224, 8192/128=64 |
| Mixtral-8x7B | 14336 | 4096 | ✅ 14336/128=112, 4096/128=32 |

Most models satisfy this since hidden sizes are typically powers of 2 or multiples of 256.

**Implementation approach:**
1. Swap A and B operand pointers in the GEMM call
2. Tile shape: M=128 (fixed), N=8/16/32 (small token dim)
3. Output LayoutC = ColumnMajor (writes D^T which matches D in row-major)

**Reference:** [FlashInfer PR 2327](https://github.com/flashinfer-ai/flashinfer/pull/2327)

### 15.3 Post-Swap: N Not Divisible by 64 (CUTLASS PR 2946)

After swapping, the original N dimension (e.g., 14336) becomes M, and the original M (tokens) becomes N.

**New problem**: After the swap, what was the fixed output dimension (14336) is now M. Since M must be processed in tiles of at least 64, the epilogue needs to handle cases where this isn't cleanly divisible.

**CUTLASS PR 2946** adds support for scenarios where:
- The swapped-N dimension (original M = tokens) is small (1-16)
- The swapped-M dimension (original N = output dim) may not be divisible by 64

**Reference:** [CUTLASS PR 2946](https://github.com/NVIDIA/cutlass/pull/2946)

### 15.4 Implementation Path Forward

Given the hardware constraint, our implementation uses M/N swap for small tiles:

| Approach | Complexity | Expected Gain | Status |
|----------|------------|---------------|--------|
| **TILE_M=128, 256 (standard)** | None | Baseline | ✅ Implemented |
| **TILE_M=8, 16, 32 (transpose)** | Medium | Full decode optimization (N min=8) | ✅ **Implemented** |
| **TILE_M=64 (standard)** | N/A | N/A | ❌ Removed (tcgen05 M≥64 constraint) |

**Implementation approach (completed):**
1. ✅ **Macro-based namespaces**: Reduce duplication with `DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE`
2. ✅ **M/N swap in launcher**: `#if TILE_M < 128` swaps A/B pointers and uses ColumnMajor output
3. ✅ **Python tile selection**: Threshold-based selection picks smallest tile that fits
4. 🔄 **Benchmark**: Validate decode performance gains with real workloads

## 14. Performance Analysis (Theoretical)

Since the `TILE_M=64` kernel currently falls back to 128 due to compilation issues, we cannot measure actual GPU time reduction yet. However, we have validated the selection logic and calculated the theoretical work reduction:

| Scenario | Logical Rows | Computed Rows (M=128) | Computed Rows (M=64) | Work Reduction |
|----------|--------------|-----------------------|----------------------|----------------|
| **1 Token Decode** | 8 | 1024 | 512 | **50%** |
| **8 Tokens Decode** | 64 | 8192 | 4096 | **50%** |
| **Skewed (64 tok)** | 512 | 13696 | 7040 | **48.6%** |

The Python-side selection logic correctly picks `TILE_M=64` for <= 64 tokens, ensuring that once the kernel compilation is fixed, these savings will be realized automatically.

## 16. CUTLASS Framework Patches for N < 128 Support (2026-01-15)

### 16.1 Root Cause Analysis

The original CUTLASS SM120 block-scaled grouped GEMM code did not support N < 128 due to scale factor TMA descriptor requirements. This was a **framework limitation**, not a hardware limitation.

**Key difference from SM100:**
- SM100's `sm100_blockscaled_mma_warpspecialized.hpp` has explicit `IsCtaN64` handling
- SM120's `sm120_blockscaled_mma_array_tma.hpp` (grouped GEMM) lacked this handling

### 16.2 Patches Applied

**File 1: `cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`**

1. Added `TileN_SFB` with ceil_div to pad N to 128:
```cpp
using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;  // = 128
static constexpr int TileN_SFB = cutlass::ceil_div(cute::size<1>(TileShape{}), Blk_MN{}) * Blk_MN{};
using TileShape_SFB = decltype(cute::make_shape(cute::Int<TileN_SFB>{}, cute::size<2>(TileShape{})));
```

2. Added `IsCtaN64` flag:
```cpp
static constexpr bool IsCtaN64 = cute::size<1>(TileShape{}) == 64;
```

3. Updated TMA_SFB type to use `TileShape_SFB{}` instead of `make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))`

4. Made MMA assertion conditional:
```cpp
if constexpr (!IsCtaN64) { CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum)); }
```

**File 2: `cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`**

Applied same ceil_div logic for SmemLayoutSFB:
```cpp
static constexpr int TileN_SFB = cutlass::ceil_div(cute::size<1>(TileShape_MNK{}), Blk_MN{}) * Blk_MN{};
using sSFB_shapeN = decltype(prepend(Int<TileN_SFB>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<TileN_SFB>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
```

### 16.3 Verified Tile Configurations

All three configurations now compile successfully:
| Tile (M, N) | SWAP_AB | Physical Tile | Use Case |
|-------------|---------|---------------|----------|
| (128, 128)  | 0       | (128, 128)    | Prefill  |
| (128, 64)   | 0       | (128, 64)     | Smaller K |
| (64, 128)   | 1       | (128, 64)     | Decode (small M) |

### 16.4 Next Steps

1. Runtime testing to verify correctness
2. Performance benchmarking with different tile sizes
3. Consider upstreaming patches to CUTLASS

---

## 17. M=64 Native Hardware Support (2026-01-15)

### 17.1 Discovery

Investigation revealed that the `tcgen05.mma` hardware instruction **natively supports M=64**. The constraint was not in the hardware but in the CUTLASS framework's scale factor layout computation.

From `cute/arch/mma_sm100_umma.hpp`:
```cpp
static_assert(M == 64 || M == 128, "SM100_MMA_TF32 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
```

And from the descriptor encoding (`mma_sm100_desc.hpp`):
```cpp
m_dim_: 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
```

### 17.2 CUTLASS Framework Patches for M < 128 and N < 128

The same issue that affected N < 128 (see Section 16) also affects M < 128. Applied the same fix using `cute::ceil_div` (NOT `cutlass::ceil_div` - see note below):

**File: `sm120_blockscaled_mma_builder.inl`**
```cpp
// M dimension must be rounded up to at least Blk_MN (128) for TMA and UTCCP to work.
static constexpr int TileM_SFA = cute::ceil_div(cute::size<0>(TileShape_MNK{}), Blk_MN{}) * Blk_MN{};
using sSFA_shapeM  = decltype(prepend(Int<TileM_SFA>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<TileM_SFA>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));

// N dimension padding for SFB (same approach)
static constexpr int TileN_SFB = cute::ceil_div(cute::size<1>(TileShape_MNK{}), Blk_MN{}) * Blk_MN{};
```

**File: `sm120_blockscaled_mma_array_tma.hpp`**
```cpp
// Scale factor A tile shape - M dimension padded to at least 128 for TMA
static constexpr int TileM_SFA = cute::ceil_div(cute::size<0>(TileShape{}), Blk_MN{}) * Blk_MN{};
using TileShape_SFA = decltype(cute::make_shape(cute::Int<TileM_SFA>{}, cute::size<2>(TileShape{})));
static constexpr bool IsCtaMSmall = cute::size<0>(TileShape{}) < 128;

// Scale factor B tile shape - N dimension padded to at least 128 for TMA
static constexpr int TileN_SFB = cute::ceil_div(cute::size<1>(TileShape{}), Blk_MN{}) * Blk_MN{};
using TileShape_SFB = decltype(cute::make_shape(cute::Int<TileN_SFB>{}, cute::size<2>(TileShape{})));
static constexpr bool IsCtaNSmall = cute::size<1>(TileShape{}) < 128;

// Skip size assertions for padded cases
if constexpr (!IsCtaMSmall) { CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum)); }
if constexpr (!IsCtaNSmall) { CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum)); }
```

**CRITICAL: Use `cute::ceil_div`, NOT `cutlass::ceil_div`**

The `cutlass::ceil_div` function is designed for runtime integer division, while `cute::ceil_div` handles compile-time `cute::Int` constants correctly. Using the wrong one causes template instantiation failures in TMA copy creation.

### 17.3 Eliminating SWAP_AB

With M=64 natively supported, the `SWAP_AB` (transposed mode) hack is no longer needed:

**File: `flashinfer/jit/fused_moe.py`**
```python
# swap_ab (transposed mode) is no longer needed.
# The tcgen05 hardware supports M=64 directly, and we've patched CUTLASS to handle
# the scale factor layout padding for M < 128.
swap_ab = False
```

### 17.4 Verified Tile Configurations

All supported tile configurations now compile successfully:

| Tile (M, N) | Scale Factor Padding | Notes |
|-------------|---------------------|-------|
| (64, 8) | SFA, SFB padded | Smallest N |
| (64, 16) | SFA, SFB padded | Small N |
| (64, 32) | SFA, SFB padded | Minimal decode |
| (64, 64) | SFA, SFB padded | Small decode |
| (64, 128) | SFA padded | Standard decode |
| (64, 256) | SFA padded | Large N decode |
| (128, 8) | SFB padded | Smallest N |
| (128, 16) | SFB padded | Small N |
| (128, 32) | SFB padded | |
| (128, 64) | SFB padded | |
| (128, 128) | None | Standard, default |

**Note:** (128, 256) exceeds shared memory capacity (requires Stages >= 2, only 1 fits).

**Constraints:**
- M must be multiple of 64 (tcgen05 hardware minimum)
- N must be power of 2: 8, 16, 32, 64, 128, or 256 (smem layout atom constraint)
- Non-power-of-2 N values fail due to `ldmatrix` copy atom alignment requirements

### 17.5 Shared Memory Constraints

SM120 (GB10) has limited shared memory compared to SM100 (Blackwell B200):

```
SM120 (GB10):     101,376 bytes (~99 KB)
SM100 (B200):     232,448 bytes (~227 KB)
```

These values can be queried from the hardware:
```python
import torch
props = torch.cuda.get_device_properties(0)
print(props.shared_memory_per_block_optin)  # 101376 on GB10
```

**Shared Memory Budget:**

The total shared memory is divided between:
1. **Epilogue carveout** - Reserved for output staging and fusion operations
2. **Mainloop stages** - A/B tensors, scale factors, pipeline barriers

**Epilogue Optimization:**

The original SM120 epilogue used a prescriptive (64, 32) tile, wasting ~6 KB of shared memory.
We optimized this to (64, 16) for most tiles:

| Tile | Old Epilogue | New Epilogue | Savings |
|------|--------------|--------------|---------|
| (64, 32+) | (64, 32) = 13,312 bytes | (64, 16) = 7,168 bytes | 6,144 bytes |
| (128, 32+) | (64, 32) = 13,312 bytes | (64, 16) = 7,168 bytes | 6,144 bytes |

**Hardware Constraint:** Epilogue tile M must be divisible by MMA tile M (64), limiting how small we can make the epilogue.

**Pipeline Stage Calculation:**

CUTLASS requires >= 2 pipeline stages for double-buffering (latency hiding). Stage count is:
```
stages = (smem_capacity - epilogue_carveout) / stage_bytes
```

Where `stage_bytes` includes:
- A tensor (FP8 stored as uint8_t): M × K bytes
- B tensor (FP4 stored as uint8_t): N × K bytes (NOT N × K / 2!)
- Scale factors A: ~512 bytes
- Scale factors B: ~N × 4 bytes
- Pipeline barriers: ~16 bytes

**Why (128, 256) Fails:**

Even with optimized epilogue (7,168 bytes):
```
Available for mainloop: 101,376 - 7,168 = 94,208 bytes
Per stage:
  A: 128 × 128 = 16,384 bytes
  B: 256 × 128 = 32,768 bytes (FP4 stored as uint8_t!)
  SF: ~1,536 bytes
  Pipeline: ~16 bytes
  Total: ~50,704 bytes

Stages: 94,208 / 50,704 = 1.86 (need >= 2)
```

**Key Insight:** The B tensor uses `uint8_t` storage (not packed FP4), doubling its footprint.
If CUTLASS supported true 4-bit packing, B would be 16,384 bytes and (128, 256) would fit.

### 17.6 Complete Tile Support with swap_ab

With `swap_ab=True`, logical (M, N) becomes physical (N, M). This enables:
- Small logical M (8, 16, 32) by making physical M = logical N (64 or 128)
- Large logical M (256) when physical layout fits in smem

**Complete Logical Tile Matrix:**

| Logical M | Supported Logical N | swap_ab | Physical (M, N) |
|-----------|---------------------|---------|-----------------|
| 8 | 64, 128 | True | (64, 8), (128, 8) |
| 16 | 64, 128 | True | (64, 16), (128, 16) |
| 32 | 64, 128 | True | (64, 32), (128, 32) |
| 64 | 8, 16, 32, 64, 128, 256 | False | (64, N) |
| 128 | 8, 16, 32, 64, 128 | False | (128, N) |
| 256 | 64 | True | (64, 256) |

**Summary:**
- **Verified (no swap):** 11 tiles - M ∈ {64, 128}, N ∈ {8, 16, 32, 64, 128, 256} excluding (128, 256)
- **Theoretical (with swap):** 7 tiles - M ∈ {8, 16, 32}, N ∈ {64, 128} plus (256, 64)
- **Total potential:** 18 unique logical tile configurations

**Known to Fail (smem overflow):**
- (128, 256): only 1 stage fits
- (256, 128), (256, 256): physical too large
- (512, *): any M=512 configuration

### 17.7 Benefits

1. **Simpler code**: No transposed/swap logic complexity (for no-swap tiles)
2. **Fewer kernel variants**: Single code path for all M sizes
3. **Native efficiency**: Hardware directly supports M=64 without layout transformations
4. **Expanded tile selection**: More options for autotuning
5. **Small-batch decode**: swap_ab enables M < 64 for single-token inference

---

## 18. FP4 Shared Memory: Why uint8 Layout is Correct (Not a Bug)

> **⚠️ INVESTIGATION RESOLVED (2026-01-15):** The uint8 shared memory layout for FP4 weights
> is **not** a bug - it's required by hardware. See section 19 for full explanation.

### 18.1 Initial Hypothesis (Incorrect)

We initially believed that FP4 shared memory was "over-allocated" by 2×:
- Expected: 8,192 bytes for 16,384 FP4 elements (0.5 bytes each)
- Actual: 16,384 bytes (uint8 layout, 1 byte each)

This appeared to be a bug that could be fixed to gain +1 pipeline stage.

### 18.2 Investigation: Attempted Optimizations

Several approaches were tried to reduce SMEM usage:

1. **Separate SmemLayoutTypeB (FP4) vs SmemCopyTypeB (uint8)**
   - Result: Compilation succeeded, but "illegal instruction" at runtime
   - Cause: Layout strides in 4-bit units, but ldmatrix expects byte addresses

2. **Use FP4 as Copy_Atom ValType**
   - Result: Compilation failed
   - Cause: `TiledCopy uses too few vals for selected CopyAtom`
   - The MMA's ValTypeB is uint8, creating layout incompatibility

3. **recast<uint8_t>(sB) before partition_S**
   - Result: Compilation failed
   - Cause: Halves element count, breaks `size<2>(tCsA) == size<2>(tCsB)` assertion

### 18.3 Root Cause: Hardware Padding Requirement

The `.b4x16_p64` suffix in `ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` means:
- **16 × 4-bit values + 64 bits padding = 128 bits = 16 bytes per chunk**
- So 16 FP4 values (8 bytes payload) require **16 bytes of SMEM**

This is a **hardware contract**. NVIDIA's Blackwell documentation states:
> "For `.kind::f8f6f4`, SMEM is stored in a 16B-aligned padded format...
> allocate SMEM as if sub-byte operands were byte operands.
> Fully compressed contiguous data in SMEM is not supported."

### 18.4 Correct Implementation

The uint8 layout is correct and matches SM100:

```cpp
// sm120_blockscaled_mma_builder.inl
using SmemAllocTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
using SmemLayoutTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
using SmemCopyTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
```

### 18.5 Verified Tile Configurations

All these tiles work correctly with the uint8 layout:

**Native (M >= 64):**
- M=64: (64, 8), (64, 16), (64, 32), (64, 64), (64, 128)
- M=128: (128, 8), (128, 16), (128, 32), (128, 64), (128, 128)
- M=256: (256, 8), (256, 16), (256, 32), (256, 64)

**Swapped (logical M < 64):**
- Logical M=8: (8, 64), (8, 128)
- Logical M=16: (16, 64), (16, 128)
- Logical M=32: (32, 64), (32, 128)

### 18.6 Lesson Learned

The "2× overhead" in SMEM is **mandated by hardware**, not a fixable inefficiency.
When working with sub-byte types on Blackwell, always check:
1. Which ldmatrix variant is being used
2. Whether it has padding requirements (e.g., `_p64`)
3. The PTX documentation for format requirements

---

## 19. FP4 Shared Memory: Hardware Padding Requirement (`.b4x16_p64`)

### 19.1 The Constraint

When using the `ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` instruction for FP4→FP8 expansion, **the shared memory must use padded format**:

- **`.b4x16_p64`** = 16 × 4-bit values + 64 bits padding = 128 bits = 16 bytes
- So 16 FP4 values (8 bytes of payload) require **16 bytes of SMEM**

This is a **hardware contract**, not a software inefficiency. NVIDIA's Blackwell sub-byte documentation states:
> "For `.kind::f8f6f4`, SMEM is stored in a 16B-aligned padded format... allocate SMEM as if sub-byte operands were byte operands. Fully compressed contiguous data in SMEM is not supported."

### 19.2 Why uint8 Layout is Correct

Both `SmemLayoutTypeB` and `SmemAllocTypeB` must use `uint8_t` because:

1. The `_p64` suffix means 64 bits of padding per 16×4-bit chunk
2. The ldmatrix instruction expects byte-aligned addresses (not 4-bit)
3. CuTe's swizzle/layout operates in byte units, not sub-byte

The "extra" bytes are **required padding**, not waste.

### 19.3 Attempted Optimizations (Why They Failed)

Several approaches were tried to reduce SMEM usage:

1. **FP4 Layout + uint8 Copy Atom**: Causes address mismatch (4-bit vs 8-bit strides)
2. **FP4 as Copy Atom ValType**: Incompatible with MMA's ValTypeB (uint8)
3. **recast<uint8_t>(sB)**: Halves element count, breaks CUTLASS assertions

All fail because they violate the hardware's padded format requirement.

### 19.4 Correct Implementation

The SM100-compatible approach (uint8 for everything):

```cpp
// sm120_blockscaled_mma_builder.inl
using SmemAllocTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
using SmemLayoutTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
using SmemCopyTypeB = cute::conditional_t<UseMxf8f6f4, uint8_t, ...>;
```

This allocates 2× the FP4 payload size in SMEM, but:
- TMA writes only the actual FP4 bytes
- ldmatrix reads from correctly-aligned addresses
- The "extra" space is the mandated padding

---

## 20. Two FP4 Instruction Paths on SM121f (2026-01-15)

### 20.1 Discovery

Investigation revealed that SM121f (GB10) has **two distinct instruction paths** for FP4 operations, with different shared memory requirements:

| Path | PTX Instruction | Operand Types | K dim | SMEM Format |
|------|-----------------|---------------|-------|-------------|
| **`kind::f8f6f4`** | `mma.sync.aligned.kind::f8f6f4.m16n8k32` | Mixed (e.g., FP8×FP4) | 32 | **Padded** (`_p64`) |
| **`kind::mxf4`** | `mma.sync.aligned.kind::mxf4.block_scale.m16n8k64` | FP4×FP4 only | 64 | **Packed** (no padding) |

### 20.2 The Packed FP4 Path (`kind::mxf4`)

The `kind::mxf4` instruction path:
- Uses standard `SM75_U32x4_LDSM_N` ldmatrix (no `_p64` suffix)
- Both operands must be FP4 (e2m1)
- K dimension is 64 (vs 32 for `kind::f8f6f4`)
- **No padding required** - true 0.5 bytes/element SMEM usage

This was verified to compile and run on SM121f:
```cpp
// Works: kind::mxf4 with packed FP4 from registers
asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
    "%14, {%15, %16}, %17, {%18, %19};\n"
    ...
);
```

### 20.3 Why We Use `kind::f8f6f4` (Padded Path)

Our use case is **FP8 activations × FP4 weights** (MXFP4):
- A operand: FP8 (e4m3) from `mxfp8_quantize(hidden_states)`
- B operand: FP4 (e2m1) from pre-quantized model weights

The `kind::mxf4` path requires **both operands to be FP4**:
```cpp
// Fails: kind::mxf4 rejects mixed types
"mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale...f32.e4m3.e2m1.f32..."
// Error: Incorrect instruction type specified for mma with shape '.m16n8k64'
```

Therefore, we must use `kind::f8f6f4` which mandates the padded SMEM format.

### 20.4 Trade-off: Packed FP4 Path

If we were willing to change to **FP4×FP4**:
- Quantize FP8 activations → FP4 (additional quantization step)
- Use `kind::mxf4` with packed SMEM
- **Benefit**: 50% SMEM reduction for both A and B operands
- **Cost**: Potential accuracy degradation from FP8→FP4 quantization

This is a **future optimization opportunity** if:
1. Accuracy testing shows FP4 activations are acceptable
2. The SMEM savings enable larger tiles or more pipeline stages
3. The K=64 instruction provides better throughput

### 20.5 CUTLASS Support

CUTLASS has MMA traits for `kind::mxf4` on SM120:
```cpp
// cute/atom/mma_traits_sm120.hpp
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<a_type, b_type, c_type, sf_type, VS>>
{
  using ValTypeA = uint4_t;  // Native 4-bit
  using ValTypeB = uint4_t;  // Native 4-bit
  using Shape_MNK = Shape<_16,_8,_64>;
  ...
};
```

When `UseMxf8f6f4 = false`, the builder uses:
- `SmemCopyTypeB = TiledMma::ValTypeB` (uint4_t, not uint8_t)
- Standard `SM75_U32x4_LDSM_N` ldmatrix (no `_p64` padding)

### 20.6 Verification Scripts

The following were used to verify instruction availability:

```bash
# Test kind::mxf4 compilation and runtime
docker exec vllm-dev nvcc -gencode arch=compute_121f,code=sm_121f test_mxf4.cu

# Verify macros enabled
# CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED: YES
# CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED: YES
```

### 20.7 Summary

| Question | Answer |
|----------|--------|
| Is there a packed FP4 path on SM121f? | **Yes** - `kind::mxf4` |
| Can we use it for FP8×FP4? | **No** - requires FP4×FP4 |
| Is the `uint8` padding a bug? | **No** - mandated by `kind::f8f6f4` |
| Could we switch to FP4×FP4? | **Maybe** - accuracy trade-off |

---

## 21. Empty Expert Handling (M=0) - Bug Fix (2026-01-16)

### 21.1 Problem

When some experts receive 0 tokens (common in MoE routing), the grouped GEMM crashed with "illegal instruction" or "illegal memory access" errors.

**Root Cause:** The code was artificially setting `gemm_m_for_problem = 128` for empty experts (M=0) to satisfy a perceived block-scale alignment requirement:

```cpp
// INCORRECT - causes stride/shape mismatch
constexpr int64_t kBlockScaleAlignment = 128;
auto const gemm_m_for_problem = (has_block_scale_fc1 || has_block_scale_fc2) && gemm_m == 0
    ? kBlockScaleAlignment : gemm_m;  // This sets M=128 for empty experts
```

However, the strides and pointers were still computed using the real `gemm_m = 0`. This mismatch caused the kernel to try to access 128 rows of data while the memory layout was configured for 0 rows.

### 21.2 Solution

Use the actual `gemm_m` value directly, including M=0 for empty experts. CUTLASS grouped GEMM properly handles groups with M=0 by skipping them during execution.

```cpp
// CORRECT - CUTLASS skips groups with M=0
auto const gemm_m_for_problem = gemm_m;
```

### 21.3 Changed File

`csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` around line 1290

### 21.4 Testing

Verified with:
- 1 expert (M=2): Pass
- 2 experts (each with 1 token): Pass
- 8 experts (only 1 active, others M=0): **Now passes** (previously crashed)
- 128 experts (gpt-oss-120b dimensions): Pass

---

## 22. Known Issue: Duplicate Expert Selection Crash (2026-01-16)

### 22.1 Problem

The SM120 MXFP4 CUTLASS kernel crashes with "illegal memory access" when a token selects the **same expert multiple times** within its top-k selection.

**Root Cause:** The `finalizeMoeRoutingKernel` reads from `unpermuted_row_to_permuted_row` which contains garbage values when duplicate experts are selected. The fused prologue kernel's radix sort ranking algorithm doesn't correctly handle duplicate expert IDs for the same token.

**Symptoms:**
- Kernel reports "initialize succeeded, calling run()..." (GEMM succeeds)
- `finalizeMoeRoutingKernel` crashes with "invalid __global__ read"
- Access is ~96GB out of bounds (garbage pointer value in permutation map)

**Reproducer:**
```python
# FAILS - token 3 selects expert 74 twice
topk_ids = torch.zeros((256, 4), dtype=torch.int32, device='cuda')
topk_ids[3] = torch.tensor([74, 74, 87, 116])  # Duplicate expert 74

# PASSES - all distinct experts per token  
topk_ids[i] = torch.tensor([i*4, i*4+1, i*4+2, i*4+3])
```

### 22.2 Analysis

With seed 42 random routing for 256 tokens with top-4:
- **14 out of 256 tokens** have duplicate expert selections
- Example duplicates:
  - Token 3: [74, 74, 87, 116] - expert 74 selected twice
  - Token 12: [61, 61, 46, 61] - expert 61 selected THREE times

### 22.3 Affected Scenarios

| Scenario | Result |
|----------|--------|
| All tokens select distinct experts per top-k | **PASS** |
| Uniform/deterministic routing (no duplicates) | **PASS** |
| Random routing (may have duplicates) | **FAIL** |
| vLLM inference (learned gating may have duplicates) | **FAIL** |

### 22.4 Fix Options

**Option A: Input Validation (Quick)**
Add check in Python wrapper to reject/deduplicate expert selections before calling kernel.

**Option B: Kernel Fix (Proper)**
Modify `fusedBuildExpertMapsSortFirstTokenKernel` to handle duplicate expert IDs by assigning distinct permuted indices even for same-expert selections.

**Option C: Use Unfused Prologue**
Force fallback to `threeStepBuildExpertMapsSortFirstToken` which may handle duplicates correctly.

### 22.5 Workaround

Use MARLIN backend which dequantizes weights to BF16 and uses standard GEMM:

```bash
export VLLM_MXFP4_BACKEND=MARLIN
vllm serve openai/gpt-oss-120b --quantization mxfp4 ...
```

The MARLIN backend works correctly but may have lower throughput than native FP4 compute.

### 22.6 Additional Issue: CUDA Graph Capture Crash

**Update (2026-01-16):** vLLM's actual gating uses `torch.topk` which ALWAYS returns distinct indices. The duplicate expert issue only affects test code using `torch.randint`.

A separate issue exists during CUDA graph capture:
- CUTLASS kernel runs successfully during initial warmup (many "initialize succeeded" messages)
- Crashes with "illegal instruction" during CUDA graph capture phase
- `--enforce-eager` bypasses CUDA graphs but issue persists during actual inference

This suggests a deeper issue with kernel execution under certain conditions, possibly related to batch size, memory layout, or SM121 instruction support.

### 22.7 Status

- **Priority:** HIGH (blocks native CUTLASS path for real inference)
- **Root Causes:**
  1. Duplicate expert selection bug (test-only, doesn't affect vLLM gating)
  2. CUDA graph capture crash (under investigation)
- **Fallback:** MARLIN backend available (~33 tok/s decode)

---

## 23. References

### 23.1 Internal
- **FlashInfer GEMM layout:** `csrc/nv_internal/.../moe_gemm_kernels.h` (line 53-55)
- **FlashInfer MoE JIT:** `flashinfer/jit/fused_moe.py` (`gen_cutlass_fused_moe_module`)
- **TRT-LLM tile configs:** `trtllmGen_bmm_export/config.json`
- **llama.cpp tile selection:** `ggml-cuda/mmq.cuh:3980-4048`
- **CUTLASS SM120 examples:** `cutlass/examples/92_blackwell_moe_gemm/`
- **FlashInfer JIT system:** `flashinfer/jit/`
- **FlashInfer autotuner:** `flashinfer/autotuner.py`
- **Current launcher:** `moe_gemm_sm120_mixed_input_launcher.inl`

### 23.2 External PRs (M/N Swap Solution)
- **FlashInfer PR 2327:** [M/N swap for small-M decode](https://github.com/flashinfer-ai/flashinfer/pull/2327) - Core solution for tcgen05 M≥64 constraint
- **CUTLASS PR 2946:** [N not divisible by 64 handling](https://github.com/NVIDIA/cutlass/pull/2946) - Required after M/N swap for proper epilogue
