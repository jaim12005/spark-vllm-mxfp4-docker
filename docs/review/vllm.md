# Review: Blackwell-class enablement, sinks gating, FlashInfer multi-group attention, and unified MXFP4 backend/layer selection

This review is meant to bring a mentor up to speed on what changed, why it changed, and what to watch out for. It’s organized by theme (attention, quant/MoE, platform/env, and warmup/debug), and calls out the main behavioral deltas plus risk areas.

---

## 1) Attention: sinks gating + Blackwell-class policy + FlashInfer “multi-hyperparam-group” support

### 1.1 Sinks gating via `VLLM_ATTENTION_SINKS`
**Files:**
- `vllm/attention/backends/abstract.py`
- `vllm/attention/layer.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/utils.py` (`get_per_layer_parameters`)

**Change:**
- Introduces env var `VLLM_ATTENTION_SINKS` with values `auto|false` (as used here).
- If model has sinks (`extra_impl_args["sinks"] is not None`) and env is `"false"`:
  - `Attention.has_sink` is forced to `False` (layer-level).
  - Per-layer parameters mark `has_sinks=False` (metadata-level).
  - FlashAttention backend explicitly sets `sinks=None` and warns.

**Intent:**
- Allow bringing up new hardware/backends or CI environments where sinks are present in the model config but not supported by the chosen kernel path. This avoids “hard fail” selection logic and enables A/B testing.

**Mentor note / risk:**
- The validation logic looks inverted in `abstract.py` and `flash_attn.py`:
  - Current logic adds “sink not supported” **only when `VLLM_ATTENTION_SINKS == "false"`**, which is the opposite of what you’d expect (you’d normally *skip* sink validation if sinks are disabled).
  - If this is intentional, it should be documented as “treat sinks as required unless explicitly disabled”, but the rest of the diff treats `"false"` as “disable sinks and proceed.”
- Recommendation: make the rule consistent everywhere:
  - If `VLLM_ATTENTION_SINKS == "false"`, treat sinks as disabled and *do not* reject backends for lack of sink support.

---

### 1.2 “Blackwell-class” detection unified: SM10x/SM11x/SM12x
**Files:**
- `vllm/attention/utils/fa_utils.py`
- `vllm/platforms/cuda.py`
- `vllm/platforms/interface.py`
- Many call sites swapped from `is_device_capability_family(100)` to `is_blackwell_class()`.

**Change:**
- Defines “Blackwell-class” as major in `{10, 11, 12}`.
- Uses it in:
  - FlashAttention version selection: FA3 is disallowed on Blackwell-class (fallback to FA2).
  - MLA backend routing defaults (CutlassMLA vs FlashMLA).
  - Various feature gates previously “SM100-family only” now expanded to include SM12x (GB10 / DGX Spark / Thor).

**Intent:**
- Make SM12x first-class without duplicating `major==12` checks everywhere.
- Keep SM11x reserved for future variants.

**Mentor note / risk:**
- There’s a conceptual mismatch in some places between:
  - “Blackwell-class” (major 10/11/12)
  - “TRT-LLM cubin availability” (really SM100/SM103 only)
- Example: `supports_trtllm_attention()` now returns:
  - `False` for major==12 (good), but still uses `is_blackwell_class()` later in the function which would include SM12x if not explicitly excluded.
- Recommendation: be careful with naming:
  - `is_blackwell_class()` is a *broad* architecture grouping.
  - TRT-LLM attention is a *binary artifact availability* constraint (sm100 only).

---

### 1.3 FlashInfer backend: multi-group wrappers for speculative decoding + FA2 sink module on Blackwell
**Files:**
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/backends/utils.py` (new grouping utility + updated semantics)

**What changed (core):**
FlashInfer previously assumed *all layers share the same*:
- `window_left`
- `logits_soft_cap`
- `sm_scale`
- `has_sinks`

This diff replaces that assumption with **hyperparameter grouping**:

#### a) New `PerLayerParameters` semantics
- `infer_global_hyperparameters()` no longer asserts uniformity.
- It now returns the first layer’s parameters and sets:
  - `has_same_window_lefts`
  - `has_same_all_params`
  as informational flags only.

#### b) New grouping helper
- `group_layers_by_hyperparameters(per_layer_params)` groups layers by:
  `(window_left, logits_soft_cap, sm_scale, has_sinks)`
- Sorts groups so group 0 is *likely* the main model:
  1. `has_sinks=True` first
  2. more layers first
  3. tie-break by first layer name

#### c) Builder now creates wrappers per group
- `FlashInferMetadataBuilder.__init__`:
  - computes per-layer params
  - groups them
  - builds `layer_to_group_id` and `group_id_to_params`

- Prefill:
  - plans **one wrapper per group**
  - stores as `FIPrefill(wrappers_by_group=...)` (or DCP equivalent)

- Decode:
  - plans primary wrapper as usual
  - for multi-group and **non-cudagraph decode**, also plans per-group wrappers
  - stores as `FIDecode(wrappers_by_group=...)`

- Forward path:
  - looks up `layer.layer_name` and maps to `group_id`
  - uses `attn_metadata.prefill.get_wrapper(group_id)` / decode analog

#### d) Sinks support logic updated
- `FlashInferImpl.supports_sink()` now returns True if:
  - TRTLLM attention is supported (SM100 artifacts), respecting forced disable, OR
  - on Blackwell-class (SM12x) via “FA2 sink module”

- Planning passes `use_sinks=...` into wrappers, and `run()` passes `sinks=self.sinks` only when wrapper indicates sinks are enabled (`_use_sinks`).

**Intent:**
- Enable speculative decoding where main + draft models can have different attention configs.
- On SM12x, allow sinks without TRT-LLM cubins (by using a FlashAttention2 sink module integrated into FlashInfer).

**Mentor note / risk:**
- **Layer name dependency:** forward asserts `layer.layer_name` exists. That requires upstream `Attention` to set `self.layer_name = prefix` consistently for all attention layers. If any path doesn’t set it, multi-group will hard fail.
- **Group ordering heuristic:** sorting by `has_sinks` and layer count is plausible, but it’s still heuristic. If a draft model also has sinks, or layer counts match unexpectedly, group 0 could become unstable.
- **Workspace sharing:** builder explicitly shares a single workspace buffer across all wrappers. That’s OK today because vLLM executes sequentially, but it’s a constraint if concurrent execution ever appears.

---

## 2) Compilation: Torch inductor config probe
**File:** `vllm/compilation/decorators.py`

**Change:**
- Wraps setting `torch._inductor.config.assume_32bit_indexing` in a try/except because some NGC builds may not expose the field even if version check passes.

**Intent:**
- Prevent runtime failure in specific PyTorch builds.

---

## 3) MXFP4: unified backend selection, CLI/env plumbing, and expanding quantization targets (qkv/o/lm_head)

### 3.1 New user-facing config: `mxfp4_backend` and `mxfp4_layers`
**Files:**
- `vllm/config/model.py`
- `vllm/engine/arg_utils.py`
- `vllm/envs.py`

**Change:**
- Adds:
  - `ModelConfig.mxfp4_backend: str = "auto"`
  - `ModelConfig.mxfp4_layers: str = "moe"`
- Adds CLI args:
  - `--mxfp4-backend`
  - `--mxfp4-layers`
- Adds env:
  - `VLLM_MXFP4_BACKEND=auto|marlin|cutlass|triton|trtllm|trtllm_mxfp8`
- Removes legacy env toggles:
  - `VLLM_USE_FLASHINFER_MOE_MXFP4_*`, `VLLM_MXFP4_USE_MARLIN`, etc.
  (but code still reads them directly via `os.environ` for deprecation warnings)

**Intent:**
- Replace a pile of backend-specific booleans with one coherent selector.
- Allow incremental expansion of quantized layer coverage beyond MoE.

---

### 3.2 Backend selection refactor (and new SM12x CUTLASS predicate)
**Files:**
- `vllm/model_executor/layers/quantization/mxfp4.py`
- `vllm/utils/flashinfer.py`

**Change:**
- Renames/reorganizes `Mxfp4Backend` enum into clearer “ENGINE_ARCH_QUANT” names:
  - Universal: `MARLIN`, `TRITON`
  - CUTLASS: `CUTLASS_BLACKWELL_FP4FP8`, `CUTLASS_SM90_FP4BF16`
  - TRTLLM: `TRTLLM_SM100_FP4FP8`, `TRTLLM_SM100_FP4BF16`

- Adds:
  - `has_flashinfer_sm12x_cutlass_moe()` which requires:
    - CUTLASS fused MoE support
    - current GPU major==12
    - presence of `flashinfer.mxfp8_quantize` and `flashinfer.mxfp4_quantize`

- Backend selection priority in `get_mxfp4_backend()`:
  1. CLI `--mxfp4-backend` (via model_config)
  2. env `VLLM_MXFP4_BACKEND`
  3. legacy env vars (warn + map)
  4. if LoRA enabled: force LoRA-compatible path
  5. hardware-based auto-select

- Auto-select highlights:
  - SM12x: prefer CUTLASS FP8×FP4 (if FlashInfer sm12x support available)
  - SM100: default TRTLLM BF16×FP4
  - Fallback: Triton if supported, else Marlin

**Intent:**
- Make SM12x “native” path explicit and robust.
- Keep SM100 upstream behavior unless user overrides.

**Mentor note / risk:**
- The SM12x predicate checks only for function presence, not that kernels were built with the right arch (`12.1f` vs `12.1a`). If there’s a common failure mode here, consider checking FlashInfer version/build tags.
- The auto-select for SM100 keys off `capability.major == 10`, which includes SM100/SM101 but also any future major=10 variants. That’s probably fine, but be aware of SM103 naming in comments.

---

### 3.3 Expanding MXFP4 to `qkv_proj`, `o_proj`, and `lm_head`
**File:** `vllm/model_executor/layers/quantization/mxfp4.py`

#### a) QKV/O: `Mxfp4LinearMethod` (Marlin-based)
- Adds a new `LinearMethodBase` implementation that:
  - creates BF16 weights
  - after load, quantizes to MXFP4 via `mxfp4_e2m1_quantize()`
  - repacks via `prepare_fp4_layer_for_marlin`
  - applies via `apply_fp4_marlin_linear` (fused dequant+GEMM)

- Gated by `--mxfp4-layers` tokens:
  - quantizes only prefixes ending in `.qkv_proj` / `.o_proj`
  - skips if LoRA enabled (conservative)

#### b) lm_head: `Mxfp4LMHeadMethod` for `VocabParallelEmbedding`
- Adds a specialized quant method because GPT-OSS uses `ParallelLMHead(VocabParallelEmbedding)` not `LinearBase`.
- Gating:
  - only for `prefix == "lm_head"` or suffix `.lm_head` (strict match)
  - only when `lm_head` token is enabled in `--mxfp4-layers`
  - only on CUDA + Blackwell-class
  - *disables* if:
    - LoRA enabled
    - `tie_word_embeddings=True` (would quantize embeddings table too)
    - weight storage pointer changes after create (detects tying/aliasing)

- If quantization is skipped, falls back to BF16 `F.linear`.

#### c) Model wiring: pass quant_config into `ParallelLMHead`
**File:** `vllm/model_executor/models/gpt_oss.py`
- `ParallelLMHead(..., quant_config=vllm_config.quant_config, ...)`
This is what allows lm_head to actually use the new quant method.

**Intent:**
- Make it possible to quantize more of the dense path (QKV/O/lm_head), not just MoE experts.
- Use Marlin as the pragmatic initial approach (weight-only FP4 compression + BF16 compute).

**Mentor note / risk:**
- These new methods are Marlin-only; they do not use native FP8×FP4 MMA. That’s fine as an incremental step, but expectations should be set: this improves bandwidth, not tensor core utilization.
- The lm_head `embedding()` fallback explicitly does *not* implement VocabParallelEmbedding sharding semantics—relies on the fact that the tied-embedding path should never call lm_head embedding. That’s subtle but probably OK.

---

### 3.4 MoE: CUTLASS init robustness + debug knobs
**File:** `vllm/model_executor/layers/quantization/mxfp4.py` (MoE apply path)

**Changes:**
- Ensures `tune_max_num_tokens >= actual num tokens` for both TRTLLM and CUTLASS paths. This avoids internal init failures when runtime shapes exceed the “tuning bucket” size.
- Adds warmup-only routing override:
  - if forward context indicates dummy run AND `VLLM_MOE_WARMUP_FORCE_UNIFORM=1`,
  - force a deterministic “uniform-ish” routing assignment to avoid empty experts (M=0 groups) that can break some grouped GEMM init paths.
- Adds optional routing distribution logging (`VLLM_MOE_ROUTING_LOG=1`).
- Adds optional mxfp8 quant logging (`VLLM_MXFP8_QUANT_LOG=1`) with stats + pointer alignment checks.

**Intent:**
- Make CUTLASS grouped-GEMM init more reliable during warmup/profile runs.
- Provide high-signal debug telemetry for alignment/NaN/Inf/shape mismatches.

**Mentor note / risk:**
- The warmup routing override writes directly to stderr via `os.write` and does extra GPU work. It’s gated, but ensure it cannot be triggered in real runs (only dummy run + env).
- Logging reductions (`amin/amax/isnan/isinf`) will synchronize and can distort profiling if enabled accidentally.

---

## 4) Utility changes: Marlin warning adjustment + Marlin-compatible MXFP4 quant
### 4.1 Marlin warning only for pre-Blackwell/Hopper
**File:** `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
- Previously warned unconditionally about “no native FP4 support.”
- Now warns only if device capability major < 10.

**Intent:**
- SM100/SM12x have native FP4, so Marlin is “fallback choice,” not “hardware limitation.”

### 4.2 `mxfp4_e2m1_quantize` (Marlin-compatible scale layout)
**File:** `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`

**Change:**
- Adds `mxfp4_e2m1_quantize(x)` which uses `flashinfer.fp4_quantize` but forces:
  - `is_sf_swizzled_layout=False`
  - because Marlin expects linear scale layout.

**Intent:**
- Avoid mixing FlashInfer’s swizzled SF layout with Marlin’s expected packing.

---

## 5) Forward context: tagging dummy/profile runs
**Files:**
- `vllm/forward_context.py`
- `vllm/v1/worker/gpu_model_runner.py`

**Change:**
- `set_forward_context(..., extra_forward_kwargs=...)` merges into `additional_kwargs`.
- Dummy run now tags:
  - `vllm_dummy_run=True`
  - `vllm_profile_run=<bool>`

**Intent:**
- Allow kernel warmup / tuning code to detect dummy runs and apply safe overrides (like uniform routing) without impacting real inference.

---

## 6) Biggest “mentor should watch this” items

### A) Sinks gating is inconsistent in validation
As written, some checks reject sinks *only when the user set sinks=false*. That likely needs flipping so `"false"` means “ignore sinks” rather than “enforce sink support.”

### B) Multi-group attention depends on `layer.layer_name`
The forward path asserts `layer.layer_name` exists and matches keys used in grouping. This is fragile unless enforced globally across Attention construction. Make sure the Attention layer sets it reliably.

### C) Group ordering heuristic may be brittle
Picking “group 0” by `has_sinks` and layer count is reasonable for main vs draft, but it’s still a heuristic. If there’s any chance of multiple groups with sinks, consider making grouping explicit via model IDs or a stable tag.

### D) MXFP4 lm_head method is carefully gated but subtle
- Correctly avoids tied embeddings and LoRA.
- Uses data_ptr replacement detection, which is safe but may skip quantization in some non-tying cases (acceptable tradeoff).
- The embedding fallback is intentionally incomplete; document this clearly to avoid surprises.

### E) SM12x “CUTLASS available” predicate is necessary but not sufficient
Checking for function symbols is good, but SM12x instruction set quirks (`12.1a` vs `12.1f`, FP4 ldmatrix constraints, etc.) often require checking build flags and kernel availability. Consider adding a version/build capability probe if failures are common.

---

## “What changed for users” (practical summary)

- New CLI:
  - `--mxfp4-backend auto|marlin|cutlass|triton|trtllm|trtllm-mxfp8`
  - `--mxfp4-layers moe|qkv|o|lm_head|all` (comma-separated)
- New env:
  - `VLLM_MXFP4_BACKEND=...`
  - `VLLM_ATTENTION_SINKS=auto|false`
- Speculative decoding: FlashInfer attention can now handle main/draft models with different attention hyperparams by using per-group wrappers.

---
