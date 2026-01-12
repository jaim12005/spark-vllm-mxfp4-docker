# SGLang Analysis (SM121 / gpt-oss-120b on DGX Spark / GB10)

## What we know (reference data)

SGLang achieves **~52 tok/s decode** on SM121 with `gpt-oss-120b` using MXFP4 weights (reported as “current highest TPS” on DGX Spark).

Reference benchmark (as collected):

```
Engine: SGLang
Model: gpt-oss-120b (MXFP4)
GPU: NVIDIA GB10 (SM121)

Output token throughput: 52.37 tok/s
TTFT: 49.87 ms
TPOT: 18.83 ms (time per output token)
```

## Why this is plausible (where decode time goes on SM121)

Our own kernel profiling on SM121 shows decode is dominated by **MoE GEMM + dense GEMV/GEMM**, not attention:

- `docs/analysis/VLLM_BASELINE_ANALYSIS.md`: attention ~1.5% GPU time; dense GEMV/GEMM ~38%; MoE ~34%.
- `docs/BENCHMARK_RESULTS.md`: even after quantizing `lm_head` to MXFP4 we only reach **~34 tok/s**, leaving a big gap to 52.

**Implication**: SGLang’s 52 tok/s likely comes from improvements in:
- Dense decode path (QKV/O projections, residual/norm/activation fusion, and/or quantized dense matmuls)
- MoE decode path (kernel choice + setup overhead + activation format)
- Runtime/scheduler overhead (CUDA graphs, async scheduling, fewer launches, tighter C++ control plane)

## Technical facts from code (SGlang “Spark” implementation)

This section replaces the earlier hypotheses with concrete code-level behavior from the local Spark-focused SGLang checkout at `~/projects/sglang-spark` (branch `spark`).

### Fact 1: On SM121, GPT-OSS uses Triton attention by default (not FA2/FA3)

In `python/sglang/srt/server_args.py` the default attention backend for `GptOssForCausalLM` is:
- SM100 → `trtllm_mha`
- SM90 → `fa3`
- Otherwise (including SM121) → **`triton`**

See: `python/sglang/srt/server_args.py:L927-L949`.

Why this matters on DGX Spark:
- gpt-oss-120b uses a non-standard head dim (45), where FA2-style kernels are typically constrained. Triton is the “always works” choice.

### Fact 2: GPT-OSS MoE defaults to `triton_kernel` backend when available (EP must be 1)

Still in `python/sglang/srt/server_args.py`, GPT-OSS sets the MoE runner backend to `triton_kernel` when:
- `moe_runner_backend=auto`
- `ep_size == 1`
- Triton kernels are available

See: `python/sglang/srt/server_args.py:L969-L981`.

Important detail:
- There is an explicit `if False:` block that would enable `flashinfer_mxfp4` MoE (TRTLLM FP4 block-scale) but it is **currently disabled** in this Spark branch.
  - See: `python/sglang/srt/server_args.py:L963-L968`.

### Fact 3: MXFP4 is used for *expert weights*, but dense projections use FP8

SGLang’s GPT‑OSS implementation uses:
- **MXFP4 for MoE expert weights**
  - `GptOssForCausalLM.load_weights()` splits weights into “experts” vs “normal”, and only the expert weights go through the MXFP4 path.
  - See: `python/sglang/srt/models/gpt_oss.py:L720-L749` (`.experts` filter and `_load_mxfp4_experts_weights()`).
- **FP8 for QKV projection, output projection, and lm_head**
  - QKV uses `quant_config=Fp8Config()` and O-proj uses `quant_config=Fp8Config()`.
  - See: `python/sglang/srt/models/gpt_oss.py:L239-L270`.
  - lm_head uses `quant_config=Fp8Config()`.
  - See: `python/sglang/srt/models/gpt_oss.py:L581-L587`.

This is a big architectural clue:
- SGLang is **not** “MXFP4 everywhere”; it is closer to “MXFP4 for routed experts + FP8 for dense always-on linears”.

### Fact 4: Attention sinks are wired explicitly in the model and always passed to attention

GPT‑OSS attention keeps per-head sinks and passes `sinks=self.sinks` on every call:
- Sinks dtype: `float32` for `trtllm_mha`, `bfloat16` otherwise.
- See: `python/sglang/srt/models/gpt_oss.py:L252-L258` (dtype selection) and `L325-L329` (passing sinks).

### Fact 5: CUDA graphs are enabled by default (and heavily parameterized)

SGLang has explicit CUDA-graph knobs and defaults to graphs enabled:
- `disable_cuda_graph: bool = False` (default is “graphs on” when supported)
- `cuda_graph_max_bs`, `cuda_graph_bs`, and “piecewise cuda graph” knobs are present.

See: `python/sglang/srt/server_args.py:L462-L491`.

### Fact 6: Benchmark harness exists and uses SGLang-specific throughput tooling

The `sglang` repo includes instructions for benchmarking `openai/gpt-oss-120b` with `bench_one_batch_server`:
- See: `benchmark/gpt_oss/README.md` (e.g. launch + benchmark commands).

Note: those instructions are written for H100/B200 examples, but they confirm the intended measurement harness and flags.

### Fact 7: `lm_head` is computed in LogitsProcessor (not `ParallelLMHead.forward()`), via matmul or quant method

In SGLang, `ParallelLMHead` is a sharded weight container and **its `forward()` is not used**:
- See: `python/sglang/srt/layers/vocab_parallel_embedding.py:L497-L573` (it raises `"LMHead's weights should be used in the sampler."`).

Logits for next-token sampling are computed in `LogitsProcessor._get_logits()`:
- If `lm_head.quant_method` exists, it calls `lm_head.quant_method.apply(lm_head, hidden_states)`.
- Otherwise it falls back to `torch.matmul(hidden_states, lm_head.weight.T)` (with dtype conversions).

See: `python/sglang/srt/layers/logits_processor.py:L794-L851`.

Practical implication:
- In decode (batch size ~1), this matmul is an **M=1 GEMM** (often discussed as “GEMV-like”), but the implementation path is GEMM/matmul-style.

### Fact 8: Experts use GEMM-style routed/grouped matmuls (not per-expert GEMV loops)

The Spark GPT‑OSS MoE path defaults to `moe_runner_backend=triton_kernel` (see Fact 2).
The Triton-kernels MoE implementation uses `matmul_ogs()` for both FC1 and FC2, operating on a routed/expanded batch of size `M * n_expts_act`:

- See: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:L124-L162`.

This is a **grouped/batched GEMM-style** formulation even when the original token batch `M` is small.

### Fact 9: Concrete dtype story (Spark GPT‑OSS)

SGLang’s Spark GPT‑OSS implementation is explicitly **mixed precision**:

- **Dense projections and lm_head use FP8 quantization config**:
  - QKV and O projections are constructed with `quant_config=Fp8Config()`:
    - See: `python/sglang/srt/models/gpt_oss.py:L239-L270`.
  - lm_head is constructed with `quant_config=Fp8Config()`:
    - See: `python/sglang/srt/models/gpt_oss.py:L581-L587`.
  - FP8 weight storage for serialized FP8 checkpoints uses `torch.float8_e4m3fn` on CUDA:
    - See: `python/sglang/srt/layers/quantization/fp8.py:L556-L620` (`params_dtype = torch.float8_e4m3fn` when `is_checkpoint_fp8_serialized`).

- **MXFP4 is applied to MoE expert weights** (experts-only weight loading split):
  - See: `python/sglang/srt/models/gpt_oss.py:L720-L749`.
  - MXFP4 expert parameters are stored as packed bytes:
    - weights: `torch.uint8`, scales: `torch.uint8`
    - See: `python/sglang/srt/layers/quantization/mxfp4.py:L286-L382`.

- **MoE Triton-kernels path expects BF16 activations**:
  - The new Triton MoE kernel asserts `hidden_states.dtype == torch.bfloat16`:
    - See: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:L107-L110`.

### Fact 10: Where the “Triton code” lives (and what is/ isn’t vendored here)

There are two layers:

1. **SGLang’s integration/wrapper code (in this repo)**:
   - `python/sglang/srt/layers/moe/fused_moe_triton/` (glue + configs)
   - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` (calls `matmul_ogs`, routing, activation fusion)

2. **The actual Triton kernel package** referenced by the MoE path:
   - The code imports `triton_kernels.matmul_ogs` / `triton_kernels.routing`:
     - See: `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:L9-L18`.
   - In this `sglang-spark` checkout, `triton_kernels` is **not present under `python/`** (the only top-level package there is `sglang`), so `triton_kernels` is provided by the Python environment, not vendored here.

Additionally:
- This repo vendors CUDA/C++ kernels via the `sgl-kernel` subproject (`/sgl-kernel`) and lists `sgl-kernel` as a Python dependency (`python/pyproject.toml`).

## What we can learn / what to consider for vLLM on DGX Spark (actionable)

Tie each item to *our measured bottlenecks*.

### Highest priority: shrink/accelerate the “dense GEMV/GEMM” bucket (~30–40% of decode)

SGLang’s most important “technical fact” is that GPT‑OSS uses **FP8** for QKV/O and lm_head (see `python/sglang/srt/models/gpt_oss.py:L239-L270` and `L581-L587`).

Investigations for vLLM:
- **Implement/enable FP8 linear methods for GPT‑OSS dense projections** (QKV/O first).
  - Success metric: reduce `gemvx::kernel<bf16>` share substantially in our nsys profile.
- **Consider FP8 for lm_head in vLLM** as an alternative to MXFP4+Marlin.
  - This is not guaranteed better than FP4 weight-only for M=1, but it’s a real, code-backed strategy SGLang uses.

Candidates:
- **Quantize additional dense linears** (QKV/O projections first; they fire every layer/token)
- Add **specialized M=1 kernels** for these projections (warp-specialized, persistent, or int8/DP4A-like where numerically acceptable)
- **Fuse** common patterns (RMSNorm+linear, bias+activation, residual adds) to reduce launches and memory traffic

### High priority: reduce MoE “overhead + conversions” (and recover CUTLASS decode regression)

SGLang’s Spark GPT‑OSS defaults to **Triton-kernels MoE** (`moe_runner_backend=triton_kernel`) rather than FlashInfer MXFP4 MoE (disabled in this branch).
See: `python/sglang/srt/server_args.py:L969-L981` and `L963-L968`.

Investigations for vLLM:
- **Compare our FlashInfer CUTLASS MoE path vs a Triton-kernels-style MoE path** at decode shapes (M≈1–2).
  - If FlashInfer remains the path, prioritize:
    - activation persistence (MXFP8),
    - fusing BF16→FP8 quantization into MoE,
    - runner/setup caching,
    - SM12x tile tuning.

### Medium priority: CUDA graphs stability and capture coverage

SGLang ships with CUDA-graph-on-by-default behavior and significant tuning knobs (see `python/sglang/srt/server_args.py:L462-L491`).
We often run `--enforce-eager` for stability; that leaves performance on the table.
Work items:
- Fix graph-capture incompatibilities for the gpt-oss-120b execution path
- Ensure sinks/speculation paths are graph-safe (or provide a stable fallback)

### Conditional: speculative decoding, but only if acceptance is good

Work items:
- Measure acceptance rate under our **best numerics** (FlashInfer + sinks + MXFP4 configs)
- Identify which module(s) cause mismatch vs the draft model
- If needed: align quantization/rounding choices to match what the draft model expects

## How to validate the SGLang “special branch” claims (what data we need)

To turn hypotheses into facts, we need:

1. **Reproducible benchmark metadata**
   - Same prompt/tg lengths, concurrency, warmup, and measurement definition as our `llama-benchy` runs
   - Confirm whether TTFT includes tokenization, queueing, and first-kernel latency

2. **`nsys` decode profile**
   - Kernel time breakdown (top kernels, categories)
   - Presence/absence of CUDA graphs
   - Evidence of speculation (batch verification patterns)

3. **Kernel/library attribution**
   - Which matmul kernels (CUTLASS vs Triton vs custom)
   - Whether dense projections are quantized (weight formats and load sizes)

## Open questions (still unanswered)

The implementation questions are now answered at code level for the Spark branch; the remaining questions are measurement/profiling questions:

1. What is the **kernel-level breakdown** (nsys) for SGLang on DGX Spark that yields 52 tok/s?
2. Which **FP8 kernel path** is used on SM121 for the FP8 linears (CUTLASS vs torch scaled-mm vs marlin-fp8), and how much time does it take?
3. Are CUDA graphs actually capturing decode in the benchmark run, and for which shapes?
4. Is speculative decoding enabled in the 52 tok/s measurement, and what acceptance rate is achieved (if enabled)?
5. Are TTFT/TPOT measured under the same conditions as our benchy runs (queueing, tokenization, warmup)?

## Concrete next steps (if we want to close this fast)

- Capture an `nsys` trace for SGLang decode (pp=2048, tg=32/128) and categorize top kernels.
- Run the *same* workload under our best vLLM config and diff kernel categories.
- Prioritize ports in this order for vLLM: **quantize QKV/O projections → activation persistence/fusion → runner caching → graphs → speculation acceptance**.
