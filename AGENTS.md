# AGENTS.md - Project Context for AI Assistants

## Mission

Make **vLLM the fastest inference engine for gpt-oss-120b** on NVIDIA GB10 (SM121), outperforming SGLang and llama.cpp by leveraging:

- **Native FP4 hardware features** (SM121 block-scaled MMA)
- **MXFP4 quantized weights** (4-bit weights, group size 32)
- **FlashAttention-2 with attention sinks** for long context stability

---

## Critical: Implementation Stack

### WE ARE USING

| Component | Implementation | Location |
|-----------|----------------|----------|
| **MoE GEMM** | CUTLASS SM120/121 grouped GEMM | `flashinfer/fused_moe/` |
| **Attention** | FlashAttention-2 (FA2) | `flashinfer/prefill.py`, `flashinfer/decode.py` |
| **Attention Sinks** | FA2 variant | `flashinfer/jit/attention/variants.py` |
| **FP4 Quantization** | FlashInfer `mxfp4_quantize()` | `flashinfer/fp4_quantization.py` |
| **Framework** | vLLM with FlashInfer backend | Local repos mounted in Docker |

### WE ARE NOT USING

| ❌ Component | Why Not |
|-------------|---------|
| **TensorRT-LLM (trtllm)** | We use CUTLASS directly, not TRT-LLM wrappers |
| **cuDNN** | Not for MoE or attention in this project |
| **trtllm_fp4_block_scale_moe** | This is SM100-only, crashes on SM121 |
| **FA3** | SM90 only; we're on SM121 which uses FA2 |
| **nvfp4** | We use MXFP4 (group size 32), not NVFP4 (group size 16) |

---

## Correct API Usage

### MoE GEMM on SM121

```python
# ✓ CORRECT: Use cutlass_fused_moe
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

output = cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_indices,
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_weights,
    fc2_expert_weights=fc2_weights,
    output_dtype=torch.bfloat16,
    quant_scales=[],  # Empty for BF16, populated for FP4/FP8
    activation_type=ActivationType.Swiglu,
)
```

```python
# ❌ WRONG: Do NOT use trtllm_fp4_block_scale_moe on SM121
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe  # SM100 ONLY!
# This will fail with: "No supported CUDA architectures found for major versions [10]"
```

### MXFP4 Quantization

```python
# ✓ CORRECT: Use mxfp4_quantize for MXFP4 (group size 32)
from flashinfer import mxfp4_quantize, mxfp4_dequantize

weight_fp4, weight_scale = mxfp4_quantize(weight_bf16)
# weight_fp4: [M, K/2] uint8 (packed nibbles)
# weight_scale: uint8 (UE8M0 format, group size 32)
```

```python
# ❌ WRONG: nvfp4_quantize is for NVFP4 (group size 16), different format
from flashinfer import nvfp4_quantize  # Different format, don't mix!
```

### Attention with Sinks (FA2)

```python
# ✓ CORRECT: FA2 attention with sinks
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

wrapper = BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer,
    backend="fa2",  # Explicitly FA2
)
wrapper.plan(..., use_sinks=True)  # Enable attention sinks
```

---

## Architecture: SM121 (GB10)

- **Compute Capability**: 12.1
- **Memory**: 120GB (not "limited" - this is desktop Blackwell with full memory)
- **Block-scaled MMA**: Supports FP4×FP4, FP8×FP4, but NOT BF16×FP4 directly
- **MXFP4 Path**: BF16 activations → quantize to FP8 → FP8×FP4 kernel

### MXFP4 on SM121 (How It Actually Works)

SM121 block-scaled MMA only accepts FP8/FP6/FP4 inputs. For MXFP4 (W4A16) with BF16 activations:

1. **FlashInfer's higher-level API** accepts BF16 activations
2. **Internally quantizes** BF16 → FP8 (e4m3)
3. **Dispatches to FP8×FP4 CUTLASS kernel**

This is why you don't see "MXFP4" tile configs at the kernel level—they go through the FP8×FP4 path.

---

## Repository Layout

**Host paths:**
- **vLLM**: `~/projects/vllm`
- **FlashInfer**: `~/projects/flashinfer`
- **This repo (mxfp4)**: `~/projects/ai/mxfp4`

```
~/projects/
├── flashinfer/          # Local FlashInfer development (CUTLASS kernels)
│   ├── flashinfer/      # Python package
│   │   ├── fused_moe/   # MoE implementation (cutlass_fused_moe)
│   │   ├── prefill.py   # Prefill attention (FA2 + sinks)
│   │   ├── decode.py    # Decode attention
│   │   └── fp4_quantization.py  # mxfp4_quantize, nvfp4_quantize
│   ├── include/         # CUDA headers (framework-agnostic)
│   └── csrc/            # CUDA sources (TVM-FFI bindings)
│
├── vllm/                # Local vLLM (uses FlashInfer as backend)
│
└── ai/mxfp4/            # THIS REPO - Docker config + test harness
    ├── docker-compose.dev.yml   # Dev container config
    ├── docker-compose.yml       # Production config
    └── scripts/
        ├── sm121_vllm_test_harness.sh      # Crash-proof testing
        ├── verify_mxfp4_moe_kernel.py      # MXFP4 MoE verification
        ├── verify_sm121_mxfp4_moe_fa2_sinks.py  # Integration tests
        └── profile_sm121_decode_performance.py  # Profiling
```

---

## Docker Environment (Dev Container)

### How It Works

The dev container (`vllm-dev`) mounts your local FlashInfer and vLLM repos as volumes. This allows:

1. **Live code editing** - Changes to kernel code on your host are immediately visible in the container
2. **JIT recompilation** - FlashInfer's JIT system detects file changes and recompiles kernels
3. **No reinstall needed** - Edit `.cuh` files on host → run in container → changes take effect

### Volume Mounts

```yaml
# docker-compose.dev.yml
volumes:
  # Source code - your local repos become /workspace/* in container
  - ~/projects/vllm:/workspace/vllm
  - ~/projects/flashinfer:/workspace/flashinfer
  
  # Persistent caches - survive container restarts
  - ~/.cache/huggingface:/root/.cache/huggingface    # Model weights
  - ./.cache/flashinfer:/root/.cache/flashinfer      # JIT compiled kernels
  - ./.cache/vllm:/root/.cache/vllm                  # vLLM caches
  - ./.cache/torch:/root/.cache/torch                # PyTorch caches
```

**Important**: The cache mounts are relative to the docker-compose directory (`~/projects/ai/mxfp4/.cache/`).

### PYTHONPATH: Why It's Required

FlashInfer uses Python's namespace package mechanism. When you have both:
- An installed wheel (`/usr/local/lib/python3.12/site-packages/flashinfer/`)
- A local repo mount (`/workspace/flashinfer/`)

Python may prefer the installed wheel! Setting `PYTHONPATH` forces the local mount to take precedence:

```yaml
environment:
  - PYTHONPATH=/workspace/flashinfer:/workspace/vllm${PYTHONPATH:+:$PYTHONPATH}
```

The `${PYTHONPATH:+:$PYTHONPATH}` suffix preserves any existing PYTHONPATH from the base image.

### Quick Verification Commands

```bash
# 1. Check you're using the local FlashInfer (CRITICAL)
docker exec vllm-dev python3 -c "import flashinfer; print(flashinfer.__file__)"
# ✓ Should print: /workspace/flashinfer/flashinfer/__init__.py
# ❌ NOT: /usr/local/lib/python3.x/site-packages/flashinfer/...

# 2. Check you're using the local vLLM
docker exec vllm-dev python3 -c "import vllm; print(vllm.__file__)"
# ✓ Should print: /workspace/vllm/vllm/__init__.py

# 3. Check both imports at once
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 -c "import flashinfer, vllm; print('flashinfer:', flashinfer.__file__); print('vllm:', vllm.__file__)"
```

### Two Services: `dev` vs `serve`

| Service | Purpose | Command |
|---------|---------|---------|
| `dev` | Interactive development | `docker compose -f docker-compose.dev.yml up -d` then `docker exec -it vllm-dev bash` |
| `serve` | Run vLLM server | `docker compose -f docker-compose.dev.yml --profile serve up serve` |

The `serve` service has the full `vllm serve` command pre-configured with MXFP4 settings.

### Workflow: Editing Kernels

1. **Edit kernel on host**: `vim ~/projects/flashinfer/include/flashinfer/moe/...`
2. **Clear JIT cache** (if needed): `docker exec vllm-dev rm -rf /root/.cache/flashinfer/`
3. **Run your test/server**: FlashInfer JIT detects changes and recompiles
4. **No container restart needed**

### Workflow: Editing Python Code

Python changes are immediate (no JIT involved):

1. **Edit Python on host**: `vim ~/projects/flashinfer/flashinfer/fused_moe/core.py`
2. **Run your test/server**: Changes take effect immediately

### Common Issues

**Issue: "Using wrong FlashInfer"**
```
# You see site-packages path instead of /workspace
import flashinfer
print(flashinfer.__file__)  # /usr/local/.../site-packages/flashinfer/__init__.py
```

**Fix**: Ensure PYTHONPATH is set correctly:
```bash
# Option A: Pass explicitly
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev python3 ...

# Option B: Use shell in container (inherits env from compose)
docker exec -it vllm-dev bash
python3 ...
```

**Issue: "Old kernel code running"**

FlashInfer caches compiled kernels. If you change `.cuh` files but see old behavior:

```bash
# Clear the JIT cache
docker exec vllm-dev rm -rf /root/.cache/flashinfer/
# Or from host:
rm -rf ~/projects/ai/mxfp4/.cache/flashinfer/
```

**Issue: "Container can't see my changes"**

Verify the volume mount is working:
```bash
# Create a test file on host
echo "test" > ~/projects/flashinfer/TEST_FILE

# Check it's visible in container
docker exec vllm-dev cat /workspace/flashinfer/TEST_FILE
```

### Environment Variables (Full List)

```yaml
environment:
  # Python path (CRITICAL for local repos)
  - PYTHONPATH=/workspace/flashinfer:/workspace/vllm${PYTHONPATH:+:$PYTHONPATH}
  
  # MXFP4 backend selection
  - VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
  - VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0
  - VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=0
  - VLLM_FLASHINFER_MOE_BACKEND=throughput
  - VLLM_USE_FLASHINFER_MOE_FP4=1
  
  # Performance settings
  - VLLM_ATTENTION_BACKEND=FLASHINFER    # Use FlashInfer for attention
  - VLLM_USE_CUDA_GRAPH=1                 # Enable CUDA graphs
  - FLASHINFER_NVCC_THREADS=4             # Parallel JIT compilation
  
  # Debugging (set to 1 to enable)
  - FLASHINFER_LOGLEVEL=0                 # 0=off, 1=basic, 3=detailed, 5=stats
  - FLASHINFER_JIT_VERBOSE=0              # Show JIT compilation output
```

---

## Known Issues / Gotchas

### 1. Wrong MoE API causes SM121 crash

**Symptom**: `RuntimeError: No supported CUDA architectures found for major versions [10]`

**Cause**: Using `trtllm_fp4_block_scale_moe` which targets SM100

**Fix**: Use `cutlass_fused_moe` instead

### 2. Previous M>=320 crash was API-specific

The old test (`verify_sm121_mxfp4_moe_fa2_sinks.py`) crashed at M>=320 tokens. This was due to using a different internal code path, NOT a CUTLASS kernel bug. The production `cutlass_fused_moe` API works for all M values.

### 3. KV cache layout is HND, not NHD

vLLM forces `HND` layout for KV cache on SM121. Ensure profiling/testing uses matching layout:

```python
wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    kv_layout="HND",  # Match vLLM's layout
)
```

### 4. PYTHONPATH required for editable install in Docker

FlashInfer's namespace package structure requires explicit PYTHONPATH. Without it, Python may import from site-packages instead of the mounted local repo.

### 5. Container recreation loses installed packages

When the container is **recreated** (not just restarted), FlashInfer and vLLM need to be reinstalled because they're installed from mounted volumes. However, `fastsafetensors` and `llama-benchy` are baked into the Dockerfile.dev image.

After `docker compose up -d`:

```bash
# Reinstall FlashInfer and vLLM from mounted repos
cd /workspace/flashinfer && uv pip install --no-build-isolation -e .
cd /workspace/vllm && python3 use_existing_torch.py && uv pip install -r requirements/build.txt && uv pip install --no-build-isolation -e .
```

**Note**: The Dockerfile.dev already includes `fastsafetensors`, `llama-benchy`, and tiktoken encodings.

### 6. CUDA graphs crash with MXFP4 path (requires --enforce-eager)

**Symptom**: Server crashes during "Capturing CUDA graphs" with `cudaErrorLaunchFailure`

**Cause**: CUDA graph capture fails with the MXFP4/MXFP8 kernel path on SM121

**Workaround**: Use `--enforce-eager` when starting the server

**Impact**: Decode performance may be affected (no CUDA graph optimization). This is a priority issue to fix.

### 7. Tiktoken encodings (baked into image)

gpt-oss-120b requires tiktoken encodings. The Dockerfile.dev downloads and installs them at `/workspace/tiktoken_encodings` and sets `TIKTOKEN_ENCODINGS_BASE` automatically. If you see tiktoken errors, ensure you're using the updated Dockerfile.dev image.

### 8. First-run JIT compilation takes ~3 minutes

FlashInfer compiles MoE CUTLASS kernels on first use. This JIT compilation:
- Takes **~3-5 minutes** on first server start
- Caches to `/root/.cache/flashinfer/` for subsequent runs
- May show zombie nvcc processes during compilation (normal)

If startup stalls, check `ps aux | grep nvcc` to see if compilation is in progress.

**Pre-warming the cache** (optional):
```bash
# Run after installing FlashInfer to avoid delay on first vLLM start
docker exec vllm-dev python3 /workspace/scripts/warmup_jit_cache.py
```

---

## Performance Targets (Scores to Beat)

### Current State (2026-01-08, enforce-eager mode)

| Test | Throughput (t/s) | TTFR (ms) |
|------|------------------|-----------|
| pp2048 | **4832 ± 35** | 425 ± 3 |
| tg32 | **29.54 ± 0.04** | - |
| pp2048 @ d512 | 3807 ± 19 | 673 ± 3 |
| pp2048 @ d1024 | 4218 ± 35 | 729 ± 6 |
| pp2048 @ d1536 | 4584 ± 14 | 783 ± 2 |
| tg32 @ depths | ~29 t/s | - |

**Note**: These results are with `--enforce-eager` (CUDA graphs disabled) due to a graph capture crash on SM121 with MXFP4 path.

### Targets to Beat

| Engine | pp2048 (t/s) | tg32 (t/s) |
|--------|--------------|------------|
| **llama.cpp** | 2449.83 | **57.85** |
| **SGLang** | - | **~52** |
| **Our vLLM** | **4808** ✓ | 29.26 ❌ |

### The Problem: Decode is 2x slower than competition

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **tg32** | 29.26 t/s | ≥52 t/s | **+78% needed** |
| **pp2048** | 4808 t/s | 2449 t/s | ✓ (2x better) |

**Prefill is excellent** - almost 2x llama.cpp. No work needed there.

**Decode is the blocker** - we need to nearly double decode throughput.

### SGLang Reference (single prompt)

```
Output token throughput: 52.37 tok/s
TTFT: 49.87 ms  
TPOT: 18.83 ms (time per output token)
```

### Full Benchmark Data

<details>
<summary>llama.cpp (llama-bench, build f5acfb2ff)</summary>

```
model                      test              t/s
gpt-oss 120B MXFP4 MoE    pp2048            2449.83 ± 10.27
gpt-oss 120B MXFP4 MoE    tg32              57.85 ± 0.44
gpt-oss 120B MXFP4 MoE    pp2048 @ d4096    2293.59 ± 8.99
gpt-oss 120B MXFP4 MoE    tg32 @ d4096      54.81 ± 0.30
gpt-oss 120B MXFP4 MoE    pp2048 @ d8192    2147.98 ± 10.64
gpt-oss 120B MXFP4 MoE    tg32 @ d8192      52.14 ± 0.50
gpt-oss 120B MXFP4 MoE    pp2048 @ d16384   1845.71 ± 7.11
gpt-oss 120B MXFP4 MoE    tg32 @ d16384     48.53 ± 0.36
gpt-oss 120B MXFP4 MoE    pp2048 @ d32768   1404.70 ± 7.36
gpt-oss 120B MXFP4 MoE    tg32 @ d32768     41.72 ± 0.18
```

</details>

<details>
<summary>vLLM (llama-benchy, single Spark)</summary>

```
model                 test               t/s           e2e_ttft (ms)
openai/gpt-oss-120b   pp2048             4663.70±42    614.72±3
openai/gpt-oss-120b   tg32               33.55±0.05    
openai/gpt-oss-120b   pp2048 @ d4096     3172.51±16    821.97±4
openai/gpt-oss-120b   tg32 @ d4096       32.63±0.02    
openai/gpt-oss-120b   pp2048 @ d8192     2687.84±9     941.50±3
openai/gpt-oss-120b   tg32 @ d8192       31.50±0.10    
openai/gpt-oss-120b   pp2048 @ d16384    2044.47±8     1186.10±5
openai/gpt-oss-120b   tg32 @ d16384      29.55±0.01    
openai/gpt-oss-120b   pp2048 @ d32768    1398.80±4     1659.47±5
openai/gpt-oss-120b   tg32 @ d32768      26.65±0.01    
```

</details>

### Decode Bottleneck Analysis

The decode gap (29 vs 58 tok/s = **2x slower**) is the critical issue.

**Possible causes (need profiling to confirm):**

1. **MoE routing overhead** - Per-token expert selection has fixed cost that doesn't amortize in decode
2. **FA2 decode kernel efficiency** - Is the SM121 decode kernel optimized?
3. **Python/IPC overhead** - vLLM's scheduler, ZMQ, async engine between tokens
4. **CUDA graph effectiveness** - Are graphs actually capturing decode iterations?
5. **KV cache access pattern** - Memory bandwidth during single-token decode
6. **Quantization overhead** - BF16→FP8 conversion on every decode step?

**Investigation findings (2026-01-08):**

1. ~~**CUDA graphs crash during capture**~~ → **RESOLVED**: CUDA graphs now work on SM121 with MXFP4
2. **CUDA graphs don't improve decode performance** (~28.7 tok/s with graphs vs ~29 tok/s without)
   - Bottleneck is kernel execution time, not launch overhead
3. **BF16 MoE kernel is very slow** for decode (M=1): ~2.3ms/call → 137ms/token (60 layers) → max 7.3 tok/s
4. The **MXFP4 path must be faster** since vLLM achieves ~30 tok/s with it
5. **Container recreation** loses all pip-installed packages

**nsys Profiling Results (2026-01-08):**

Captured ~52 decode tokens with nsys profiling. Top kernel breakdown:

| Kernel | Time % | Total (s) | Instances | Avg (μs) |
|--------|--------|-----------|-----------|----------|
| GEMV (attn proj) | 31.8% | 1.88s | 12312 | 153 |
| **CUTLASS MoE (FC1)** | 28.2% | 1.67s | 6300 | **266** |
| GEMV (other) | 19.4% | 1.15s | 6331 | 181 |
| **CUTLASS MoE (FC2)** | 12.3% | 0.73s | 6300 | **115** |
| Activation (FP8) | 1.9% | 0.11s | 6300 | 18 |
| Expand input rows | 1.9% | 0.11s | 6300 | 18 |

**Key observations:**
- **MoE kernels take 40.5%** of GPU time (FC1 + FC2)
- Per-token MoE time: (266 + 115) μs × 60 layers ≈ **22.9ms**
- Per-token GEMV time: 153μs × 237 calls ≈ **36ms**
- Total kernel time per token: ~60ms → theoretical **16.7 tok/s**
- Actual: 29 tok/s → overlap/parallelism helping

**Bottleneck conclusion:**
The **CUTLASS grouped GEMM** (MoE) is 40% of decode time. FC1 kernel (266μs) is 2.3x slower than FC2 (115μs) - likely due to SwiGLU fused activation. Attention GEMV is also significant.

**Root cause analysis (2026-01-08):**

**MoE-SPECIFIC LIMITATION DISCOVERED**: While CUTLASS supports M=64 tiles for block-scaled GEMM (see `87b_blackwell_geforce_*.cu`), the FlashInfer MoE kernel cannot use them due to scale factor layout constraints.

**Why the MoE kernel is limited to M=128:**
1. MoE uses `kSm12xBlockScaleGranularity = 128` (each scale covers 128 elements)
2. TMA layout requires tile alignment with scale granularity
3. When M=64 < 128, TMA fails: "CTA_Tile and SLayout top-level size equivalence"

**Why CUTLASS example works with M=64:**
The example uses `ScaleGranularityM = 1` (per-element scaling), which is completely different from MoE's 128-element blocks.

**What would be needed to fix:**
1. Change `kSm12xBlockScaleGranularity` from 128 to 64 or smaller
2. Redesign scale factor packing in `sm12x_activation_quantizer.cuh`
3. Change weight quantization format (scales per 64 elements)
4. This is a significant architectural change, not just tile config

**Impact for M=1 decode:**
- 128×128 tile with 1 actual row → **0.78% compute efficiency**
- This is a fundamental limitation, not a software bug

**Attempted Solution (FAILED):**
Tried adding M=64 tiles but got TMA layout errors. The MoE kernel's scale factor granularity (128 elements) makes smaller tiles incompatible.

**Viable Alternative Solutions:**

1. **Speculative decoding**: Generate multiple candidate tokens → M > 1
   - vLLM has built-in support for speculative decoding
   - Would naturally increase batch size for MoE

2. **Token batching**: Accumulate decode tokens before MoE processing
   - Requires scheduler changes in vLLM
   - Trades latency for throughput

3. **Reduce scale granularity**: Change `kSm12xBlockScaleGranularity` from 128 to 64
   - Significant MoE kernel refactoring required
   - Would need to change weight quantization format
   - High effort, high reward

**Why llama.cpp achieves 2x decode performance (58 vs 29 tok/s):**
- Likely uses a different approach entirely:
  - Custom GEMV kernel with on-the-fly dequantization
  - Different scale factor layout (per-element or smaller groups)
  - Not constrained by TMA block-scaled GEMM
- Investigation of llama.cpp's CUDA backend would reveal their strategy

---

## Speculative Decoding Investigation (2026-01-08)

### Ngram Speculation Results

Tested vLLM's built-in ngram speculative decoding:

```bash
vllm serve ... \
    --speculative-config.method=ngram \
    --speculative-config.num_speculative_tokens=4 \
    --speculative-config.prompt_lookup_max=5 \
    --speculative-config.prompt_lookup_min=1
```

**Results:**
| Prompt Type | Throughput | Baseline |
|-------------|------------|----------|
| Open-ended generation | ~17-20 tok/s | 29 tok/s |
| Repetitive patterns | Server crash | 29 tok/s |

**Conclusion:** Ngram speculation is **NOT suitable** for gpt-oss-120b because:
1. Open-ended generation rarely repeats patterns from the prompt
2. Speculation overhead exceeds benefits when speculation fails
3. Async scheduling not supported with ngram (falls back to sync)

### Alternative Approaches (Not Yet Tested)

1. **Draft model speculation** - Uses a smaller model for drafting
   - Better for open-ended generation
   - Requires a compatible draft model

2. **Suffix decoding** - Uses global response cache
   - Better for structured outputs (JSON, code)
   - `--speculative-config.method=suffix`

3. **EAGLE/MTP** - Model-specific speculation heads
   - Highest acceptance rate
   - Requires model-specific training

### GEMV Fallback Status

Created infrastructure for GEMV-based decode optimization:
- `flashinfer/gemv/` - Python module with heuristic
- `csrc/gemv/gemv_fp4_blockscaled.cu` - CUDA kernel (software fallback)
- `csrc/gemv/gemv_epilogue_bf16.h` - Custom epilogue for BF16 output

**Benchmark Results (DP4A llama.cpp-style kernel):**
```
FC1 (4096→11776): 0.31 ms at 312 GFLOPS (1.32x faster than PyTorch BF16)
FC2 (11776→4096): 0.25 ms at 380 GFLOPS (1.52x faster than PyTorch BF16)
```

### CRITICAL FINDING: GEMV is NOT the Solution for MoE

**GEMV is slower than CUTLASS grouped GEMM for MoE decode!**

| Approach | 60 Layers (TopK=8) | Why |
|----------|---------------------|-----|
| CUTLASS grouped GEMM | 182.88 ms | Batches all experts, reuses weights |
| DP4A per-expert GEMV | 270.38 ms | 8x memory traffic (no weight reuse) |

The grouped GEMM's weight reuse across all 8 experts outweighs the M=128 tile inefficiency.

### Updated Next Steps

1. ~~Investigate llama.cpp's decode kernel~~ (Done - implemented DP4A kernel)
2. ~~Benchmark GEMV vs grouped GEMM~~ (Done - grouped GEMM wins)
3. **Focus on attention GEMV** (51% of decode time per nsys)
4. **Try draft model speculation** to increase effective batch size
5. **Profile llama.cpp architecture** to understand their dense model advantage

---

## Callsite Logging (Proving Correct Dispatch)

We added one-time callsite logging to vLLM to prove the correct MXFP4 path is used.

**Expected log output on SM121:**
```
[MoE] backend=flashinfer format=MXFP4_NATIVE fn=flashinfer.fused_moe.cutlass_fused_moe
[MoE] w13_weight dtype=torch.uint8 shape=(256, 8192, 2048)
[MoE] w2_weight dtype=torch.uint8 shape=(256, 4096, 4096)
[MoE] w13_weight_scale dtype=torch.float8_e4m3fn shape=(256, 8192, 64)
[MoE] w2_weight_scale dtype=torch.float8_e4m3fn shape=(256, 4096, 128)
[MoE] activation input dtype=torch.bfloat16 shape=(32, 4096) → MXFP8 quantized
```

**Red flags to watch for:**
- `[MoE] WARNING: Using NVFP4 path` → Wrong quantization format
- `format=MXFP4_BF16` on SM121 → Should be `MXFP4_NATIVE`
- `quant_dtype=nvfp4` → Wrong path taken

**Files modified:**
- `vllm/model_executor/layers/quantization/mxfp4.py` (lines ~1007-1030)
- `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` (FlashInferExperts.apply)

---

## Testing Commands

```bash
# Verify FlashInfer import location
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 -c "import flashinfer; print(flashinfer.__file__)"

# Run MXFP4 MoE kernel verification
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 /workspace/verify_mxfp4_moe_kernel.py --quick

# Run full test harness (inside container)
cd /workspace && ./sm121_vllm_test_harness.sh --mode verify --skip-server

# Profile decode performance
docker exec -e PYTHONPATH=/workspace/flashinfer:/workspace/vllm vllm-dev \
  python3 /workspace/profile_sm121_decode_performance.py --kv-layout HND
```

---

## Summary for AI Agents

When working on this project:

1. **We use CUTLASS via FlashInfer**, not TensorRT-LLM or cuDNN
2. **The correct MoE API is `cutlass_fused_moe`**, not `trtllm_*` functions
3. **SM121 uses FA2** (not FA3), with attention sinks wired up
4. **MXFP4 uses group size 32**, goes through FP8×FP4 kernel internally
5. **Always verify PYTHONPATH** points to local FlashInfer repo
6. **KV cache is HND layout** on SM121/vLLM

If you're debugging crashes or wrong backends, start by verifying which FlashInfer you're actually importing.

