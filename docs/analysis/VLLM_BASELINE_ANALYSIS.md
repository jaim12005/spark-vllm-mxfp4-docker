# vLLM Baseline Analysis (SM121 / gpt-oss-120b)

**Status**: Complete with nsys Profile - 2026-01-09

## Executive Summary

On SM121 (GB10 DGX Spark), upstream vLLM achieves **32.14 tok/s decode** (tg32) and **4340 tok/s prefill** (pp2048). This is ~45% slower decode than llama.cpp (58 tok/s) and SGLang (52 tok/s).

**Root cause**: SM121 falls through all optimized code paths and lands on fallback implementations:
- **Attention**: TRITON_ATTN (not FlashInfer or Flash Attention)
- **MoE**: Marlin (not CUTLASS or FlashInfer)
- **FP4**: Weight-only compression (not native SM121 block-scaled MMA)

---

## Architecture Detection Issues

### SM121 is Not Recognized as "Blackwell Class"

The key issue is how vLLM classifies GPU architectures:

```python
# vllm/platforms/interface.py:329-341
def is_device_capability_family(cls, capability: int, device_id: int = 0) -> bool:
    """Returns True if the device capability is any <major>.x."""
    current_capability = cls.get_device_capability(device_id=device_id)
    return (current_capability.to_int() // 10) == (capability // 10)
```

For SM121:
- `capability.to_int()` = 121
- `121 // 10` = 12
- `is_device_capability_family(100)` checks: `12 == 10` → **False**

**SM121 is family 120, not family 100 (Blackwell)**. All FlashInfer MXFP4 paths require `is_device_capability_family(100)`.

---

## Attention Backend Selection

### Priority Order (non-MLA, SM12x)

From `vllm/platforms/cuda.py:76-82`:

```python
# For device_capability.major != 10:
return [
    AttentionBackendEnum.FLASH_ATTN,    # Priority 0
    AttentionBackendEnum.FLASHINFER,    # Priority 1
    AttentionBackendEnum.TRITON_ATTN,   # Priority 2
    AttentionBackendEnum.FLEX_ATTENTION, # Priority 3
]
```

### Why Each Backend Fails

#### FLASH_ATTN
- **supports_compute_capability**: ✅ `capability >= 8.0` → True for SM121
- **supports_sink**: ❓ Requires Flash Attention 3 (`get_flash_attn_version() == 3`)
- **Result**: Fails if FA3 not available/configured

From `vllm/attention/utils/fa_utils.py:105-111`:
```python
def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3
```

#### FLASHINFER
- **supports_compute_capability**: ✅ `7.5 <= SM <= 12.1` → True
- **supports_sink**: ❌ Requires `is_device_capability_family(100)` → False for SM121

From `vllm/v1/attention/backends/flashinfer.py:358-371`:
```python
@classmethod
def supports_sink(cls) -> bool:
    """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
    return supports_trtllm_attention()

# vllm/utils/flashinfer.py:284-286:
def supports_trtllm_attention() -> bool:
    return (
        current_platform.is_device_capability_family(100) and has_nvidia_artifactory()
    )
```

**Result**: `supports_sink()` returns False → "sink setting not supported"

#### TRITON_ATTN (Selected)
- **supports_compute_capability**: ✅ Always True
- **supports_sink**: ✅ Always True

From `vllm/v1/attention/backends/triton_attn.py:323-337`:
```python
@classmethod
def supports_sink(cls) -> bool:
    return True

@classmethod
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    return True
```

**Result**: TRITON_ATTN is selected as the fallback.

### gpt-oss-120b Has Attention Sinks

The validation error "sink setting not supported" indicates the model configuration has `has_sink=True`. This triggers the sink validation check in `validate_configuration()`.

---

## MoE Backend Selection

### Decision Logic

From `vllm/model_executor/layers/quantization/mxfp4.py:109-218`:

```python
def get_mxfp4_backend(with_lora_support: bool) -> Mxfp4Backend:
    if current_platform.is_cuda():
        # Check 1: SM90 + FlashInfer + MXFP4_BF16 env var
        if is_device_capability(90) and has_flashinfer() and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
            return Mxfp4Backend.SM90_FI_MXFP4_BF16
        
        # Check 2: SM10x (Blackwell) + FlashInfer + CUTLASS env var
        elif is_device_capability_family(100) and has_flashinfer() and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS:
            return Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        
        # Check 3: SM10x (Blackwell) + FlashInfer + MXFP8 env var  
        elif is_device_capability_family(100) and has_flashinfer() and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:
            return Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
        
        # Check 4: SM10x (Blackwell) + FlashInfer (default BF16)
        elif is_device_capability_family(100) and has_flashinfer():
            # Special handling for SM12x here - but requires is_device_capability_family(100)!
            return Mxfp4Backend.SM100_FI_MXFP4_BF16
        
        # Fallback: Triton or Marlin
        triton_kernels_supported = (
            has_triton_kernels()
            and is_torch_equal_or_newer("2.8.0")
            and (9, 0) <= current_platform.get_device_capability() < (11, 0)  # SM90-SM100 only!
        )
        
        if envs.VLLM_MXFP4_USE_MARLIN or not triton_kernels_supported:
            logger.info_once("Using Marlin backend")
            return Mxfp4Backend.MARLIN
```

### SM121 Evaluation

| Check | Condition | SM121 Result |
|-------|-----------|--------------|
| SM90 + FlashInfer | `is_device_capability(90)` | ❌ False (is 121) |
| SM10x + CUTLASS | `is_device_capability_family(100)` | ❌ False (family 120) |
| SM10x + MXFP8 | `is_device_capability_family(100)` | ❌ False |
| SM10x + BF16 | `is_device_capability_family(100)` | ❌ False |
| Triton kernels | `(9, 0) <= (12, 1) < (11, 0)` | ❌ False (12.1 >= 11.0) |

**Result**: Falls through to Marlin backend.

### Marlin Backend Characteristics

From the log:
```
WARNING: Your GPU does not have native support for FP4 computation but FP4 
quantization is being used. Weight-only FP4 compression will be used leveraging 
the Marlin kernel. This may degrade performance for compute-heavy workloads.
```

Marlin implements:
- Weight-only FP4 decompression to FP16/BF16
- Standard GEMM operations (no native FP4 tensor cores)
- No block-scaled MMA optimizations

---

## Code Paths That Would Enable Native FP4

### In mxfp4.py (Lines 169-185)

There IS code for SM12x, but it's gated incorrectly:

```python
elif current_platform.is_blackwell_class() and has_flashinfer():
    # Check if this is SM12x (GB10 DGX Spark, Thor) - use native CUTLASS path
    capability = current_platform.get_device_capability()
    if capability and capability.major == 12:
        logger.info_once(
            "Using FlashInfer MXFP4 MXFP8 CUTLASS backend for SM12x"
        )
        return Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
```

**Problem**: The outer condition `is_blackwell_class()` uses `is_device_capability_family(100)`, which returns False for SM121. The SM12x-specific code inside is never reached.

---

## Backend Capabilities Comparison

| Backend | Attention | MoE | Native FP4 | Notes |
|---------|-----------|-----|------------|-------|
| **Current (Upstream)** | TRITON_ATTN | Marlin | ❌ Weight-only | Fallback path |
| **PR #31740** | FLASHINFER | CUTLASS | ✅ Block-scaled | Requires FlashInfer SM12x |
| **Target** | FlashInfer FA2 | CUTLASS grouped | ✅ FP8×FP4 MMA | Native SM121 path |

---

## Performance Impact

### Measured Baseline

| Metric | Value |
|--------|-------|
| pp2048 (prefill) | 4340 tok/s |
| tg32 (decode) | 32.14 tok/s |
| TTFT | ~589 ms |

---

## nsys Profile Analysis (Baseline)

**Profile captured**: 2026-01-09 with 7 inference requests (2 warmup + 5 profiled)

### Top GPU Kernels by Time

| Rank | Kernel | Time % | Total (ms) | Calls | Avg (µs) |
|------|--------|--------|------------|-------|----------|
| 1 | **Marlin MoE (large)** | 20.8% | 1938 | 12168 | 159 |
| 2 | **GEMV (dense layers)** | 19.8% | 1838 | 11592 | 159 |
| 3 | **GEMV variant** | 12.0% | 1116 | 5964 | 187 |
| 4 | elementwise (activations) | 9.3% | 862 | 9216 | 94 |
| 5 | elementwise variant | 6.9% | 644 | 144 | 4472 |
| 6 | **Marlin MoE (prefill)** | 6.9% | 640 | 144 | 4445 |
| 7 | CatArrayBatchedCopy | 5.9% | 546 | 72 | 7582 |
| 8 | Fill kernel | 3.9% | 361 | 144 | 2509 |
| 9 | **gptq_marlin_repack** | 3.6% | 333 | 9216 | 36 |
| 10 | CUTLASS GEMM (attention proj) | 1.1% | 105 | 144 | 726 |
| 11 | WMMA GEMM | 0.5% | 46 | 254 | 182 |
| 12 | **Triton attention (decode)** | 0.3% | 24 | 5796 | 4 |
| 13 | **Triton attention (prefill)** | 0.0% | 1.6 | 252 | 6 |

### Key Observations

1. **MoE dominates (~31%)**: Marlin MoE + repack kernels account for ~31% of GPU time
2. **Dense GEMV is significant (~32%)**: Two GEMV kernels for attention projections
3. **Attention is cheap**: `kernel_unified_attention_3d` (decode) only 0.3%, 4µs avg
4. **Memory copies**: CatArrayBatchedCopy at 5.9% indicates tensor concatenation overhead
5. **Marlin repack overhead**: 3.6% spent just repacking weights for Marlin

### Kernel Breakdown by Category

| Category | Time % | Notes |
|----------|--------|-------|
| **MoE (Marlin)** | ~31% | Weight-only FP4, not using native FP4 tensor cores |
| **Dense GEMM/GEMV** | ~33% | Attention projections (Q, K, V, O) |
| **Memory/Copy** | ~10% | Tensor concatenation, fills |
| **Attention** | ~0.5% | Triton attention is very fast |
| **Other** | ~25% | Activations, RMSNorm, etc. |

### Memory Bandwidth

| Operation | Total (GB) | Count | Notes |
|-----------|------------|-------|-------|
| H2D (model load) | 65.3 GB | 29949 | Initial weight transfer |
| D2D copies | 0.13 GB | 23349 | Per-token copies |
| D2H copies | 0.001 GB | 168 | Output tokens |

### Decode vs Prefill

The profile captured both prefill (~200 tokens) and decode (32 tokens × 5 requests):

- **Prefill**: Dominated by large Marlin MoE calls (4.4ms avg)
- **Decode**: Many small GEMV calls (159µs avg) with 4µs attention

### Performance Bottleneck Analysis

**Why 32 tok/s instead of 58 tok/s?**

1. **Marlin is not optimal for SM121**: Uses weight-only FP4 → BF16 decompression, not native FP4 tensor cores
2. **GEMV overhead**: 32% of time in GEMV suggests small batch sizes (expected for decode)
3. **Marlin repack**: 3.6% overhead just preparing weights
4. **No CUDA graphs**: Running in eager mode

**Expected improvements with CUTLASS MoE**:
- Native FP8×FP4 block-scaled MMA on SM121
- No repack overhead
- Better memory coalescing for MoE expert selection

---

## Comprehensive Benchmark Results

**Test Date**: 2026-01-09
**Configuration**: TRITON_ATTN + Marlin, enforce_eager, fastsafetensors

### Varying Prefill Length (tg=32)

| Prefill | Throughput (tok/s) | TTFT (ms) | Decode (tok/s) |
|---------|-------------------|-----------|----------------|
| pp512 | 2209 ± 97 | 332 ± 12 | 32.22 ± 0.07 |
| pp1024 | 3386 ± 81 | 418 ± 2 | 31.96 ± 0.02 |
| pp2048 | 4341 ± 12 | 555 ± 14 | 31.63 ± 0.04 |
| pp4096 | 4008 ± 23 | 1051 ± 9 | 30.96 ± 0.06 |

**Observations**:
- Prefill throughput peaks around pp2048 (~4.3k tok/s)
- Decode performance is stable ~31-32 tok/s regardless of context length
- TTFT scales linearly with prefill length

### Varying Decode Length (pp=2048)

| Decode | Throughput (tok/s) |
|--------|-------------------|
| tg16 | 31.53 ± 0.06 |
| tg32 | 31.58 ± 0.06 |
| tg64 | 31.62 ± 0.03 |
| tg128 | 31.62 ± 0.05 |
| tg256 | 31.12 ± 0.35 |

**Observations**:
- Decode throughput is consistent ~31.5 tok/s
- Slight degradation at tg256 (context length growing)
- Memory-bound, not compute-bound

### Comparison to Targets

| Engine | tg32 (tok/s) | Gap to Baseline |
|--------|--------------|-----------------|
| vLLM (baseline) | 32.14 | - |
| SGLang | 52.37 | +63% faster |
| llama.cpp | 57.85 | +80% faster |

---

### Expected Impact of Fixes

1. **FlashInfer Attention**: Replace TRITON_ATTN with FlashInfer FA2
   - Expected: 10-20% improvement
   
2. **CUTLASS MoE**: Replace Marlin with CUTLASS grouped GEMM
   - Expected: 20-40% improvement
   
3. **Native FP4**: Enable block-scaled FP8×FP4 MMA
   - Expected: 10-30% improvement

---

## Required Fixes

### 1. Architecture Detection (vLLM)

Either:
- **Option A**: Treat SM12x as "Blackwell class" (expand `is_device_capability_family(100)` to include 12x)
- **Option B**: Add explicit SM12x handling alongside SM10x checks

### 2. Attention Sink Support

For FlashInfer to work on SM121:
- `supports_sink()` must return True for SM121
- Or model must be configured without attention sinks

### 3. FlashInfer SM12x Kernels

FlashInfer `upstream/main` lacks SM12x JIT compilation support. The error:
```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

Indicates FlashInfer's compilation context needs SM12x entries.

---

## Verification Commands

```bash
# Check which backend is selected
FLASHINFER_LOGLEVEL=3 vllm serve openai/gpt-oss-120b --quantization mxfp4 ... 2>&1 | grep -i "backend\|marlin\|cutlass\|triton"

# Check device capability detection
python3 -c "
from vllm.platforms import current_platform
cap = current_platform.get_device_capability()
print(f'Capability: SM{cap.major}{cap.minor}')
print(f'Family 100: {current_platform.is_device_capability_family(100)}')
print(f'Family 120: {current_platform.is_device_capability_family(120)}')
"
```

---

## Data Representation

### MXFP4 Weight Format

From `mxfp4.py:270-275` and MX spec:

| Component | Format | Size | Description |
|-----------|--------|------|-------------|
| **Weight data** | E2M1 (4-bit) | 2 elements/byte | Packed FP4 nibbles |
| **Block scale** | E8M0 (8-bit) | 1 byte/32 elements | Shared exponent per block |
| **Block size** | 32 elements | - | MX format standard |

```python
# Weight tensor shapes (from create_weights):
w13_weight: [num_experts, 2*intermediate_size, hidden_size//2]  # dtype=uint8
w13_weight_scale: [num_experts, 2*intermediate_size, hidden_size//32]  # dtype=uint8
```

### Activation Quantization by Backend

| Backend | Activation Input | Quantization | Notes |
|---------|-----------------|--------------|-------|
| **Marlin** | BF16 | None | Weight-only FP4, activations stay BF16 |
| **SM100_FI_MXFP4_BF16** | BF16 | None | BF16 activations into FP4 weights |
| **SM100_FI_MXFP4_MXFP8** | BF16→MXFP8 | `mxfp8_quantize()` | Activations quantized to MXFP8 |
| **SM100_FI_MXFP4_MXFP8_CUTLASS** | BF16→MXFP8 | `mxfp8_quantize(x, True, 32)` | Block size 32 |

From `mxfp4.py:996-998`:
```python
# MXFP8 activation quantization (when enabled)
from flashinfer import mxfp8_quantize
x_quant, x_scale = mxfp8_quantize(x, True, 32)  # rowwise=True, block_size=32
```

---

## Attention Sinks in gpt-oss-120b

### What Are Attention Sinks?

Attention sinks are learned parameters that help stabilize attention patterns, particularly for long sequences. They act as "anchor" positions that prevent attention drift.

### Implementation in gpt-oss

From `vllm/model_executor/models/gpt_oss.py:88-127`:

```python
class OAIAttention(nn.Module):
    def __init__(self, ...):
        # Per-head learned sink parameters
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size, requires_grad=False)
        )
        
        self.attn = Attention(
            ...,
            sinks=self.sinks,  # Passed to attention layer
        )
```

### Why This Matters

1. **Backend requirement**: Any attention backend must support `has_sink=True`
2. **FlashInfer limitation**: Only supports sinks via TRTLLM attention (SM100 only in upstream)
3. **TRITON_ATTN fallback**: Always claims `supports_sink()=True`, but may not use them optimally

---

## KV Cache Layout

### HND vs NHD

| Layout | Order | When Used |
|--------|-------|-----------|
| **HND** | [num_heads, num_tokens, head_dim] | Blackwell-class (SM10x, SM12x) |
| **NHD** | [num_tokens, num_heads, head_dim] | Other architectures |

From `vllm/v1/attention/backends/flashinfer.py:374-382`:

```python
@classmethod
def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
    capability = current_platform.get_device_capability()
    # Blackwell-class: SM10x, SM11x, SM12x (GB10)
    if capability is not None and capability.major in (10, 11, 12):
        return "HND"
    return None
```

**Note**: PR #31740 already includes SM12x in the HND layout requirement.

---

## Padding Requirements by Backend

From `mxfp4.py` `create_weights()`:

| Backend | Intermediate Size | Hidden Size | Reason |
|---------|-------------------|-------------|--------|
| Marlin | round_up(n, 128) | round_up(n, 256) | Marlin kernel constraints |
| SM100_FI_MXFP4_TRTLLM | round_up(n, 256) | round_up(n, 256) | TMA alignment |
| SM100_FI_MXFP4_CUTLASS | round_up(n, 128) | round_up(n, 128) | CUTLASS tile size |
| Triton | varies | varies | Platform-specific |

---

## Source File References

| File | Key Functions |
|------|---------------|
| `vllm/platforms/cuda.py:45-82` | `_get_backend_priorities()` - attention backend order |
| `vllm/platforms/interface.py:329-341` | `is_device_capability_family()` - arch detection |
| `vllm/model_executor/layers/quantization/mxfp4.py:109-218` | `get_mxfp4_backend()` - MoE kernel selection |
| `vllm/v1/attention/backends/flashinfer.py:358-371` | `supports_sink()` - FlashInfer validation |
| `vllm/utils/flashinfer.py:274-286` | `supports_trtllm_attention()` - SM100 check |
| `vllm/attention/backends/abstract.py:206-260` | `validate_configuration()` - backend validation |

---

## Conclusions

1. **SM121 is in a "blind spot"** - neither recognized as SM10x (Blackwell) nor SM90 (Hopper)
2. **All optimized paths are gated** behind `is_device_capability_family(100)` which is False for SM12x
3. **FlashInfer lacks SM12x support** in upstream - JIT compilation fails
4. **The 80% decode performance gap** (32 vs 58 tok/s) is due to using fallback implementations

The fixes in PR #31740 address these issues by:
- Adding SM12x detection as a variant of Blackwell
- Enabling FlashInfer attention with SM12x support
- Using CUTLASS grouped GEMM instead of Marlin
