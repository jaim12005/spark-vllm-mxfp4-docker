# vLLM Baseline Analysis (SM121 / gpt-oss-120b)

**Status**: Complete - 2026-01-09

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

### Comparison to Targets

| Engine | tg32 (tok/s) | Gap to Baseline |
|--------|--------------|-----------------|
| vLLM (baseline) | 32.14 | - |
| SGLang | 52.37 | +63% faster |
| llama.cpp | 57.85 | +80% faster |

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
