# Building Flash Attention 2 with Native SM121 Support

## Background

The Dao-AILab Flash Attention 2 library (used by vLLM as `vllm_flash_attn`) ships with SM80 cubins and PTX for forward compatibility. On SM121 (GB10/DGX Spark), the PTX JIT compilation produces **numerically incorrect results** - outputs are garbage/NaN.

This document describes how to build FA2 with native SM121 SASS code.

## The Problem

Default FA2 CMakeLists.txt sets:
```cmake
cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
```

This means:
- Native SM80 cubins are included
- PTX is included for forward compatibility
- On SM121, CUDA driver JIT-compiles PTX → SM121 SASS at runtime
- **This JIT path produces incorrect numerical results on Blackwell**

See: https://github.com/Dao-AILab/flash-attention/issues/1969

## Solution: Build with Native SM121

### Prerequisites

- CUDA 13.0+ (required for SM121 support)
- Container with NVIDIA PyTorch base image
- vLLM source at `/workspace/vllm`

### Step 1: Locate FA2 Source

vLLM includes FA2 source at:
```bash
/workspace/vllm/.deps/vllm-flash-attn-src/
```

### Step 2: Patch CMakeLists.txt

Edit `/workspace/vllm/.deps/vllm-flash-attn-src/CMakeLists.txt`:

**Before (line ~140):**
```cmake
cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
```

**After:**
```cmake
cuda_archs_loose_intersection(FA2_ARCHS "8.0;12.1" "${CUDA_ARCHS}")
```

This removes PTX and adds native SM121.

### Step 3: Verify CUDA_SUPPORTED_ARCHS Includes 12.1

Check that 12.1 is in the supported archs list (around line 13-17):
```cmake
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    list(APPEND CUDA_SUPPORTED_ARCHS "10.0" "11.0" "12.0" "12.1")
```

If not present, add it.

### Step 4: Configure with CMake

```bash
cd /workspace/vllm/.deps/vllm-flash-attn-src
rm -rf build/
mkdir -p build && cd build

export TORCH_CUDA_ARCH_LIST='8.0;12.1'

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFA3_ENABLED=OFF  # Optional: skip FA3 to speed up build
```

Verify output shows:
```
-- FA2_ARCHS: 8.0;12.1
-- Added CUDA NVCC flags for: -gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_121,code=sm_121
```

### Step 5: Build

```bash
cmake --build . --parallel 12
```

Build takes ~10-15 minutes with 12 parallel jobs.

### Step 6: Verify SM121 in Binary

```bash
strings _vllm_fa2_C.abi3.so | grep -E 'sm_[0-9]+' | sort -u
```

Expected output:
```
-arch sm_121 -m 64 
-arch sm_80 -m 64 
```

### Step 7: Install

Copy the built .so to vLLM's flash_attn directory:
```bash
cp _vllm_fa2_C.abi3.so /workspace/vllm/vllm/vllm_flash_attn/
```

### Step 8: Test

```python
import torch
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

# Test parameters
batch, seqlen, nheads, headdim = 2, 128, 8, 64

q = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
k = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
v = torch.randn(batch * seqlen, nheads, headdim, dtype=torch.float16, device='cuda')
cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device='cuda')

out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, causal=True)

print(f'Output has NaN: {torch.isnan(out).any().item()}')  # Should be False
print(f'Output has Inf: {torch.isinf(out).any().item()}')  # Should be False
print('SM121 FA2 test PASSED!')
```

## Pre-built Binary

A pre-built `_vllm_fa2_C.abi3.so` with SM80+SM121 is available at:
```
/home/swank/projects/ai/mxfp4/fa2_sm121.so
```

To use:
```bash
cp /home/swank/projects/ai/mxfp4/fa2_sm121.so /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
```

## Notes

1. **Do NOT use pip install for FA2** - it will try to satisfy `torch == 2.4.0` dependency and may overwrite NVIDIA's custom torch with a CPU-only version.

2. **Build parallelism** - Use `--parallel 12` or lower. Higher values with model loaded can cause OOM.

3. **FA3 not needed** - FA3 is SM90-only (Hopper). Disable with `-DFA3_ENABLED=OFF` to speed up build.

4. **CUTLASS warnings** - Deprecation warnings about `long4`, `ulong4`, etc. are harmless and can be ignored.

## Verification with vLLM

After installing, test with vLLM:
```bash
export VLLM_ATTENTION_SINKS=false

python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --quantization mxfp4 \
    --attention-config '{"backend": "FLASH_ATTN"}' \
    --max-model-len 8192 \
    --enforce-eager
```

Send a test request and verify coherent output (not garbage).
