# GPT-OSS-120B with MXFP4 on DGX Spark (SM121/GB10)

This Docker setup enables running GPT-OSS-120B with native MXFP4 quantization on NVIDIA DGX Spark (SM121/GB10 Blackwell-class GPU).

## Key Components

| Component | Version/Details |
|-----------|-----------------|
| Base Image | `nvcr.io/nvidia/pytorch:25.12-py3` |
| CUDA | 13.0+ (13.1 in NGC container) |
| cuDNN | 9.15+ (required for MXFP4 on SM121) |
| vLLM | PR #31740 (adds SM121 Blackwell-class support) |
| FlashInfer | Latest with CUTLASS/cuDNN backends |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GPT-OSS-120B Model                      │
│                    (MXFP4 Quantized Weights)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         vLLM                                │
│   (PR #31740: is_blackwell_class() includes SM121)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FlashInfer                             │
│   Fused MoE Backend Selection for SM121:                    │
│   ├── CUTLASS Backend ✓ (SM90_FI_MXFP4_BF16)                │
│   ├── cuDNN Backend ✓ (MXFP4 supported, requires ≥9.14)     │
│   └── TRTLLM Backend ✗ (SM100/SM103 only)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  DGX Spark (SM121/GB10)                     │
│   Blackwell-class GPU with FP4 Tensor Core support         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Deep Dive

### Critical: CUDA Architecture Suffix

**The most important configuration for SM121 FP4 support:**

```bash
# ❌ WRONG - Software FP4 emulation, will fail with cvt.e2m1x2 error
FLASHINFER_CUDA_ARCH_LIST="12.1"

# ✅ CORRECT - Hardware FP4 path enabled
FLASHINFER_CUDA_ARCH_LIST="12.1a"
```

The `a` suffix (or `f` for some architectures) enables `__CUDA_ARCH_FAMILY_SPECIFIC__`, which:
- Activates hardware FP4 conversion instructions (`cvt.rn.satfinite.e2m1x2.f32`)
- Enables architecture-specific optimizations
- Without it, ptxas will error: `Feature 'cvt.e2m1x2.f32' not supported on .target 'sm_121'`

### Backend Comparison

| Backend | SM121 Support | MXFP4 | NVFP4 | Notes |
|---------|--------------|-------|-------|-------|
| **CUTLASS** | ✅ | ❌ | ✅ | 128x128x128 tiles, 1x1x1 cluster only |
| **cuDNN** | ✅ | ✅ | ✅ | Requires cuDNN ≥ 9.14, preferred for MXFP4 |
| **TRTLLM** | ❌ | - | - | SM100/SM103 only, hardcoded architecture |
| **Deep GEMM** | ❌ | - | - | SM90 (Hopper) only |

### What is Deep GEMM?

Deep GEMM is NVIDIA's high-performance GEMM library from TensorRT-LLM, specifically optimized for Hopper (SM90). It provides significant performance gains through Hopper-specific features but **cannot be ported to SM121**.

#### Why Deep GEMM Cannot Work on SM121

| Evidence | Location | Finding |
|----------|----------|---------|
| Hardcoded architecture | `compiler.cuh:282` | `--gpu-architecture=sm_90a` |
| Runtime assertion | `fp8_gemm_impl.cuh:435,820` | `"This kernel only support sm_90a"` |
| WGMMA instructions | `fp8_gemm_impl.cuh` | 43 references to SM90-only warp-group MMA |
| Cluster/multicast | `fp8_gemm.cuh` | 19 references to TMA multicast (SM121: 1×1×1 only) |
| MMA tile structures | `mma_utils.cuh` | 16 SM90-specific structs (`SM90_64xNx32_*`) |

#### SM90 vs SM121 Feature Comparison

| Feature | SM90 (Hopper) | SM121 (Blackwell Consumer) |
|---------|---------------|----------------------------|
| **MMA Instruction** | `wgmma.mma_async` (warp-group) | `mma.sync.aligned` (traditional) |
| **MMA Tile Size** | 64×N×32 (async) | 16×8×32 (sync) |
| **TMA Support** | ✅ Full | ✅ Limited (no multicast) |
| **Cluster Support** | ✅ Up to 16 blocks | ❌ 1×1×1 only |
| **TMA Multicast** | ✅ | ❌ |
| **DSMEM** | ✅ | ❌ |

#### Could Deep GEMM Be Ported?

**Theoretically possible, but practically a complete rewrite:**

1. **WGMMA → mma.sync**: ~140 instruction references, completely different programming model
2. **Tile restructuring**: SM90 uses 64×N tiles vs SM120's 16×8 tiles
3. **Remove cluster dependencies**: ~225 references to multicast/cluster features
4. **Rebuild MMA structures**: All 16 `SM90_64xNx32_*` structs need SM120 equivalents

**Bottom line**: A "port" would be writing a new DeepGEMM-SM120 from scratch. The existing CUTLASS SM120 collectives in FlashInfer are the appropriate solution.

#### References

From [sglang's spark branch](https://github.com/sgl-project/sglang/compare/main...yvbbrjdr:sglang:spark):
```python
def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version != 90:  # Only SM90!
        return False
```

Key DeepGEMM code showing SM90 dependencies:
```cpp
// compiler.cuh:281-282 - Hardcoded architecture
std::vector<std::string> flags = {"-std=c++17",
                                  "--gpu-architecture=sm_90a",  // Cannot be changed

// fp8_gemm_impl.cuh:435 - Runtime check
DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");

// fp8_gemm.cuh:207-211 - Requires cluster multicast
// Clusters for TMA multicast
attr.val.clusterDim = {num_tma_multicast, 1, 1};  // SM121 limited to 1,1,1
```

### What is Fused MoE?

**Fused Mixture of Experts** combines multiple operations into a single kernel:

```
Traditional MoE (multiple kernel launches):
  Gate → TopK → Permute → GEMM1 → Activation → GEMM2 → Unpermute

Fused MoE (single kernel):
  [Gate + TopK + Permute + GEMM1 + Activation + GEMM2 + Unpermute]
```

**Advantages:**
- Reduced kernel launch overhead
- Better memory locality
- Higher GPU utilization
- Critical for MoE models like GPT-OSS-120B

### CUTLASS vs cuDNN for SM121

| Aspect | CUTLASS | cuDNN |
|--------|---------|-------|
| **MXFP4 Support** | ❌ (NVFP4 only) | ✅ |
| **Tile Configuration** | 128×128×128 only | Flexible |
| **Cluster Shape** | 1×1×1 only | Flexible |
| **Fused MoE** | ✅ (FlashInfer) | ❌ (standalone GEMM) |
| **Performance** | Potentially higher | Good baseline |

**Recommendation:** For SM121 with MXFP4:
- Use cuDNN for standalone GEMM operations
- Use CUTLASS-based fused MoE (SM90 path in FlashInfer)

### Triton Considerations

The [sglang spark branch](https://github.com/sgl-project/sglang/compare/main...yvbbrjdr:sglang:spark) uses a custom Triton fork:

```dockerfile
git clone --branch=spark https://github.com/yvbbrjdr/triton.git
```

This may be needed for:
- SM121-specific kernel generation
- CUDA 13.x compatibility
- FP4 operation support in Triton kernels

### Flash Attention 3

FA3 is **disabled for CUDA 13+** in sglang:

```cmake
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" AND "${CUDA_VERSION}" VERSION_LESS "13.0")
    set(SGL_KERNEL_ENABLE_FA3 ON)
```

The FlashInfer attention backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`) should be used instead.

---

## Quick Start

### Build

```bash
docker build -t vllm-dgx-spark-mxfp4 .
```

### Run with Docker

```bash
# Basic run
docker run --gpus all -p 8000:8000 vllm-dgx-spark-mxfp4 \
    vllm serve openai/gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# With model cache persistence
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm-dgx-spark-mxfp4 \
    vllm serve openai/gpt-oss-120b --quantization mxfp4
```

### Run with Docker Compose

```bash
# Start service
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

### Development Mode

For active development with mounted source code:

```bash
# Build dev image
docker build -f Dockerfile.dev -t vllm-dev .

# Start dev environment
docker compose -f docker-compose.dev.yml up -d dev

# Shell into container
docker compose -f docker-compose.dev.yml exec dev bash

# Start vLLM server (from inside container)
docker compose -f docker-compose.dev.yml --profile serve up
```

---

## Environment Variables

### MXFP4 Backend Selection

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | **Critical:** Must include `a` suffix for hardware FP4 |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | `1` | Enable MXFP4 BF16 backend (CUTLASS) |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `0` | Disable MXFP8 TRTLLM (not SM121) |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS` | `0` | Disable MXFP8 CUTLASS |
| `VLLM_FLASHINFER_MOE_BACKEND` | `throughput` | Use throughput-optimized backend |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | Enable FP4 MOE |

### Performance Tuning

| Variable | Value | Description |
|----------|-------|-------------|
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | Use FlashInfer attention |
| `VLLM_USE_CUDA_GRAPH` | `1` | Enable CUDA graphs (~60% speedup) |
| `FLASHINFER_NVCC_THREADS` | `4` | Parallel JIT compilation |

### Debugging

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASHINFER_LOGLEVEL` | `0-5` | API logging (0=off, 3=detailed) |
| `FLASHINFER_JIT_VERBOSE` | `0-1` | JIT compilation logging |

---

## Validation

The container runs validation on startup. You can also validate manually:

```bash
docker run --gpus all vllm-dgx-spark-mxfp4 python -c "
from vllm.platforms import current_platform
import torch

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
print(f'is_blackwell_class(): {current_platform.is_blackwell_class()}')
"
```

Expected output for DGX Spark:
```
GPU: NVIDIA GB10
Compute Capability: (12, 1)
is_blackwell_class(): True
```

---

## API Usage

Once running, the vLLM OpenAI-compatible API is available at `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Troubleshooting

### `cvt.e2m1x2.f32 not supported on .target 'sm_121'`

**Root Cause:** Missing `a` suffix in architecture specification.

**Fix:**
1. Ensure `FLASHINFER_CUDA_ARCH_LIST="12.1a"` (with the `a` suffix)
2. Clear FlashInfer cache: `rm -rf ~/.cache/flashinfer/` (or `./.cache/flashinfer/` if using volume mounts)
3. Rebuild/restart container

### MXFP4 not using FlashInfer

If you see "Using Marlin backend" instead of FlashInfer:

1. Verify SM121 is detected as Blackwell-class:
   ```bash
   python -c "from vllm.platforms import current_platform; print(current_platform.is_blackwell_class())"
   ```

2. Check that PR #31740 is applied:
   ```bash
   cd /workspace/vllm && git log --oneline -3
   ```

3. Verify cuDNN version is >= 9.14:
   ```bash
   python -c "import cudnn; print(cudnn.backend_version())"
   ```

### JIT compilation errors

Clear FlashInfer cache and retry:
```bash
rm -rf ~/.cache/flashinfer/
# Or if using volume mounts:
sudo rm -rf ./.cache/flashinfer/
```

### Slow builds on DGX Spark

DGX Spark uses ARM CPUs which are slower for C++/CUDA compilation:

- First build may take 30-60+ minutes
- Use ccache (configured in Dockerfile)
- Limit parallel jobs: `BUILD_JOBS=2` or `BUILD_JOBS=4`
- Subsequent builds will be faster due to caching

### Out of memory

- Reduce `--max-model-len`
- Reduce `--max-num-seqs`
- Lower `--gpu-memory-utilization` (default 0.70)
- Enable tensor parallelism if multiple GPUs available

---

## Lessons from sglang Spark Branch

The [sglang spark branch](https://github.com/sgl-project/sglang/compare/main...yvbbrjdr:sglang:spark) provides valuable insights:

1. **MXFP4 FlashInfer backend was disabled** - indicates ongoing issues
2. **Deep GEMM disabled** for SM121 (only SM90)
3. **CUTLASS FP8 disabled** entirely
4. **Custom Triton fork** required for SM121
5. **FA3 disabled** for CUDA 13+
6. **FP8 for attention** layers even when using MXFP4 for MoE:
   ```python
   self.qkv_proj = QKVParallelLinear(..., quant_config=Fp8Config())
   self.o_proj = RowParallelLinear(..., quant_config=Fp8Config())
   self.lm_head = ParallelLMHead(..., quant_config=Fp8Config())
   ```

---

## Technical References

### PR #31740 Changes

The PR adds `is_blackwell_class()` which groups SM10x, SM11x, SM12x as Blackwell-class:

```python
def is_blackwell_class(cls, device_id: int = 0) -> bool:
    capability = cls.get_device_capability(device_id=device_id)
    return capability.major in (10, 11, 12)
```

### Hardware FP4 Path

The `__CUDA_ARCH_FAMILY_SPECIFIC__` macro enables hardware FP4 conversion:

| Architecture Flag | Macro Defined | FP4 Path |
|-------------------|---------------|----------|
| `-arch=sm_121` | ❌ | Software emulation |
| `-arch=sm_121a` | ✅ (1210) | Hardware (`cvt.e2m1x2`) |
| `-arch=sm_121f` | ✅ (1210) | Hardware (`cvt.e2m1x2`) |

### SM121 CUTLASS Limitations & Optimization Opportunities

**Current FlashInfer SM120/121 Configuration:**
- Tile: 128×128×128 only
- Cluster: 1×1×1 only (hardware limitation for GeForce-class)
- Schedule: Cooperative only

**What CUTLASS Officially Supports for SM120:**

| Configuration | Tile Shape (MxNxK) | Schedule | Use Case |
|---------------|-------------------|----------|----------|
| ✅ Cooperative | 128×128×128 | Cooperative | General GEMM |
| ❌ **PingPong** | **64×128×128** | PingPong | Small-batch MoE |
| ❌ **Sparse** | **128×128×256** | WarpSpecialized | Large-K GEMM |

**FlashInfer TODO** (from `gemm_groupwise_sm120.cuh`):
```cpp
// TODO (yongwww): add PingPong schedule (64x128x128)
```

### Optimal Shapes for GPT-OSS-120B

**Model Dimensions:**
- `hidden_size`: 2880
- `intermediate_size`: 2880
- `num_experts`: 128
- `experts_per_token`: 4 (Top-4 routing)

**GEMM Analysis:**

| Operation | M | N | K | Optimal Tile |
|-----------|---|---|---|--------------|
| MoE Gate | batch | 128 | 2880 | 64×128×128 (small M) |
| MoE Up | ~batch/4 | 2880 | 2880 | 64×128×128 (small M) |
| MoE Down | ~batch/4 | 2880 | 2880 | 64×128×128 (small M) |
| QKV Proj | batch | 2880×3 | 2880 | 128×128×128 |

**Recommendations for FlashInfer SM121 Performance:**

1. **Add PingPong Schedule (64×128×128)** - High Priority
   - Critical for MoE where tokens-per-expert is small
   - With Top-4 routing and batch=64: only ~16 tokens per expert
   - 64×M tiles better match this workload

2. **Add K=256 Support (128×128×256)** - Medium Priority
   - Reduces memory traffic for large K dimensions
   - 2880 % 256 = 64 (requires padding)
   - 2880 % 128 = 64 (same padding needed)
   - Larger K tiles still beneficial for memory bandwidth

3. **Cluster Shape: 1×1×1** - Hardware Limitation
   - This is a GeForce-class constraint, not a software limitation
   - Cannot be optimized without SM100+ hardware

---

## Adding MXFP4 Support to FlashInfer for SM120/SM121

### Current State

| Format | cuDNN | CUTLASS Standalone | CUTLASS Fused MoE |
|--------|-------|-------------------|-------------------|
| **NVFP4** | ✅ | ✅ | ✅ |
| **MXFP4** | ✅ | ❌ | ❌ |

**Problem**: GPT-OSS-120B uses MXFP4 quantization, but FlashInfer's CUTLASS kernels for SM120 only support NVFP4.

### MXFP4 vs NVFP4 Differences

| Property | NVFP4 | MXFP4 |
|----------|-------|-------|
| Scale format | FP8 E4M3 | FP8 E8M0 |
| Block size | 16 elements | 32 elements |
| Scale granularity | Per-16 | Per-32 |

### Implementation Plan

FlashInfer already has MXFP4 support for SM100. The plan is to port this to SM120:

#### Step 1: Create SM120 MXFP4 Groupwise GEMM

**Reference implementation**: `include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm100.cuh`

**New file**: `include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm120.cuh`

Key adaptations:
- Tile shape: 128×128×128 (fixed for SM120)
- Cluster shape: 1×1×1 (hardware limitation)
- MMA traits: `SM120_16x8x32_TN` instead of SM100 variants
- Scale granularity: 128 for M, N, K dimensions

#### Step 2: Update FP4 GEMM Dispatcher

**File**: `include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`

Add:
```cpp
template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchMXFP4xMXFP4GemmClusterShapeSm120(...);

// In dispatchFP4GemmSm120:
if constexpr (fp4GemmType == FP4GemmType::W4A4_MXFP4_MXFP4) {
    return dispatchMXFP4xMXFP4GemmCTAShapeSm120<T>(...);
}
```

#### Step 3: Remove SM120 MXFP4 Restrictions

**File**: `flashinfer/gemm/gemm_base.py`

Update `_cutlass_gemm_fp4_requirement`:
```python
@supported_compute_capability([100, 103, 110, 120, 121])
def _cutlass_gemm_fp4_requirement(...):
    # Remove: if not use_nvfp4: raise ValueError(...)
    return True
```

**File**: `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp`

Remove lines 618 and 631:
```cpp
// Remove: TLLM_THROW("Not Implemented: SM120 GEMM only supports nvfp4.");
// Remove: TLLM_THROW("Not Implemented: SM120 group GEMM only supports nvfp4.");
```

#### Step 4: Add Fused MoE MXFP4 Support

**Reference**: TRT-LLM fused MoE for SM100 with MXFP4

**New file**: `csrc/cutlass_fused_moe_sm120.cu`

Adapt the SM100 fused MoE kernel with:
- SM120 tile constraints (128×128×128)
- SM120 cluster constraints (1×1×1)
- E8M0 scale format handling

### Optimal Tile Configurations for SM120

Based on CUTLASS examples and GPT-OSS-120B dimensions:

| Schedule | Tile (M×N×K) | Use Case | Status |
|----------|--------------|----------|--------|
| Cooperative | 128×128×128 | General GEMM | ✅ Implemented |
| PingPong | 64×128×128 | Small-batch MoE | ❌ TODO |
| Sparse | 128×128×256 | Large-K GEMM | ❌ TODO |

### GPT-OSS-120B Specific Optimizations

**Model dimensions**:
- `hidden_size`: 2880
- `intermediate_size`: 2880  
- `num_experts`: 128
- `experts_per_token`: 4

**Recommended tile for MoE**:
- **64×128×128 (PingPong)** - With Top-4 routing, each expert sees ~batch/4 tokens
- For batch=64: ~16 tokens per expert → 64×M tiles are more efficient

### Estimated Development Effort

| Task | Effort | Priority |
|------|--------|----------|
| Port MXFP4 standalone GEMM (SM100→SM120) | 2 days | High |
| Port MXFP4 fused MoE (SM100→SM120) | 3 days | High |
| Add PingPong schedule (64×128×128) | 2 days | Medium |
| Add K=256 tile support | 1 day | Low |
| Testing & validation | 2-3 days | High |
| **Total** | **~10-12 days** | |

### Files to Modify/Create

| Action | File |
|--------|------|
| Create | `include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm120.cuh` |
| Create | `include/flashinfer/gemm/fp4_gemm_mxfp4_template_sm120.h` |
| Create | `csrc/cutlass_fused_moe_mxfp4_sm120.cu` |
| Modify | `include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h` |
| Modify | `flashinfer/gemm/gemm_base.py` |
| Modify | `csrc/nv_internal/.../cutlass_heuristic.cpp` |
| Modify | `flashinfer/jit/gemm/*.py` (JIT module generators) |

---

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Production image with pre-built vLLM |
| `Dockerfile.dev` | Development image with mounted source |
| `docker-compose.yml` | Production deployment |
| `docker-compose.dev.yml` | Development environment |
| `fastsafetensors.patch` | Patch for faster distributed model loading |
| `mxfp4_sm121.patch` | Route SM121 to CUTLASS backend (not TRT-LLM) |
| `decorators_ngc.patch` | Fix NGC PyTorch `assume_32bit_indexing` compatibility |

---

## Development Considerations

### FlashInfer JIT Cache Management

FlashInfer uses JIT compilation with a disk cache at `~/.cache/flashinfer/`. The cache is organized by:
```
~/.cache/flashinfer/{version}/{arch}/cached_ops/{module_name}/
```

**Example structure:**
```
~/.cache/flashinfer/0.5.3/121a/cached_ops/
├── fused_moe_120/          # SM120 fused MoE kernels
├── fused_moe_90/           # SM90 fused MoE kernels  
├── batch_decode_*/         # Decode attention kernels
├── batch_prefill_*/        # Prefill attention kernels
├── fp4_gemm_*/             # FP4 GEMM kernels
└── ...
```

### Selective Cache Clearing (Faster Testing)

**Don't delete the entire cache!** Only clear the specific module you modified:

```bash
# Only clear fused MoE SM120 kernels (what we're working on)
rm -rf /root/.cache/flashinfer/*/cached_ops/fused_moe_120/

# Clear all fused MoE kernels (if modifying shared MoE code)
rm -rf /root/.cache/flashinfer/*/cached_ops/fused_moe_*/

# Clear FP4 GEMM kernels
rm -rf /root/.cache/flashinfer/*/cached_ops/fp4_gemm_*/

# Nuclear option - clear everything (SLOW - forces full recompilation)
rm -rf /root/.cache/flashinfer/
```

### Module to File Mapping

| Module Name | Affected By Changes To |
|-------------|------------------------|
| `fused_moe_120` | `flashinfer/jit/fused_moe.py`, `moe_gemm_*.h/.cpp`, `moe_tma_warp_specialized_traits.h` |
| `fused_moe_90` | Same as above (SM90 variant) |
| `fp4_gemm_*` | `flashinfer/jit/gemm/core.py`, `fp4_gemm_*.cu/.cuh` |
| `batch_decode_*` | Attention kernel files |
| `batch_prefill_*` | Attention kernel files |

### Compilation Parallelism and OOM

CUTLASS kernel instantiations are **extremely memory-hungry** - each can consume 5-10+ GB of RAM during compilation. With high parallelism (`MAX_JOBS=16`), peak memory usage can exceed 100GB, causing the Linux OOM killer to terminate `cicc` (CUDA compiler) processes.

**Symptom:** `Killed` in build output, ninja exits with code 255.

**Example error:**
```
/bin/bash: line 1: 80034 Killed  "$CICC_PATH/cicc" --c++17 ...
```

**Solution - Reduce parallelism:**

```bash
# Before running vLLM, reduce parallel compilation jobs
export MAX_JOBS=4      # Default is 16, which can cause OOM

# Then start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --quantization mxfp4 ...
```

**Memory guidelines:**
| MAX_JOBS | Peak RAM Usage (approx) | Recommended For |
|----------|-------------------------|-----------------|
| 16 | ~100-160GB | 256GB+ systems |
| 8 | ~50-80GB | 128GB+ systems |
| 4 | ~25-40GB | 64GB+ systems |
| 2 | ~12-20GB | 32GB+ systems |

**Note:** First compilation is always slow. Subsequent runs use cached `.so` files and don't need recompilation.

### Debugging JIT Compilation

```bash
# Enable verbose JIT logging
export FLASHINFER_JIT_VERBOSE=1

# Enable debug builds (slower but better error messages)
export FLASHINFER_JIT_DEBUG=1

# Check build logs for a specific module
cat /root/.cache/flashinfer/*/cached_ops/fused_moe_120/*.log

# Watch ninja build in real-time
tail -f /root/.cache/flashinfer/*/cached_ops/fused_moe_120/*.log
```

### Architecture-Specific Compilation

The `FLASHINFER_CUDA_ARCH_LIST` environment variable controls target architectures:

```bash
# SM121 with architecture-specific features (hardware FP4)
export FLASHINFER_CUDA_ARCH_LIST="12.1a"

# Multiple architectures (for broader compatibility)
export FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a 12.1a"
```

**Important:** The `a` suffix enables architecture-specific features. Without it, some hardware paths (like FP4 conversion) fall back to software emulation.

