# =============================================================================
# Dockerfile for GPT-OSS-120B with MXFP4 on DGX Spark (SM121/GB10)
# =============================================================================
#
# Optimized MXFP4 inference achieving 59.4 tok/s decode on GB10.
#
# Build:
#   docker build -t vllm-mxfp4-spark .
#
# Run with docker-compose (recommended):
#   docker compose up -d
#
# Run standalone:
#   docker run --gpus all -p 8000:8000 \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       vllm-mxfp4-spark \
#       vllm serve openai/gpt-oss-120b --quantization mxfp4
#
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL maintainer="MXFP4 Optimization Project"
LABEL description="vLLM with optimized MXFP4 for DGX Spark (SM121/GB10)"

# =============================================================================
# Pinned versions - our tested configurations
# =============================================================================

ARG VLLM_SHA=045293d82b832229560ac4a13152a095af603b6e
ARG FLASHINFER_SHA=1660ee8d740b0385f235519f9e2750db944d1838
ARG CUTLASS_SHA=11af7f02ab52c9130e422eeb4b44042fbd60c083

ARG VLLM_REPO=https://github.com/christopherowen/vllm.git
ARG FLASHINFER_REPO=https://github.com/christopherowen/flashinfer.git
ARG CUTLASS_REPO=https://github.com/christopherowen/cutlass.git

# Build parallelism
ARG BUILD_JOBS=16

# =============================================================================
# Environment configuration
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# UV package manager
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

# Ccache for faster rebuilds
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# FlashInfer settings
ENV FLASHINFER_CUDA_ARCH_LIST="12.1f"
ENV FLASHINFER_JIT_VERBOSE=0
ENV FLASHINFER_LOGLEVEL=0
ENV FLASHINFER_NVCC_THREADS=4

# CUDA architecture - SM120/SM121 only (DGX Spark)
# This avoids compiling for SM80/SM90 which wastes build time
# TORCH_CUDA_ARCH_LIST: Used by PyTorch/vLLM extension builds
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"

# Model cache (HF_HOME is the modern unified path)
ENV HF_HOME=/root/.cache/huggingface

# Use local repos
ENV PYTHONPATH=/workspace/flashinfer:/workspace/vllm

WORKDIR /workspace

# =============================================================================
# System dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv \
    && pip uninstall -y flash-attn 2>/dev/null || true  # Remove NGC flash-attn to avoid operator conflicts

# =============================================================================
# Clone repositories at pinned versions
# =============================================================================

# Clone FlashInfer
RUN git clone ${FLASHINFER_REPO} /workspace/flashinfer && \
    cd /workspace/flashinfer && \
    git checkout ${FLASHINFER_SHA}

# Update CUTLASS submodule to our fork
RUN cd /workspace/flashinfer && \
    git submodule update --init --recursive && \
    cd 3rdparty/cutlass && \
    git remote set-url origin ${CUTLASS_REPO} && \
    git fetch origin && \
    git checkout ${CUTLASS_SHA}

# Clone vLLM
RUN git clone ${VLLM_REPO} /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout ${VLLM_SHA} && \
    git submodule update --init --recursive

# =============================================================================
# Build FlashInfer
# =============================================================================

WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    uv pip install --no-build-isolation -e .

# =============================================================================
# Build vLLM
# =============================================================================

WORKDIR /workspace/vllm

# Prepare build (use existing torch from NGC container)
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt 2>/dev/null || true

# Install build dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

# Install runtime dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/cuda.txt -r requirements/common.txt 2>/dev/null || \
    uv pip install -r requirements.txt 2>/dev/null || true

# Build and install vLLM
# Note: This step compiles CUDA kernels and takes 20-40 minutes on first build.
# Ccache speeds up subsequent rebuilds significantly.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=vllm-build,target=/workspace/vllm/build \
    uv pip install --no-build-isolation --no-deps -e .

# Install additional tools
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install fastsafetensors llama-benchy

# =============================================================================
# Download tiktoken encodings
# =============================================================================

RUN mkdir -p /workspace/tiktoken_encodings && \
    wget -q -O /workspace/tiktoken_encodings/o200k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -q -O /workspace/tiktoken_encodings/cl100k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# =============================================================================
# Create entrypoint script (validation only)
# =============================================================================

RUN cat > /workspace/entrypoint.sh << 'ENTRYPOINT_SCRIPT'
#!/bin/bash
set -e

echo "=============================================="
echo "MXFP4 vLLM for DGX Spark (SM121/GB10)"
echo "=============================================="
echo ""
echo "Git SHAs:"
echo "  vLLM:       $(cd /workspace/vllm && git rev-parse --short HEAD)"
echo "  FlashInfer: $(cd /workspace/flashinfer && git rev-parse --short HEAD)"
echo "  CUTLASS:    $(cd /workspace/flashinfer/3rdparty/cutlass && git rev-parse --short HEAD)"
echo ""

# Validate GPU
python3 -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability()
print(f'Compute Capability: SM{cc[0]}{cc[1]}')
"

echo ""
echo "=============================================="
echo ""

exec "$@"
ENTRYPOINT_SCRIPT

RUN chmod +x /workspace/entrypoint.sh

# =============================================================================
# Runtime configuration
# =============================================================================

WORKDIR /workspace

RUN mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
