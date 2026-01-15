# =============================================================================
# Dockerfile for GPT-OSS-120B with MXFP4 Quantization on DGX Spark (SM121/GB10)
# =============================================================================
#
# This Dockerfile creates an environment for running GPT-OSS-120B with native
# MXFP4 quantization on NVIDIA DGX Spark (SM121/GB10 Blackwell-class GPU).
#
# Key components:
# - Base: NGC PyTorch container with CUDA 13.0 and cuDNN 9.15+
# - vLLM with PR #31740 for SM121 Blackwell-class support
# - FlashInfer with cuDNN backend for MXFP4 GEMM operations
#
# Build:
#   docker build -t vllm-dgx-spark-mxfp4 .
#
# Run:
#   docker run --gpus all -p 8000:8000 vllm-dgx-spark-mxfp4 \
#       vllm serve openai/gpt-oss-120b --quantization mxfp4
#
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL maintainer="DGX Spark MXFP4 Setup"
LABEL description="vLLM with MXFP4 support for DGX Spark (SM121/GB10)"

# =============================================================================
# Build arguments
# =============================================================================

# Build parallelism - adjust based on available RAM (lower if OOM during build)
ARG BUILD_JOBS=16

# vLLM reference: can be a branch, tag, commit SHA, or PR reference
ARG VLLM_REF=pr-31740
ARG VLLM_PR_NUMBER=31740

# Workspace directory
ARG VLLM_BASE_DIR=/workspace

# =============================================================================
# Build parallelism controls
# =============================================================================

ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# =============================================================================
# Environment setup
# =============================================================================

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# UV package manager settings (faster than pip)
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

# =============================================================================
# Ccache configuration for incremental C++/CUDA compilation
# =============================================================================

ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# =============================================================================
# System dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# =============================================================================
# FlashInfer installation
# =============================================================================

WORKDIR ${VLLM_BASE_DIR}

# Install FlashInfer with SM121 architecture target
# JIT compilation will generate kernels for SM121 (DGX Spark)
ENV FLASHINFER_CUDA_ARCH_LIST="12.1f"

# Install FlashInfer using UV with cache mount
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install flashinfer-python

# FlashInfer environment configuration
ENV FLASHINFER_JIT_VERBOSE=0
ENV FLASHINFER_LOGLEVEL=0

# =============================================================================
# vLLM installation with SM121 support (PR #31740)
# Uses smart git clone pattern to avoid re-cloning on rebuilds
# =============================================================================

# =============================================================================
# Step 1: Clone/fetch vLLM (cached separately)
# =============================================================================
RUN --mount=type=cache,id=repo-cache-vllm,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "=== Cache miss: Cloning vLLM from scratch ===" && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
    else \
        echo "=== Cache hit: Fetching vLLM updates ===" && \
        cd vllm && \
        git checkout main 2>/dev/null || git checkout -b main origin/main && \
        git fetch --all && \
        git submodule update --init --recursive && \
        git gc --auto && \
        cd ..; \
    fi && \
    cd vllm && \
    echo "=== Fetching PR #${VLLM_PR_NUMBER} ===" && \
    git fetch origin pull/${VLLM_PR_NUMBER}/head:${VLLM_REF} --force && \
    echo "=== Checking out ${VLLM_REF} ===" && \
    git checkout ${VLLM_REF} && \
    if [ "${VLLM_REF}" = "main" ]; then \
        git reset --hard origin/main; \
    fi && \
    cd .. && \
    rm -rf ${VLLM_BASE_DIR}/vllm && \
    cp -a /repo-cache/vllm ${VLLM_BASE_DIR}/

WORKDIR ${VLLM_BASE_DIR}/vllm

# =============================================================================
# Step 2: Prepare build (modifies requirements to use existing torch)
# =============================================================================
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt 2>/dev/null || true

# =============================================================================
# Step 3: Install build dependencies (cached, rarely changes)
# =============================================================================
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

# =============================================================================
# Step 4: Install vLLM dependencies ONLY (without building vLLM itself)
# This layer is cached and only rebuilds when requirements change
# =============================================================================
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install -r requirements/cuda.txt -r requirements/common.txt 2>/dev/null || \
    uv pip install -r requirements.txt 2>/dev/null || true

# =============================================================================
# Step 5: Build and install vLLM (uses ccache for C++ compilation)
# =============================================================================
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    echo "=== ccache stats before build ===" && \
    ccache -s && \
    uv pip install --no-build-isolation --no-deps -e . && \
    echo "=== ccache stats after build ===" && \
    ccache -s

# =============================================================================
# Step 6: Install fastsafetensors and apply patches
# =============================================================================
COPY fastsafetensors.patch ${VLLM_BASE_DIR}/vllm/
COPY mxfp4_sm121.patch ${VLLM_BASE_DIR}/vllm/
COPY decorators_ngc.patch ${VLLM_BASE_DIR}/vllm/
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install fastsafetensors && \
    cd ${VLLM_BASE_DIR}/vllm && \
    echo "=== Applying fastsafetensors patch ===" && \
    patch -p1 < fastsafetensors.patch && \
    echo "=== Applying SM121 MXFP4 backend patch ===" && \
    patch -p1 < mxfp4_sm121.patch && \
    echo "=== Applying NGC PyTorch compatibility patch ===" && \
    patch -p1 < decorators_ngc.patch && \
    rm fastsafetensors.patch mxfp4_sm121.patch decorators_ngc.patch

# =============================================================================
# Build-time Configuration for SM121
# =============================================================================

# FlashInfer JIT compilation settings
ENV FLASHINFER_NVCC_THREADS=4

# Target architecture for runtime
ENV TORCH_CUDA_ARCH_LIST="12.1"

# NOTE: vLLM runtime configuration (VLLM_*, attention backends, MoE kernels)
# should be set at runtime via docker-compose.yml or command line.
# This allows flexibility in testing different configurations.
#
# Example runtime configuration (set in docker-compose.yml or at startup):
#   VLLM_MXFP4_MOE_KERNEL=marlin       # MoE kernel: auto, marlin, gemm, gemv, triton
#   VLLM_ATTENTION_SINKS=false         # Sink control: auto, true, false
#   --attention-config '{"backend": "TRITON_ATTN"}'  # Attention backend

# =============================================================================
# Tiktoken encodings (for tokenizer support)
# Uses cache mount to avoid re-downloading on each build
# =============================================================================

RUN --mount=type=cache,id=tiktoken-cache,target=/tiktoken-cache \
    mkdir -p /workspace/tiktoken_encodings && \
    if [ ! -f "/tiktoken-cache/o200k_base.tiktoken" ]; then \
        echo "=== Cache miss: Downloading tiktoken encodings ===" && \
        wget -q -O /tiktoken-cache/o200k_base.tiktoken \
            "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
        wget -q -O /tiktoken-cache/cl100k_base.tiktoken \
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"; \
    else \
        echo "=== Cache hit: Using cached tiktoken encodings ==="; \
    fi && \
    cp /tiktoken-cache/o200k_base.tiktoken /workspace/tiktoken_encodings/ && \
    cp /tiktoken-cache/cl100k_base.tiktoken /workspace/tiktoken_encodings/


# =============================================================================
# llama-benchy installation (benchmarking tool)
# =============================================================================

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install llama-benchy

# =============================================================================
# Model cache configuration
# =============================================================================

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE}

# =============================================================================
# Entrypoint and healthcheck
# =============================================================================

WORKDIR /workspace

# Create an entrypoint script for warmup and validation
RUN cat > /workspace/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "DGX Spark MXFP4 Environment Validation"
echo "=============================================="

# Validate GPU
echo ""
echo "GPU Information:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}')"
python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'  Compute Capability: SM{cc[0]}{cc[1]} (expected: SM121)')"

# Validate CUDA/cuDNN versions
echo ""
echo "Software Versions:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
python -c "import cudnn; print(f'  cuDNN backend: {cudnn.backend_version()}')" 2>/dev/null || echo "  cuDNN: (cudnn module not available, using torch.backends)"

# Validate FlashInfer
echo ""
echo "FlashInfer:"
python -c "import flashinfer; print(f'  Version: {flashinfer.__version__}')" 2>/dev/null || echo "  FlashInfer: (version not available)"

# Validate vLLM
echo ""
echo "vLLM:"
python -c "import vllm; print(f'  Version: {vllm.__version__}')" 2>/dev/null || echo "  vLLM: (version not available)"

# Show vLLM git ref
echo ""
echo "vLLM Git Reference:"
cd /workspace/vllm && git log --oneline -1 2>/dev/null || echo "  (git info not available)"

# Check is_blackwell_class
echo ""
echo "Blackwell-class Detection:"
python -c "
from vllm.platforms import current_platform
is_blackwell = current_platform.is_blackwell_class()
print(f'  is_blackwell_class(): {is_blackwell}')
if is_blackwell:
    print('  ✓ SM121 detected as Blackwell-class - MXFP4 FlashInfer backends enabled')
else:
    print('  ✗ Warning: SM121 not detected as Blackwell-class')
"

# Show ccache stats
echo ""
echo "Ccache Stats:"
ccache -s 2>/dev/null | head -5 || echo "  (ccache stats not available)"

echo ""
echo "=============================================="
echo "Environment ready. Executing command..."
echo "=============================================="
echo ""

exec "$@"
EOF

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]

CMD ["vllm", "serve", "--help"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
