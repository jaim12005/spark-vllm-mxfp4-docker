#!/bin/bash
# Profile vLLM decode performance with nsys
#
# Usage:
#   ./scripts/profile_decode_nsys.sh [delay_seconds] [duration_seconds]
#
# This script:
# 1. Starts vLLM server with nsys profiler attached
# 2. Waits for server to be ready
# 3. Runs decode benchmark during profiling window
# 4. Generates profile report
#
# Prerequisites:
#   - vllm-dev container running
#   - GPU memory clear (no other models loaded)

set -e

DELAY=${1:-60}      # Delay before profiling starts (server warmup)
DURATION=${2:-30}   # Profiling duration

CONTAINER="vllm-dev"
PROFILE_OUTPUT="/tmp/vllm_decode_profile"

echo "=== vLLM Decode Performance Profiler ==="
echo "Delay: ${DELAY}s, Duration: ${DURATION}s"
echo ""

# Check if server is already running
if docker exec $CONTAINER curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "WARNING: vLLM server already running. Stop it first:"
    echo "  docker exec $CONTAINER pkill -9 -f 'vllm.entrypoints'"
    exit 1
fi

# Check GPU memory
GPU_FREE=$(docker exec $CONTAINER python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'{f/1e9:.0f}')")
echo "GPU memory free: ${GPU_FREE}GB"
if [ "$GPU_FREE" -lt 100 ]; then
    echo "WARNING: Less than 100GB GPU memory free. Kill any GPU processes first."
    exit 1
fi

# Start server with nsys profiling
echo ""
echo "Starting vLLM server with nsys profiler..."
docker exec $CONTAINER bash -c "
cd /workspace/vllm && \\
PYTHONPATH=/workspace/flashinfer:/workspace/vllm:\$PYTHONPATH \\
TIKTOKEN_ENCODINGS_BASE=/workspace/tiktoken_encodings \\
VLLM_ATTENTION_BACKEND=FLASHINFER \\
VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \\
nsys profile --output=$PROFILE_OUTPUT \\
  --trace=cuda,nvtx \\
  --force-overwrite=true \\
  --delay=$DELAY \\
  --duration=$DURATION \\
  python -m vllm.entrypoints.openai.api_server \\
    --host 0.0.0.0 --port 8000 \\
    --model openai/gpt-oss-120b \\
    --max-model-len 4096 \\
    --quantization mxfp4 \\
    --load-format fastsafetensors \\
    --enforce-eager \\
    --gpu-memory-utilization 0.70 \\
  > /tmp/vllm_profile.log 2>&1 &
"

echo "Server starting (nsys will profile from ${DELAY}s to $((DELAY+DURATION))s)..."

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 90); do
    if docker exec $CONTAINER curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after ~$((i*2))s!"
        break
    fi
    sleep 2
done

# Verify server is up
if ! docker exec $CONTAINER curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start. Check logs:"
    echo "  docker exec $CONTAINER tail -50 /tmp/vllm_profile.log"
    exit 1
fi

# Run decode benchmark
echo ""
echo "Running decode benchmark during profiling window..."
docker exec $CONTAINER python3 /workspace/decode_benchmark.py --tokens 64 --runs 5

# Wait for nsys to finish
echo ""
echo "Waiting for nsys profiling to complete..."
sleep $((DURATION + 10))

# Check for profile output
if docker exec $CONTAINER ls ${PROFILE_OUTPUT}.nsys-rep > /dev/null 2>&1; then
    echo ""
    echo "=== Profile Analysis ==="
    docker exec $CONTAINER nsys stats --report cuda_gpu_kern_sum ${PROFILE_OUTPUT}.nsys-rep 2>&1 | head -60
    
    echo ""
    echo "Profile saved to: ${PROFILE_OUTPUT}.nsys-rep"
    echo "To copy to host:  docker cp ${CONTAINER}:${PROFILE_OUTPUT}.nsys-rep ."
    echo "To view in GUI:   nsys-ui ${PROFILE_OUTPUT}.nsys-rep"
else
    echo "ERROR: Profile not generated. Check server logs."
fi

