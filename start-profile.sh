#!/bin/bash
# =============================================================================
# Start vLLM under nsys profiling with precise capture control
# Profiles only during actual inference, not startup
# =============================================================================

set -e

# Configuration
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-/tmp/nsys_profile/vllm_inference}"
SESSION_NAME="vllm_profile"

echo "=============================================="
echo "vLLM MXFP4 Profiler with Precise Capture"
echo "=============================================="
echo "Model: $MODEL"
echo "Endpoint: http://$HOST:$PORT"
echo "Profile output: $PROFILE_OUTPUT"
echo "Benchmark: llama-benchy --pp 2048 --tg 32 128"
echo "NVTX: --enable-layerwise-nvtx-tracing"
echo "=============================================="

# Create script to run inside container
docker exec -it vllm-dev bash -c "
set -e
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
mkdir -p /tmp/nsys_profile

echo ''
echo '=== Phase 1: Launching vLLM under nsys (capture paused) ==='
echo ''

# Launch vLLM under nsys with session name and trace options
nsys launch \\
    --session-new=${SESSION_NAME} \\
    --cuda-memory-usage=true \\
    -- python -m vllm.entrypoints.openai.api_server \\
        --model $MODEL \\
        --host $HOST \\
        --port $PORT \\
        --served-model-name gpt-oss-120b \\
        --quantization mxfp4 \\
        --mxfp4-backend CUTLASS \\
        --mxfp4-layers moe,qkv,o,lm_head \\
        --attention-backend FLASHINFER \\
        --kv-cache-dtype fp8 \\
        --tensor-parallel-size 1 \\
        --gpu-memory-utilization 0.70 \\
        --max-model-len 131072 \\
        --max-num-seqs 2 \\
        --max-num-batched-tokens 8192 \\
        --enable-prefix-caching \\
        --load-format fastsafetensors \\
        --enable-layerwise-nvtx-tracing &

VLLM_PID=\$!
echo \"vLLM PID: \$VLLM_PID\"

echo ''
echo '=== Phase 2: Waiting for server to be ready ==='
echo ''

# Poll for server health
MAX_WAIT=600  # 10 minutes max
WAITED=0
while [ \$WAITED -lt \$MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo \"Server ready after \${WAITED}s!\"
        break
    fi
    sleep 5
    WAITED=\$((WAITED + 5))
    echo \"  Waiting... (\${WAITED}s)\"
done

if [ \$WAITED -ge \$MAX_WAIT ]; then
    echo \"ERROR: Server did not become ready within \${MAX_WAIT}s\"
    kill \$VLLM_PID 2>/dev/null || true
    exit 1
fi

# Give server a moment to stabilize
sleep 2

echo ''
echo '=== Phase 3: Warmup request (not profiled) ==='
echo ''

# Send a warmup request to ensure CUDA graphs are captured, JIT is done, caches are hot
echo 'Sending warmup request...'
curl -s http://localhost:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{\"model\": \"gpt-oss-120b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 16}' \\
    > /dev/null
echo 'Warmup complete'

sleep 1

echo ''
echo '=== Phase 4: Starting profiling capture ==='
echo ''

# Start capture with output file (after warmup, so profile is clean)
nsys start \\
    --session=${SESSION_NAME} \\
    --output=${PROFILE_OUTPUT} \\
    --force-overwrite=true

echo \"Profiling started!\"

# Immediately send a request to fill the gap while llama-benchy starts
curl -s http://localhost:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{\"model\": \"gpt-oss-120b\", \"messages\": [{\"role\": \"user\", \"content\": \"Quick test\"}], \"max_tokens\": 8}' \\
    > /dev/null &

echo ''
echo '=== Phase 5: Running benchmark (llama-benchy) ==='
echo ''

# Run the exact benchmark we use for performance testing
# --pp 2048: 2048 token prefill
# --tg 32 128: decode with 32 and 128 output tokens
llama-benchy \\
    --base-url http://localhost:$PORT/v1 \\
    --model gpt-oss-120b \\
    --tokenizer openai/gpt-oss-120b \\
    --pp 2048 \\
    --tg 32 128

echo \"Benchmark completed\"

echo ''
echo '=== Phase 6: Stopping capture and saving ==='
echo ''

# Stop capture and save
nsys stop --session=${SESSION_NAME}

echo ''
echo '=============================================='
echo \"Profile saved to: ${PROFILE_OUTPUT}.nsys-rep\"
echo ''
echo 'Analyze with:'
echo \"  ./analyze-profile.sh ${PROFILE_OUTPUT}.nsys-rep\"
echo '=============================================='

# Cleanup
kill \$VLLM_PID 2>/dev/null || true
"
