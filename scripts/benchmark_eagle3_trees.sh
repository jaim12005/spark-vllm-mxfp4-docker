#!/bin/bash
# Eagle3 Speculative Decoding Tree Structure & K-Sweep Benchmark

set -e

export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export VLLM_MXFP4_MOE_KERNEL=marlin

PROMPT="Write a detailed explanation of how neural networks learn through backpropagation."
MAX_TOKENS=128
PORT=8000

# Results file
RESULTS_FILE="/tmp/eagle3_tree_benchmark.md"

echo "# Eagle3 Tree Structure Benchmark" > $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "## Results" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "| Model | K | Tree | Tokens | Time(s) | tok/s | Accept Rate |" >> $RESULTS_FILE
echo "|-------|---|------|--------|---------|-------|-------------|" >> $RESULTS_FILE

run_test() {
    local MODEL_NAME=$1
    local DRAFT_MODEL=$2
    local K=$3
    local TREE=$4
    local TREE_DESC=$5
    
    echo ""
    echo "=============================================="
    echo "Testing: $MODEL_NAME, K=$K, Tree=$TREE_DESC"
    echo "=============================================="
    
    # Kill any existing server
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
    
    # Start server
    vllm serve openai/gpt-oss-120b \
        --host 0.0.0.0 \
        --port $PORT \
        --served-model-name gpt-oss-120b \
        --quantization mxfp4 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.70 \
        --max-model-len 131072 \
        --max-num-seqs 2 \
        --max-num-batched-tokens 8192 \
        --enforce-eager \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --attention-config '{"backend": "TRITON_ATTN"}' \
        --speculative-config "{
            \"method\": \"eagle3\",
            \"model\": \"$DRAFT_MODEL\",
            \"num_speculative_tokens\": $K,
            \"speculative_token_tree\": \"$TREE\"
        }" &
    
    SERVER_PID=$!
    
    # Wait for server
    echo "Waiting for server..."
    for i in {1..120}; do
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "ERROR: Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
    
    # Warmup
    curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"gpt-oss-120b\", \"prompt\": \"Hello\", \"max_tokens\": 10}" > /dev/null
    
    # Benchmark with Python for calculations
    python3 << EOF
import requests
import time
import json

start = time.time()
response = requests.post(
    "http://localhost:$PORT/v1/completions",
    headers={"Content-Type": "application/json"},
    json={"model": "gpt-oss-120b", "prompt": "$PROMPT", "max_tokens": $MAX_TOKENS}
)
elapsed = time.time() - start

data = response.json()
tokens = data["usage"]["completion_tokens"]
throughput = tokens / elapsed
output = data["choices"][0]["text"][:80].replace("\n", " ")

print(f"Tokens: {tokens}, Time: {elapsed:.2f}s, Throughput: {throughput:.1f} tok/s")
print(f"Output: {output}...")

# Append to results file
with open("$RESULTS_FILE", "a") as f:
    f.write(f"| $MODEL_NAME | $K | $TREE_DESC | {tokens} | {elapsed:.2f} | {throughput:.1f} | - |\n")
EOF
    
    # Get spec decoding metrics from logs
    sleep 2
    
    # Cleanup
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
}

# Eagle3 models
EAGLE_LONG="nvidia/gpt-oss-120b-Eagle3-long-context"
EAGLE_SHORT="nvidia/gpt-oss-120b-Eagle3-short-context"
EAGLE_THRU="nvidia/gpt-oss-120b-Eagle3-throughput"

# Tree configurations
TREE_CHAIN_1='[(0,)]'
TREE_CHAIN_2='[(0,), (0, 0)]'
TREE_CHAIN_3='[(0,), (0, 0), (0, 0, 0)]'
TREE_CHAIN_4='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]'
TREE_CHAIN_6='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)]'
TREE_BINARY='[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]'
TREE_WIDE='[(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]'

echo ""
echo "====== K-SWEEP: long-context model ======"
run_test "long-ctx" "$EAGLE_LONG" 1 "$TREE_CHAIN_1" "chain-1"
run_test "long-ctx" "$EAGLE_LONG" 2 "$TREE_CHAIN_2" "chain-2"
run_test "long-ctx" "$EAGLE_LONG" 3 "$TREE_CHAIN_3" "chain-3"
run_test "long-ctx" "$EAGLE_LONG" 4 "$TREE_CHAIN_4" "chain-4"
run_test "long-ctx" "$EAGLE_LONG" 6 "$TREE_CHAIN_6" "chain-6"

echo ""
echo "====== TREE STRUCTURES: long-context model ======"
run_test "long-ctx" "$EAGLE_LONG" 6 "$TREE_BINARY" "binary"
run_test "long-ctx" "$EAGLE_LONG" 9 "$TREE_WIDE" "wide-9"

echo ""
echo "====== MODEL COMPARISON (k=3) ======"
run_test "short-ctx" "$EAGLE_SHORT" 3 "$TREE_CHAIN_3" "chain-3"
run_test "throughput" "$EAGLE_THRU" 1 "$TREE_CHAIN_1" "chain-1"

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
cat $RESULTS_FILE
