#!/bin/bash
# =============================================================================
# SM121 MXFP4 MoE GEMM + FA2 Sinks Testing Harness for vLLM
# =============================================================================
# This script provides crash-proof testing with proper process cleanup,
# backend verification, and comprehensive diagnostics.
#
# Location: ~/projects/ai/mxfp4/scripts/
#
# Usage (from host):
#   cd ~/projects/ai/mxfp4
#   ./scripts/sm121_vllm_test_harness.sh [--mode all|verify|stress|benchmark]
#
# Usage (inside Docker):
#   docker exec -it vllm-dev bash
#   cd /workspace  # or copy scripts to /workspace
#   /host/scripts/sm121_vllm_test_harness.sh --mode verify
#
# Docker Setup:
#   - vllm-dev container running (docker-compose.dev.yml)
#   - FlashInfer mounted at /workspace/flashinfer
#   - vLLM mounted at /workspace/vllm
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MXFP4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect if running inside Docker
if [[ -f /.dockerenv ]]; then
    IN_DOCKER=true
    FLASHINFER_ROOT="/workspace/flashinfer"
    VLLM_ROOT="/workspace/vllm"
    LOG_DIR="${LOG_DIR:-/workspace/test_logs}"
else
    IN_DOCKER=false
    FLASHINFER_ROOT="${FLASHINFER_ROOT:-$HOME/projects/flashinfer}"
    VLLM_ROOT="${VLLM_ROOT:-$HOME/projects/vllm}"
    LOG_DIR="${LOG_DIR:-$MXFP4_DIR/test_logs}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="$LOG_DIR/run_$TIMESTAMP"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
TIMEOUT_STARTUP="${TIMEOUT_STARTUP:-600}"  # 10 minutes for model loading
TIMEOUT_REQUEST="${TIMEOUT_REQUEST:-120}"  # 2 minutes per request
STRESS_REQUESTS="${STRESS_REQUESTS:-200}"
STRESS_CONCURRENCY="${STRESS_CONCURRENCY:-8}"

# Docker configuration
DOCKER_CONTAINER="${DOCKER_CONTAINER:-vllm-dev}"
DOCKER_COMPOSE_DIR="${DOCKER_COMPOSE_DIR:-$MXFP4_DIR}"

# Debug toggles (only enable when debugging crashes)
DEBUG_MODE="${DEBUG_MODE:-0}"
if [[ "$DEBUG_MODE" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_SHOW_CPP_STACKTRACES=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=INFO
fi

# FlashInfer logging
export FLASHINFER_LOGLEVEL="${FLASHINFER_LOGLEVEL:-1}"
export FLASHINFER_LOGDEST="${FLASHINFER_LOGDEST:-$RUN_LOG_DIR/flashinfer_api.log}"

# Paths for scripts (in mxfp4 dir) vs FlashInfer benchmarks
SCRIPTS_DIR="$SCRIPT_DIR"

# =============================================================================
# ENVIRONMENT VARIABLES (from docker-compose.dev.yml and docker-compose.yml)
# =============================================================================
# These should match docker-compose environment sections
EXPECTED_ENV_VARS=(
    # MXFP4 configuration
    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1"
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0"
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=0"
    "VLLM_FLASHINFER_MOE_BACKEND=throughput"
    "VLLM_USE_FLASHINFER_MOE_FP4=1"
    # Performance
    "VLLM_ATTENTION_BACKEND=FLASHINFER"
    "VLLM_USE_CUDA_GRAPH=1"
    "FLASHINFER_NVCC_THREADS=4"
)

# vLLM serve command parameters (from docker-compose.yml)
# These are used when the harness starts vLLM itself
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-gpt-oss-120b}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-mxfp4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Create log directory
setup_logging() {
    mkdir -p "$RUN_LOG_DIR"
    log_info "Logs will be written to: $RUN_LOG_DIR"
    
    # Log system info
    {
        echo "=== Run Started: $(date) ==="
        echo "=== Environment ==="
        echo "In Docker: $IN_DOCKER"
        echo "MXFP4 Dir: $MXFP4_DIR"
        echo "FlashInfer Root: $FLASHINFER_ROOT"
        echo "vLLM Root: $VLLM_ROOT"
        echo ""
        echo "=== System Info ==="
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo "Python: $(python3 --version 2>&1)"
        echo ""
        echo "=== GPU Info ==="
        nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi failed"
        echo ""
        echo "=== CUDA Info ==="
        nvcc --version 2>/dev/null || echo "nvcc not found"
    } > "$RUN_LOG_DIR/system_info.log"
}

# =============================================================================
# PROCESS MANAGEMENT (A1-A3 Requirements)
# =============================================================================

# Global variable to track server PID
SERVER_PID=""
SERVER_PGID=""

# Cleanup function - kills entire process group
cleanup() {
    local exit_code=$?
    log_info "Running cleanup (exit code: $exit_code)..."
    
    if [[ -n "$SERVER_PGID" ]]; then
        log_info "Killing process group: -$SERVER_PGID"
        kill -- -"$SERVER_PGID" 2>/dev/null || true
        
        # Wait and force kill if needed
        sleep 2
        kill -9 -- -"$SERVER_PGID" 2>/dev/null || true
    fi
    
    if [[ -n "$SERVER_PID" ]]; then
        log_info "Killing server PID: $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Kill any orphaned vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -f "EngineCore" 2>/dev/null || true
    
    # Check for zombies
    check_zombies
    
    log_info "Cleanup complete"
}

trap cleanup EXIT INT TERM

# Check for zombie processes
check_zombies() {
    log_info "Checking for zombie processes..."
    local zombies
    zombies=$(ps auxwf 2>/dev/null | grep -E 'vllm|EngineCore|api_server|zmq' | grep -v grep || true)
    
    if [[ -n "$zombies" ]]; then
        log_warn "Found lingering processes:"
        echo "$zombies"
        
        # Check for defunct (zombie) processes
        local defunct
        defunct=$(ps aux 2>/dev/null | grep -E 'Z.*<defunct>' | grep -v grep || true)
        if [[ -n "$defunct" ]]; then
            log_error "ZOMBIE PROCESSES DETECTED:"
            echo "$defunct"
            return 1
        fi
    fi
    
    log_success "No zombie processes found"
    return 0
}

# =============================================================================
# CRASH DIAGNOSTICS (A3 Requirements)
# =============================================================================

collect_crash_diagnostics() {
    local log_file="$1"
    local diag_dir="$RUN_LOG_DIR/crash_diagnostics_$(date +%H%M%S)"
    mkdir -p "$diag_dir"
    
    log_error "Collecting crash diagnostics to: $diag_dir"
    
    # Copy the server log
    if [[ -f "$log_file" ]]; then
        cp "$log_file" "$diag_dir/server_log.txt"
    fi
    
    # Collect dmesg (GPU Xid / OOM / illegal access)
    {
        echo "=== dmesg (last 200 lines) ==="
        dmesg -T 2>/dev/null | tail -n 200 || echo "dmesg failed (might need sudo or not in container)"
    } > "$diag_dir/dmesg.log"
    
    # Collect nvidia-smi
    {
        echo "=== nvidia-smi memory/utilization/clock ==="
        nvidia-smi -q -d MEMORY,UTILIZATION,CLOCK 2>/dev/null | tail -n 200 || echo "nvidia-smi failed"
    } > "$diag_dir/nvidia_smi.log"
    
    # Collect process info
    {
        echo "=== Process Tree ==="
        ps auxwf 2>/dev/null | grep -E 'vllm|EngineCore|api_server|zmq|python' | head -50 || true
    } > "$diag_dir/processes.log"
    
    # Check for Xid errors
    if dmesg 2>/dev/null | grep -i "xid" | tail -5; then
        log_error "GPU Xid errors detected in dmesg!"
    fi
    
    # FlashInfer API log
    if [[ -f "$RUN_LOG_DIR/flashinfer_api.log" ]]; then
        cp "$RUN_LOG_DIR/flashinfer_api.log" "$diag_dir/"
    fi
}

# =============================================================================
# ENVIRONMENT VALIDATION (D Requirements)
# =============================================================================

print_env_vars() {
    log_info "=== Environment Variables ==="
    
    echo "--- Docker Compose Expected Variables ---"
    for expected in "${EXPECTED_ENV_VARS[@]}"; do
        var_name="${expected%%=*}"
        expected_value="${expected#*=}"
        actual_value="${!var_name:-<not set>}"
        
        if [[ "$actual_value" == "$expected_value" ]]; then
            echo -e "  ${GREEN}✓${NC} $var_name = $actual_value"
        elif [[ "$actual_value" == "<not set>" ]]; then
            echo -e "  ${YELLOW}!${NC} $var_name = <not set> (expected: $expected_value)"
        else
            echo -e "  ${RED}✗${NC} $var_name = $actual_value (expected: $expected_value)"
        fi
    done
    
    echo ""
    echo "--- FlashInfer Variables ---"
    echo "  FLASHINFER_LOGLEVEL = ${FLASHINFER_LOGLEVEL:-<not set>}"
    echo "  FLASHINFER_LOGDEST = ${FLASHINFER_LOGDEST:-<not set>}"
    echo "  FLASHINFER_JIT_VERBOSE = ${FLASHINFER_JIT_VERBOSE:-<not set>}"
    echo "  FLASHINFER_CUDA_ARCH_LIST = ${FLASHINFER_CUDA_ARCH_LIST:-<auto>}"
    echo "  FLASHINFER_NVCC_THREADS = ${FLASHINFER_NVCC_THREADS:-<not set>}"
    
    echo ""
    echo "--- Paths ---"
    echo "  PYTHONPATH = ${PYTHONPATH:-<not set>}"
    
    # Validate FlashInfer is in PYTHONPATH or installed
    if [[ "${PYTHONPATH:-}" == *"flashinfer"* ]] || [[ "${PYTHONPATH:-}" == *"/workspace"* ]]; then
        log_success "FlashInfer path detected in PYTHONPATH"
    else
        log_warn "Local FlashInfer NOT in PYTHONPATH - using installed version"
    fi
}

validate_compose_env_parity() {
    log_info "=== Docker Compose Environment Parity Check ==="
    
    local compose_file="$DOCKER_COMPOSE_DIR/docker-compose.dev.yml"
    local mismatches=0
    
    if [[ ! -f "$compose_file" ]]; then
        log_warn "Cannot find docker-compose.dev.yml at $compose_file"
        return 0
    fi
    
    # Extract environment variables from compose file
    local compose_vars
    compose_vars=$(grep -E '^\s*-\s+[A-Z_]+=.+$' "$compose_file" | sed 's/^\s*-\s*//' | head -20)
    
    echo "Checking compose environment parity..."
    
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then continue; fi
        
        var_name="${line%%=*}"
        expected_value="${line#*=}"
        actual_value="${!var_name:-<not set>}"
        
        if [[ "$actual_value" != "$expected_value" && "$actual_value" != "<not set>" ]]; then
            log_warn "Mismatch: $var_name = $actual_value (compose has: $expected_value)"
            ((mismatches++))
        fi
    done <<< "$compose_vars"
    
    if [[ $mismatches -eq 0 ]]; then
        log_success "Environment matches docker-compose.dev.yml"
    else
        log_warn "$mismatches environment variable mismatches detected"
    fi
}

validate_environment() {
    log_info "Validating environment..."
    
    # Check GPU
    if ! nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not available"
        return 1
    fi
    
    # Check compute capability
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    log_info "GPU Compute Capability: $cc"
    
    if [[ "$cc" != "121" ]]; then
        log_warn "Expected SM121 (compute_cap 12.1), got: $cc"
    else
        log_success "SM121 GPU detected"
    fi
    
    # Check Python imports
    python3 -c "import flashinfer; print(f'FlashInfer version: {flashinfer.__version__}')" 2>&1 || {
        log_error "FlashInfer import failed"
        return 1
    }
    
    # Check vLLM (if in Docker, it should be importable)
    if $IN_DOCKER; then
        python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>&1 || {
            log_error "vLLM import failed"
            return 1
        }
    fi
    
    print_env_vars | tee "$RUN_LOG_DIR/environment.log"
    
    if $IN_DOCKER; then
        validate_compose_env_parity | tee -a "$RUN_LOG_DIR/environment.log"
    fi
}

# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

start_vllm_server() {
    local model="$1"
    local log_file="$RUN_LOG_DIR/vllm_server.log"
    
    log_info "Starting vLLM server..."
    log_info "Model: $model"
    log_info "Served model name: $VLLM_SERVED_MODEL_NAME"
    log_info "Quantization: $VLLM_QUANTIZATION"
    log_info "Log file: $log_file"
    
    # Build server command - match docker-compose.yml settings EXACTLY
    local cmd=(
        python3 -m vllm.entrypoints.openai.api_server
        --model "$model"
        --host 0.0.0.0
        --port "$VLLM_PORT"
        --served-model-name "$VLLM_SERVED_MODEL_NAME"
        --trust-remote-code
    )
    
    # Quantization (from docker-compose: --quantization mxfp4)
    cmd+=(--quantization "$VLLM_QUANTIZATION")
    
    # Tensor parallelism (from docker-compose: --tensor-parallel-size 1)
    cmd+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
    
    # GPU memory and batching (from docker-compose)
    cmd+=(
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --max-num-seqs "$VLLM_MAX_NUM_SEQS"
        --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    )
    
    # Generation config override (from docker-compose.yml production)
    cmd+=(--override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":0}')
    
    # Performance options
    if [[ "${VLLM_USE_CUDA_GRAPH:-1}" == "1" ]]; then
        cmd+=(--enforce-eager=False)
    fi
    
    # Async scheduling and prefix caching (from docker-compose)
    cmd+=(--async-scheduling)
    cmd+=(--enable-prefix-caching)
    
    # Tool use for GPT-OSS (from docker-compose.yml production)
    cmd+=(--enable-auto-tool-choice)
    cmd+=(--tool-call-parser=openai)
    cmd+=(--reasoning-parser=openai_gptoss)
    
    # Fast loading (from docker-compose)
    cmd+=(--load-format fastsafetensors)
    
    log_info "Command: ${cmd[*]}"
    
    # Start server with line-buffered output, preserving stderr
    # A1: Never pipe through grep - write directly to file
    setsid stdbuf -oL -eL "${cmd[@]}" > "$log_file" 2>&1 &
    SERVER_PID=$!
    SERVER_PGID=$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')
    
    log_info "Server PID: $SERVER_PID, PGID: $SERVER_PGID"
    
    # Wait for health endpoint
    log_info "Waiting for server to be ready (timeout: ${TIMEOUT_STARTUP}s)..."
    log_info "This may take several minutes for large models..."
    local start_time
    start_time=$(date +%s)
    
    while true; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_error "Server process died during startup"
            log_error "Last 100 lines of server log:"
            tail -100 "$log_file"
            collect_crash_diagnostics "$log_file"
            return 1
        fi
        
        if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
            log_success "Server is ready!"
            break
        fi
        
        local elapsed
        elapsed=$(($(date +%s) - start_time))
        
        # Progress update every 30 seconds
        if [[ $((elapsed % 30)) -eq 0 && $elapsed -gt 0 ]]; then
            log_info "Still waiting... ${elapsed}s elapsed"
            # Show last line of log for progress
            tail -1 "$log_file" 2>/dev/null || true
        fi
        
        if [[ $elapsed -ge $TIMEOUT_STARTUP ]]; then
            log_error "Server startup timeout after ${elapsed}s"
            log_error "Last 100 lines of server log:"
            tail -100 "$log_file"
            collect_crash_diagnostics "$log_file"
            return 1
        fi
        
        sleep 2
    done
    
    return 0
}

stop_vllm_server() {
    log_info "Stopping vLLM server..."
    
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while kill -0 "$SERVER_PID" 2>/dev/null && [[ $count -lt 10 ]]; do
            sleep 1
            ((count++))
        done
        
        # Force kill if still running
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log_warn "Force killing server..."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    
    SERVER_PID=""
    SERVER_PGID=""
}

# =============================================================================
# BACKEND VERIFICATION (B1 Requirements)
# =============================================================================

verify_backends() {
    log_info "=== Backend Verification ==="
    local log_file="$RUN_LOG_DIR/vllm_server.log"
    local verification_passed=true
    
    # Create summary file
    local summary_file="$RUN_LOG_DIR/backend_summary.log"
    
    {
        echo "=== Backend Summary ==="
        echo "Timestamp: $(date)"
        echo "Docker Container: ${DOCKER_CONTAINER:-N/A}"
        echo ""
        
        # Check attention backend from environment
        echo "--- Attention Backend ---"
        if [[ "${VLLM_ATTENTION_BACKEND:-}" == "FLASHINFER" ]]; then
            echo "VLLM_ATTENTION_BACKEND: FLASHINFER (configured)"
        else
            echo "VLLM_ATTENTION_BACKEND: ${VLLM_ATTENTION_BACKEND:-not set}"
        fi
        
        # Check from server logs if available
        if [[ -f "$log_file" ]]; then
            if grep -qi "flashinfer" "$log_file" 2>/dev/null; then
                echo "Server Log: FlashInfer detected"
            fi
            
            if grep -qi "attention.*sink\|sink.*attention" "$log_file" 2>/dev/null; then
                echo "Sinks: Detected in logs"
            fi
        fi
        
        echo ""
        echo "--- MoE Backend ---"
        echo "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: ${VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:-not set}"
        echo "VLLM_USE_FLASHINFER_MOE_FP4: ${VLLM_USE_FLASHINFER_MOE_FP4:-not set}"
        echo "VLLM_FLASHINFER_MOE_BACKEND: ${VLLM_FLASHINFER_MOE_BACKEND:-not set}"
        
        # Check from server logs
        if [[ -f "$log_file" ]]; then
            if grep -qiE "mxfp4|fp4.*moe|cutlass.*moe" "$log_file" 2>/dev/null; then
                echo "Server Log: MXFP4/FP4 MoE detected"
            fi
        fi
        
        echo ""
        echo "--- CUDA Graph ---"
        echo "VLLM_USE_CUDA_GRAPH: ${VLLM_USE_CUDA_GRAPH:-not set}"
        
        echo ""
        echo "--- Paths ---"
        if $IN_DOCKER; then
            echo "Running inside Docker: $DOCKER_CONTAINER"
            echo "FlashInfer mount: /workspace/flashinfer"
            echo "vLLM mount: /workspace/vllm"
        else
            echo "Running on host"
            echo "FlashInfer: $REPO_ROOT"
        fi
        
        echo ""
        echo "--- Verification Status ---"
        
        # Check critical settings
        if [[ "${VLLM_ATTENTION_BACKEND:-}" != "FLASHINFER" ]]; then
            echo "WARN: VLLM_ATTENTION_BACKEND != FLASHINFER"
            verification_passed=false
        else
            echo "OK: VLLM_ATTENTION_BACKEND = FLASHINFER"
        fi
        
        if [[ "${VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:-0}" != "1" ]]; then
            echo "WARN: VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 != 1"
        else
            echo "OK: MXFP4 BF16 MoE enabled"
        fi
        
    } | tee "$summary_file"
    
    if $verification_passed; then
        log_success "Backend verification passed"
    else
        log_warn "Backend verification had issues - check $summary_file"
    fi
    
    return 0
}

# =============================================================================
# SMOKE TESTS (B2 Requirements)
# =============================================================================

run_smoke_tests() {
    log_info "=== Running Smoke Tests ==="
    local failed=0
    local results_file="$RUN_LOG_DIR/smoke_tests.log"
    local model_name="${VLLM_SERVED_MODEL_NAME:-gpt-oss-120b}"
    
    # Test 1: Short prompt + short generation
    log_info "Test 1: Short prompt + short generation..."
    local response
    response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$model_name"'",
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        }' 2>&1)
    
    if echo "$response" | jq -e '.choices[0].text' >/dev/null 2>&1; then
        local text
        text=$(echo "$response" | jq -r '.choices[0].text')
        if [[ -n "$text" && "$text" != "null" ]]; then
            log_success "Test 1 passed: Got response of ${#text} chars"
            echo "Test 1: PASSED - ${#text} chars" >> "$results_file"
        else
            log_error "Test 1 failed: Empty response"
            echo "Test 1: FAILED - Empty response" >> "$results_file"
            ((failed++))
        fi
    else
        log_error "Test 1 failed: Invalid JSON response"
        echo "$response" >> "$results_file"
        ((failed++))
    fi
    
    # Test 2: Chat completions (for tool use path)
    log_info "Test 2: Chat completions API..."
    response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$model_name"'",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
            "temperature": 0.7
        }' 2>&1)
    
    if echo "$response" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        local content
        content=$(echo "$response" | jq -r '.choices[0].message.content')
        if [[ -n "$content" && "$content" != "null" ]]; then
            log_success "Test 2 passed: Chat API works"
            echo "Test 2: PASSED - Chat API" >> "$results_file"
        else
            log_error "Test 2 failed: Empty chat response"
            echo "Test 2: FAILED - Empty chat response" >> "$results_file"
            ((failed++))
        fi
    else
        log_error "Test 2 failed: Chat API error"
        echo "$response" >> "$results_file"
        ((failed++))
    fi
    
    # Test 3: Check for NaN/Inf in outputs
    log_info "Test 3: NaN/Inf check..."
    response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$model_name"'",
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.0,
            "logprobs": 1
        }' 2>&1)
    
    if echo "$response" | grep -qiE "nan|inf|-inf"; then
        log_error "Test 3 failed: NaN/Inf detected in response"
        echo "Test 3: FAILED - NaN/Inf detected" >> "$results_file"
        echo "$response" >> "$results_file"
        ((failed++))
    else
        log_success "Test 3 passed: No NaN/Inf in output"
        echo "Test 3: PASSED - No NaN/Inf" >> "$results_file"
    fi
    
    # Test 4: Multi-request concurrency (4 parallel)
    log_info "Test 4: Concurrent requests (4 parallel)..."
    local pids=()
    local concurrent_results="$RUN_LOG_DIR/concurrent_results"
    mkdir -p "$concurrent_results"
    
    for i in {1..4}; do
        (
            curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "'"$model_name"'",
                    "prompt": "Count from 1 to 10: 1, 2, 3,",
                    "max_tokens": 30,
                    "temperature": 0.5
                }' > "$concurrent_results/response_$i.json" 2>&1
        ) &
        pids+=($!)
    done
    
    # Wait for all requests
    local concurrent_failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((concurrent_failed++))
        fi
    done
    
    # Check results
    for i in {1..4}; do
        if ! jq -e '.choices[0].text' "$concurrent_results/response_$i.json" >/dev/null 2>&1; then
            ((concurrent_failed++))
        fi
    done
    
    if [[ $concurrent_failed -eq 0 ]]; then
        log_success "Test 4 passed: All 4 concurrent requests succeeded"
        echo "Test 4: PASSED - 4 concurrent requests" >> "$results_file"
    else
        log_error "Test 4 failed: $concurrent_failed concurrent requests failed"
        echo "Test 4: FAILED - $concurrent_failed failures" >> "$results_file"
        ((failed++))
    fi
    
    # Test 5: Repeated runs (check for memory leaks)
    log_info "Test 5: Repeated runs (memory stability)..."
    local initial_mem
    initial_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    for i in {1..5}; do
        curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$model_name"'",
                "prompt": "Test iteration '"$i"': Hello",
                "max_tokens": 20
            }' >/dev/null 2>&1
    done
    
    local final_mem
    final_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    local mem_diff=$((final_mem - initial_mem))
    
    if [[ $mem_diff -lt 1000 ]]; then  # Less than 1GB increase
        log_success "Test 5 passed: Memory stable (delta: ${mem_diff}MB)"
        echo "Test 5: PASSED - Memory stable (delta: ${mem_diff}MB)" >> "$results_file"
    else
        log_warn "Test 5 warning: Memory increased by ${mem_diff}MB"
        echo "Test 5: WARN - Memory increased by ${mem_diff}MB" >> "$results_file"
    fi
    
    # Summary
    echo ""
    echo "=== Smoke Test Summary ===" | tee -a "$results_file"
    if [[ $failed -eq 0 ]]; then
        log_success "All smoke tests passed!"
        echo "Status: ALL PASSED" >> "$results_file"
    else
        log_error "$failed smoke tests failed"
        echo "Status: $failed FAILED" >> "$results_file"
        return 1
    fi
    
    return 0
}

# =============================================================================
# STRESS TESTS (B3 Requirements)
# =============================================================================

run_stress_tests() {
    log_info "=== Running Stress Tests ==="
    log_info "Requests: $STRESS_REQUESTS, Concurrency: $STRESS_CONCURRENCY"
    
    local results_file="$RUN_LOG_DIR/stress_tests.log"
    local failed=0
    local success=0
    local model_name="${VLLM_SERVED_MODEL_NAME:-gpt-oss-120b}"
    
    # Initial memory baseline
    local initial_mem
    initial_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    # Create stress test script
    cat > "$RUN_LOG_DIR/stress_worker.py" << 'STRESS_SCRIPT'
#!/usr/bin/env python3
import sys
import json
import time
import random
import requests
import argparse

def make_request(port, model, seed):
    random.seed(seed)
    
    # Randomized prompt length
    prompt_len = random.choice([10, 50, 100, 200])
    prompt = " ".join(["word"] * prompt_len)
    
    # Randomized parameters
    max_tokens = random.choice([10, 20, 50])
    temperature = random.uniform(0.1, 1.0)
    top_p = random.uniform(0.8, 1.0)
    
    start = time.time()
    try:
        response = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=120
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            if data.get("choices") and data["choices"][0].get("text"):
                return {"success": True, "latency": latency, "status": 200}
            else:
                return {"success": False, "error": "Empty response", "latency": latency, "status": 200}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "latency": latency, "status": response.status_code}
    except Exception as e:
        return {"success": False, "error": str(e), "latency": time.time() - start, "status": 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    result = make_request(args.port, args.model, args.seed)
    print(json.dumps(result))
STRESS_SCRIPT
    
    chmod +x "$RUN_LOG_DIR/stress_worker.py"
    
    # Run stress test with concurrency control
    log_info "Starting stress test..."
    local active_pids=()
    local http_5xx=0
    
    for i in $(seq 1 "$STRESS_REQUESTS"); do
        # Check server health periodically
        if [[ $((i % 50)) -eq 0 ]]; then
            if ! curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
                log_error "Server became unhealthy at request $i"
                collect_crash_diagnostics "$RUN_LOG_DIR/vllm_server.log"
                return 1
            fi
            
            # Check memory growth
            local current_mem
            current_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            local mem_growth=$((current_mem - initial_mem))
            log_info "Progress: $i/$STRESS_REQUESTS requests, Memory growth: ${mem_growth}MB"
        fi
        
        # Wait if we have too many concurrent requests
        while [[ ${#active_pids[@]} -ge $STRESS_CONCURRENCY ]]; do
            local new_pids=()
            for pid in "${active_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    wait "$pid" 2>/dev/null || true
                fi
            done
            active_pids=("${new_pids[@]}")
            
            if [[ ${#active_pids[@]} -ge $STRESS_CONCURRENCY ]]; then
                sleep 0.1
            fi
        done
        
        # Launch request
        (
            python3 "$RUN_LOG_DIR/stress_worker.py" \
                --port "$VLLM_PORT" \
                --model "$model_name" \
                --seed "$i" > "$RUN_LOG_DIR/stress_result_$i.json" 2>&1
        ) &
        active_pids+=($!)
    done
    
    # Wait for all to complete
    log_info "Waiting for remaining requests to complete..."
    for pid in "${active_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    
    # Analyze results
    log_info "Analyzing stress test results..."
    local total_latency=0
    local max_latency=0
    
    for i in $(seq 1 "$STRESS_REQUESTS"); do
        local result_file="$RUN_LOG_DIR/stress_result_$i.json"
        if [[ -f "$result_file" ]]; then
            local result
            result=$(cat "$result_file")
            if echo "$result" | jq -e '.success == true' >/dev/null 2>&1; then
                ((success++))
                local latency
                latency=$(echo "$result" | jq -r '.latency')
                total_latency=$(echo "$total_latency + $latency" | bc)
                if (( $(echo "$latency > $max_latency" | bc -l) )); then
                    max_latency=$latency
                fi
            else
                ((failed++))
                local error status
                error=$(echo "$result" | jq -r '.error // "Unknown"')
                status=$(echo "$result" | jq -r '.status // 0')
                if [[ "$status" -ge 500 ]]; then
                    ((http_5xx++))
                fi
                echo "Request $i failed: $error (status: $status)" >> "$results_file"
            fi
        else
            ((failed++))
            echo "Request $i: No result file" >> "$results_file"
        fi
    done
    
    # Final memory check
    local final_mem
    final_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    local total_mem_growth=$((final_mem - initial_mem))
    
    # Summary
    {
        echo "=== Stress Test Summary ==="
        echo "Total requests: $STRESS_REQUESTS"
        echo "Concurrency: $STRESS_CONCURRENCY"
        echo "Successful: $success"
        echo "Failed: $failed"
        echo "HTTP 5xx errors: $http_5xx"
        if [[ $success -gt 0 ]]; then
            local avg_latency
            avg_latency=$(echo "scale=3; $total_latency / $success" | bc)
            echo "Average latency: ${avg_latency}s"
            echo "Max latency: ${max_latency}s"
        fi
        echo "Memory growth: ${total_mem_growth}MB"
    } | tee -a "$results_file"
    
    # Acceptance criteria
    if [[ $failed -gt 0 ]]; then
        log_error "Stress test failed: $failed requests failed"
        return 1
    fi
    
    if [[ $http_5xx -gt 0 ]]; then
        log_error "Stress test failed: $http_5xx HTTP 5xx errors"
        return 1
    fi
    
    if [[ $total_mem_growth -gt 5000 ]]; then  # 5GB threshold
        log_warn "Memory growth exceeded threshold: ${total_mem_growth}MB"
    fi
    
    log_success "Stress test passed: $success/$STRESS_REQUESTS requests succeeded"
    return 0
}

# =============================================================================
# PERFORMANCE BENCHMARKS (C Requirements)
# =============================================================================

run_benchmarks() {
    log_info "=== Running Performance Benchmarks ==="
    local results_file="$RUN_LOG_DIR/benchmark_results.csv"
    local model_name="${VLLM_SERVED_MODEL_NAME:-gpt-oss-120b}"
    
    # CSV header
    echo "test_type,prompt_len,generation_len,batch_size,ttft_ms,tpot_ms,throughput_tok_s" > "$results_file"
    
    # Run FlashInfer microbenchmarks (if available)
    if [[ -f "$FLASHINFER_ROOT/benchmarks/sm121_mxfp4_moe_gemm_bench.py" ]]; then
        log_info "Running FlashInfer MoE GEMM benchmark..."
        python3 "$FLASHINFER_ROOT/benchmarks/sm121_mxfp4_moe_gemm_bench.py" \
            --regime both \
            --hidden-dim 4096 \
            --num-experts 8 \
            --iters 50 \
            2>&1 | tee "$RUN_LOG_DIR/moe_gemm_bench.log" || log_warn "MoE GEMM benchmark failed"
    fi
    
    # Prefill benchmark (TTFT)
    log_info "Running prefill benchmarks (TTFT)..."
    for prompt_len in 512 1024 2048 4096; do
        local prompt
        prompt=$(python3 -c "print('word ' * ($prompt_len // 4))")
        
        local start_time end_time ttft
        start_time=$(python3 -c "import time; print(time.time())")
        
        local response
        response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$model_name"'",
                "prompt": "'"$prompt"'",
                "max_tokens": 1,
                "temperature": 0.0
            }' 2>&1)
        
        end_time=$(python3 -c "import time; print(time.time())")
        ttft=$(echo "scale=3; ($end_time - $start_time) * 1000" | bc)
        
        echo "prefill,$prompt_len,1,1,$ttft,N/A,N/A" >> "$results_file"
        log_info "Prefill $prompt_len tokens: TTFT=${ttft}ms"
    done
    
    # Decode benchmark (TPOT, throughput)
    log_info "Running decode benchmarks..."
    for gen_len in 32 64 128; do
        local start_time end_time total_time tpot throughput
        start_time=$(python3 -c "import time; print(time.time())")
        
        local response
        response=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$model_name"'",
                "prompt": "Tell me a story about",
                "max_tokens": '"$gen_len"',
                "temperature": 0.7
            }' 2>&1)
        
        end_time=$(python3 -c "import time; print(time.time())")
        total_time=$(echo "scale=3; $end_time - $start_time" | bc)
        tpot=$(echo "scale=3; ($total_time * 1000) / $gen_len" | bc)
        throughput=$(echo "scale=1; $gen_len / $total_time" | bc)
        
        echo "decode,10,$gen_len,1,N/A,$tpot,$throughput" >> "$results_file"
        log_info "Decode $gen_len tokens: TPOT=${tpot}ms, Throughput=${throughput} tok/s"
    done
    
    log_info "Benchmark results saved to: $results_file"
    cat "$results_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

SM121 MXFP4 MoE GEMM + FA2 Sinks Testing Harness

Options:
    --mode MODE     Test mode: all, verify, stress, benchmark (default: all)
    --model PATH    Model path or name (default: $VLLM_MODEL)
    --port PORT     vLLM server port (default: $VLLM_PORT)
    --debug         Enable debug mode (CUDA_LAUNCH_BLOCKING, etc.)
    --skip-server   Skip starting server (use existing)
    --help          Show this help message

Docker Usage:
    # From host:
    cd ~/projects/ai/mxfp4
    ./scripts/sm121_vllm_test_harness.sh --mode verify
    
    # Or from inside vllm-dev container (mount the script or copy it):
    docker exec -it vllm-dev /host/scripts/sm121_vllm_test_harness.sh --mode verify

Environment Variables (from docker-compose.dev.yml):
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
    VLLM_ATTENTION_BACKEND=FLASHINFER
    VLLM_USE_CUDA_GRAPH=1
    
    # Stress test settings
    STRESS_REQUESTS=200
    STRESS_CONCURRENCY=8
    TIMEOUT_STARTUP=600

EOF
}

main() {
    local mode="all"
    local skip_server=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                mode="$2"
                shift 2
                ;;
            --model)
                VLLM_MODEL="$2"
                shift 2
                ;;
            --port)
                VLLM_PORT="$2"
                shift 2
                ;;
            --debug)
                DEBUG_MODE=1
                export CUDA_LAUNCH_BLOCKING=1
                export TORCH_SHOW_CPP_STACKTRACES=1
                shift
                ;;
            --skip-server)
                skip_server=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Setup
    setup_logging
    
    echo "============================================================"
    echo "SM121 MXFP4 MoE GEMM + FA2 Sinks Testing Harness"
    echo "============================================================"
    echo "Mode: $mode"
    echo "Model: $VLLM_MODEL"
    echo "In Docker: $IN_DOCKER"
    echo "Log directory: $RUN_LOG_DIR"
    echo "============================================================"
    
    # Validate environment
    validate_environment || exit 1
    
    # Start server if needed
    if ! $skip_server; then
        # Check if server is already running
        if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
            log_info "Server already running on port $VLLM_PORT"
        else
            start_vllm_server "$VLLM_MODEL" || exit 1
        fi
    fi
    
    # Run tests based on mode
    local exit_code=0
    
    case $mode in
        all)
            verify_backends || exit_code=1
            run_smoke_tests || exit_code=1
            run_stress_tests || exit_code=1
            run_benchmarks || exit_code=1
            ;;
        verify)
            verify_backends || exit_code=1
            run_smoke_tests || exit_code=1
            ;;
        stress)
            verify_backends || exit_code=1
            run_stress_tests || exit_code=1
            ;;
        benchmark)
            verify_backends || exit_code=1
            run_benchmarks || exit_code=1
            ;;
        *)
            log_error "Unknown mode: $mode"
            print_usage
            exit 1
            ;;
    esac
    
    # Check for zombies
    check_zombies || exit_code=1
    
    # Final summary
    echo ""
    echo "============================================================"
    echo "Test Run Complete"
    echo "============================================================"
    echo "Log directory: $RUN_LOG_DIR"
    echo "Exit code: $exit_code"
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed - check logs for details"
    fi
    
    exit $exit_code
}

main "$@"
