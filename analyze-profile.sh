#!/bin/bash
# =============================================================================
# Analyze nsys profile from start-profile.sh
# =============================================================================

PROFILE="${1:-/tmp/nsys_profile/vllm_inference.nsys-rep}"

if [ ! -f "$PROFILE" ]; then
    # Check inside docker
    docker exec vllm-dev test -f "$PROFILE" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Profile not found: $PROFILE"
        echo "Usage: $0 [profile.nsys-rep]"
        exit 1
    fi
    EXEC_PREFIX="docker exec vllm-dev"
else
    EXEC_PREFIX=""
fi

echo "=============================================="
echo "Analyzing: $PROFILE"
echo "=============================================="

echo ""
echo "=== Top 20 CUDA Kernels by Total Time ==="
$EXEC_PREFIX nsys stats "$PROFILE" --report cuda_gpu_kern_sum --force-export=true 2>/dev/null | head -30

echo ""
echo "=== Top 20 Kernels (sorted by total time) ==="
echo ""
echo "The first report above shows kernel names. Here's a simplified view:"
echo ""
# Parse the table output instead of CSV (kernel names have commas)
$EXEC_PREFIX nsys stats "$PROFILE" --report cuda_gpu_kern_sum --force-export=true 2>/dev/null | \
    grep -E "^\s+[0-9]" | head -20 | \
    while read line; do
        # Extract percentage and a shortened kernel name
        pct=$(echo "$line" | awk '{print $1}')
        time_ns=$(echo "$line" | awk '{print $2}')
        instances=$(echo "$line" | awk '{print $3}')
        # Get kernel name (everything after the 8th field)
        name=$(echo "$line" | awk '{for(i=9;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/^[ ]*//' | cut -c1-70)
        time_ms=$(echo "scale=2; $time_ns / 1000000" | bc 2>/dev/null || echo "N/A")
        printf "%5s%%  %8s ms  %5s calls  %s\n" "$pct" "$time_ms" "$instances" "$name"
    done

echo ""
echo "=== CUDA API Summary ==="
$EXEC_PREFIX nsys stats "$PROFILE" --report cuda_api_sum --force-export=true 2>/dev/null | head -20

echo ""
echo "=== Memory Operations ==="
$EXEC_PREFIX nsys stats "$PROFILE" --report cuda_gpu_mem_time_sum --force-export=true 2>/dev/null | head -15

echo ""
echo "=============================================="
echo "For full interactive analysis:"
echo "  nsys-ui $PROFILE"
echo "=============================================="
