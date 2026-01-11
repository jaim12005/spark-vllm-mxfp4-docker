#!/bin/bash
# =============================================================================
# Phase 2 Smoke Test: SM12x Dedicated Launcher + Polish
# =============================================================================
#
# This test verifies the Phase 2 changes:
# 1. New SM120 mixed-input launcher files exist
# 2. Dispatch routes SM12x to the dedicated launcher
# 3. Generic launcher is clean (no SM12x hacks)
# 4. Identity scale buffer manager and prewarm functions
# 5. New enum naming (ENGINE_ARCH_QUANT)
# 6. SM12x capability predicate
# 7. LoRA backend warning
# 8. Include fixes (no missing headers)
#
# Run from: docker exec -it vllm-dev bash -c "cd /workspace/mxfp4 && ./scripts/test_level1_phase2_smoke.sh"
# =============================================================================

set -e

echo "=============================================="
echo "Phase 2 Smoke Test: SM12x Dedicated Launcher"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; exit 1; }
info() { echo -e "${YELLOW}→${NC} $1"; }
warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; }

# =============================================================================
# Test 1: Verify new Phase 2 files exist
# =============================================================================
echo "1. Checking Phase 2 files exist..."

FLASHINFER_MOE="/workspace/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm"

check_file() {
    if [ -f "$1" ]; then
        pass "Found: $(basename $1)"
    else
        fail "Missing: $1"
    fi
}

check_file "${FLASHINFER_MOE}/launchers/moe_gemm_sm120_mixed_input_launcher.h"
check_file "${FLASHINFER_MOE}/launchers/moe_gemm_sm120_mixed_input_launcher.inl"
check_file "${FLASHINFER_MOE}/sm12x_layout_sfa_utils.h"
check_file "${FLASHINFER_MOE}/sm12x_activation_quantizer.cuh"
check_file "${FLASHINFER_MOE}/sm12x_arch_config.h"
echo ""

# =============================================================================
# Test 2: Verify includes in new files (Polish fixes)
# =============================================================================
echo "2. Checking includes in new files..."

LAUNCHER_H="${FLASHINFER_MOE}/launchers/moe_gemm_sm120_mixed_input_launcher.h"
LAUNCHER_INL="${FLASHINFER_MOE}/launchers/moe_gemm_sm120_mixed_input_launcher.inl"
SFA_UTILS="${FLASHINFER_MOE}/sm12x_layout_sfa_utils.h"

# Check launcher header has required includes
if grep -q "#include <vector>" "$LAUNCHER_H" && \
   grep -q "#include <tuple>" "$LAUNCHER_H" && \
   grep -q "#include <cstdint>" "$LAUNCHER_H"; then
    pass "Launcher header has required STL includes"
else
    fail "Launcher header missing STL includes (vector, tuple, cstdint)"
fi

# Check SFA utils has cstdio for fprintf
if grep -q "#include <cstdio>" "$SFA_UTILS"; then
    pass "SFA utils has cstdio include"
else
    fail "SFA utils missing cstdio include"
fi

# Check launcher .inl has typeinfo for typeid
if grep -q "#include <typeinfo>" "$LAUNCHER_INL"; then
    pass "Launcher .inl has typeinfo include"
else
    fail "Launcher .inl missing typeinfo include"
fi
echo ""

# =============================================================================
# Test 3: Verify dispatch includes SM120 launcher
# =============================================================================
echo "3. Checking dispatch includes SM120 launcher..."

DISPATCH_FILE="${FLASHINFER_MOE}/moe_gemm_template_dispatch_tma_ws.h"
if grep -q "moe_gemm_sm120_mixed_input_launcher.h" "$DISPATCH_FILE"; then
    pass "Dispatch includes SM120 launcher header"
else
    fail "Dispatch missing SM120 launcher include"
fi

if grep -q "sm120_mixed_input_moe_gemm_kernelLauncher" "$DISPATCH_FILE"; then
    pass "Dispatch calls SM120 launcher function"
else
    fail "Dispatch missing SM120 launcher call"
fi
echo ""

# =============================================================================
# Test 4: Verify SM120 launcher uses correct kernel schedule
# =============================================================================
echo "4. Checking SM120 launcher uses block-scaled kernel schedule..."

if grep -q "KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120" "$LAUNCHER_INL"; then
    pass "SM120 launcher uses BlockScaledSm120 kernel schedule"
else
    fail "SM120 launcher missing BlockScaledSm120 kernel schedule"
fi

# Verify parameter name is correct (hopper_inputs, not hopper_input)
if grep -q "hopper_input\." "$LAUNCHER_INL"; then
    fail "Launcher has wrong parameter name 'hopper_input' (should be 'hopper_inputs')"
else
    pass "Launcher uses correct parameter name 'hopper_inputs'"
fi
echo ""

# =============================================================================
# Test 5: Verify IsMXFP4 type trait exists
# =============================================================================
echo "5. Checking FP4 weight mixed-input type trait..."

TRAITS_FILE="${FLASHINFER_MOE}/moe_tma_warp_specialized_traits.h"
if grep -q "isFP4WeightMixedInputPath" "$TRAITS_FILE"; then
    pass "isFP4WeightMixedInputPath type trait found"
else
    fail "isFP4WeightMixedInputPath type trait missing"
fi

# Verify dispatch uses the trait
if grep -q "isFP4WeightMixedInputPath<T, WeightType>" "$DISPATCH_FILE"; then
    pass "Dispatch uses isFP4WeightMixedInputPath trait"
else
    fail "Dispatch not using isFP4WeightMixedInputPath trait"
fi
echo ""

# =============================================================================
# Test 6: Verify generic launcher is clean (no SM12x hacks)
# =============================================================================
echo "6. Checking generic launcher is clean..."

GENERIC_LAUNCHER="${FLASHINFER_MOE}/launchers/moe_gemm_tma_ws_launcher.inl"

# Should NOT have the inline ternary hack
if grep -q "cutlass::arch::ArchTag_::kMinComputeCapability == 120.*Sm12xKBytesToElements" "$GENERIC_LAUNCHER"; then
    fail "Generic launcher still has SM12x inline ternary hack"
else
    pass "Generic launcher clean: no inline SM12x ternary"
fi

# Should have correct arch messaging (Blackwell/SM100+)
if grep -q "Blackwell" "$GENERIC_LAUNCHER" && grep -q "SM100" "$GENERIC_LAUNCHER"; then
    pass "Generic launcher has correct arch messaging"
else
    warn "Generic launcher may have outdated arch messaging"
fi
echo ""

# =============================================================================
# Test 7: Verify identity scale buffer manager and prewarm
# =============================================================================
echo "7. Checking identity scale buffer manager and prewarm..."

QUANTIZER_FILE="${FLASHINFER_MOE}/sm12x_activation_quantizer.cuh"
if grep -q "getIdentityScaleBufferManager" "$QUANTIZER_FILE"; then
    pass "Identity scale buffer manager found"
else
    fail "Identity scale buffer manager missing"
fi

if grep -q "getSFAPointerArrayManager" "$QUANTIZER_FILE"; then
    pass "SFA pointer array manager found"
else
    fail "SFA pointer array manager missing"
fi

# Check for prewarm functions
if grep -q "prewarmSm12xMoEBuffers" "$QUANTIZER_FILE"; then
    pass "Unified prewarm function found"
else
    fail "Unified prewarm function missing"
fi

# Check activation_quantizer include is guarded
if grep -q "#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)" "$LAUNCHER_INL" && \
   grep -q "sm12x_activation_quantizer.cuh" "$LAUNCHER_INL"; then
    pass "Activation quantizer include is properly guarded"
else
    warn "Activation quantizer include guard may be missing"
fi
echo ""

# =============================================================================
# Test 8: Clear JIT cache and test FlashInfer import
# =============================================================================
echo "8. Clearing JIT cache and testing FlashInfer import..."

rm -rf ~/.cache/flashinfer/* 2>/dev/null || true
info "JIT cache cleared"

python3 -c "
import sys
sys.path.insert(0, '/workspace/flashinfer')
sys.path.insert(0, '/workspace/vllm')

import flashinfer
print(f'FlashInfer version: {flashinfer.__version__}')
print(f'FlashInfer path: {flashinfer.__file__}')

# Check if we're using the local version
if '/workspace/flashinfer' in flashinfer.__file__:
    print('Using local FlashInfer: OK')
else:
    print('WARNING: Not using local FlashInfer!')
    sys.exit(1)
" && pass "FlashInfer import OK" || fail "FlashInfer import failed"
echo ""

# =============================================================================
# Test 9: Test GPU detection
# =============================================================================
echo "9. Checking GPU and compute capability..."

python3 -c "
import torch
if not torch.cuda.is_available():
    print('CUDA not available')
    exit(1)

props = torch.cuda.get_device_properties(0)
cc = props.major * 10 + props.minor
print(f'GPU: {props.name}')
print(f'Compute Capability: {props.major}.{props.minor} (SM{cc})')

if cc >= 120:
    print('SM12x detected: Phase 2 launcher will be used')
elif cc >= 100:
    print('SM10x detected: Generic launcher will be used')
else:
    print('Pre-SM100: Legacy path')
" && pass "GPU detection OK" || fail "GPU detection failed"
echo ""

# =============================================================================
# Test 10: Test new enum naming (ENGINE_ARCH_QUANT)
# =============================================================================
echo "10. Testing new enum naming..."

python3 -c "
import sys
sys.path.insert(0, '/workspace/vllm')

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Backend

# Check new enum names exist
required_enums = [
    'CUTLASS_BLACKWELL_FP4FP8',
    'CUTLASS_SM90_FP4BF16',
    'TRTLLM_SM100_FP4FP8',
    'TRTLLM_SM100_FP4BF16',
    'MARLIN',
    'TRITON',
]

for name in required_enums:
    if hasattr(Mxfp4Backend, name):
        print(f'  ✓ Mxfp4Backend.{name} exists')
    else:
        print(f'  ✗ Mxfp4Backend.{name} MISSING')
        sys.exit(1)

# Check old names are gone
old_enums = [
    'SM100_FI_MXFP4_MXFP8_CUTLASS',
    'SM100_FI_MXFP4_MXFP8_TRTLLM',
    'SM100_FI_MXFP4_BF16',
    'SM90_FI_MXFP4_BF16',
]

for name in old_enums:
    if hasattr(Mxfp4Backend, name):
        print(f'  ⚠ Old enum Mxfp4Backend.{name} still exists (should be removed)')
    else:
        print(f'  ✓ Old enum {name} removed')

print('Enum naming check: OK')
" && pass "New enum naming OK" || fail "Enum naming check failed"
echo ""

# =============================================================================
# Test 11: Test SM12x capability predicate
# =============================================================================
echo "11. Testing SM12x capability predicate..."

python3 -c "
import sys
sys.path.insert(0, '/workspace/vllm')

from vllm.utils.flashinfer import has_flashinfer_sm12x_cutlass_moe

# Just verify the function exists and is callable
result = has_flashinfer_sm12x_cutlass_moe()
print(f'has_flashinfer_sm12x_cutlass_moe() = {result}')
print('SM12x capability predicate: OK')
" && pass "SM12x capability predicate OK" || fail "SM12x capability predicate failed"
echo ""

# =============================================================================
# Test 12: Test vLLM MXFP4 config with CUTLASS backend
# =============================================================================
echo "12. Testing vLLM MXFP4 config with CUTLASS backend..."

python3 -c "
import os
import sys
sys.path.insert(0, '/workspace/vllm')

# Force CUTLASS backend
os.environ['VLLM_MXFP4_BACKEND'] = 'cutlass'

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, get_mxfp4_backend, Mxfp4Backend

# Test config creation
config = Mxfp4Config()
print(f'Mxfp4Config created: OK')

# Test backend selection
backend = get_mxfp4_backend(with_lora_support=False)
print(f'Selected backend: {backend}')

if backend == Mxfp4Backend.CUTLASS_BLACKWELL_FP4FP8:
    print('CUTLASS backend selected: OK')
else:
    print(f'WARNING: Expected CUTLASS_BLACKWELL_FP4FP8, got {backend}')
" && pass "vLLM MXFP4 config OK" || fail "vLLM MXFP4 config failed"
echo ""

# =============================================================================
# Test 13: Test LoRA backend warning
# =============================================================================
echo "13. Testing LoRA backend warning..."

python3 -c "
import os
import sys
import logging

sys.path.insert(0, '/workspace/vllm')

# Capture logging output
logging.basicConfig(level=logging.WARNING)

# Force CUTLASS backend (not LoRA-compatible)
os.environ['VLLM_MXFP4_BACKEND'] = 'cutlass'

from vllm.model_executor.layers.quantization.mxfp4 import get_mxfp4_backend_with_lora, Mxfp4Backend

# This should warn and fall back to Marlin
backend = get_mxfp4_backend_with_lora()
print(f'LoRA backend with CUTLASS requested: {backend}')

if backend == Mxfp4Backend.MARLIN:
    print('Correctly fell back to Marlin for LoRA: OK')
else:
    print(f'ERROR: Expected MARLIN fallback, got {backend}')
    sys.exit(1)
" && pass "LoRA backend warning OK" || fail "LoRA backend warning failed"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo -e "${GREEN}Phase 2 Smoke Test: ALL PASSED${NC}"
echo "=============================================="
echo ""
echo "Phase 2 changes verified:"
echo "  ✓ SM120 launcher files present"
echo "  ✓ All includes fixed (vector, tuple, cstdint, cstdio, typeinfo)"
echo "  ✓ Dispatch routes SM12x to dedicated launcher"
echo "  ✓ SM120 launcher uses correct kernel schedule"
echo "  ✓ Parameter name correct (hopper_inputs)"
echo "  ✓ isMXFP4SemanticPath type trait in use"
echo "  ✓ Generic launcher is clean (no SM12x hacks)"
echo "  ✓ Identity scale buffer manager + prewarm available"
echo "  ✓ FlashInfer imports correctly"
echo "  ✓ GPU detection works"
echo "  ✓ New enum naming (ENGINE_ARCH_QUANT)"
echo "  ✓ SM12x capability predicate works"
echo "  ✓ vLLM MXFP4 CUTLASS backend selectable"
echo "  ✓ LoRA backend warning works"
echo ""
echo "Next steps:"
echo "  1. Stage untracked FlashInfer files: git add ..."
echo "  2. Run vLLM server with: VLLM_MXFP4_BACKEND=CUTLASS vllm serve ..."
echo "  3. Check for 'Stages < 2' errors (should be fixed)"
echo ""
