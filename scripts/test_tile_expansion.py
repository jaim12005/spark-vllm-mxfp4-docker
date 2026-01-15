#!/usr/bin/env python3
"""Test SM120 MoE tile expansion compilation.

This script validates that both TILE_M=32 (Transpose Mode) and TILE_M=128 variants
compile successfully on SM121 (GB10).
"""

import os
import sys
import time
from pathlib import Path

# Force minimal fused-MoE build: compile only MXFP4 path.
os.environ["FLASHINFER_FUSED_MOE_BUILD_PROFILE"] = "mxfp4_minimal"

# Detect environment and set paths
if Path("/workspace/flashinfer").exists():
    # Docker environment
    FLASHINFER_PATH = "/workspace/flashinfer"
    VLLM_PATH = "/workspace/vllm"
else:
    # Local environment (assume sibling directories)
    project_root = Path(__file__).parent.parent.resolve()
    # Check if flashinfer is in ../../flashinfer (relative to mxfp4) or ../flashinfer (relative to ai)
    potential_root_1 = project_root.parent.parent
    if (potential_root_1 / "flashinfer").exists():
        workspace_root = potential_root_1
    else:
        workspace_root = project_root.parent

    FLASHINFER_PATH = str(workspace_root / "flashinfer")
    VLLM_PATH = str(workspace_root / "vllm")

print(f"Using FlashInfer path: {FLASHINFER_PATH}")
print(f"Using vLLM path: {VLLM_PATH}")

os.environ["PYTHONPATH"] = f"{FLASHINFER_PATH}:{VLLM_PATH}:{os.environ.get('PYTHONPATH', '')}"
sys.path.insert(0, FLASHINFER_PATH)
sys.path.insert(0, VLLM_PATH)


def test_module_loading():
    """Test that FlashInfer modules can be imported."""
    print("=" * 60)
    print("Testing FlashInfer module import...")
    print("=" * 60)
    
    import flashinfer
    print(f"FlashInfer path: {flashinfer.__file__}")
    
    from flashinfer.fused_moe.core import (
        get_cutlass_fused_moe_module,
        select_tile_mn_for_sm120,
    )
    
    print(f"select_tile_mn_for_sm120(1) = {select_tile_mn_for_sm120(1)}")
    print(f"select_tile_mn_for_sm120(32) = {select_tile_mn_for_sm120(32)}")
    print(f"select_tile_mn_for_sm120(128) = {select_tile_mn_for_sm120(128)}")
    print(f"select_tile_mn_for_sm120(129) = {select_tile_mn_for_sm120(129)}")
    print()


def test_jit_compile_tile_128x128():
    """Test compilation of logical tile (128,128) (baseline)."""
    print("=" * 60)
    print("Testing tile_mn=(128,128) (baseline) JIT compilation...")
    print("=" * 60)
    
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    start = time.time()
    try:
        # This will JIT compile if not cached
        module_128 = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_mn=(128, 128),
        )
        elapsed = time.time() - start
        print(f"✓ tile_mn=(128,128) module loaded successfully ({elapsed:.1f}s)")
        print(f"  Module type: {type(module_128)}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ tile_mn=(128,128) compilation FAILED ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        return False


def test_jit_compile_tile_128x32():
    """Test compilation of logical tile (128,32) (standard small-N)."""
    print()
    print("=" * 60)
    print("Testing tile_mn=(128,32) (standard small-N) JIT compilation...")
    print("=" * 60)
    
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    start = time.time()
    try:
        # This will JIT compile if not cached
        module_32 = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_mn=(128, 32),
        )
        elapsed = time.time() - start
        print(f"✓ tile_mn=(128,32) module loaded successfully ({elapsed:.1f}s)")
        print(f"  Module type: {type(module_32)}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ tile_mn=(128,32) compilation FAILED ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jit_compile_tile_32x128():
    """Test compilation of logical tile (32,128) (swap-hack)."""
    print()
    print("=" * 60)
    print("Testing tile_mn=(32,128) (swap-hack) JIT compilation...")
    print("=" * 60)
    
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    start = time.time()
    try:
        # This will JIT compile if not cached
        module_256 = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_mn=(32, 128),
        )
        elapsed = time.time() - start
        print(f"✓ tile_mn=(32,128) module loaded successfully ({elapsed:.1f}s)")
        print(f"  Module type: {type(module_256)}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ tile_mn=(32,128) compilation FAILED ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print(" SM120 MoE Tile Expansion Test")
    print("=" * 60 + "\n")
    
    # Clear MoE kernel cache to force recompilation
    cache_dir = "/root/.cache/flashinfer"
    print(f"Note: JIT cache at {cache_dir}")
    print("To force recompile, run: rm -rf ~/.cache/flashinfer/*/cached_ops/fused_moe_*")
    print()
    
    try:
        test_module_loading()
    except Exception as e:
        print(f"Failed to load module: {e}")
        return 1
    
    success_128 = test_jit_compile_tile_128x128()
    success_32 = test_jit_compile_tile_128x32()
    success_256 = test_jit_compile_tile_32x128()
    
    print()
    print("=" * 60)
    print(" Test Summary")
    print("=" * 60)
    print(f"  tile_mn=(128,128): {'✓ PASS' if success_128 else '✗ FAIL'}")
    print(f"  tile_mn=(128,32):  {'✓ PASS' if success_32 else '✗ FAIL'}")
    print(f"  tile_mn=(32,128):  {'✓ PASS' if success_256 else '✗ FAIL'}")
    print()
    
    if success_128 and success_32 and success_256:
        print("All tile expansion tests PASSED!")
        
        # Optional: Test prewarming
        print("\n" + "=" * 60)
        print(" Testing prewarm_moe_tiles()")
        print("=" * 60)
        try:
            from flashinfer.fused_moe.core import prewarm_moe_tiles
            # This should be fast since they are already compiled
            start = time.time()
            prewarm_moe_tiles()
            print(f"✓ prewarm_moe_tiles() completed in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"✗ prewarm_moe_tiles() failed: {e}")
            
        return 0
    else:
        print("Some tile expansion tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())