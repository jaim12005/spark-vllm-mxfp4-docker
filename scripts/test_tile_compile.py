#!/usr/bin/env python3
"""
Compile test for SM120/121 MoE GEMM tile configurations.
Tests JIT compilation of different tile sizes without running a full inference.
"""
import sys
import time


def test_tile_compile(tile_mn: tuple[int, int]) -> bool:
    """Test JIT compilation for a specific tile configuration."""
    import torch

    # Clear JIT cache for this tile to force recompile
    import shutil
    import os
    cache_dir = os.path.expanduser("~/.cache/flashinfer/")
    if os.path.exists(cache_dir):
        # Don't clear the whole cache, just let JIT handle it
        pass

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    
    if backend not in ("120", "121"):
        print(f"[SKIP] Not an SM120/121 device (got SM{backend})")
        return True

    print(f"\n{'='*60}")
    print(f"Testing tile_mn={tile_mn} on SM{backend}")
    print(f"{'='*60}")

    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    start = time.time()
    try:
        mod = get_cutlass_fused_moe_module(
            backend=backend,
            use_fast_build=True,
            tile_mn=tile_mn,
        )
        elapsed = time.time() - start
        print(f"[OK] tile_mn={tile_mn} compiled in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] tile_mn={tile_mn} failed after {elapsed:.1f}s")
        print(f"  Error: {e}")
        # Print more details for compile errors
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test configurations in order
    # SM120 MXFP4 TMA constraints: M >= 64, N >= 32 (padded to 128 internally for SF)
    configs = [
        (128, 128),  # Standard: 128x128 (default)
        (128, 64),   # Standard: 128x64 (smaller N)
        (128, 32),   # Standard: 128x32 (smallest N per CUTLASS builder)
        (64, 128),   # M=64 (tests TileShape_SFA padding)
        (64, 64),    # Both M and N small
        (64, 32),    # Smallest practical tile for decode
    ]
    
    results = {}
    for tile_mn in configs:
        ok = test_tile_compile(tile_mn)
        results[tile_mn] = ok
        if not ok:
            print(f"\n[STOP] tile_mn={tile_mn} failed, stopping further tests")
            break
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    for tile_mn, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {tile_mn}: {status}")
    
    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
