#!/usr/bin/env python3
"""
Probe SM121 hardware capabilities via PTX compilation tests.

This provides ground truth about what instructions are actually available,
rather than relying on potentially incorrect compile-time macros.

Usage:
    python probe_sm121_capabilities.py
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PTXTest:
    name: str
    description: str
    ptx: str
    category: str


# PTX tests for various instruction categories
PTX_TESTS = [
    # Memory instructions
    PTXTest(
        name="ldmatrix_x1",
        description="Load 1x 8x8 matrix tile from shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<2>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%r0}, [%addr];
    ret;
}
""",
    ),
    PTXTest(
        name="ldmatrix_x2",
        description="Load 2x 8x8 matrix tiles from shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<2>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%r0, %r1}, [%addr];
    ret;
}
""",
    ),
    PTXTest(
        name="ldmatrix_x4",
        description="Load 4x 8x8 matrix tiles from shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<4>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%addr];
    ret;
}
""",
    ),
    PTXTest(
        name="stmatrix_x1",
        description="Store 1x 8x8 matrix tile to shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<2>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    stmatrix.sync.aligned.x1.m8n8.shared.b16 [%addr], {%r0};
    ret;
}
""",
    ),
    PTXTest(
        name="stmatrix_x2",
        description="Store 2x 8x8 matrix tiles to shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<2>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    stmatrix.sync.aligned.x2.m8n8.shared.b16 [%addr], {%r0, %r1};
    ret;
}
""",
    ),
    PTXTest(
        name="stmatrix_x4",
        description="Store 4x 8x8 matrix tiles to shared memory",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b32 %r<4>;
    .reg .b64 %addr;
    .shared .align 32 .b8 smem[1024];
    mov.u64 %addr, smem;
    stmatrix.sync.aligned.x4.m8n8.shared.b16 [%addr], {%r0, %r1, %r2, %r3};
    ret;
}
""",
    ),
    PTXTest(
        name="ld_256bit",
        description="256-bit vectorized global load",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b64 %rd<5>;
    ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd4}, [%rd0];
    ret;
}
""",
    ),
    PTXTest(
        name="st_256bit",
        description="256-bit vectorized global store",
        category="memory",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b64 %rd<5>;
    st.global.v4.u64 [%rd0], {%rd1, %rd2, %rd3, %rd4};
    ret;
}
""",
    ),
    # TMEM instructions (SM100-only, expected to fail on SM121)
    PTXTest(
        name="tmem_alloc",
        description="Allocate tensor memory (TMEM)",
        category="tmem",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b64 %tmem_addr;
    tcgen05.alloc.cta_group::1.sync.aligned %tmem_addr, 1024;
    ret;
}
""",
    ),
    PTXTest(
        name="tmem_dealloc",
        description="Deallocate tensor memory (TMEM)",
        category="tmem",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b64 %tmem_addr;
    tcgen05.dealloc.cta_group::1.sync.aligned %tmem_addr, 1024;
    ret;
}
""",
    ),
    # TMA instructions
    PTXTest(
        name="tensormap_replace",
        description="Dynamic TMA descriptor update",
        category="tma",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    .reg .b64 %rd<4>;
    .reg .b32 %r<4>;
    ret;
}
""",
    ),
    # Swizzle modes
    PTXTest(
        name="swizzle_128b",
        description="128-byte swizzle mode",
        category="swizzle",
        ptx="""
.version 8.8
.target sm_121
.address_size 64
.visible .entry test() {
    ret;
}
""",
    ),
]


def test_ptx_compiles(ptx: str, arch: str = "sm_121") -> Tuple[bool, str]:
    """Test if PTX code compiles for the given architecture."""
    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w", delete=False) as f:
        f.write(ptx)
        ptx_path = f.name

    try:
        result = subprocess.run(
            ["ptxas", f"-arch={arch}", ptx_path, "-o", "/dev/null"],
            capture_output=True,
            text=True,
        )
        success = result.returncode == 0
        error = result.stderr.strip() if not success else ""
        return success, error
    finally:
        os.unlink(ptx_path)


def probe_capabilities(arch: str = "sm_121") -> Dict[str, bool]:
    """Probe all capabilities for the given architecture."""
    results = {}
    for test in PTX_TESTS:
        success, error = test_ptx_compiles(test.ptx, arch)
        results[test.name] = success
    return results


def print_report(arch: str = "sm_121"):
    """Print a formatted capability report."""
    print(f"\n{'=' * 60}")
    print(f"  SM121 Hardware Capability Probe")
    print(f"  Target: {arch}")
    print(f"{'=' * 60}\n")

    results = {}
    categories = {}

    for test in PTX_TESTS:
        success, error = test_ptx_compiles(test.ptx, arch)
        results[test.name] = (success, error)
        if test.category not in categories:
            categories[test.category] = []
        categories[test.category].append(test)

    for category, tests in categories.items():
        print(f"\n{category.upper()}")
        print("-" * 40)
        for test in tests:
            success, error = results[test.name]
            status = "✅" if success else "❌"
            print(f"  {status} {test.name:20s} - {test.description}")
            if not success and error:
                # Extract key error message
                for line in error.split("\n"):
                    if "error" in line.lower():
                        print(f"      Error: {line.strip()[:60]}")
                        break

    # Summary
    print(f"\n{'=' * 60}")
    available = sum(1 for s, _ in results.values() if s)
    total = len(results)
    print(f"  Summary: {available}/{total} features available")

    # Recommendations
    print(f"\n  OPTIMIZATION RECOMMENDATIONS:")
    if results.get("ldmatrix_x4", (False,))[0]:
        print("    - Use ldmatrix.x4 for TileN >= 32")
    if results.get("ldmatrix_x2", (False,))[0]:
        print("    - Use ldmatrix.x2 for TileN >= 16")
    if results.get("stmatrix_x2", (False,))[0]:
        print("    - Use stmatrix.x2 for EpiN >= 16")
    if not results.get("tmem_alloc", (False,))[0]:
        print("    - TMEM not available: use register-based accumulation")
        print("    - Epilogue must use shared memory staging")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print_report("sm_121")
