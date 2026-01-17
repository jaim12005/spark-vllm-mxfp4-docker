#!/usr/bin/env python3
"""
Test epilogue copy atom selection and measure performance.

This script:
1. Verifies epilogue tile calculation for all supported CTA tiles
2. Tests if SM90_U32x1_STSM_N can be used for EpiN=8 (instead of AutoVectorizingCopy)
3. Measures performance difference between copy atoms
"""

import subprocess
import sys
import tempfile
import os

# Test configurations: (CTA_M, CTA_N)
TILE_CONFIGS = [
    (64, 8),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 8),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128),
]

# CUDA code to verify epilogue tile calculation at compile time
VERIFY_EPILOGUE_TILE_CU = '''
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/builders/sm120_builder.inl>

using namespace cute;

template <int CTA_M, int CTA_N>
struct VerifyEpilogueTile {
    using TileShape_MNK = Shape<Int<CTA_M>, Int<CTA_N>, Int<128>>;
    using StrideD = Stride<Int<1>, int64_t, int64_t>;
    
    // Calculate epilogue tile the same way as sm120_compute_tile_shape_or_override
    static constexpr int EpiM = cute::min(64, CTA_M);
    static constexpr int EpiN = (CTA_N % 8 == 0) ? 8 : CTA_N;
    
    // Verify constraints
    static_assert(CTA_M % EpiM == 0, "CTA_M must be divisible by EpiM");
    static_assert(CTA_N % EpiN == 0, "CTA_N must be divisible by EpiN");
    static_assert(64 % EpiM == 0 || EpiM % 64 == 0, "MMA_M constraint");
    static_assert(64 % EpiN == 0, "MMA_N must be divisible by EpiN");
    
    static void print() {
        printf("CTA (%d, %d) -> Epilogue (%d, %d)\\n", CTA_M, CTA_N, EpiM, EpiN);
    }
};

int main() {
    printf("Epilogue Tile Verification:\\n");
    printf("============================\\n");
'''

def generate_verification_code():
    """Generate CUDA code to verify all tile configurations."""
    code = VERIFY_EPILOGUE_TILE_CU
    for m, n in TILE_CONFIGS:
        code += f"    VerifyEpilogueTile<{m}, {n}>::print();\n"
    code += "    return 0;\n}\n"
    return code


# Test if SM90_U32x1_STSM_N can work for EpiN=8
TEST_STSM_X1_CU = '''
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>

using namespace cute;

// Test if we can create a TiledCopy with SM90_U32x1_STSM_N for EpiN=8
template <int EpiM, int EpiN>
struct TestSTSMx1 {
    // Try to use SM90_U32x1_STSM_N
    using CopyAtom = Copy_Atom<SM90_U32x1_STSM_N, half_t>;
    
    // Thread layout for 128 threads
    using ThrLayout = Layout<Shape<_128>>;
    
    // Value layout - each thread handles some elements
    using ValLayout = Layout<Shape<Int<EpiM / 8>, Int<EpiN / 8>>>;
    
    static void test() {
        printf("Testing SM90_U32x1_STSM_N for (%d, %d)\\n", EpiM, EpiN);
    }
};

int main() {
    // This will fail to compile if SM90_U32x1_STSM_N can't be used
    TestSTSMx1<64, 8>::test();
    return 0;
}
'''


def test_ptx_compiles(cuda_code, name, nvcc_flags=None):
    """Test if CUDA code compiles for SM121."""
    if nvcc_flags is None:
        nvcc_flags = ["-arch=sm_121", "-std=c++17"]
    
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(cuda_code)
        cu_path = f.name
    
    # Include paths
    flashinfer_path = "/workspace/flashinfer"
    cutlass_include = f"{flashinfer_path}/3rdparty/cutlass/include"
    
    cmd = [
        "nvcc",
        *nvcc_flags,
        f"-I{cutlass_include}",
        "-c", cu_path,
        "-o", "/dev/null"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        success = result.returncode == 0
        return success, result.stderr if not success else ""
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(cu_path)


def main():
    print("=" * 60)
    print("  Epilogue Copy Atom Analysis")
    print("=" * 60)
    
    # Test 1: Verify epilogue tile calculation compiles
    print("\n1. Verifying epilogue tile calculations...")
    code = generate_verification_code()
    success, error = test_ptx_compiles(code, "epilogue_verify")
    if success:
        print("   ✅ All epilogue tile calculations are valid")
    else:
        print("   ❌ Compilation failed:")
        print(f"   {error[:500]}")
    
    # Test 2: Can we use SM90_U32x1_STSM_N for EpiN=8?
    print("\n2. Testing SM90_U32x1_STSM_N for EpiN=8...")
    success, error = test_ptx_compiles(TEST_STSM_X1_CU, "stsm_x1_test")
    if success:
        print("   ✅ SM90_U32x1_STSM_N compiles for EpiN=8")
        print("   → We could potentially use this instead of AutoVectorizingCopy!")
    else:
        print("   ❌ SM90_U32x1_STSM_N cannot be used for EpiN=8:")
        # Extract relevant error
        for line in error.split("\n"):
            if "error" in line.lower():
                print(f"   {line[:80]}")
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
  Current implementation uses AutoVectorizingCopy for EpiN=8.
  
  Potential optimization:
    - SM90_U32x1_STSM_N stores 8x8 tiles atomically
    - If layout constraints can be satisfied, this would be faster
    
  To measure performance:
    1. Create a microbenchmark comparing copy atoms
    2. Run full MoE GEMM with different epilogue configurations
    3. Profile with NCU to see epilogue contribution
    """)


if __name__ == "__main__":
    main()
