#!/usr/bin/env python3
"""Analyze CUTLASS shared memory calculation for SM120 MXFP4."""

# Constants from CUTLASS
SM120_SMEM = 101376
CARVEOUT = 0

def analyze_tile(tile_m, tile_n, tile_k=128):
    """Replicate CUTLASS stage count calculation."""
    
    print(f"\nCUTLASS Stage Calculation for ({tile_m}, {tile_n}, {tile_k})")
    print("=" * 60)
    
    # SmemAllocTypeA and SmemAllocTypeB are both uint8_t for MXFP4
    # This is how CUTLASS handles sub-byte types in shared memory
    a_bits = 8  # uint8_t
    b_bits = 8  # uint8_t (FP4 packed into bytes)
    
    # A tensor: M x K elements, each stored as uint8_t
    a_bytes = (a_bits * tile_m * tile_k) // 8
    
    # B tensor: N x K elements, each stored as uint8_t
    # NOTE: For FP4, the TileN in the shape already accounts for packing!
    # The GEMM shape N=256 means 256 FP4 outputs, stored as 128 bytes
    b_bytes = (b_bits * tile_n * tile_k) // 8
    
    # Scale factor layouts (approximate)
    # SFA: ceil(M/128) blocks, each with (K/32) scale factors, 4 bytes per group
    tile_m_padded = ((tile_m + 127) // 128) * 128
    tile_n_padded = ((tile_n + 127) // 128) * 128
    sfa_bytes = (tile_m_padded // 128) * (tile_k // 32) * 4
    sfb_bytes = (tile_n_padded // 128) * (tile_k // 32) * 4
    
    # Pipeline overhead (barriers, etc.)
    pipeline_bytes = 64
    
    stage_bytes = a_bytes + b_bytes + sfa_bytes + sfb_bytes + pipeline_bytes
    stages = (SM120_SMEM - CARVEOUT) // stage_bytes
    
    print(f"\nInputs:")
    print(f"  SmemAllocTypeA = uint8_t (a_bits = {a_bits})")
    print(f"  SmemAllocTypeB = uint8_t (b_bits = {b_bits})")
    
    print(f"\nPer-stage storage:")
    print(f"  A tensor: {tile_m} x {tile_k} x {a_bits}/8 = {a_bytes:,} bytes")
    print(f"  B tensor: {tile_n} x {tile_k} x {b_bits}/8 = {b_bytes:,} bytes")
    print(f"  SFA: {sfa_bytes:,} bytes")
    print(f"  SFB: {sfb_bytes:,} bytes")
    print(f"  Pipeline: ~{pipeline_bytes} bytes")
    print(f"  TOTAL: {stage_bytes:,} bytes per stage")
    
    print(f"\nStage calculation:")
    print(f"  ({SM120_SMEM:,} - {CARVEOUT}) / {stage_bytes:,} = {stages}")
    
    if stages < 2:
        print(f"\n>>> FAILS: Only {stages} stage fits, need >= 2 <<<")
    else:
        print(f"\n>>> OK: {stages} stages fit <<<")
    
    return stages

# Analyze the failing tile
print("=" * 70)
print("WHY DOES (128, 256) FAIL?")
print("=" * 70)

stages_128_256 = analyze_tile(128, 256)

print("\n" + "=" * 70)
print("COMPARISON WITH WORKING TILES")
print("=" * 70)

for m, n in [(64, 256), (128, 128), (128, 64)]:
    analyze_tile(m, n)

print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)
print("""
The issue is that CUTLASS uses uint8_t (8 bits) as the SmemAllocType
for BOTH FP8 (A tensor) and FP4 (B tensor). This seems counterintuitive
since FP4 only needs 4 bits per element.

However, this is intentional! In shared memory:
- FP4 data is stored in packed uint8_t format (2 FP4 per byte)
- But the TileN dimension in TileShape_MNK represents FP4 elements, not bytes
- So TileN=256 means 256 FP4 elements, which SHOULD be 128 bytes

Let's check if this is a bug in the calculation...
""")

# The real question: does TileN represent FP4 elements or bytes?
print("If TileN=256 represents 256 FP4 ELEMENTS (128 bytes):")
print(f"  B tensor should be: 256 * 128 / 2 = {256 * 128 // 2:,} bytes (FP4 packing)")

print("\nIf TileN=256 represents 256 uint8_t (256 bytes, 512 FP4 elements):")
print(f"  B tensor would be: 256 * 128 = {256 * 128:,} bytes")

print("\nThe CUTLASS calculation uses the SECOND interpretation (uint8_t).")
print("This may be intentional for alignment or may be a conservative estimate.")
