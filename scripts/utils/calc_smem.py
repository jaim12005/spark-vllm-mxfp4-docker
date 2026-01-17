#!/usr/bin/env python3
"""Calculate shared memory usage for SM120 MXFP4 tile configurations."""

import math

# From cutlass/arch/arch.h
SM120_SMEM = 101376  # SM120 (GB10) - only ~99 KB!
SM100_SMEM = 232448  # SM100 (Blackwell) - ~227 KB

def calc_stage_bytes(M, N, K):
    """Calculate bytes per pipeline stage for a tile configuration."""
    # A tensor: FP8 (8 bits = 1 byte per element)
    a_bytes = M * K
    
    # B tensor: FP4 (4 bits = 0.5 bytes per element)
    b_bytes = N * K // 2
    
    # Scale Factor A: padded to 128 in M dimension
    TileM_SFA = math.ceil(M / 128) * 128
    sfa_bytes = (TileM_SFA // 128) * math.ceil(K / 32) * 4
    
    # Scale Factor B: padded to 128 in N dimension
    TileN_SFB = math.ceil(N / 128) * 128
    sfb_bytes = (TileN_SFB // 128) * math.ceil(K / 32) * 4
    
    # Pipeline overhead (barriers, etc.)
    pipeline_bytes = 64
    
    stage_bytes = a_bytes + b_bytes + pipeline_bytes + sfa_bytes + sfb_bytes
    return stage_bytes, a_bytes, b_bytes, sfa_bytes, sfb_bytes

def main():
    print("=" * 80)
    print(f"SM120 (GB10) Shared Memory: {SM120_SMEM:,} bytes (~99 KB)")
    print(f"SM100 (Blackwell) Shared Memory: {SM100_SMEM:,} bytes (~227 KB)")
    print("=" * 80)
    print()
    print("Pipeline requires >= 2 stages for double-buffering (hide memory latency)")
    print()
    print("Tile (M,N,K)     A(FP8)   B(FP4)    SFA    SFB   Stage  SM120  Status")
    print("-" * 75)
    
    for m, n, k in [(64,128,128), (64,256,128), (128,64,128), (128,128,128), (128,256,128)]:
        stage, a, b, sfa, sfb = calc_stage_bytes(m, n, k)
        stages = SM120_SMEM // stage
        status = "OK" if stages >= 2 else "FAIL"
        print(f"({m:3},{n:3},{k:3}) {a:8,} {b:8,} {sfa:6,} {sfb:6,} {stage:7,} {stages:5}  {status}")
    
    print()
    s = calc_stage_bytes(128, 256, 128)[0]
    print(f"(128, 256) needs {s:,} bytes/stage but SM120 only has {SM120_SMEM:,} bytes total")
    print(f"That allows only {SM120_SMEM // s} stage, but 2+ are required!")
    print()
    print(f"On SM100: {SM100_SMEM:,} bytes / {s:,} = {SM100_SMEM // s} stages (would work)")

if __name__ == "__main__":
    main()
