# SGLang Analysis (SM121 / gpt-oss-120b)

**Status**: Reference data collected, analysis pending

## Overview

SGLang achieves ~52 tok/s decode on SM121 with gpt-oss-120b, approximately 1.8x faster than vLLM baseline (29 tok/s).

## Reference Benchmark

```
Engine: SGLang
Model: gpt-oss-120b (MXFP4)
GPU: NVIDIA GB10 (SM121)

Output token throughput: 52.37 tok/s
TTFT: 49.87 ms  
TPOT: 18.83 ms (time per output token)
```

## Key Differences from vLLM

| Aspect | SGLang | vLLM | Notes |
|--------|--------|------|-------|
| Scheduler | ? | V1 busy-loop | |
| MoE Implementation | ? | FlashInfer CUTLASS | |
| Attention Backend | ? | FlashInfer FA2 | |
| CUDA Graphs | ? | Supported | |

## Questions to Investigate

1. What MoE kernel does SGLang use on SM121?
2. How does their scheduler differ?
3. What's their TTFT vs ours?
4. Do they use speculative decoding by default?

## Profiling TODO

- [ ] Profile SGLang decode with nsys
- [ ] Compare kernel-level breakdown
- [ ] Identify architectural differences
- [ ] Document optimization techniques we can adopt
