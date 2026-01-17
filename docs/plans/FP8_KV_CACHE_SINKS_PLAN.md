# FP8 E4M3 KV Cache + Attention Sinks

## Overview

Enable FP8 E4M3 KV cache with attention sinks for gpt-oss-120b on SM121. This plan first validates that the kernel infrastructure supports FP8 KV, then relaxes the guard with proper validation and correctness tests.

**Expected Impact:** +1.8 to +3.7 tok/s (48.9 -> 50.7-52.6 tok/s)

---

## Validation: Kernel Infrastructure Already Supports FP8 KV

**Finding:** The sink kernel path IS instantiated for FP8 KV cache. Here's the evidence:

### 1. JIT System Maps FP8 Types Correctly

From `flashinfer/jit/utils.py` lines 33-44:

```python
dtype_map = {
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",  # Supported
    torch.float8_e5m2: "__nv_fp8_e5m2",
    ...
}
```

### 2. Sink Module Generator Passes dtype_kv to Template

From `flashinfer/jit/attention/modules.py` lines 1114-1118:

```python
def gen_batch_prefill_attention_sink_module(..., dtype_kv, ...):
    return gen_customize_batch_prefill_module(
        backend, uri,
        dtype_q, dtype_kv,  # <-- FP8 kv passes through
        dtype_o, dtype_idx, ...
        fp8_enabled=False,  # Only gates FP8 QUERY tensor cores
    )
```

### 3. Kernel Templates Are Dtype-Generic

From `csrc/batch_prefill_customize_config.jinja`:

```cpp
using DTypeKV = {{ dtype_kv }};  // Templated, works with any dtype in dtype_map
struct PagedParams {
    paged_kv_t<DTypeKV, IdType> paged_kv;  // Generic template
};
```

### 4. paged_kv_t Template Is Fully Generic

From `include/flashinfer/page.cuh`:

```cpp
template <typename DType, typename IdType>  // Works with FP8
struct paged_kv_t {
    DType* k_data;
    DType* v_data;
    ...
};
```

### 5. Existing FP8 KV Tests Confirm Infrastructure Works

From `tests/attention/test_batch_decode_kernels.py`:

```python
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])  # FP8 tested!
def test_batch_decode_with_paged_kv_cache(...):
    wrapper.plan(..., data_type=kv_dtype, q_data_type=q_dtype)  # Works
```

### 6. The `fp8_enabled=False` Flag Only Gates FP8 Query

From `flashinfer/jit/attention/modules.py` lines 987-990:

```python
# use `fp8_enabled` flag to use separate kernel template
# this is used for fp8 tensor core computation
# KV-only quant is not influenced by this flag  <-- EXPLICIT COMMENT
fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]
```

**Conclusion:** The Python check at lines 1073-1078 is overly conservative. The kernel infrastructure already supports FP8 KV with BF16 Q.

---

## Code Changes

### Change 1: FlashInfer decode.py - PRIMARY

**File:** `~/projects/flashinfer/flashinfer/decode.py`
**Location:** Lines 1066-1093

This is the **primary change**. Sinks are decode-oriented.

**Current:**

```python
if use_sinks:
    if self._backend != "fa2":
        raise NotImplementedError(...)
    # Check for FP8 dtypes - sink module is compiled with fp8_enabled=False
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if q_data_type in fp8_dtypes or kv_data_type in fp8_dtypes:
        raise NotImplementedError(
            "Attention sinks are not supported with FP8 inputs. "
            "Use BF16/FP16 inputs or set use_sinks=False."
        )
    self._cached_module = get_batch_prefill_attention_sink_module(...)
```

**Proposed:**

```python
if use_sinks:
    if self._backend != "fa2":
        raise NotImplementedError(...)
    
    # Supported query dtypes: BF16, FP16
    supported_q_dtypes = (torch.bfloat16, torch.float16)
    if q_data_type not in supported_q_dtypes:
        raise NotImplementedError(
            f"Attention sinks require BF16/FP16 query, got {q_data_type}."
        )
    
    # Supported KV cache dtypes: BF16, FP8 E4M3 (including fnuz variant)
    supported_kv_dtypes = (torch.bfloat16, torch.float8_e4m3fn)
    if hasattr(torch, "float8_e4m3fnuz"):
        supported_kv_dtypes = supported_kv_dtypes + (torch.float8_e4m3fnuz,)
    if kv_data_type not in supported_kv_dtypes:
        raise NotImplementedError(
            f"Attention sinks support BF16 or FP8 E4M3 KV cache, got {kv_data_type}."
        )
    
    # Log the selected kernel for verification
    import logging
    logging.info(f"Attention sink kernel: kv_dtype={kv_data_type}, q_dtype={q_data_type}")
    
    self._cached_module = get_batch_prefill_attention_sink_module(...)
```

### Change 2: FlashInfer prefill.py - DEFERRED

**File:** `~/projects/flashinfer/flashinfer/prefill.py`
**Location:** Lines 1960-1987

**Rationale:** Sinks are decode-oriented. Prefill may use a different attention kernel entirely.
Only change prefill.py if:
1. We confirm prefill actually uses sinks for gpt-oss-120b
2. We have test coverage for prefill-with-sinks + FP8 KV

**Investigation needed first:**
```python
# Check if prefill uses sinks by examining the code path
# Look for: self._use_sinks in prefill wrapper's run() method
```

---

## Test Plan

### Test 1: Path Coverage - Assert Sinks Are Actually Used

**Critical:** Verify the sink kernel path is exercised, not a fallback.

```python
def test_attention_sink_fp8_path_verification():
    """Verify FP8 KV cache uses the sink kernel, not a fallback."""
    import os
    os.environ["FLASHINFER_LOGLEVEL"] = "3"  # Enable verbose logging
    
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
    wrapper.plan(
        ...,
        kv_data_type=torch.float8_e4m3fn,
        q_data_type=torch.bfloat16,
        use_sinks=True,
    )
    
    # Check the compiled module URI contains both sink and e4m3
    module = wrapper._cached_module
    assert module is not None, "No cached module - sink path not compiled"
    
    # The URI should be: batch_prefill_with_attention_sink_...dtype_kv_e4m3_...
    # This confirms:
    # 1. Sink variant was selected (not DefaultAttention)
    # 2. FP8 E4M3 KV dtype was passed to template
    
    # Run and verify output is not NaN/Inf
    output = wrapper.run(q, kv_cache, sinks=sink_values)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
```

### Test 2: Correctness - Greedy Decode Token Match

**Better than 5% tolerance:** Compare top-1 token match rate with temperature=0.

```python
def test_attention_sink_fp8_greedy_match():
    """Compare greedy decode tokens between BF16 and FP8 KV cache."""
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In 1969, humans first",
    ]
    
    for prompt in prompts:
        # Run with BF16 KV cache
        tokens_bf16 = greedy_decode(prompt, kv_dtype=torch.bfloat16, max_tokens=32)
        
        # Run with FP8 KV cache  
        tokens_fp8 = greedy_decode(prompt, kv_dtype=torch.float8_e4m3fn, max_tokens=32)
        
        # Token match rate should be very high (>95%)
        match_rate = (tokens_bf16 == tokens_fp8).float().mean()
        assert match_rate > 0.95, f"Token match rate {match_rate:.1%} too low for prompt: {prompt}"
```

### Test 3: Attention Output Tensor Comparison

```python
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 1024])  # Include longer for bandwidth visibility
@pytest.mark.parametrize("num_qo_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [8])  # GQA ratio matching gpt-oss-120b
@pytest.mark.parametrize("head_dim", [64])
def test_attention_sink_fp8_kv_correctness(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim):
    """Compare FP8 KV cache output against BF16 KV cache reference."""
    q_dtype = torch.bfloat16
    device = "cuda:0"
    
    # Generate inputs
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype, device=device)
    kv_bf16 = torch.randn(batch_size, 2, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)
    sink = torch.rand(num_qo_heads, dtype=torch.float32, device=device) * 5
    
    # Run with BF16 KV
    wrapper_bf16 = BatchDecodeWithPagedKVCacheWrapper(...)
    wrapper_bf16.plan(..., kv_data_type=torch.bfloat16, use_sinks=True)
    output_bf16 = wrapper_bf16.run(q, kv_bf16, sinks=sink)
    
    # Run with FP8 KV
    wrapper_fp8 = BatchDecodeWithPagedKVCacheWrapper(...)
    wrapper_fp8.plan(..., kv_data_type=torch.float8_e4m3fn, use_sinks=True)
    output_fp8 = wrapper_fp8.run(q, kv_fp8, sinks=sink)
    
    # Use absolute + relative tolerance appropriate for FP8 storage
    # FP8 E4M3 has ~0.1% relative error for typical values
    torch.testing.assert_close(
        output_fp8, output_bf16,
        rtol=1e-2,  # 1% relative
        atol=1e-3,  # Absolute for near-zero values
    )
```

### Test 4: Error Cases - Unsupported Configs Throw NotImplementedError

```python
def test_attention_sink_rejects_fp8_query():
    """FP8 query with sinks should throw NotImplementedError."""
    with pytest.raises(NotImplementedError, match="require BF16/FP16 query"):
        wrapper.plan(..., q_data_type=torch.float8_e4m3fn, use_sinks=True)

def test_attention_sink_rejects_unsupported_kv_dtype():
    """Unsupported KV cache dtype with sinks should throw NotImplementedError."""
    with pytest.raises(NotImplementedError, match="FP8 E4M3 KV cache"):
        wrapper.plan(..., kv_data_type=torch.float32, use_sinks=True)
```

---

## Before/After Profiling

### Step 1: Baseline (BF16 KV cache)

```bash
# Start server with BF16 KV cache (current default)
docker exec vllm-dev bash -c 'export PYTHONPATH=/workspace/flashinfer:/workspace/vllm && \
  cd /workspace/vllm && \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --kv-cache-dtype auto'

# Benchmark baseline
llama-benchy \
  --model gpt-oss-120b \
  --endpoint http://localhost:8000 \
  --prompt-length 2048 \
  --output-lengths 32,128 \
  --num-requests 10

# Record: expected ~48.9 tok/s decode
```

### Step 2: Apply Changes and Clear Cache

```bash
# Clear ENTIRE FlashInfer cache to avoid stale kernels
# This is slower but removes ambiguity when validating dtype path changes
docker exec vllm-dev rm -rf /root/.cache/flashinfer/*/cached_ops/*

# Alternative: disable cache entirely for this run
# export FLASHINFER_JIT_CACHE=0
```

### Step 3: FP8 KV Cache Benchmark

```bash
# Start server with FP8 KV cache
docker exec vllm-dev bash -c 'export PYTHONPATH=/workspace/flashinfer:/workspace/vllm && \
  cd /workspace/vllm && \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --kv-cache-dtype fp8'

# Benchmark with FP8 KV
llama-benchy \
  --model gpt-oss-120b \
  --endpoint http://localhost:8000 \
  --prompt-length 2048 \
  --output-lengths 32,128 \
  --num-requests 10

# Expected: ~50.7-52.6 tok/s decode (+1.8 to +3.7 tok/s)
```

### Step 4: Correctness Check

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-oss-120b", "prompt": "The capital of France is", "max_tokens": 32}'
```

---

## Expected Performance Impact

**Conservative estimate:**

- Attention is ~25% of decode step time for MoE models
- FP8 KV halves KV bandwidth, giving ~15-30% attention speedup
- End-to-end: 0.25 x 0.15-0.30 = **3.75-7.5%** improvement
- On 48.9 tok/s baseline: **+1.8 to +3.7 tok/s -> 50.7-52.6 tok/s**

Hitting 52 tok/s is plausible; 56 is unlikely without additional optimizations.

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Silent fallback to non-sink kernel | Path coverage test verifies JIT URI contains `attention_sink` + `e4m3` |
| FP8 KV accuracy drift | Greedy decode token match test (>95% match rate) |
| Stale cached kernels | Clear entire FlashInfer cache, not just sink ops |
| E4M3fnuz variant not handled | `hasattr(torch, "float8_e4m3fnuz")` guard |
| Prefill accidentally changed | Defer prefill.py changes until decode.py is validated |

---

## Todos

### Phase 1: Decode Path Only

- [ ] Run baseline benchmark with BF16 KV cache (llama-benchy)
- [ ] Update decode.py ONLY: block FP8 query, allow BF16/FP8-E4M3 KV
- [ ] Add logging to confirm sink kernel selection
- [ ] Clear ENTIRE FlashInfer JIT cache
- [ ] Run FP8 KV cache benchmark with llama-benchy
- [ ] Add path verification test (assert sink kernel URI contains `attention_sink` + `e4m3`)
- [ ] Add greedy decode token match test (>95% match rate)
- [ ] Add attention output tensor comparison test (rtol=1e-2, atol=1e-3)
- [ ] Add error case tests (FP8 query, E5M2 KV)
- [ ] Document before/after results in BENCHMARK_RESULTS.md

### Phase 2: Prefill Path (If Needed)

- [ ] Investigate if prefill uses sinks for gpt-oss-120b
- [ ] If yes: update prefill.py with same pattern
- [ ] Add prefill-with-sinks + FP8 KV test coverage
