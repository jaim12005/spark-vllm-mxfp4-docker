# GPT-OSS-120B with MXFP4 on DGX Spark (SM121/GB10)

**Fastest gpt-oss-120b inference on DGX Spark** - 57-60 tok/s decode, beating SGLang and llama.cpp.

## Benchmark Results

| Context | Prefill (t/s) | Decode tg32 (t/s) | Decode tg128 (t/s) |
|---------|---------------|-------------------|---------------------|
| **Short (512)** | 1,854 | **60.02** | **60.07** |
| **Medium (2048)** | 4,573 | **59.36** | **59.47** |
| **Long (8192)** | 6,628 | **57.52** | **57.81** |

### Key Observations

- ✅ Decode consistently **57-60 tok/s** across all context lengths
- ✅ Prefill scales well: 1.8K → 4.6K → 6.6K t/s as batch size increases
- ✅ Long context (8K) only ~3% decode slowdown vs short context

### vs Competitors

| Engine | Decode (t/s) | Status |
|--------|--------------|--------|
| SGLang | 52 | ✅ Beat by 10-15% |
| llama.cpp | 58 | ✅ Beat at short/medium context |
| **vLLM (this)** | **57-60** | **Winner** |

See the [discussion on NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651/18) for more details.

---

## Quick Start

### Build

```bash
docker build -t vllm-mxfp4-spark .
```

### Run with Docker Compose (Recommended)

```bash
# Start the server
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Run with Docker

```bash
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./.cache/flashinfer:/root/.cache/flashinfer \
    vllm-mxfp4-spark
```

The volume mounts persist the model cache and JIT-compiled kernels between runs.

---

## Replicate Benchmark Results

Once the server is running, run the benchmark:

```bash
docker compose exec vllm-mxfp4 llama-benchy \
    --base-url http://localhost:8000/v1 \
    --model gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --pp 512 2048 8192 \
    --tg 32 128 \
    --runs 5
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Configuration

The server starts with optimized settings for gpt-oss-120b:

| Setting | Value |
|---------|-------|
| Quantization | MXFP4 with CUTLASS backend |
| Attention | FlashInfer with FP8 KV cache |
| Max context | 131072 tokens |
| GPU memory | 70% utilization |

See `docker-compose.yml` for full configuration.

---

## Development

For development with local FlashInfer/vLLM repos:

```bash
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml exec dev bash
```

---

## Documentation

- [AGENTS.md](AGENTS.md) - Project context and AI assistant guide
- [docs/reference/](docs/reference/) - Technical deep dive (SM121 architecture, CUTLASS details)
- [docs/porting/](docs/porting/) - Feature implementation documentation

---

## Pinned Versions

| Component | SHA | Repository |
|-----------|-----|------------|
| vLLM | `a461bc39` | [christopherowen/vllm](https://github.com/christopherowen/vllm/tree/mxfp4_v2) |
| FlashInfer | `1660ee8d` | [christopherowen/flashinfer](https://github.com/christopherowen/flashinfer/tree/mxfp4_v2) |
| CUTLASS | `11af7f02` | [christopherowen/cutlass](https://github.com/christopherowen/cutlass/tree/mxfp4_v2) |
