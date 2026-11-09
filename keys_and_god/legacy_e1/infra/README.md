# Local LLM server (vLLM) for the swarm

- **Endpoint:** `http://localhost:8000/v1` (OpenAI-compatible; any `api_key`, e.g. `"EMPTY"`)
- **Model name:** `Qwen/Qwen3-8B` (bf16, max context 16384, `--gpu-memory-utilization 0.75`)
- **Env:** conda `swarm` (Python 3.12, vllm 0.33.0, torch 2.14.0+cu130, sm_120 supported)
- **Weights:** `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` (standard HF cache; same checkpoint works for nnsight/transformers)

## Commands

```bash
cd $SWARM_ROOT/infra
./serve.sh            # starts in background, waits for /health (~30s warm, ~2 min first time), logs -> vllm.log
./stop.sh             # kills the server's process group, frees the GPU
PYTHONNOUSERSITE=1 $PY bench.py            # 32 concurrent, 256 tok
PYTHONNOUSERSITE=1 $PY bench.py -n 256 -c 128 --max-tokens 512
```

serve.sh env overrides: `PORT`, `HOST` (default 127.0.0.1), `GPU_UTIL`, `MAX_LEN`, `MAX_SEQS` (256),
`REASONING_PARSER` (default `qwen3`; `REASONING_PARSER= ./serve.sh` disables), `WAIT=0`.
A different model: `./serve.sh Qwen/Qwen3-4B` (must already be in the HF cache; `HF_HUB_OFFLINE=1` is set).

## Client usage

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = c.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # turn Qwen3 thinking OFF
)
print(r.choices[0].message.content)
# with thinking ON (the default), the chain of thought is in r.choices[0].message.reasoning (not in .content)
```

curl: add `"chat_template_kwargs": {"enable_thinking": false}` at the top level of the JSON body.

## Measured (RTX 5090, thinking off, short prompts)

| requests | concurrency | max_tokens | output tok/s | p50 latency |
|---|---|---|---|---|
| 32  | 32  | 256 | ~2,480 | 3.26 s |
| 128 | 64  | 512 | ~3,890 | 6.32 s |
| 256 | 128 | 512 | ~6,290 | 7.92 s |

## Quirks

- **Thinking is ON by default** in Qwen3. Pass `enable_thinking: false` per request (above) for fast agent turns.
  With thinking on, even trivial questions use ~1k tokens.
- **KV cache is small: ~49.6k tokens (6.8 GiB)** at util 0.75, since the desktop/Xwayland holds ~3.6 GB and weights take 15.3 GB.
  That's fine for many short requests, but e.g. only ~12 concurrent 4k-token contexts fit at once (the rest queue).
  If agents carry long histories, run `GPU_UTIL=0.85 ./serve.sh` (~10 GiB KV), provided nothing else needs the GPU.
- Default sampling comes from the model's generation_config: temperature 0.6, top_k 20, top_p 0.95 (override per request).
- `~/.local/lib/python3.12/site-packages` holds old torch/transformers/openai that **shadow the env**. serve.sh sets
  `PYTHONNOUSERSITE=1`, and so does `conda activate swarm` (it's in the env vars). Set it yourself when calling the env's python directly.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is set in serve.sh. Without it, startup crashes: FlashInfer tries to JIT-compile and
  there is no nvcc / CUDA toolkit. Attention uses FlashAttention 2, which is fine.
- Binds to 127.0.0.1 only. Use `HOST=0.0.0.0` if something outside WSL needs it.
- Tool calling is not enabled. If needed, add `--enable-auto-tool-choice --tool-call-parser hermes` to serve.sh (untested).
- Uses ~24 GB VRAM while running. Run `./stop.sh` before nnsight work on the same GPU.
