#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server in the background. Logs -> infra/vllm.log, PID -> infra/vllm.pid
# Usage: ./serve.sh [MODEL]   (default Qwen/Qwen3-8B)
# Env overrides: PORT, HOST, GPU_UTIL, MAX_LEN, MAX_SEQS, REASONING_PARSER (default qwen3; set empty to disable), WAIT=0 (don't block)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_SEQS="${MAX_SEQS:-256}"
EXTRA=()
REASONING_PARSER="${REASONING_PARSER-qwen3}"
[[ -n "${REASONING_PARSER:-}" ]] && EXTRA+=(--reasoning-parser "$REASONING_PARSER")
VLLM="${VLLM:-vllm}"

if [[ -f "$DIR/vllm.pid" ]] && kill -0 "$(cat "$DIR/vllm.pid")" 2>/dev/null; then
  echo "vLLM already running (pid $(cat "$DIR/vllm.pid")). Run ./stop.sh first."; exit 1
fi

export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.12 has old torch/transformers/openai that shadow the env
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer sampler JIT-compiles and needs nvcc (not installed) -> crash at startup
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights are already in ~/.cache/huggingface; don't hit the network
setsid nohup "$VLLM" serve "$MODEL" \
  --port "$PORT" --host "$HOST" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching \
  "${EXTRA[@]}" \
  > "$DIR/vllm.log" 2>&1 &
echo $! > "$DIR/vllm.pid"
echo "Started vLLM pid $(cat "$DIR/vllm.pid") model=$MODEL port=$PORT; log: $DIR/vllm.log"

if [[ "${WAIT:-1}" == "1" ]]; then
  echo -n "Waiting for /health"
  for _ in $(seq 1 300); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then echo " ready."; exit 0; fi
    if ! kill -0 "$(cat "$DIR/vllm.pid")" 2>/dev/null; then echo " server died; tail of log:"; tail -30 "$DIR/vllm.log"; exit 1; fi
    echo -n "."; sleep 2
  done
  echo " timed out (still starting?). Check $DIR/vllm.log"; exit 1
fi
