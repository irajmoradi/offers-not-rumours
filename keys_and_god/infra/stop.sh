#!/usr/bin/env bash
# Stop the vLLM server started by serve.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$DIR/vllm.pid" ]]; then echo "No pid file; nothing to stop."; exit 0; fi
PID="$(cat "$DIR/vllm.pid")"
if kill -0 "$PID" 2>/dev/null; then
  # serve.sh starts vLLM with setsid, so PID == process-group id; this also kills the EngineCore child
  kill -TERM -- "-$PID" 2>/dev/null || kill "$PID"
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && { echo "Force killing group $PID"; kill -9 -- "-$PID"; }
  echo "Stopped vLLM (pid $PID)."
else
  echo "Process $PID not running."
fi
rm -f "$DIR/vllm.pid"
