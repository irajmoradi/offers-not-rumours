#!/bin/bash
# sweep100 queue: one grid_driver per block, strictly in priority order 1 -> 2 -> 6 -> 7 -> 8 -> 3 -> 4 -> 5.
# Stops the queue when a block's driver aborts / exits non-zero (watchdog: ledger stop, projection, errors, parse,
# truncation, stall, RAM) or when the world gate skipped any world (the $1,300 sweep cap is reached).
#   cd keys_and_god && setsid nohup infra/sweep100/run_queue.sh > runs/sweep100_queue.out 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
export PYTHONNOUSERSITE=1
LEDGER=${SWARM_ROOT:-.}/runs/adaption_sweep_ledger.jsonl
LOG=runs/sweep100_driver.log
CELLS=(
  "sweep100_glm_high:340:2:n=100,g=0,seeds=5,shared=2"
  "sweep100_glm_low:160:2:n=100,g=0,seeds=5,shared=2"
  "sweep100_mixed_effort:90:1:n=100,g=0,seeds=5,shared=0"
  "sweep100_scripted:340:2:n=100,g=0,seeds=5,shared=2"
  "sweep50_glm_high:170:2:n=50,g=0,seeds=5,shared=2"
  "sweep100_glm_high_k14:340:2:n=100,g=0,seeds=5,shared=2"
  "sweep100_deepseek:140:2:n=100,g=0,seeds=5,shared=2,trunc=10"
  "sweep100_gemma:400:2:n=100,g=0,seeds=5,shared=2,trunc=10"
)
for cell in "${CELLS[@]}"; do
  name="${cell%%:*}"
  if [ -f "runs/$name/manifest.json" ] && grep -q '"status": "done"' "runs/$name/manifest.json" && \
     ! grep -q '"status": "skipped"' "runs/$name/manifest.json"; then
    echo "$(date +%T) QUEUE skip $name (already done)" | tee -a "$LOG"; continue
  fi
  echo "$(date +%T) QUEUE start $name" | tee -a "$LOG"
  "$PY" infra/grid_driver.py --cell "$cell" --seeds 5 --rounds 80 --ledger "$LEDGER" --ledger-stop 1300 \
        --stall-min 20 --log "$LOG"
  rc=$?
  echo "$(date +%T) QUEUE end $name rc=$rc" | tee -a "$LOG"
  if [ $rc -ne 0 ]; then echo "$(date +%T) QUEUE STOP: $name driver rc=$rc (abort or failed cell)" | tee -a "$LOG"; exit $rc; fi
  if grep -q '"status": "skipped"' "runs/$name/manifest.json" 2>/dev/null; then
    echo "$(date +%T) QUEUE STOP: world gate skipped worlds in $name (sweep cap)" | tee -a "$LOG"; exit 5
  fi
done
echo "$(date +%T) QUEUE DONE" | tee -a "$LOG"
