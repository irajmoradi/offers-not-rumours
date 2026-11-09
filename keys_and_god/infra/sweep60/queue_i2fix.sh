#!/usr/bin/env bash
# Starts the I2 fixed-render rerun (sweep60_or_glm_i2fix, cap $12) once the C2 / C3 control drivers have exited
# (lead 15:40: do not compete with them for OpenRouter capacity). Detached: nohup setsid infra/sweep60/queue_i2fix.sh &
cd ${SWARM_ROOT:-.} || exit 1
LOG=runs/sweep60_or_glm_i2fix_queue.log
echo "$(date +%H:%M:%S) waiting for C2/C3 drivers: $(pgrep -f 'grid_driver.py --cell sweep60_or_c[23]_' | tr '\n' ' ')" >> $LOG
while pgrep -f "grid_driver.py --cell sweep60_or_c[23]_" >/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) C2/C3 done; launching sweep60_or_glm_i2fix" >> $LOG
PYTHONNOUSERSITE=1 exec ${PY:-python} infra/grid_driver.py \
  --cell sweep60_or_glm_i2fix:12:2:n=60,g=0,seeds=2,shared=2 --seeds 2 --rounds 80 \
  --ledger ${SWARM_ROOT:-.}/runs/openrouter_ledger.jsonl --ledger-stop 130 \
  --primary-provider Relace --stall-min 20 --log runs/sweep60_or_glm_i2fix_driver.log >> runs/sweep60_or_glm_i2fix_driver.out 2>&1
