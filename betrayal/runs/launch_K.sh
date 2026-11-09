#!/bin/bash
cd ${PLANC_ROOT:-.}
until grep -qE "FINAL|Traceback|same error" runs/arm_J_gemma_log.txt; do sleep 15; done
read SEEDS CAP <<< $(python3 -c "
import json
u = json.load(open('out/arm_J_gemma/cost_tally.json'))['usd']
rem = 1.50 - u; s = min(4, int(rem / 0.105))
print(s, round(rem, 3))")
echo "$(date +%T) J done; K seeds=$SEEDS cap=$CAP" >> runs/launch_K_log.txt
[ "$SEEDS" -ge 2 ] || { echo "not enough budget for K" >> runs/launch_K_log.txt; exit 0; }
K_MODEL=google/gemma-3-12b-it ARMS=K SEEDS_K=$SEEDS OUT=${PLANC_ROOT:-.}/out/arm_K_gemma COST_CAP=$CAP PAR_CELLS=12 API_CONCURRENCY=64 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 3600 ${PY:-python} -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_K_gemma_log.txt 2>&1
echo "$(date +%T) K exit $?" >> runs/launch_K_log.txt
