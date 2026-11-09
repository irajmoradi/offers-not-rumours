#!/bin/bash
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
echo "$(date +%T) launching SE" >> runs/launch_SE_log.txt
ARMS=SE SEEDS_SE=20 OUT=${PLANC_ROOT:-.}/out/arm_SE_levers COST_CAP=1.00 PAR_CELLS=10 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_SE_log.txt 2>&1
echo "$(date +%T) SE exit $?" >> runs/launch_SE_log.txt
