#!/bin/bash
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
echo "$(date +%T) launching DB" >> runs/launch_DB_log.txt
ARMS=DB SEEDS_DB=20 OUT=${PLANC_ROOT:-.}/out/arm_DB_disclosure COST_CAP=0.75 PAR_CELLS=10 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_DB_log.txt 2>&1
echo "$(date +%T) DB exit $?" >> runs/launch_DB_log.txt
