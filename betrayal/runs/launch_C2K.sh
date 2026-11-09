#!/bin/bash
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
echo "$(date +%T) launching C2K" >> runs/launch_C2K_log.txt
C_N=16 C_SIZES=2 C_LABEL=C2K ARMS=C SEEDS_C=40 OUT=${PLANC_ROOT:-.}/out/arm_C2K_contagion COST_CAP=1.00 PAR_CELLS=10 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_C2K_log.txt 2>&1
echo "$(date +%T) C2K exit $?" >> runs/launch_C2K_log.txt
