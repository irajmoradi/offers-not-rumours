#!/bin/bash
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
echo "$(date +%T) launching T" >> runs/launch_T_C16_log.txt
ARMS=T SEEDS_T=40 OUT=${PLANC_ROOT:-.}/out/arm_T_paytiming COST_CAP=1.20 PAR_CELLS=10 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_T_log.txt 2>&1
echo "$(date +%T) T exit $?; launching C16" >> runs/launch_T_C16_log.txt
C_N=16 ARMS=C SEEDS_C=40 OUT=${PLANC_ROOT:-.}/out/arm_C16_contagion COST_CAP=2.50 PAR_CELLS=10 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_C16_log.txt 2>&1
echo "$(date +%T) C16 exit $?" >> runs/launch_T_C16_log.txt
