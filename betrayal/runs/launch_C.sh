#!/bin/bash
cd ${PLANC_ROOT:-.}
until grep -qE "FINAL|Traceback|same error" runs/arm_UF_main_log.txt; do sleep 15; done
echo "$(date +%T) UF finished; launching C" >> runs/launch_C_log.txt
ARMS=C SEEDS_C=40 OUT=${PLANC_ROOT:-.}/out/arm_C_contagion COST_CAP=2.00 PAR_CELLS=12 API_CONCURRENCY=96 PYTHONHASHSEED=0 MPLBACKEND=Agg \
  timeout 7200 ${PY:-python} -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_C_log.txt 2>&1
echo "$(date +%T) C exit $?" >> runs/launch_C_log.txt
