#!/bin/bash
cd ${PLANC_ROOT:-.}
until grep -qE "FINAL|Traceback|same error" runs/arm_F_pilot3_log.txt; do sleep 10; done
if grep -qE "Traceback|same error" runs/arm_F_pilot3_log.txt; then echo "pilot3 failed; not launching" > runs/arm_F_main_log.txt; exit 1; fi
ARMS=F SEED_START_F=1 SEEDS_F=10 ADAPTION_KEY_FILE="${ADAPTION_KEY_FILE:?set ADAPTION_KEY_FILE to your key file path}" ADAPTION_CAP_USD=2.50 ADAPTION_MAX_CONCURRENCY=8 \
ADAPTION_TPM=40000 ADAPTION_TPM_FLOOR=10000 OUT=${PLANC_ROOT:-.}/out/arm_F_main COST_CAP=0.30 PAR_CELLS=2 PYTHONHASHSEED=0 MPLBACKEND=Agg \
timeout 14400 ${PY:-python} -s -P -u harness/launch.py "${ADAPTION_KEY_FILE:?set ADAPTION_KEY_FILE to your key file path}" \
  runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_F_main_log.txt 2>&1
