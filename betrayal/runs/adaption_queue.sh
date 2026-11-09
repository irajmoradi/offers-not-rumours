#!/bin/bash
# Stop arm F once seeds 1-6 are complete (with pilot seed 0 = 7 seeds), then run the Adaption queue G -> GR -> H(N=4).
cd ${PLANC_ROOT:-.}
FPID=3350787
until python3 -c "
import json,sys
s={json.loads(l)['seed'] for l in open('out/arm_F_main/cells.jsonl')}
sys.exit(0 if set(range(1,7)) <= s else 1)"; do
  kill -0 $FPID 2>/dev/null || break; sleep 20; done
kill $FPID 2>/dev/null; sleep 5
echo "$(date +%T) arm F stopped after seeds 0-6" >> runs/adaption_queue_log.txt
COMMON="PROVIDER=adaption ADAPTION_KEY_FILE=${ADAPTION_KEY_FILE:?set ADAPTION_KEY_FILE to your key file path} ADAPTION_MAX_CONCURRENCY=8 ADAPTION_TPM=30000 ADAPTION_TPM_FLOOR=10000 COST_CAP=0.02 PYTHONHASHSEED=0 MPLBACKEND=Agg"
for spec in "G SEEDS_G=3 ADAPTION_CAP_USD=0.60" "GR SEEDS_GR=3 ADAPTION_CAP_USD=0.30" "H SEEDS_H=3 SIZES_H=4 ADAPTION_CAP_USD=0.20"; do
  set -- $spec; A=$1; shift
  echo "$(date +%T) start $A on Adaption" >> runs/adaption_queue_log.txt
  env $COMMON "$@" ARMS=$A OUT=${PLANC_ROOT:-.}/out/arm_${A}_ada PAR_CELLS=2 timeout 7200 \
    ${PY:-python} -s -P -u harness/launch.py "${ADAPTION_KEY_FILE:?set ADAPTION_KEY_FILE to your key file path}" \
    runs/v2run ${PLANC_ROOT:-.}/harness/run_v2.py > runs/arm_${A}_ada_log.txt 2>&1
  echo "$(date +%T) done $A rc=$?" >> runs/adaption_queue_log.txt
done
