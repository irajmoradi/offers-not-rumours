#!/bin/bash
# Noor smoke 2 (paid, excluded from analysis): Gemma short vs long re-run after the provider-error fix in noor_decode.py (smoke 1 Gemma was invalid: 69% provider timeouts scored as empty).
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
sm() {  # $1 tag  $2 model  $3 short  $4 s0  $5 s1
  OUT=${PLANC_ROOT:-.}/out/noor_smoke2_$1 MODEL=$2 SHORT=$3 SEED_START=$4 SEED_END=$5 NOOR_SMOKE=1 NOOR_RAW=1 COST_CAP=0.08 \
  PAR_CELLS=2 API_CONCURRENCY=16 PYTHONHASHSEED=0 MPLBACKEND=Agg timeout 1500 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run \
  ${PLANC_ROOT:-.}/harness/run_noor.py > ${PLANC_ROOT:-.}/out/noor_smoke2_log_$1.txt 2>&1
  echo "$(date +%T) $1 exit $?" >> ${PLANC_ROOT:-.}/out/noor_smoke2_exit.txt
}
sm gemma_short google/gemma-3-12b-it 1 690 692 & sm gemma_long google/gemma-3-12b-it 0 690 692 &
wait; echo "$(date +%T) all done" >> ${PLANC_ROOT:-.}/out/noor_smoke2_exit.txt
