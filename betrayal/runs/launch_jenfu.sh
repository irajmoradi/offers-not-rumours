#!/bin/bash
# Jenifer follow-ups (prespec_jenfu.md). NOT YET LAUNCHED: wait for the lead's go. Key read by path inside launch.py.
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
run() {
  OUT=${PLANC_ROOT:-.}/out/jenfu_$1 ARMS=$1 SEED_START=300 SEED_END=320 COST_CAP=$2 PAR_CELLS=5 API_CONCURRENCY=24 \
  PYTHONHASHSEED=0 MPLBACKEND=Agg timeout 5400 $PY -s -P -u harness/launch.py "${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}" runs/v2run \
  ${PLANC_ROOT:-.}/harness/run_jenfu.py > ${PLANC_ROOT:-.}/out/jenfu_log_$1.txt 2>&1
  echo "$(date +%T) $1 exit $?" >> ${PLANC_ROOT:-.}/out/jenfu_exit.txt
}
run JF123 0.75 & run JF4 0.45 & run JF5 0.42 & run JF6 0.55 &
wait; echo "$(date +%T) all done" >> ${PLANC_ROOT:-.}/out/jenfu_exit.txt
