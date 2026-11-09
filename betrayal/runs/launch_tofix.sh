#!/bin/bash
# TOFIX (prespec_tofix.md). $1 = smoke|main. Key read by path inside launch.py (never printed). No auto-relaunch.
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
KEY="${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}"
if [ "$1" = smoke ]; then
  OUT=${PLANC_ROOT:-.}/out/tofix_smoke TOFIX_SMOKE=1 SEED_START=899 SEED_END=900 COST_CAP=0.06 PAR_CELLS=1 API_CONCURRENCY=16 \
  PYTHONHASHSEED=0 MPLBACKEND=Agg timeout 1800 $PY -s -P -u harness/launch.py "$KEY" runs/v2run ${PLANC_ROOT:-.}/harness/run_tofix.py \
  > ${PLANC_ROOT:-.}/out/tofix_smoke_log.txt 2>&1
  echo "$(date +%T) smoke exit $?" >> ${PLANC_ROOT:-.}/out/tofix_exit.txt
else
  OUT=${PLANC_ROOT:-.}/out/tofix SEED_START=800 SEED_END=820 COST_CAP=0.60 PAR_CELLS=5 API_CONCURRENCY=24 \
  PYTHONHASHSEED=0 MPLBACKEND=Agg timeout 7200 $PY -s -P -u harness/launch.py "$KEY" runs/v2run ${PLANC_ROOT:-.}/harness/run_tofix.py \
  > ${PLANC_ROOT:-.}/out/tofix_log.txt 2>&1
  echo "$(date +%T) main exit $?" >> ${PLANC_ROOT:-.}/out/tofix_exit.txt
fi
