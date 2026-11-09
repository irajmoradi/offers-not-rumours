#!/bin/bash
# Noor T1-core main runs (prespec_noor.md). LAUNCH ONLY AFTER THE TEAM LEAD'S "GO". Key read by path (never printed).
# 1) US-reference slot probe (pre-registered rule) -> out/noor_prereg/gemma_slot.json; 2) one process per model,
# seeds 700-723, short prompts, temperature 0.1 + top_k 5 (require_parameters). No auto-relaunch.
cd ${PLANC_ROOT:-.}
PY=${PY:-python}
KEY="${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}"
GEM=$($PY -I harness/noor_gemma_probe.py "$KEY" out/noor_prereg/gemma_slot.json | tail -1)
echo "$(date +%T) gemma slot: $GEM" >> out/noor_exit.txt
GEMMA_OK=1
if [ "$GEM" != "google/gemma-3-12b-it" ] && [ "$GEM" != "google/gemma-3-27b-it" ]; then
  GEMMA_OK=0; echo "$(date +%T) probe failed for both Gemma models; Gemma slot PENDING, other 3 models launch" >> out/noor_exit.txt
fi
run() {  # $1 tag  $2 model  $3 cap
  OUT=${PLANC_ROOT:-.}/out/noor_$1 MODEL=$2 SHORT=1 SEED_START=700 SEED_END=724 COST_CAP=$3 PAR_CELLS=4 API_CONCURRENCY=16 \
  PYTHONHASHSEED=0 MPLBACKEND=Agg timeout 7200 $PY -s -P -u harness/launch.py "$KEY" runs/v2run \
  ${PLANC_ROOT:-.}/harness/run_noor.py > ${PLANC_ROOT:-.}/out/noor_log_$1.txt 2>&1
  echo "$(date +%T) $1 exit $?" >> ${PLANC_ROOT:-.}/out/noor_exit.txt
}
[ $GEMMA_OK = 1 ] && { run gemma "$GEM" 0.75 & }
run phi4 microsoft/phi-4 0.62 &
run mistral mistralai/mistral-small-3.2-24b-instruct 0.70 & run qwen qwen/qwen-2.5-7b-instruct 0.63 &
wait; echo "$(date +%T) all done" >> ${PLANC_ROOT:-.}/out/noor_exit.txt
