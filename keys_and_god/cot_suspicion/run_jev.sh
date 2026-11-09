#!/bin/bash
# pass 1: every 2nd round (balanced subsample), pass 2: fill remaining rounds (cached items skipped)
set -u
# NOTE: only the stride-2 pass was run for nogod (Jev cost ~$0.00035/item; full set > $6 budget)
for S in 2; do
  python3 -I ${SWARM_ROOT:-.}/cot_suspicion/cot_judge.py judge --out ${SWARM_ROOT:-.}/cot_suspicion --tag jev_nogod_s$S --round-stride $S --cap 6.5 "Game Env=${SWARM_ROOT:-.}/runs/sweep60_or_oldframe/games/g00-nogod" "Game Env=${SWARM_ROOT:-.}/runs/sweep60_or_oldframe/games/g01-nogod" "Game Env=${SWARM_ROOT:-.}/runs/sweep60_or_oldframe/games/g04-nogod" "Realistic Env=${SWARM_ROOT:-.}/runs/sweep60_or_glm/games/g00-nogod" "Realistic Env=${SWARM_ROOT:-.}/runs/sweep60_or_glm/games/g01-nogod" "Realistic Env=${SWARM_ROOT:-.}/runs/sweep60_or_glm/games/g04-nogod" || exit 1
  grep -q "HARD STOP" ${SWARM_ROOT:-.}/cot_suspicion/judge_run_jev.log && exit 2
done
