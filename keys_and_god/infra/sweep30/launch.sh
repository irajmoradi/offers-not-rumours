#!/usr/bin/env bash
# sweep30 launcher (infra/sweep30/make_configs.py), v3 (18:58): GLM A arms + gpt-oss B arms on the SEPARATE sweep30
# ledger (--ledger-stop 32; configs global_cap_usd 32), never the shared openrouter_ledger.jsonl.
# Detached: nohup setsid infra/sweep30/launch.sh >/dev/null 2>&1 &
# 5 lanes (grid drivers; each runs its cells one after another, at most 5 sweep30 processes / 15 games in flight):
#   L1: A1 facility -> A3 minimal -> A4 low      L2: A2 oldframe -> A5 high
#   L3: A6 k10 -> B1 oss k1                      L4: A7 k20            L5: B0 oss -> B5 oss k5 -> B10 oss k10
# L3 / L4 start once every A1 game has reached round 3 (God wakes from r3): the God arms replay A1's rounds 1-2 from
# the shared cache (paired prefix). The B God arms run after B0 in L5 / after A6 in L3, so B0's cache holds rounds 1-2.
# A1 / A2 were first started 18:49 with global_cap 20 and stopped at r5-7 (dirs runs/aborted_sweep30_a*_cap20);
# their restart replays those rounds from the cache. Each start waits for MemAvailable > 20 GB, staggered by 90 s.
cd ${SWARM_ROOT:-.} || exit 1
PY=${PY:-python}
LEDGER=${SWARM_ROOT:-.}/runs/openrouter_ledger_sweep30.jsonl
LOG=runs/sweep30_launch.log
touch "$LEDGER"                       # grid_driver.ledger_total() opens it before the first paid call
# grid_driver projection cost model per arm (call_weight shape fitted to GLM's measured per-round cost: prompt grows
# ~800 tok/round to ~17k, completion ~700 tok (high ~2250); g=800,pcap=17000 overestimated round-1 projections 1.7-2.8x)
OPT="n=30,g=436,pcap=9110,seeds=3"      # facility medium / minimal / low, the God arms, gpt-oss
OPT_OLD="n=30,g=350,pcap=11000,seeds=3" # agent/v1 prompts
OPT_HIGH="n=30,g=200,pcap=4550,seeds=3" # effort high
say() { echo "$(date +%H:%M:%S) $*" >> $LOG; }
memok() { awk '/MemAvailable/ {exit !($2 > 20*1024*1024)}' /proc/meminfo; }
start() {   # start <tag> <cell spec>...
  local tag=$1; shift
  until memok; do say "$tag waiting: MemAvailable <= 20 GB"; sleep 60; done
  local cells=(); for c in "$@"; do cells+=(--cell "$c"); done
  PYTHONNOUSERSITE=1 nohup setsid $PY infra/grid_driver.py "${cells[@]}" --seeds 3 --rounds 30 \
    --ledger "$LEDGER" --ledger-stop 32 --primary-provider Relace --stall-min 20 \
    --log runs/sweep30_${tag}_driver.log >> runs/sweep30_${tag}_driver.out 2>&1 < /dev/null &
  say "started $tag pid $! cells $*; MemAvailable $(awk '/MemAvailable/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
  sleep 90
}
say "v3 launcher start"
start L1 sweep30_a1_facility:2.0:1:$OPT sweep30_a3_minimal:2.0:1:$OPT sweep30_a4_low:2.0:1:$OPT
start L2 sweep30_a2_oldframe:1.5:1:$OPT_OLD sweep30_a5_high:3.0:1:$OPT_HIGH
start L5 sweep30_b0_oss:1.6:1:$OPT sweep30_b5_oss_k5:2.0:1:$OPT sweep30_b10_oss_k10:2.6:1:$OPT
say "waiting for every A1 game to reach round 3"
for i in $(seq 1 60); do
  n=0
  for g in runs/sweep30_a1_facility/games/g0{0,1,4}-nogod/llm_calls.jsonl; do
    [ -f "$g" ] && tail -n 5 "$g" | grep -q '"round": \?\([3-9]\|[1-9][0-9]\),' && n=$((n+1))
  done
  [ $n -eq 3 ] && break
  sleep 60
done
say "A1 round-3 gate: $n/3 games past round 2 (after $i checks)"
start L3 sweep30_a6_k10:3.5:1:$OPT sweep30_b1_oss_k1:1.8:1:$OPT
start L4 sweep30_a7_k20:5.5:1:$OPT
say "all 5 lanes started"
