#!/usr/bin/env bash
# sweep30 v4 (19:03, Iraj via lead: "do it all at once"): every remaining arm as its own grid driver, in parallel, instead
# of chained lanes. A1 / A2 / B0 are already running (watched by --attach drivers L1 / L2 / L5). Safety gate: each start
# waits for MemAvailable > 20 GB, then 60 s to settle before the next is considered (other jobs' RAM guards are 15 GB).
# Order: A God arms (A1 is past round 2), A effort arms, soft-k1 (sweep60, 60 agents x 40 rounds), then the B God arms
# once every B0 game has reached round 3 (paired prefix from B0's cache).
# Detached: nohup setsid infra/sweep30/launch_all.sh >/dev/null 2>&1 &
cd ${SWARM_ROOT:-.} || exit 1
PY=${PY:-python}
LEDGER=${SWARM_ROOT:-.}/runs/openrouter_ledger_sweep30.jsonl
LOG=runs/sweep30_launch.log
OPT="n=30,g=436,pcap=9110,seeds=3"; OPT_HIGH="n=30,g=200,pcap=4550,seeds=3"; OPT60="n=60,g=436,pcap=9110,seeds=3"
say() { echo "$(date +%H:%M:%S) [v4] $*" >> $LOG; }
mem() { awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo; }
memok() { awk '/MemAvailable/ {exit !($2 > 20*1024*1024)}' /proc/meminfo; }
start() {   # start <cell spec> <rounds> <ledger stop>
  local name=${1%%:*}
  until memok; do say "$name waiting: MemAvailable $(mem) GB <= 20"; sleep 60; done
  PYTHONNOUSERSITE=1 nohup setsid $PY infra/grid_driver.py --cell "$1" --seeds 3 --rounds $2 \
    --ledger "$LEDGER" --ledger-stop $3 --primary-provider Relace --stall-min 20 \
    --log runs/${name}_driver.log >> runs/${name}_driver.out 2>&1 < /dev/null &
  say "started $name driver pid $!; MemAvailable $(mem) GB"
  sleep 60
}
gate() {    # gate <run> <arm label>: every game of the run has reached round 3
  for i in $(seq 1 60); do
    n=0
    for g in runs/$1/games/g0{0,1,4}-$2/llm_calls.jsonl; do
      [ -f "$g" ] && tail -n 5 "$g" | grep -q '"round": \?\([3-9]\|[1-9][0-9]\),' && n=$((n+1))
    done
    [ $n -eq 3 ] && break
    sleep 30
  done
  say "$1 round-3 gate: $n/3"
}
say "v4 all-at-once launcher start; MemAvailable $(mem) GB"
gate sweep30_a1_facility nogod
start sweep30_a6_k10:3.5:1:$OPT 30 32
start sweep30_a7_k20:5.5:1:$OPT 30 32
start sweep30_a3_minimal:2.0:1:$OPT 30 32
start sweep30_a4_low:2.0:1:$OPT 30 32
start sweep30_a5_high:3.0:1:$OPT_HIGH 30 32
start sweep60_or_soft_perm_k1:4.2:1:$OPT60 40 36.5
start sweep60_or_soft_silent_k1:4.2:1:$OPT60 40 36.5
gate sweep30_b0_oss nogod
start sweep30_b1_oss_k1:1.8:1:$OPT 30 32
start sweep30_b5_oss_k5:2.0:1:$OPT 30 32
start sweep30_b10_oss_k10:2.6:1:$OPT 30 32
say "v4: all arms started"
