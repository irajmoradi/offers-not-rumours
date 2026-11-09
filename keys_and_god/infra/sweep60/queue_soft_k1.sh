#!/usr/bin/env bash
# Queues sweep60_or_soft_perm_k1 then sweep60_or_soft_silent_k1 (infra/sweep60/make_soft_k1.py; lead 19:05, caps $4.2
# each) behind the sweep30 lanes: starts only after all 5 sweep30 lanes have started and fewer than 5 sweep30 / soft-k1
# grid drivers are alive (<= 5 lanes), with MemAvailable > 20 GB. Each on the sweep30 ledger, --ledger-stop 36.5.
# Detached: nohup setsid infra/sweep60/queue_soft_k1.sh >/dev/null 2>&1 &
cd ${SWARM_ROOT:-.} || exit 1
PY=${PY:-python}
LEDGER=${SWARM_ROOT:-.}/runs/openrouter_ledger_sweep30.jsonl
LOG=runs/sweep30_launch.log
say() { echo "$(date +%H:%M:%S) [soft-k1 queue] $*" >> $LOG; }
lanes() { pgrep -f "grid_driver.py --cell (sweep30_|sweep60_or_soft_)" | wc -l; }
memok() { awk '/MemAvailable/ {exit !($2 > 20*1024*1024)}' /proc/meminfo; }
until grep -q "all 5 lanes started" $LOG; do sleep 60; done
for name in sweep60_or_soft_perm_k1 sweep60_or_soft_silent_k1; do
  until [ "$(lanes)" -lt 5 ] && memok; do sleep 60; done
  PYTHONNOUSERSITE=1 nohup setsid $PY infra/grid_driver.py --cell $name:4.2:1:n=60,g=436,pcap=9110,seeds=3 \
    --seeds 3 --rounds 40 --ledger "$LEDGER" --ledger-stop 36.5 --primary-provider Relace --stall-min 20 \
    --log runs/${name}_driver.log >> runs/${name}_driver.out 2>&1 < /dev/null &
  say "started $name driver pid $!; lanes now $(lanes); MemAvailable $(awk '/MemAvailable/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
  sleep 120
done
say "both soft-k1 runs started"
