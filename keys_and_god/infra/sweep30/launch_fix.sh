#!/usr/bin/env bash
# sweep30 v5 (19:20): relaunch B0 (gpt-oss) and sweep60_or_soft_perm_k1 after FALSE projection aborts at ~1% done
# (B0 $1.674 > $1.60 at 1.7%; soft_perm_k1 $4.235 > $4.20 at 0.9%); partial dirs -> runs/aborted_*_proj, the restart
# replays their cache. gpt-oss is completion-heavy (Near AI: ~$0.03/M in, $0.15/M out, 700-900 completion tokens), so
# its cells use the completion-heavy cost model (g=200,pcap=4550); every cell's driver threshold = the config's hard cap
# (llm.budget_usd, unchanged) + headroom, so only the hard cap stops a run on spend. B1 / B5 / B10 start after every
# B0 game has reached round 3 (paired prefix in B0's cache). RAM gate > 20 GB, 60 s settle per start.
cd ${SWARM_ROOT:-.} || exit 1
PY=${PY:-python}
LEDGER=${SWARM_ROOT:-.}/runs/openrouter_ledger_sweep30.jsonl
LOG=runs/sweep30_launch.log
OSS="n=30,g=200,pcap=4550,seeds=3"; SOFT="n=60,g=436,pcap=9110,seeds=3,free=1,shared=2"
say() { echo "$(date +%H:%M:%S) [v5] $*" >> $LOG; }
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
start sweep30_b0_oss:2.0:1:$OSS 30 32                      # hard cap 1.6
start sweep60_or_soft_perm_k1:4.8:1:$SOFT 40 36.5           # hard cap 4.2
for i in $(seq 1 60); do
  n=0
  for g in runs/sweep30_b0_oss/games/g0{0,1,4}-nogod/llm_calls.jsonl; do
    [ -f "$g" ] && tail -n 5 "$g" | grep -q '"round": \?\([3-9]\|[1-9][0-9]\),' && n=$((n+1))
  done
  [ $n -eq 3 ] && break
  sleep 30
done
say "sweep30_b0_oss round-3 gate: $n/3"
start sweep30_b1_oss_k1:2.2:1:$OSS 30 32                   # hard cap 1.8
start sweep30_b5_oss_k5:2.4:1:$OSS 30 32                   # hard cap 2.0
start sweep30_b10_oss_k10:3.0:1:$OSS 30 32                 # hard cap 2.6
say "v5: B0, soft_perm_k1, B1, B5, B10 started"
