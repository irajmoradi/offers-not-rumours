#!/usr/bin/env bash
# Resume the runs stopped by the OpenRouter key limit (HTTP 403 at 19:59:18 EST), ONLY after the lead says "resume"
# (Iraj raising the key limit to ~$165). Priority order (lead): oldframe_k1, i2fix k1, A7, B5, B10, B1; then
# soft_perm_k1 g00 (3 rounds left; found partial by final_report.py).
# Each is a new run id resume_<name> on the unfinished worlds (the runner's resume pattern, infra/sweep100/lanes.py):
# the same game ids, so every played round replays from the cache at $0, and play continues from the stop round.
# Driver cost model: free=1,shared=<min completed rounds> so the projection only counts the rounds still to pay.
# Spend: ONE separate ledger for all resumed runs (llm.openrouter.ledger_path / global_cap_usd overrides, operational, unhashed),
# driver --ledger-stop 12 on it. RAM gate > 20 GB, 60 s settle per start. (The 22:30 no-start rule was dropped, lead.)
# Detached: nohup setsid infra/sweep30/resume_after_403.sh [KEY_FILE_PATH] >/dev/null 2>&1 &
# KEY_FILE_PATH (optional, a path only): exported as SWARM_OPENROUTER_KEY_FILE so every driver / run loads the
# OpenRouter key from that file (swarm/llm/secrets.py), e.g. the team key ${OPENROUTER_KEY_FILE:?set OPENROUTER_KEY_FILE to your key file path}.
cd ${SWARM_ROOT:-.} || exit 1
if [ -n "$1" ]; then
  [ -f "$1" ] || { echo "$(date +%H:%M:%S) [resume] key file path not found; nothing started" >> runs/sweep30_launch.log; exit 2; }
  export SWARM_OPENROUTER_KEY_FILE="$1"
fi
PY=${PY:-python}
H=${SWARM_ROOT:-.}/runs
LOG=runs/sweep30_launch.log
RL=$H/openrouter_ledger_resume.jsonl   # lead 20:2x: every resumed run on its OWN ledger, stop $12
touch "$RL"
say() { echo "$(date +%H:%M:%S) [resume] $*" >> $LOG; }
mem() { awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo; }
memok() { awk '/MemAvailable/ {exit !($2 > 20*1024*1024)}' /proc/meminfo; }
start() {   # start <cell spec> <rounds> <ledger> <ledger stop> <override>...
  local cell=$1 rounds=$2; shift 4                  # (args 3-4, the original ledger / stop, are kept for the record)
  local ledger=$RL stop=12
  local name=${cell%%:*}
  until memok; do say "$name waiting: MemAvailable $(mem) GB <= 20"; sleep 30; done
  local sets=(--set "llm.openrouter.ledger_path=$RL" --set "llm.openrouter.global_cap_usd=12.0")
  for s in "$@"; do sets+=(--set "$s"); done
  PYTHONNOUSERSITE=1 nohup setsid $PY infra/grid_driver.py --cell "$cell" --run-prefix resume_ --seeds 3 --rounds $rounds \
    --ledger "$ledger" --ledger-stop $stop --primary-provider Relace --stall-min 20 "${sets[@]}" \
    --log runs/resume_${name}_driver.log >> runs/resume_${name}_driver.out 2>&1 < /dev/null &
  say "started resume_$name driver pid $!; MemAvailable $(mem) GB"
  sleep 60
}
OSS=n=30,g=200,pcap=4550
start sweep60_or_oldframe_k1:9.0:1:n=60,g=350,pcap=11000,seeds=3,free=1,shared=32 80 $H/openrouter_ledger_oldframe_k1.jsonl 8.5 \
  "runner.game_indices=[0, 1, 4]"
start sweep60_or_glm_i2fix:12:1:n=60,g=436,pcap=9110,seeds=2,free=1,shared=47 80 $H/openrouter_ledger.jsonl 130 \
  "runner.game_indices=[0, 1]" "runner.arms=[{label: k1, god: {budget: {k: 1}}}]"
start sweep30_a7_k20:5.5:1:n=30,g=436,pcap=9110,seeds=3,free=1,shared=21 30 $H/openrouter_ledger_sweep30.jsonl 32 \
  "runner.game_indices=[0, 1, 4]"
start sweep30_b5_oss_k5:2.4:1:$OSS,seeds=3,free=1,shared=21 30 $H/openrouter_ledger_sweep30.jsonl 32 "runner.game_indices=[0, 1, 4]"
start sweep30_b10_oss_k10:3.0:1:$OSS,seeds=3,free=1,shared=17 30 $H/openrouter_ledger_sweep30.jsonl 32 "runner.game_indices=[0, 1, 4]"
start sweep30_b1_oss_k1:2.2:1:$OSS,seeds=1,free=1,shared=29 30 $H/openrouter_ledger_sweep30.jsonl 32 "runner.game_indices=[1]"
start sweep60_or_soft_perm_k1:4.8:1:n=60,g=436,pcap=9110,seeds=1,free=1,shared=37 40 $H/openrouter_ledger_sweep30.jsonl 36.5 \
  "runner.game_indices=[0]"                                    # g00-k1 stopped at r37 of 40 by the 403
say "resume: all 7 started (key file: ${SWARM_OPENROUTER_KEY_FILE:-default})"
