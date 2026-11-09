#!/usr/bin/env bash
# RAM watchdog: prevents WSL OOM crashes (2026-11-07 crash = hundreds of concurrent rg).
# < 6GB available or > 40 rg procs: kill all rg. < 3GB: also kill the single largest python/node/bun (never claude).
# Launch:  setsid nohup infra/ramguard.sh >/dev/null 2>&1 < /dev/null &
LOG=infra/ramguard.log
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  nrg=$(pgrep -xc rg || true)
  if [ "$avail" -lt 6144 ] || [ "${nrg:-0}" -gt 40 ]; then
    echo "$(date -Is) avail=${avail}MB rg=${nrg} -> killing rg; parents + sample args:" >> "$LOG"
    ps -o ppid= -p "$(pgrep -xd, rg)" 2>/dev/null | sort | uniq -c | sort -rn | head -3 | while read n pp; do
      echo "   $n rg under ppid $pp: $(ps -o args= -p "$pp" | cut -c1-200)" >> "$LOG"; done
    p1=$(pgrep -xo rg); [ -n "$p1" ] && echo "   rg args: $(tr '\0' ' ' < /proc/$p1/cmdline | cut -c1-300) cwd=$(readlink /proc/$p1/cwd)" >> "$LOG"
    pkill -9 -x rg
  fi
  if [ "$avail" -lt 3072 ]; then
    victim=$(ps -eo pid=,rss=,comm= --sort=-rss | awk '$3!~/^claude/ && $3~/python|node|bun|MainThread|vllm/ {print $1" "$3; exit}')
    echo "$(date -Is) avail=${avail}MB CRITICAL -> killing $victim" >> "$LOG"
    [ -n "$victim" ] && kill -9 "${victim%% *}"
  fi
  sleep 1
done
