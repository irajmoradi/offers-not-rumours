#!/usr/bin/env bash
# Continuous RAM/rg flight recorder (survives a VM crash: fsync after each write).
# Every 3s: time, MemAvailable, swap used, rg count, top-4 RSS processes.
# When rg > 5: also who spawned the rg's (parent cmdline), one rg's args + cwd.
LOG=infra/memlog.log
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  swap=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)
  nrg=$(pgrep -xc rg || true)
  top=$(ps -eo rss=,comm= --sort=-rss | head -4 | awk '{printf "%s:%dMB ", $2, $1/1024}')
  {
    echo "$(date +%H:%M:%S) avail=${avail}MB swap=${swap}MB rg=${nrg:-0} | $top"
    if [ "${nrg:-0}" -gt 5 ]; then
      ps -o ppid= -p "$(pgrep -xd, rg)" 2>/dev/null | sort | uniq -c | sort -rn | head -3 | while read -r n pp; do
        echo "   $n rg <- ppid $pp: $(ps -o args= -p "$pp" | cut -c1-250)"; done
      p1=$(pgrep -xo rg)
      [ -n "$p1" ] && echo "   rg args: $(tr '\0' ' ' < /proc/$p1/cmdline 2>/dev/null | cut -c1-300) | cwd=$(readlink /proc/$p1/cwd)"
    fi
  } >> "$LOG"
  sync "$LOG" 2>/dev/null
  sleep 3
done
