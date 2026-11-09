#!/usr/bin/env bash
# Verify every pre-registration hash in betrayal/ (prespec bodies, frozen code, Jenifer's package).
# Prints one line per entry and exits 1 on any mismatch. No API, no Python packages needed.
cd "$(dirname "$0")/../betrayal" || exit 1
rc=0
for f in PREREG_*SHA256.txt; do
  while read -r h p; do
    [ -z "$h" ] && continue
    case "$p" in
      *"(body)") fn=${p%(body)}; got=$(tail -n +2 "$fn" | sha256sum | cut -d' ' -f1) ;;  # body = line 2..end
      *) got=$(sha256sum "$p" | cut -d' ' -f1) ;;
    esac
    if [ "$got" = "$h" ]; then echo "OK        $f  $p"; else echo "MISMATCH  $f  $p"; rc=1; fi
  done < "$f"
done
sha256sum -c pkg_module_sha256.txt --quiet && echo "OK        pkg_module_sha256.txt  (17 modules of pkg/swarm_llm)" || rc=1
exit $rc
