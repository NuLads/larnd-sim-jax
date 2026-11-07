#!/bin/bash
# Move FINISHED staging runs to /sdf/data: rsync, verify bytes+file count, repoint fit_result link, drop scratch copy.
set -uo pipefail
ST=/sdf/scratch/neutrino/pgranger/fit_result_staging; DST=/sdf/data/neutrino/pgranger/larnd-sim-jax/fit_result
FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result; LOG=/sdf/scratch/neutrino/pgranger/review_2026_09/move_staging.log
count() { find "$1" -type f | wc -l; }; bytes() { du -sb --apparent-size "$1" | cut -f1; }
for n in "$@"; do
  s=$ST/$n; d=$DST/$n; l=$FR/$n
  [ -d "$s" ] && [ -n "$n" ] || { echo "SKIP $n (no staging dir)" | tee -a $LOG; continue; }
  [ "$(readlink "$l")" = "$s" ] || { echo "SKIP $n (link not -> staging)" | tee -a $LOG; continue; }
  rsync -a "$s/" "$d/" || { echo "RSYNC FAIL $n" | tee -a $LOG; continue; }
  if [ "$(bytes $s)" = "$(bytes $d)" ] && [ "$(count $s)" = "$(count $d)" ]; then
    ln -sfn "$d" "$l" && [ -e "$l" ] && rm -rf -- "$s" && echo "MOVED $n ($(du -sh $d | cut -f1))" | tee -a $LOG
  else echo "VERIFY FAIL $n" | tee -a $LOG; fi
done
df -h /sdf/data/neutrino | tail -1
