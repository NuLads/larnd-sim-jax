#!/bin/bash
# STEP 2 of the data move (approved by Pierre, 2026-09-25): for every item verified by copy_data.sh,
# re-verify, remove the original and leave a symlink to the /sdf/data copy.
set -uo pipefail
SRC=/sdf/group/neutrino/pgranger
OUT=/sdf/scratch/neutrino/pgranger/review_2026_09
MAN=$OUT/copy_manifest.txt; LOG=$OUT/swap.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { find -L "$1" -type f 2>/dev/null | wc -l; }
bytes() { du -sbL --apparent-size "$1" 2>/dev/null | cut -f1; }
ok=0; skipped=0
say "=== swap start ($(wc -l < "$MAN") manifest items)"
while IFS=$'\t' read -r rel real dst || [ -n "$rel" ]; do
  link="$SRC/$rel"
  # guards: only our own paths, never an empty component
  case "$rel" in larnd-sim-jax/?*) ;; *) say "SKIP bad rel '$rel'"; skipped=$((skipped+1)); continue;; esac
  case "$real" in "$SRC"/larnd-sim-jax/?*|/sdf/scratch/neutrino/pgranger/fit_result/?*) ;;
       *) say "SKIP unexpected source '$real'"; skipped=$((skipped+1)); continue;; esac
  case "$dst" in /sdf/data/neutrino/pgranger/larnd-sim-jax/?*) ;; *) say "SKIP bad dest '$dst'"; skipped=$((skipped+1)); continue;; esac
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$dst" ]; then say "ALREADY $rel"; continue; fi
  [ -e "$dst" ] || { say "SKIP missing copy $dst"; skipped=$((skipped+1)); continue; }
  b0=$(bytes "$real"); c0=$(count "$real"); b1=$(bytes "$dst"); c1=$(count "$dst")
  if [ "$b0" != "$b1" ] || [ "$c0" != "$c1" ]; then
    say "SKIP re-verify failed $rel: bytes $b0/$b1 files $c0/$c1"; skipped=$((skipped+1)); continue
  fi
  if [ -L "$link" ]; then            # symlink into scratch: repoint, then drop the scratch copy
    ln -sfn "$dst" "$link" && rm -rf -- "$real"
  else
    rm -rf -- "$link" && ln -s "$dst" "$link"
  fi
  if [ -L "$link" ] && [ -e "$link" ]; then ok=$((ok+1)); else say "ERROR after swap: $rel"; fi
done < "$MAN"
say "=== swap done: $ok swapped, $skipped skipped"
df -h /sdf/group/neutrino | tail -1 | tee -a "$LOG"
