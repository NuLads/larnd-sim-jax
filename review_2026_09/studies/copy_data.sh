#!/bin/bash
# STEP 1 of the data move (non-destructive): copy data from /sdf/group to /sdf/data/neutrino/pgranger
# and verify each item (file count + apparent bytes). Nothing is deleted or relinked here; the
# verified list is written to copy_manifest.txt for the (separately approved) swap-to-symlink step.
set -uo pipefail
SRC=/sdf/group/neutrino/pgranger
DST=/sdf/data/neutrino/pgranger
OUT=/sdf/scratch/neutrino/pgranger/review_2026_09
LOG=$OUT/copy_data.log; MAN=$OUT/copy_manifest.txt
mkdir -p "$DST"; : > "$MAN"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { find -L "$1" -type f 2>/dev/null | wc -l; }
bytes() { du -sbL --apparent-size "$1" 2>/dev/null | cut -f1; }

copy_one() {   # $1 = path relative to $SRC (file, dir, or symlink to a scratch dir)
  local rel="$1" s="$SRC/$1" d="$DST/$1" real
  real=$(readlink -f "$s")
  mkdir -p "$(dirname "$d")"
  if [ -d "$real" ]; then rsync -a "$real/" "$d/"; else rsync -a "$real" "$d"; fi
  local rc=$? nb0 nb1 cb0 cb1
  nb0=$(bytes "$real"); cb0=$(count "$real"); nb1=$(bytes "$d"); cb1=$(count "$d")
  if [ $rc -eq 0 ] && [ "$nb0" = "$nb1" ] && [ "$cb0" = "$cb1" ]; then
    say "VERIFIED $rel ($(numfmt --to=iec "$nb1"), $cb1 files) from $real"
    printf '%s\t%s\t%s\n' "$rel" "$real" "$d" >> "$MAN"
  else
    say "MISMATCH $rel: rc=$rc bytes $nb0 vs $nb1, files $cb0 vs $cb1"
  fi
}

say "=== copy start"
cd "$SRC/larnd-sim-jax"
for e in fit_result/*; do
  [ "$e" = "fit_result/hessian_analysis" ] && continue                 # holds the one tracked file
  git ls-files --error-unmatch "$e" >/dev/null 2>&1 && { say "SKIP tracked: $e"; continue; }
  copy_one "larnd-sim-jax/$e"
done
for f in *.h5 *.root *.npy *.pkl; do
  [ -f "$f" ] && [ ! -L "$f" ] || continue
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue
  [ "$(stat -c %s "$f")" -lt 50000000 ] && continue
  copy_one "larnd-sim-jax/$f"
done
say "=== copy done: $(wc -l < "$MAN") items verified"
