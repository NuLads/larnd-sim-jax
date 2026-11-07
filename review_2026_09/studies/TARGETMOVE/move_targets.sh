#!/bin/bash
# 4 Nov (Pierre: "Please move them in data behind links"): move the kept target dirs target_* from the repo root on /sdf/group
# to /sdf/data/neutrino/pgranger/larnd-sim-jax/ (mirror of the repo layout) and leave a symlink in place. Only the dirs listed
# in manifest.txt ("name n_files n_bytes", built with every queued job's dirs excluded) are touched. Per dir: refuse if the
# destination exists; mv (removes the source only after a complete copy); check file count and bytes at the destination
# against the manifest; ln -s. Stops at the first mismatch. Log: moved.txt / errors.txt.
SRC=/sdf/group/neutrino/pgranger/larnd-sim-jax; DST=/sdf/data/neutrino/pgranger/larnd-sim-jax
W=/sdf/scratch/neutrino/pgranger/review_2026_09/TARGETMOVE
cd $SRC || exit 1
while read -r d nf nb; do
  [ -L "$d" ] && { echo "$d already a link, skipped" >> $W/moved.txt; continue; }
  [ -d "$d" ] || { echo "$d missing" >> $W/errors.txt; exit 1; }
  [ -e "$DST/$d" ] && { echo "$DST/$d exists" >> $W/errors.txt; exit 1; }
  mv "$d" "$DST/" || { echo "mv failed: $d" >> $W/errors.txt; exit 1; }
  n2=$(find "$DST/$d" -type f | wc -l); b2=$(du -sb "$DST/$d" | cut -f1)
  [ "$n2" = "$nf" ] && [ "$b2" = "$nb" ] || { echo "MISMATCH $d: files $nf -> $n2, bytes $nb -> $b2 (now at $DST/$d, no link made)" >> $W/errors.txt; exit 1; }
  ln -s "$DST/$d" "$d" || { echo "ln failed: $d" >> $W/errors.txt; exit 1; }
  echo "$d $nf $nb" >> $W/moved.txt
done < $W/manifest.txt
echo "DONE $(date +%T): $(grep -c . $W/moved.txt) moved" >> $W/moved.txt
