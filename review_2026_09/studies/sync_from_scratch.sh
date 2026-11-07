#!/bin/bash
# Refresh this copy of the study code from the scratch workspaces (see README.md).
# Copies every file listed in MANIFEST.txt (paths relative to the scratch review dir); never deletes.
set -e
SRC=/sdf/scratch/neutrino/pgranger/review_2026_09/
DST=$(cd "$(dirname "$0")" && pwd)
missing=$(cd "$SRC" && while read -r p; do [ -f "$p" ] || echo "$p"; done < "$DST/MANIFEST.txt")
[ -z "$missing" ] || { echo "missing on scratch (not copied):"; echo "$missing"; }
rsync -a --files-from=<(cd "$SRC" && while read -r p; do [ -f "$p" ] && echo "$p"; done < "$DST/MANIFEST.txt") "$SRC" "$DST/"
echo "synced; review with: git -C $DST status --short ."
