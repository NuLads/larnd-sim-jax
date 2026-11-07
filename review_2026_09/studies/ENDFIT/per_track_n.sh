#!/bin/bash
# CPU per-track residuals vs truth at 20000 for the (n) end points (exploratory; (n) is complete).
L=/sdf/group/neutrino/pgranger/larnd-sim-jax; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
run1() { cd $D && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu \
  PYTHONPATH=$L/src:$L python3 per_track_res.py $1 '' 20000 > $D/per_track_$1_20000.out 2>&1; echo "done $1"; }
RUNS="$*"; n=0
for r in $RUNS; do
  [ -s $D/per_track_${r}_20000.json ] && { echo "skip $r"; continue; }
  run1 $r & n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
done; wait; echo ALL DONE
