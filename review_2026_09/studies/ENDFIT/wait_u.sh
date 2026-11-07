#!/bin/bash
# ENDFIT (u): wait for every job in u_jobs.txt to leave the queue, then sacct + release-log check + read_u.py.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
J=$(paste -sd, $E/u_jobs.txt)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%14,State,Elapsed,NodeList | grep -E "EXP|JobID|---"
for s in 7 8 9 10 11 12; do r=$(ls $K/logs/EXP${s}-*.out | tail -1); c=$(ls $K/logs/EXP${s}C75-*.out | tail -1)
  echo "s$s: ref coarse lines $(grep -c 'coarse-to-fine geometry' $r); C75: $(grep -o -m1 'until iteration [0-9]*' $c)"; done
cd $E && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_u.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
