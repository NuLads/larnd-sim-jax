#!/bin/bash
# ENDFIT (t): wait for every job in t_jobs.txt to leave the queue, then sacct + release-log check + read_t.py.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
J=$(paste -sd, $E/t_jobs.txt)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%14,State,Elapsed,NodeList | grep -E "KN10N|JobID|---"
for s in 1 2 4 10 11 12; do f=$(ls $K/logs/KN10N${s}C75-*.out | tail -1); echo "$(basename $f): $(grep -o -m1 'until iteration [0-9]*' $f)"; done
cd $E && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_t.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
