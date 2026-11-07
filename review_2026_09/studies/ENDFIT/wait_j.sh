#!/bin/bash
# ENDFIT (j): wait for all 6 jobs of j_jobs.txt to leave the queue, then sacct + read_j.py.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; J=$(awk '{print $2}' $E/j_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName,State,Elapsed,NodeList
cd $E && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_j.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
