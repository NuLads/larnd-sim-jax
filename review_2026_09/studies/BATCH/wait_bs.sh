#!/bin/bash
# 28 Oct batch-size probe: wait for the 4 jobs, then read_bs.py.
B=/sdf/scratch/neutrino/pgranger/review_2026_09/BATCH; J=$(awk '{print $2}' $B/bs_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
cd $B && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_bs.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
