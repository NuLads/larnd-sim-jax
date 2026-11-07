#!/bin/bash
# ENDFIT (k): wait for all 6 jobs of k_jobs.txt to leave the queue, then sacct + read_k.py.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; J=$(awk '{print $2}' $E/k_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName,State,Elapsed,NodeList
LG=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs
while read X j; do grep -a "JOINT-NEWTON\] step .* APPLIED\|no candidate\|Error\|RESOURCE_EXHAUSTED" $LG/$X-$j.out | grep -v "cuda\|CUDA\|plugin" | sed 's/^.*INFO - //' | cut -c1-400; done < $E/k_jobs.txt
cd $E && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_k.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
