#!/bin/bash
U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
J=$(awk '{print $2}' $U/pilot2_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%12,State,Elapsed,NodeList | grep -E "UQ_|JobID|---"
for k in $(awk '{print $1}' $U/pilot2_jobs.txt); do f=$(ls /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/UQ_$k-*.out | tail -1); echo "$k: $(grep -o 'dumped the profile pieces.*' $f | cut -c1-300)"; done
ls -la $U/dumps/
cd $U && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_pilot2.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
