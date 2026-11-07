#!/bin/bash
U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
J=$(awk '{print $2}' $U/pilot_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%12,State,Elapsed,NodeList | grep -E "UQ_|JobID|---"
for k in P3 O3 C3 P3e3; do f=$(ls /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/UQ_$k-*.out | tail -1); grep -o "dumped the profile pieces.*" $f | cut -c1-400; done
cd $U && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_pilot.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
