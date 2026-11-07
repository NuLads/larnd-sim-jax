#!/bin/bash
# ENDFIT (i): wait for all 18 jobs (G, S, SL x EXS/KN10N 1/2/4), then sacct + read_opt.py (RO_ARMS=S,G,SL).
J=$(awk '{print $2","$3","$4}' /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/opt_jobs.txt | paste -sd,)
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 300; done
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L RO_ARMS=S,G,SL \
  python3 $E/read_opt.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
