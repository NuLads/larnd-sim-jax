#!/bin/bash
# ENDFIT (h) (26 Oct 03:18): wait for the 6 fast continuations, then sacct + read_fast.py.
J=41980603,41980604,41980605,41980606,41980607,41980608
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/read_fast.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
