#!/bin/bash
# ENDFIT (h2) (26 Oct 04:27): wait for the 6 fast continuations with matched chain LR, then sacct + read_h2.py.
J=41984914,41984915,41984916,41984917,41984918,41984919
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/read_h2.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
