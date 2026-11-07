#!/bin/bash
# ENDFIT (f) (25 Oct 15:37): wait for the 6 long-continuation controls, then sacct + read_ctrl.py.
J=41932229,41932230,41932231,41932232,41932233,41932234
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 300; done
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/read_ctrl.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
