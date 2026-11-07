#!/bin/bash
# ENDFIT option (b) block (25 Oct 00:31): wait for all 18 jobs, then sacct + read_endc.py.
J="41873499 41873500 41873501 41873502 41873503 41873504 41873505 41873506 41873508 41873509 41873510 41873511 41873512 41873513 41873514 41873515 41873517 41873518"
L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j ${J// /,} -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j ${J// /,} -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_endc.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
