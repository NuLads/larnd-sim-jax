#!/bin/bash
# ENDFIT (g) (26 Oct 05:29): wait for all 18 jobs (6 chains: fit -> FR @ 14000 -> L @ 44000), then sacct + read_g.py.
J=41986747,41986748,41986749,41986750,41986751,41986752,41986753,41986754,41986756,41986757,41986758,41986759,41986760,41986761,41986762,41986763,41986764,41986765
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 300; done
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/read_g.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
