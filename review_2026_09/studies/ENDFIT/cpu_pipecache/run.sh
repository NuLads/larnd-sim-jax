#!/bin/bash
# 29 Oct: CPU check of a per-PIPELINE persistent JAX cache (existing mode LARND_JAX_PERSISTENT_CACHE=shared with an
# explicit dir on /sdf/scratch; no code change). A cold / B warm same config (must be BITWISE vs A) / D default per-job
# cache (BITWISE vs A) / E warm, FR-like hyperparameters (freeze off, calib lr 1e-3, chain lr 2e-3): new cache entries.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/cpu_pipecache; L=/sdf/group/neutrino/pgranger/larnd-sim-jax
P=/sdf/scratch/neutrino/pgranger/opt/equiv/prof2; SIF=$L/larndsim-jax_main.sif; C=$E/cache_C
mkdir -p $C
run () {  # name mode extra-args...
  N=$1; M=$2; shift 2
  t0=$(date +%s)
  apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu RUNDIR=$E/$N BL=100 SAVE_FREQ=100000 \
    LARND_JAX_PERSISTENT_CACHE=$M JAX_COMPILATION_CACHE_DIR=$C bash $P/run_fit.sh $L pc_$N 4 2 --cpu_only "$@" > $E/$N.log 2>&1
  echo "$N rc=$? wall $(( $(date +%s) - t0 )) s; cache files now $(ls $C | wc -l)"
}
run A shared
run B shared
run D job
run E shared --lr 1e-3 --chain_lr 2e-3
h () { ls $E/$1/fit_result/pc_$1/history_iter*.pkl | tail -1; }
X="apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L"
for r in B D; do echo "== A vs $r (must be BITWISE)"; $X python3 $P/tools/cmp_hist.py "$(h A)" "$(h $r)" 2>&1 | grep -v -i "cuda\|plugin" | tail -3; done
