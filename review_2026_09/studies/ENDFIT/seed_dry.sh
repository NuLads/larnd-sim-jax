#!/bin/bash
# ENDFIT (d), 25 Oct: CPU dry run of the hit-based endpoint seed (LARND_SEED_DIAG_ONLY=1: every batch seeded once, no
# step, checkpoint saved) on sample 1 as EZS1 (100 batches x 400 cm, endpoints smeared 1 mm), plus the default-path
# check (HEAD vs live, 6 iterations, 2 x 100 cm: must be BITWISE).
O=/sdf/scratch/neutrino/pgranger/opt; P=$O/equiv/prof2; E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT
D=$E/cpu; S=$(date +%H%M); L=/sdf/group/neutrino/pgranger/larnd-sim-jax; H=$E/head_tree; SIF=$L/larndsim-jax_main.sif
mkdir -p $D
run () {  # tree name iterations nbatch BL ENV=VAL...
  T=$1; N=$2; IT=$3; NB=$4; B=$5; shift 5
  ( for kv in "$@"; do export "$kv"; done
    apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu RUNDIR=$D/$N-$S BL=$B SAVE_FREQ=100000 \
      bash $P/run_fit.sh $T epb_$N $IT $NB --cpu_only ) > $D/$N-$S.log 2>&1
  echo "cpu $N rc=$? $(date +%T)"
}
( run $L seed_dry 200 100 400 LARND_SEED_DIAG_ONLY=1 LARND_FIT_ENDPOINTS=1 LARND_ENDPOINT_SMEAR_MM=1.0 ) &
( run $H d_head_def 6 2 100; run $L d_live_def 6 2 100 ) &
wait
h () { ls $D/$1-$S/fit_result/epb_$1/history_iter*.pkl | tail -1; }
C="apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L"
echo "== default path: d_head_def vs d_live_def (must be BITWISE)"; $C python3 $P/tools/cmp_hist.py "$(h d_head_def)" "$(h d_live_def)" 2>&1 | tail -3
echo "== seed dry run checkpoint: $(h seed_dry)"
grep -h "SEED-DIAG\|Traceback\|Error" $D/seed_dry-$S.log | grep -v "cuda\|CUDA\|plugin" | tail -5
