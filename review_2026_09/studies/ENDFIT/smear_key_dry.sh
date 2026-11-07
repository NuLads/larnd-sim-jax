#!/bin/bash
# ENDFIT (g), 25 Oct: CPU dry run of LARND_ENDPOINT_SMEAR_KEY=entry (per-track smear) on samples 1 and 2 as EXP<s>
# (100 batches x 400 cm, 1 mm, ends held; LARND_SEED_DIAG_ONLY=1: contexts built, no step, checkpoint saved with
# training_history['endpoint_smear']), then ENDFIT/smear_key_check.py.
O=/sdf/scratch/neutrino/pgranger/opt; P=$O/equiv/prof2; E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT
D=$E/cpu; S=$(date +%H%M); L=/sdf/group/neutrino/pgranger/larnd-sim-jax; SIF=$L/larndsim-jax_main.sif
mkdir -p $D
run () {  # name data_seed ENV=VAL...   (run_fit.sh hard-codes --data_seed 1; the appended one wins)
  N=$1; DS=$2; shift 2
  ( for kv in "$@"; do export "$kv"; done
    apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu RUNDIR=$D/$N-$S BL=400 SAVE_FREQ=100000 \
      bash $P/run_fit.sh $L epb_$N 200 100 --cpu_only --data_seed $DS ) > $D/$N-$S.log 2>&1
  echo "cpu $N rc=$? $(date +%T)"
}
( run key_s1 1 LARND_SEED_DIAG_ONLY=1 LARND_ENDPOINT_SMEAR_MM=1.0 LARND_ENDPOINT_SMEAR_KEY=entry ) &
( run key_s2 2 LARND_SEED_DIAG_ONLY=1 LARND_ENDPOINT_SMEAR_MM=1.0 LARND_ENDPOINT_SMEAR_KEY=entry ) &
wait
h () { ls $D/$1-$S/fit_result/epb_$1/history_iter*.pkl | tail -1; }
for n in key_s1 key_s2; do grep -h "SEED-DIAG\|Traceback\|Error" $D/$n-$S.log | grep -v "cuda\|CUDA\|plugin" | tail -3; done
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/smear_key_check.py "$(h key_s1)" "$(h key_s2)" 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
