#!/bin/bash
# ENDFIT option (b), 24 Oct: CPU checks before the GPU block (short fit, 2 batches x 100 cm, 6 iterations).
#  1. head_def vs live_def: the new LARND_ENDPOINT_DRIFT_ZERO_MEAN code, off, leaves the production path BITWISE.
#  2. ep_def vs ep_off: the endpoint-fit path (LARND_FIT_ENDPOINTS=1, 1 mm smear) under the speed defaults vs every
#     speed flag off -> round-off level (the speed checks never covered this path).
#  3. ep_zm (constraint on) and ep_zm_rel (+ LARND_CHAIN_FREEZE_DRIFT=0, the continuation's freeze release): batch
#     sum of the endpoint drift shifts ~0 (check_zero_mean.py); ep_def for contrast.
# usage: bash cpu_check_b.sh   (login node; runs in two parallel streams)
O=/sdf/scratch/neutrino/pgranger/opt; P=$O/equiv/prof2; E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT
D=$E/cpu; S=$(date +%H%M); L=/sdf/group/neutrino/pgranger/larnd-sim-jax; H=$E/head_tree
SIF=$L/larndsim-jax_main.sif
mkdir -p $D
[ -d $H/optimize ] || { mkdir -p $H && git -C $L archive HEAD optimize src | tar -x -C $H; }
EP="LARND_FIT_ENDPOINTS=1 LARND_ENDPOINT_SMEAR_MM=1.0"
OFF="LARND_SIG_REMAT=0 LARND_SIG_DROP_GARBAGE=0 LARND_SIG_COMPACT=0 LARND_RESP_CUM_CACHE=0 LARND_PPP_JIT=0
 LARND_FITSIDE_JIT=0 LARND_OPT_JIT=0 LARND_HITS_PAD=0 LARND_FEE_PIXEL_STOP=0 LARND_FEE_PEEL0=0 LARND_FEE_CMDBUF=0
 LARND_FEW_SYNC=0 LARND_STRONG_INIT=0 LARND_SEARCH_UNROLLED=0 LARND_MLIR_SHARED_CTX=0 LARND_SEG_PAD=0 LARND_ROI_SKIP=0
 LARND_PRECOMP_LEAN=0 LARND_FEE_FAST=0 LARND_FEE_FAST_LOGMERGE=0 LARND_CHAIN_FLAT_ADAM=0 LARND_FITSIDE_FAST=0
 LARND_HOST_BATCH_CACHE=0 LARND_LOSS_FAST=0 LARND_SPLIT_T_FAST=0 LARND_WARP_ONE_SCATTER=0"
run () {  # tree name ENV=VAL...
  T=$1; N=$2; shift 2
  ( for kv in "$@"; do export "$kv"; done
    apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu RUNDIR=$D/$N-$S BL=100 SAVE_FREQ=100000 \
      bash $P/run_fit.sh $T epb_$N 6 2 --cpu_only ) > $D/$N-$S.log 2>&1
  echo "cpu $N rc=$? $(date +%T)"
}
( run $H head_def; run $L ep_def $EP; run $L ep_zm $EP LARND_ENDPOINT_DRIFT_ZERO_MEAN=1 ) &
( run $L live_def; run $L ep_off $EP $OFF; run $L ep_zm_rel $EP LARND_ENDPOINT_DRIFT_ZERO_MEAN=1 LARND_CHAIN_FREEZE_DRIFT=0 ) &
wait
h () { ls $D/$1-$S/fit_result/epb_$1/history_iter*.pkl | tail -1; }
C="apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $SIF env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L"
echo "== 1. head_def vs live_def (must be BITWISE)"; $C python3 $P/tools/cmp_hist.py "$(h head_def)" "$(h live_def)" 2>&1 | tail -3
echo "== 2. ep_def vs ep_off (round-off level expected)"; $C python3 $P/tools/cmp_rel.py "$(h ep_off)" "$(h ep_def)" 2>&1 | tail -6
echo "== 3. endpoint drift sums (ep_def: free; ep_zm, ep_zm_rel: must be ~0)"
$C python3 $E/check_zero_mean.py "$(h ep_def)" "$(h ep_zm)" "$(h ep_zm_rel)" 2>&1 | grep -v "^ERROR\|^Traceback\|^RuntimeError\|^  File\|^    "
grep -h "endpoint drift zero-mean constraint\|FROZEN on" $D/ep_zm-$S.log $D/ep_zm_rel-$S.log | sed 's/.*\[CHAIN\]/[CHAIN]/' | sort -u | head -8
