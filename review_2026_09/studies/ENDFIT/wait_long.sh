#!/bin/bash
# ENDFIT (e) (25 Oct 10:44): keep the EZS<s>L checkpoints at 16000, 20000, ... (iter % 4000 == 0) as hard links in
# long_ckpt/ (the fit deletes each checkpoint 2000 iterations later; same filesystem -> no copy), wait for the 6 jobs,
# then sacct + read_long.py.
J=41912038,41912039,41912040,41912041,41912042,41912043
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT
L=/sdf/group/neutrino/pgranger/larnd-sim-jax
keep () {
  for s in 1 2 4; do
    mkdir -p $E/long_ckpt/sci_full_EZS${s}L
    for f in $S/sci_full_EZS${s}L/history_iter*.pkl; do
      [ -e "$f" ] || continue
      it=$(basename $f | sed 's/history_iter\([0-9]*\)_.*/\1/')
      [ $((it % 4000)) = 0 ] && [ ! -e $E/long_ckpt/sci_full_EZS${s}L/$(basename $f) ] && ln $f $E/long_ckpt/sci_full_EZS${s}L/ && echo "$(date +%T) kept EZS${s}L @ $it"
    done
  done
}
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do keep; sleep 150; done
keep
sacct -j $J -X -o JobID,JobName%10,State,Elapsed,NodeList%14
apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch $L/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONPATH=$L/src:$L \
  python3 $E/read_long.py 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback\|^  File\|^    "
