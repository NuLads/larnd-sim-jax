#!/bin/bash
# 22 Oct: the calibration half of the PRF1 rule: iterated joint step (fitted geometry, nsteps 3) at the PRA1 and PRF1
# 14 000 end points, throttled to <= 6 jobs.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; PJ=/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS
for r in PRA PRF; do
  until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 60; done
  id=$(cd $PJ && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=07:00:00 --mem-per-cpu=48g --job-name=jit_${r}1 run_paper.sbatch "export LARND_JAX_LRU=32768 LARND_UNIQUE_BUCKET=1 LARND_POSRES_VEC=1; python3 $D/joint_polish2.py --run $r --sample 1 --iter 14000 --eps 3e-3 --nsteps 3 --out $D/iter_${r}1.json --state $D/iter_${r}1_state.npz")
  echo "$(date +%T) jit_${r}1 $id (PRF1 rule, calibration half)" | tee -a $D/submitted.txt; sleep 70
done
