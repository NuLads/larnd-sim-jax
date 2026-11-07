#!/bin/bash
# 22 Oct: G3 (truth-free filter) then G4 (oracle filter), each nsteps 3 from KN10S1, after G2 is in the queue; <= 6 jobs.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; PJ=/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS
SB="sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=08:00:00 --mem-per-cpu=48g"
SPEED="export LARND_JAX_LRU=32768 LARND_UNIQUE_BUCKET=1 LARND_POSRES_VEC=1"
slot() { until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 50; done; }
until grep -q jog_KN10S1 $D/submitted.txt; do sleep 60; done
slot; g3=$(cd $PJ && $SB --job-name=jdf_KN10S1 run_paper.sbatch "$SPEED; python3 $D/joint_polish3.py --run KN10S --sample 1 --iter 10000 --eps 3e-3 --nsteps 3 --drop flag --drop_q 0.8 --out $D/df_KN10S1.json --state $D/df_KN10S1_state.npz")
echo "$(date +%T) jdf_KN10S1 $g3 (G3: truth-free filter, flag q 0.8)" | tee -a $D/submitted.txt; sleep 5
slot; g4=$(cd $PJ && $SB --job-name=jdo_KN10S1 run_paper.sbatch "$SPEED; python3 $D/joint_polish3.py --run KN10S --sample 1 --iter 10000 --eps 3e-3 --nsteps 3 --drop outlier --thr_um 200 --out $D/do_KN10S1.json --state $D/do_KN10S1_state.npz")
echo "$(date +%T) jdo_KN10S1 $g4 (G4: oracle filter, |r| > 200 um)" | tee -a $D/submitted.txt
