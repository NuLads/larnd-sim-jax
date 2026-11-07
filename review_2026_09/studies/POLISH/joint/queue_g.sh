#!/bin/bash
# 22 Oct: submit G1 (oracle loss split), PRF1 (freeze off + re-heat), then G2 (oracle joint step, after G1) as slots
# free up; <= 6 jobs under this user in total (1 left for the paper session).
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
PJ=/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS; SB="sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
SPEED="export LARND_JAX_LRU=32768 LARND_UNIQUE_BUCKET=1 LARND_POSRES_VEC=1"
slot() { until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 45; done; }
slot; g1=$(cd $PJ && $SB --time=01:30:00 --mem-per-cpu=48g --job-name=geomloss2_S1 run_paper.sbatch "$SPEED; python3 $D/check_geom_loss2.py KN10S 1")
echo "$(date +%T) geomloss2_S1 $g1" | tee -a $D/submitted.txt; sleep 5
slot; f1=$(cd $K && $SB --time=06:00:00 --job-name=PRF1 run_knot.sbatch fit $K/inner_sci_full_PRF1.sh)
echo "$(date +%T) PRF1 $f1 (freeze off + re-heat)" | tee -a $D/submitted.txt; sleep 5
until [ -z "$(squeue -h -j $g1 -o %i 2>/dev/null)" ]; do sleep 60; done
slot; g2=$(cd $PJ && $SB --time=10:00:00 --mem-per-cpu=48g --job-name=jog_KN10S1 run_paper.sbatch "$SPEED; python3 $D/joint_polish3.py --run KN10S --sample 1 --iter 10000 --eps 3e-3 --nsteps 6 --geom outlier --thr_um 200 --out $D/og_KN10S1.json --state $D/og_KN10S1_state.npz")
echo "$(date +%T) jog_KN10S1 $g2 (oracle: outliers > 200 um at truth, nsteps 6)" | tee -a $D/submitted.txt
sacct -n -X -j $g1 -o JobName%14,State,Elapsed
