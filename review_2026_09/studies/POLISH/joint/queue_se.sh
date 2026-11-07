#!/bin/bash
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
until grep -q jdo_KN10S1 $D/submitted.txt; do sleep 60; done; sleep 10
until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 45; done
id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=10:00:00 --job-name=SE0_1 run_knot.sbatch fit $K/inner_sci_full_SE0_1.sh)
echo "$(date +%T) SE0_1 $id (full fit, LARND_HITS_SEED_E1=0)" | tee -a $D/submitted.txt
