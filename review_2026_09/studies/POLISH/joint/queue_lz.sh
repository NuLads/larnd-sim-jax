#!/bin/bash
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; S=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED
until grep -q "SAF1 " $D/submitted.txt; do sleep 120; done; sleep 30
until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 60; done
id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=05:00:00 --job-name=BENCHLZ --output=$S/lazy/%x-%j.out run_knot.sbatch fit $S/lazy/job_lz.sh)
echo "$(date +%T) BENCHLZ $id (lazy ROI bench: lazy arm then control, 600 it each)" | tee -a $D/submitted.txt $S/submitted.txt
