#!/bin/bash
# 22 Oct: SA block, samples 2-6 (alias gate 500 um, fresh 1e4 fits), throttled to <= 6 jobs of this user.
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
for s in 2 3 4 5 6; do
  until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 60; done
  id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=10:00:00 --job-name=SA$s run_knot.sbatch fit $K/inner_sci_full_SA$s.sh)
  echo "$(date +%T) SA$s $id (SA block: alias gate 500 um, sample $s)" | tee -a $D/submitted.txt; sleep 90
done
