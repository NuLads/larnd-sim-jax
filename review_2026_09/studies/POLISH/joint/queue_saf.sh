#!/bin/bash
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
until grep -q "SA6 " $D/submitted.txt; do sleep 120; done; sleep 30
until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 60; done
id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=06:00:00 --job-name=SAF1 run_knot.sbatch fit $K/inner_sci_full_SAF1.sh)
echo "$(date +%T) SAF1 $id (gate + freeze release continuation from SA1)" | tee -a $D/submitted.txt
