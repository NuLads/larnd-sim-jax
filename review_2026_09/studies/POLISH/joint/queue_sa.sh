#!/bin/bash
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
until [ -f $D/SA1_GO ] && grep -q SE0_1 $D/submitted.txt; do sleep 60; done; sleep 10
until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 45; done
id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=10:00:00 --job-name=SA1 run_knot.sbatch fit $K/inner_sci_full_SA1.sh)
echo "$(date +%T) SA1 $id (full fit, e1 seed with alias gate 500 um)" | tee -a $D/submitted.txt
