#!/bin/bash
# SAF1 then BENCHLZ, each when the user has < 6 jobs; shares the optimisation team's lock (opt/gpu.lock).
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; S=/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED
exec 9>/sdf/scratch/neutrino/pgranger/opt/gpu.lock
sub() {   # sub <name> <sbatch args...>
  while true; do
    flock 9
    if [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; then
      id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 "${@:2}")
      echo "$(date +%T) $1 $id" | tee -a $D/submitted.txt; sleep 15; flock -u 9; return
    fi
    flock -u 9; sleep 60
  done
}
sub "SAF1 (gate + freeze release continuation from SA1)" --time=06:00:00 --job-name=SAF1 run_knot.sbatch fit $K/inner_sci_full_SAF1.sh
sub "BENCHLZ (lazy ROI bench: lazy arm then control, 600 it each)" --time=05:00:00 --job-name=BENCHLZ --output=$S/lazy/%x-%j.out run_knot.sbatch fit $S/lazy/job_lz.sh
