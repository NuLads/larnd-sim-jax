#!/bin/bash
# Submits the 12 Newton polishes (REVIEW.md 21 Oct) as slots free up: never more than 6 of my jobs queued or running;
# a replica is polished only after its fit job has left the queue with its 1e4 checkpoint. Then waits for all of them.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/replica; FR=/sdf/scratch/neutrino/pgranger/fit_result_staging
declare -A JOB=([KN10V1b]=41484832 [KN10V1c]=41484833 [KN10V2b]=41484834 [KN10V2c]=41484835 [KN10V3b]=41484836 [KN10V3c]=41484837)
IDS=""
for name in KN10V1 KN10V2 KN10V3 KN10S1 KN10S2 KN10S3 KN10V1b KN10V1c KN10V2b KN10V2c KN10V3b KN10V3c; do
  run=${name:0:5}; s=${name:5:1}; suf=${name:6}
  while :; do
    if [ -n "${JOB[$name]}" ] && [ -n "$(squeue -h -j ${JOB[$name]} -o %i 2>/dev/null)" ]; then sleep 120; continue; fi
    if ! ls $FR/sci_full_$name/history_iter10000_*.pkl >/dev/null 2>&1; then echo "$(date +%T) $name: no 1e4 checkpoint, SKIPPED" | tee -a $D/submitted.txt; continue 2; fi
    [ "$(squeue -h -u $USER | wc -l)" -lt 6 ] && break; sleep 120
  done
  id=$(cd /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 \
       --time=02:00:00 --job-name=pol_$name run_paper.sbatch \
       "python3 /sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/newton_polish.py --run $run --sample $s ${suf:+--suffix $suf} --iter 10000 --out $D/polish_$name.json")
  echo "$(date +%T) pol_$name $id" | tee -a $D/submitted.txt; IDS="$IDS,$id"; sleep 5
done
IDS=${IDS#,}; echo "all submitted: $IDS"
until [ -z "$(squeue -h -j $IDS -o %i 2>/dev/null)" ]; do sleep 300; done
echo "polish replicas done"; sacct -n -X -j $IDS -o JobName%14,State,Elapsed
