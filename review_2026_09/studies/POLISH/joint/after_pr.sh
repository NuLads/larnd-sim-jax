#!/bin/bash
# After a re-heat pair (PRA<s>, PRC<s>) finishes: submit the iterated joint step (fitted geometry, nsteps 3) at both
# 14 000 end points (REVIEW.md 21-22 Oct rule), throttled to <= 6 jobs, then wait for them.
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; s=$1
until [ -z "$(squeue -h -u $USER -n PRA$s,PRC$s -o %i)" ]; do sleep 120; done
sacct -n -X -S 2026-10-21T23:00 -u $USER -o JobName%10,State,Elapsed | grep "PR[AC]$s "
IDS=""
for r in PRC PRA; do
  ls /sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_$r$s/history_iter14000_*.pkl >/dev/null || { echo "no 14000 checkpoint for $r$s"; continue; }
  until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 60; done
  id=$(cd /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=07:00:00 --mem-per-cpu=48g --job-name=jit_$r$s run_paper.sbatch "export LARND_JAX_LRU=32768 LARND_UNIQUE_BUCKET=1 LARND_POSRES_VEC=1; python3 $D/joint_polish2.py --run $r --sample $s --iter 14000 --eps 3e-3 --nsteps 3 --out $D/iter_$r$s.json --state $D/iter_$r${s}_state.npz")
  echo "$(date +%T) jit_$r$s $id" | tee -a $D/submitted.txt; IDS="$IDS,$id"; sleep 3
done
IDS=${IDS#,}; [ -z "$IDS" ] && exit 1
until [ -z "$(squeue -h -j $IDS -o %i 2>/dev/null)" ]; do sleep 300; done
echo "jit pair $s done"; sacct -n -X -j $IDS -o JobName%12,State,Elapsed
