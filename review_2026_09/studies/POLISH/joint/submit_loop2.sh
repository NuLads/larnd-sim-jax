#!/bin/bash
# As submit_loop.sh, but leaves R slots free for hand-submitted jobs: submits only while (jobs of this user, queued or
# running, the paper session's included) < 6 - R, with R read from reserve.txt at every check (default 0).
# usage: submit_loop2.sh EPS ITER NAME[@ITER]...
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint; EPS=$1; ITER=$2; shift 2
IDS=""
cap() { R=$(cat $D/reserve.txt 2>/dev/null); echo $((6 - ${R:-0})); }
for spec in "$@"; do
  name=${spec%@*}; it=$ITER; [ "$spec" != "$name" ] && it=${spec#*@}
  run=${name:0:5}; rest=${name:5}; s=${rest%%[!0-9]*}; suf=${rest#$s}
  until [ "$(squeue -h -u $USER | wc -l)" -lt "$(cap)" ]; do sleep 120; done
  id=$(cd /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 \
       --time=05:00:00 --job-name=jnt_$name run_paper.sbatch \
       "python3 $D/joint_polish.py --run $run --sample $s ${suf:+--suffix $suf} --iter $it --eps $EPS --out $D/joint_$name.json")
  echo "$(date +%T) jnt_$name $id (eps $EPS, iter $it)" | tee -a $D/submitted.txt; IDS="$IDS,$id"; sleep 5
done
IDS=${IDS#,}; echo "all submitted: $IDS"
until [ -z "$(squeue -h -j $IDS -o %i 2>/dev/null)" ]; do sleep 300; done
echo "joint steps done"; sacct -n -X -j $IDS -o JobName%14,State,Elapsed
