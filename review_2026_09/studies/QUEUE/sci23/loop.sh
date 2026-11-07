#!/bin/bash
# 23 Oct science queue: submits queue.txt lines (NAME|knot|-|inner.sh  or  NAME|paper|HH:MM:SS|cmd) in order, keeping
# science jobs (names not starting with opt_) <= CAP (3 until 22:00, then 6 - opt jobs) and all jobs <= 6. Edit queue.txt
# freely; submitted names are recorded in done.txt. Idles when everything is submitted; touch stop to end it.
Q=/sdf/scratch/neutrino/pgranger/review_2026_09/QUEUE/sci23; K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; PJ=/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS
touch $Q/done.txt
while true; do
  line=$(grep -v '^#' $Q/queue.txt | while IFS='|' read -r n t w c; do grep -qx "$n" $Q/done.txt || { echo "$n|$t|$w|$c"; break; }; done)
  [ -f $Q/stop ] && exit 0
  [ -z "$line" ] && { sleep 300; continue; }
  IFS='|' read -r n t w c <<< "$line"
  tot=$(squeue -h -u $USER | wc -l); opt=$(squeue -h -u $USER -o %j | grep -c '^opt_'); sci=$((tot - opt))
  if [ $(date +%H) -ge 22 ] || [ $(date +%H) -lt 8 ]; then cap=$((6 - opt)); else cap=3; fi
  if [ $tot -lt 6 ] && [ $sci -lt $cap ]; then
    if [ "$t" = knot ]; then id=$(cd $K && sbatch --parsable --account=neutrino:dune-ml --job-name=$n run_knot.sbatch fit $c)
    else id=$(cd $PJ && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=$w --mem-per-cpu=48g --job-name=$n run_paper.sbatch "$c"); fi
    [ -n "$id" ] && { echo "$n" >> $Q/done.txt; echo "$(date +%T) $n $id" | tee -a $Q/submitted.txt; }
    sleep 20
  else sleep 120; fi
done
