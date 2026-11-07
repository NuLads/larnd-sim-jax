#!/bin/bash
# ENDFIT (i): early health check of the G arm -- returns when every G run has written its 14100 checkpoint (end of the
# first sweep epoch) or left the queue; prints the per-run sweep summary of that first sweep.
F=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result; LG=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs
G=$(awk '{print $2}' /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/opt_jobs.txt)
while true; do n=0
  for X in EXS1 EXS2 EXS4 KN10N1 KN10N2 KN10N4; do j=$(awk -v x=$X '$1==x{print $2}' /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/opt_jobs.txt)
    if ls $F/sci_full_${X}G/history_iter14100_*.pkl >/dev/null 2>&1 || [ -z "$(squeue -h -j $j -o %i 2>/dev/null)" ]; then n=$((n+1)); fi; done
  [ $n -eq 6 ] && break; sleep 120; done
for X in EXS1 EXS2 EXS4 KN10N1 KN10N2 KN10N4; do j=$(awk -v x=$X '$1==x{print $2}' /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/opt_jobs.txt)
  f=$LG/${X}G-$j.out; echo "== ${X}G ($j): $(sacct -j $j -X -n -o State,Elapsed | head -1)"
  grep -a "\[GEOM-SWEEP\] it14" $f | sed 's/^.*INFO - //' | awk '{for(i=1;i<=NF;i++){if($i=="dJ"&&$(i-2)!="best"){s+=$(i+1);n++} if($i=="lowers")r++}} END{printf "   newton steps accepted %d, rejected %d, summed dJ %.0f\n", n, r, s}'
  grep -a "Error\|Traceback\|Killed\|OOM\|out of memory" $f | grep -v "cuda\|CUDA\|plugin" | head -3
  ls -la $F/sci_full_${X}G/ 2>/dev/null | grep history | awk '{print "   ", $NF}'; done
