#!/bin/bash
# 28 Oct: wait for the option 1 / 2 GPU smokes, then print their logs' key lines.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; LG=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs
J=$(awk '{print $2}' $E/k_smoke_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName,State,Elapsed,NodeList
while read X j; do f=$LG/$X-$j.out; echo "=================== $X ($j)"
  grep -a "EPOCH-ACCEL\|JOINT-NEWTON\] \(step\|it\)" $f | sed 's/^.*INFO - //' | cut -c1-600 | head -40
  grep -a "JOINT-NEWTON\] assemble" $f | sed 's/^.*INFO - //' | head -3
  grep -a "JOINT-NEWTON\] assemble" $f | awk '{for(i=1;i<=NF;i++) if($i=="|") t+=$(i+1)} END{if(NR) printf "assembly: %d batches, mean %.1f s/batch\n", NR, t/NR}'
  grep -a "Error\|error:\|Killed\|RESOURCE_EXHAUSTED\|out of memory" $f | grep -v "cuda\|CUDA\|plugin" | head -5
  tail -3 $f; done < $E/k_smoke_jobs.txt
