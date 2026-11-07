#!/bin/bash
# wait for the cache-test pipeline KN10N3PC (jobs 42401876-78) to finish; print first-epoch times and walls vs KN10N3P
while [ -n "$(squeue -h -j 42401876,42401877,42401878 -o %i 2>/dev/null)" ]; do sleep 120; done
G=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs
ep () { grep -a -o -E "^?[ ]*100/$2 \[[0-9:]+<|[^0-9]100/$2 \[[0-9:]+<" $G/$1 | head -1 | grep -o -E "\[[0-9:]+" | tr -d '['; }
for st in "" FR H; do
  n=$([ -z "$st" ] && echo 10000 || { [ $st = FR ] && echo 4000 || echo 6000; })
  a=$(ls $G/KN10N3P$st-4237*.out); b=$(ls $G/KN10N3PC$st-4240*.out)
  echo "stage ${st:-fit}: first epoch P $(ep $(basename $a) $n) PC $(ep $(basename $b) $n); cache line: $(grep -a -c 'pipe_KN10N3PC (mode=shared' $b)"
done
sacct -j 42371063,42371064,42371065,42401876,42401877,42401878 -X -n -o JobName%12,Elapsed,State,NodeList
du -sh /sdf/scratch/neutrino/pgranger/jax_job_cache/pipe_KN10N3PC
