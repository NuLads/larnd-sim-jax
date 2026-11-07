#!/bin/bash
# cache test 2: wait for KN10N5PE (job 42437667) to finish its first epoch, cancel it, report the time and the new entries
J=42437667; L=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/KN10N5PE-$J.out
C=/sdf/scratch/neutrino/pgranger/jax_job_cache/pipe_ENS_test
while :; do
  if [ -f $L ] && grep -a -q -E "[^0-9]100/10000 \[[0-9:]+<" $L; then break; fi
  [ -z "$(squeue -h -j $J -o %i 2>/dev/null)" ] && { echo "job $J left the queue before epoch 1"; tail -5 $L; exit 1; }
  sleep 30
done
scancel $J; sleep 90
echo "KN10N5PE first epoch: $(grep -a -o -E "[^0-9]100/10000 \[[0-9:]+<" $L | head -1 | grep -o -E "\[[0-9:]+" | tr -d '[')  (cold KN10N5 P/CF/SD 09:10 / 09:24 / 09:15)"
echo "cache line: $(grep -a -c 'pipe_ENS_test (mode=shared' $L); entries $(ls $C | wc -l) (was 1984); $(du -sh $C | cut -f1)"
grep -a "=== start" $L; sacct -j $J -X -n -o JobName%10,Elapsed,State,NodeList
