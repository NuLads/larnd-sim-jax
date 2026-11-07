#!/bin/bash
# wait until the cache-test FR stage (KN10N3PCFR, job 42401877) has done its first epoch, then print its timing lines
F=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/KN10N3PCFR-42401877.out
until [ -f $F ] && grep -a -q -E " 1[0-9]{2}/4000 \[" $F; do
  st=$(squeue -h -j 42401877 -o %T 2>/dev/null); [ -z "$st" ] && [ ! -f $F ] && { echo "FR job gone without a log"; exit 1; }
  sleep 60; done
grep -a "Persistent JAX" $F | cut -c1-200
grep -a -o -E "[0-9]+/4000 \[[0-9:]+<" $F | awk -F'[ /[<]' '$1==1||$1==2||$1==10||$1==50||$1==100||$1==101||$1==120' | sort -n -u -k1,1 | tr '\n' ';'; echo
sacct -j 42401876 -X -n -o JobName,Elapsed,State
