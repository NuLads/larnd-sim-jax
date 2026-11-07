#!/bin/bash
# wait until all given job ids are observed in a terminal state via sacct
ids="$@"
while true; do
  alldone=1
  for j in $ids; do
    st=$(sacct -j $j -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
    case "$st" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) ;;
      *) alldone=0;;
    esac
  done
  [ $alldone = 1 ] && break
  sleep 30
done
for j in $ids; do sacct -j $j -X -n -o JobID,State,Elapsed,NodeList; done
