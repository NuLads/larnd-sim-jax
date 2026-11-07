#!/bin/bash
# UQ TEST: wait until every job of a set is in a terminal state, then print the states. usage: wait_set.sh <label> <jobid...>
L=$1; shift; IDS=$(echo "$@" | tr ' ' ',')
while true; do
  st=$(sacct -X -n -P -j $IDS -o JobID,JobName,State,Elapsed 2>/dev/null)
  n=$(echo "$st" | grep -c .); nt=$(echo "$st" | grep -c "|COMPLETED|\||FAILED|\||CANCELLED\||TIMEOUT|\||OUT_OF_ME\||NODE_FAIL|")
  if [ "$n" -eq $# ] && [ "$nt" -eq $# ]; then echo "SET $L DONE $(date +%T)"; echo "$st"; exit 0; fi
  sleep 300
done
