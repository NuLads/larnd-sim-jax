#!/bin/bash
# UQ TEST: chain watchdog. Exits as soon as a job ends in a non-COMPLETED terminal state (a failed stage blocks its afterok
# dependents forever) or when every job is COMPLETED. usage: watch_health.sh <jobid...>
IDS=$(echo "$@" | tr ' ' ',')
while true; do
  st=$(sacct -X -n -P -j $IDS -o JobID,JobName,State,Elapsed 2>/dev/null)
  bad=$(echo "$st" | grep -E "\|(FAILED|CANCELLED[^|]*|TIMEOUT|OUT_OF_ME[^|]*|NODE_FAIL|PREEMPTED)\|")
  if [ -n "$bad" ]; then echo "PROBLEM $(date +%T)"; echo "$bad"; exit 1; fi
  nc=$(echo "$st" | grep -c "|COMPLETED|")
  if [ "$nc" -eq $# ]; then echo "ALL COMPLETED $(date +%T)"; exit 0; fi
  sleep 300
done
