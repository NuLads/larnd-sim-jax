#!/bin/bash
# Robust SLURM waiter: a failed squeue/sacct query must NOT be read as "job finished".
# Only a positively-observed terminal state ends the wait.
JOBS="$1"; MAXWAIT=${2:-14400}; T0=$(date +%s)
while true; do
  st=$(timeout 30 sacct -j "$JOBS" --format=JobID,State -n 2>/dev/null | grep -E "^[0-9]+(_[0-9]+)? " | awk '{print $2}')
  if [ -z "$st" ]; then
    echo "[wait] query failed or empty - retrying (NOT assuming completion)"
  else
    pend=$(echo "$st" | grep -cE "PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING" || true)
    if [ "$pend" -eq 0 ]; then echo "[wait] all terminal:"; echo "$st" | sort | uniq -c; break; fi
  fi
  [ $(( $(date +%s) - T0 )) -gt "$MAXWAIT" ] && { echo "[wait] TIMEOUT"; break; }
  sleep 45
done
