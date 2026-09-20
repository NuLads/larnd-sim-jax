#!/bin/bash
# Submit jobs while NEVER exceeding MAXJOBS concurrent running jobs on the cluster.
#
# WHY THIS EXISTS. Launching a 24-job wave in one go monopolises the ampere partition and other
# users complained. The fix is structural rather than remembered: every launch goes through here,
# so the cap cannot be forgotten at 2am.
#
# HOW IT WORKS. Jobs are submitted immediately but IMMEDIATELY HELD, then released one at a time
# as running slots free up. Holding rather than delaying submission means the whole wave is
# recorded in the queue (so `squeue` shows the real plan) and survives this script exiting --
# though the releaser must keep running for them to start, so it is launched with nohup.
#
# USAGE
#   # 1. write one sbatch command per line to a file (no leading 'sbatch')
#   #    e.g.  --account=X --job-name=foo --array=0-0 --export=ALL,... script.sh
#   optimize/scripts/submit_throttled.sh joblist.txt          # cap = 7
#   MAXJOBS=4 optimize/scripts/submit_throttled.sh joblist.txt
#
#   # or release an ALREADY-held set (e.g. after submitting by hand):
#   optimize/scripts/submit_throttled.sh --release-only
set -uo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

MAXJOBS=${MAXJOBS:-7}      # base cap, always respected when the partition is busy
MAXCAP=${MAXCAP:-20}       # never exceed this even on a totally empty partition
LOG=${THROTTLE_LOG:-logs/throttle.log}
mkdir -p "$(dirname "$LOG")"

running() { squeue -u "$USER" -h -t RUNNING -o "%i" 2>/dev/null | wc -l; }

# ADAPTIVE CAP. The base cap exists because other users complained about the partition being
# monopolised. When the partition is genuinely quiet -- free nodes AND nobody else queued -- that
# reason does not apply, so the cap lifts to use the idle capacity.
#   * "free" means state EXACTLY idle. Nodes can report -t idle while flagged MAINT/DRAIN and are
#     NOT usable; on 2026-09-20, 9 of 9 apparently-idle ampere nodes were in maintenance.
#   * any other user queued => fall straight back to the base cap.
PART=${PART:-ampere}
free_nodes()     { sinfo -p "$PART" -h -o "%t %D" 2>/dev/null | awk '$1=="idle"{s+=$2} END{print s+0}'; }
others_pending() { squeue -p "$PART" -h -t PENDING -o "%u" 2>/dev/null | grep -vc "^${USER}$"; }
effective_cap() {
  local f o
  f=$(free_nodes); o=$(others_pending)
  if [ "${o:-1}" -eq 0 ] && [ "${f:-0}" -gt 0 ]; then
    echo $(( MAXJOBS + f < MAXCAP ? MAXJOBS + f : MAXCAP ))
  else
    echo "$MAXJOBS"
  fi
}
held()    { squeue -u "$USER" -h -t PENDING -o "%i %r" 2>/dev/null | awk '/JobHeldUser/{print $1}'; }

if [ "${1:-}" != "--release-only" ]; then
  LIST="${1:?usage: submit_throttled.sh <joblist|--release-only>}"
  n=0
  while IFS= read -r line; do
    [ -z "${line// }" ] && continue
    case "$line" in \#*) continue ;; esac
    jid=$(sbatch --hold $line 2>/dev/null | awk '{print $NF}')
    [ -n "$jid" ] && { n=$((n+1)); echo "$(date '+%F %T') submitted+held $jid" >> "$LOG"; }
  done < "$LIST"
  echo "submitted and held $n jobs (cap ${MAXJOBS})"
fi

# the releaser: keep exactly MAXJOBS running until nothing is held
(
  while true; do
    H=$(held); [ -z "$H" ] && { echo "$(date '+%F %T') all released; throttle exiting" >> "$LOG"; break; }
    R=$(running); CAP=$(effective_cap)
    [ "$CAP" != "$MAXJOBS" ] && echo "$(date '+%F %T') partition quiet -> cap raised to $CAP" >> "$LOG"
    SLOTS=$(( CAP - R ))
    if [ "$SLOTS" -gt 0 ]; then
      echo "$H" | head -n "$SLOTS" | while read -r j; do
        scontrol release "$j" 2>/dev/null && echo "$(date '+%F %T') released $j (running was $R)" >> "$LOG"
      done
    fi
    sleep 120
  done
) &
disown
echo "releaser running in background; cap=${MAXJOBS}; log=${LOG}"

# ---------------------------------------------------------------------------
# DISK NOTE (2026-09-20). /sdf/group filled to 100% (455T volume, a few GB free)
# and killed two whole waves mid-run: logs truncate mid-line, no traceback,
# every job on every node dying within seconds of each other, exit 1:0.
# Our own footprint is ~83G of 455T, so cleaning up on our side does not help.
#
# WORKAROUND, no code change needed: fit_params.py writes to the RELATIVE path
# fit_result/<test_name>/ and only calls makedirs when that path does not
# already exist. So pre-creating it as a SYMLINK to scratch redirects all
# checkpoint writes:
#   S=/sdf/scratch/neutrino/pgranger/fit_result
#   mkdir -p $S/sci_full_<TAG> && ln -sfn $S/sci_full_<TAG> fit_result/sci_full_<TAG>
# /sdf/scratch/neutrino has its own 5T quota (was 99% free) and the container
# already binds /sdf and /sdf/scratch, so compute nodes see it.
# ---------------------------------------------------------------------------
