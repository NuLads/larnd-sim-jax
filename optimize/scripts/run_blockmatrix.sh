#!/bin/bash
# THE BLOCK MATRIX: warm-start EVERY block from BT2, then freeze subsets and re-anneal the rest.
#
# THE QUESTION. The fit has three blocks -- calibration (C, 5 params), per-segment dEdx (D, ~4000
# nuisances) and per-track spline geometry (G). They are optimised simultaneously and, per the
# degeneracy matrix (report fig 58), they are coupled: each block's optimum MOVES as the others
# move. So each block may be sitting where it is not because the data wants it there, but because
# it co-adapted to the others while they were themselves wrong. Freezing a block at its fitted
# value and re-annealing the rest asks, for every subset: is there a better point reachable once
# this block stops moving?
#
# ALL SEVEN ARMS START FROM BT2'S CONVERGED VALUES -- nothing restarts from scratch:
#   calibration  SCIINIT        (--set_init_params, verified exact in report fig 62)
#   geometry     SCICHAININIT   (chain_cache, keyed by track_id)
#   dEdx         SCIDEDXINIT    (dedx_cache; NEW -- this knob did not exist before today)
#
#   arm    frozen        fitted        what it asks
#   ----   -----------   -----------   -------------------------------------------------------
#   BM0    none          C, D, G       CONTROL: what does a re-anneal alone buy? Every other arm
#                                      must be read against this, not against BT2.
#   BMC    C             D, G          do the nuisances improve once calibration stops moving?
#   BMD    D             C, G          the dEdx block is prior-limited (fig 46/47) -- does pinning
#                                      it free calibration?
#   BMG    G             C, D          = ALT/ALTW, but now with dEdx warm-started too
#   BMCD   C, D          G             geometry alone, against a fully static nuisance background
#   BMCG   C, G          D             dEdx alone
#   BMDG   D, G          C             calibration alone -- the cleanest read of what the data says
#                                      about the 5 physics parameters
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE SCHEDULE, WHICH IS THE WHOLE REASON THIS IS NOT A REPEAT OF FIG 62.
#
# ALTW failed twice and both failures are fixed here:
#   (1) its dEdx block restarted COLD and dragged calibration off its warm start within ~200
#       iterations. SCIDEDXINIT fixes that; it is why the knob was written.
#   (2) it ran 3 000 iterations on a schedule that anneals over 10 000, so it was read out with
#       its calibration LR at 0.059 of peak against BT2's 8.0e-5 -- a factor 700 -- and its
#       "result" was a sample of an oscillation.
#
# So BOTH schedules are COMPRESSED to land on BT2's terminal LR at iteration 3 000:
#   SCILRDECAY  0.91   -> 0.73     calibration decays per EPOCH: 0.73^30  = 7.9e-5 ~ 0.91^100
#   SCICHDECAY  0.9997 -> 0.999    geometry decays per ITERATION: 0.999^3000 = 0.050 = 0.9997^10000
# Every arm is then read at the same schedule phase as BT2's endpoint, and the comparison is
# between converged numbers rather than between a frozen one and a moving one.
#
# The peak LRs are left at BT2's values deliberately. Starting from a converged point at peak LR is
# a genuine re-anneal: it kicks the FITTED blocks away and lets them re-converge, which is the only
# way to discover a different optimum. A frozen block cannot be kicked, so nothing that is being
# tested can be erased -- that was ALTW's specific bug, not a property of warm starts.
#
# n = 2 SEEDS. This is a SCREEN over 7 arms, sized to find a large effect, not to resolve a small
# one -- fig 62 is the standing reminder that a median over 2 seeds is the mean of 2. Any arm that
# wins here needs confirming at n = 6 before it is quoted.
set -euo pipefail
cd /sdf/group/neutrino/pgranger/larnd-sim-jax

SEEDS=${SEEDS:-"0 1"}
ITERS=${ITERS:-3000}
ONLY=${ONLY:-}
ACCOUNTS=(neutrino:cider-nu neutrino:dune-ml mli:cider-ml mli:nu-ml-dev)
CKDIR=fit_result/sci_full_BT2
BASE="SCIMODE=full,SCINBATCH=100,SCILEN=400,SCIBASIS=spline,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
# compressed so 3 000 iterations end where BT2's 10 000 did -- see the block comment above
SCHED="SCICLR=1e-2,SCICHDECAY=0.999,SCILRDECAY=0.73"

# name | freeze_calib | freeze_dedx | freeze_geom
ARMS=("BM0|0|0|0" "BMC|1|0|0" "BMD|0|1|0" "BMG|0|0|1"
      "BMCD|1|1|0" "BMCG|1|0|1" "BMDG|0|1|1")

k=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r NAME FC FD FG <<< "${arm}"
  if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${NAME} "* ]]; then continue; fi
  for S in ${SEEDS}; do
    CK=$(ls ${CKDIR}/history_iter10000_*seed${S}.pkl 2>/dev/null | head -1)
    [ -z "${CK}" ] && { echo "SKIP ${NAME} seed ${S}: no BT2 checkpoint"; continue; }

    # BT2's converged calibration for this seed, colon-separated (sbatch --export forbids spaces).
    # Tail median, not the last sample: BT2 is annealed flat by then (spread 0.001%) so the two
    # agree, but the tail median is the quantity every other comparison in the report uses.
    INIT=$(/sdf/group/neutrino/pgranger/conda/bin/python3.12 -c "
import pickle,numpy as np
h=pickle.load(open('${CK}','rb')); out=[]
for p in ['Ab','eField','tran_diff','long_diff','lifetime']:
    v=np.ravel(h[p+'_iter']); out += [p, '%.8g' % np.median(v[int(len(v)*.8):])]
print(':'.join(out))")

    # dEdx freeze is expressed as a freeze ITERATION; 0 = frozen from the very first step.
    DFZ=""
    [ "${FD}" = "1" ] && DFZ="SCIDEDXFREEZE=0,"

    ACC=${ACCOUNTS[$((k % ${#ACCOUNTS[@]}))]}; k=$((k + 1))
    echo "${NAME} seed ${S} on ${ACC}  (freeze C=${FC} D=${FD} G=${FG})"
    sbatch --account="${ACC}" --job-name="${NAME}" --array="${S}-${S}" --time=12:00:00 \
      --export=ALL,${BASE},${REFIT},${SCHED},SCITAG=_${NAME},SCIITER=${ITERS},\
SCIINIT=${INIT},SCICHAININIT=${CK},SCIDEDXINIT=${CK},\
SCICALIBFREEZE=${FC},${DFZ}SCICHFREEZE=${FG} \
      optimize/scripts/start_sci_case.sh
  done
done
echo "submitted; results in fit_result/sci_full_BM*/"
