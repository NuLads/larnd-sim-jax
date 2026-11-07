#!/bin/bash
# UQ: submit one profile-Hessian diagnostic of an end point. usage: submit_dump.sh <RUN> <DNAME> [LARND_X=value ...]
#   RUN = the end point's run name without sci_full_ (e.g. KN10N3PH); its stage script KNOT/inner_sci_full_<RUN>.sh and its
#   last checkpoint in fit_result_staging/sci_full_<RUN>/ are used. Output: UQ/dumps/<DNAME>.npz; the held run writes to
#   fit_result_staging/sci_full_UQ_<DNAME> (linked from the live fit_result/). Prints "<DNAME> <jobid>".
RUN=$1; DN=$2; shift 2
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
IN=$K/inner_sci_full_$RUN.sh; [ -f $IN ] || { echo "no $IN"; exit 2; }
CK=$(ls $S/sci_full_$RUN/history_iter*_*.pkl | sort -V | tail -1); [ -f "$CK" ] || { echo "no checkpoint for $RUN"; exit 2; }
[ -e $U/dumps/$DN.npz ] && { echo "$U/dumps/$DN.npz exists"; exit 2; }
r=sci_full_UQ_$DN
if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r exists and is not a link"; exit 2; fi
mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
apptainer exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu python3 $U/make_jn_dump.py $IN $CK $r $U/dumps/$DN.npz "$@" > $U/inner/inner_$DN.sh 2>/dev/null || { echo "build failed"; exit 1; }
grep -q "LARND_JOINT_NEWTON_DUMP=$U/dumps/$DN.npz" $U/inner/inner_$DN.sh || { echo "bad inner"; exit 1; }
J=$(sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --job-name=UQ_$DN $K/run_knot.sbatch fit $U/inner/inner_$DN.sh) || exit 1
echo "$(date +%F' '%T) $DN $J :: $RUN $CK $*" >> $U/submitted.txt; echo "$DN $J"
