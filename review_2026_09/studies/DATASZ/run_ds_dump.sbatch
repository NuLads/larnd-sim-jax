#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=6:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/%x-%j.out
# DATASZ copy of UQ/run_dump.sbatch (3 Nov), building with DATASZ/make_ds_dump.py (--iterations N_b + 1). UQ (3 Nov): one diagnostic dump of an end point (as submit_dump.sh) but with its command built AT JOB START, so the job
# can be chained before the checkpoint exists. usage: sbatch --job-name=UQ_<DNAME> run_dump.sbatch <RUN> <DNAME> <IT> [LARND_X=v ...]
#   uses KNOT/inner_sci_full_<RUN>.sh and fit_result_staging/sci_full_<RUN>/history_iter<IT>_*.pkl; writes UQ/dumps/<DNAME>.npz;
#   the held run writes to fit_result_staging/sci_full_UQ_<DNAME> (made and linked by the submitter).
RUN=$1; DN=$2; IT=$3; shift 3
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging
IN=$K/inner_sci_full_$RUN.sh; [ -f $IN ] || { echo "no $IN"; exit 2; }
CK=$(ls $S/sci_full_$RUN/history_iter${IT}_*.pkl) || { echo "no $IT checkpoint for $RUN"; exit 2; }
[ -e $U/dumps/$DN.npz ] && { echo "$U/dumps/$DN.npz exists"; exit 2; }
apptainer exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu python3 /sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ/make_ds_dump.py $IN $CK sci_full_UQ_$DN $U/dumps/$DN.npz "$@" > $U/inner/inner_$DN.sh 2>/dev/null || { echo "build failed"; exit 1; }
grep -q "LARND_JOINT_NEWTON_DUMP=$U/dumps/$DN.npz" $U/inner/inner_$DN.sh || { echo "bad inner"; exit 1; }
echo "[UQ] $RUN @ $IT -> $DN :: $CK $*"
bash $K/run_knot.sbatch fit $U/inner/inner_$DN.sh
