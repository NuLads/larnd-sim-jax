#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=6:00:00
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/%x-%j.out
# PRODUCTION POLISH stage (6 Nov, Pierre: "Lets adopt POLISH"; REVIEW.md "POLISH RESULT" and "POLISH adoption"): after the H
# stage, one held epoch at sci_full_<NAME>H @ 20000 writes the calibration-only E1 dump (exactly the validated UQ/run_dump.sbatch
# construction: UQ/make_jn_dump.py on the H stage's inner, LARND_JOINT_NEWTON_DUMP_CALIB=1), then KNOT/polish_estimate.py turns
# it into the final calibration. usage: sbatch --job-name=<NAME>P run_polish.sbatch <NAME>     (DRY=1: build and print only)
# Writes: fit_result_staging/sci_full_<NAME>H/polish_E1_iter20000.npz and polish_estimate.json (the production result);
# the held run's own output goes to fit_result_staging/sci_full_<NAME>P (made and linked by submit_prod.sh).
NAME=$1; [ -n "$NAME" ] || { echo "usage: run_polish.sbatch <NAME>"; exit 2; }
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; IMG=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
IN=$K/inner_sci_full_${NAME}H.sh; [ -f $IN ] || { echo "no $IN"; exit 2; }
CK=$(ls $S/sci_full_${NAME}H/history_iter20000_*.pkl) || { echo "no 20000 checkpoint for ${NAME}H"; exit 2; }
DUMP=$S/sci_full_${NAME}H/polish_E1_iter20000.npz; OUT=$S/sci_full_${NAME}H/polish_estimate.json
PIN=$K/inner_sci_full_${NAME}P.sh; [ "$DRY" = 1 ] && PIN=$(mktemp)
[ "$DRY" = 1 ] || { [ -e $DUMP ] && { echo "$DUMP exists"; exit 2; }; }
apptainer exec -B /sdf $IMG env JAX_PLATFORMS=cpu python3 $U/make_jn_dump.py $IN $CK sci_full_${NAME}P $DUMP LARND_JOINT_NEWTON_DUMP_CALIB=1 > $PIN 2>/dev/null || { echo "build failed"; exit 1; }
grep -q "LARND_JOINT_NEWTON_DUMP=$DUMP" $PIN && grep -q "LARND_JOINT_NEWTON_DUMP_CALIB=1" $PIN || { echo "bad inner"; exit 1; }
if [ "$DRY" = 1 ]; then cat $PIN; exit 0; fi
echo "[POLISH] ${NAME}H @ 20000 -> $DUMP :: $CK"
bash $K/run_knot.sbatch fit $PIN || exit 1
[ -e $DUMP ] || { echo "no dump written"; exit 1; }
apptainer exec -B /sdf $IMG env JAX_PLATFORMS=cpu python3 $K/polish_estimate.py $DUMP $CK > $OUT 2>/dev/null || { echo "polish_estimate failed"; exit 1; }
cat $OUT
