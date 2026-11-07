#!/bin/bash
# DATASZ (3 Nov): submit one epoch-matched pipeline (fit -> FR -> H, afterok) and link its staging dirs.
# usage: [DEP=<slurm dep>] submit_ds.sh <NAME> <data_seed> <data_sz> <N_b> <L>     prints "<NAME> <J1> <J2> <J3>"
NAME=$1; shift
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
DS=/sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ
for r in sci_full_$NAME sci_full_${NAME}FR sci_full_${NAME}H; do
  if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r exists and is not a link: refusing" >&2; exit 2; fi
  [ -e $S/$r ] && [ -n "$(ls -A $S/$r)" ] && { echo "$S/$r not empty: refusing" >&2; exit 2; }
  mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
done
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
DA=${DEP:+--dependency=$DEP}
J1=$(sbatch --parsable $A --job-name=$NAME $DA $DS/run_ds.sbatch fit $NAME "$@") || exit 1
J2=$(sbatch --parsable $A --job-name=${NAME}FR --dependency=afterok:$J1 $DS/run_ds.sbatch cont $NAME ${NAME}FR) || exit 1
J3=$(sbatch --parsable $A --job-name=${NAME}H --dependency=afterok:$J2 $DS/run_ds.sbatch fast ${NAME}FR ${NAME}H) || exit 1
echo "$NAME $J1 $J2 $J3"
