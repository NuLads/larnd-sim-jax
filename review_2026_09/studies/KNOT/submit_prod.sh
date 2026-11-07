#!/bin/bash
# PRODUCTION launcher (24 Oct, Pierre's OK): the 1e4-iteration KN10 fit, then -- chained with afterok -- the 4000-iteration
# freeze-release continuation (run_knot.sbatch prodcont = CONT/make_prod_cont.py = the validated PRF recipe; REVIEW.md
# "FB block = CONFIRMED"), then -- 28 Oct, Pierre: "Let's make this part of standard production now" -- the 6000-iteration
# fast continuation of its 14000 checkpoint (run_knot.sbatch prodfast = CONT/make_prod_fast.py = the (h) recipe; REVIEW.md
# "(j) RESULT"). The production end point is the fast continuation's 20000 checkpoint, sci_full_<NAME>H (until 28 Oct:
# sci_full_<NAME>FR @ 14000).
# usage: submit_prod.sh <NAME> <inner_fit.sh>    (the inner script must use --test_name sci_full_<NAME>)
# Outputs: /sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_<NAME>{,FR,H}, linked from the live fit_result/.
# Submits 4 chained jobs since 6 Nov (each waits for the previous one); refuses if 3 or more of this user's jobs are already queued.
# Chaining (submit_prod_chains.sh): DEP=<slurm dependency> makes the fit wait for it; no queue-size check then (a
# chained triple adds no running job). DEP=none: no dependency and no queue check (chain heads of
# submit_prod_chains.sh: the caller bounds the running jobs by the number of chains).
# The last line printed ends with "fit <J1> -> cont <NAME>FR <J2> -> fast <NAME>H <J3> -> polish <NAME>P <J4> :: <inner>".
# 31 Oct (Pierre: "Ok, let's adopt this"; REVIEW.md "ENDFIT (u) RESULT"): the fit runs with the coarse-to-fine geometry C75,
# LARND_CHAIN_COARSE_ITER=7500 (only the lowest sine modes free until iteration 7500), passed to the fit job only; the FR and H
# stages inherit it from the checkpoint provenance (inactive there, iteration >= 10000), exactly as in the validated (r)-(u) runs.
# An inner script that exports its own value wins; LARND_CHAIN_COARSE_ITER= (empty) or 0 in the caller's environment turns it off.
# 6 Nov (Pierre: "Lets adopt POLISH"; "Let's adopt the cache and write targets directly to data"; REVIEW.md "6 Nov adoptions"):
# (1) a 4th chained job, run_polish.sbatch = the POLISH stage (afterok on H): one held epoch at <NAME>H @ 20000, calibration-only
#     E1 dump, then the final calibration in fit_result_staging/sci_full_<NAME>H/polish_estimate.json (the production result);
# (2) one persistent JAX compilation cache per PIPELINE (the 29 Oct tested path, ENDFIT/cache_adoption);
# (3) target dirs under LARND_TARGET_ROOT=/sdf/data/neutrino/pgranger/larnd-sim-jax instead of the repo root on /sdf/group.
# Set any of these variables before calling this script to override (LARND_JAX_PERSISTENT_CACHE=job: the old per-job cache;
# LARND_TARGET_ROOT= (empty): the old behaviour).
NAME=$1; INNER=$2
[ -n "$NAME" ] && [ -f "$INNER" ] || { echo "usage: submit_prod.sh <NAME> <inner_fit.sh>"; exit 2; }
grep -q -- "--test_name sci_full_$NAME " "$INNER" || { echo "$INNER does not set --test_name sci_full_$NAME"; exit 2; }
if [ -z "$DEP" ]; then n=$(squeue -h -u $USER | wc -l); [ $n -lt 3 ] || { echo "$n jobs queued: no room for 4 more (limit 6)"; exit 75; }; fi
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
for r in sci_full_$NAME sci_full_${NAME}FR sci_full_${NAME}H sci_full_${NAME}P; do
  if [ -e $FR/$r ] && [ ! -L $FR/$r ]; then echo "$FR/$r exists and is not a link: refusing"; exit 2; fi
  mkdir -p $S/$r; [ -L $FR/$r ] || ln -s $S/$r $FR/$r
done
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
DA=${DEP:+--dependency=$DEP}; [ "$DEP" = none ] && DA=
CI=${LARND_CHAIN_COARSE_ITER-7500}
# (2) the four jobs get both variables from this env (sbatch --export=ALL); the dir is under the per-job sweeper's root, so it
# is removed after 2 idle days. (3) LARND_* variable: stored in the fit checkpoint, so FR, H and POLISH inherit it.
export LARND_JAX_PERSISTENT_CACHE=${LARND_JAX_PERSISTENT_CACHE-shared}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR-/sdf/scratch/neutrino/$USER/jax_job_cache/pipe_$NAME}
export LARND_TARGET_ROOT=${LARND_TARGET_ROOT-/sdf/data/neutrino/pgranger/larnd-sim-jax}
J1=$(sbatch --parsable $A --job-name=$NAME $DA --export=ALL,LARND_CHAIN_COARSE_ITER=$CI $K/run_knot.sbatch fit "$INNER") || exit 1
J2=$(sbatch --parsable $A --job-name=${NAME}FR --dependency=afterok:$J1 $K/run_knot.sbatch prodcont sci_full_$NAME sci_full_${NAME}FR) || exit 1
J3=$(sbatch --parsable $A --job-name=${NAME}H --dependency=afterok:$J2 $K/run_knot.sbatch prodfast sci_full_${NAME}FR sci_full_${NAME}H) || exit 1
J4=$(sbatch --parsable $A --job-name=${NAME}P --dependency=afterok:$J3 $K/run_polish.sbatch $NAME) || exit 1
echo "$(date +%F' '%T) prod $NAME [coarse ${CI:-off}; cache ${LARND_JAX_PERSISTENT_CACHE}; targets ${LARND_TARGET_ROOT:-cwd}] fit $J1 -> cont ${NAME}FR $J2 -> fast ${NAME}H $J3 -> polish ${NAME}P $J4 :: $INNER" | tee -a $K/submitted.txt
