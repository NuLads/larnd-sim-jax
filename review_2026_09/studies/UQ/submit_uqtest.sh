#!/bin/bash
# UQ TEST (3 Nov; REVIEW.md "UQ TEST of the recommendations"): all jobs as 6 dependency chains (each task waits for the
# previous one of its chain, afterany), so <= 6 run at once. Per chain c = 0..5: three T1 dumps of EXP<c+1>C75 (10000,
# 14000, 20000), the A pipelines EXP<17+c>C75 and EXP<23+c>C75, the R pipeline EXP<17+c>C75R, then chain 0/1: R pipeline
# EXP<23+c>C75R and one D dump; chains 2-5: two H2 replicas and two or three D dumps. H2 also waits (afterok) for its A
# pipeline's FR job, a D dump for its A pipeline's H job. Job ids -> UQ/uqtest_jobs.txt.
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT; U=/sdf/scratch/neutrino/pgranger/review_2026_09/UQ
S=/sdf/scratch/neutrino/pgranger/fit_result_staging; FR=/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result
A="--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003"
LOG=$U/uqtest_jobs.txt
declare -A JFR JH
link() { if [ -e $FR/$1 ] && [ ! -L $FR/$1 ]; then echo "$FR/$1 exists and is not a link"; exit 2; fi; mkdir -p $S/$1; [ -L $FR/$1 ] || ln -s $S/$1 $FR/$1; }
dep() { [ -n "$LAST" ] && echo "--dependency=afterany:$LAST$1" || { [ -n "$1" ] && echo "--dependency=${1#,}"; }; }
dump() {   # RUN DNAME IT [extra dependency ',afterok:J']
  [ -e $U/dumps/$2.npz ] && { echo "$2 exists"; exit 2; }
  link sci_full_UQ_$2
  J=$(sbatch --parsable $A --job-name=UQ_$2 $(dep "$4") $U/run_dump.sbatch $1 $2 $3 LARND_JOINT_NEWTON_DUMP_CALIB=1) || exit 1
  echo "dump $2 $J :: $1 @ $3 $(dep "$4")" >> $LOG; LAST=$J; }
prod() {   # NAME
  out=$(DEP=${LAST:+afterany:$LAST} ; [ -z "$DEP" ] && DEP=none; DEP=$DEP $K/submit_prod.sh $1 $K/inner_sci_full_$1.sh | tail -1) || exit 1
  set -- $1 $(echo "$out" | sed 's/.* fit \([0-9]*\) -> cont [^ ]* \([0-9]*\) -> fast [^ ]* \([0-9]*\) ::.*/\1 \2 \3/')
  [ -n "$4" ] || { echo "parse failed: $out"; exit 1; }
  JFR[$1]=$3; JH[$1]=$4; echo "prod $1 $2 $3 $4" >> $LOG; LAST=$4; }
h2() {     # NAME (its A pipeline must be submitted already)
  link sci_full_${1}H2
  J=$(sbatch --parsable $A --job-name=${1}H2 $(dep ",afterok:${JFR[$1]}") $K/run_knot.sbatch prodfast sci_full_${1}FR sci_full_${1}H2) || exit 1
  echo "h2 ${1}H2 $J :: afterok ${JFR[$1]}" >> $LOG; LAST=$J; }
echo "# submitted $(date +%F' '%T)" >> $LOG
# pass 1: the A pipelines of every chain must exist before any H2 / D dump refers to them, so build chains in two passes
declare -A CLAST
for c in 0 1 2 3 4 5; do
  LAST=; s=$((c + 1))
  dump EXP${s}C75 T1_EXP${s}_10k 10000; dump EXP${s}C75FR T1_EXP${s}_14k 14000; dump EXP${s}C75H T1_EXP${s}_20k 20000
  prod EXP$((17 + c))C75; prod EXP$((23 + c))C75; prod EXP$((17 + c))C75R
  CLAST[$c]=$LAST
done
for c in 0 1; do LAST=${CLAST[$c]}; prod EXP$((23 + c))C75R; dump EXP$((17 + c))C75H D_EXP$((17 + c)) 20000 ",afterok:${JH[EXP$((17 + c))C75]}"; done
c=2; LAST=${CLAST[$c]}; h2 EXP17C75; h2 EXP18C75; for s in 19 20 21; do dump EXP${s}C75H D_EXP$s 20000 ",afterok:${JH[EXP${s}C75]}"; done
c=3; LAST=${CLAST[$c]}; h2 EXP19C75; h2 EXP20C75; for s in 22 23 24; do dump EXP${s}C75H D_EXP$s 20000 ",afterok:${JH[EXP${s}C75]}"; done
c=4; LAST=${CLAST[$c]}; h2 EXP21C75; h2 EXP22C75; for s in 25 26; do dump EXP${s}C75H D_EXP$s 20000 ",afterok:${JH[EXP${s}C75]}"; done
c=5; LAST=${CLAST[$c]}; h2 EXP23C75; h2 EXP24C75; for s in 27 28; do dump EXP${s}C75H D_EXP$s 20000 ",afterok:${JH[EXP${s}C75]}"; done
echo "done: $(grep -c . $LOG) lines in $LOG"
