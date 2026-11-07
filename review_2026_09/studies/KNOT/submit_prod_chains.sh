#!/bin/bash
# Production ensemble as SLURM dependency chains (24 Oct): each chain runs its samples one after the other, each sample =
# submit_prod.sh triple (fit, the freeze-release continuation <NAME>FR, the fast continuation <NAME>H; 28 Oct). At most
# one job per chain runs, so the number of running jobs = the number of chains. A sample's fit starts when the previous
# sample's POLISH stage ends (any state; 6 Nov: 4th job <NAME>P), or as soon as the previous fit, FR or H fails (the later
# jobs never start).
# usage: submit_prod_chains.sh <head_dependency|none> "<NAME list of chain 1>" "<NAME list of chain 2>" ...
#   (none: the chain heads start at once; running jobs = number of chains)
#   e.g. submit_prod_chains.sh afterok:41815295 "KN10N2 KN10N5" "KN10N3 KN10N6"   (inner = KNOT/inner_sci_full_<NAME>.sh)
K=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT
HEAD=$1; shift
[ -n "$HEAD" ] && [ $# -ge 1 ] || { echo "usage: submit_prod_chains.sh <head_dependency> \"<names>\" ..."; exit 2; }
for chain in "$@"; do
  dep=$HEAD
  for NAME in $chain; do
    out=$(DEP=$dep bash $K/submit_prod.sh $NAME $K/inner_sci_full_$NAME.sh) || { echo "$out"; echo "chain stopped at $NAME"; exit 1; }
    echo "$out"
    j1=$(echo "$out" | tail -1 | sed -n 's/.* fit \([0-9]*\) -> cont .*/\1/p'); j2=$(echo "$out" | tail -1 | sed -n 's/.* cont [^ ]* \([0-9]*\) -> fast .*/\1/p')
    j3=$(echo "$out" | tail -1 | sed -n 's/.* fast [^ ]* \([0-9]*\) -> polish .*/\1/p')
    j4=$(echo "$out" | tail -1 | sed -n 's/.* polish [^ ]* \([0-9]*\) ::.*/\1/p')
    [ -n "$j1" ] && [ -n "$j2" ] && [ -n "$j3" ] && [ -n "$j4" ] || { echo "could not parse job ids for $NAME"; exit 1; }
    dep="afterany:$j4?afternotok:$j1?afternotok:$j2?afternotok:$j3"
  done
done
