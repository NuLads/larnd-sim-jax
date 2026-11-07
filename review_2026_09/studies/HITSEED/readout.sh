#!/bin/bash
# Readout for HITSEED / GTRU (samples 1/2/4) against production DSREP and each other, per the stated rules.
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
S=/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ/compare_arms.py
H=${1:-HITSEED}; G=${2:-GTRU}; B=${3:-DSREP}
echo "=== $G vs $B";  python $S $G $B 1,2,4 2>/dev/null
echo; echo "=== $H vs $B"; python $S $H $B 1,2,4 2>/dev/null | sed -n '/paired/,$p'
echo; echo "=== $H vs $G"; python $S $H $G 1,2,4 2>/dev/null | sed -n '/paired/,$p'
echo; echo "=== drift-shape error (median / mean, um)"
python /sdf/scratch/neutrino/pgranger/review_2026_09/DRIFTHITS/shape_err.py ${B}1 ${H}1 ${G}1 ${B}2 ${H}2 ${G}2 ${B}4 ${H}4 ${G}4 2>/dev/null
