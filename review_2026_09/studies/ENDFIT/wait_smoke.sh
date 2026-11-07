#!/bin/bash
# ENDFIT (i) smoke (27 Oct): wait for SMGA / SMGB / SMSA, then print their sweep / SAGA log lines.
J=42101363,42101364,42101365
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 60; done
sacct -j $J -X -o JobID,JobName%8,State,Elapsed,NodeList%14
for j in 42101363 42101364 42101365; do
  f=$(ls /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs/*-$j.out); echo "== $f"
  grep -a "\[GEOM-SWEEP\] it\|CALIB-SAGA\] table\|Error\|Traceback\|NaN\|Iter 14" $f | sed 's/^.*INFO - //' | tail -30
done
