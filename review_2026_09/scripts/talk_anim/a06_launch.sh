#!/bin/bash
# usage: a06_launch.sh <cpus per sweep> <points per parameter>; resumes: points already saved are skipped. Grid comes from a06_grid.F on the compute node.
T=/sdf/scratch/neutrino/pgranger/talk_animations; D=/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/talk_anim
for p in Ab eField tran_diff long_diff lifetime; do
  nohup $T/run.sh $1 $D/a06_sweep.py --param $p --n $2 --nfd ${3:-4} >> $T/logs/a06/sw_$p.log 2>&1 &
done
