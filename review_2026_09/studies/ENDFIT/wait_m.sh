#!/bin/bash
# ENDFIT (m): wait for every job of one set (sched = SK, FL, FLH; lm = LM) to leave the queue, then sacct + read_m.py <set>.
E=/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT; SET=$1
case $SET in sched) PAT='(SK|FL|FLH)$';; lm) PAT='LM$';; *) echo "usage: wait_m.sh sched|lm"; exit 2;; esac
J=$(awk -v p="$PAT" '$1 ~ p {print $2}' $E/m_jobs.txt | paste -sd,)
while [ -n "$(squeue -h -j $J -o %i 2>/dev/null)" ]; do sleep 120; done
sacct -j $J -X -o JobID,JobName%12,State,Elapsed,NodeList
cd $E && apptainer exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH=/sdf/group/neutrino/pgranger/larnd-sim-jax/src:/sdf/group/neutrino/pgranger/larnd-sim-jax python3 read_m.py $SET 2>&1 | grep -v "cuda\|CUDA\|plugin\|^Traceback (most recent call last):$"
