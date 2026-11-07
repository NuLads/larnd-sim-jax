#!/bin/bash
D=/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint
until [ "$(squeue -h -u $USER | wc -l)" -lt 6 ]; do sleep 10; done
id=$(cd /sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS && sbatch --parsable --account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003 --time=01:30:00 --mem-per-cpu=48g --job-name=seedloss_S1 run_paper.sbatch "export LARND_JAX_LRU=32768 LARND_UNIQUE_BUCKET=1 LARND_POSRES_VEC=1; python3 $D/seed_loss_test.py")
echo "$(date +%T) seedloss_S1 $id" | tee -a $D/submitted.txt
until [ -z "$(squeue -h -j $id -o %i 2>/dev/null)" ]; do sleep 60; done; sacct -n -X -j $id -o JobName%12,State,Elapsed
