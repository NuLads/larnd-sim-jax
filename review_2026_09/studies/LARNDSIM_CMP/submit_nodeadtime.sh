#!/bin/bash
# Test (9 Oct): does removing the reference's ADC dead time (ADC_BUSY_DELAY, RESET_CYCLES) reproduce the
# larnd-sim-jax hit counts? Reference stages 1 and 3 on matched_input.h5 with REF_NO_DEADTIME=1. ~5 min.
#SBATCH --job-name=larndsim_nodead
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/slurm-nodead-%j.out
CMP_DIR=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP
REF_DIR=$CMP_DIR/reference_copy
for stage in 1 3; do
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/images/develop.sif /bin/bash -c "
    cd $REF_DIR && export SKIP_CUPY_INSTALL=1 PYTHONNOUSERSITE=1 PYTHONPATH=$REF_DIR:$CMP_DIR/pydeps REF_NO_DEADTIME=1 && \
    python3 $CMP_DIR/run_reference_stage.py --reference-dir $REF_DIR --stage $stage --input $CMP_DIR/matched_input.h5 \
        --warmup-input $CMP_DIR/tiny_input.h5 --output $CMP_DIR/out_stage${stage}_ref_nodead.h5" > $CMP_DIR/log_stage${stage}_ref_nodead.txt 2>&1 \
    || echo "stage $stage FAILED"
  grep -E "constants applied|perf\]|Simulated" $CMP_DIR/log_stage${stage}_ref_nodead.txt
done
