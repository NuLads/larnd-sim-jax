#!/bin/bash
# Debug (9 Oct): dump exact FEE inputs/outputs of the reference (stage 1, noise off) to find why stored pixel_signals
# re-digitised with the unmodified kernel give more hits than the recorded output.
#SBATCH --job-name=larndsim_feedump
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/slurm-feedump-%j.out
CMP_DIR=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP
REF_DIR=$CMP_DIR/reference_copy
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/images/develop.sif /bin/bash -c "
  cd $REF_DIR && export SKIP_CUPY_INSTALL=1 PYTHONNOUSERSITE=1 PYTHONPATH=$REF_DIR:$CMP_DIR/pydeps REF_DUMP_FEE=$CMP_DIR/fee_dump/dump && \
  python3 $CMP_DIR/run_reference_stage.py --reference-dir $REF_DIR --stage 1 --input $CMP_DIR/matched_input.h5 \
      --warmup-input $CMP_DIR/tiny_input.h5 --output $CMP_DIR/out_stage1_ref_dump.h5" 2>&1 | grep -v -i warn | tail -15
ls -la $CMP_DIR/fee_dump
