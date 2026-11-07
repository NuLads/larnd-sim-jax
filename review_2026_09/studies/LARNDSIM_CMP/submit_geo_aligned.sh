#!/bin/bash
# 9 Oct: reference geometry aligned to larnd-sim-jax (tpc_offsets 0; both drift volumes towards the cathode at
# +-0.15875 cm), full batch_350cm_input.h5 (127 segments, 438 cm) on BOTH sides; stages 1-3. Outputs *_geo.*
#SBATCH --job-name=larndsim_geo
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/slurm-geo-%j.out
CMP_DIR=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP
REF_DIR=$CMP_DIR/reference_copy
IN=$CMP_DIR/batch_350cm_input.h5
for stage in 1 2 3; do
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/images/develop.sif /bin/bash -c "
    cd $REF_DIR && export SKIP_CUPY_INSTALL=1 PYTHONNOUSERSITE=1 PYTHONPATH=$REF_DIR:$CMP_DIR/pydeps && \
    python3 $CMP_DIR/run_reference_stage.py --reference-dir $REF_DIR --stage $stage --input $IN \
        --warmup-input $CMP_DIR/tiny_input.h5 --output $CMP_DIR/out_stage${stage}_ref_geo.h5" > $CMP_DIR/log_stage${stage}_ref_geo.txt 2>&1 || echo "REF $stage FAILED"
  echo "== ref $stage"; grep -E "Simulated|perf\]" $CMP_DIR/log_stage${stage}_ref_geo.txt | tail -3
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif /bin/bash -c "
    python3 $CMP_DIR/run_jax_direct.py --stage $stage --input $IN --output $CMP_DIR/out_stage${stage}_jax_geo.h5 --max-batch-len 1000" \
    > $CMP_DIR/log_stage${stage}_jax_geo.txt 2>&1 || echo "JAX $stage FAILED"
  echo "== jax $stage"; grep -E "\[data\]|\[perf\]" $CMP_DIR/log_stage${stage}_jax_geo.txt | tail -3
done
