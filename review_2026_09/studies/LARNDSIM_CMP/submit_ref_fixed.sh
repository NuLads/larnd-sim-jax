#!/bin/bash
# Reference rerun with the overrides re-applied after importlib.reload(fee) (9 Oct): stages 1-3 on matched_input,
# stage 1 without dead time, and stage 3 on a 10x input (marginal per-cm cost, compile amortised). FEE dump on stage 1.
#SBATCH --job-name=larndsim_reffix
#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --exclude=sdfampere038,sdfampere003
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/slurm-reffix-%j.out
CMP_DIR=/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP
REF_DIR=$CMP_DIR/reference_copy
run() {  # stage input output extra_env
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/images/develop.sif /bin/bash -c "
    cd $REF_DIR && export SKIP_CUPY_INSTALL=1 PYTHONNOUSERSITE=1 PYTHONPATH=$REF_DIR:$CMP_DIR/pydeps $4 && \
    python3 $CMP_DIR/run_reference_stage.py --reference-dir $REF_DIR --stage $1 --input $2 \
        --warmup-input $CMP_DIR/tiny_input.h5 --output $3" > ${3%.h5}.log 2>&1 || echo "FAILED $3"
  echo "== $3"; grep -E "re-applied|timed run|perf\]|Simulated" ${3%.h5}.log | cut -c1-260
}
run 1 $CMP_DIR/matched_input.h5 $CMP_DIR/out_stage1_ref.h5 "REF_DUMP_FEE=$CMP_DIR/fee_dump/dump"
run 2 $CMP_DIR/matched_input.h5 $CMP_DIR/out_stage2_ref.h5 ""
run 3 $CMP_DIR/matched_input.h5 $CMP_DIR/out_stage3_ref.h5 ""
run 1 $CMP_DIR/matched_input.h5 $CMP_DIR/out_stage1_ref_nodead.h5 "REF_NO_DEADTIME=1"
run 3 $CMP_DIR/matched_input_x10.h5 $CMP_DIR/out_stage3_ref_x10.h5 ""
