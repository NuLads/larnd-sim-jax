#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --account=neutrino:dune-ml
#SBATCH --job-name=dist_scan
#SBATCH --output=logs/dist_scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=2:00:00
#SBATCH --array=0-15

# Setup paths and environment
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
INPUT_FILE=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

# Define Parameters and Distribution Features
PARAMS=("Ab" "kb" "shift_x" "shift_y")
DIST_FEATURES=("charge" "time" "x" "y")

# Map SLURM_ARRAY_TASK_ID to PARAM and FEATURE
NUM_FEATURES=${#DIST_FEATURES[@]}
PARAM_IDX=$(($SLURM_ARRAY_TASK_ID / $NUM_FEATURES))
FEAT_IDX=$(($SLURM_ARRAY_TASK_ID % $NUM_FEATURES))

PARAM=${PARAMS[$PARAM_IDX]}
FEATURE=${DIST_FEATURES[$FEAT_IDX]}

echo "Scanning Parameter: $PARAM with Distribution Loss Feature: $FEATURE"

LABEL="dist_scan_${PARAM}_${FEATURE}_$(uuidgen)"

# Run scanning
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH;
pip install .
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 100 \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE} \
    --input_file_tgt ${INPUT_FILE} \
    --out_label ${LABEL} \
    --test_name distribution_scans \
    --iterations 50 \
    --max_batch_len 300 \
    --electron_sampling_resolution 0.1 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mode 'lut' \
    --probabilistic_sim \
    --loss_fn 'distribution' \
    --dist_feature ${FEATURE} \
    --fit_type 'scan' \
    --scan_tgt_nom \
    --sim_seed_strategy 'same' \
    --detector_props src/larndsim/detector_properties/module0.yaml \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz
"
