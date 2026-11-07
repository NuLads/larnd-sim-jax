#!/bin/bash
#SBATCH --job-name=pos_demo
#SBATCH --account=neutrino:cider-nu
#SBATCH --partition=ampere
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_pos_demo_%j.log
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_pos_demo_%j.log

SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

echo "=== Job started: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv \
    --bind /sdf,/lscratch \
    "$SIF" \
    bash -c '
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH

python3 -m optimize.example_run \
    --params Ab \
    --input_file_sim data/pos_demo_sim.h5 \
    --input_file_tgt data/pos_demo_tgt.h5 \
    --fit_type chain \
    --iterations 300 \
    --max_nbatch 10 \
    --max_batch_len 50 \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --track_z_bound 28 \
    --data_seed 1 \
    --seed 0 \
    --lr 1e-10 \
    --optimizer_fn Adam \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd \
    --probabilistic_sim \
    --sim_seed_strategy different \
    --non_deterministic \
    --no-noise \
    --fit_chain_positions \
    --chain_lr 3e-4 \
    --chain_start_iter 0 \
    --mcs_prior_weight 0.1 \
    --chain_step_len 2.0 \
    --chain_momentum_GeV 3.0 \
    --out_label pos_demo_300iter \
    --test_name pos_demo \
    --save_freq 50
'

echo "=== Job finished: $(date) ==="
