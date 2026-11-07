#!/bin/bash
#SBATCH --job-name=chain_geom_test
#SBATCH --account=neutrino
#SBATCH --partition=turing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_chain_geom_%j.log
#SBATCH --error=/sdf/group/neutrino/pgranger/larnd-sim-jax/slurm_chain_geom_%j.log

SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv \
    --bind /sdf,/lscratch \
    "$SIF" \
    bash -c '
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH

python3 -m optimize.example_run \
    --params Ab \
    --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 \
    --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
    --fit_type chain \
    --iterations 100 \
    --max_nbatch 6 \
    --max_batch_len 50 \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --track_z_bound 28 \
    --data_seed 1 \
    --seed 0 \
    --lr 1e-1 \
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
    --fit_dedx \
    --dedx_prior_weight 0.5 \
    --dedx_lr 1e-2 \
    --dedx_start_iter 0 \
    --dedx_use_split_t True \
    --dedx_student_nu_l 4.785 \
    --dedx_student_nu_r 2.073 \
    --dedx_student_scale_l 0.1204 \
    --dedx_student_scale_r 0.1058 \
    --dedx_soft_barrier_threshold 8.5 \
    --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight 0.0 \
    --fit_chain_positions \
    --chain_lr 3e-5 \
    --chain_start_iter 0 \
    --mcs_prior_weight 1.0 \
    --chain_step_len 2.0 \
    --chain_momentum_GeV 3.0 \
    --out_label chain_geom_test_100iter_b50 \
    --test_name chain_geom_test \
    --save_freq 9999
'

echo "=== Job finished: $(date) ==="
