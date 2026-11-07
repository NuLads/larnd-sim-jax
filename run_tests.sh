#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=00:15:00
#SBATCH --job-name=test_prior
#SBATCH --output=logs/fit_noise/test_prior-%j.out

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

# Run Fit A: with prior and constraint (regularized)
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 1 \
    --params Ab \
    --lr 1e-1 \
    --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 \
    --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed 1 \
    --out_label test_with_prior \
    --test_name test_prior_comparison \
    --seed 0 \
    --optimizer_fn Adam \
    --random_nevent \
    --iterations 100 \
    --max_batch_len 200 \
    --track_z_bound 28 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mc_diff \
    --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd \
    --sim_seed_strategy random \
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 20}' \
    --max_clip_norm_val 100 \
    --fit_dedx \
    --dedx_prior_weight 0.5 \
    --dedx_lr 1e-2 \
    --dedx_start_iter 0 \
    --dedx_freeze_iter 200 \
    --probabilistic_sim \
    --dedx_mean_constraint_weight 100000.0 \
    --dedx_mean_constraint_target 1.887
"

# Run Fit B: without prior and constraint (unregularized)
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 1 \
    --params Ab \
    --lr 1e-1 \
    --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 \
    --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed 1 \
    --out_label test_no_prior \
    --test_name test_prior_comparison \
    --seed 0 \
    --optimizer_fn Adam \
    --random_nevent \
    --iterations 100 \
    --max_batch_len 200 \
    --track_z_bound 28 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mc_diff \
    --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd \
    --sim_seed_strategy random \
    --lr_scheduler warmup_exponential_decay_schedule \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 20}' \
    --max_clip_norm_val 100 \
    --fit_dedx \
    --dedx_prior_weight 0.0 \
    --dedx_lr 1e-2 \
    --dedx_start_iter 0 \
    --dedx_freeze_iter 200 \
    --probabilistic_sim \
    --dedx_mean_constraint_weight 0.0 \
    --dedx_mean_constraint_target 1.887
"
