#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=chainopt
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/chainopt/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=2:00:00
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/chainopt
SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
BATCH=0
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
python3 -m optimize.example_run \
    --params long_diff --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 \
    --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
    --fit_type chain --iterations 250 --max_nbatch 100 --max_batch_len 50 \
    --data_seed 1 --seed 0 --lr 0 --optimizer_fn Adam \
    --lr_scheduler warmup_exponential_decay_schedule --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
    --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd --probabilistic_sim --sim_seed_strategy same --no-noise \
    --fit_chain_positions --chain_lr 1e-4 --chain_update_freq 1 --mcs_prior_weight 0.5 \
    --chain_step_len 2.0 --chain_momentum_GeV 3.0 \
    --benchmark_chain_opt ${BATCH} \
    --out_label chainopt --test_name chainopt 2>&1 | grep -iE 'BENCH|Error|Traceback|resid' | tail -30
"
echo "=== done ==="
