#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=scan_bs
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/scan_bs/job-%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=2:00:00

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/scan_bs
SIF=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif

run_one () {
  local blen="$1"
  echo "================ BATCH_LEN = ${blen} cm ================"
  ( peak=0; for k in $(seq 1 300); do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m; echo "$peak" > /tmp/pk_${blen}.txt; sleep 2
    done ) & SAMP=$!
  apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF} /bin/bash -c "
    cd /sdf/group/neutrino/pgranger/larnd-sim-jax
    export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
    export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
    python3 -m optimize.example_run --params long_diff --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 --fit_type chain --iterations 40 --max_nbatch 100 --max_batch_len ${blen} --data_seed 1 --seed 0 --lr 1e-1 --optimizer_fn Adam --lr_scheduler warmup_exponential_decay_schedule --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 500}' --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic --no-noise --fit_dedx --dedx_lr 1e-2 --dedx_prior_weight 0.5 --dedx_use_split_t True --dedx_student_nu_l 4.785 --dedx_student_nu_r 2.073 --dedx_student_scale_l 0.1204 --dedx_student_scale_r 0.1058 --dedx_soft_barrier_threshold 8.5 --dedx_soft_barrier_weight 1.0 --dedx_mean_constraint_weight 100000.0 --fit_chain_positions --chain_lr 1e-4 --chain_start_iter 0 --chain_update_freq 1 --mcs_prior_weight 0.5 --chain_step_len 2.0 --chain_momentum_GeV 3.0 --pos_residual_freq 25 --out_label bs${blen} --test_name scan_bs_tmp --save_freq 10000 2>&1 | grep -iE 'RESOURCE_EXHAUSTED|out of memory|OOM|Padding from shape.*pad value -1|it/s|it\]|number of batches|nbatch|Traceback' | grep -vE 'target shape' | tail -30
  "
  local rc=$?
  kill $SAMP 2>/dev/null
  echo ">>> BATCH_LEN=${blen}: exit=${rc}  PEAK_MEM=$(cat /tmp/pk_${blen}.txt 2>/dev/null) MiB"
  rm -rf fit_result/scan_bs_tmp 2>/dev/null
}

for bl in 50 100 200 400; do run_one $bl; done
echo "===== DONE ====="
