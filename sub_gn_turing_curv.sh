#!/bin/bash
#SBATCH --partition=turing
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=gn_probe
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case/gncurv-%A_%a.out
#SBATCH --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32g
#SBATCH --gpus-per-node=geforce_rtx_2080_ti:1 --time=06:00:00
#SBATCH --array=0-1
# Measure whether the HVP Hessian path fits an 11 GB card, and at what batch count.
NB=100
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then CURV=ggn; else CURV=exact; fi
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
echo "=== NB=${NB} batches curvature=${CURV} ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv -B /sdf,/fs,/lscratch larndsim-jax_main.sif /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python3 -m optimize.example_run \
  --params optimize/scripts/param_list_nd_noshift.yaml \
  --input_file_sim /sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5 \
  --input_file_tgt /sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5 \
  --fit_type gn_calib \
  --gn_resume_joint fit_result/sci_full_ANNEALLONG/history_iter10000_sci_full_b100_len400_seed0.pkl \
  --iterations 1 --max_nbatch ${NB} --gn_curvature ${CURV} --max_batch_len 400 --data_seed 1 --seed 0 \
  --lr 1e-1 --optimizer_fn Adam --lr_scheduler constant_schedule \
  --lr_kw '{\"decay_rate\":0.999,\"init_value\":0,\"warmup_steps\":500}' \
  --max_clip_norm_val 100 --electron_sampling_resolution 0.01 --number_pix_neighbors 4 \
  --signal_length 150 --mode lut --lut_file src/larndsim/detector_properties/response_44.npy \
  --loss_fn llhd --probabilistic_sim --sim_seed_strategy different --non_deterministic \
  --fit_dedx --dedx_lr 1e-2 --dedx_prior_weight 5.0 --dedx_use_split_t True \
  --dedx_student_nu_l 4.785 --dedx_student_nu_r 2.073 \
  --dedx_student_scale_l 0.1204 --dedx_student_scale_r 0.1058 \
  --dedx_soft_barrier_threshold 8.5 --dedx_soft_barrier_weight 1.0 \
  --dedx_mean_constraint_weight 100000.0 \
  --fit_chain_positions --chain_basis spline --chain_spline_knot_cm 40 \
  --chain_lr 1e-2 --mcs_prior_weight 0.5 --chain_step_len 2.0 --chain_momentum_GeV 3.0 \
  --out_label gncurv_${CURV} --test_name gn_memprobe --save_freq 10000
" 2>&1 | grep -vE "UserWarning|warnings.warn" | tail -30
echo "=== done NB=${NB} CURV=${CURV}: $(date) ==="
