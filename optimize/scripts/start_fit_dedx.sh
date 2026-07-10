#!/bin/bash

#SBATCH --partition=ampere

#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --array=1-5

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list_wokb.yaml
BATCH_SIZE=200
ITERATIONS=8000
MAX_CLIP_NORM_VAL=100
DATA_SEED=1
LOSS=llhd
SEED_STRATEGY=random
SAMPLING_STEP=0.01 # cm
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
MAX_NBATCH=80
SIGNAL_LENGTH=150

# per-segment dEdx settings: fit_dedx is True and start_iter is 0 to allow optimization
# of dEdx starting from the prior centre (MPV = 1.864 MeV/cm)
DEDX_PRIOR_WEIGHT=0.5
DEDX_LR=1e-2  # typically lower than global LR
DEDX_START_ITER=0  # start fitting dEdx from the beginning
DEDX_FREEZE_ITER=8200  # do not freeze dEdx before the end of the run

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

### true stopping muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_dEdx_mod0.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_new.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0_dEdx_gaus1smear.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0_dEdx_gaus10smear.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx+20.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0_dEdx+20.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_CSDA_dEdx+20.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_CSDA_dEdx+20.h5

##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_new.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_force_agree_1MeVcm_new.h5
#
## full truth
##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
## 'reco' dE/dx
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_dEdx_new.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_force_agree_1MeVcm_new.h5

## true through going muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
LABEL=true_throughmuons_6par_dedxfit_priw${DEDX_PRIOR_WEIGHT}_dlr${DEDX_LR}_dsi${DEDX_START_ITER}_noise_tgtsim_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}

#LABEL=stopping_mu_range_dEdx_6par_n_neigh${N_NEIGH}_mode_${MODE}_noise_e_sampling_${SAMPLING_STEP}cm_seed_strategy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#LABEL=stopping_mu_range_0.5-5cm_dEdx_force_agree_1MeVcm_6par_no_noise_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#LABEL=stopping_mu_range_0.5-5cm_dEdx_force_agree_1MeVcm_6par_no_noise_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_proton_range_0.1cm_dEdx_6par_no_noise_guess_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopmu_range_0.2-cm_dEdx+20_edep_mod0_6par_noise_tgt_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopmu_range_0.2-cm_dEdx+20_edep_CSDA_6par_noise_tgtsim_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopmu_range_0.5-5cm_dEdx_CSDA_6par_noise_tgt_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#DECLARATIONS
#LABEL=true_stopp_closure_rg0.1-cm_6par_lr8e-3_noise_tgtsim_seed_${SEED_STRATEGY}_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_signalL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopp_closure_rg0.1-cm_6par_lr8e-3_lossnoq_noise_tgtsim_seed_${SEED_STRATEGY}_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_signalL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopp_closure_rg0.1-cm_6par_lr8e-3_oldloss_noise_tgtsim_seed_${SEED_STRATEGY}_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_signalL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
ONAME=fit_noise_dedx_${SLURM_ARRAY_JOB_ID}
nvidia-smi


# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH; \
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name $ONAME \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --lr_scheduler ${LR_SCHEDULER} \
    --fit_dedx \
    --dedx_prior_weight ${DEDX_PRIOR_WEIGHT} \
    --dedx_lr ${DEDX_LR} \
    --dedx_start_iter ${DEDX_START_ITER} \
    --dedx_freeze_iter ${DEDX_FREEZE_ITER} \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --probabilistic_sim \
    --dedx_mean_constraint_weight 100000.0 \
    --dedx_mean_constraint_target 1.887
    #--random_ntrack \
    #--no-noise-guess \
    #--no-noise \
    #--lr_scheduler exponential_decay \
    #--lr_kw '{\"decay_rate\" : 0.99}' \
    #--live_selection
    #--chamfer_match_z \
    #--print_input
    # --loss_fn SDTW \
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized'
    # --profile_gradient 
    # --loss_fn space_match
"

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
