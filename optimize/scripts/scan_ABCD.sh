#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=2:00:00
#SBATCH --array=0,1,2,3,4,5,6,7,8


# --- CONFIGURATION SELECTION ---
# Example format: A1-B1-C1-D1
# Change this variable or pass it as an argument
CONFIG="A2-B2-C2-D5"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=200
MAX_NBATCH=50
ITERATIONS=50
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
#LOSS=mse_adc 
SEED_STRATEGY=same #same #different 
SAMPLING_STEP=0.01 
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=200
DEDX_DENSITY_MODE=flow #flow #histogram

# --- LOGIC FOR COMBINATIONS ---

# Parse CONFIG (A, B, C, D)
IFS='-' read -r CONF_A CONF_B CONF_C CONF_D <<< "$CONFIG"

# A: DX / TGT File / Chop Logic
if [ "$CONF_A" == "A1" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
    CHOP_FLAG="" # No --no_chop for A1
    DX_LABEL="stopp_dxvaried"
elif [ "$CONF_A" == "A2" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
    CHOP_FLAG="--no_chop" # Add --no_chop for A2
    DX_LABEL="stopp_dx0.01"
elif [ "$CONF_A" == "A3" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.2cm.h5
    # CHOP_FLAG="--no_chop" # Add --no_chop for A2
    DX_LABEL="stopp_dx0.1_chopped"
elif [ "$CONF_A" == "A5" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm.h5
    CHOP_FLAG="--no_chop" # No --no_chop for A3
    DX_LABEL="thrumu_dx0.01"
elif [ "$CONF_A" == "A6" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3407823/job_27035649_0001/output_27035649_0001-edepsim_lbl_trklen2cm_containment2cm_costheta0.966.h5
    CHOP_FLAG="--no_chop" # No --no_chop for A3
    DX_LABEL="stopmu_dx0.01"
fi

# B: SIM Input File
if [ "$CONF_A" == "A1" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
        B_LABEL="closure"
    else
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm_dEdx.h5
        B_LABEL="reco_dE"
    fi
elif [ "$CONF_A" == "A2" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density"
        # B_LABEL="reco_dE"
    elif [ "$CONF_B" == "B3" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_true_traj_start_end_reco_seg_step_0.01cm_range_0.05cm.h5
        B_LABEL="reco_traj_st_ed_pos_dE"
    elif [ "$CONF_B" == "B4" ]; then
	    INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
        B_LABEL="reco_posdE"
    elif [ "$CONF_B" == "B5" ]; then
	    INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="reco_posdEdensity"
    fi
elif [ "$CONF_A" == "A3" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.2cm.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.2cm.h5
        B_LABEL="reco_dE"
    fi
elif [ "$CONF_A" == "A5" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density"
    fi
elif [ "$CONF_A" == "A6" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3407823/job_27035649_0001/output_27035649_0001-edepsim_lbl_trklen2cm_containment2cm_costheta0.966.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3407823/job_27035649_0001/output_27035649_0001-edepsim_lbl_trklen2cm_containment2cm_costheta0.966.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density"
    fi
fi

# C: Normalization / Params
if [ "$CONF_C" == "C1" ]; then
    NORM=divide
    PARAMS=optimize/scripts/param_list.yaml
    LR_LABEL="2e-3"
elif [ "$CONF_C" == "C2" ]; then
    NORM=sigmoid
    if [ "$CONF_B" == "B1" ]; then
    	PARAMS=optimize/scripts/param_list_main.yaml
	LR_LABEL="0.03"
    elif [ "$CONF_B" == "B2" ]; then
    	PARAMS=optimize/scripts/param_list_C2_B2.yaml
	LR_LABEL="0.1"
    elif [ "$CONF_B" == "B3" ]; then
    	PARAMS=optimize/scripts/param_list_main_reco.yaml
	LR_LABEL="1"
    elif [ "$CONF_B" == "B4" ]; then
    	PARAMS=optimize/scripts/param_list_main_reco.yaml
	LR_LABEL="1"
    fi
elif [ "$CONF_C" == "C3" ]; then
    NORM=exp_log
    PARAMS=optimize/scripts/param_list_main.yaml
    LR_LABEL="1"
fi

# D: Noise / Probabilistic Flag
# LOSS_KW is a JSON blob passed via --loss_fn_kw. Keys inside must be double-quoted
# and escaped (\") because this string is embedded in the bash -c "..." block later.
if [ "$CONF_D" == "D1" ]; then
    PROB_FLAG=""
    D_LABEL="stoc_noise"
    LOSS=mse_adc
    LOSS_KW="{}"
elif [ "$CONF_D" == "D2" ]; then
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_noise"
    LOSS=llhd
    LOSS_KW="{}"
elif [ "$CONF_D" == "D3" ]; then
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_noise_dqdt"
    LOSS=dqdt_radial
    LOSS_KW="{\"mask_empty\": false}"
elif [ "$CONF_D" == "D4" ]; then
    PROB_FLAG=""
    D_LABEL="stoc_noise_dqdt"
    LOSS=dqdt_radial
    LOSS_KW="{\"mask_empty\": false}"
elif [ "$CONF_D" == "D5" ]; then
    # Stochastic sim + dqdt_radial + MMD distributional comparison.
    # No per-slice pairing: MMD compares (Q_hit, tick_hit) 2D point clouds.
    # Bandwidths scale the two axes to O(1) so the RBF kernel (sigma=1) is
    # sensible; retune mmd_sigma_Q_ke ~ typical Q-hit scale (ke),
    # mmd_sigma_drift_tick ~ typical drift-tick spread if the loss looks
    # too flat or too spiky.
    PROB_FLAG=""
    D_LABEL="stoc_noise_dqdt_mmd"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0}"
fi

#PARAMS=optimize/scripts/param_list_${CONFIG}.yaml

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Generate Label
LABEL="${PARAM}_${CONFIG}_${DX_LABEL}_${B_LABEL}_${D_LABEL}_${DEDX_DENSITY_MODE}_tgtsimsd${SEED_STRATEGY}_nngh${N_NEIGH}_${MODE}_esamp${SAMPLING_STEP}cm_sigL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_dtsd${DATA_SEED}_adam_${LOSS}_Qonly_${NORM}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "Running Configuration: $CONFIG"
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type scan \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name scan \
    --scan_tgt_nom \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --loss_fn_kw '${LOSS_KW}' \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --dedx_density_mode ${DEDX_DENSITY_MODE} \
    ${PROB_FLAG} \
    ${CHOP_FLAG} \
    ${USE_DENSITY_FLAG} \
    --print_input
"
