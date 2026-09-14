#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=18:00:00
#SBATCH --array=1,2,3,4,5

# --- CONFIGURATION SELECTION ---
# Example format: A1-B1-C1-D1
# Change this variable or pass it as an argument
CONFIG="A2-B5-C2-D10"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=200
MAX_NBATCH=500
ITERATIONS=15000
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
#LOSS=mse_adc 
SEED_STRATEGY=different 
SAMPLING_STEP=0.01 
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=200
DEDX_DENSITY_MODE=flow #flow #histogram
FLOW_EXPECTATION_MODE=sample

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
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="reco_pos_dE_density"
    elif [ "$CONF_B" == "B5" ]; then
	    INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="reco_pos_dE_density_no_diffusion_par_lr2"
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
    elif [ "$CONF_B" == "B3" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm_straight_line_reco_idx0_10000.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density_reco_pos"
    elif [ "$CONF_B" == "B4" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm_straight_line_reco_idx0_10000.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density_reco_pos_Ab_Efield_lifetime"
    elif [ "$CONF_B" == "B5" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_3576201/job_23729838_0000/output_23729838_0000-edepsim_lbl_range_0.05cm_straight_line_reco_idx0_10000.h5
        USE_DENSITY_FLAG="--use_dedx_density"
        B_LABEL="dE_density_reco_pos_kb_Efield_lifetime"
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
    elif [ "$CONF_B" == "B5" ]; then
    	PARAMS=optimize/scripts/param_list_main_reco.yaml
	LR_LABEL="1"
    fi
elif [ "$CONF_C" == "C3" ]; then
    NORM=exp_log
    PARAMS=optimize/scripts/param_list_main.yaml
    LR_LABEL="1"
fi

# D: Noise / Probabilistic Flag
# LOSS_KW is a JSON blob passed via --loss_fn_kw. Keys inside must be double-
# quoted and escaped (\") because this string is embedded in the bash -c "..."
# block later, same pattern as --lr_kw.
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
    # (Q_hit, tick_hit) 2D point-cloud MMD; retune sigmas if the loss looks
    # too flat or too spiky.
    PROB_FLAG=""
    D_LABEL="stoc_noise_dqdt_mmd"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0}"
elif [ "$CONF_D" == "D10" ]; then
    # Pixel-marginalized LLHD: for each target hit at (t_i, Q_i), score the
    # sim's marginal-over-pixel density there. LLHD-quality info leverage on
    # tick shape + Q, but position-insensitive because pixel identity is
    # summed out (unlike D2's full LLHD which requires pixel matching).
    #
    # Four toggles below, all reflected in the label:
    #   D10_INCLUDE_Q      "true" | "false"       — include Q axis (Gaussian
    #                                                kernel) or drop to tick-only
    #   D10_SIGMA_Q        e.g. "0.5" | "3.0"     — Q-kernel width in ke.
    #                                                Wider tolerates larger
    #                                                target-sim Q mismatch (e.g.
    #                                                B5 reco-position case) at
    #                                                the cost of Q-shape signal.
    #   D10_ALPHA_MODE     "off" | "per_track" | "per_batch"
    #                                              — closed-form per-track ΣQ
    #                                                nuisance, rescales sim
    #                                                Q_grid before the kernel
    #                                                so target-sim ΣQ align.
    #   D10_ALPHA_PRIOR    e.g. "0.5" | "null"    — log-α² Gaussian prior
    #                                                width; ignored if
    #                                                D10_ALPHA_MODE=off.
    #
    # Presets — set D10_PRESET to select one and skip editing the 4 knobs.
    #   "default"    — σ_Q=3.0, α_prior=0.5. Original balanced setting.
    #   "sweep_best" — σ_Q=5.0, α_prior=0.2. From the A2-B5 hyperparameter
    #                  sweep summary: smallest bias on Ab/kb/lifetime with
    #                  frac_underflow ≈ 0 and tight α (no ΣQ absorption).
    #                  Recommended fit-time working point on B5.
    #   "custom"     — script honors whatever you edit below.
    D10_PRESET="sweep_best"

    if [ "$D10_PRESET" = "sweep_best" ]; then
        D10_INCLUDE_Q="true"
        D10_SIGMA_Q="5.0"
        D10_ALPHA_MODE="per_track"
        D10_ALPHA_PRIOR="0.2"
    elif [ "$D10_PRESET" = "default" ]; then
        D10_INCLUDE_Q="true"
        D10_SIGMA_Q="3.0"
        D10_ALPHA_MODE="per_track"
        D10_ALPHA_PRIOR="0.5"
    else
        # "custom" — edit below directly.
        D10_INCLUDE_Q="true"
        D10_SIGMA_Q="3.0"
        D10_ALPHA_MODE="per_track"
        D10_ALPHA_PRIOR="0.5"
    fi

    # --- derive label suffixes from the toggles ---
    if [ "$D10_INCLUDE_Q" = "true" ]; then
        _Q_TAG="tQ"
    else
        _Q_TAG="t"
    fi
    _SQ_TAG="${D10_SIGMA_Q/./p}"   # "3.0" -> "3p0", "0.5" -> "0p5"

    if [ "$D10_ALPHA_MODE" = "off" ]; then
        _AT_TAG="atOff"
        _PRIOR_JSON="null"
    elif [ "$D10_ALPHA_PRIOR" = "null" ]; then
        _AT_TAG="at_${D10_ALPHA_MODE}_noPr"
        _PRIOR_JSON="null"
    else
        _PRIOR_TAG="${D10_ALPHA_PRIOR/./p}"
        _AT_TAG="at_${D10_ALPHA_MODE}_ls${_PRIOR_TAG}"
        _PRIOR_JSON="${D10_ALPHA_PRIOR}"
    fi

    # --- Shape penalty (per-track log(Q)-vs-drift-tick slope term) ---
    # Breaks the (Ab, τ) and (kb, τ) ΣQ-preserving valleys. See
    # scan_ABCD.sh header comment for the diag/prod/inert mode explanation.
    # Per-kind σ calibrated from the diag dry-run's batch-to-batch σ of
    # `mean_slope_sim` at truth.
    SLOPE_PENALTY_KIND="between_track"    # within_track | between_track | per_hit
    SLOPE_PENALTY_SIGMA_BT="3e-5"
    SLOPE_PENALTY_SIGMA_PH="4e-5"
    SLOPE_PENALTY_SIGMA_WT="1e-3"
    case "$SLOPE_PENALTY_KIND" in
        within_track)  SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_WT";;
        between_track) SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_BT";;
        per_hit)       SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_PH";;
    esac
    SLOPE_PENALTY_WEIGHT="0.0"
    SLOPE_PENALTY_DIAG="false"
    SLOPE_PENALTY_TICK_MIN="200"    # drop pixel-electronics ramp-up region

    # See scan_ABCD.sh for the tag encoding.
    case "$SLOPE_PENALTY_KIND" in
        within_track)  _SP_K_TAG="wt";;
        between_track) _SP_K_TAG="bt";;
        per_hit)       _SP_K_TAG="ph";;
        *) echo "ERROR: unknown SLOPE_PENALTY_KIND=$SLOPE_PENALTY_KIND"; exit 3;;
    esac
    _SP_TAG=""
    if awk -v w="$SLOPE_PENALTY_WEIGHT" 'BEGIN{exit !(w+0>0)}'; then
        _SP_W_TAG="${SLOPE_PENALTY_WEIGHT/./p}"
        _SP_S_TAG="${SLOPE_PENALTY_SIGMA/./p}"
        _SP_TAG="_sp${_SP_K_TAG}${_SP_W_TAG}sg${_SP_S_TAG}"
    elif [ "$SLOPE_PENALTY_DIAG" = "true" ]; then
        _SP_TAG="_spD${_SP_K_TAG}"
    elif [ "$SLOPE_PENALTY_KIND" != "within_track" ]; then
        _SP_TAG="_sp0${_SP_K_TAG}"
    fi

    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_mllhd_${_Q_TAG}_sQ${_SQ_TAG}_${_AT_TAG}${_SP_TAG}"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"marginal_llhd\", \"marginal_llhd_include_Q\": ${D10_INCLUDE_Q}, \"marginal_llhd_sigma_Q_ke\": ${D10_SIGMA_Q}, \"marginal_llhd_eps\": 1e-10, \"per_track_scale_mode\": \"${D10_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}, \"slope_penalty_weight\": ${SLOPE_PENALTY_WEIGHT}, \"slope_penalty_sigma\": ${SLOPE_PENALTY_SIGMA}, \"slope_penalty_diag\": ${SLOPE_PENALTY_DIAG}, \"slope_penalty_kind\": \"${SLOPE_PENALTY_KIND}\", \"slope_penalty_tick_min\": ${SLOPE_PENALTY_TICK_MIN}}"
elif [ "$CONF_D" == "D11" ]; then
    # D5 (stochastic sim + 2-D MMD on (Q_hit, tick_hit)) + D7-style per-track
    # ΣQ nuisance. No probabilistic collapse — the cheapest MMD path plus α.
    # Toggle α off (D11_ALPHA_MODE="off") to recover pure D5. Prior width
    # log-α² Gaussian: D11_ALPHA_PRIOR (log-sigma). "null" disables the prior
    # and lets α float free.
    D11_ALPHA_MODE="per_track"    # "off" | "per_track" | "per_batch"
    D11_ALPHA_PRIOR="0.5"         # e.g. "0.5" or "null"

    if [ "$D11_ALPHA_MODE" = "off" ]; then
        _AT_TAG="atOff"
        _PRIOR_JSON="null"
    elif [ "$D11_ALPHA_PRIOR" = "null" ]; then
        _AT_TAG="at_${D11_ALPHA_MODE}_noPr"
        _PRIOR_JSON="null"
    else
        _PRIOR_TAG="${D11_ALPHA_PRIOR/./p}"
        _AT_TAG="at_${D11_ALPHA_MODE}_ls${_PRIOR_TAG}"
        _PRIOR_JSON="${D11_ALPHA_PRIOR}"
    fi

    PROB_FLAG=""
    D_LABEL="stoc_mmd_${_AT_TAG}"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0, \"per_track_scale_mode\": \"${D11_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}}"
fi

PARAMS=optimize/scripts/param_list_${CONFIG}.yaml

# --- Optimizer choice ---
#   "Adam"          — legacy per-parameter Adam via optax.multi_transform.
#   "SGD"           — legacy SGD (per-parameter LR schedules).
#   "PreAdam"       — full-matrix Adam. Maintains a K×K second-moment
#                     matrix and applies V^(-1/2) as a preconditioner
#                     (via GPU Newton-Schulz iteration — no cusolver dep).
#                     Absorbs the (Ab, τ) valley curvature so along-valley
#                     drift is suppressed. Uses max of per-param LR
#                     schedules as a single scalar.
#   "NatGrad"       — natural-gradient / one-step-Newton preconditioning.
#                     Same streaming state as PreAdam, but applies V^(-1)
#                     (not V^(-1/2)) via jnp.linalg.solve. Stronger
#                     Newton-like preconditioning. Same memory as Adam.
#                     Faster convergence near the optimum; potentially
#                     more aggressive far from it.
#   "DampedNewton"  — full-Hessian Levenberg-Marquardt. Solves
#                     (H + λ·I)·step = g each iteration. Fastest convergence
#                     for 6-10 physics params with strong parameter
#                     correlations. Requires with_hess=True → K× extra
#                     memory for the Hessian pass.
OPTIMIZER_FN="NatGrad"

# LR-label / tag in filename for optimizer choice — kept short so pickle
# labels stay under the OS filename limit.
case "$OPTIMIZER_FN" in
    Adam)         _OPT_TAG="adam";;
    SGD)          _OPT_TAG="sgd";;
    PreAdam)      _OPT_TAG="preadam";;      # full-matrix Adam, V^{-1/2}·m̂ via Newton-Schulz
    NatGrad)      _OPT_TAG="natgrad";;      # Fisher-preconditioned, V^{-1}·m̂ via jnp.linalg.solve
    DampedNewton) _OPT_TAG="dnewton";;
    *) echo "ERROR: unknown OPTIMIZER_FN=$OPTIMIZER_FN"; exit 3;;
esac

# Generate Label
LABEL="${CONFIG}_4par_${B_LABEL}_${DX_LABEL}_${D_LABEL}_tgtsimsd_${SEED_STRATEGY}_nneigh${N_NEIGH}_${MODE}_sigL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_nbt${MAX_NBATCH}_dtsd${DATA_SEED}_${_OPT_TAG}_${LOSS}_${NORM}_tgtsd${TARGET_SEED}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "Running Configuration: $CONFIG"
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
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
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --optimizer_fn ${OPTIMIZER_FN} \
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
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --dedx_density_mode ${DEDX_DENSITY_MODE} \
    ${PROB_FLAG} \
    ${CHOP_FLAG} \
    ${USE_DENSITY_FLAG} \
    --print_input
"
