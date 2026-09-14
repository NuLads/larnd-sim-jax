#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --exclude=sdfampere024,sdfampere038,sdfampere002
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --array=0,1,2,3,4,5
##SBATCH --array=0,1,2,3,4,5,6,7,8


# --- CONFIGURATION SELECTION ---
# Example format: A1-B1-C1-D1
# Change this variable or pass it as an argument

CONFIG="A2-B5-C2-D10"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=300
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
FLOW_EXPECTATION_MODE=sample #grid #quadrature #sample

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
        # B_LABEL="reco_dE_validation"
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
elif [ "$CONF_D" == "D6" ]; then
    # Probabilistic sim + MMD on collapsed per-hit (E[Q], E[tick]) point clouds.
    # get_average_hit_values collapses each (pixel, hit_slot) to weighted mean
    # tick / ADC, with weight lambda_per_hit = P(hit exists). Target side is
    # still the stochastic hit list loaded from the input h5. Lower estimator
    # variance than D5 at fixed batch because prediction is a distribution, not
    # a single sample. Same bandwidth tuning notes as D5.
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_noise_dqdt_mmd_collapsed"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0}"
elif [ "$CONF_D" == "D7" ]; then
    # D6 + per-track ΣQ nuisance α = (ΣQ_tgt + eps) / (ΣQ_sim + eps), closed-form
    # each forward call, gradient flowing through α so the loss is ΣQ-invariant
    # per track (absorbs reco-vs-truth input-dE mismatch). Soft Gaussian prior on
    # log α — sigma=0.5 → α within ~[0.61, 1.65] softly (prior weight 1/(2σ²)=2.0,
    # was 12.5 at σ=0.2 which pinned α too close to 1 and left ΣQ mismatch for
    # Ab/kb/lifetime to absorb). Switch to per_track_scale_mode="per_batch" for
    # a single α on the whole batch (keeps track-to-track spread as signal,
    # kills only the bulk offset). The "ls05" tag encodes prior_log_sigma=0.5.
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_mmd_alphat_ls05"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0, \"per_track_scale_mode\": \"per_track\", \"per_track_scale_prior_log_sigma\": 0.5}"
elif [ "$CONF_D" == "D8" ]; then
    # D7 + option 1 (4-D MMD features: adds √Var[Q], √Var[tick] axes from prob
    # sim's per-slot 2nd moments — recovers the DL/diffusion signal that the
    # per-slot-mean collapse discards) + option 2 (global moment-matching term
    # on (mean_Q, mean_t, [var_Q, var_t]), symmetric between sim and target,
    # extra constraint on the ΣQ scale and diffusion). Toggle option 2 off by
    # setting "moment_match_weight": 0. Toggle option 1 off with
    # "mmd_extra_features": "none". Label suffix encodes moment-match variant:
    #   mmv1  = moment_match_use_var=true  (mean + var, weight 1)
    #   mmm1  = moment_match_use_var=false (mean-only,   weight 1)
    D8_MM_USE_VAR="true"    # "true" or "false" — must match LOSS_KW below
    if [ "$D8_MM_USE_VAR" = "true" ]; then
        _MM_TAG="mmv1"
    else
        _MM_TAG="mmm1"
    fi
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_mmd_at_ls05_v4d_${_MM_TAG}"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0, \"per_track_scale_mode\": \"per_track\", \"per_track_scale_prior_log_sigma\": 0.5, \"mmd_extra_features\": \"std\", \"mmd_sigma_std_Q_ke\": 2.0, \"mmd_sigma_std_tick\": 100.0, \"moment_match_weight\": 1.0, \"moment_match_use_var\": ${D8_MM_USE_VAR}}"
elif [ "$CONF_D" == "D9" ]; then
    # D8 + per-slot skew and kurt of tick as extra MMD axes (option A: higher
    # moments describe the *shape* of each slot's tick distribution beyond
    # mean+var, capturing Bragg-peak / recombination asymmetry and heavy-tail
    # diffusion structure). Point cloud coordinates:
    #   (E[Q], E[t], √Var[Q], √Var[t], skew[t], kurt[t])   -> 6-D
    # Label suffix encodes moment-match variant (same convention as D8):
    #   mmv1  = moment_match_use_var=true
    #   mmm1  = moment_match_use_var=false
    # Same target=0 caveat on the extra axes as D8's Var axes — sim's non-zero
    # skew/kurt vs target's 0 tends to prefer *Gaussian-shaped* slot
    # distributions; watch for DL / long_diff bias in the scan.
    D9_MM_USE_VAR="false"    # "true" or "false" — must match LOSS_KW below
    if [ "$D9_MM_USE_VAR" = "true" ]; then
        _MM_TAG="mmv1"
    else
        _MM_TAG="mmm1"
    fi
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_mmd_at_ls05_v6d_${_MM_TAG}"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"mmd\", \"mmd_sigma_Q_ke\": 5.0, \"mmd_sigma_drift_tick\": 500.0, \"per_track_scale_mode\": \"per_track\", \"per_track_scale_prior_log_sigma\": 0.5, \"mmd_extra_features\": \"std\", \"mmd_sigma_std_Q_ke\": 2.0, \"mmd_sigma_std_tick\": 100.0, \"mmd_include_skew_tick\": true, \"mmd_include_kurt_tick\": true, \"mmd_sigma_skew_tick\": 1.0, \"mmd_sigma_kurt_tick\": 2.0, \"moment_match_weight\": 1.0, \"moment_match_use_var\": ${D9_MM_USE_VAR}}"
elif [ "$CONF_D" == "D10" ]; then
    # Pixel-marginalized LLHD: for each target hit at (t_i, Q_i), score the
    # sim's marginal-over-pixel density there. LLHD-quality info leverage on
    # tick shape + Q, but position-insensitive because pixel identity is
    # summed out (unlike D2's full LLHD which requires pixel matching).
    #
    # Three toggles below, all reflected in the label:
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
    #   D10_METRIC         "marginal_llhd"        — joint marginal LLHD (L_joint):
    #                                                p_sim(t, Q) with tick·Q
    #                                                correlation inside the sum
    #                      "marginal_llhd_prod"   — product-of-marginals (L_prod):
    #                                                p_tick(t) * p_Q(Q) with two
    #                                                separately-marginalized 1-D
    #                                                densities. Tolerant of
    #                                                per-cell tick-Q correlation
    #                                                mismatches (better on B5).
    D10_INCLUDE_Q="true"
    D10_SIGMA_Q="5.0"
    D10_ALPHA_MODE="per_track"
    D10_ALPHA_PRIOR="0.2"
    D10_METRIC="marginal_llhd"

    # --- (Ab, τ) shape penalty (per-track log(Q)-vs-drift-tick slope term) ---
    #
    # THREE MODES driven by (WEIGHT, DIAG):
    #   WEIGHT>0                   — PRODUCTION.  Term is *added to the loss*
    #                                so gradients see it and Adam pulls
    #                                parameters toward matching slope_sim to
    #                                slope_tgt. σ sets the residual scale.
    #   WEIGHT=0, DIAG=true        — DIAGNOSTIC.  Term is computed and
    #                                aux fields populated but the loss is
    #                                unchanged, so this is *pure
    #                                measurement*, no effect on
    #                                optimization. Use to observe how the
    #                                observable behaves across a scan
    #                                (τ-leverage check, σ calibration).
    #   WEIGHT=0, DIAG=false       — INERT.  The sp code path is skipped
    #                                entirely; identical to vanilla D10.
    #                                (A `_sp0<kind>` label tag is still
    #                                emitted when kind != within_track so
    #                                the pickle doesn't collide with
    #                                vanilla — bookkeeping only.)
    #
    # Slope-observable variant (SLOPE_PENALTY_KIND):
    #   "within_track"  — legacy per-track log(Q)-vs-tick regression; Bragg-
    #                     dominated on stopping muons, τ-blind.
    #   "between_track" — one point per track (⟨t⟩, ⟨log Q⟩), regressed
    #                     across tracks. Uses inter-track drift lever arm.
    #                     τ leverage / Ab leverage ≈ 6.7× on this dataset →
    #                     good ΣQ-valley breaker. NOT α-invariant.
    #   "per_hit"       — pool all hits into one regression; higher absolute
    #                     τ SNR (~12×) but also picks up Ab response (~2.6×),
    #                     so τ/Ab specificity is only ~4.6×. Weaker
    #                     valley-breaker than between_track. NOT α-invariant.
    SLOPE_PENALTY_KIND="per_hit"

    # Per-kind σ, calibrated from the diag dry-run's batch-to-batch σ of
    # `mean_slope_sim` at truth (see aux). Both values live here so
    # switching KIND above picks the matching σ automatically without
    # re-editing anything else.
    SLOPE_PENALTY_SIGMA_BT="3e-5"    # between_track
    SLOPE_PENALTY_SIGMA_PH="4e-5"    # per_hit
    SLOPE_PENALTY_SIGMA_WT="1e-3"    # within_track (legacy calibration)
    case "$SLOPE_PENALTY_KIND" in
        within_track)  SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_WT";;
        between_track) SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_BT";;
        per_hit)       SLOPE_PENALTY_SIGMA="$SLOPE_PENALTY_SIGMA_PH";;
    esac

    SLOPE_PENALTY_WEIGHT="1.0"    # ≈ 10% × mllhd_only(11.4) at truth
    SLOPE_PENALTY_DIAG="false"
    # Drop hits with tick < this value from the slope regression. Excludes
    # the pixel-electronics signal-integrator ramp-up region at very short
    # drift (~first 100 ticks), which has non-exponential Q(t) and biases
    # any linear slope fit. 0 = no cut (legacy).
    SLOPE_PENALTY_TICK_MIN="200"

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

    # Label prefix per metric — keeps L_joint and L_prod pickles distinct.
    if [ "$D10_METRIC" = "marginal_llhd_prod" ]; then
        _METRIC_TAG="mllhdP"
    else
        _METRIC_TAG="mllhd"
    fi

    # Slope-penalty label suffix — encoded so the report generator (and you)
    # can tell modes apart at a glance and so pickles never collide across
    # variants:
    #
    #   weight > 0                          → _sp<kind><w>sg<σ>
    #     Production; penalty is in the loss.
    #
    #   weight = 0, diag = true             → _spD<kind>
    #     Dry-run; observable computed for aux only.
    #
    #   weight = 0, diag = false, kind != within_track  →  _sp0<kind>
    #     Inert but user explicitly picked a non-default observable — tag it
    #     so this pickle doesn't overwrite the vanilla D10 baseline. (The
    #     "0" prefix makes clear the penalty is off.)
    #
    #   weight = 0, diag = false, kind == within_track  →  (no tag)
    #     Backward-compat: same label as the historical vanilla D10 scan.
    #
    # `<kind>` is a two-letter abbrev: wt=within_track, bt=between_track,
    # ph=per_hit.
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
    D_LABEL="prob_${_METRIC_TAG}_${_Q_TAG}_sQ${_SQ_TAG}_${_AT_TAG}${_SP_TAG}"
    LOSS=dqdt_radial
    LOSS_KW="{\"distance_metric\": \"${D10_METRIC}\", \"marginal_llhd_include_Q\": ${D10_INCLUDE_Q}, \"marginal_llhd_sigma_Q_ke\": ${D10_SIGMA_Q}, \"marginal_llhd_eps\": 1e-10, \"per_track_scale_mode\": \"${D10_ALPHA_MODE}\", \"per_track_scale_prior_log_sigma\": ${_PRIOR_JSON}, \"slope_penalty_weight\": ${SLOPE_PENALTY_WEIGHT}, \"slope_penalty_sigma\": ${SLOPE_PENALTY_SIGMA}, \"slope_penalty_diag\": ${SLOPE_PENALTY_DIAG}, \"slope_penalty_kind\": \"${SLOPE_PENALTY_KIND}\", \"slope_penalty_tick_min\": ${SLOPE_PENALTY_TICK_MIN}}"
elif [ "$CONF_D" == "D11" ]; then
    # D5 (stochastic sim + 2-D MMD on (Q_hit, tick_hit)) + D7-style per-track
    # ΣQ nuisance. No 4-D/6-D features, no probabilistic collapse — just the
    # cheapest MMD path plus α on top. Useful when the probabilistic path in
    # D6/D7 hits the (Npix·Nhits)² kernel-matrix OOM on shift scans:
    # stochastic sim keeps the point cloud at the *actual hit count* (a few
    # thousand) instead of the padded (Npix·Nhits) ~ 30k.
    #
    # Toggles (mirroring D10 conventions):
    #   D11_ALPHA_MODE    "off" | "per_track" | "per_batch"
    #   D11_ALPHA_PRIOR   "0.5" | "null" — log-sigma of the log-α² prior
    D11_ALPHA_MODE="per_track"
    D11_ALPHA_PRIOR="0.5"

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

#PARAMS=optimize/scripts/param_list_${CONFIG}.yaml

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Optimizer for the scan machinery — mostly cosmetic (scans don't take
# gradient steps in the physics sense), but we keep the choice in the
# label for consistency with fits. See start_fit_ABCD.sh for full docs.
OPTIMIZER_FN="Adam"
case "$OPTIMIZER_FN" in
    Adam)         _OPT_TAG="adam";;
    SGD)          _OPT_TAG="sgd";;
    PreAdam)      _OPT_TAG="preadam";;
    NatGrad)      _OPT_TAG="natgrad";;
    DampedNewton) _OPT_TAG="dnewton";;
    *) echo "ERROR: unknown OPTIMIZER_FN=$OPTIMIZER_FN"; exit 3;;
esac

# Generate Label
LABEL="${CONFIG}_${DX_LABEL}_${B_LABEL}_${D_LABEL}_${DEDX_DENSITY_MODE}_${FLOW_EXPECTATION_MODE}_nngh${N_NEIGH}_${MODE}_esamp${SAMPLING_STEP}cm_sigL${SIGNAL_LENGTH}_grclp${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_nbt${MAX_NBATCH}_dtsd${DATA_SEED}_${_OPT_TAG}_${LOSS}_${NORM}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "==========================================================="
echo "Running Configuration: $CONFIG"
echo "   hostname          = $(hostname)"
echo "   nvidia-smi first line = $(nvidia-smi -L 2>&1 | head -1)"
echo "Label: $LABEL"
echo "==========================================================="

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
# Preflight: verify JAX can initialize a CUDA device on this host. Some
# Ampere nodes (sdfampere024/038/002 have been seen) list the GPU via
# nvidia-smi but XLA's StreamExecutor init fails with CUDA_ERROR_UNKNOWN
# or CUDA_ERROR_CONTAINED. Exit 75 (EX_TEMPFAIL) → SLURM --requeue drops
# the task onto a different node instead of burning the 4h wall clock.
if ! python3 -c 'import jax; assert any(d.platform==\"gpu\" for d in jax.devices()), \"no GPU\"' 2>/dev/null; then
    echo \"CUDA preflight FAILED on \$(hostname). Requesting requeue.\"
    exit 75
fi
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}; \
# Serialize pip install across concurrent array tasks. Multiple tasks
# racing on user site-packages rename each other's dist-info dir mid-read
# (\"~arndsim-*/METADATA missing\"). If larndsim is already importable —
# most likely case for a repeat job — skip the install entirely.
if ! python3 -c 'import larndsim' >/dev/null 2>&1; then \
    (flock -x 200; python3 -c 'import larndsim' >/dev/null 2>&1 || pip3 install .) 200> /tmp/pip_larndsim.lock; \
fi; \
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
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --dedx_density_mode ${DEDX_DENSITY_MODE} \
    ${PROB_FLAG} \
    ${CHOP_FLAG} \
    ${USE_DENSITY_FLAG} \
    --print_input \
    --flow_expectation_mode ${FLOW_EXPECTATION_MODE} \
"
