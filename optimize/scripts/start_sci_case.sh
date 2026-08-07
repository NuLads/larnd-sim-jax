#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=sci_case
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=40:00:00
#SBATCH --array=0-4%5

# Science-case experiment: isolate the effect of POSITION handling on Ab recovery.
# All three modes share the v12 config (param_list_nd_noshift, --fit_dedx, 50cm, 100 batches);
# they differ ONLY in INPUT_SIM and whether chain positions are fitted.
#   SCIMODE=floor    -> INPUT_SIM=linear_guess, positions FROZEN at imperfect guess (no --fit_chain_positions)
#   SCIMODE=full     -> INPUT_SIM=linear_guess, positions FITTED via Adam chain (--fit_chain_positions)  [= v12]
#   SCIMODE=ceiling  -> INPUT_SIM=true,          positions = TRUE, frozen (no --fit_chain_positions)
SCIMODE=${SCIMODE:-full}

PARAMS=${SCIPARAMS:-optimize/scripts/param_list_nd_noshift.yaml}

ITERATIONS=${SCIITER:-5000}
MAX_NBATCH=${SCINBATCH:-100}
# SCITRUTHPOINT=<a completed history_*.pkl> switches the run into truth-point EVALUATION mode
# (no fitting; see _eval_truth_point). One epoch visits every batch exactly once, which is all
# the evaluation needs, so iterations is pinned to the batch count rather than left at SCIITER.
SCITRUTHPOINT=${SCITRUTHPOINT:-}
if [ -n "${SCITRUTHPOINT}" ]; then
  if [ ! -f "${SCITRUTHPOINT}" ]; then
    echo "SCITRUTHPOINT=${SCITRUTHPOINT} does not exist"; exit 1
  fi
  ITERATIONS=${MAX_NBATCH}
  echo "=== TRUTH-POINT MODE: evaluating against ${SCITRUTHPOINT}, ITERATIONS pinned to ${ITERATIONS} ==="
fi
# dEdx mean-constraint weight and parametrisation. Defaults reproduce the previous behaviour
# EXACTLY (1e5 / per-segment). See the GN report: 1e5 is measured to be far too stiff.
SCIMEANW=${SCIMEANW:-100000.0}
SCIDXPARAM=${SCIDXPARAM:-perseg}
MAX_BATCH_LEN=${SCILEN:-50}
DATA_SEED=1
SEED=${SLURM_ARRAY_TASK_ID:-0}
LOSS=llhd
N_NEIGH=4
MODE=lut
LR_SCHEDULER=warmup_exponential_decay_schedule

# dEdx settings (identical across modes)
DEDX_PRIOR_WEIGHT=${SCIDEDXPRIOR:-0.5}
DEDX_DRIFT_PROFILE_WEIGHT=${SCIDPW:-0.0}
DEDX_LR=${SCIDEDXLR:-1e-2}
DEDX_START_ITER=0
# Default = ITERATIONS+200, i.e. never fires. SCIDEDXFREEZE makes it settable: freezing dEdx at
# iters/2 is what broke the lifetime<->dEdx co-adaptation in the 5-D study.
DEDX_FREEZE_ITER=${SCIDEDXFREEZE:-$((ITERATIONS+200))}
# dEdx LR annealing: final LR as a FRACTION of the initial one, reached at the end of the run.
# Empty = constant LR = previous behaviour (the dEdx block historically never annealed at all,
# unlike calibration which decays per epoch and geometry which decays per iteration).
SCIDEDXFF=${SCIDEDXFF:-}
# dEdx mean-constraint TARGET (the value <dEdx> is pinned to). Distinct from SCIMEANW, which is
# how hard it is pinned. Deliberately mis-setting this is the dedx-mean bias study.
SCIDMEANTGT=${SCIDMEANTGT:-1.887}
# Split student-t prior SHAPE. Defaults reproduce the previous hardcoded values exactly. These
# parameters describe the true per-segment dEdx distribution AT A GIVEN SEGMENTATION and are not
# transferable between grids: the defaults below are the MLE optimum on the TARGET file's
# segmentation (dx 2.32 cm) and are badly mismatched on the guess file's (dx 0.99 cm), where the
# true spread is 1.6x narrower. See fig 48(j).
SCIDXNUL=${SCIDXNUL:-4.785}
SCIDXNUR=${SCIDXNUR:-2.073}
SCIDXSL=${SCIDXSL:-0.1204}
SCIDXSR=${SCIDXSR:-0.1058}
SCIDXLOC=${SCIDXLOC:-}

# Chain position settings (only used when SCIMODE=full)
CHAIN_LR=${SCICLR:-1e-4}
# Iteration at which geometry fitting is released. >0 runs an initial phase with geometry
# held at the (imperfect) linear guess so calibration/dEdx settle first. Note chain_decay_start
# defaults to this, so the geometry LR schedule also starts here rather than at iteration 0.
CHAIN_START_ITER=${SCICHSTART:-0}
CHAIN_UPDATE_FREQ=1
MCS_PRIOR_WEIGHT=0.5
CHAIN_STEP_LEN=${SCISTEP:-2.0}
CHAIN_MOMENTUM_GEV=3.0

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
LINEAR_GUESS=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
JAX_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache

# Mode-dependent settings
case ${SCIMODE} in
  floor)
    INPUT_FILE_SIM=${LINEAR_GUESS}
    CHAIN_FLAG=""
    ;;
  full)
    INPUT_FILE_SIM=${SCISIMFILE:-${LINEAR_GUESS}}
    CHAIN_FLAG="--fit_chain_positions"
    ;;
  ceiling)
    INPUT_FILE_SIM=${INPUT_FILE_TGT}
    CHAIN_FLAG=""
    ;;
  *)
    echo "Unknown SCIMODE=${SCIMODE} (expected floor|full|ceiling)"; exit 1;;
esac

TEST_NAME=sci_${SCIMODE}${SCITAG:-}
# The label MUST encode every knob that changes the fit, or concurrent configurations silently
# overwrite each other's checkpoints (this has now bitten three times in one campaign, and the
# survivors look exactly like a clean set of seeds). SCIMEANW/SCIDXPARAM are appended whenever
# they differ from the defaults, so previously-submitted configurations keep their old paths.
_KNOBTAG=""
[ "${SCIMEANW}" != "100000.0" ] && _KNOBTAG="${_KNOBTAG}_mw${SCIMEANW}"
[ "${SCIDXPARAM}" != "perseg" ] && _KNOBTAG="${_KNOBTAG}_${SCIDXPARAM}"
[ "${LARND_DEDX_MIN_DX:-0.0}" != "0.0" ] && _KNOBTAG="${_KNOBTAG}_mdx${LARND_DEDX_MIN_DX}"
[ "${SCIDMEANTGT}" != "1.887" ] && _KNOBTAG="${_KNOBTAG}_mt${SCIDMEANTGT}"
[ -n "${SCIDEDXFF}" ]           && _KNOBTAG="${_KNOBTAG}_dff${SCIDEDXFF}"
[ "${SCIDEDXLR:-1e-2}" != "1e-2" ] && _KNOBTAG="${_KNOBTAG}_dlr${SCIDEDXLR}"
[ -n "${SCIDEDXFREEZE}" ]       && _KNOBTAG="${_KNOBTAG}_dfz${SCIDEDXFREEZE}"
[ "${SCIDXNUL}" != "4.785" ]    && _KNOBTAG="${_KNOBTAG}_refit"
LABEL=sci_${SCIMODE}_b${MAX_NBATCH}_len${MAX_BATCH_LEN}${_KNOBTAG}_seed${SEED}

mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/sci_case

echo "=== Job started: $(date) ==="
echo "=== SCIMODE=${SCIMODE} SEED=${SEED} INPUT_SIM=${INPUT_FILE_SIM} CHAIN_FLAG='${CHAIN_FLAG}' ==="
echo "=== MEANW=${SCIMEANW} MEANTGT=${SCIDMEANTGT} DXPARAM=${SCIDXPARAM} ITERATIONS=${ITERATIONS} CHAIN_START_ITER=${CHAIN_START_ITER} CHAIN_LR=${CHAIN_LR} CHDECAY=${SCICHDECAY:-0.999} LEN=${MAX_BATCH_LEN} LRDECAY=${SCILRDECAY:-0.999} DEDXPRIOR=${SCIDEDXPRIOR:-} DEDXLR=${DEDX_LR} DEDXFF=${SCIDEDXFF:-none} DEDXFREEZE=${DEDX_FREEZE_ITER} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}
# Geometry basis frame + drift-projected displacement penalty (explicit, so the setting is
# visible in the job log rather than relying on apptainer env passthrough).
export LARND_DEDX_PARAM=${SCIDXPARAM}
export LARND_DEDX_MIN_DX=${LARND_DEDX_MIN_DX:-0.0}
export LARND_DEDX_LR_FINAL_FRAC=${SCIDEDXFF}
export LARND_TRUTHPOINT=${SCITRUTHPOINT}
export LARND_CHAIN_FRAME=${LARND_CHAIN_FRAME:-drift}
export LARND_CHAIN_DRIFT_W=${LARND_CHAIN_DRIFT_W:-0}
echo \"[CHAIN-ENV] frame=\$LARND_CHAIN_FRAME drift_w=\$LARND_CHAIN_DRIFT_W\"
echo \"[DEDX-ENV] lr=${DEDX_LR} final_frac='\$LARND_DEDX_LR_FINAL_FRAC' freeze=${DEDX_FREEZE_ITER} mean_target=${SCIDMEANTGT} mean_w=${SCIMEANW}\"

python3 -m optimize.example_run \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain \
    --iterations ${ITERATIONS} \
    --max_nbatch ${MAX_NBATCH} \
    --max_batch_len ${MAX_BATCH_LEN} \
    --data_seed ${DATA_SEED} \
    --seed ${SEED} \
    --lr 1e-1 \
    --optimizer_fn Adam \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : '${SCILRDECAY:-0.999}', \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val 100 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn ${LOSS} ${SCISETP:+--set_params $SCISETP} ${SCIINIT:+--set_init_params $SCIINIT} ${SCITGT:+--set_target_vals $SCITGT} \
    --probabilistic_sim ${SCIPROBTGT:+--probabilistic_sampling_target} \
    --sim_seed_strategy different ${SCISHUFFLE:+--shuffle_bt} \
    --non_deterministic \
    ${NOISEFLAG:-} \
    ${DEDXFLAG:---fit_dedx} \
    --dedx_lr ${DEDX_LR} \
    --dedx_start_iter ${DEDX_START_ITER} \
    --dedx_freeze_iter ${DEDX_FREEZE_ITER} \
    --dedx_prior_weight ${DEDX_PRIOR_WEIGHT} \
    --dedx_use_split_t True \
    --dedx_student_nu_l ${SCIDXNUL} \
    --dedx_student_nu_r ${SCIDXNUR} \
    --dedx_student_scale_l ${SCIDXSL} \
    --dedx_student_scale_r ${SCIDXSR} \
    ${SCIDXLOC:+--dedx_student_loc ${SCIDXLOC}} \
    --dedx_soft_barrier_threshold 8.5 \
    --dedx_soft_barrier_weight 1.0 \
    --dedx_mean_constraint_weight ${SCIMEANW} \
    --dedx_mean_constraint_target ${SCIDMEANTGT} \
    --dedx_drift_profile_weight ${DEDX_DRIFT_PROFILE_WEIGHT} \
    ${CHAIN_FLAG} ${SCIBASIS:+--chain_basis $SCIBASIS --chain_spline_knot_cm 40} ${SCIGEOMOPT:+--geom_optimizer $SCIGEOMOPT} \
    --chain_lr ${CHAIN_LR} \
    --chain_start_iter ${CHAIN_START_ITER} \
    --chain_update_freq ${CHAIN_UPDATE_FREQ} \
    --chain_decay_rate ${SCICHDECAY:-0.999} \
    --mcs_prior_weight ${MCS_PRIOR_WEIGHT} \
    --chain_step_len ${CHAIN_STEP_LEN} \
    --chain_momentum_GeV ${CHAIN_MOMENTUM_GEV} \
    --out_label ${LABEL} \
    --test_name ${TEST_NAME} \
    --save_freq 200
"

echo "=== Job finished: $(date) ==="
