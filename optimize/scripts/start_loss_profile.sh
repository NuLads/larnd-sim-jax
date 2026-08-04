#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=lossprof
#SBATCH --output=/sdf/group/neutrino/pgranger/larnd-sim-jax/logs/loss_profile/job-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=20:00:00
#SBATCH --array=0-2
#
# STUDY 1 — 1-D loss profile through the TRUE parameter point.
#
# THE QUESTION: lifetime moves monotonically further from truth the more the optimizer works
# (ANNEAL SumLR 134.5 -> -2.01%, ANNEALLONG 136.0 -> -2.70%, SLOWANNEAL 239.4 -> -5.02%).
# Either (a) the loss minimum IS at truth and annealing has not arrived, or (b) the minimum is
# genuinely displaced and better optimization converges to a wrong answer. NO FIT CAN SEPARATE
# THESE. This scan does, because it never runs an optimizer: it evaluates the loss on a grid.
#
# HOW: LikelihoodProfiler (--fit_type scan) walks each of the 5 calibration parameters across
# its ranges.py [down, up] interval, holding the OTHER four at the values given by
# --set_init_params. We set both the target (--scan_tgt_nom, which pins the target to
# ranges[p]['nom']) and the init to those same nominal values, so every scan is a 1-D slice
# through the TRUE point. If the per-parameter minimum sits at nominal, the objective is
# unbiased for that parameter; if it is displaced, we have measured a real bias and its size.
#
# This is a 1-D SLICE, not a profile likelihood: the other parameters are held fixed rather than
# re-minimised. That is the right first measurement (it isolates the objective from optimizer and
# from parameter correlations), but a displaced minimum here must be re-checked with the other
# parameters profiled before being called a bias.
#
# GEOMETRY sets the condition:
#   GEOM=true   -> sim geometry = target geometry (the S2 condition: isolates the objective)
#   GEOM=guess  -> sim geometry = straight-line guess (the S3 condition: adds geometry error)
set -e
SEED=${SLURM_ARRAY_TASK_ID:-0}
STEPS=${PROFSTEPS:-21}
NB=${PROFNBATCH:-30}
LEN=${PROFLEN:-400}
GEOM=${GEOM:-true}

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
LINEAR_GUESS=/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5
# SIMFILE lets the quality-ladder files (optimize/scripts/make_quality_ladder.py) be scanned:
# controlled degradations of the TRUE file along ONE axis (position RMS, or dE/dx spread),
# so the scan measures how the loss-minimum location depends on that quality alone.
if   [ -n "${SIMFILE:-}" ];  then INPUT_FILE_SIM=$SIMFILE
elif [ "$GEOM" = "guess" ]; then INPUT_FILE_SIM=$LINEAR_GUESS
else                              INPUT_FILE_SIM=$INPUT_FILE_TGT; fi

# Scan centre = the nominal values that --scan_tgt_nom pins the target to (optimize/ranges.py).
NOM="Ab 0.8 eField 0.5 tran_diff 8.8e-6 long_diff 4.0e-6 lifetime 2200"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
mkdir -p /sdf/group/neutrino/pgranger/larnd-sim-jax/logs/loss_profile

echo "=== Job started: $(date) ==="
echo "=== STUDY1 loss profile: GEOM=${GEOM} SEED=${SEED} STEPS=${STEPS} NB=${NB} LEN=${LEN} ==="
echo "=== sim=${INPUT_FILE_SIM} ==="
echo "=== scan centre (also the target, via --scan_tgt_nom): ${NOM} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/sdf/group/neutrino/pgranger/.jax_cache
# SCANWINDOW>0 narrows every scan to nom*(1 +/- w) instead of the full ranges.py interval.
# Unset/0 = previous behaviour exactly. Cost is steps x batches and does NOT depend on the span,
# so a narrow window buys resolution for free.
export LARND_SCAN_WINDOW=${SCANWINDOW:-0}
echo \"[SCAN-ENV] window=\$LARND_SCAN_WINDOW steps=${STEPS} nbatch=${NB}\"
python3 -m optimize.example_run \
    --params optimize/scripts/param_list_nd_noshift.yaml \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type scan \
    --scan_tgt_nom \
    --iterations ${STEPS} \
    --max_nbatch ${NB} \
    --max_batch_len ${LEN} \
    --data_seed 1 \
    --seed ${SEED} \
    --set_init_params ${NOM} \
    --lr 1e-1 --optimizer_fn Adam --lr_scheduler constant_schedule --lr_kw '{}' \
    --max_clip_norm_val 100 \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mode lut \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn llhd \
    --probabilistic_sim \
    --sim_seed_strategy different \
    --non_deterministic \
    --out_label prof_${GEOM}${PROFTAG:+_${PROFTAG}}_seed${SEED} \
    --test_name loss_profile \
    --save_freq 100000
"
echo "=== Job finished: $(date) ==="
