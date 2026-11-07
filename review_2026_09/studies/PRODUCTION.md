# Production pipeline (state of 6 Nov 2026)

C75 coarse-to-fine fit + freeze-release (FR) + fast continuation (H) to 20000 iterations + POLISH.
Adopted step by step: 24 Oct (fit + FR), 28 Oct (H), 31 Oct (C75), 6 Nov (POLISH, per-pipeline JAX
cache, targets on /sdf/data). Acceptance run: EXP41C75, PASS (REVIEW.md, "Acceptance pipeline
EXP41C75 RESULT").

The scripts run from `/sdf/scratch/neutrino/pgranger/review_2026_09/` (absolute paths inside them).
The copies here are the record of that code; see `README.md` for how to refresh them.

## Launch

```
bash KNOT/submit_prod.sh <NAME> <inner_fit.sh>     # inner must set --test_name sci_full_<NAME>
DEP=... bash KNOT/submit_prod_chains.sh ...         # an ensemble as dependency chains
```

`<inner_fit.sh>` is a copy of `KNOT/inner_sci_full_EXP3C75.sh` with only `--data_seed` and `--test_name`
changed (there is no generator for it). The production samples use realistic ends:
`LARND_ENDPOINT_SMEAR_MM=1.0`, `LARND_ENDPOINT_SMEAR_KEY=entry`.

## The four chained jobs (each `afterok` on the previous)

| Stage | Job | Built by | End point |
|---|---|---|---|
| fit, 10000 its, C75 (`LARND_CHAIN_COARSE_ITER=7500`) | `KNOT/run_knot.sbatch fit` | the inner script | `sci_full_<NAME>` |
| FR, freeze-release continuation to 14000 | `KNOT/run_knot.sbatch prodcont` | `CONT/make_prod_cont.py` (`CONT/param_list_cont.yaml`, calib LR 1e-3) | `sci_full_<NAME>FR` |
| H, fast continuation to 20000 | `KNOT/run_knot.sbatch prodfast` | `CONT/make_prod_fast.py` (`ENDFIT/param_list_cont_1e-2.yaml`, calib LR 1e-2) | `sci_full_<NAME>H` @ 20000 |
| POLISH, one held epoch + calibration-only E1 dump | `KNOT/run_polish.sbatch` | `UQ/make_jn_dump.py` (`LARND_JOINT_NEWTON=20000`, `_DUMP`, `_DUMP_CALIB=1`) | `sci_full_<NAME>H/polish_estimate.json` |

The production result is `polish_estimate.json`, written by `KNOT/polish_estimate.py` (imports
`UQ/uq_lib.py`): m = e + 100 jac(-H^-1 sum_b g_c) at the H end point. The quoted uncertainty is the
ensemble spread (REVIEW.md, UQ thread); the per-fit derivative widths are diagnostics.

## Environment set by `submit_prod.sh`

- `LARND_JAX_PERSISTENT_CACHE=shared`, `JAX_COMPILATION_CACHE_DIR=/sdf/scratch/neutrino/$USER/jax_job_cache/pipe_<NAME>`
  (one cache per pipeline, ~45 MB, swept after 2 idle days).
- `LARND_TARGET_ROOT=/sdf/data/neutrino/pgranger/larnd-sim-jax` (targets on /sdf/data, not the repo).
- SLURM: `--account=neutrino:dune-ml --exclude=sdfampere038,sdfampere003`; refuses if 3 or more jobs are queued.
- Outputs: `/sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_<NAME>{,FR,H,P}`, linked from `fit_result/`.
- Ledger: every launch is appended to `KNOT/submitted.txt`.

## Repo side

`optimize/example_run.py`, `optimize/fit_params.py`, `optimize/ranges.py`,
`optimize/scripts/param_list_nd_noshift.yaml`, `src/larndsim/detector_properties/response_44.npy`, run in
`larndsim-jax_main.sif` (not in git). Input data: `/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5`,
`/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5`.

## Absolute paths to change if the pipeline ever moves

- `submit_prod.sh`: `K=` (KNOT dir), `S=` (fit_result_staging), `FR=` (live fit_result), the cache and target roots.
- `run_knot.sbatch`, `run_polish.sbatch`: `#SBATCH --output=.../KNOT/logs/` (cannot use variables); they write the
  generated `inner_sci_full_<NAME>{FR,H,P}.sh` into `KNOT/`, and later stages read them back.
- `polish_estimate.py`: `sys.path` entries for `UQ/` and the repo.
- `make_prod_cont.py`, `make_prod_fast.py`: the `--params` YAML paths (also stored in each checkpoint's provenance).
