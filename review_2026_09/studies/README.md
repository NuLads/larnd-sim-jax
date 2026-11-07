# Study code of the review campaign

A copy of the hand-written code in the study workspaces under
`/sdf/scratch/neutrino/pgranger/review_2026_09/<STUDY>/`. It holds the submit scripts, sbatch templates,
builders, readouts and patches of the review copies. It also has the small text records: `*_result.txt`
readout outputs, notes, and the job ledgers `*_jobs.txt`, `set*_ids.txt` and `submitted.txt`. First
snapshot: 7 Nov 2026.

The scripts contain absolute paths and are run **from /sdf/scratch**, not from here. This directory
records what ran. The rules and the verdict of each study are in `../REVIEW.md` (grep the study name).
The production pipeline is described in `PRODUCTION.md`.

## Not included

- Data, logs, checkpoints, caches and fit results.
- The `tree/` review copies of the repo. Their changes are kept as patches:
  `DXPROF/patch_dxprof.py`, `GNPROF/patch_gnprof.py` (GNREAL reuses the GNPROF tree) and
  `TRUTHC/patch_truthc.diff`. The last one only exists here; it was rebuilt on 7 Nov from `TRUTHC/tree`
  against the production `optimize/fit_params.py`, and it holds 50 added lines for TRUTHC and TRUTHC2.
- Generated scripts (`inner_*.sh`, `*.built.sh`, ~990 files). One example per family is kept; the
  generators are here. Main families:

  | Family | Generator |
  |---|---|
  | `KNOT/inner_sci_full_<NAME>{FR,H}.sh` | `KNOT/run_knot.sbatch` + `CONT/make_prod_{cont,fast}.py` |
  | `KNOT/inner_sci_full_<NAME>P.sh` | `KNOT/run_polish.sbatch` + `UQ/make_jn_dump.py` |
  | `KNOT/inner_sci_full_{B,V}<n>s<n>*.sh` | `DATASZ/run_ds.sbatch` + `DATASZ/make_ds.py` |
  | `KNOT/inner_sci_full_*HNC.sh` | `NOCLIP/make_noclip.py` |
  | `UQ/inner/*`, `TRUTHC/inner/*` | `UQ/make_jn_dump.py`, `TRUTHC/make_tc.py` |

  The stage-1 production inner scripts have no generator: each one is a copy of
  `KNOT/inner_sci_full_EXP3C75.sh` with `--data_seed` and `--test_name` changed.
- Backup copies of repo files (`*.orig`, `*.new*`, `*.before_*`, `*.pre_*`). Their content is in the
  git history.

## Refreshing the copy

After editing scripts on /sdf/scratch, run `bash review_2026_09/studies/sync_from_scratch.sh` and commit.
It copies the files listed in `MANIFEST.txt` and never deletes anything. To track a new file, add its
path (relative to the scratch review directory) to `MANIFEST.txt` first.

## Studies

| Study | What |
|---|---|
| KNOT | Production launchers and SLURM templates, `polish_estimate.py`, the launch ledger, and the KN10 ensemble and paper-support readouts |
| CONT | Builders of the FR and H continuations |
| UQ | Uncertainty library (`uq_lib.py`, `uqt_common.py`), the POLISH dump builder, and readouts of the UQ test, F1/F2 and bias diagnostics |
| POLISH | Conditional-minimum (POLISH) estimator studies, including the joint-polish prototypes |
| DXPROF, GNPROF, GNREAL | Derivative-based widths: dE/dx-profiled sandwich, Gauss-Newton profile, check on production fits (6 Nov) |
| TRUTHC | Truth-point checks TRUTHC and TRUTHC2, and endpoint resolution |
| NOCLIP | Is the end-point gap the calibration clip? |
| DATASZ | Accuracy versus data volume and batch size |
| ENDFIT | Endpoints from the hits, then fitted: threads (a) to (u), including C75 and the cache adoption |
| SPEED | Speed and memory package (22 to 24 Oct): benchmarks and adoption patches |
| TARGETMOVE | Moving the target dirs to /sdf/data |
| LARNDSIM_CMP | Comparison with the reference larnd-sim (scripts and configs only) |
| NEIGH | Neighbour-pixel induction |
| FIG4, PAPERJOBS, figs | Paper-figure jobs |
| BATCH, CLIP, CURVEK6, DRIFTHITS, DRIFTTRU_resume, E_truthdiag, GEOFRZ, HITSEED, LAD, L_laplace, NAIVE, NZLAD, PROBSTOCH, PST, QUEUE, ROUND1, ROUND2, SCHED, STRFRZ, TW0 | September and October review studies (see `../REVIEW.md` and the September report) |
| top-level `*.sh` | The 25 Sep data move from /sdf/group to /sdf/data (copy, verify, swap for symlinks) and moving finished staging runs to /sdf/data |
