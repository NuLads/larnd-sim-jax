# Dossier: making the Gauss-Newton / Newton path cheaper (memory, compute, compile time)

This is an **engineering** brief, not a physics one. The science question (does a second-order
step help) is settled elsewhere. What is asked here is: **how do we make the curvature
computation fit and run fast?**

## 1. The computation

`larnd-sim-jax` is a JAX differentiable LArTPC simulation. Per batch it:
- takes ~8.6 tracks (a "400 cm batch": 100 batches, 864 tracks total),
- drifts/diffuses charge onto a pixel grid, convolves a current LUT, builds waveforms over
  ~2000 time ticks per pixel, applies a front-end/ADC model,
- and evaluates a marked-Poisson-point-process NLL against a target hit list.

Differentiated parameters:
- **5 global calibration params** (`Ab`, `eField`, `lifetime`, `tran_diff`, `long_diff`),
  sigmoid-bounded, held in a flax `struct.dataclass` (`norm_params`).
- **~4000 per-segment `log_dedx` nuisances** (student-t prior + soft barrier + mean constraint).
- **per-track geometry**: transverse displacement in a sine basis, 2K coefficients per track
  with K=clip(round(len/40),3,12), 864 tracks.

For the GN polish the nuisances are **frozen** (passed in as fixed arrays) and only the 5x5
calibration Hessian is needed.

Hardware: A100 40 GB (`ampere`, currently 0 of 120 free); RTX 2080 Ti 11 GB (`turing`,
12 of 140 free). All arithmetic is float32 — `jax_enable_x64` appears nowhere in the tree.

## 2. Measured costs and failures

| observation | value |
|---|---|
| `jax.jacfwd(jax.jacrev(loss))` on ONE 400 cm batch | requested **129 GB**, OOM |
| earlier prototype OOMs (per-batch jitted closures) | 4.7 GB, 8.3 GB, 18.2 GB, 80.3 GB |
| exact Hessian, len50 config | ~38 s for 7 batches |
| Fisher/GGN, len50, once compiled | ~0.4 s per cached batch |
| GN polish at len50/33 batches | ~5-7 min per GN iteration |
| matrix-free GGN/CG on the **geometry** block | `xla.gpu.CompilationResultProto exceeded maximum protobuf size of 2GB: 3084031029`, then **445 s/iteration** |
| a chain/geometry L-BFGS attempt | fresh compile per batch, ~170 s, flat in problem size |
| exact-vs-GGN Frobenius mismatch | 1.75-4.27x, **growing** with batch count (1.74x at 1 batch -> 4.88x at 7) |

## 3. What has already been done

- **`data-as-args`** (in `optimize/prototype_gn_calib.py`): per-batch jitted closures baked the
  batch data in as device constants, producing one compiled program *per batch* plus constants
  -> OOM. Fixed by padding all batches to global-max shapes and passing data as arguments, so a
  single compiled program is shared across batches.
- **Static shapes**: `unique_size` (padded count of unique pixels) and `roi_override` (ROI
  counts padded to 128-multiples) are pinned so the traced program has static shapes;
  `pad_to_closest_multiple` has an early-return when already padded.
- **HVP Hessian** (added today, `_hess_by_hvp` in `optimize/fit_params.py`): replaces
  `jacfwd(jacrev(f))` with P sequential `jax.jvp(jax.grad(f), (x,), (e_k,))` calls. Rationale:
  `jacfwd` pushes all P tangents through one reverse-mode tape simultaneously (~P x tape
  memory); P sequential HVPs cost ~1 x tape each. Unit-tested to reproduce
  `jacfwd(jacrev(.))` to **0.0** max absolute difference on a non-quadratic test function with
  the same pytree shape. **Not yet measured on real data.**
- `JAX_COMPILATION_CACHE_DIR` is set in some launch scripts.
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` appear in
  two batch-size scan scripts but **not** in the production or GN launch scripts.

## 4. Relevant code

- `optimize/fit_params.py`
  - `compute_loss` (~line 1720): the single entry point; flags `with_loss/with_grad/with_hess`,
    `log_dedx`, `parent_ids`, `chain_angles_per_track`, `unique_size`, `roi_override`,
    `return_loss_closure`.
  - the `with_hess` branches (~1798, ~2091, ~2126) — three separate `jacfwd(jacrev(...))` sites.
  - `_hess_by_hvp` (~line 435), `GaussNewtonCalibFitter` (~line 4130),
    `_full_batch_lgh` (~line 4318, accumulates loss/grad/Hessian over batches),
    `_get_fisher_fn` / `fisher` (~line 4270, the GGN path),
    the LM loop with backtracking (~line 4670).
- `optimize/strategies.py` — `LUTProbabilisticSimulation.predict`, `ProbabilisticLossStrategy.compute`.
- `src/larndsim/sim_jax.py` — `simulate_wfs`, `get_roi_counts`, `pad_to_closest_multiple`.
- `src/larndsim/detsim_jax.py` — drift, diffusion, current model, pixel/tick assembly.
- `optimize/prototype_gn_calib.py` — the standalone prototype with data-as-args and its own
  `_hvp_da` / `_hess_seq` implementation.

## 5. Your task

Propose **concrete, implementable methods to reduce memory, wall-clock, and compilation time**
for computing curvature (Hessian, Hessian-vector products, or a usable approximation) for the
5 calibration parameters on this simulation.

Consider anything relevant, e.g.: checkpointing/rematerialisation strategy and where to place
it; forward-over-reverse vs reverse-over-forward; whether the 5x5 structure permits a
fundamentally cheaper scheme; batching/chunking and accumulation order; `donate_argnums` and
buffer reuse; allocator and preallocation settings; avoiding recompiles (shape polymorphism,
static-argnum hygiene, persistent compilation cache); mixed precision; whether the Fisher/GGN
can be made trustworthy or cheaper; stochastic/randomised curvature estimators; exploiting
the frozen nuisances; and whether any of it can fit in 11 GB.

For each proposal state: what changes concretely (file/function level), the expected saving and
your reasoning for it, the risk of changing numerical results, and how to verify it cheaply.
Rank by (expected saving) / (implementation risk + effort).

**Verify against the code before asserting** — read the functions above rather than assuming
what they do. Do NOT launch SLURM jobs and do NOT modify repo files. `/lscratch` is nearly
full: keep temp files tiny and delete them. You may run small CPU-only JAX experiments inside
the container (`apptainer exec -B /sdf,/fs,/lscratch larndsim-jax_main.sif python3 ...`,
with `JAX_PLATFORMS=cpu`) to measure things like tape memory scaling on toy problems.

Answer in under 700 words.
