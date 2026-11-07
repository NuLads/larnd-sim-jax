# Gauss-Newton on the calibration toy — convergence, cost and scaling

*larnd-sim-jax · measurements taken 2026-07-30 on SLURM `turing` (RTX 2080 Ti, 11 GiB)*

## Setup

A deliberately small testbed for optimiser work: **pure 5-parameter calibration**
(`Ab`, `eField`, `tran_diff`, `long_diff`, `lifetime`), truth positions, truth dE/dx, **no
nuisances**. The same input file is used as both simulation and target, with explicit
`--set_target_vals` / `--set_init_params` placing the initial guess a controlled fraction from
truth. **Electronics noise is ON** throughout — the loss under study depends on it.

Scripts: `optimize/scripts/start_gn_toy.sh` (GN) and `optimize/scripts/start_adam_toy.sh` (Adam).
Runs complete in 1–10 min, so the loop is minutes rather than the ~14 h of a production fit.

---

## 1. A GN configuration that converges

The failure mode that had blocked GN was **not** a bad search direction. The oversized step *does*
lower the loss, so Levenberg–Marquardt accepts it — backtracking only fires on a loss *increase* —
and a post-hoc saturation guard then reverted the entire run. The sigmoid box `|θ| ≤ 3.5` was
simply an unmodelled constraint.

Four env-gated additions to `optimize/fit_params.py`, **all defaulting to prior behaviour**:

| knob | what it does | why |
|---|---|---|
| `LARND_GN_DAMP_MODE=abseig` | saddle-free Newton: `V·diag(max(\|λ\|,floor))·Vᵀ` | the Hessian is **indefinite at nearly every iterate**; the raw Newton step partly *ascends*. Cut the first step 46.4 → 6.35 |
| `LARND_GN_BOUND_TR=0.95` | fraction-to-boundary step scaling | stops an accepted step leaving the box |
| `LARND_GN_ACTIVE_SET=1` | drop coordinates pinned at a bound and pushed outward, re-solve on the free block | saturation-aware; **releases** coordinates as the gradient turns inward |
| `LARND_GN_STALL_PATIENCE=10` | tolerate N consecutive non-improving iterations, refreshing curvature | LM used to terminate permanently on **one** failed iteration |

**Recommended configuration**

```
--gn_curvature exact
LARND_GN_DAMP_MODE=abseig   LARND_GN_BOUND_TR=0.95   LARND_GN_ACTIVE_SET=1
LARND_GN_STEP_CAP=1.0       LARND_GN_STALL_PATIENCE=10
```

Result (2-batch toy, loss floor 1914.66):

| init offset | final loss | iters | wall | retries |
|---|---|---|---|---|
| 1 % | **1914.66** | 10 | 214 s | 0 |
| 5 % | **1914.66** | 29 | ~416 s | 0 (**3/3 repeats identical**) |
| *baseline, all knobs off* | 3508 / 5613 | 1 | — | **reverts, zero progress** |

Both offsets reach the *same* floor from different starting points, and `|step|` decays to ~5e-6.
The old rank-based `LARND_GN_SOFT_MODES` projection is **strictly dominated** (5–10 worse) because
it discards a subspace permanently.

`STEP_CAP=1.0` is what makes the 5 % case deterministic: its init sits near `Ab`'s 0.9 bound and
the uncapped first step (|step| = 3.2) drives it in. `0.5` is too tight (41 iters vs 29).

**GGN is not worth it.** At 1 % it reaches the same floor but the economics cancel exactly —
2.8× cheaper per iteration, 2.7× more iterations, **200 s vs 204 s**. At 5 % it never converged
(60 iterations, still descending, λ collapsed to 1e-12, `Ab` pinned). Use exact curvature.

---

## 2. Adam vs GN

![Adam vs GN convergence](../plots/fig1_adam_vs_gn.png)

Like-for-like: a **single 48.5 cm batch** (`NB=1 LEN=50`), so one Adam step and one GN gradient see
identical data and the losses are the same quantity (verified — Adam `losses_iter[0]` = 1605.0289
matches GN `init: loss` = 1605.029297). Adam's loss is very noisy (1605 → 1010 → 1732 in three
steps), so a **running minimum** is used. Threshold: loss ≤ 999.5.

| offset | method | iters to thr | wall to thr | s/iter | final best |
|---|---|---|---|---|---|
| 1 % | GN (recipe) | **5** | 58 s | 11.50 | **999.19** |
| 1 % | Adam const LR | 40 | **28 s** | 0.69 | 999.32 |
| 1 % | Adam annealed | 52 | 36 s | 0.70 | 999.29 |
| 5 % | GN (recipe) | **22** | 243 s | 11.04 | **999.19** |
| 5 % | Adam const LR | never | — | 0.69 | 1000.62 |
| 5 % | Adam annealed | never | — | 0.70 | 1030.94 |

**Near the optimum Adam wins wall-clock 2.1×.** A GN iteration costs ~17× an Adam iteration and
buys only ~8× fewer, so the premium is not repaid.

**GN's value is robustness, not speed.** From 5 % Adam never reaches the threshold in 400
iterations, plateauing 1.4 (const) to 32 (annealed) above the floor, while GN converges to 999.19
from both starts.

**Memory: GN 6487 MiB vs Adam 5167 MiB — only 1.26×.** The HVP-built Hessian keeps GN near
gradient-level memory rather than the naive (P+1) = 6×.

Annealing *hurt* Adam here (1030.94 vs 1000.62 at 5 %). Not a contradiction of the production
result, where annealing over 10 000 iterations at 400 cm with nuisances cured a co-adaptation
instability; on a 400-iteration 5-parameter toy it just shrinks the step prematurely.

---

## 3. Cost structure and batch scaling

![GN cost structure and scaling](../plots/fig2_gn_scaling.png)

**Batch *count* — memory flat, time linear.** Peak GPU 6487 / 6489 / 6489 / 6489 MiB for
1 / 2 / 4 / 8 batches (0.03 % spread). `_full_batch_lgh` walks batches sequentially, so peak memory
is the largest *single* batch, never the sum. **Raising the batch count costs time, never memory.**

**Batch *size* — the binding constraint.** Over all 10 converged points:

> **peak_MiB = 3471 + 70.1 × (cm/batch)** → **~30.8 GiB at 400 cm** (bootstrap 16–84 %:
> 27.9–35.0; slope 68.4 [61.3–83.2] MiB/cm)

So GN at production 400 cm/batch **fits on a 40 GiB A100**, and never on an 11 GiB card
(practical ceiling ~85–90 cm/batch; 110–116 cm OOMs). This agrees with the independently measured
25.17 GiB for a plain Adam gradient at 400 cm.

**Throughput saturates early**: 2.3× total gain from 19 → 80 cm/batch, but **2.06× of it is already
had by ~43 cm**; 43 → 80 cm buys only **+11 %** while memory goes 6.2 → 9.4 GiB. There is little
throughput reason to push batch size toward the memory limit.

---

## Measurement traps encountered (all cost a wrong answer at least once)

1. **`peak_mem` silently reads 0.00 GiB.** `XLA_PYTHON_CLIENT_ALLOCATOR=platform` makes jax
   `memory_stats()` return nothing. Do **not** "fix" this by switching to `bfc` — that OOM'd all 8
   arms even at LEN=50/NB=2; the platform allocator is what makes these runs fit in 11 GiB. Use the
   `MEMSAMPLE=1` nvidia-smi poller instead (allocator-independent).
2. **Compile contamination.** `[GN-PROF]` lines are emitted for every sweep and **each batch
   compiles separately**: a 4-batch run logged 49.8 / 14.8 / 13.9 / 25.5 s against a ~6.3 s
   steady state. Correct extraction: filter `with_hess=True`, take the **min per batch index across
   sweeps**, then the median. Reading the first sweep gives you the compiler's timings, not the
   optimiser's — this produced an incorrect "no time gain from larger batches" conclusion.
3. **`--max_batch_len` is a cap, not a target**, and requested `--max_nbatch` is not achieved.
   Always read the actual `number of simulation batches is N` and `total track length` from the log.
4. **A subset fit misled the extrapolation.** Six points gave 98 MiB/cm → ~40 GiB at 400 cm; all ten
   give 70 MiB/cm → ~31 GiB. Bootstrap before extrapolating 5× beyond the data.
5. **The toy's `rel` init mode clips.** `Ab ∈ [0.75, 0.9]` around truth 0.83488, so nominal offsets
   of 20 % and 50 % **both** clip `Ab` to exactly 0.9 and `eField` to exactly 0.6 — those two arms
   share a starting point and begin with two parameters already at a bound. Use
   `INITMODE=frac2bound` for larger perturbations.
