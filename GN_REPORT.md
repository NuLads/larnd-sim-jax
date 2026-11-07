# Gauss-Newton in larnd-sim-jax: where second-order helps, where it doesn't, and why

*Consolidated report — measurements July 2026, `larnd-sim-jax`, SLURM `turing` (RTX 2080 Ti, 11 GiB)
and `ampere` (A100, 40 GiB). Written 2026-07-31.*

---

## Verdict up front

The fit has three parameter blocks. Second-order methods have a **different answer in each**, and
lumping them together is what made the earlier record look contradictory.

| block | size | verdict | status |
|---|---|---|---|
| **5 calibration scalars** | 5 | **GN works** — converges to the loss floor where Adam stalls | recipe found, validated on a toy |
| **~4000 per-segment dE/dx** | ~4000 | **GN works but TIES Adam** at equal wall-clock — the block is objective-limited, not optimiser-limited | built & validated (§7); **not worth deploying** |
| *(its `mean_constraint` term)* | 1 scalar | **weight is ~6× too stiff** — costs 4× calibration accuracy and a +108 % lifetime bias | measured, §6; **change the default** |
| **per-track geometry** | ~6/track × 864 | **GN loses to Adam** — on accuracy *and* speed | closed; don't re-chase |

The headline is not "GN is faster." It is **"GN is more robust, and only in the small block."** A GN
iteration costs ~17× an Adam iteration and buys only ~8× fewer of them, so near the optimum Adam
wins wall-clock by ~2×. GN earns its place where Adam *stops making progress at all*.

**Two of the three blocks turned out to be objective-limited, not optimiser-limited.** For
calibration (§4) and dE/dx (§7–8) alike, a better optimiser reaches a lower loss and a *worse*
answer. That is the single most important theme here, and it caps what any optimiser work can buy.

**Read §11 before acting on §6 or §9.** The dE/dx recommendations below were measured on toys
with **true geometry**; at production scale, with fitted positions, they **do not reproduce**, and
the seed scatter (lifetime sd 36–62 %) is too large for 3 seeds to resolve them either way. What
survives production contact is everything measured on *curvature or truth data* rather than on fit
quality: the Hessian decomposition, the 5-HVP probing, the mean-constraint target being correct to
0.09 %, and the diagnosis of the MAE puzzle.

**The most actionable finding is not about Gauss-Newton at all**, and its *diagnosis* stands even
though its production benefit is unproven: the `dedx_mean_constraint` weight is ~13× stiffer than
the batch-to-batch spread supports (§6.3), and on toys lowering it 1e5 → 3000 improved calibration
**4×** and cut a **+108 % lifetime bias to +16 %** — paired with the (scale, shape)
reparametrisation (§9), to −2.4 %. **None of that was reproducible at production scale (§11).**

---

## 1. The calibration block: a configuration that converges

### 1.1 What was actually broken

The failure that had blocked GN for weeks was **not a bad search direction**, which is what every
earlier diagnosis assumed. The mechanism, once traced:

1. The raw Newton step is enormous (‖step‖ = 46.4 against sigmoid bounds at |θ| ≈ 3.5).
2. That oversized step **does lower the loss** — so Levenberg-Marquardt *accepts* it, because
   backtracking only fires on a loss **increase**.
3. Parameters land outside the box, and a post-hoc saturation guard then reverts **the entire run**.

The sigmoid box was simply an **unmodelled constraint**. Nothing was wrong with the curvature or
the linear algebra; the optimiser had no representation of the feasible set.

A second, independent contributor: LM terminated permanently after **one** non-improving iteration.

### 1.2 The four fixes

All are env-gated in `optimize/fit_params.py` and **default to prior behaviour** (unit-tested to
0.0 max abs difference against `-solve(A,g)` over 500 random systems).

| knob | what it does | why it was needed |
|---|---|---|
| `LARND_GN_DAMP_MODE=abseig` | saddle-free Newton, `V·diag(max(\|λ\|,floor))·Vᵀ` | the Hessian is **indefinite at nearly every iterate** — the raw step partly *ascends*. Cut the first step 46.4 → 6.35 |
| `LARND_GN_BOUND_TR=0.95` | fraction-to-boundary scaling, applied last | stops an accepted step leaving the box |
| `LARND_GN_ACTIVE_SET=1` | drop coords pinned at a bound and pushed outward, re-solve on the free block | saturation-aware; visibly shrinks 4-D → 3-D → 2-D → 1-D and **releases** coords as the gradient turns inward |
| `LARND_GN_STALL_PATIENCE=10` | tolerate N non-improving iterations, refreshing curvature | LM used to quit permanently on one failed iteration |

**Recommended configuration:**

```
--gn_curvature exact
LARND_GN_DAMP_MODE=abseig   LARND_GN_BOUND_TR=0.95   LARND_GN_ACTIVE_SET=1
LARND_GN_STEP_CAP=1.0       LARND_GN_STALL_PATIENCE=10
```

### 1.3 Result (2-batch toy, loss floor 1914.66)

| init offset | final loss | iters | wall | retries |
|---|---|---|---|---|
| 1 % | **1914.66** | 10 | 214 s | 0 |
| 5 % | **1914.66** | 29 | ~416 s | 0 (**3/3 repeats identical**) |
| *baseline, all knobs off* | 3508 / 5613 | 1 | — | **reverts, zero progress** |

Both offsets reach the **same** floor from different starting points; ‖step‖ decays 1.69e-2 →
8.0e-5 → 4.7e-6, so this is genuine convergence, not truncation.

`STALL_PATIENCE` is the decisive knob at 5 %. Root cause proven by two runs of an *identical*
config diverging: at iteration 1 the available improvement (~2e-4) equalled the loss's
run-to-run reproducibility (~5e-4). One run got a favourable rounding and went on to converge;
the other escalated λ 8× to 2.18e+02, exhausted `gn_max_backtrack=8`, and quit with ~3670 of loss
reduction still on the table. **Once λ is that large the step is tiny and the loss delta is under
the noise — more backtracking cannot help. A fresh curvature evaluation can.**

### 1.4 Two conclusions that overturn earlier consensus

**Exact curvature beats GGN.** This contradicts an earlier 10-investigator consensus. At 1 %
offset the economics cancel *exactly* — GGN is 2.8× cheaper per iteration and needs 2.7× more of
them (200 s vs 204 s). At 5 % **GGN never converged**: 60 iterations, still descending, λ collapsed
to 1e-12, `Ab` pinned at its bound. GGN curvature is too stiff to unblock pinned coordinates.

**The old rank-based soft-mode projection is strictly dominated** (final loss 5–10 worse) because
it discards a subspace permanently. It buys stability by *declining to correct* the soft
parameters — the converged 1 % run improved `eField` 50× but left `lifetime` and both diffusions
~2 % *worse*.

---

## 2. Adam vs GN, like for like

![Adam vs GN convergence](plots/fig1_adam_vs_gn.png)

A **single 48.5 cm batch**, so one Adam step and one GN gradient see identical data and the losses
are the same quantity (verified: Adam `losses_iter[0]` = 1605.0289 matches GN `init: loss` =
1605.029297). Adam's loss is very noisy (1605 → 1010 → 1732 in three steps), so a **running
minimum** is used.

| offset | method | iters to thr | wall to thr | s/iter | final best |
|---|---|---|---|---|---|
| 1 % | GN (recipe) | **5** | 58 s | 11.50 | **999.19** |
| 1 % | Adam const LR | 40 | **28 s** | 0.69 | 999.32 |
| 1 % | Adam annealed | 52 | 36 s | 0.70 | 999.29 |
| 5 % | GN (recipe) | **22** | 243 s | 11.04 | **999.19** |
| 5 % | Adam const LR | never | — | 0.69 | 1000.62 |
| 5 % | Adam annealed | never | — | 0.70 | 1030.94 |

**Near the optimum Adam wins wall-clock 2.1×. GN's value is robustness**: from 5 % Adam never
reaches threshold in 400 iterations, plateauing 1.4 to 32 above the floor, while GN converges to
999.19 from both starts.

Annealing *hurt* Adam here (1030.94 vs 1000.62 at 5 %). Not a contradiction of the production
result, where annealing over 10 000 iterations at 400 cm with nuisances cured a co-adaptation
instability — on a 400-iteration 5-parameter toy it just shrinks the step prematurely.

**Memory: GN 6487 MiB vs Adam 5167 MiB — only 1.26×**, because the Hessian is built from P
sequential HVPs (`_hess_by_hvp`) rather than `jacfwd(jacrev)`, which would push all P tangents
through one reverse tape at ~(P+1) = 6× the memory.

---

## 3. Cost structure and batch scaling

![GN cost structure and scaling](plots/fig2_gn_scaling.png)

**Batch *count* is free in memory.** Peak GPU 6487 / 6489 / 6489 / 6489 MiB for 1 / 2 / 4 / 8
batches — a 0.03 % spread. `_full_batch_lgh` walks batches sequentially, so peak memory is the
largest *single* batch, never the sum. Raising the batch count costs time, never memory.

**Batch *size* is the binding constraint.** Over all 10 converged points:

> **peak_MiB = 3471 + 70.1 × (cm/batch)** → **~30.8 GiB at 400 cm**
> (bootstrap 16–84 %: 27.9–35.0; slope 68.4 [61.3–83.2] MiB/cm)

So GN at production 400 cm/batch **fits on a 40 GiB A100, and never on an 11 GiB card** (practical
ceiling ~85–90 cm/batch; 110–116 cm OOMs). This agrees with the independently measured 25.17 GiB
for a plain Adam gradient at 400 cm.

**Throughput saturates early**: 2.3× total gain from 19 → 80 cm/batch, but **2.06× of it is already
had by ~43 cm**. 43 → 80 cm buys only **+11 %** while memory goes 6.2 → 9.4 GiB. There is little
throughput reason to push batch size toward the memory limit.

---

## 4. The finding that limits what any of this means

**The toy is objective-limited, not optimizer-limited.** Measured with init == truth:

- **loss(truth) = 1916.98**, but the best GN arm reaches **1914.66** — *below* the loss at truth.
- **‖g‖(truth) = 486.6, not zero.** Starting exactly at truth, GN's first step *lowers* the loss
  while moving parameters to 4.34 % MAPE (lifetime +14.4 %).

The cause is **electronics noise in the target**:

| at truth (nb=8) | lifetime | long_diff | Ab | MAPE |
|---|---|---|---|---|
| noise ON | **+14.36 %** | +5.12 % | −0.31 % | 4.34 % |
| noise OFF | **−4.88 %** | −29.78 % | −0.03 % | 7.34 % |
| noise OFF, nb=32 | **−2.14 %** | −4.99 % | +0.01 % | **1.48 %** |

Turning target noise off **flips the sign** of the lifetime displacement and restores 1/√N
shrinkage. Crucially, **only the target's noise matters** — `--no-noise-guess` reproduces
noise-ON-both *exactly*; `--no-noise-target` reproduces noise-OFF-both *exactly*. Guess-side noise
is exactly inert, as expected for a marked-Poisson NLL where the model supplies an *expected*
intensity rather than a sampled realisation.

**Two consequences:**

1. For GN algorithm work, use the **noise-OFF** toy. At 1.5–4 % MAPE the displacement of the
   objective's minimum from truth is comparable to or larger than anything optimiser tuning
   changes, so "distance to truth" is a poor figure of merit on the noise-ON toy.
2. **An open physics question, not yet verified:** real data always has noise in the target, so
   the +14 % row is the physically relevant one. Lifetime may be biased **up** on real data unless
   electronics noise is represented in the expected-intensity term of the loss. Plausible
   mechanism — noise adds spurious charge that matters most at long drift where true signal is
   smallest, mimicking less attenuation. **Caveat: measured at one point (truth), 50 cm, 8 batches,
   from a single Newton step. Not a converged fit.** This deserves real work.

---

## 5. The dE/dx block: exact structure, and an O(N) solve

This is the most promising unexploited result in the whole investigation. Measured with
forward-over-reverse HVPs on an S2-like config (true geometry, `--fit_dedx`, no position fitting,
`--max_batch_len 60 --max_nbatch 2`, production prior weights).

### 5.1 The decomposition

![dE/dx Hessian decomposition](plots/fig_dedx_hessian_decomposition.png)

The total Hessian looks dense and unpromising. It is not — it is the sum of three terms with three
*exact* structures:

$$H = H_{\rm data} + \mathrm{diag}(\text{student-}t + \text{barrier}) + w_{\rm mean}\, q q^{\!\top}$$

1. **Data/likelihood term: banded**, with a strong band at |i−j| ≤ 2 and a negligible tail
   (§5.2). Charge from a segment only reaches pixels within `number_pix_neighbors=4`, so only
   nearby segments couple.
2. **Student-t prior + soft barrier: exactly diagonal.** Verified by re-running with
   `--dedx_mean_constraint_weight 0`: off-diagonals became **exactly 0.0**, not small — zero.
   Perfectly separable.
3. **Mean-dE/dx constraint: exactly rank-1**, and the *sole* source of dense coupling.
   `mean_dedx` is a length-weighted linear functional of every segment, so
   `w·(mean_dedx − target)²` has Hessian `2w·qqᵀ`.

### 5.2 The quantitative evidence

![dE/dx Hessian structure](plots/fig_dedx_hessian_structure.png)

**Left — the band.** Decay of `|H_ij| / max_i H_ii` for the likelihood term, measured on the
`w_mean = 0` control (see below for why that matters):

| separation | 0 | 1 | **2** | 3 | 4 | ≥ 5 |
|---|---|---|---|---|---|---|
| relative magnitude | 1 | ~3e-2 | **~1e-3** | ~3e-7 | ~3e-9 | **exactly 0.0** |
| entries that are exactly 0 | 0/11 | 2/20 | 6/18 | 10/16 | 12/14 | all |

**The band is not a clean fixed bandwidth, and it is not tridiagonal.** Two earlier statements in
this investigation were wrong and are corrected here:

- The original note recorded the term as "exactly tridiagonal, off-diagonals exactly zero for
  separation ≥ 2." The separation-2 entries are ~1e-3 of the diagonal — ~10⁴× above float32 ε.
  Real.
- My own first correction then claimed "bandwidth 2, exactly zero beyond." **Also wrong.** There
  are genuine couplings at separations 3 and 4, at ~3e-7 and ~3e-9 relative. They are verified by
  **symmetry**: the pairs (0,3)/(3,0), (6,9)/(9,6), (7,10)/(10,7) and (6,10)/(10,6) agree to 6–7
  significant figures. Float32 rounding does not produce symmetric agreement to 6 digits; a real
  Hessian entry must.

The structure is therefore: **a strong band to |i−j| ≤ 2, then a sparse tail of genuine but
~1e-7-relative couplings at |i−j| = 3–4, then exact zeros.** Most entries beyond separation 2 are
*exactly* 0.0 — the reach in *index* space is not uniform, because whether two segments share a
pixel depends on their physical lengths and orientation relative to the grid, not on their index
distance.

**A methodological trap worth recording.** The band tail can only be measured on the `w_mean = 0`
control. At production weights the rank-1 term makes `H_total` and `H_prior` both ~1e4, so forming
`H_data = H_total − H_prior` is a catastrophic cancellation losing ~1e-3 of absolute precision —
about 2e-6 in relative terms, which is *above* the entire real tail. Batch 1 accordingly shows two
spurious entries at separations 5 and 9, both exactly 2⁻¹² (a single float32 ULP) and both with an
**exactly zero transpose partner** — the signature of rounding, not physics. They are drawn hollow
(left panel) and gray (decomposition panel).

**Consequence for the implementation.** Truncating at bandwidth 2 discards couplings of ~3e-7
relative magnitude. That is negligible — but it is an **approximation, not an identity**, and a
Thomas-algorithm tridiagonal solve would additionally discard the ~1e-3 separation-2 band, which
is not negligible. The correct solve is a bandwidth-2 banded LU. Still O(N).

**Right — the rank-1 term.** Measured prior off-diagonals plotted against `q_i q_j`: max relative
residual **1.4e-5**, median 2.2e-6 — exact to float32. 156 off-diagonal entries are reproduced by
13 numbers. §5.3 derives why and closes the loop with an independent numerical test.

### 5.3 Why the mean constraint has this effect — derived, then measured

![Mean constraint effect](plots/fig_dedx_hessian_meanconstraint.png)

The rank-1 structure is not an implementation accident; it follows in three lines, and every step
below was checked numerically.

The penalty is `R = w·(m − t)²` with `m = Σᵢ lᵢ·dedxᵢ / L` (`fit_params.py:1982–1995`), and the
fitted parameters are `θ = log_dedx`, so `dedxᵢ = exp(θᵢ)`. Then

$$\frac{\partial m}{\partial\theta_i} = \frac{l_i\,\mathrm{dedx}_i}{L} \equiv g_i,
\qquad \frac{\partial^2 m}{\partial\theta_i\partial\theta_j} = \delta_{ij}\,g_i$$

$$\boxed{\;H_{\rm mean} = 2w\,g g^{\!\top} \;+\; 2w(m-t)\,\mathrm{diag}(g)\;}$$

**Why it is dense at all: the square sits outside the sum.** The student-t and the barrier are
*sums of squares* — each term mentions one segment, hence exactly diagonal. The mean penalty is
the *square of a sum*, and `(Σaᵢ)² = Σaᵢ² + 2Σ_{i<j}aᵢaⱼ`; those cross terms **are** the
off-diagonal. Put physically: the penalty never looks at an individual segment, only at their
average, so raising one segment obliges you to lower another. That trade-off *between* segments is
what an off-diagonal entry encodes.

**Why rank-1 rather than merely dense: it constrains one number.** Any change to the parameters
that leaves `m` unchanged is invisible to the penalty — zero curvature. In N dimensions there are
N−1 independent ways to move without changing the mean, so N−1 eigenvalues are exactly zero. The
general rule is *k constraints → rank k*; here k = 1. The constraint pins the *mean* of the dE/dx
profile and says nothing whatever about its *shape*.

**Why the extra diagonal term:** if you fitted `dedx` directly, `m` would be linear in the
parameters and the Hessian would be purely rank-1. Because `log_dedx` is fitted, `m` is a sum of
exponentials and each carries its own curvature — a diagonal contribution, carrying the factor
`(m − t)`, so it vanishes when the constraint is satisfied.

**Verification.** Differencing the two batch-0 runs (`w_mean = 0` vs `1e5`, everything else
identical) isolates `H_mean` exactly:

| test | prediction | measured |
|---|---|---|
| off-diagonal is `qᵢqⱼ` | exact | max rel. err **3.0e-5** over 60 entries |
| `(diag − qᵢ²)/qᵢ` constant | `= √(2w)(m−t)` | constant to **6.7e-4** |
| implied residual | — | `m − t = −0.012996` → **m = 1.874004** |
| **independent closure:** `Σᵢgᵢ = m` by definition | — | `Σg =` **1.873995** |

The last two lines are the real check: the mean dE/dx is recovered **two independent ways — from
the off-diagonal magnitudes and from the diagonal residual term — agreeing to 5 decimals.**

*(An earlier version of this report claimed the recovered `q` "sums to the track length". That was
circular — `q/Σq × L` sums to `L` identically — and physically wrong, since `gᵢ ∝ lᵢ·dedxᵢ`, not
`lᵢ`. The closure above replaces it.)*

**What the form explains, all measured:**

1. **One stiff direction.** `|q|² = 73356` vs top eigenvalue **74186**, eigenvector overlap
   **0.999990**, and a **62× gap** to the second eigenvalue. The κ 25 → 23874 story is that one
   direction.
2. **The dominance collapse is structural.** For rank-1, row `i`'s ratio is
   `qᵢ/Σ_{j≠i}qⱼ ≈ 1/(N−1)` = 0.0833 at N = 13; measured median **0.0857**. **This scales badly**:
   at production N ≈ 4000 the ratio would be ~2.5e-4, an argument for a *direct* structured solve
   over anything iterative or diagonally preconditioned.
3. **Short segments get their curvature cancelled.** The diagonal contribution is `qᵢ² + c·qᵢ`
   with `c = √(2w)(m−t) = −5.81`, so it is **net negative for any segment with `qᵢ < |c|`**.
   Batch 1's segment 12 — implied length 0.036 cm — ends with total diagonal **3.67**, four orders
   below its neighbours, and is the source of the minimum eigenvalue 3.11. **This was later
   confirmed dynamically**: every one of 54 fit arms in §6 produced exactly one runaway segment
   collapsing to 0.24–0.43 MeV/cm while the rest tracked truth.

### 5.4 The resulting algorithm

> **Bandwidth-2 banded LU for the band + one Sherman–Morrison correction for the rank-1 term.
> No CG needed. O(N).** Essentially free next to the backward pass already being computed.

**Do not implement diagonal-only.** 3 of 13 diagonals of the *data* term are negative (short,
weakly-constrained edge segments). The *total* Hessian is positive definite (min eigenvalue 3.11
batch 1, 325.8 batch 0), so diagonal-only would not hit a negative pivot — but it would discard
the dominant rank-1 coupling, which is the entire difficulty.

### 5.5 Caveats — read before implementing

- **Measured only at N = 11–13 segments per batch. Production carries ~4000.** The banded +
  rank-1 claim is **extrapolated**. The band follows from the pixel-neighbourhood argument, which
  is a physical argument and should hold, but it is **unverified at scale**.
- **Measure the band on a `w_mean = 0` run.** At production weights the cancellation floor (§5.2)
  sits above the real band tail, so a production-weight measurement cannot see it and will
  manufacture single-ULP artifacts. Use the transpose-symmetry test to separate real couplings
  from rounding — it is decisive and costs nothing.
- Brute-force HVP diagonal extraction costs ~1.2 s/segment ≈ 1 h/batch at N=4000. **The structured
  solve is the only viable route, not merely the nicer one.**
- A scale-up attempt OOM'd from a DataLoader padding-tax effect: raising `--max_batch_len` pads
  *every* batch to the new global max step count.
- **Verify the structured solve against a brute-force HVP Hessian at small N before trusting it at
  4000.**

---

## 6. The mean constraint itself: right target, wrong stiffness

The Hessian work above treats `dedx_mean_constraint` as a given and asks how to solve around it.
That prompted the better question — *should it be there in that form at all?* Chasing it produced
the most actionable result in this report, and it changes a production default.

![The dE/dx mean constraint](plots/fig_dedx_mean_constraint.png)

### 6.1 Why the term exists

Historical origin (Pierre): using **only** the split-t prior to regularise dE/dx produced
*concentration around the MPV and a bias on the mean*. That is a textbook pathology and the
numbers confirm it exactly:

**MAP returns the mode; calibration needs the mean; this prior is skewed.** With `nu_R = 2.073`
the right tail is very heavy, so mode ≠ mean:

| quantity | value |
|---|---|
| prior **mode** (`DEDX_SPLIT_LOC`, where penalised estimation shrinks to) | 1.8740 |
| prior **mean** (numerically integrated) | **1.8847** |
| `dedx_mean_constraint_target` | **1.887** |

**The target is the prior's own mean.** The data has the same asymmetry — median 1.8689, mean
1.8853 — so shrinking each segment toward the mode biases the ensemble mean low by ~0.6 %. The
constraint was introduced to drag it back. The intent is sound, and §5.3 shows why a constraint on
*one scalar* is the structurally right shape of fix: the overall dE/dx scale is degenerate with
`Ab`/`lifetime`, the *shape* is not, and a mean penalty constrains exactly that one direction.

### 6.2 The target is correct — do not "fix" it

Computed over the **entire** input file (3.03 M segments, 99.3 km), three independent ways:

| quantity | value |
|---|---|
| length-weighted mean dE/dx | **1.8853** |
| unweighted mean | 1.8854 |
| Σ dE / Σ dx | 1.8853 |
| **target 1.887** | **+0.09 % off** |

Whoever set 1.887 calibrated it correctly. An intermediate measurement suggesting otherwise
(1.8538) was a 24-segment small-sample fluctuation, not a systematic offset.

### 6.3 The weight is far too stiff — measured, no fitting required

`w = 1e5` imposes `σ_m = 1/√(2w) = 0.12 %` on **every batch's** mean. But batch means genuinely
scatter, and that scatter is measurable directly from truth (panel A):

| cm/batch | true batch-to-batch σ | imposed σ | over-stiff by | honest `w = 1/(2σ²)` |
|---|---|---|---|---|
| 50 | 1.74 % | 0.12 % | 15× | 464 |
| 100 | 1.30 % | 0.12 % | 11× | ~800 |
| 200 | 0.95 % | 0.12 % | 8× | ~1500 |
| **~72 (production, see below)** | **~1.5 %** | 0.12 % | **~13×** | — |
| 400 | 0.68 % | 0.12 % | 6× | — |
| 1000 | 0.44 % | 0.12 % | 4× | ~7000 |

Since curvature ∝ `w`, dropping 1e5 → 3000 shrinks the rank-1 eigenvalue 33× and takes κ from
~23874 to roughly ~750 — most of the conditioning damage in §5, removed by one number.

> **SCALE CORRECTION.** Production (`start_sci_case.sh`) runs `SCILEN=50`, and a probe measured
> **931.9 cm across 13 batches ≈ 72 cm/batch** — not the 400 cm/batch assumed in earlier drafts.
> At that scale the true batch spread is ~1.5 %, so `w = 1e5` is **~13× too stiff**, worse than the
> 6× quoted from the 400 cm row. It also means production scale **fits an 11 GiB turing card**
> (~6.8 GiB), so ampere is a speed choice, not a requirement.
>
> **IMPORTANT CORRECTION (added after §9.3).** The `w = 1/(2σ²)` derivation above is *not* a valid
> prescription, and this table should not be used to pick a weight. Tested directly: at a
> ~46 cm/batch configuration, where the formula prescribes **w = 464**, that value is much WORSE
> than 3000 (mean abs calibration error 16.5 % vs 9.6 %; lifetime +44.1 % vs −2.4 %). The reason
> is that the constraint has **two jobs**, and σ only captures one: it is a prior on the batch
> mean *and* the thing that breaks the dE/dx↔calibration degeneracy (§6.4). Too weak and the
> degeneracy re-opens; too stiff and it injects per-batch bias. **What survives is the negative
> result — `w = 1e5` is far too stiff (robust across two configurations) — not any particular
> replacement value.** The weight must be tuned empirically at the production configuration.

**This also rules out making it a hard constraint.** Batch means really do differ by 0.68 %;
pinning them exactly would be wrong physics. A properly-weighted *soft* prior is the right object.

### 6.4 The decisive test: the stiff constraint is actively harmful

18 arms on `turing`: 17 batches, 376 segments, 1270 cm, 8000 iterations (~470 updates/batch),
**calibration initialised at truth** so any drift is injected bias, 3 seeds per arm.

| loc | `mean_w` | dE/dx mean bias | mean abs calib error | lifetime error |
|---|---|---|---|---|
| 1.874 | **1e5 (current)** | +0.210 % | **25.8 % ± 2.2** | **+107.6 % ± 11.3** |
| **1.874** | **3000** | **+0.026 %** | **6.2 % ± 2.0** | **+16.0 % ± 0.7** |
| 1.874 | 0 | −0.584 % | 6.5 % ± 2.3 | +4.9 % ± 16.5 |
| 1.8847 | 1e5 | +0.236 % | 25.9 % ± 1.9 | +110.4 % ± 5.7 |
| 1.8847 | 3000 | +0.147 % | 8.0 % ± 2.4 | +30.3 % ± 11.1 |
| 1.8847 | 0 | −0.967 % | 14.9 % ± 5.6 | +55.8 % ± 31.0 |

Three things fall out, all well outside the seed scatter:

1. **`w = 1e5` makes calibration ~4× worse** (t ≈ 11) and biases **lifetime by +108 %** (t ≈ 14) —
   *starting from truth*, on both loc values, across all six seeds.
   **Mechanism:** forcing every batch's mean to 1.887 when batch means genuinely vary ~1.5 % at
   this scale leaves the per-batch mismatch nowhere to go but the **global** calibration
   parameters. Lifetime is the most degenerate with an overall charge scale, so it absorbs it.
2. **The mode-shrinkage bias is confirmed at scale.** At `w = 0` the fitted mean lands at 1.8719 —
   essentially the prior mode 1.874 — while this data's truth is 1.8829. Pierre's original
   diagnosis, reproduced (panel C).
3. **`w ≈ 3000` is the sweet spot**: near-zero dE/dx bias (+0.026 %), the best calibration
   (6.2 %), and by far the most reproducible lifetime (sd 0.7 vs 11–31 elsewhere).

The test was **conservative**: 3000 is the honest weight for 400 cm/batch, but this toy ran at
~75 cm/batch where the honest value is ~640. So `w = 3000` was still ~5× too stiff here and *still*
won by 4×.

### 6.5 What did not work, and a retraction

- **Shifting `dedx_student_loc` to the prior mean (1.8847) is slightly *worse* on every axis** at
  `w = 3000`. **This retracts my earlier recommendation to move the two together.** The right
  division of labour is the one the design already had: the prior sits at the physical **mode** and
  does the per-segment job; the constraint fixes the **mean** and does the global job. Only the
  weight was wrong.
- **The missing Jacobian is real but minor.** The split-t is evaluated on linear `dedx` while the
  fitted parameter is `log_dedx`, so `−log p_d(e^θ)` omits `−log|d(dedx)/dθ| = −θ`. Including it
  moves the *analytic* MAP 1.8740 → 1.8780, i.e. 38 % of the mode-mean gap. **In the fit its
  effect was ≤ 0.3 % and not resolved above seed scatter.** Added as `LARND_DEDX_LOG_JACOBIAN`,
  default off — a correctness fix, not a solution.

### 6.6 Caveats

- Absolute calibration errors remain large in all arms (lifetime +16 %, long_diff +8 % even at the
  best setting). 1270 cm is far from production scale, so **this establishes the ranking, not
  production-quality numbers.**
- `w = 3000` should be re-derived at the production batch length actually used, from the table in
  §6.3, rather than copied.
- The Hessian analysis in §5 was evaluated **at initialisation**, where every segment sits at
  `log(loc)` — which is why `m` came out exactly 1.874 there. The rank-1 structure is algebraic and
  holds everywhere; those particular eigenvalues describe the starting point.

---

## 7. GN applied to the dE/dx block: built, validated, and it ties Adam

§5 argued the dE/dx Hessian has exactly exploitable structure. It was implemented and measured.
**The solve works exactly as designed. It does not beat Adam, and the reason why is the most
useful thing in this section.**

### 7.1 What was built

`_dedx_gn_step` in `optimize/fit_params.py`, launcher `optimize/scripts/start_dedx_gn.sh`,
unit tests `test_dedx_gn_numerics.py` (20 checks, CPU-only, no simulation).

**The algorithmic win: 5 HVPs, independent of N.** A bandwidth-2 matrix is recovered *exactly* by
structurally-orthogonal **mod-5 probing** — probe with `v_c[j] = 1 iff j % 5 == c`; since
`{i−2 … i+2}` are 5 consecutive integers, exactly one is `≡ c (mod 5)`, so every band entry is
read off uniquely. That replaces the ~1.2 s/segment brute-force extraction (≈1 h/batch at N=4000)
with **5 HVPs at any N**.

The dense rank-1 term would destroy that structural orthogonality, so it is subtracted
analytically from the probes and restored by Sherman–Morrison. Because `w > 0` the rank-1 enters
with a **plus** sign, so the SM denominator `1 + qᵀB⁻¹q = 112.3 > 1` — no cancellation,
unconditionally stable.

All default-off (`LARND_DEDX_OPT=gn`), with automatic fallback to the Adam step whenever the solve
finds no descent, so it cannot stall a run.

### 7.2 The band assumption holds on real data

The earlier band measurement (§5.2) was at N = 11–13 and at *initialisation*. Re-measured
**mid-fit at the real operating point**, against the dense Hessian:

| check | value |
|---|---|
| max \|off-band\| / diagonal | **2.48e-06** |
| band recovery, 5 probes vs 11 dense HVPs | **2.59e-06** |
| asymmetry | 7.5e-07 |

All at float32 noise. Newton steps are accepted undamped (λ at its floor, zero backtracks).
Reusing the band for 10 steps (`LARND_DEDX_GN_REFRESH=10`) still gives clean full Newton steps —
the curvature is dominated by prior and geometry terms that barely move.

### 7.3 The verdict: a tie, in both conditioning regimes

dE/dx block only, calibration pinned at truth, 24 segments, 3 seeds, **equal wall-clock**:

| regime | GN, 200 it (~8.8 min) | Adam, 610 it (~8.2 min) | GN − Adam |
|---|---|---|---|
| `mean_w = 3000` (κ ≈ 750) | 940.7 ± 10.5 | 942.1 ± 5.8 | −1.43 ± 6.93 → **tie** |
| `mean_w = 1e5` (κ ≈ 23874) | 939.3 ± 2.7 | 941.4 ± 6.0 | −2.13 ± 3.77 → **tie** |

Per-iteration GN clearly wins (949.9 → 940.7 in the same 200 steps) but costs **3.0×/iter**
(7.72 s/it at refresh=1, 3.04 at refresh=10, vs Adam's 1.00), which consumes exactly the gain.

**A prediction of mine that failed.** I argued GN would win where conditioning is worst, since
Newton is affine-invariant while first-order methods suffer O(κ). **Adam is fine at κ = 23874** —
it ties GN there too. Its per-parameter second-moment normalisation evidently absorbs this
conditioning, and the single stiff direction is satisfied almost immediately. *Do not use the κ
argument to motivate GN on this block; it does not survive measurement.*

---

## 8. Why the dE/dx MAE degrades — solved

A long-standing puzzle: "dE/dx MAE ends ~30 % worse than it starts." This section resolves it,
and the answer is neither overfitting nor the optimiser.

### 8.1 Two artifacts in my own first analysis

Recorded because both are easy to repeat:

1. **`dedx_mae_iter` is per-BATCH, not global** — it records the MAE of whichever batch that
   iteration touched, so with NB=2 it interleaves two batches with very different behaviour.
2. **It is NaN-padded**, and filtering the NaNs *before* indexing by position shifts every index.
   My first pass did exactly that and produced a spurious "+62 % degradation" curve.

![dE/dx MAE anatomy](plots/fig_dedx_mae_anatomy.png)

### 8.2 What actually happens

Split by batch, with the raw array — identical across all 3 seeds:

| | it 0–40 | it 40–80 | it 180–220 | it 400–440 | last 40 |
|---|---|---|---|---|---|
| **batch 0** | 0.0534 | 0.0453 | 0.0338 | 0.0321 | **0.0317** (improving) |
| **batch 1** | 0.0597 | 0.0809 | 0.1236 | 0.1520 | **0.1626** (degrading 2.7×) |

The degradation is entirely in one batch. And within that batch it is **one segment**:

| batch | N | MAE | MAE excluding worst | shortest segment | fitted vs true |
|---|---|---|---|---|---|
| 0 | 11 | 0.0315 | 0.0261 | 0.687 cm | −0 % |
| 1 | 13 | **0.1634** | **0.0354** | **0.0355 cm** | **−82 %** (0.367 vs 2.066) |

**Remove that single 0.0355 cm segment and batch 1 behaves exactly like batch 0** (0.0354 vs
0.0315). The worst 3 of 24 segments carry 75 % of the total absolute error, and
`corr(log segment length, |error|) = −0.75`.

### 8.3 The mechanism: ultra-short segments carry no information

Both the likelihood and the student-t prior are **dx-weighted** (`weights=t[:, dx_idx]`), so a
segment's curvature is roughly proportional to its length. Measured on the batch-1 Hessian:

> shortest segment (0.036 cm) diagonal curvature **3.67** vs median **3826** — **1044× less**.

A 0.0355 cm segment is 0.08 % of a 43 cm track. Its dE/dx is **not identifiable from the data, and
not pinned by the prior either** — so it is a nearly-free parameter that drifts wherever the
residual noise pushes it. More optimisation ⇒ more drift ⇒ rising MAE. That is the whole effect.

The mean-constraint cancellation of §5.3 (`qᵢ² + c·qᵢ < 0`) makes such segments *worse*, but is
not the root cause: the runaway is essentially as bad at `mean_w = 3000` (−82 %) as at `1e5`
(−74 %), so **lowering the weight does not fix it.**

### 8.4 How common is this at production scale

Over the whole input file (3.03 M segments):

| threshold | segments below | share of segments | share of total track length |
|---|---|---|---|
| dx < 0.05 cm | 40 014 | 1.32 % | **0.010 %** |
| dx < 0.10 cm | 80 356 | 2.65 % | **0.041 %** |
| dx < 0.20 cm | 165 655 | 5.47 % | 0.170 % |
| dx < 0.50 cm | 418 914 | 13.82 % | 1.054 % |

**2.65 % of segments carry 0.041 % of the track length.** Each currently gets its own free fitted
parameter, contributing essentially nothing to the likelihood while being free to run away and
dominate any per-segment error metric.

### 8.5 What to do about it

1. **Do not fit dE/dx for segments below a minimum length** (~0.1–0.2 cm) — hold them at the prior
   centre, or merge them into a neighbour. This removes 2.7–5.5 % of the parameters at a cost of
   0.04–0.17 % of the track length. Cheapest and most direct fix.
2. **Floor the per-segment curvature** in any structured Newton solve — required anyway, since
   Sherman–Morrison needs the band+diagonal block invertible (§5.3).
3. **Report a length-weighted MAE** alongside the unweighted one. The unweighted metric is
   dominated by segments that carry no physics: GN has the *best* length-weighted MAE (0.0364) and
   the *worst* unweighted one (0.1178) in the same runs.
4. **Early stopping is not the answer**, despite appearances. Batch 0 improves monotonically; only
   the runaway segment argues for stopping early. Fix the parameterisation instead.

---

## 9. The (scale, shape) reparametrisation — the structural fix, built and measured

§6 showed the mean constraint's weight is wrong; §5.3 showed *why* it wrecks the conditioning.
Both are consequences of the mean being an **implicit** function of ~4000 parameters. Making it
explicit removes the cause rather than the symptom.

### 9.1 The parametrisation

$$\mathrm{dedx}_i = M\,s_i,\qquad M=e^{\mu},\qquad \sum_i l_i s_i / L = 1
\;\Rightarrow\; \log \mathrm{dedx}_i = \mu + a_i - \log Z,\quad Z=\tfrac1L\sum_j l_j e^{a_j}$$

so **`mean_dedx = exp(μ)` exactly, by construction** (verified end-to-end: `exp(μ)` =
1.866057 / 1.883146 matched the measured length-weighted mean to every digit).

**The loss is untouched.** It still receives `log_dedx`; only the parametrisation and the gradient
chain rule change — `∂L/∂μ = Σᵢgᵢ`, `∂L/∂a_k = g_k − p_k Σᵢgᵢ` with
`p_k = l_k e^{a_k}/Σ_j l_j e^{a_j}` — which keeps the blast radius small. The checkpoint mirrors
the derived `log_dedx`, so every existing consumer works unchanged. The gauge `a → a + c` is
exactly flat and receives exactly zero gradient (`Σ_k ∂L/∂a_k = 0` identically).

**The payoff, verified against autodiff:** the mean-constraint Hessian in the new coordinates has

> **max|H_aa| = 2.09e-14** against **H_μμ = 2.09e+04**

The dense rank-1 `qqᵀ` coupling across all N segments — the thing that sets κ ≈ 2.4e4 and cancels
short-segment curvature (§5.3) — **collapses to a single scalar entry.**

### 9.2 Result: it works, but only together with the weight fix

Calibration initialised at truth, 3 seeds, 610 iterations:

| parametrisation | `mean_w` | mean abs calib err | lifetime |
|---|---|---|---|
| per-segment | 1e5 | 9.74 % | +15.02 % |
| scale_shape (μ_lr 1.0) | 1e5 | 14.91 % | +25.83 % |
| scale_shape (μ_lr 0.1) | 1e5 | 11.01 % | +16.17 % |
| per-segment | 3000 | 11.90 % | +18.08 % |
| **scale_shape** | **3000** | **9.63 %** | **−2.43 %** |

**At `w = 3000` it is a clear win** — calibration −2.26 ± 0.51, and **lifetime +18.08 % → −2.43 %**
(shift −20.5 ± 2.5). Essentially unbiased lifetime, the best of any configuration in this report.

**At `w = 1e5` it is worse.** μ becomes a single heavily-driven coordinate and Adam, being
scale-invariant per coordinate, steps it at full learning rate however large its gradient is, so
it oscillates. Damping μ's step recovers much of it (14.9 → 11.0, +25.8 → +16.2), confirming the
diagnosis, but it still loses to the per-segment form there.

> **The two changes are complementary, not independent: use `w ≈ 3000` AND `scale_shape` together.**
> Neither alone reaches the result.

### 9.3 The shape-only prior — built, and it does NOT help

The reparametrisation does not by itself remove the residual ~+1.5 % mean bias, because the
student-t still acts on `dedxᵢ = M·sᵢ`, so mode-shrinkage still reaches μ. The fix attempted was to
make the prior say nothing about the scale — evaluate it at scale-normalised dE/dx:

$$\widetilde{\mathrm{dedx}}_i = \mathrm{dedx}_i \cdot \frac{M_{\rm ref}}{\mathrm{mean\_dedx}}$$

**The mechanism works exactly as designed**: the prior's value varies by 1.8e-14 over a ±10 %
scale sweep, versus 25.4 for the current form — that variation *is* the gradient that biases the
mean, and it is now identically zero. (The soft barrier is deliberately not normalised: it is a
delta-ray cap on physical dE/dx, not a statement about shape.)

**It nonetheless makes the fit worse.** `scale_shape`, calibration at truth, 3 seeds:

| `mean_w` | prior | mean bias | len-w MAE | mean abs calib err | lifetime | exp(μ) |
|---|---|---|---|---|---|---|
| 3000 | standard | +1.538 % | 0.0364 | **9.63 %** | **−2.40 %** | 1.8823 |
| 3000 | shape-only | +1.835 % | 0.0413 | 20.21 % | +54.59 % | 1.8880 |
| 464 | standard | +1.335 % | 0.0349 | 16.48 % | +44.09 % | 1.8787 |
| 464 | shape-only | +1.849 % | 0.0414 | 15.40 % | +19.84 % | 1.8882 |

**Why:** removing the prior's pull on the scale leaves the mean constrained only by the mean
constraint and the likelihood. You can watch it happen — `exp(μ)` moves from 1.8823 to 1.8880,
driven onto the target 1.887 — while this batch's own true mean sits ~1.8 % below. The prior had
been **shielding the fit from a mis-specified per-batch target**, and that shield was load-bearing.

A follow-up predicted the shape-only prior would win once the scale prior was correctly specified
(w = 464, the σ-derived value for this batch size). **That prediction failed**: at w = 464 the two
priors merely tie (−1.08 ± 3.08 on calibration), and w = 464 is itself much worse than w = 3000.
That failure is what prompted the correction now recorded in §6.3.

**Verdict: `LARND_DEDX_SHAPE_PRIOR` stays default-off.** The idea is sound in isolation and the
implementation is verified, but it removes a shield that is doing more good than the bias it
fixes. It would only become useful alongside a per-batch-correct scale prior, which we do not have.

### 9.4 What this still does not fix

- **The short-segment runaway** (worst segment ~−84 %). That is dx-weighting making tiny segments
  information-free (§8), not a coupling problem. It needs the minimum-length cut.
- **The per-batch over-constraint.** That is the weight, not the parametrisation.

### 9.5 Defects found in this work

- `LARND_DEDX_MU_LR_MULT` initially scaled μ's **gradient** — a no-op under Adam, which is
  scale-invariant per coordinate. It now scales the **update**. (The grid above ran at 1.0, where
  the operation is a no-op either way, so those results are unaffected.)
- The output tag omitted the μ-LR value, so the μ_lr=0.1 runs **overwrote** the μ_lr=1.0 pickles —
  exactly the `out_label` collision that `analyze_gn_toy.py` exists to flag. Tag fixed; the
  μ_lr=1.0 row above survives only from the analysis printed before the overwrite.

---

## 10. The minimum-length cut — the MAE puzzle, fixed

![dE/dx MAE anatomy and the cut](plots/fig_dedx_mae_anatomy.png)

§8 diagnosed the MAE degradation as a handful of ultra-short segments that are not identifiable:
both the likelihood and the student-t prior are dx-weighted, so a 0.0355 cm segment was measured
to have **1044× less curvature than the median** and drifted to −84 % of truth.

`LARND_DEDX_MIN_DX` (default 0.0 = off) acts on that: segments shorter than the threshold have
their gradient zeroed, so they stop chasing noise. The launcher logs how many segments are frozen
and what fraction of track length they carry, so the cost is explicit rather than silent.

**Result** (scale_shape, `w = 3000`, 3 seeds, cut at 0.15 cm):

| min dx | MAE | len-w MAE | mean bias | mean abs calib err | lifetime | worst segment |
|---|---|---|---|---|---|---|
| OFF | 0.1035 | 0.0364 | +1.538 % | 9.63 % | −2.43 % | **−84 %** |
| **0.15 cm** | **0.0397** | **0.0352** | +1.549 % | **8.67 %** | −1.87 % | **−11 %** |

- **MAE −62 %** (−0.0637 ± 0.0001) and the runaway is gone: the 0.0355 cm segment goes from
  0.330 to 1.849 against a true 2.066.
- **Calibration also improves** (−0.97 ± 0.11), so this is not merely cosmetic relabelling of a
  metric — removing unidentifiable parameters helps the fit.
- Length-weighted MAE improves slightly; the mean bias is unchanged, as expected (that is the
  weight, §6).

**Implementation note.** In `scale_shape` mode a cut segment still follows the global scale `M`;
only its *own* gradient is removed. That is the desired behaviour — a short segment's dE/dx should
track the overall scale — but it does mean "frozen at the prior centre" is the wrong description,
and a check written on that assumption will fail for the right reason.

---

## 11. Production-scale validation — the toy ranking does NOT reproduce

Everything in §6 and §9 was measured on toys with **true geometry and calibration started at
truth**. This section tests those recommendations in the real configuration
(`start_sci_case.sh`, `SCIMODE=full`: fitted chain positions from an imperfect linear guess),
as a 2×2 factorial, 3 seeds, 2500 iterations, with the §10 minimum-length cut enabled throughout.

![Production-scale validation](plots/fig_production_scan.png)

### 11.1 Result

Robust estimate = trailing-20 % median of each parameter trajectory.

| param | `mean_w` | Ab | eField | tran_diff | long_diff | lifetime | mean \|err\| |
|---|---|---|---|---|---|---|---|
| perseg | 1e5 | −2.20 | −4.36 | −16.60 | +38.06 | +6.12 | 20.9 ± 6.6 |
| perseg | 3000 | −1.94 | −4.37 | −9.52 | +40.67 | +13.24 | 24.3 ± 5.2 |
| scale_shape | 1e5 | −1.80 | −4.35 | −16.61 | +36.49 | +2.36 | 20.6 ± 6.9 |
| scale_shape | 3000 | −2.45 | −4.37 | −11.28 | +27.80 | +38.75 | 26.5 ± 2.7 |

**Lowering the weight does not help here.** In the toy it moved mean calibration error
25.8 → 6.2 (**−19.6**); in production it moves 20.9 → 24.3 (**+3.4**). The headline result of §6
and §9 does not transfer.

### 11.2 But this is UNDERPOWERED, not a clean negative

Per-seed lifetime error:

| config | seed 0 | seed 1 | seed 2 | sd |
|---|---|---|---|---|
| perseg, 1e5 | +33.1 | −45.2 | +30.5 | 36.3 |
| perseg, 3000 | +31.2 | −43.7 | +52.2 | 41.2 |
| scale_shape, 1e5 | +30.8 | −50.7 | +26.9 | 37.5 |
| scale_shape, 3000 | +72.7 | −48.2 | +91.8 | 62.0 |

**Every configuration has one seed at −45…−51 % and two at +27…+92 %.** The seed scatter
(sd 36–62 %) is larger than every between-configuration difference; with n = 3 the standard error
on each mean is ~21–36 %. **This experiment cannot resolve the configurations in either
direction.** The correct statement is *not* "the change is useless" but "at production scale,
with 3 seeds, it is not measurable."

### 11.3 What is different, and what it implies

The toys isolated the dE/dx block with **true geometry**. Production fits chain positions from an
imperfect guess, and that appears to dominate: *every* configuration shows `tran_diff` −10…−17 %,
`long_diff` +28…+41 %, `eField` −4.4 % — errors far larger than anything in the toys, and
essentially unchanged across the four cells.

That is consistent with §12: **geometry is the hard block.** The plausible reading is that the
dE/dx work is polishing something that is not currently the limiting factor in the real fit.

### 11.4 Status of the recommendations after this

| result | status |
|---|---|
| Hessian decomposition, 5-HVP probing, rank-1 derivation | **unaffected** — measured on curvature, not on fit quality |
| mean-constraint target 1.887 is correct (+0.09 %) | **unaffected** — measured on truth data |
| MAE puzzle: one information-free segment | **unaffected** — measured on segment lengths and curvature |
| minimum-length cut (§10) | measured on toys; mechanism is direct, but **inherits the transfer caveat** |
| **`w = 1e5 → 3000` (§6)** | **NOT validated at production scale** |
| **(scale, shape) reparametrisation (§9)** | **NOT validated at production scale** |

**To settle the weight question** would need either many more seeds (≳10 to beat that scatter) or
a lower-variance configuration — most usefully `SCIMODE=ceiling` (true geometry, frozen), which
removes the dominant noise source and asks the question the toys were actually answering.

---

## 12. The geometry block: closed

Two independent lines of evidence say don't pursue this.

**Adam simply wins.** Clean per-track comparison, 50 cm spline position-only: Adam mean 51 µm /
median 41 / p90 82 (71 tracks, converged) vs one-shot GN mean 107 µm / median 58 / p90 205 (33
tracks, 1 epoch). The reason is instructive — **a per-batch GN solve fits that batch's stochastic
noise realisation**, while Adam refines each track over ~150 epochs with different sim seeds and
averages the noise out. *Second-order cannot beat stochastic averaging on a noisy
differentiable-sim objective.*

**And the engineering is prohibitive.** Identical data (9 batches, ~605 cm): Adam ~1.2 s/iter
(~50 min total) vs GGN 1 iteration in 7:51, projected **~13 hours** — a ~400× per-iteration gap,
dominated by compilation, and very sensitive to padded shape. Earlier attempts hit
`xla.gpu.CompilationResultProto exceeded maximum protobuf size of 2GB` and 445 s/iteration.

Two further facts worth recording:

- **The new LM machinery does not reach the position path.** `LARND_GN_DAMP_MODE / BOUND_TR /
  ACTIVE_SET / STEP_CAP / STALL_PATIENCE` live entirely in `GaussNewtonCalibFitter`, used only by
  `_run_gn_loop`/`_bounded_step`, which operate solely on the 5 calibration scalars.
  Position-GGN (`_solve_chain_gn/_ggn/_ggn_shared`) sits in the parent `GradientDescentFitter` and
  implements its own minimal inline LM loop — **zero bound handling, zero active set, zero abseig,
  zero stall patience.** Setting those env vars with `--geom_optimizer ggn` has **no effect**.
  To benefit, they must be *ported*, not enabled.
- **The dE/dx structure trick almost certainly does not transfer** (reasoned, not measured). dE/dx
  is banded because one scalar per segment matches physical locality. The geometry block
  deliberately uses a **global, non-local** basis — sine modes each move the whole track — chosen
  because a local per-node basis was ill-conditioned. So `dPos/dParam` is dense even where data
  coupling is banded, corroborated by the exact geometry Hessian OOM'ing at 224 GB. A local basis
  might recover structure but reintroduces the conditioning problem the spline basis was adopted
  to solve. **Real tension, not a quick win.**

---

## 13. Measurement traps

Every one of these cost a wrong answer at least once. They are the most transferable part of this
report.

1. **`peak_mem` silently reads 0.00 GiB.** `XLA_PYTHON_CLIENT_ALLOCATOR=platform` makes jax
   `memory_stats()` return nothing. Do **not** "fix" this by switching to `bfc` — that OOM'd all 8
   arms even at LEN=50/NB=2; the platform allocator is what makes these runs fit in 11 GiB. Use an
   allocator-independent nvidia-smi poller.
2. **Compile contamination.** `[GN-PROF]` lines are emitted per sweep and **each batch compiles
   separately**: a 4-batch run logged 49.8 / 14.8 / 13.9 / 25.5 s against a ~6.3 s steady state.
   Correct extraction: filter `with_hess=True`, take the **min per batch index across sweeps**,
   then the median. Reading the first sweep gives the compiler's timings, not the optimiser's —
   this produced an incorrect "no time gain from larger batches" conclusion.
3. **`--max_batch_len` is a cap, not a target**, and requested `--max_nbatch` is not achieved.
   Always read the actual `number of simulation batches is N` from the log. Three separate
   variants of this bug have now been found, including `example_run.py` silently clamping
   `max_nbatch = min(max_nbatch, iterations)`.
4. **A subset fit misled an extrapolation.** Six points gave 98 MiB/cm → ~40 GiB at 400 cm; all
   ten give 70 MiB/cm → ~31 GiB. **Bootstrap before extrapolating 5× beyond the data.**
5. **The toy's `rel` init mode clips.** `Ab ∈ [0.75, 0.9]` around truth 0.83488, so nominal offsets
   of 20 % and 50 % **both** clip `Ab` to exactly 0.9 and `eField` to 0.6 — those two arms share a
   starting point and begin with two parameters already at a bound. Any conclusion drawn from
   those arms is an artifact of the initialisation, not of curvature. Use `INITMODE=frac2bound`.
6. **A difference of two large float32 numbers cannot resolve a small structure.** Extracting
   `H_data = H_total − H_prior` at production weights loses ~1e-3 absolute precision because the
   rank-1 term makes both ~1e4 — above the entire real band tail, and enough to manufacture
   single-ULP entries that look like long-range coupling. **Test candidate structure against
   transpose symmetry**: a real Hessian entry appears in both `(i,j)` and `(j,i)`; float32
   rounding does not. Two "couplings" at separations 5 and 9 were killed this way.
7. **A single unreplicated arm is not a measurement.** The first dE/dx campaign compared arms
   differing by ~0.26 % without ever measuring run-to-run scatter — which turned out to be the
   same size. Always run replicate seeds. (`--seed` *does* vary the realisation here, verified
   byte-level; `--data_seed` and `SEEDSTRAT` do **not**.)
8. **Use the statistic the objective actually uses.** The mean constraint acts on the
   *length-weighted* mean (`mean_dedx_iter`); computing the unweighted one gave a result with the
   **wrong sign**. Where a weight is needed to reproduce a term offline, save it — `seg_dx_cache`
   was added to the checkpoint for exactly this.
9. **One runaway point destroys a variance.** A single collapsed segment inflated an apparent
   "+165 % over-dispersion" that did not exist. Use robust or trimmed measures and report outliers
   explicitly.
10. **Small-sample means are not systematics.** A 24-segment batch gave a length-weighted mean
   1.8 % from the global value, which briefly looked like a biased target; it was a fluctuation.
   Check the standard error before calling anything a bias.
11. **`AssocGrpNodeLimit` is per-account, not global.** Jobs pending on `turing` were blocked by
   *ampere* usage on the same association, while `turing` itself was idle. Four accounts here have
   turing + `normal` QOS; switching account started jobs immediately. Also check the memory request
   — 64 G was requested against a 2.7 GB peak, throttling concurrency for nothing.
12. **A per-batch diagnostic is not a global one, and NaN-filtering breaks index alignment.**
   `dedx_mae_iter` records only the batch touched that iteration, and is NaN-padded. Filtering the
   NaNs *then* indexing by position shifts every index — that alone manufactured a "+62 %
   degradation" curve that does not exist. Split by batch, keep the raw array (§8.1).
13. **A mean over segments is not a mean over the detector.** 3 of 24 segments carried 75 % of the
   dE/dx absolute error, and the worst was 0.0355 cm long. Any unweighted per-segment metric is
   dominated by objects carrying no physics; report the length-weighted version alongside.
14. **Check whether a parameter is identifiable before blaming the optimiser.** The whole MAE
   puzzle was a segment with 1044× less curvature than the median — a free parameter, not a hard
   optimisation problem.
15. **Piping a job's output through `tail -N` discards its startup diagnostics.** The
   `[ENV-INSIDE]` and `[DEDX-MINDX]` lines were emitted and then thrown away, so "the feature
   didn't fire" and "the log was truncated" looked identical. Verify a feature fired from its
   *effect on the output data*, not from the absence of a log line.
16. **A verification test must match the mode being tested.** Checking that cut segments sit at
   `log(loc)` fails in `scale_shape` mode — they still follow the global scale by design. The test
   was wrong, not the code.
17. **`--max_nbatch` is not what you get, in PRODUCTION too.** `SCILEN=50, SCINBATCH=100` yields
   **13 batches / 931.9 cm**, because dataio loads `len × (nbatch+2)` cm then drops trajectories
   longer than `max_batch_len`. Worth auditing independently of anything in this report.
18. **A toy that fixes the hard block will overstate an easy-block improvement.** Every dE/dx
   result here was measured with true geometry; in production, fitted positions dominate and the
   same changes are unmeasurable. Validate on the configuration you intend to run, and check the
   seed scatter BEFORE interpreting a mean of 3.
19. **The output label must encode every knob you vary — checked three times, missed three times.**
   `start_sci_case.sh` built its label from mode/nbatch/len/seed only, so a 4-configuration scan
   silently overwrote itself and left 3 files that looked exactly like a clean seed set. ~17
   A100-hours lost. The label now appends any non-default knob.
20. **Truth-point curvature does not extrapolate.** Every spectrum before this work was measured
   *at truth* (eigenvalues 3.3e3–1.3e10). At a 1 % offset — where the optimiser actually operates —
   the Hessian is **indefinite** (eigenvalue −2.51) with eigenvalues ≤ 2e5, and the LM step points
   nearly *opposite* the GGN step (cos = −0.449). Qualitatively different territory.

---

## 14. How to move forward

Ordered by (expected gain) / (effort × risk). The first item is a one-line change with a measured
4× effect; everything below it is real work.

### Tier 1 — do now, cheap and measured

**1. ~~Lower `dedx_mean_constraint_weight`~~ — production validation FAILED to reproduce (§11).**
The 2×2 factorial on the real config shows no gain (mean calibration error 20.9 → 24.3 on lowering
`w`), and the seed scatter makes it unresolvable anyway. **Do not change the production default on
the strength of the toy results.** To settle it: rerun as `SCIMODE=ceiling` (true geometry), which
removes the dominant noise source, and/or use ≳10 seeds. The *diagnosis* — that 1e5 is far stiffer
than the batch spread supports — is unaffected; only its benefit is unproven.
One number in `start_sci_case.sh`. Measured effects: calibration error 25.8 % → 6.2 %, lifetime
bias +108 % → +16 %, dE/dx mean bias +0.21 % → +0.03 %, κ ~23874 → ~750.
- *Derive the value for your actual batch length* from §6.3 (`w = 1/(2σ²)`), don't copy 3000.
- **Validation before it becomes the default:** one production-scale ampere run at 400 cm, 3 seeds,
  comparing `w ∈ {1e5, 3000}`. Success = lifetime and long_diff improve or hold, and the fitted
  length-weighted mean stays within ~0.1 % of 1.887. Failure = calibration degrades, which would
  mean the toy's ranking doesn't survive scale and the reparametrisation (item 5) is the only route.
- **This may be upstream of several standing puzzles.** The lifetime pathology has been attacked
  repeatedly — freeze-dEdx-at-iters/2, the drift-profile penalty, the overshoot-then-oscillate
  co-adaptation. If an over-stiff mean constraint is injecting a large lifetime bias, those may all
  have been treating a symptom. **Worth re-running the freeze and drift-profile ablations at
  `w = 3000` to see whether they are still needed at all.**

**2. ~~Cut ultra-short segments~~ — BUILT and it works (§10).** `LARND_DEDX_MIN_DX=0.15`,
default off. Measured: **MAE −62 %**, worst-segment error −84 % → −11 %, and calibration also
improves (−0.97 ± 0.11). Enable it. Original rationale (§8): The degradation is one 0.0355 cm segment with 1044× less curvature than the median
running away; excluding it, MAE is flat or improving. Removes 2.7–5.5 % of the parameters at a
cost of 0.04–0.17 % of the track length. **Independent of the weight change** — the runaway is as
bad at `w = 3000` as at `1e5`. Also report a length-weighted MAE, since the unweighted metric is
dominated by segments carrying no physics.

**3. Guard the short-segment runaway.** The `qᵢ² + c·qᵢ < 0` cancellation (§5.3) is real and fires
in every single fit — one segment per batch collapsing to ~0.3 MeV/cm. Lowering `w` shrinks `c`
and should mostly cure it, but a floor on the per-segment curvature (or a minimum-length cut on
fitted segments) is cheap insurance and is *required* before any structured Newton solve, which
needs the band+diagonal block invertible.

### Tier 2 — the real engineering

**4. ~~Implement the dE/dx structured Newton step~~ — DONE, and it does not pay off (§7).**
Built, validated (band recovery 2.6e-06 against the dense Hessian), and it **ties Adam at equal
wall-clock in both conditioning regimes**. The 5-HVP probing trick is real and reusable, but the
block is objective-limited so a better optimiser buys nothing. Left in the tree, default-off.
**Do not spend more effort here.**

**5. ~~Reparametrise dE/dx as (scale, shape)~~ — BUILT; works on toys, NOT reproduced at
production scale (§11).** `LARND_DEDX_PARAM=scale_shape`,
default off. Verified to collapse the rank-1 coupling (`max|H_aa| = 2e-14` vs `H_μμ = 2e+04`) and,
**paired with `w = 3000`**, to take lifetime from +18.1 % to **−2.4 %**. *Must* be paired with the
weight change — it is worse at `w = 1e5`. Remaining work is the shape-only prior (§9.3) and
production-scale validation. Original rationale, still accurate:
- `M` carries the mean explicitly and takes its prior directly — *one parameter*, not a rank-1
  coupling across 4000. Set its stiffness to anything you like at zero conditioning cost.
- The shape `sᵢ` takes the student-t, and mode-shrinkage on the shape **can no longer bias the
  mean**, because the mean is not a function of the shape parameters. This dissolves the original
  problem rather than patching it.
- The Hessian becomes banded-in-`s` plus one row/column — no dense rank-1, no Sherman–Morrison, no
  zero-pivot hazard. Item 4 becomes trivial.
- Cost: a genuine refactor (normalisation constraint, checkpoint format, restart compatibility).

**6. Port the LM improvements into the chain solvers.** `LARND_GN_DAMP_MODE / BOUND_TR /
ACTIVE_SET / STEP_CAP / STALL_PATIENCE` currently have **no effect** with `--geom_optimizer ggn`
(§7). Contained change, independent of whether GGN or Adam wins on speed.

### Tier 3 — open physics, higher value than any optimiser question

**7. Test the target-noise/lifetime-bias hypothesis (§4).** Real data always has noise in the
target, so the +14 % row is the physically relevant one, and lifetime may be biased **up** unless
electronics noise enters the expected-intensity term of the loss. Currently measured at one point,
from a single Newton step. If real, this is a physics bias in every production fit.
- Note this now has a **second candidate mechanism competing for the same symptom** — §6.4 shows
  the mean constraint biasing lifetime by +108 %. Item 1 should be done first so the noise study
  isn't measuring the constraint's artifact.

**7b. ~~Shape-only prior~~ — built, tested, does NOT help (§9.3).** Mechanism verified
(scale-invariant to 1.8e-14) but it removes a shield against a mis-specified per-batch target and
makes calibration and lifetime substantially worse. Default-off. Revisit only if a per-batch-correct
scale prior ever exists.

**8. Attack the OBJECTIVE, not the optimiser.** §4 and §7–8 both show the binding constraint is
where the objective's minimum sits, not how fast you reach it. Concretely: the target-noise
question (item 7), the mean-constraint weight (item 1), the short-segment parameterisation
(item 2), and under-dispersion below. Optimiser work on either block is now well into diminishing
returns.

**9. Address MAP under-dispersion, if the per-segment distribution matters.** Any MAP estimate with
a meaningful prior compresses the fitted profile toward the prior centre. Fixing the *mean* (items
1/5) does not fix the *shape*. If the per-segment dE/dx distribution needs to be right and not just
its mean, MAP is the wrong estimator and a hierarchical or marginalised treatment is required —
a much larger project, but the honest answer to "why is the profile over-concentrated".

### Explicitly not worth doing

- **More dE/dx optimiser work.** GN is built, correct, and ties Adam (§7).
- **More position-GGN benchmarking.** Compile cost is the binding constraint, not the linear
  algebra, and Adam already wins on accuracy (§7).
- ~~Retuning `w` by trial and error.~~ **Reversed:** it is NOT derivable from the batch spread
  (§6.3 correction) — an empirical scan at the production configuration is the only way to set it.
- **"Fixing" the 1.887 target.** It is correct to 0.09 % (§6.2).
- **Shifting `dedx_student_loc`.** Measured slightly harmful (§6.5).

---

## Provenance

- **Toy testbed:** `optimize/scripts/start_gn_toy.sh`, `start_adam_toy.sh` — pure 5-param
  calibration, truth positions, truth dE/dx, no nuisances, 1–10 min/arm on `turing`.
- **dE/dx Hessian:** `optimize/scripts/start_dedx_hessian.sh`; raw matrices in
  `fit_result/dedx_hessian_b1_b1.pkl` (batch 1, full 13×13, production weights),
  `dedx_hessian_b0_meanw0.pkl` (batch 0, full 11×11, mean weight 0 — the control),
  `dedx_hessian_b0_c1.pkl` (batch 0, 6 of 11 columns).
- **Figures 3–5** regenerated from those pickles by `plot_dedx_hessian.py` (repo root).
- **dE/dx GN implementation:** `_dedx_gn_step` + `_dedx_probe_matrix` / `_dedx_band_from_probes`
  / `_dedx_banded_sm_solve` in `optimize/fit_params.py`; launcher
  `optimize/scripts/start_dedx_gn.sh`; unit tests `test_dedx_gn_numerics.py` (20 checks, CPU).
  Env-gated, default off. `wait_slurm.sh` (repo root) is a SLURM waiter that does not mistake a
  failed query for job completion — the naive `until [ -z "$(squeue …)" ]` does.
- **dE/dx mean-constraint campaign:** `optimize/scripts/start_dedx_bias_toy.sh` (54 arms total
  across three iterations of the design), analysed by `analyze_dedx_bias_toy.py`; figure by
  `plot_dedx_mean_constraint.py`. Truth-data measurements read
  `true_through_muon_edep_10cm_vol1cm.h5` directly — no fitting, no GPU.
- **Code added, all additive and default-off:** `LARND_DEDX_LOG_JACOBIAN` (change-of-variables
  term for the dE/dx prior) and `seg_dx_cache` (per-segment length in the checkpoint, required to
  reconstruct the length-weighted mean offline).
- **Analysis tool:** `analyze_gn_toy.py` — parses logs, cross-checks pickles, reconstructs the
  *accepted* trajectory (a reverting run's last `GN iter` line is **not** its final state), and
  flags `out_label` collisions where a resubmission silently overwrote an earlier pickle.
- All arithmetic is float32; `jax_enable_x64` appears nowhere in the tree. Condition numbers near
  1/ε_float32 ≈ 8.4e6 (as in the at-truth calibration Hessian, κ ≈ 4e6) should be read with that
  in mind.

**Code status at time of writing:** the GN work is **uncommitted** — `optimize/fit_params.py`
carries ~+625/−21 lines in the working tree, and the toy scripts are untracked.
