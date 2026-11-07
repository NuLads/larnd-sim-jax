# Second-order methods in larnd-sim-jax — figure-by-figure summary

*Condensed companion to `GN_REPORT.md`. Same evidence, same figures, discussion compressed to
what each test asks and what it settles. July–August 2026.*

---

## Verdict

| block | verdict | status |
|---|---|---|
| **5 calibration scalars** | GN converges where Adam stalls; Adam still faster near the optimum | recipe validated |
| **~4000 per-segment dE/dx** | GN works, **ties Adam** at equal wall-clock — block is objective-limited | built, not worth deploying |
| **per-track geometry** | Adam beats GN on accuracy *and* speed | closed |

**The one thing to carry away:** two of the three blocks are **objective-limited, not
optimiser-limited** — a better optimiser reaches a lower loss and a *worse* answer. That caps what
any optimiser work can buy here.

### How much to trust each result

| evidence type | examples | status |
|---|---|---|
| measured on **curvature or truth data** | Hessian structure, 5-HVP probing, target = 1.8853, MAE diagnosis | **robust** |
| measured on **toy fit quality** (true geometry) | weight change, reparametrisation | **did not reproduce at production scale (§9)** |

---

## 1. Does GN beat Adam on the calibration block?

![Adam vs GN convergence](plots/fig1_adam_vs_gn.png)

**Tests:** identical single 48.5 cm batch, so one Adam step and one GN gradient see the same data
and the same loss. Two starting offsets from truth.

**Concludes:**
- GN reaches threshold in **5 iterations vs Adam's 40** — but 58 s vs **28 s**. Adam wins
  wall-clock **2.1×** near the optimum.
- From a 5 % offset **Adam never reaches threshold in 400 iterations**; GN converges from both
  starts to the same floor.
- **GN's value is robustness, not speed.** It earns its place only where Adam stops progressing.
- The failure that had blocked GN was not a bad direction: the oversized step *lowered* the loss,
  so LM accepted it and a saturation guard reverted the whole run. The sigmoid box was an
  unmodelled constraint.
- Exact curvature **beats** GGN (contradicting an earlier consensus); GGN never converged from 5 %.

## 2. What does GN cost, and where does it fit?

![GN cost and scaling](plots/fig2_gn_scaling.png)

**Tests:** peak GPU memory and time against batch count and batch size.

**Concludes:**
- **Batch count is free in memory** (6487→6489 MiB for 1→8 batches): batches are walked
  sequentially, so peak is the largest single batch. Scale data with batch *count*.
- **Batch size is the binding constraint:** `peak_MiB = 3471 + 70.1 × cm/batch`.
- Memory overhead over Adam is only **1.26×**, because the Hessian is built from sequential HVPs
  rather than `jacfwd(jacrev)` (which would cost ~6×).
- Throughput saturates early — 2.06× of a 2.3× gain is already had by ~43 cm/batch.

## 3. Does the dE/dx Hessian have structure worth exploiting?

![dE/dx Hessian decomposition](plots/fig_dedx_hessian_decomposition.png)

**Tests:** the full per-segment dE/dx Hessian, decomposed into its three physical contributions.

**Concludes:**
- `H = H_data(banded) + diag(student-t + barrier) + w·qqᵀ`. The dense-looking total is three
  exactly-structured pieces.
- The prior is **exactly diagonal** (off-diagonals identically 0.0 with the mean constraint off).
- The mean constraint is the **sole** source of dense coupling.
- ⇒ an **O(N) solve exists**: banded LU + one Sherman–Morrison correction.

## 4. How good is that structure, quantitatively?

![dE/dx Hessian structure](plots/fig_dedx_hessian_structure.png)

**Tests:** decay of the likelihood term with segment separation; whether the coupling really is
rank-1.

**Concludes:**
- Strong band to **|i−j| ≤ 2**, then a *sparse but real* tail at separation 3–4 (~1e-7 relative,
  verified by transpose symmetry), then exact zeros. **Not tridiagonal, and not "exactly zero
  beyond bandwidth 2"** — two earlier claims corrected.
- The rank-1 form reproduces 156 off-diagonal entries from 13 numbers, to **1.4e-5**.
- **Method trap:** at production weights, forming `H_data = H_total − H_prior` is catastrophic
  cancellation whose floor sits *above* the real band tail — it manufactures single-ULP artifacts.
  The band can only be measured with the mean constraint switched off.

## 5. What does the mean constraint do to the conditioning?

![Mean constraint effect](plots/fig_dedx_hessian_meanconstraint.png)

**Tests:** the same Hessian with the mean-constraint weight at 0 versus production.

**Concludes:**
- Derived and verified: `H_mean = 2w·ggᵀ + 2w(m−t)·diag(g)`. Rank-1 because **any** penalty on a
  single scalar functional contributes `∇f∇fᵀ`.
- It pins exactly **one** direction: `|q|² = 73356` vs top eigenvalue **74186**, eigenvector
  overlap 0.999990, 62× gap to the second. The other N−1 "shape" directions are untouched.
- **κ goes 25 → 23874** purely because of that one direction.
- It **cancels short-segment curvature**: the diagonal term is negative when the running mean is
  below target, driving one segment's total curvature to 3.67 against a median of 3826.
- Diagonal dominance collapses as ~1/(N−1) — worse as N grows, arguing for a *direct* structured
  solve over anything iterative.

## 6. Is the mean constraint itself correctly specified?

![The dE/dx mean constraint](plots/fig_dedx_mean_constraint.png)

**Tests:** (A) true batch-to-batch spread of the mean dE/dx, from truth data, versus the σ the
weight imposes; (B, C) an 18-arm toy scanning weight and prior location.

**Concludes:**
- **The target 1.887 is correct.** True length-weighted mean over 3.03 M segments = **1.8853**,
  three independent computations agreeing (+0.09 %). Do not change it.
- **The weight is far too stiff.** It imposes σ = 0.12 % on every batch while batch means genuinely
  scatter by ~1.5 % at production's ~72 cm/batch — **~13×** over-constrained.
- On the toy, `w = 1e5` **costs 4× calibration accuracy** and injects a **+108 % lifetime bias**
  starting from truth (25.8 % → 6.2 % mean error on lowering to 3000).
- The unconstrained fit lands on the prior *mode*, confirming the original mode-vs-mean diagnosis:
  MAP shrinks to 1.874 while the data's mean is 1.885.
- **Retracted:** the `w = 1/(2σ²)` prescription. Its own prediction for a 46 cm/batch toy (464) is
  much worse than 3000 — the constraint doubles as the dE/dx↔calibration degeneracy breaker, which
  σ does not capture. **The weight must be scanned, not derived.**

## 7. Does GN actually pay off on the dE/dx block? *(no figure — table)*

**Tests:** the structured Newton step (5 HVPs via mod-5 probing + Sherman–Morrison) against Adam
at **equal wall-clock**, in both conditioning regimes.

| regime | GN 200 it (~8.8 min) | Adam 610 it (~8.2 min) | difference |
|---|---|---|---|
| `w = 3000` (κ ≈ 750) | 940.7 ± 10.5 | 942.1 ± 5.8 | −1.4 ± 6.9 → **tie** |
| `w = 1e5` (κ ≈ 23874) | 939.3 ± 2.7 | 941.4 ± 6.0 | −2.1 ± 3.8 → **tie** |

**Concludes:**
- **A tie.** GN wins per-iteration (949.9 → 940.7 in the same 200 steps) but costs 3.0×/iteration,
  which consumes exactly the gain.
- The band assumption **holds on real data mid-fit**: off-band leakage 2.5e-06, and 5 probes
  reproduce 11 dense HVPs to 2.6e-06 — at N=4000 that is 5 HVPs instead of ~4000.
- **A prediction of mine that failed:** I argued GN would win where κ is worst, since Newton is
  affine-invariant. **Adam is fine at κ = 23874.** Do not use the κ argument to motivate GN here.
- Verdict: correct, validated, **not worth deploying**.

## 8. Why did the dE/dx MAE get *worse* with more fitting?

![dE/dx MAE anatomy](plots/fig_dedx_mae_anatomy.png)

**Tests:** (A) MAE per batch over the trajectory; (B) per-segment error against segment length;
(C) freezing sub-threshold segments.

**Concludes:**
- **Not overfitting and not the optimiser.** One batch improves monotonically; the other degrades
  2.7× — and within it, **one 0.0355 cm segment** carries the effect.
- Both the likelihood and the prior are **dx-weighted**, so that segment has **1044× less
  curvature than the median**: an unidentifiable free parameter that drifts. `corr(log dx, |err|)
  = −0.75`.
- At production scale **2.65 % of segments carry 0.041 % of the track length**, each with its own
  fitted parameter.
- **The fix works:** freezing sub-0.15 cm segments gives **MAE −62 %**, worst-segment error
  −84 % → −11 %, and calibration also improves (−0.97 ± 0.11) — so it is not metric cosmetics.
- **Report a length-weighted MAE.** The unweighted metric is dominated by segments carrying no
  physics: GN has the *best* length-weighted and *worst* unweighted MAE in the same runs.

## 9. Do any of the dE/dx recommendations survive production? *(the reality check)*

![Production-scale validation](plots/fig_production_scan.png)

**Tests:** 2×2 factorial (parametrisation × weight, 3 seeds) on the real `start_sci_case.sh`
config — fitted chain positions from an imperfect guess, not true geometry.

**Concludes:**
- **The toy gain does not transfer.** Lowering `w` moved mean calibration error 25.8 → 6.2
  (−19.6) on the toy; in production **20.9 → 24.3 (+3.4)**.
- **But this is underpowered, not a clean negative.** Every configuration has one seed at
  −45…−51 % and two at +27…+92 %; seed sd 36–62 % swamps every between-config difference. With
  n = 3 nothing is resolvable in either direction.
- **Geometry dominates.** Every cell shows `tran_diff` −10…−17 %, `long_diff` +28…+41 %,
  essentially unchanged across configurations — far worse than any toy. The dE/dx work is
  plausibly polishing something that is not currently limiting.
- ⇒ **Do not change the production default on the toy evidence.** To settle it: `SCIMODE=ceiling`
  (true geometry, removing the dominant noise source) and/or ≳10 seeds.

## 10. Two more tests that did not pay off *(no figures)*

**(scale, shape) reparametrisation** — `dedxᵢ = M·sᵢ`, making the mean an explicit parameter.
- Verified to work: `mean_dedx = exp(μ)` exactly, and the rank-1 coupling collapses
  (`max|H_aa| = 2e-14` vs `H_μμ = 2e+04`).
- On toys, paired with `w = 3000`: **lifetime +18.1 % → −2.4 %**, the best number anywhere.
- **Worse at `w = 1e5`** — μ becomes one heavily-driven coordinate and Adam, scale-invariant per
  coordinate, oscillates it. The two changes are complementary, never independent.
- **Not reproduced at production scale (§9).**

**Shape-only prior** — evaluate the prior at scale-normalised dE/dx so it says nothing about the mean.
- Mechanism verified **exactly**: prior value varies by 1.8e-14 over a ±10 % scale sweep, vs 25.4
  for the current form.
- **Still makes the fit much worse** (calibration 9.6 → 20.2 %, lifetime −2.4 → +54.6 %): removing
  the prior's pull on the scale exposes the fit fully to a mis-specified per-batch target. The
  prior was a load-bearing shield.
- Default off.

**The pattern across §7, §6 and this section:** three mechanism-level arguments of mine were each
verified *exactly* in isolation and each failed end-to-end. **Verifying a mechanism says little
about whether it improves the fit.**

## 11. The geometry block *(no figure — closed)*

- **Adam beats GN on accuracy**: 51 µm mean per-track residual vs 107 µm. A per-batch GN solve
  fits that batch's noise realisation; Adam averages over seeds. Second-order cannot beat
  stochastic averaging on a noisy differentiable-sim objective.
- **And on cost**: ~400× per-iteration gap, compile-dominated.
- The LM improvements from §1 **do not reach this code path** — setting those env vars with
  `--geom_optimizer ggn` has no effect at all.

---

## What to do next

1. **Enable the minimum-length cut** (`LARND_DEDX_MIN_DX≈0.15`). Direct mechanism, measured
   benefit, independent of everything else.
2. **Settle the weight question properly** — `SCIMODE=ceiling` and/or ≳10 seeds. Do not adopt
   `w = 3000` on the toy evidence.
3. **Audit the batching.** Production requests 100 batches and gets **13** (`SCILEN=50` loads
   `len × (nbatch+2)` cm, then drops trajectories longer than `max_batch_len`). You may be fitting
   on far less data than intended — this is independent of everything above and possibly larger.
4. **Attack the objective, not the optimiser.** §1 and §7–8 both show the binding constraint is
   where the objective's minimum sits, not how fast you reach it. The target-noise/lifetime
   hypothesis is the highest-value open item.
5. **Stop**: dE/dx optimiser work, position-GGN, deriving the weight from σ, "fixing" the 1.887
   target, shifting `dedx_student_loc`.

## Traps worth keeping

- **`--max_nbatch` is not what you get** — including in production (13 batches, not 100).
- **The output label must encode every knob you vary.** Missed three times; cost ~17 A100-hours on
  a scan that silently overwrote itself and left files resembling a clean seed set.
- **Check seed scatter before interpreting a mean of 3.**
- **A toy that fixes the hard block overstates easy-block improvements.**
- **NaN-filtering breaks index alignment**, and a per-batch diagnostic is not a global one —
  together these manufactured a "+62 % degradation" curve that does not exist.
- **Piping job output through `tail -N` discards startup diagnostics** — "feature didn't fire" and
  "log truncated" then look identical.
- **A difference of two large float32 numbers cannot resolve a small structure**; test candidate
  structure against transpose symmetry.
- **Truth-point curvature does not extrapolate** to where the optimiser actually operates.
- **`AssocGrpNodeLimit` is per-account, shared across partitions** — a different account bypasses it.
