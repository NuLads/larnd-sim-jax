# The dE/dx prior: every tuning study, and what each production value rests on

Consolidated 16 Sep 2026 from `NOISE_CAMPAIGN_REPORT.md` (§6d, §6h, §6m, §6o–§6w), `GN_REPORT.md`
(§6, §6.3, §6.5) and the memory notes. Figures are in `plots/noise_report/`.

---

## 1. Current production values, and what each is justified by

```
BASE="...,SCIDEDXPRIOR=1.0088,SCIMEANW=2500.0"
REFIT="SCIDXNUL=21.760,SCIDXNUR=4.720,SCIDXLOC=1.8786,SCIDXSL=0.0782,SCIDXSR=0.0801"
```

| knob | value | status |
|---|---|---|
| `dedx_prior_weight` | 1.0088 (= 1/dx, β = 1) | **TUNED** — §6s tempering derivation + BT2 (6 paired seeds) |
| `dedx_student_nu_l / nu_r / scale_l / scale_r` | 21.760 / 4.720 / 0.0782 / 0.0801 | **TUNED** — MLE refit on the production grid, split-half CV (+0.9744 held out vs +0.9751 in-sample) |
| `dedx_student_loc` | 1.8786 | **BUNDLED, never independently tested.** The *old* value 1.874 was separately defended (GN §6.5: "do NOT shift loc to 1.8847"); 1.8786 arrived only as part of the refit |
| `dedx_mean_constraint_weight` | 2500 | **DERIVED, by a formula later declared invalid** — see §4 |
| `dedx_mean_constraint_target` | 1.887 | **VERIFIED** — true mean 1.88656, so +0.023% off; budget requires 0.12% to keep \|A_b bias\| < 0.1 pt |
| `dedx_use_split_t` | True (hard-coded) | **NEVER JUSTIFIED.** No split-vs-symmetric comparison exists. Its asymmetry is what creates the mode≠mean gap the mean constraint exists to patch |
| `dedx_soft_barrier_threshold / weight` | 8.5 / 1.0 (hard-coded, not even an env knob) | **NEVER STUDIED.** Measured contribution to the loss: exactly 0 — it never fires |
| `dedx_drift_profile_weight` | 0.0 | **TUNED TO ZERO by three independent nulls** (§6m, `schDP`, and the retraction of the original dpw=1e6 result) |

⚠ **Every launcher default is now stale** relative to what the drivers export: `start_sci_case.sh`
still defaults to prior 0.5, ν_L 4.785, ν_R 2.073, s_L 0.1204, s_R 0.1058, mean weight 1e5. Any job
launched without the `BASE`+`REFIT` strings silently runs the *old, worse* prior.

---

## 2. The diagnosis — the prior was right for the wrong segmentation (§6s, fig 48)

The split-t shape was an MLE of the true per-segment spread on the **target** grid (dx = 2.32 cm)
but is applied on the **guess** grid (dx = 0.99 cm), where the true spread is **1.6× narrower**
(0.0843 vs 0.1354). Refitting on the target grid gains +0.0002 nat — it was already optimal there;
refitting on the production grid gains **+0.1532 nat/segment**.

Weight enters as a tempering exponent β = w·dx (identity `ν_eff = β(ν+1) − 1` verified to 4e-14).
At production β = 4.96 this caused two harms at once:
* **over-shrinkage** — prior 0.046 wide against a true 0.084; the fitted field collapsed to **32% of
  the true spread in 100% of 900 batches** (Q-Q slope 0.36);
* **mean displacement** — tempering an asymmetric density moves its mean, 1.8819 → 1.8666 vs a true
  1.8869.

Closure check: 0.281 nat/seg × w 5 × 404 seg/batch = +568/batch vs +562 measured.

**Density ranking on the true field** (higher = better prior): refit β=1 **+0.981** > width-matched
β=1.8 +0.954 > current β=1 +0.828 > current w=5 **+0.430** > current w=0.5 +0.326. Production was
the worst of the five.

## 3. Why re-weighting cannot fix it — the α-scan (§6s, fig 52)

Walking the field from fit (α=0) to truth (α=1): the **likelihood's** optimum is **α\* = 0.50** —
the data wants 66% of the true spread, not truth, because with finite hits/segment the ML field is
itself shrunk. Some shrinkage is *correct*.

| prior weight | β | α preferred |
|---|---|---|
| 5 (old production) | 4.96 | 0.000 |
| 3 | 2.97 | 0.059 |
| 1.009 (β=1) | 1.00 | 0.227 |
| 0.5 (launcher default) | 0.50 | 0.319 |
| 0 (no prior) | 0 | 0.464 |

**No weight reaches 0.50**, structurally: the prior as applied (0.046) is narrower than the data's
optimum (0.056), so every weight pulls α *down*. A correctly-calibrated prior would pull α *up*.
Per-batch: 97% of batches want more dispersion than the fit, 100% want less than truth.

Removing the prior entirely is *not* safe: the likelihood moves only −10 nat/batch against the
prior's +562 — the data cannot see ~4000 per-segment parameters individually, and an unconstrained
field would absorb calibration error instead.

## 4. The mean constraint — an A_b systematic, not a lifetime lever (§6u, fig 53; GN §6)

Mis-setting the target is absorbed **linearly by A_b** (−0.837 ± 0.018 pts/% at w=1e5, 47σ;
−0.264 ± 0.022 at w=3000) and **lifetime does not respond at all** (−0.16 ± 0.48 and −0.25 ± 0.44).
Stiffness sets how much the fitted mean follows a wrong target: **93% at 1e5, 30% at 3000**.

The GN toy (18 arms, calibration initialised at truth) is where 1e5 was first convicted:

| loc | mean_w | mean bias | mean abs calib err | lifetime |
|---|---|---|---|---|
| 1.874 | 1e5 | +0.210% | 25.8 ± 2.2% | +107.6 ± 11.3% |
| 1.874 | **3000** | +0.026% | **6.2 ± 2.0%** | **+16.0 ± 0.7%** |
| 1.874 | 0 | −0.584% | 6.5 ± 2.3% | +4.9 ± 16.5% |

⚠ **The `w = 1/(2σ²)` derivation behind 2500 was itself later declared invalid** (GN §6.3): at a
~46 cm/batch configuration the formula prescribes w = 464, and 464 loses badly to 3000 (calib error
16.5% vs 9.6%; lifetime +44.1% vs −2.4%). *What survives is the negative result — 1e5 is far too
stiff — not any particular replacement value.* So 2500 is a defensible operating point, not a
derived optimum.

Also note the mean constraint is only **0.1% of the loss** (+8.9 nats) against the student-t's
**22.2%** (−2178) — the shape term is where the leverage is.

## 5. The decisive experiment — the BT arms (§6v, figs 54–55)

Three production arms, 6 paired seeds, differing only in the prior:

| arm | shape | w | mean w | A_b | lifetime | dE/dx MAE | ⟨dE/dx⟩ err | spread |
|---|---|---|---|---|---|---|---|---|
| BT0 (control) | current | 5 | 1e5 | +0.77 | −5.44 | 0.0581 | −0.454% | 0.32 |
| BT1 (weight only) | current | 1.0088 | 2500 | +0.94 | +2.97 | 0.0710 | −1.019% | 0.95 |
| **BT2 (refit)** | **refit** | 1.0088 | 2500 | **+0.27** | **+0.28** | **0.0569** | **−0.199%** | 0.70 |

A_b better on **6/6 seeds** (paired −0.50 ± 0.09, 5.6σ); |lifetime error| 5.97 → 1.62%.
**BT1 is decisively worse than the control** — exactly as the α-scan predicted. *Correcting the
weight without the shape is a regression, not a partial improvement.*

Validated afterwards on the BT2 fit (§6w, fig 56): total-vs-likelihood optima now differ by 0.17 in
α against 0.50 with the old prior, and the data's preferred spread reads 63% on the refit fit vs 66%
on the old — two different fits, two different priors, the same answer.

## 6. What is still open

1. **The refit's sample-independence.** Its parameters are an MLE of *this* MC sample's true
   per-segment dE/dx. CV says it is not overfitted, but not that it is sample-independent. An
   independent shape (e.g. from the dE/dx flow model in `detector_properties/`) would settle it.
2. **`dedx_use_split_t` has never been tested.** The asymmetry is the root of the mode≠mean problem
   the mean constraint patches — a symmetric-t arm is a one-line, never-run comparison.
3. **`loc = 1.8786` was never isolated**, and the value it replaced *was* defended by a dedicated study.
4. **The soft barrier is dead code in practice** (contributes exactly 0) yet is hard-coded in seven
   launchers and not exposed as a knob.
5. **The prior remains the campaign's one proven objective-side defect**: even after the refit,
   moving dE/dx to truth costs +568 nats of which **+561.9 is the student-t prior**, and the fit
   still sits at α ≈ 0.23–0.32 against the data's 0.50.
6. ⚠ **The headline motivation is partly undercut.** The +0.27%, 6/6-positive A_b signal that
   motivated the prior-shape work was later shown (fig 73, `DSREP`) to be common-mode from the single
   shared track sample — on random samples A_b is unbiased at the 0.04% level. The lifetime and
   dE/dx-spread gains are unaffected.

> **Standing caution from §6s, worth repeating:** α = 0.50 is the *likelihood's* optimum, not a
> target to chase — pushing the field there would be fitting noise. The criterion that matters is
> whether A_b and lifetime improve, which only BT-style paired arms can measure.

---

## 7. ⚠ CORRECTION (17 Sep): "the data wants 66% of the true spread" — right number, wrong reason,
## and the wrong target for calibration (fig 94)

§6s justified the 66% with: *"with finite hits per segment the maximum-likelihood field is itself a
shrunk version of truth, because the per-segment values are individually noisy. Some shrinkage is
correct."* **The first sentence is backwards.** A noisy estimate is OVER-dispersed:
Var(estimate) = Var(truth) + Var(noise). Measured on 602k core segments across the 16 replicas:

| quantity | value |
|---|---|
| true per-segment spread | 0.078 |
| implied per-segment measurement noise | **0.102** |
| per-segment SNR | **0.77** — each segment is measured worse than it varies |
| width of the *unshrunk* ML field | **165% of truth**, not below it |
| our fitted field | 71% of truth |
| our shrinkage slope λ | 0.457 (Bayes-optimal for this SNR: 0.369) |

So what is shrunk is the **MAP/posterior-mean** estimate, not the ML one, and the fit is doing that
near-optimally. That is the correct estimator for **per-segment MSE**. It is the wrong one for
**calibration**, and `DXTRU` (fig 93) measures the cost: pinning the field at 100% buys
**−10.4 ± 3.3 points** of lifetime bias, 3/3, while the loss gets *worse*.

**Why shrinkage specifically poisons lifetime.** The shrinkage bias is *coherent*: every segment is
pulled toward the same centre in proportion to its own deviation. A coherent, drift-correlated
distortion of the charge field is exactly what the lifetime parameter exists to absorb — so the
calibration block eats it. This is the Neyman–Scott / nuisance-parameter problem in its standard
form: the estimator that is optimal for the ~4000 incidental parameters is not the one that leaves
the structural parameter unbiased.

**The second reason the α-scan cannot see 100%:** its α = 0 endpoint is a field fitted to the *same
data* being scored, so it has already absorbed that realisation's noise and partially retaining it
scores well **in sample**. The scan cannot separate "correct shrinkage" from "overfitting", which
is precisely why the §6s caution was right even though the reasoning beside it was not.

### The deployable test, launched 17 Sep (`WIDE16D*`, `WIDE25D*`, 6 jobs)

`DXTRU` pins the field at truth, so it is an upper bound, not a recipe. The **deployable** version
is to stop the MAP shrinkage from happening in the first place, by **deliberately widening the
prior beyond the true spread**. Since λ = s_p²/(s_p² + σ_n²), a wider prior scale raises λ and the
fitted width follows:

| prior scale | predicted λ | predicted fitted width |
|---|---|---|
| ×1.0 (production, = `DSREP*` control) | 0.37 | 61% of truth |
| **×1.6** (`WIDE16`) | **0.60** | **99% — the target** |
| ×2.5 (`WIDE25`) | 0.79 | 130% — deliberate overshoot |

Only `--dedx_student_scale_l/r` change (0.0782/0.0801 → 0.1251/0.1282 and 0.1955/0.2003); shape ν,
`loc`, the prior weight and the mean constraint are all untouched, and the ×1.0 arm already exists
as the replica baselines on the same three seeds. **This is a legitimate recipe, not a truth
injection**: the widening factor is calibrated on simulation, exactly as the refit shape already is.

Reads as: `WIDE16` recovers a large share of `DXTRU`'s −10.4 points ⇒ the shrinkage *is* the
mechanism and the fix is a one-knob change. `WIDE16` fails while `DXTRU` works ⇒ what matters is
the per-segment *noise*, not the width, and only marginalisation (or longer segments) will do.
`WIDE25` says whether the effect is monotone in λ or has an optimum — and guards against an
unconstrained field running away, since the soft barrier is measured never to fire.

**Consequences for what to aim at:**
1. **Aim at 100%, not 66%,** whenever the deliverable is calibration rather than the dE/dx field.
2. **Do not tune the prior toward α\* = 0.50** — that target was derived from the wrong criterion.
3. **The fix is to marginalise, not to re-weight**: integrating the block out removes the coherent
   bias that optimising it creates. The exact banded+rank-1 dE/dx Hessian machinery already exists.
4. **Or raise the per-segment SNR at source.** SNR ∝ √dx, so reaching SNR ≈ 2 needs ~7× longer
   segments (~6.6 cm vs today's 0.99 cm) — which trades away real dE/dx structure, so it needs its
   own study rather than a guess.
