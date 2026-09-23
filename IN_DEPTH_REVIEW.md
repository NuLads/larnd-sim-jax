# In-depth review — what actually improves the fits

23 September 2026. Every arm re-read across **all** convergence metrics: the five calibration
parameters, the dE/dx field (MAE and recovered width), geometry (position residual), the data
likelihood, and settledness. Arms are only compared **within a matched-target family**.

---

## The headline, and it is a correction

**⚠ The ×0.40 prior recipe does not deliver what I reported.** It was found on seeds 3/7/8 and
validated on the same three. On five seeds it had never seen:

| | baseline mean \|err\| | ×0.40 | paired Δ |
|---|---|---|---|
| seeds 3/7/8 (**tuned on**) | 7.26% | 2.95% | **−4.31 ± 1.48** (3/3) |
| seeds 1/2/4/5/6 (**fresh**) | 3.42% | 3.08% | **−0.34 ± 0.65** (3/5) |

The original "7.26 → 2.95%" was **mostly regression to the mean**: seeds 3/7/8 happen to have
unusually bad baselines (7.26% against 3.42% for the fresh five), so almost any intervention
improves them. On fresh data the gain is not significant.

Lifetime specifically does better than the aggregate — |lifetime| 5.69 → 3.56%, **−2.14 ± 1.66**
(4/5 improved) — but that is ~1.3σ, suggestive only.

**Revised status: ×0.40 is a plausible small improvement, not a validated recipe.** The honest
claim is "≲2 points of lifetime, unproven at n=5", not "matches truth-pinning".

## What the constraint ladder really does (this part holds)

All n=3 on the matched-target family, seeds 3/7/8:

| arm | A_b | E field | tran | long | lifetime | mean\|e\| | dxMAE | dx width | pos | likelihood |
|---|---|---|---|---|---|---|---|---|---|---|
| ×2.5 | 0.01 | −0.03 | 4.0 | 34.3 | 20.34 | 11.77 | 0.0771 | 137% | 192 | 9794 |
| ×1.6 | −0.13 | −0.04 | 1.7 | 29.9 | 19.73 | 10.32 | 0.0650 | 109% | 186 | 9714 |
| **×1.0 production** | −0.07 | −0.04 | 0.6 | 21.1 | 13.37 | 7.26 | **0.0552** | 71% | 184 | 9872 |
| ×0.60 | −0.01 | −0.03 | 0.7 | 5.8 | 7.76 | 3.36 | 0.0559 | 36% | 168 | 9952 |
| ×0.40 | 0.11 | −0.05 | 1.2 | −8.2 | 4.85 | **2.95** | 0.0598 | 18% | 163 | 10041 |
| ×0.25 | 0.22 | −0.04 | 0.4 | −15.0 | 1.71 | 4.18 | 0.0648 | 8% | 159 | 10172 |
| ×0.15 | 0.52 | −0.03 | −0.2 | −18.9 | −2.24 | 4.95 | 0.0685 | 3% | **153** | 10286 |
| dEdx frozen@own | −0.07 | −0.04 | 1.3 | 13.0 | 10.60 | 5.13 | 0.0551 | 71% | 153 | 9815 |
| dEdx frozen@TRUTH | 0.01 | −0.03 | −1.5 | 6.2 | 3.00 | 3.19 | 0 | 100% | 148 | 9907 |
| both nuis@TRUTH | 0.17 | 0.00 | 1.1 | −9.7 | −3.74 | 2.94 | 0 | 100% | 173 | **8994** |

**Four things are solid across every metric:**

1. **Only the soft plane moves.** A_b stays within ±0.5% and E field within ±0.05% across a
   17-fold change in prior scale. Whatever the constraint does, it does it entirely to
   (lifetime, long_diff). This is consistent in every arm including the truth-pinned ones.
2. **There is a genuine three-way trade** (fig 101c). The dE/dx field's *own* accuracy is best at
   production (dxMAE 0.0552); the *calibration* is best near ×0.40 (dxMAE 0.0598). Optimising the
   nuisance and optimising the deliverable are different objectives — Neyman–Scott, measured.
3. **The objective disagrees with every physics metric.** The likelihood worsens monotonically
   along the ladder, 9714 → 10286, while lifetime, long_diff and position all improve. Position
   improves 192 → 153 µm over the same range. Nothing about this is visible to the fit.
4. **Everything is converged.** Settledness (peak-to-peak over the last 2000 iterations) is
   ≤0.5% on every parameter in every arm — the endpoint differences are real, not drift.

## What does NOT improve the fits — the measured nulls

| intervention | verdict |
|---|---|
| Wider prior (×1.6, ×2.5) | **Actively harmful**: mean \|e\| 7.3 → 10.3 → 11.8. The restored width is noise (dxMAE 0.055 → 0.065 → 0.077) |
| All schedule work | null or harmful (gating +55%, warmup +59%, multi-pass re-draws the landing) |
| Initialisation | −2.4 ± 2.8 at n=6, not significant |
| Tying the diffusions | **untested** — the linked fit put both diffusions ~+56% off with an out-of-range derived parameter, and the likelihood could not distinguish it from the control (−19 ± 54 nats) |
| Every dE/dx prior knob | weight, shape, scale, reparametrisation, mean target, drift penalty — all null |
| Second-order optimisers | tie or lose to Adam everywhere |

## What genuinely improves things

Ranked by how well established each is, not by effect size:

1. **Freezing the dE/dx block mid-fit** — position 184 → 153 µm, needs no truth, and the
   `DXFRZ`/`DXTRU` pair shows it is the *freezing*, not the correctness, that buys the geometry
   (31 of the 37 µm). This is the most robust positive result we have.
2. **Constraining the dE/dx prior** (×0.40–×0.25) — real on the soft plane and on position, but
   the size is unproven off the tuning seeds (see above).
3. **The calibration curve** — not a fit improvement but the only thing that converts the residual
   bias into a quoted number: τ_true = (τ_fit + 204)/1.232, ±2.9% over 800–3200 µs.

## Two structural facts that constrain everything

- **The ceiling.** With both nuisance blocks pinned at truth the calibration still lands at
  lifetime −3.74 ± 0.39%, mean \|e\| 2.94%. No amount of nuisance work reaches zero.
- **The floor of the degeneracy.** The link test showed ±56% on both diffusions costs ~0 nats of
  likelihood. The diffusion direction is essentially unconstrained by the data at this level.

## Methodological conclusion

Four results this month held at small n and reversed when tested properly: the multi-pass
bootstrap, the init ladder, the diffusion mechanism test, and now the ×0.40 recipe. Three of them
failed the *same* way — **evaluated on the data that produced them**. The rule going forward is
explicit: **any tuned quantity must be validated on seeds not used in the tuning, before it is
reported.** The ×0.40 scan should have included fresh seeds from the start; it cost a round trip.
