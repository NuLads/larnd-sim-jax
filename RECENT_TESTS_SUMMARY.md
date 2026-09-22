# Where we stand — the September test programme

Status as of 22 September 2026. Every number below is measured; ⚠ marks a claim I reported earlier
and have since had to correct.

---

## The one-paragraph version

We set out to remove a **+8.8% lifetime bias** and instead learned what it actually is. It is real
and universal (survives 4 truth points), it is **not** an optimiser problem (every schedule,
restart and initialisation lever is a measured null), and it is **not** a prior-calibration problem
(every prior knob is a null). What it *is*: a **constant additive error in the attenuation rate** —
the fit behaves as if 0.8% too little charge were lost over the 30 cm drift — and that single
number reproduces the entire τ-dependence. Two things came out of it that we can actually use: a
**one-line prior change (×0.40) that cuts calibration error from 7.3% to 3.0%**, matching what
truth-pinned nuisances achieve; and a **calibration curve** that converts the residual bias into a
quoted ±2.9% uncertainty. What we still do not have is the *cause* of the rate error.

---

## 1. What is now established

| finding | evidence | figure |
|---|---|---|
| The bias is real and sample-independent | +8.85 ± 1.18% over **16 replicas** | 85 |
| It survives a change of truth point | 4 truth points, 15/16 fits high, grand mean +8.0% | 92 |
| It is **not** optimiser-limited | schedules, restarts, multi-pass, init: all null or harmful | 77, 78, 91 |
| It is **not** prior-mis-calibration | weight, shape, scale, reparam, mean-target, drift: all null | 90, 95 |
| The objective is **blind** to it | loss ↔ lifetime r = +0.02; best-loss selection picks *worse* lifetimes on 3/4 seeds | 82, 83 |
| Perfect nuisances do **not** give an unbiased fit | both blocks at truth → lifetime **−3.74 ± 0.39%** | — |
| **Constraining the nuisance beats estimating it** | ×0.40 prior scale → mean \|err\| **2.95%** vs 7.26% production | 97, 99 |
| The bias is a constant **attenuation-rate** offset | Δ(1/τ) constant to 17% across τ = 800–3200 µs | 98 |

## 2. The two deliverables

**(a) A better recipe — the ×0.40 prior scale.** Constraining the ~4000 per-segment dE/dx
parameters harder improves *every* calibration parameter, monotonically, to an optimum:

| prior scale | ×1.0 (production) | ×0.6 | **×0.40** | ×0.25 | ×0.15 |
|---|---|---|---|---|---|
| mean \|error\| (5 params) | 7.26% | 3.36% | **2.95%** | 4.18% | 4.95% |
| lifetime | +13.37 | +7.76 | +4.85 | **+1.71** | −2.24 |
| position (µm) | 184 | 168 | 163 | 159 | **153** |

All points n = 3. **×0.40 matches pinning both nuisance blocks at truth (2.94%) while using no
truth information.** The gain is entirely in the soft plane — A_b and E field never move — so it
costs nothing. Use **×0.40** as the default, **×0.25** if lifetime is the deliverable.

*Why it works:* the prior was tuned to be a good **density model of dE/dx**; the right criterion is
to **minimise calibration bias**. Those give different answers. Constraining a nuisance block below
its own statistical optimum is correct when the deliverable is a structural parameter — textbook
Neyman–Scott.

**(b) A corrected estimator — the calibration curve.** Over τ = 800–3200 µs the response is linear:

> **τ_true = (τ_fit + 204) / 1.232**, residual **±35 µs = ±2.9%**

against a raw bias running to +17.6%. Above ~3500 µs the estimator is noise-limited (only 4% of
charge lost over the drift) — do not extrapolate.

## 3. The mechanism — what we know and what we don't

Re-expressed in the variable the data actually measures, the bias is **constant**: the error in the
attenuation rate 1/τ is −4.42 ± 0.75 ×10⁻⁵/µs over 800–3200 µs (17% spread, versus 63% for the
relative bias). Physically: **the fit behaves as if 0.81% too little charge were lost over 30 cm**.
Everything else — the τ-dependence, the apparent 1/lever law, the whole calibration curve — follows
from that one number arithmetically.

⚠ **The leading candidate is not confirmed.** Over-estimated longitudinal diffusion (+19% in these
runs) would spread far-drift hits, push charge below threshold, and be compensated by a longer
lifetime — right sign, right drift-dependence. Pinning long_diff at truth gave **−6.71 points at
n = 2**, which I reported as confirmation. **At n = 3 it is −4.34 ± 2.88 (1.5σ), 2/3 seeds**, with
seed 7 flatly reversing (+20.55 → +20.93). **Status: suggestive, not established.**

Running now: the **linked-diffusion test** — tying long_diff to tran_diff through the mobility
relation D_L/D_T = 1 + (E/μ)dμ/dE, against a link-consistent truth. It probes the same physics from
the other direction, and matters because tran_diff is measured **6× more precisely** than long_diff
(±1.5% vs ±9.5%) with r = −0.84 between them.

## 4. Dead ends (measured, not assumed)

- **Optimiser side:** multi-pass bootstrap (ensemble mean 11.7→13.6→13.7%), gentle relaxation (own
  attractor at +5–7%), calibration gating (+55%), warmup tuning (+59%), schedule variants,
  initialisation (−2.4 ± 2.8, n.s. at n = 6).
- **Prior side:** weight (α-scan: structurally cannot reach the optimum), shape (BT1 regressed),
  **scale widening** (restores width, *worsens* physics — the extra width is noise), (scale,shape)
  reparametrisation, mean target (an A_b knob only), drift penalty.
- **Second-order:** GN from scratch, GGN/Fisher, L-BFGS geometry, banded dE/dx Newton — all lose to
  or tie Adam; GN geometry is dead at production scale (~400× per-iteration cost).

## 5. What I got wrong, and the rule it produced

Three claims this month were reported and then corrected, all the same shape — **a result that held
at n ≤ 2 and reversed at n = 3**:

1. the multi-pass bootstrap "converges to truth" (rep 8's 16.4→0.65% was a lucky draw);
2. the calibration-init ladder's monotone law (2/6 seeds at n = 6);
3. the diffusion mechanism test (61% → 31%, 1.5σ).

Plus two where a *failed query* was read as a *negative answer*: a target-verification that passed
while checking zero runs, and a job watcher that read a `squeue` outage as "wave complete".

**Rules now in force:** never report a mechanism at n ≤ 2; make every check count what it examined
and fail loudly on an empty set; distinguish "nothing found" from "could not look".

## 6. Next

1. **Read the linked-diffusion test** (~2 h) — the last live probe of the rate-error mechanism.
2. **Adopt ×0.40** as the production prior scale; re-run the replica ensemble under it to confirm
   the gain holds off the three scan seeds.
3. **Find the residual rate error.** The cleanest unrun diagnostic: evaluate the forward model at
   truth and histogram the charge residual (simulation vs target) against drift distance. If a
   ~0.8% drift-correlated discrepancy is already there with everything at truth, it is a
   forward-model defect and fixable at source — no fitting involved, evaluation only.
