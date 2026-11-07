# S4 calibration — consolidated ranked plan
Date: 2026-07-27. Folds in the LR-anneal result + all 10 proposer outputs.

## Problem statement (what is actually still open)

Solved: S1/S2 at 400 cm noise-ON — all 5 calib params at the statistical floor
(sigma(lifetime)=1.6%, sigma(long_diff)=1.4%). Ab/eField/tran_diff also fine in S4.

Open, in order of severity:
1. **long_diff in S4**: +10.7 / +50.2 / -22.4% (anneal run, mid-flight). Annealing does NOT fix it.
2. **lifetime in S4**: +4.9 / -4.8 / +2.0% under annealing — oscillating around truth, not walking.
   Was +38+-26% (biased walk) before. Near-acceptable but not yet demonstrated converged.
3. **Geometry residual**: 287-299 um, vs 26 um the spline basis can represent and ~9 um
   statistical floor. 1-2 orders looser than the data supports.

## Verified facts that constrain every proposal

- **V1** Calib LR never anneals (dEdx 100%, calib ~93%, geometry 0.03% of initial by run end).
- **V2** Fitted spline displacement is 379 um ALONG THE DRIFT AXIS (59% of magnitude,
  isotropic; |e2.zhat|=0.83). Drift position comes from timing -> geometry can manufacture
  apparent attenuation. Physically wrong.
- **V3** S4 opens a NEW soft direction: leading increment mode overlaps the static Hessian
  soft plane at only 0.51-0.77 (S2: 0.91-0.92).
- **V4** Batches built once by sequential floor-division, never reshuffled (dataio.py:375).
  Per-batch drift lever-arm std varies 6x (200cm) to 78x (50cm). Matches lag-100 autocorrelation.
- **V5** Only 864 / 216,091 tracks used = 0.4% of the file.
- **V6** Truth file drift span is +-31 cm = 16% of lambda=385 cm, baked into the file.
- **V7** GN/Fisher differs from exact Hessian by 1.75-4.27x relative Frobenius, and the gap
  GROWS with more batches. Structural, not sampling noise.
- **V8** Rigid geometry modes are pinned to microns by data (sigma dz=1.3um, dx=0.5um);
  marginalizing them costs almost nothing (lifetime sigma 3.75->3.79%) EXCEPT eField (x119).
- **NOT VERIFIED**: the FEE stochastic-vs-analytic charge-bias claim. Probe returned
  COULD-NOT-TEST. Do not cite it.

---

# TIER 0 — finish what is in flight (no new work)

**0.1 Let the 3 anneal seeds + S2 control reach 5000 it.** Currently 2600-2800.
Decision rule fixed in advance, to avoid a third endpoint-reading burn:
accept only if (a) |lifetime| < 10% on all 3 seeds AND (b) the tau trajectory over the
last 30% of iterations shows no monotone drift > 5 points. Otherwise treat as inconclusive.

---

# TIER 1 — verified defects, cheap, do regardless of anneal outcome

**1.1 Drift-axis decomposition of the geometry fit.** [V2, V3 | proposer: physics]
Decompose spline displacement into drift-projected (z-hat) and wire-plane components;
freeze or heavily penalize the drift-projected part. This is a real modelling defect, not a
tuning knob: the fit is currently allowed to move charge in the one coordinate that is
measured by timing rather than inferred. Directly targets the mechanism by which geometry
poisons lifetime, and is orthogonal to the annealing fix.
Cost: moderate code (basis change in _chain_mcs_spline / chain warp). Highest expected payoff.

**1.2 Tighten the MCS prior.** [V8 | proposer af86 #4]
Geometry is being fitted 2-3 orders looser (287-500 um) than the data supports (1.3 um on
rigid modes; 26 um basis-representable; ~9 um statistical floor). mcs_prior_weight=1.0 is
not binding. Pre-check is CPU-only and free: recompute _chain_mcs_spline's implied sigma per
mode at current sigma_mcs and compare against the observed residual, BEFORE spending GPU.
Then try x5-10. Caveat: inferred from the 3-rigid-mode Hessian, not the full spline space.
Cost: one CPU check + one config-only run. Stacks with 1.1.

**1.3 Shuffle/stratify batches per epoch.** [V4 | proposers #7, a979 #2]
Two independent angles converged on this. Bin tracks by drift-position quantile and
round-robin across batches, or at minimum re-permute batch_traj_indices each epoch.
Cost: near-trivial code. Pre-test is free (diff the per-batch z-std table before/after).
Expected: removes the epoch-periodic gradient bias; narrows seed-to-seed spread.

---

# TIER 2 — config-only curriculum experiments (cheap GPU, no code)

**2.1 Start geometry LATE, not early.** [proposer a8b7 #3] **Best single untested idea.**
Every variant tried so far either starts geometry at iteration 0 or freezes it late (after
contamination). Nobody has tried the calibration-lab order of operations: nail the global
constant (lifetime) first from an S2-like phase, THEN release local per-track shape.
Set chain_start_iter ~2000-3000. Mechanism is exactly right: S1/S2 recover lifetime to a few
% whenever geometry is exact or rigid, so the fix is to spend the early iterations in that regime.
Cost: one env var. Should be launched alongside Tier 1 work.

**2.2 Stagger calibration parameters instead of gating all 5 together.** [proposer ab04 #4]
_calib_start_iter gates all 5 as one block, but they are not one group: eField is decoupled
(|r|<0.10, own eigenvalue 1.28e10) while Ab-lifetime is the strongest late co-mover
(r=+0.81..0.94). Let eField run throughout; delay Ab relative to lifetime/diffusion.
Cost: small code (per-key start-iter dict). Lower priority than 2.1.

---

# TIER 3 — free offline diagnostics (no GPU, run before spending more GPU)

**3.1 Per-seed adaptive stopping / plateau detection.** [proposer ab04 #5]
The reported lifetime error shifts by up to 9 points of mean and 6 of std depending on the
averaging window; seeds plateau at different iterations. Every staging comparison in the
campaign compared seeds that were not equally converged. Build a plateau detector from
existing training_history logs and re-read ALL past variants at each seed's own plateau.
This may retroactively change which variants looked good. Do this first — it is free and it
audits everything else.

**3.2 Hessian stability away from truth.** [proposer af86 #5]
Every eigen-analysis was done AT TRUTH with 3 rigid modes. Recompute the exact Hessian at an
Adam iterate (e.g. it 1000) and check whether the soft (lifetime, long_diff) eigenvector is
stable. If it isn't, several proposals built on that eigenbasis are void.

**3.3 Model-free lifetime pilot.** [proposer a979 #3] **High payoff if it works.**
Bin charge vs RAW drift-time (no geometry correction) on existing S1/S2 outputs and fit a
single exponential; compare to truth and to the joint-fit value. If it lands within a few %,
lifetime can be estimated outside the joint fit entirely — or used as a warm start / prior.
Sidesteps the whole degeneracy. Computable from data already on disk.

**3.4 Ensembling as a bias-vs-multimodality check.** [proposer a979 #4]
S4 seed errors were all POSITIVE (+128/+8.6/+51) — suggests systematic bias, not variance.
Rerun 3-4 seeds with different geometry init; if the bias direction is init-invariant it is a
single bad basin, if init-dependent it is multimodal. This is a check on everyone else's
assumptions, including mine. Cheap (no new code).

---

# TIER 4 — moderate code, do only if Tiers 1-2 leave a gap

**4.1 Low-rank dEdx basis (mean + K<=3 residual modes).** [proposer a817 #4]
Strongest empirical lead in the dossier: the best S4 result came from raising
dedx_prior_weight 0.5->5, i.e. shrinking dEdx toward the track mean; and at low prior weight
per-segment dEdx MAE grows 0.035->0.137 MeV/cm over exactly the iterations lifetime drifts.
A hard low-rank basis removes the capacity to absorb the drift-depth trend rather than merely
discouraging it. Test at the CURRENT weak prior weight — if it matches weight=5 without the
prior's help, it is a structural fix.
Caveat: [[spline-dedx-suitability]] says per-segment dEdx is ~2x noisier in a spline basis.

**4.2 Split rigid global geometry modes from per-track local modes.** [proposer a817 #5, V8]
The 3 rigid modes are stiff (1e7-1e10) and solvable in 2-3 Newton steps from an Adam warm
start; the local per-track residuals are the slow part. Pull rigid modes out and solve them
near-exactly; leave local shape on Adam. Prerequisite: extend the exact-Hessian script to
include real per-track spline coefficients alongside the 5 calib params (the 8-param study
excluded the ~4000 dEdx dof, so the coupling is unmeasured).

**4.3 Repeated Newton polish at multiple checkpoints.** [proposer a54c P2]
GN-as-polisher is already validated. Instead of one correction at the end, insert
freeze-geometry-and-dEdx + exact 5x5 Newton at it 2000/3500/5000. Each insertion snaps
lifetime back to the local conditional optimum, capping how far the walk drifts between
corrections. Must stay damped/trust-region: full Newton from nominal diverges catastrophically
(lifetime 7% -> 1003% while loss still falls monotonically).
Cost: negligible (2-3 five-param Newton steps vs 6.7 h/run).

**4.4 Robust (Student-t) charge likelihood.** [proposer a581 #3]
sigma_charge is already a constructor kwarg on ProbabilisticLossStrategy; swapping the two
Gaussian log-density blocks for Student-t (nu~4-5) is localized. Motivated by V7 (the
quadratic model is a poor local fit) and by the threshold anomaly.
Targets long_diff/lifetime variance rather than bias — worth trying since long_diff is the
remaining ragged parameter.

**4.5 Aggregate drift-time lifetime term.** [proposer a581 #1]
Add a loss term that measures attenuation from ensemble charge-vs-drift-time, decoupled from
per-track geometry. Related to 3.3 — do the free pilot (3.3) FIRST; it tells you whether this
signal is clean enough to be worth building.

---

# TIER 5 — guardrails and things NOT to do

- **Do NOT build any K-FAC / natural-gradient / GN preconditioner.** [V7, proposer a54c P3]
  GN/Fisher is structurally wrong here by 1.75-4.27x and worsens with more batches. Only the
  exact (cheap, 5x5) Hessian is trustworthy as a curvature source.
- **Do NOT expect more data to fix S4.** [V5] Yes, only 0.4% of the file is used, and yes that
  is free statistics — but S2 already sits at the sigma=1.6% Hessian floor. More data buys
  variance, not bias. Worth doing for the final numbers; will not fix the lifetime.
- **Do NOT change the threshold based on the existing evidence.** [proposer a581 #4]
  Halving it (5000->2500) made lifetime worse (+29.4/+1.1 vs -0.3/+3.3), opposite to naive
  expectation. Diagnose the per-hit weighting near threshold first.
- **Do NOT cite the FEE charge-bias claim.** Unverified; the probe returned COULD-NOT-TEST.
  If it matters, run get_adc_values vs get_adc_values_average_noise_chunked directly.
- **Do NOT read endpoints of unconverged oscillating fits.** Burned twice. Use 3.1.
- **Cox-Reid profile-likelihood correction: likely a red herring.** [proposer a5bd #3]
  Expected magnitude is O(1/n), consistent with S2-level few-% residuals, NOT the S4 blowup.
  Falsifiable for <1 CPU-day (finite-difference det(I_geom,geom) around truth); if flat, drop it.
- **A wider-drift truth file** [V6] would raise lifetime Fisher info ~25x, but requires
  regenerating input data upstream (cyifan's pipeline). Park unless someone else is regenerating.

---

# Recommended execution order

Week 1, parallel:
  - Tier 0 (wait) + 3.1 (free audit of all past variants) + 3.3 (free lifetime pilot)
  - 1.2 pre-check (free CPU) then launch 1.3 + 2.1 as GPU jobs
Week 2:
  - Implement 1.1 (drift-axis decomposition), launch with 1.2 and 1.3 folded in
  - 3.2 + 3.4 as free diagnostics alongside
Week 3+, only if a gap remains:
  - 4.3 (cheapest of Tier 4) -> 4.1 -> 4.4 -> 4.2/4.5

Single highest-value item if only one thing can be done: **1.1 (drift-axis decomposition)**,
because it is a verified physics defect rather than a tuning knob.
Single highest-value FREE item: **3.1**, because it may invalidate past variant rankings.
