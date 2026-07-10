# Gauss–Newton as a fitting method — investigation

**Question:** can Gauss–Newton (GN) improve the joint calibration fit, and where?

**Short answer:** textbook GN does *not* apply to the current likelihood loss (it is not a
sum of squares), but the correct generalization — **Generalized Gauss–Newton / Fisher scoring
with Levenberg–Marquardt damping** — does, and it is an excellent fit for the **5-parameter
calibration block**, which is low-dimensional, ill-conditioned, and where the Hessian machinery
already exists. It is *not* a good fit for the large blocks (per-segment dEdx, geometry angles).

---

## 1. What GN requires

GN minimizes ½‖r(θ)‖² for a residual vector `r`. Update: solve `(JᵀJ + λI)Δθ = −Jᵀr`, with
`J = ∂r/∂θ`. It approximates the Hessian by `JᵀJ` (drops the `r·∇²r` term), so it needs only
first derivatives, and `JᵀJ` is PSD → stable. Levenberg–Marquardt (LM) adds damping `λ` to
interpolate between GN (fast, local) and gradient descent (safe, global).

**Requirement:** the objective is (or is well-approximated by) a sum of squares.

## 2. Our loss is NOT least-squares

The `llhd` loss (`strategies.py:840–939`) is a *marginalized point-process negative
log-likelihood*:
- tick-sequence intensity term (Hawkes/Poisson) — **dominant, not LS**
- Gaussian charge term `−½((target_charge − E[Q])/σ)²` — **this sub-term is LS**
- survival ("none") term + expected-hit count (Poisson normalization) — **dominant, not LS**

So classic GN applies only to the charge sub-term. **However**, the loss is a *smooth,
deterministic* likelihood — the stochasticity is integrated out analytically into
`log_p1 / log_T / expected_Q`. That smoothness is the property that makes the correct
generalization applicable.

## 3. The correct form: Generalized Gauss–Newton / Fisher scoring

For any NLL, the GN generalization is `H_GGN = Jᵀ (∇²_z L) J`, where `z` = model outputs
(per-pixel/tick log-intensities and expected charge), `J = ∂z/∂θ`, and `∇²_z L` is the loss
curvature in output space (PSD for the exponential-family / Poisson / Gaussian terms).
Equivalently, the Fisher Information `F = E[∇L ∇Lᵀ]`, with update
`θ ← θ − (F+λI)⁻¹ ∇L` (Fisher scoring / natural gradient).

It inherits GN's advantages: **PSD curvature** (stable), **only first-order Jacobians through
the sim** (avoids fragile 2nd derivatives through `argsort`/`cummax`/`erf`/`lax.cond` in the
FEE), and **scale/reparameterization invariance**.

## 4. Where to apply it — the 5-parameter calibration block

| block | dim | GN suitability |
|---|---|---|
| **calibration** (Ab, eField, tran_diff, long_diff, lifetime) | **5** | **ideal** — 5×5 curvature, direct solve |
| per-segment dEdx | 50–100 / track | GGN + matrix-free CG possible, heavier |
| chain geometry angles | ~2 / segment | the L-BFGS target; GGN possible but bigger |

The calibration block is the prime candidate:
- **5×5 curvature** → direct solve, no CG.
- **It is ill-conditioned.** Physical scales span `1e-5` (diffusion) to `2400` (lifetime); even
  in normalized space, eField/lifetime/Ab are correlated (all move total collected charge). The
  empirical signature: Adam needs hand-set **per-parameter learning rates** and *still* lands
  **lifetime +27%** off in the full fit — the classic profile of a poorly-conditioned problem
  where a curvature-aware step wins.
- Curvature-aware steps are scale-invariant → they would remove the per-param LR tuning and
  should fix the lifetime bias.

## 5. Feasibility here — already ~90% wired

- `compute_loss(..., with_hess=True)` already computes the exact 5×5 Hessian in normalized space
  (`fit_params.py:1730`, `jacfwd∘jacrev`).
- `HessianCalculator` (`fit_type='hess'`) already computes & logs Hessian + gradient per batch.
- **Missing = only the step:** replace the Adam update on `norm_params` with a damped solve
  `Δ = −(H+λI)⁻¹ g` (Newton/LM), or the PSD GGN/Fisher variant. ~30–50 lines.
- **Cost:** exact 5×5 Hessian ≈ a handful of sim passes; GGN via 5 jvps. Compile is heavier but
  paid once (persistent cache). The 5×5 solve is free. Expected convergence ~tens of iters
  (quadratic near optimum) vs ~thousands for Adam.

## 6. Obstacles / caveats

- **Exact Newton** needs 2nd-order autodiff through the sim; `argsort`/`cummax`/`erf`/`lax.cond`
  in the FEE have zero or non-smooth 2nd derivatives → the exact Hessian can be noisy/indefinite
  far from the optimum. → prefer **GGN/Fisher** (PSD, first-order only) + LM damping.
- **Batch structure:** curvature must be summed over the 100 batches per step. Prefer
  **full-batch GN** (accumulate `F` and `g` over batches, one LM solve/step) — cheap for 5 params
  and most stable — over per-batch stochastic steps.
- The `no_match_penalty` counting term is non-smooth; GGN handles the smooth part and damping
  absorbs the rest.
- **Diminishing returns for the big blocks** (dEdx/geometry). GN's edge is strongest exactly
  where Adam struggles most: the low-dim, ill-conditioned calibration block.

## 7. Recommendation

Implement **damped Generalized Gauss–Newton / Fisher scoring on the 5 calibration parameters**
as an alternative to the Adam calibration step, **full-batch** (accumulate `F`, `g` over batches,
one LM solve per step), reusing the existing `with_hess` machinery. Contrasted with the shelved
L-BFGS geometry effort, this is **low-risk**: 5×5, infra exists, a clean smooth deterministic
objective, and **no per-batch compile explosion** (one small extra graph, not 100 fresh compiles).

**Next steps**
1. *(running)* Empirical: compute the actual 5×5 Hessian condition number at the solution
   (`fit_type=hess` probe) — quantifies the ill-conditioning. → results appended below.
2. Prototype a `calib_optimizer=gn` that accumulates `F,g` over batches and takes an LM step;
   compare iters-to-tolerance and final bias vs Adam on the 50 cm set (same methodology: time-avg
   ≥300 iters, multi-seed).
3. If it removes the lifetime bias and cuts iteration count, adopt it as the calibration block in
   the joint fit.

---

## Appendix — empirical Hessian conditioning (probe result)

5×5 normalized-space Hessian at the **true params**, 1 batch, 50 cm (`fit_type=hess`):

```
params (norm space): [Ab, eField, tran_diff, long_diff, lifetime]
eigenvalues:  [0.659, 6.84, 13.97, 198.8, 138550.3]
condition number (|max/min|):  2.10e5
negative eigenvalues: 0 of 5   (PSD at the optimum; damping still needed far from it)
gradient (at a nearby point):  [-412.1, -5.57, -9.64, 2.73, 3.86]
curvature correlation matrix:
   Ab      eField  tran    long    life
[[ 1.00    0.13    0.06   -0.18   -0.20 ]   Ab       (nearly decoupled)
 [ 0.13    1.00    0.84   -0.72   -0.69 ]   eField
 [ 0.06    0.84    1.00   -0.77   -0.67 ]   tran_diff
 [-0.18   -0.72   -0.77    1.00    0.87 ]   long_diff
 [-0.20   -0.69   -0.67    0.87    1.00 ]]  lifetime
LM Newton step:  [ 0.0029, 0.0022, 0.0061, -0.0016, -0.0021]   (balanced across params)
neg-grad (Adam-like), normalized:  [1.0, 0.014, 0.023, -0.007, -0.009]  (Ab-dominated)
```

**Interpretation — this is the crux of why GN should help:**
- **Condition number ≈ 2.1×10⁵** even after normalization → gradient descent (Adam) mixes fast
  and slow directions badly.
- **PSD at the optimum** (0 negative eigenvalues) → exact Newton is safe near the solution, but
  LM damping is still needed farther out where the non-smooth FEE ops can flip curvature signs.
- **eField/tran_diff/long_diff/lifetime are strongly coupled** (|corr| 0.67–0.87) — they all move
  collected/diffused charge — while **Ab is nearly decoupled** but **dominates the gradient**
  (−412 vs a few).
- Consequence: **Adam's step is almost entirely along Ab and under-serves the coupled 4-param
  subspace** — exactly where lifetime sits, which is why the full fit lands **lifetime +27%**.
  The **LM step rebalances** to comparable magnitudes across all five by dividing out curvature.

This concretely predicts GN/LM will (a) converge in far fewer iterations and (b) remove the
lifetime bias. The prototype (`GaussNewtonCalibFitter`, `--fit_type gn_calib`) tests exactly
this against Adam on ground-truth tracks.

---

## Empirical verdict (32-batch head-to-head, ground-truth tracks, calibration-only)

> **CORRECTION / RETRACTION**: the table below compared MISMATCHED datasets.
> `example_run.py` silently capped `max_nbatch` at `iterations`, so Adam (3000 it) fit
> 32 batches / 2048 cm while exact-GN and GGN (40 it) fit only 13 batches / 929 cm, the
> polish run 4 batches, and the covariance job 1 batch. The cap is now exempted for
> `gn_calib` (full-batch) and the comparison re-run on matched 32-batch data
> (jobs 31145475/76/77). The garden-path *behavior* (monotonic descent into the bounds)
> was real on GN's own surface, but the cross-optimizer loss numbers below are
> apples-to-oranges — superseded by the matched re-run.

Jobs 31140129 (exact-GN), 31140130 (GGN/Fisher), 31140131 (Adam); identical data/init/loss.

| | exact-GN (23 it, 20 min) | GGN (12 it, 33 min) | Adam (3000 it, 46 min) |
|---|---|---|---|
| full-batch loss | 63,328 (stalled) | 63,831 (stalled) | ~43,660 |
| Ab | −10.2% | −6.3% | **−0.00%** |
| eField | −3.9% | −4.1% | **−0.10%** |
| lifetime | **+149% (sigmoid cap)** | +144% | −6.8% |
| long_diff | **−83% (floor)** | −83% | +5.2% |

**The prediction in the appendix above was WRONG.** Both curvature methods descend
*monotonically* (every LM step accepted) straight down the shallow curved
lifetime/long_diff/tran_diff valley and pin those params at their sigmoid bounds — a
*garden path*: each big tangent step lowers the loss, but the valley bends, and the method
never follows the bend. Adam's small per-batch steps track the curve to the true minimum.
The same over-commitment pathology killed the L-BFGS geometry block — this is now seen twice
and is a property of this likelihood's geometry, not of one optimizer.

Mechanical footnotes: (1) an LM damping bug (no floor on the Marquardt diagonal scaling at
sigmoid saturation → |step| up to 3e6) was found and fixed — the verdict above is post-fix;
(2) `dataio` starves small `max_nbatch` values (loads `len×(n+2)` cm then drops trajectories
longer than `len`; n=8 → only 2 usable batches) — use `max_nbatch=100` → 32 batches at 50 cm.

**Fisher/GGN validation** (batch 0, at truth): F is PSD, same conditioning structure as H,
eigenvalues 1.4–4× larger, LM-step cosine 0.59; eval 0.3 s cached vs ~19 s exact-H
(compile 155 s once per batch).

## GGN compile-time fix: data-as-args (measured)

The per-batch Fisher compile storm (32 × ~155 s) was caused by baking each batch's `tracks`
array into the jitted closure as XLA constants — 32 distinct programs. Since `dataio` already
pads all batches to one global shape, passing `tracks` as a TRACED argument (jit cache keyed
on `(usize, roi, shape)`, ROI unified to the global max, plus a pre-scan pass so the ROI max
is final before jitting) collapses this to ~2 compiles (one per unique-pixel bucket, 128/256).

Measured (job 31155083 vs 31145476, same 32-batch config):
- baked: **>3.5 h elapsed, zero iterations reached** (cancelled)
- data-as-args: **11 min to first iteration, full 40-iter run in 58 min** (~70 s/full-batch iter)

This same pattern (traced data + static shape buckets) is the fix for the L-BFGS geometry
compile storm as well.

## Remaining niche being tested: GN as final polish + uncertainties

Near the optimum the surface is a PSD bowl (probe) — Newton's home turf. Job 31144534
(`GNMODE=gn_polish`) starts exact-GN from the Adam 32-batch endpoint: if it converges in a
few iterations it gives (a) last-mile precision beyond Adam's noise floor and (b) the 5×5
H⁻¹ covariance of the calibration parameters for free. That — not global optimization — is
the realistic role for GN in this fit.
