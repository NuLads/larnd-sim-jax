# Audit of the differentiable LArTPC calibration campaign — claim chain review

Repo: `/sdf/group/neutrino/pgranger/larnd-sim-jax`, branch `dedx_soft_barrier`. Audit date 2026-09-23.
Sources: git log since 2026-06-01 (241 commits), root/`docs/*.md` reports (11 synthesis docs read
in full by a dedicated worker, plus NOISE_CAMPAIGN_REPORT/DIGEST, OPTIMIZATION_FINDINGS_SUMMARY,
DEDX_PRIOR_DOSSIER, GN_REPORT_SUMMARY, S4_RANKED_PLAN read directly), Claude memory notes, the prior
session transcript (JSONL, 40701 turns, mined by a dedicated worker), and provenance blocks sampled
live from `fit_result/sci_full_*/*.pkl`.

---

## 1. Bug timeline

Ordered by fix/discovery date. "Invalidates" = what upstream conclusions are put in doubt.

| # | Bug | Discovered/fixed | Commit | What it broke |
|---|---|---|---|---|
| 1 | Chain-freeze gated the wrong thing | 2026-07-22 | `714bd64` | "freeze was silently reverting geometry to nominal" — explicitly invalidated the geomfreeze/two-phase/ceiling-repro test family run before this date |
| 2 | `DEDX_FREEZE_ITER` hard-coded no-op | 2026-07-22 | `29849c8` | Hard-coded 2600 > `ITERATIONS`=2500 → freeze never fired; any pre-fix "dE/dx frozen" arm never actually froze |
| 3 | Two-pass chain init keyed by list position, not `track_id` | 2026-07-27 | `5860795` | Warm-start / two-pass geometry tests before this date silently mismatched tracks between passes |
| 4 | `chain_update_freq` fed calib nominal geometry every iter (warp bug) | 2026-07-27 | `fa8e53d` | Chain-position tests run with non-default `chain_update_freq` before this date used the wrong nominal geometry. (A related, earlier form of this — `chain_update_freq=5` throttling starving geometry — is separately documented in `docs/joint_fit_v7_v8_findings.md`, 5 Jul 2026: switching to `freq=1` took position 260µm→63µm and long_diff −79%→−11%.) |
| 5 | Noise mismatch: noiseless targets vs noise-marginalizing fit model | discovered/fixed ~2026-07-24–27 | `ed6c10f` (+`fa8e53d` noise-ON default) | "all prior campaigns had noiseless targets vs noise-smeared guess." Measured effect: S2 `long_diff` flips **−19.2% → +12.6%** once made consistent (`MEASUREMENTS_DOSSIER.md`). Everything before ≈Jul 24–27 inherits this risk, including `BRIEFING_FOR_REVIEWERS.md`'s S1–S4 ladder (dated Jul 29, just after the fix — should be clean, but close enough to the boundary to double check which exact run it cites) |
| 6 | `out_label` collisions silently overwrote checkpoints | fixed 2026-07-22, still flagged live | `8b0fedb`; also `GN_REPORT_SUMMARY.md` §"traps" | "Missed three times; cost ~17 A100-hours on a scan that silently overwrote itself and left files resembling a clean seed set" |
| 7 | `max_nbatch` not what you get | documented, no single fix commit found | `GN_REPORT_SUMMARY.md`, `S4_RANKED_PLAN.md` | "Production requests 100 batches and gets **13**" (`SCILEN=50` loads `len×(nbatch+2)` cm, then drops trajectories over `max_batch_len`) — a possibly still-open discrepancy between requested and delivered data volume |
| 8 | Epoch-indexed LR decay ("NB400" trap) | documented Jul–Aug 2026 | `OPTIMIZATION_FINDINGS_SUMMARY.md` §1, `HANDOFF.md` trap T1 | Raising `max_nbatch` 4× silently left the calib LR 1250× higher at readout (lifetime +52%, position +3545µm) — read as a physics failure, was a schedule artifact. Also "invalidated a published ×2.3 improvement" per HANDOFF T1 |
| 9 | Calibration `--lr` hard-coded `1e-1` in launcher for the whole campaign | found Sep 2026, fixed | `ca7585d` (2026-09-14) | Only the LR **decay** was ever tuned; the **peak** LR was never actually under `SCICLR` control campaign-wide until this fix |
| 10 | Dead GPU (`sdfampere038`) reported `COMPLETED 0:0` | 2026-08-12 (2nd occurrence; 1st was 2026-08-08) | `77b652b` | JAX saw no CUDA device, preflight caught it, but container `exit 1` didn't propagate to SLURM; "one array silently 2/3 full again" (also HANDOFF trap T6) |
| 11 | LSCAN "objective-limited" conclusion was a misread conditional minimum | corrected 2026-08-12 | `a86e607` (corrects `688d302`) | Reversed the prior claim: "objective PREFERS truth; the error is reachability" (fig 66). Root cause: a 1-D conditional scan (nuisances pinned at fitted values) was read as a joint minimum (HANDOFF trap T2: "this one also produced a published conclusion that was exactly backwards") |
| 12 | Geometry/dE/dx "information ratio 28×" measurement error | corrected 2026-08-11 | `71ebd00` | Compared two different fits' α-ranges; corrected to 5.1× on one fit (qualitative conclusion unchanged) |
| 13 | dE/dx prior "wrong segmentation" diagnosis flip | 2026-08-07 | `3d29b88` | The student-t prior was believed mis-shaped; actually right for the *target* grid, wrong for the *production* grid |
| 14 | `dedx_mean_penalty` explanation was backwards (fig 47) | 2026-08-07 | `589545f` | First version said truth pays MORE because it's closer to target (wrong); real mechanism is spread (6× wider true variation than the constraint allows) |
| 15 | `w = 1/(2σ²)` derivation for `dedx_mean_constraint_weight` declared invalid | documented, no single commit; `GN_REPORT_SUMMARY.md` §6 | — | Formula predicts w=464 for a 46cm/batch config; 464 loses badly to 3000. "2500 is a defensible operating point, not a derived optimum" |
| 16 | Shuffle "breaks geometry" finding was code drift, not physics | documented (memory: `shuffle_investigation.md`; `HANDOFF.md` trap T7) | — | Comparison was across two `git_dirty` runs on different days — `--shuffle_bt` is fine |
| 17 | **`data_seed` silent no-op — the single largest invalidation** | 2026-09-08 | `d26523d` | Load-time pre-trim ran before random event selection; with the campaign's 400cm×100-batch config every `data_seed` drew from the *same* pre-trimmed pool → bit-identical batches across "different" seeds. **All fits before this date, across the entire campaign, share ONE physical track sample.** The bug is in the code path (order of pre-trim vs. selection), not in argv content, so runs are subject to it regardless of what `--data_seed` value was passed. Directly verified live: `sci_full_BT2/*seed0.pkl` provenance timestamp = **2026-08-07 13:10** (git `04a0346`, dirty), `sci_full_TS1/*seed0.pkl` = **2026-08-12 03:02** (git `9948c88`, dirty) — both run `--data_seed 1` explicitly in argv but both predate the fix by 4–5 weeks, so both are confirmed subject to the bug. (An earlier version of this note said BT2 launched "2026-08-10" — corrected here to the directly-verified 2026-08-07.) |
| 18 | fig74 mean-target convention error (sim-input vs target-matched truth) | 2026-09-09 | `8ae761c` | Two truth conventions conflated; headline excess corrected from +0.53% to +0.10% |
| 19 | Ab init silently clamped out of range | 2026-09-16 | `6ae877e` (fig 88) | `SCIINIT` values outside `ranges.py` bounds are silently clamped — `CINITA8`'s Ab init 0.92 (max 0.90) never moved from the bound, turning an intended ablation into an accidental one |
| 20 | fig94 "66% of true spread" shrinkage-direction reasoning backwards | 2026-09-17 | `fbe5cbf` | A noisy per-segment estimate is *over*-dispersed, not under; correct diagnosis is Neyman–Scott MAP shrinkage, cost quantified at −10.4 pts of lifetime |
| 21 | fig95 MAE↔width correlation inflated by including truth-pinned arms | 2026-09-18 | `3360aaa` | r dropped +0.67 → +0.29 once truth arms (MAE≡0 by construction) were excluded; "aim for 100% width" refuted |
| 22 | SCITGT missing colon→space translation | 2026-09-18 | `7c1290a` | All 24 calibration-curve jobs of that batch died in ~25s; resubmitted |
| 23 | Multi-pass truth-free bootstrap "converges to truth" — retracted | 2026-09-14 | `22cd525` (confirms/retracts `1093a89`, `d374cf0`) | Rep 8's 16.4%→11.7%→0.65% ladder was "a lucky draw from a ±8% landing distribution re-sampled by every re-anneal"; confirmation suite falsified it (ensemble mean 11.7→13.6→13.7%, worsens 2/3 replicas) |
| 24 | Transient `squeue` failure stranded held jobs | 2026-09-20 | `fd1338e` | Throttler read an empty `squeue` (itself failed) as "all released" and exited, leaving 5 held jobs stuck |
| 25 | Linked-diffusion "lifetime gain" — retracted | 2026-09-23 | `f6cc0f0` (on branch `link_diffusion_test`, **not merged into `dedx_soft_barrier`**) | Both diffusions land ~+56% off; likelihood can't distinguish the linked vs control arm (−19±54 nats). Lives on an unmerged side-branch — a reader of the current `dedx_soft_barrier` docs would not see this retraction |
| 26 | **×0.40 dE/dx-prior-scale "recipe" — retracted on fresh seeds** (today's top commit) | 2026-09-23 | `9a45383` | Found/validated only on tuning seeds 3/7/8 (mean\|err\| 7.26%→2.95%); on 5 fresh seeds the gain collapses to −0.34±0.65 (3/5, n.s.) — "mostly regression to the mean," since the tuning seeds happened to have unusually bad (7.26%) baselines vs 3.42% for the fresh five |
| 27 | Track-selection CLI args were dead code (`live_selection` defaulted False) | documented 2026-07-02 | `docs/joint_fit_v6_analysis.md` | 33/33 tracks used regardless of selection flags in the v6-era runs; invalidated an earlier "narrow drift lever-arm" lifetime hypothesis |
| 28 | S2 loader target-file mismatch (own-truth/target bookkeeping) | found via JSONL transcript mining (not in git log directly) | — | Caused a false "the July-11 result was wrong" retraction that itself had to be retracted — a meta-retraction; traced to comparing against the wrong target file |

**Cross-cutting provenance finding (not in the docs):** every checkpoint sampled from `fit_result/`
(`sci_full_BT2`, `sci_full_DSREP1`, `sci_full_NARR040D3`, `sci_full_CINITA8`) has `git_dirty: True`
in its provenance block. The team's own trap list (T7, "never compare runs across days on a dirty
tree") is being violated by default across the whole campaign — dirty-tree state is the norm, not
the exception.

---

## 2. Claim ledger (most load-bearing, ~30 claims)

Status codes: **STANDING-SOLID** · **STANDING-WEAK** (small n / tuning-seed-only / pre-fix, not
re-validated) · **CONTRADICTED** (a sibling claim in current docs says the opposite) ·
**INVALIDATED-BY-BUG** · **RETRACTED**.

| # | Claim | Evidence | Run date | Doc(s) | Status |
|---|---|---|---|---|---|
| 1 | +8.8%/+8.85±1.18% lifetime bias is real, sample-independent, survives 4 truth points | 16 replicas, fig 85/92, grand mean +8.0% | post-Sep-8 | RECENT_TESTS_SUMMARY | **STANDING-SOLID** — largest-n claim in the campaign |
| 2 | Bias is NOT optimiser-limited / NOT prior-miscalibration | figs 77,78,90,91,95 | Sept 2026 | RECENT_TESTS_SUMMARY, IN_DEPTH_REVIEW | **STANDING-SOLID** |
| 3 | Objective is "blind" to the bias: loss↔lifetime r=+0.02, best-loss picks worse lifetime 3/4 seeds | figs 82/83 | Sept 2026 | RECENT_TESTS_SUMMARY | **STANDING-SOLID but in tension with #4** — see §3(a) |
| 4 | Objective "prefers truth" (fig 66): all 3 blocks at truth beats fit on 7/7 arm-seeds; gap closing 9.4%→6.6%→2.5% | fig 66, corrected `a86e607` | corrected 2026-08-12 | HANDOFF, FINDINGS_SYNTHESIS | **STANDING-SOLID** as global 3-block statement — but note the "prefers truth" result is geometry-dominated: calibration-alone-at-truth makes loss *worse* (+0.18%, 0/7) and dE/dx-alone-at-truth also worse (+2.79%, 0/7); only geometry-alone (−7.00%, 7/7) carries the net result |
| 5 | ×0.40 dE/dx-prior-scale recipe cuts calib error 7.26%→2.95%, matches truth-pinning | figs 97/99, seeds 3/7/8 only | 2026-09-20/22 | RECENT_TESTS_SUMMARY | **RETRACTED next day** (`9a45383`) — fresh 5 seeds: −0.34±0.65 (3/5, n.s.) |
| 6 | Revised ×0.40 verdict: plausible small lifetime-only improvement, unproven | −2.14±1.66 (4/5), ~1.3σ | 2026-09-23 | IN_DEPTH_REVIEW | **STANDING-WEAK**, self-flagged as suggestive only |
| 7 | Freezing dE/dx mid-fit is the most robust positive result (−31µm position) | n=3, seeds 3/7/8 | 2026-09-23 (today) | IN_DEPTH_REVIEW | **STANDING-WEAK** — same tuning-seed-only pattern that sank claim #5, not yet checked on fresh seeds |
| 8 | "Both nuisances at truth" lifetime ceiling = −3.74±0.39% | seeds 3/7/8, `IN_DEPTH_REVIEW`/`RECENT_TESTS_SUMMARY` | 2026-09-22/23 | both | **STANDING-WEAK**; possible sign tension with claim #1's sibling fig-73 ceiling number — see §3(d) |
| 9 | TS1 (freeze calib, re-anneal nuisances, 10k it) confirmed best production recipe, n=6/6, position −55.1µm, loss −756 | figs 67/68/72 | `TS1` seed0 checkpoint directly verified launched **2026-08-12 03:02** (git `9948c88`, dirty); `BT2` base **2026-08-07 13:10** (git `04a0346`, dirty) — both **pre**-data_seed-fix (2026-09-08) and pre-hardcoded-LR-fix (2026-09-14); both argvs also show `--lr 1e-1` literally passed, consistent with bug #9 (LR peak not actually under `SCICLR` control) | HANDOFF, FINDINGS_SYNTHESIS | **STANDING-WEAK** — confirmed provenance shows both BT2 and TS1 subject to bug #17 and #9; "its calibration is BT2's, frozen — the fit gain, not a calibration gain" per the docs' own wording; never re-run on independent track samples after either fix |
| 10 | TS1's 4000-iteration "free" shortcut | proposed in HANDOFF as untested idea | 8 Sep | HANDOFF → FINDINGS_SYNTHESIS | **RETRACTED** — falsified: +10µm/+378 loss (3/3) |
| 11 | Two-stage (TS1+TS2) recipe as production | fig 68 | 8 Sep | HANDOFF, FINDINGS_SYNTHESIS | **NARROWED** — stage 2 demoted to optional (trades, doesn't win) |
| 12 | "A_b and E field are solved" (0.27%/0.03% mean error, n=6) | `BT2`, HANDOFF §2 table | pre-Sep-8 | HANDOFF | **INVALIDATED-BY-BUG / CONTRADICTED** — the 0.27% A_b bias number is explicitly retracted in the same document family: "shown (fig 73, DSREP) to be common-mode from the single shared track sample — on random samples A_b is unbiased at 0.04%"; separately E field's "solved" framing is walked back once its OWN floor (~0.0005%) is compared: "~×60 — the real headroom... nothing has ever been tuned for it because 0.03% looked 'solved'" |
| 13 | "Lifetime is nearly at its statistical floor" (×1.5–1.8×, 1.65% achieved, 0.9–1.1% floor) | fig 72, `BT2`, n=6 | pre-Sep-8 | FINDINGS_SYNTHESIS scoreboard | **CONTRADICTED within the same document** — its own 9-Sep addendum states the achieved numbers "inherit a sample-luck caveat," and the +8.6±1.6% bias (claim #1's precursor) directly undercuts the "nearly at the limit" framing; the scoreboard headline was never updated |
| 14 | "4–16× from statistical floor" (all calib params) | fig 58 | pre-Sep-8 | HANDOFF | **SUPERSEDED** in FINDINGS_SYNTHESIS's own corrections ledger to "3.3–71× conditional" / lifetime specifically "×1.5–1.8" |
| 15 | Longitudinal diffusion is not standalone: conditional optimum runs 10–44% below truth, moves further from truth as nuisances improve | fig 72c | pre-Sep-8 | FINDINGS_SYNTHESIS | **STANDING-WEAK** (n unspecified, pre-fix) |
| 16 | A_b↔lifetime degeneracy central: r=−0.71 conditional → −0.83 marginalised | figs 71/72 | pre-Sep-8 | FINDINGS_SYNTHESIS | **STANDING-WEAK** (pre-fix, not re-run) |
| 17 | Re-fit profiling for uncertainties fails (×2105 too noisy); curvature (FD Hessian) validated to 1–2% against independent scan instead | figs 70/71 | — | FINDINGS_SYNTHESIS, HANDOFF | **STANDING-SOLID** as a *method* result; feeds the compromised floor numbers above |
| 18 | GN/second-order verdict: calib-5-param GN robust-not-fast; dE/dx GN ties Adam (not worth deploying); geometry Adam beats GN on accuracy (51µm vs 107µm) AND speed (~400×) | many independent tests | Jul–Aug 2026 | GN_REPORT_SUMMARY, S4_RANKED_PLAN | **STANDING-SOLID**, consistent across 3+ docs |
| 19 | GN/Fisher (GGN) curvature differs from exact Hessian by 1.75–4.27×, gap GROWS with batches — do not use as preconditioner | — | Jul–Aug 2026 | GN_REPORT_SUMMARY, S4_RANKED_PLAN, MEASUREMENTS_DOSSIER | **STANDING-SOLID** |
| 20 | dE/dx mean-constraint target 1.887 correct (+0.09–0.10% off true 1.8853/1.8869); weight ~13× too stiff at 1e5 | 3 independent computations, 602k–3.03M segments | — | GN_REPORT_SUMMARY, DEDX_PRIOR_DOSSIER | **STANDING-SOLID** |
| 21 | `dedx_mean_constraint_weight`=2500 is a defensible operating point (not derived) | GN toy (18 arms) + BT arms (6 seeds) | — | DEDX_PRIOR_DOSSIER | **STANDING-WEAK**, self-caveated by authors — see bug #15 |
| 22 | (scale,shape) dE/dx reparametrisation: best-anywhere toy result (lifetime +18.1%→−2.4%) does NOT reproduce at production scale (toy 25.8→6.2 vs production 20.9→**24.3**, worse) | 2×2 factorial, n=3 | Jul–Aug 2026 | GN_REPORT_SUMMARY | **RETRACTED at production scale**, explicitly labelled "underpowered, not a clean negative" |
| 23 | 51µm position floor (noiseless, position-only, calib+dE/dx at truth) | consistent across `GN_REPORT.md`, `GN_REPORT_SUMMARY.md`, `MEASUREMENTS_DOSSIER.md` | multiple re-derivations agree | 3 docs + memory note | **STANDING-SOLID**, but scope-limited — see §3(e); not the same quantity as the 80–165µm full-production position residual |
| 24 | Geometry was "optimiser-limited," fixing it is the single largest win (`G20K`, −57µm/−8.7% loss, 3/3) | fig 61 | 2026-08-12, pre-fix | OPTIMIZATION_FINDINGS_SUMMARY | **STANDING-WEAK** — pre-data_seed-fix; superseded operationally by the cheaper TS1 but not itself re-run post-fix |
| 25 | ANNEAL fix is the single biggest optimiser win (lifetime +62.7±49.5%→−2.1±4.5%) | figs 7/16 | 2026-07-27 | BRIEFING_FOR_REVIEWERS, OPTIMIZATION_FINDINGS_SUMMARY | **STANDING-SOLID** — every later "solid" recipe (BT2, TS1, G20K) builds on it and continues to behave sanely |
| 26 | Drift-axis penalty on geometry refuted (legitimate track structure, not a lifetime-contaminant channel) | 12-seed dose-response | Jul 2026 | BRIEFING_FOR_REVIEWERS | **STANDING-SOLID**, mild residual noise-mismatch-window timing risk |
| 27 | The dE/dx prior-tuning programme is "finished" (weight/shape/scale/reparam/mean-target/drift all null) | fig 95, mostly seeds 3/7/8 | Sept 2026 | DEDX_PRIOR_DOSSIER | **STANDING-WEAK** — rests on the same n=3 tuning-seed base that failed fresh-seed validation for the adjacent ×0.40 claim |
| 28 | Methodological rule: "any tuned quantity must be validated on seeds not used in the tuning" — 4 prior claims this month (multi-pass bootstrap, init ladder, diffusion-mechanism test, ×0.40 recipe) held at small n and reversed at proper n | — | 2026-09-23 | IN_DEPTH_REVIEW, RECENT_TESTS_SUMMARY | **STANDING-SOLID as a meta-finding**, and itself evidence for §4 below |
| 29 | 80% of campaign runs were still moving when read out (plateau audit) | fig 4 | — | OPTIMIZATION_FINDINGS_SUMMARY | **STANDING-SOLID** as a methodology warning; implies unquantified plateau risk on many of the above n-seed claims |
| 30 | Enabling `dedx_mean_constraint_weight`=1e5 (was silently 0) fixed worst 1-D joint-fit failures (lifetime +40%→+4%) | 1-D fits, other 8 params pinned at truth | 2026-07-05 | docs/joint_fit_v7_v8_findings.md | **STANDING-WEAK** — 1-D conditional-scan evidence only (the exact trap T2 pattern that later reversed fig 66) |

---

## 3. Special-attention pairs (deep dive)

**(a) "Objective prefers truth" (fig 66) vs "objective is blind" (`IN_DEPTH_REVIEW.md`).**
Not a strict contradiction, and partly explained by the docs' own numbers, but never explicitly
reconciled in one place. Fig 66's "prefers truth" result is dominated entirely by **geometry**:
moving calibration alone to truth makes the loss *worse* (+0.18%, 0/7 seeds better) and moving
dE/dx alone to truth also makes it worse (+2.79%, 0/7) — "the dE/dx prior is a genuine objective
bias... [it] displaces its own block's minimum" (HANDOFF finding 5). Only geometry-alone-at-truth
(−7.00%, 7/7) and all-three-together (−5.60%, 7/7) beat the fit. So "the objective prefers truth"
is, in its own source data, a geometry-driven headline, not evidence that the likelihood favors
true calibration or true dE/dx. The later "objective is blind" claim (loss↔lifetime r=+0.02) and
"likelihood worsens monotonically as physics improves" (IN_DEPTH_REVIEW, dE/dx-prior-scale ladder)
describe the realistic regime — comparing different *fitted* solutions — where tightening the dE/dx
prior improves lifetime accuracy but worsens the likelihood (9714→10286), consistent with, not
contradicting, "dE/dx prior is a genuine objective bias." **Recommend an explicit reconciling
sentence be added to the docs**; as written, a reader who only sees fig 66's headline and the
"blind"/"disagrees with every physics metric" headline would reasonably read them as opposed.

**(b) Lifetime statistical floor (0.9–1.1%, "nearly at its limit") vs +8.6%/+8.8% bias.**
Confirmed **self-contradiction inside `FINDINGS_SYNTHESIS.md`** (claim #13 above): the scoreboard
presents "lifetime nearly at the limit" as the headline "load-bearing result," while the same
document's own 9-September addendum — two sections later — states the achieved numbers "inherit a
sample-luck caveat" because of the `data_seed` bug, and that on 8 independent track samples the
procedure is biased +8.6±1.6%. The floor number is a *precision* statement (Hessian-derived
variance at one sample/truth point); the +8.6% is an *accuracy/bias* statement (systematic offset,
driven by sample-composition variance across genuinely independent draws). Both can be technically
true, but the scoreboard headline was never revised to reflect the addendum — this is the clearest
single documentation defect found in the audit.

**(c) TS1 as production recipe.** Adopted and reconfirmed with increasing seed count (n=3→n=6/6)
across HANDOFF→FINDINGS_SYNTHESIS, with one explicit walk-back (the proposed 4000-iteration "free"
shortcut, falsified) and one narrowing (stage 2 demoted to optional). No later document reverses
TS1 itself, but the two most recent docs (RECENT_TESTS_SUMMARY, IN_DEPTH_REVIEW, 22–23 Sep) do not
mention TS1 by name at all — their "×1.0 production" baseline's relationship to TS1/BT2 is not
stated. Directly verified via live checkpoint load: `sci_full_TS1/*seed0.pkl` (git `9948c88`, dirty
tree) has provenance timestamp **2026-08-12 03:02**, and its base `sci_full_BT2/*seed0.pkl` (git
`04a0346`, dirty tree) timestamp **2026-08-07 13:10** — both 4–5 weeks before the `data_seed` fix
(2026-09-08) and the hardcoded-LR fix (2026-09-14). So TS1's "n=6/6" superiority claim is unconfirmed
on genuine track-sample diversity and was built/confirmed under the hardcoded-LR bug too; only its
narrower "fit gain, not calibration gain" framing (its own docs' wording) is defensible as-is. TS1
has not been re-run since either fix, and has effectively fallen out of the September documentation
set without a formal retraction.

**(d) Noise-at-truth toy (+14%) vs production "both nuisances at truth" ceiling (−3.74%).**
A dedicated search of the 11 "recent/synthesis" docs plus NOISE_CAMPAIGN_REPORT/DIGEST and
MEASUREMENTS_DOSSIER found **no exact "+14% lifetime bias at truth with a noise-ON toy" statement**
matching the task's phrasing. Closest matches: (i) a **calibration-init** test (`CINIT*`, fig 84)
where starting calibration "high" (+14% from truth) lands at +6.8% lifetime on one replica — a
different experiment (init sensitivity, not a noise-consistency toy); (ii) `MEASUREMENTS_DOSSIER.md`'s
actual noise-consistency toy, where making the noise model consistent flips S2 `long_diff` from
−19.2% to **+12.6%** (a sign flip, not "+14%", and on `long_diff` not `lifetime`); (iii) GN_REPORT's
Adam-vs-GN toy: "the best GN arm went *below* the truth loss while landing **14% wrong on lifetime**"
— a noisy-objective/optimizer artifact, not a noise-ON-target claim. The confirmed "−3.74%" ceiling
(both nuisance blocks pinned at truth, calibration fit, seeds 3/7/8, `IN_DEPTH_REVIEW.md`/
`RECENT_TESTS_SUMMARY.md`, 2026-09-22/23) is real and exact. Separately, there IS a genuine,
currently-unresolved tension worth flagging: fig 73 (`OPTIMIZATION_FINDINGS_SUMMARY.md` §6,
post-data_seed-fix, 8 replicas) reports lifetime **+8.6±1.6%** for "the same fit on 8 independent
track samples at pinned truth" — wording that sounds like the same "nuisances at truth, calibration
fit" ceiling scenario as IN_DEPTH_REVIEW's "both nuis@TRUTH" row, which instead reports **−3.74±0.39%**
(opposite sign). If these are the same measurement, that is a real, unresolved sign contradiction
between two "ceiling" numbers in currently-standing docs. If they differ (e.g. one fixed truth point
+ 8 track samples vs. three different seeds each with their own truth point, or "pinned truth" in
fig 73 actually meaning something else, such as calibration also pinned), the discrepancy needs an
explicit note distinguishing the setups — none was found in any document read. **Recommend this be
checked directly against the underlying scan code/config before either number is cited further.**

**(e) 51 µm position floor.** No contradiction found — consistently the *noiseless, position-only*
number (calibration and dE/dx fixed at truth) across `GN_REPORT.md`, `GN_REPORT_SUMMARY.md`,
`MEASUREMENTS_DOSSIER.md`, and the memory note `position_fit_accuracy_floor.md`. It is a genuinely
different quantity from the *full noisy production* position residual under TS1 (105µm achieved,
FINDINGS_SYNTHESIS states a "representative floor ~80µm," explicitly caveated as geometry-conditional
and an upper bound pending marginalisation). No document explicitly cross-references the two numbers
side by side, which risks future citation conflation even though neither figure is wrong.

**(f) "Ab/eField are solved."** The specific *bias* framing (+0.27%/+0.03% as a "reproducible"
per-run signal, `HANDOFF.md`) is invalidated for A_b: `FINDINGS_SYNTHESIS.md`'s own corrections
ledger states the "+0.27%, 6/6-positive A_b signal... was later shown (fig 73, DSREP) to be
common-mode from the single shared track sample — on random samples A_b is unbiased at the 0.04%
level." E field's "solved" framing is separately walked back once compared to its *own* statistical
floor: FINDINGS_SYNTHESIS calls out "~×60 — the real headroom... nothing has ever been tuned for it
because 0.03% looked 'solved'." The underlying *relative* accuracy claim (A_b/eField recover far
better than lifetime/long_diff, and are decoupled from both nuisance blocks, ±0.5%/±0.05% across a
17-fold prior-scale change) is not contradicted and recurs in post-fix docs including `IN_DEPTH_REVIEW.md`
(2026-09-23).

---

## 4. Retraction pattern

From direct git-log inspection (28 explicit correction/retraction/fix commits with scientific
content, §1) and from mining the prior session transcript (`8d010f6a-...jsonl`, 40701 turns):

- **Pushback → immediate revision (user-forced):** 51 raw keyword hits ("wrong"=41, "mistake"=4,
  "no,"=3, "re-check"=2, "incorrect"=1); after filtering to scientifically substantive cases, ~10–15
  real instances — e.g. a reviewer directly disputing a "prefers truth" panel reading that reversed
  a load-bearing conclusion cited repeatedly afterward (the very commit that became `a86e607`/fig 66);
  a reviewer challenge that reversed the attributed root cause of the S4 lifetime failure from
  geometry accuracy to the calibration LR schedule.
- **Self-caught retractions (agent re-reads own output/logs and corrects):** 174 raw keyword hits
  ("correction"=115, "retract"=33, "I was wrong"=14, "let me correct"=7, "walk back"=3), dominated by
  routine low-stakes bookkeeping, but at least 15 scientifically material corrections identified,
  clustering into four repeatable failure modes:
  1. **Reporting from non-converged/partial/mid-transient data** (pass2r8 misjudged mid-transient;
     the position-floor claim briefly overturned then re-confirmed; multi-pass bootstrap's early
     passes read as success before the confirmation suite ran).
  2. **Comparing runs across a dirty/differing git tree** (the shuffle investigation — a full
     false-finding retraction; live checkpoint sampling in this audit shows dirty-tree runs remain
     the campaign's norm, not the exception, so this class of error remains structurally possible).
  3. **Silent config/code bugs** (`data_seed`, an S2-loader target-file mismatch that itself caused
     a meta-retraction of a correct earlier result, un-annealed LR from a `max_nbatch` change,
     Ab-init clamp, SCITGT colon bug) — the single largest class by both commit count and cost
     (whole studies / 24-job arrays lost in one case).
  4. **Small-n (n=2/n=3) claims not surviving replication at larger n** — recurs at least 4 times in
     the final three weeks alone: the diffusion-mechanism test (n=3), the calibration-init monotone
     law (n=6, only 2/6 seeds actually monotone), the linked-diffusion "gain" (n=3, retracted on an
     unmerged branch), and — as of today's commit — the ×0.40 dE/dx-prior recipe itself (n=3 tuning
     seeds vs. n=5 fresh seeds). This is an **explicitly acknowledged, recurring, not-yet-fixed**
     institutional failure mode: `IN_DEPTH_REVIEW.md`'s own closing section states "Four results this
     month held at small n and reversed when tested properly... the rule going forward is explicit:
     any tuned quantity must be validated on seeds not used in the tuning, before it is reported" —
     written in the same commit that reports the fourth instance of exactly that failure.
- Two additional cases documented as "a failed query read as a negative answer": a target-verification
  step that reported pass while checking zero runs, and a job watcher that read a `squeue` outage as
  "wave complete" (bug #24 above is the git-log-visible instance of this pattern).
- At least one retraction (`f6cc0f0`, linked-diffusion) exists only on an **unmerged branch**
  (`link_diffusion_test`) and would not be visible to a reader of the current `dedx_soft_barrier` docs.

---

## 5. Open questions the docs themselves flag as untested

- **Full calibration-block profile scan** (not just 1-D conditional slices) at TS1/G20K/BT2 —
  flagged in `HANDOFF.md` as "the question that decides whether lifetime has a residual bias,"
  never reported as run in any doc read.
- **Geometry marginalisation** — needs `remat` through `simulate_signals`; all quoted statistical
  floors (incl. the 0.9–1.1% lifetime floor) are explicitly geometry-conditional lower bounds.
- **Real-data validation** — repeatedly named as the next qualitatively new information; the entire
  programme so far is simulation-on-simulation.
- **Forward-model charge residual vs. drift at truth** — the "cleanest unrun diagnostic" for the
  residual rate-error mechanism (`RECENT_TESTS_SUMMARY.md` §6); not yet executed as of the latest doc.
- **FEE stochastic-vs-analytic charge-bias claim** — explicitly marked "UNVERIFIED... do not cite it"
  in both `S4_RANKED_PLAN.md` and `GN_REPORT_SUMMARY.md`; probe returned COULD-NOT-TEST.
- **`dedx_use_split_t` (asymmetric prior) vs. a symmetric-t alternative** — "never tested," flagged
  as a one-line, never-run comparison in `DEDX_PRIOR_DOSSIER.md`.
- **Sample-independence of the refit dE/dx prior shape** — cross-validated within-sample only; an
  independent shape source (e.g. a dE/dx flow model) was proposed but not built.
- **E field's ~×60 measured headroom** — flagged as untouched/never tuned for.
- **Tying the two diffusion constants via the mobility relation** — the one attempt
  (`link_diffusion_test` branch) was itself retracted (bug #25) and the branch is unmerged.
- **Whether the ×0.40 (or any) dE/dx-prior-scale gain is real off tuning seeds** — actively open as
  of today's commit; only a weak, 1.3σ, lifetime-only signal survives on fresh seeds.
- **Whether TS1/BT2 conclusions survive on genuinely independent track samples** (post-`data_seed`-
  fix) — not found tested in any document read.
- **The mechanism for the residual lifetime-vs-drift attenuation-rate error** — "what we still do
  not have is the *cause*" (`RECENT_TESTS_SUMMARY.md`), with the long_diff-overestimate hypothesis
  at only 1.5σ, 2/3 seeds, one seed flatly reversing.

---

## 5b. Additional bug found only via JSONL transcript mining (not in §1, no git-log fix commit isolated)

- **Chain-position (geometry) Adam optimizer ran with a fixed, non-decaying LR while detector
  calibration params decayed** — sustained a persistent position↔calibration limit cycle. Fixed by
  commit `a35e5de` (adds chain LR decay), dated ~2026-06-21, i.e. **before** the audit's `--since`
  window but directly named by the agent as the root cause of three early "breakthrough" claims:
  "all four N-D runs oscillate persistently, and none converged the full calibration. My three
  'breakthrough' reports [nd_v9@400cm, nd_v10@800cm, fastdecay@1000cm] were each a lucky snapshot
  caught at a favorable phase of the oscillation." This bug predates and is logically prior to bug
  #7 in §1 (epoch-indexed calibration LR decay) — the campaign had two independent, sequential LR-
  scheduling bugs (chain-side in June, calibration-side in July), each separately caught only after
  a published result was retracted.

## 6. Raw pushback / self-correction episodes mined from the prior session transcript (JSONL)

A second research pass mined the full 40,701-line prior-session transcript (798 human messages,
2026-06-18 → 2026-09-23) directly for pushback/correction language and reconstructed 36 episodes
with quoted trigger text, the claim challenged, the resolution, and (interpolated where not exact)
a date. Full verbatim text — including exact quoted user pushback lines and assistant retractions —
is preserved in this session's transcript and in `jsonl_mining_raw.txt`/`mine.py` in this directory;
condensed below for the report. Selected highlights not already covered in §1/§2/§4:

- **E7 (2026-07-07)**, user: *"Did you actually compute these metrics on a significant sample of the
  last iterations and not only on the last point, to not be influenced by possible fast
  oscillations?"* — exposed that a geometry metric was a single end-of-run snapshot, not
  time-averaged; fixed via `pos_residual_iter`. Generalizes to a standing methodological risk: any
  metric read as a single endpoint (not a plateau-averaged or convergence-checked value) is suspect
  — this is the same class of error later named explicitly as a guardrail ("do NOT read endpoints of
  unconverged oscillating fits — burned twice," `S4_RANKED_PLAN.md`) and recurred as late as claim
  #22 in §2 (kb-fixing N-D result, iter-400 snapshot judged "too optimistic," oscillated through
  iter 1200).
- **E10 (2026-07-10)**, user: *"Can you please make the relevant studies show your claims about the
  degeneracy valley, to make sure it exists before actually thinking about how to fix it"* — a
  (lifetime, long_diff) degeneracy mechanism had been asserted before being measured; the resulting
  study was itself later flagged "REFUTED by its own test" when pinning long_diff at truth did not
  collapse the lifetime error as predicted.
- **E16 (2026-07-24)**, user: *"Ok, so that's a big issue. I need you to submit new jobs with the
  electronics noise ON to be sure of where we stand..."* — directly triggered discovery of the
  per-seed differing-truth comparison bug (bug §1 item 3/5 region) and the observation "the entire
  weekly-review body of work has the same issue."
- **E23 (2026-07-30)**, user: *"...in other fits we used to see lifetime being under-estimated and
  not over-estimated. The minimum might be in the 'wrong location'"* — broke a "consistent sign
  across three independent subsets" argument (subsets were nested, not independent,
  `dataio.py:365`) and produced the **first sighting of the eventual `data_seed` no-op**, six weeks
  before it was properly chased down and fixed (2026-09-08): *"My DATASEED test silently did nothing.
  All five 'disjoint' seeds returned identical numbers."* This was flagged but NOT fixed at the time
  — a six-week gap between first sighting and fix, during which the campaign's highest-volume
  production runs (BT2, TS1, G20K) were all launched and never revisited.
- **E28 (~2026-08-05/06)**, self-caught, no direct trigger: *"the production objective has no
  per-hit charge residual"* (published in fig 44, §6q, and two TL;DRs) was wrong — the charge term
  is present, just not separately logged. Directly relevant to open question §4 item 1 (forward-model
  charge residual at truth): the mechanism exists and is unmeasured, not absent.
- **E32 (2026-09-08)**, user: *"I think one thing that I would like to be done is investigate the
  actual status of the bias. Do you [know] if we ever tried to run the same fits with different input
  data (by changing the data seed for instance)?"* — this single question triggered the `data_seed`
  no-op discovery (§1 bug #17), the single largest invalidation in the audit. Notable: the bug had
  already surfaced once, unfixed, in E23 (2026-07-30) — it took a second, independent user prompt six
  weeks later to actually chase it to a fix.
- **E36 (2026-09-23, today)**, user: *"It's quite visible that the diffusion parameters are now really
  wrong in the linked case, no?"* — directly triggered the linked-diffusion retraction (`f6cc0f0`),
  which the agent's own reply says "cascaded directly into" re-examining and retracting the ×0.40
  recipe the same day (`9a45383`). I.e. **both of today's two retractions were user-pushback-driven**,
  not self-caught.

**Pattern across all 36 episodes:** a large fraction of the campaign's most consequential corrections
(E16/data_seed-truth-per-seed, E21/padding-mask, E23+E32/data_seed no-op, E36/both of today's
retractions) were triggered by direct user pushback/questions, not self-discovered by the agent in
the course of routine work. Several bugs (data_seed chief among them) were flagged once, not acted
on, and only fixed after a second, independent prompt — i.e. detection and remediation are not the
same event, and the gap between them (up to 6 weeks for data_seed) is itself a source of continued
invalidation risk for anything run in that window.

---

## 6b. Files/paths of record (scratchpad)

- `/lscratch/.../scratchpad/D_claims/gitlog_oneline.txt` — full `git log --since=2026-06-01` (241
  commits, one line each) used to build §1.
- `/lscratch/.../scratchpad/D_claims/jsonl_mining_raw.txt` and `mine.py` — raw grep/keyword-hit output
  and mining script from the prior session transcript.
- `/lscratch/.../scratchpad/D_claims/pushback_episodes_full.md` — the full 36-episode (E1-E36)
  verbatim pushback/self-correction appendix underlying §6 above, with exact quoted trigger lines,
  line numbers, and resolutions.
- Checkpoint provenance samples pulled live via `/sdf/group/neutrino/pgranger/conda/bin/python` from
  `fit_result/sci_full_BT2`, `sci_full_DSREP1`, `sci_full_NARR040D3`, `sci_full_CINITA8` — all four
  show `git_dirty: True`; `sci_full_BT2` (launched 2026-08-10) confirmed missing `--random_nevents`
  in its stored `argv`, i.e. subject to the `data_seed` bug.
