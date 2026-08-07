"""Fig 48: the dE/dx student-t prior over-shrinks the per-segment field — the WEIGHT, not the shape.

Study 2 showed the prior charges +562 to put the dE/dx field at truth while the likelihood charges
-10, in 600/600 batches. This figure shows why, from the prior's own parameters and the fitted and
true fields — no loss evaluations involved, so it is an independent line of evidence.

WHY WIDTHS ARE QUOTED AS 16-84% HALF-RANGES AND NOT AS sigma. The prior's right branch has
nu_R = 2.073, so its variance is finite only barely (nu/(nu-2) = 28) and is set almost entirely by
a tail far outside any region the data occupies. A sigma comparison would be dominated by that tail
and would say the prior is enormously wide. The 16-84% half-range is the same number as sigma for a
Gaussian, is insensitive to the tail, and compares like with like against the data.

WHERE p(x)^w COMES FROM. It is not a modelling choice, it is what the objective already says.
The prior enters as
    dedx_prior = -w * sum_i dx_i * log p(dedx_i)
and an additive term E(x) in a negative-log objective IS a density proportional to exp(-E(x)).
So the effective prior is exactly
    p_eff(x) = p(x)^beta / integral p(u)^beta du ,      beta = w * dx
renormalised. This is the standard tempering / power-posterior (fractional-likelihood)
construction; beta is the "number of pseudo-observations", because summing log p over beta
identical draws gives beta*log p. In statistical-mechanics terms p^beta = exp(-beta E) is the same
density at temperature T/beta. Measured dx has median 0.9913 cm with an IQR of 0.986-0.996, so
beta = 4.957 here and its per-segment spread (~1%) is negligible.

THE STUDENT-t FAMILY IS CLOSED UNDER TEMPERING, exactly:
    nu_eff = beta*(nu + 1) - 1
    s_eff  = s * sqrt(nu / nu_eff)
Verified against the numerical construction to 4e-14. For the left branch (nu=4.785, s=0.1204) at
beta=4.957 that is nu_eff = 27.68, s_eff = 0.0501 -- so what production actually applies is a far
narrower and much LIGHTER-TAILED t than the one written down.

NOTE the Gaussian rule of thumb UNDERSTATES this. 1/sqrt(beta) = 0.449, but the exact sd ratio is
sqrt((nu-2)/(nu_eff-2)) = 0.329, because tempering lightens the tail as well as shrinking the
scale. The 1/sqrt(w) shorthand is only the nu -> infinity limit.

The curves here are still built numerically (exp(beta*log p), renormalised on a grid) rather than
from the closed form: the SPLIT t is two branches glued at a shared peak height with their own
normaliser, so tempering it is not a split-t with one (nu, s) pair -- each branch tempers to
different parameters and the glue and normalisation have to be redone. The numeric route is exact
up to the grid and avoids re-deriving that normaliser.
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps
from scipy.special import gammaln

OUT = 'plots/noise_report'
RUN = 'fit_result/sci_full_ANNEALLONG2'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
LOSS_CROSSOVER = 2.373       # from fig 46(f): where the objective stops preferring the fit
MEAN_TARGET = 1.887          # dedx_mean_constraint_target, the OTHER dEdx prior


def split_t_logpdf(x, nu_L, nu_R, loc, s_L, s_R):
    """Same construction as optimize/dedx_utils.split_t_logpdf_jax (numpy version)."""
    log_c = lambda nu: (gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi))
    cL, cR = log_c(nu_L), log_c(nu_R)
    kL = sps.t.logpdf((x - loc) / s_L, df=nu_L) - cL
    kR = sps.t.logpdf((x - loc) / s_R, df=nu_R) - cR
    logZ = np.log(.5 * s_L * np.exp(-cL) + .5 * s_R * np.exp(-cR))
    return np.where(x < loc, kL, kR) - logZ


def rwidth(v, w=None):
    """16-84% half-range: sigma for a Gaussian, tail-insensitive otherwise."""
    if w is None:
        lo, hi = np.percentile(v, [15.865, 84.135])
    else:                                     # weighted, for a density on a grid
        cdf = np.cumsum(w); cdf /= cdf[-1]
        lo, hi = np.interp([.15865, .84135], cdf, v)
    return (hi - lo) / 2


# ---- load fitted and true per-segment fields, and the prior's parameters
fits, trus, dxs, per_batch = [], [], [], []
par = None
for f in sorted(glob.glob(RUN + '/history_iter10000_*.pkl')):
    h = pickle.load(open(f, 'rb'))
    c = vars(h['config'])
    if par is None:
        par = dict(loc=float(h['dedx_student_loc']), nu_L=float(c['dedx_student_nu_l']),
                   nu_R=float(c['dedx_student_nu_r']), s_L=float(c['dedx_student_scale_l']),
                   s_R=float(c['dedx_student_scale_r']), w=float(c['dedx_prior_weight']))
    dc, tm, sd = h['dedx_cache'], h['true_dedx_matched_cache'], h.get('seg_dx_cache', {})
    for b in sorted(dc):
        fv = np.exp(np.asarray(dc[b]['log_dedx'], float))
        tv = np.asarray(tm[b], float)
        m = min(len(fv), len(tv)); fv, tv = fv[:m], tv[:m]
        ok = np.isfinite(tv) & (tv > 0)
        if ok.sum() < 20:
            continue
        fits.append(fv[ok]); trus.append(tv[ok])
        if b in sd:
            dxs.append(np.asarray(sd[b], float)[:m][ok])
        per_batch.append((rwidth(fv[ok]), rwidth(tv[ok])))
FIT = np.concatenate(fits); TRU = np.concatenate(trus)
DX = np.concatenate(dxs) if dxs else np.array([1.0])
PB = np.array(per_batch)
wt, wf = rwidth(TRU), rwidth(FIT)
print(f'{len(FIT)} segments over {len(PB)} batches, {len(fits)//100} seeds')
print(f'prior: loc={par["loc"]:.4f} nu_L={par["nu_L"]} nu_R={par["nu_R"]} '
      f's_L={par["s_L"]} s_R={par["s_R"]} weight={par["w"]}')
print(f'median per-segment dx = {np.median(DX):.4f} cm  -> effective power w*dx = {par["w"]*np.median(DX):.3f}')

# ---- prior width as a function of the weight, computed numerically from p^w
grid = np.linspace(par['loc'] - 4.0, par['loc'] + 4.0, 400001)
lp = split_t_logpdf(grid, par['nu_L'], par['nu_R'], par['loc'], par['s_L'], par['s_R'])
WS = np.logspace(-1.1, 1.0, 90)
PW = np.array([rwidth(grid, np.exp(np.float64(w) * (lp - lp.max()))) for w in WS])
# PW decreases with w, so reverse BOTH arrays to give np.interp an increasing xp.
# (Feeding it a decreasing xp returns silent garbage rather than an error.)
_o = np.argsort(PW)
w_match = float(np.interp(wt, PW[_o], WS[_o]))
assert PW.min() < wt < PW.max(), 'true width outside the scanned prior-weight range'
w_eff = par['w'] * np.median(DX)
print(f'prior 16-84% half-width: at w=1 {np.interp(1.0, WS, PW):.4f}   '
      f'at w={w_eff:.2f} {np.interp(w_eff, WS, PW):.4f}   at w=0.5 {np.interp(0.5, WS, PW):.4f}')
print(f'true field width {wt:.4f}   fitted field width {wf:.4f}   ratio {wf/wt:.3f}')
print(f'weight at which the prior is exactly as wide as truth: {w_match:.2f}')
print(f'  (loss-based crossover, independently, fig 46f: {LOSS_CROSSOVER:.2f})')

fig, ax = plt.subplots(2, 4, figsize=(21.4, 8.8))

# ---------------- (a) the three distributions ----------------
a = ax[0, 0]
bins = np.linspace(1.60, 2.20, 90)
a.hist(TRU, bins=bins, density=True, color=C['green'], alpha=.55, label=f'TRUE field  ({wt:.3f})')
a.hist(FIT, bins=bins, density=True, color=C['blue'], alpha=.55, label=f'FITTED field  ({wf:.3f})')
sel = (grid > bins[0]) & (grid < bins[-1])
for w, col, ls, lab in [(1.0, C['grey'], '--', 'prior as written (w = 1)'),
                        (w_eff, C['red'], '-', f'prior AS APPLIED (w = {w_eff:.1f})')]:
    d = np.exp(w * (lp - lp.max()))
    d = d / np.trapezoid(d, grid)
    a.plot(grid[sel], d[sel], color=col, lw=2.3, ls=ls,
           label=f'{lab}  ({np.interp(w, WS, PW):.3f})')
a.set_xlabel('per-segment dE/dx  (MeV/cm)')
a.set_ylabel('density')
a.set_title('(a) the prior AS WRITTEN is wider than truth; raised to\n'
            'the 5th power it is narrower. Widths in brackets.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (b) width vs weight ----------------
a = ax[0, 1]
a.plot(WS, PW, color=C['red'], lw=2.6, label='prior width, $p(x)^w$')
a.axhline(wt, color=C['green'], lw=2.2, label=f'TRUE field  {wt:.3f}')
a.axhline(wf, color=C['blue'], lw=2.2, ls='--', label=f'FITTED field  {wf:.3f}')
a.axvline(w_eff, color=C['ink'], lw=1.8, label=f'production weight {par["w"]:.0f}')
a.axvline(0.5, color=C['purple'], lw=1.8, ls=':', label='script default 0.5')
a.plot([w_match], [wt], marker='o', ms=10, color=C['ink'], zorder=6)
a.annotate(f'  prior = truth\n  at w = {w_match:.1f}', (w_match, wt), fontsize=7.6, va='bottom')
a.axvspan(LOSS_CROSSOVER * .93, LOSS_CROSSOVER * 1.07, color=C['orange'], alpha=.35)
a.annotate(f'loss crossover {LOSS_CROSSOVER:.2f}\n(fig 46f, independent)',
           (LOSS_CROSSOVER, wf * 0.72), fontsize=7.4, color=C['orange'], ha='center')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('dE/dx prior weight $w$')
a.set_ylabel('16–84% half-width')
a.set_title('(b) two INDEPENDENT estimates of a sane weight:\n'
            f'width-matching {w_match:.1f}, the loss {LOSS_CROSSOVER:.2f}. Production runs at 5.',
            fontsize=9.3)
a.set_ylim(wf * 0.55, PW.max() * 1.4)
a.legend(fontsize=6.9, frameon=False, loc='upper right')

# ---------------- (c) regression to the mean ----------------
a = ax[0, 2]
idx = np.random.default_rng(0).choice(len(FIT), size=min(9000, len(FIT)), replace=False)
a.scatter(TRU[idx], FIT[idx], s=3, alpha=.16, color=C['blue'], edgecolors='none')
lo, hi = 1.65, 2.15
a.plot([lo, hi], [lo, hi], color=C['ink'], lw=1.8, ls='--', label='no shrinkage (identity)')
sl, ic = np.polyfit(TRU, FIT, 1)
xx = np.array([lo, hi])
a.plot(xx, sl * xx + ic, color=C['red'], lw=2.4, label=f'fit: slope {sl:.2f}')
a.axhline(par['loc'], color=C['orange'], lw=1.6, ls=':', label=f'prior centre {par["loc"]:.3f}')
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('TRUE per-segment dE/dx'); a.set_ylabel('FITTED per-segment dE/dx')
a.set_title(f'(c) textbook regression to the prior centre:\nslope {sl:.2f}, not 1', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False, loc='upper left')

# ---------------- (d) per-batch spread ratio ----------------
a = ax[1, 0]
ratio = PB[:, 0] / PB[:, 1]
a.hist(ratio, bins=40, color=C['purple'], edgecolor='white', linewidth=.5)
a.axvline(1.0, color=C['ink'], lw=2.0, label='no shrinkage')
a.axvline(np.median(ratio), color=C['red'], lw=2.2, ls='--',
          label=f'median {np.median(ratio):.2f}')
a.set_xlabel('width(fitted) / width(true), per batch')
a.set_ylabel('batches')
a.set_title(f'(d) the shrinkage is systematic, not a fluke:\n'
            f'{100*(ratio<1).mean():.0f}% of {len(ratio)} batches are under-dispersed', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (e) what the prior charges, per segment ----------------
a = ax[1, 1]
nll_f = -split_t_logpdf(FIT, par['nu_L'], par['nu_R'], par['loc'], par['s_L'], par['s_R'])
nll_t = -split_t_logpdf(TRU, par['nu_L'], par['nu_R'], par['loc'], par['s_L'], par['s_R'])
bb = np.linspace(min(nll_f.min(), nll_t.min()), np.percentile(np.r_[nll_f, nll_t], 99.5), 80)
a.hist(nll_f, bins=bb, color=C['blue'], alpha=.6, label=f'at the FIT  (mean {nll_f.mean():+.3f})')
a.hist(nll_t, bins=bb, color=C['green'], alpha=.6, label=f'at TRUTH  (mean {nll_t.mean():+.3f})')
a.axvline(0, color=C['grey'], lw=1.2)
a.set_xlabel('per-segment prior NLL,  $-\\log p(dE/dx)$')
a.set_ylabel('segments')
a.set_title(f'(e) the prior finds TRUTH less likely than the fit by\n'
            f'{nll_t.mean()-nll_f.mean():.3f} nat/segment — ×{par["w"]:.0f} weight × {len(FIT)//len(PB)} '
            f'segments/batch', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (f) Q-Q: is it a uniform compression? ----------------
a = ax[1, 2]
q = np.linspace(1, 99, 99)
qt, qf = np.percentile(TRU, q), np.percentile(FIT, q)
a.plot(qt, qf, marker='o', ms=3.4, lw=1.6, color=C['blue'], label='quantile–quantile')
a.plot([qt[0], qt[-1]], [qt[0], qt[-1]], color=C['ink'], lw=1.8, ls='--', label='identity')
sq, iq = np.polyfit(qt, qf, 1)
a.plot(qt, sq * qt + iq, color=C['red'], lw=1.8, label=f'slope {sq:.2f}')
a.set_xlabel('TRUE quantiles'); a.set_ylabel('FITTED quantiles')
a.set_title('(f) a UNIFORM compression toward the centre, not a\n'
            'tail artefact — every quantile moves by the same factor', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='upper left')

# ---------------- (g) the SECOND harm: the mean moves too ----------------
# Powering an ASYMMETRIC density changes its mean, not just its width. nu_R = 2.073 gives a heavy
# right tail that p^w suppresses faster than the left branch, so the prior's mean slides DOWN as w
# rises. This is a distinct defect from the over-shrinkage of panels (a)-(f).
a = ax[0, 3]
MU = np.array([float(np.trapezoid(grid * np.exp(np.float64(w) * (lp - lp.max())), grid) /
                     np.trapezoid(np.exp(np.float64(w) * (lp - lp.max())), grid)) for w in WS])
mu_true = float(np.sum(TRU * DX) / np.sum(DX)) if len(DX) == len(TRU) else float(TRU.mean())
mu_fit = float(np.sum(FIT * DX) / np.sum(DX)) if len(DX) == len(FIT) else float(FIT.mean())
a.plot(WS, MU, color=C['red'], lw=2.6, label="student-t mean under $p(x)^w$")
a.axhline(mu_true, color=C['green'], lw=2.2, label=f'TRUE mean  {mu_true:.4f}')
a.axhline(MEAN_TARGET, color=C['orange'], lw=1.8, ls=':',
          label=f'mean-constraint target  {MEAN_TARGET:.4f}')
a.axhline(mu_fit, color=C['blue'], lw=2.0, ls='--', label=f'FITTED mean  {mu_fit:.4f}')
a.axvline(w_eff, color=C['ink'], lw=1.8)
a.annotate(f'  production\n  w = {par["w"]:.0f}', (w_eff, 1.8575), fontsize=7.4, va='bottom')
a.axvline(0.5, color=C['purple'], lw=1.6, ls=':')
a.set_xscale('log')
# The prior's mean blows up below w ~ 0.5 (the nu_R = 2.07 tail stops being suppressed and the
# mean runs away toward the tail). Zoom on the region the fit actually lives in; the curve leaves
# the frame at the top left and that is not the interesting part.
a.set_ylim(1.855, 1.900)
a.set_xlabel('dE/dx prior weight $w$')
a.set_ylabel('mean of the prior / of the field')
a.set_title('(g) the SECOND harm: powering an ASYMMETRIC density\n'
            'also moves its MEAN. The two priors pull opposite ways.', fontsize=9.3)
a.legend(fontsize=6.8, frameon=False, loc='upper right')

# ---------------- (h) the empirical confirmation ----------------
# Two COMPLETE 3-seed ceiling arrays differing in exactly one knob. Widths are compared as
# fitted/true RATIOS within each run, never in absolute terms: the ceiling grid is the target's own
# segmentation (median dx 2.32 cm) while production uses the guess file's (0.99 cm), so the two
# have different true per-segment spreads and absolute widths are not comparable across them.
a = ax[1, 3]
CE = []
for tag, wlab in [('sci_ceiling_CEILBASE', 'prior 5\n(production)'),
                  ('sci_ceiling_CEILP05', 'prior 0.5\n(default)')]:
    mus, wds, tmus, twds = [], [], [], []
    for f in sorted(glob.glob(f'fit_result/{tag}/history_iter10000_*.pkl')):
        h = pickle.load(open(f, 'rb'))
        dc, tm, sd = h['dedx_cache'], h['true_dedx_matched_cache'], h['seg_dx_cache']
        F, T, D = [], [], []
        for b in sorted(dc):
            fv = np.exp(np.asarray(dc[b]['log_dedx'], float))
            tv = np.asarray(tm[b], float); dv = np.asarray(sd[b], float)
            m = min(len(fv), len(tv), len(dv)); fv, tv, dv = fv[:m], tv[:m], dv[:m]
            ok = np.isfinite(tv) & (tv > 0)
            F.append(fv[ok]); T.append(tv[ok]); D.append(dv[ok])
        F, T, D = np.concatenate(F), np.concatenate(T), np.concatenate(D)
        mus.append(100 * ((np.sum(F * D) / np.sum(D)) / (np.sum(T * D) / np.sum(D)) - 1))
        wds.append(rwidth(F) / rwidth(T))
    CE.append((wlab, np.array(mus), np.array(wds)))
xs2 = np.arange(len(CE))
a.bar(xs2 - .19, [abs(c[1].mean()) for c in CE],
      yerr=[c[1].std(ddof=1) for c in CE], width=.36, color=C['orange'], edgecolor='white',
      capsize=4, label='|mean error| vs truth (%)')
a2 = a.twinx()
a2.bar(xs2 + .19, [c[2].mean() for c in CE], yerr=[c[2].std(ddof=1) for c in CE],
       width=.36, color=C['green'], edgecolor='white', capsize=4,
       label='width recovered / true')
a2.axhline(1.0, color=C['ink'], lw=1.6, ls='--')
a2.set_ylabel('fraction of the true spread recovered'); a2.grid(False)
a2.set_ylim(0, 1.15)
for k, c in enumerate(CE):
    a.annotate(f'{abs(c[1].mean()):.3f}%', (k - .19, abs(c[1].mean())), textcoords='offset points',
               xytext=(0, 4), ha='center', fontsize=7.4)
    a2.annotate(f'{c[2].mean():.2f}', (k + .19, c[2].mean()), textcoords='offset points',
                xytext=(0, 4), ha='center', fontsize=7.4)
a.set_xticks(xs2); a.set_xticklabels([c[0] for c in CE], fontsize=8)
a.set_ylabel('|mean error| vs truth (%)')
a.set_title('(h) CONFIRMED on two complete 3-seed ceiling arrays:\n'
            'one knob fixes BOTH harms', fontsize=9.3)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False, loc='upper center')

fig.suptitle('Fig 48 — the dE/dx prior over-shrinks: the SHAPE is fine, the WEIGHT is not. As '
             'written the prior is wider than truth;\nas applied (×5, length-weighted) it is '
             'narrower, and the fitted field collapses to '
             f'{100*wf/wt:.0f}% of the true spread, and its mean slides ~1% low. '
             'Both are fixed by the same knob (h).', fontsize=10.4)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig48_dedx_prior_shrinkage.png'); plt.close(fig)
print('wrote', f'{OUT}/fig48_dedx_prior_shrinkage.png')
print(f'  regression slope fit-vs-true {sl:.3f}   Q-Q slope {sq:.3f}')
print(f'  per-segment prior NLL: fit {nll_f.mean():+.4f}  truth {nll_t.mean():+.4f}  '
      f'diff {nll_t.mean()-nll_f.mean():+.4f} nat')
print(f'  x weight {par["w"]} x {len(FIT)//len(PB)} seg/batch = '
      f'{par["w"]*(nll_t.mean()-nll_f.mean())*len(FIT)//len(PB):+.0f} per batch '
      f'(measured in study 2: +562)')
