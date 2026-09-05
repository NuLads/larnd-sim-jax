"""Fig 71 — the first real error bars: the calibration covariance, and the A_b<->lifetime degeneracy.

WHY A HESSIAN AND NOT A PROFILE. Profiling by re-fitting failed by a factor 2 105 (fig 70): each
point is an independent stochastic re-optimisation of ~9 000 nuisance parameters and where it lands
varies more in loss than the signal. A Hessian is evaluated at one point, so it has no
landing-point noise at all. Computed by central finite differences of the GRADIENT -- the
forward-over-reverse HVP path OOMs at 400 cm batches, asking 8.3 GB per allocation -- at two step
sizes, agreeing to 7-11%, which is the systematic on everything below.

PANEL (a) IS THE VALIDATION, and it is the reason to believe the rest. The fig-58 loss scan measured
the same curvature by an entirely different route: finite differences of the LOSS along each
parameter, on a +-5% grid, with the other four calibration parameters HELD FIXED. Computing the
matching quantity from the Hessian -- 1/sqrt(H_ii), which is also "others fixed" -- the two agree to
1-2% on lifetime (1.00), transverse diffusion (1.02) and A_b (0.98), and to 11% on longitudinal
diffusion. Two independent methods, one number.

THE TWO PLACES THEY DISAGREE ARE THE SCAN'S FAULT, and it is checkable. Repeating the scan's
parabola fit on +-5%, +-10% and +-20% baselines: for lifetime, transverse and longitudinal diffusion
the answer is stable (0.258/0.255/0.245 for lifetime), so the loss is genuinely quadratic there. For
E field it moves 0.0027 -> 0.0051 -> 0.0098 and for A_b 0.0166 -> 0.0187 -> 0.0366, i.e. a factor
3.6 and 2.2 across baselines. Those two are so tightly constrained that a +-5% grid sits ~1 700 and
~170 sigma from the minimum, far outside the quadratic region, and the scan is measuring an average
chord curvature rather than a local one. THE E FIELD FLOOR IN THE HANDOFF WAS 7x TOO LARGE.

PANEL (b) IS THE NEW PHYSICS. The calibration correlation matrix has never been measured in this
campaign, and it is not diagonal: RECOMBINATION AND LIFETIME ARE 71% ANTICORRELATED. Both control
how much charge arrives at the pixels -- A_b at production, lifetime during drift -- so the fit can
trade one against the other almost freely. That is the degeneracy the whole campaign has been
fighting, now quantified rather than inferred.

PANEL (c) IS WHAT IT COSTS. Letting the other four calibration parameters move inflates each 1-sigma:
lifetime x1.67, A_b x1.66, and the rest by 1.04-1.24. The two inflated by 1.67 are exactly the two
joined by the -0.71 correlation. So the honest error bar on lifetime is not 0.26% but 0.43%.

WHAT THIS IS AND IS NOT. Still CONDITIONAL on geometry and dEdx, which are held at their fitted
values, so these remain LOWER BOUNDS on the full profile uncertainty -- but they are now bounds with
the calibration correlations included, and validated against an independent method. Marginalising
the ~9 000 nuisances as well needs the Schur complement, which is the next build.
"""
import glob, os, pickle, sys, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from optimize.ranges import ranges
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = R.P
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
SCAN = {'Ab': 0.0166, 'eField': 0.0027, 'tran_diff': 0.3729, 'long_diff': 0.7218, 'lifetime': 0.2581}
QUAD = {'Ab': False, 'eField': False, 'tran_diff': True, 'long_diff': True, 'lifetime': True}
ACHIEVED = {'Ab': 0.269, 'eField': 0.030, 'tran_diff': 1.460, 'long_diff': 7.414, 'lifetime': 1.651}


def get(tag, s, hk='hessian_fd_0.001'):
    h = pickle.load(open(glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')[0], 'rb'))
    scale = float(h['normalization_scale_sigmoid'])
    Rs = h['truthpoint']
    H = np.zeros((5, 5))
    for r in Rs:
        d = r['points']['P0_fit'][hk]
        H += np.array([[d[kb][ka] for kb in P] for ka in P])
    H = 0.5 * (H + H.T)
    calib = Rs[0]['points']['P0_fit']['calib']
    J = []
    for k in P:
        lo, hi = ranges[k]['min'], ranges[k]['max']
        p = (float(calib[k]) - lo) / (hi - lo)
        J.append((hi - lo) * scale * p * (1 - p) / abs(float(calib[k])))
    return H, np.array(J)


HB = sum(get('HESSBT2', s)[0] for s in (0, 1)) / 2
JB = get('HESSBT2', 0)[1]
cond = 100 * JB / np.sqrt(np.diag(HB))
marg = 100 * JB * np.sqrt(np.diag(np.linalg.inv(HB)))
Cov = np.linalg.inv(HB)
d = np.sqrt(np.diag(Cov)); COR = Cov / np.outer(d, d)

fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.0))

# ---------------- (a) validation ----------------
a = ax[0, 0]
xs = np.arange(5); w = .38
a.bar(xs - .19, [SCAN[p] for p in P], width=w, color=C['grey'], edgecolor='white',
      label='fig 58 loss scan (others fixed)')
a.bar(xs + .19, cond, width=w, color=C['blue'], edgecolor='white',
      label='Hessian, 1/√H$_{ii}$ (others fixed)')
for i, p in enumerate(P):
    r = cond[i] / SCAN[p]
    a.text(i, max(SCAN[p], cond[i]) * 1.35, f'×{r:.2f}', ha='center', fontsize=8,
           color=C['ink'] if QUAD[p] else C['red'])
    if not QUAD[p]:
        a.text(i, max(SCAN[p], cond[i]) * 3.2, 'scan\nnon-quadratic', ha='center', fontsize=6.4,
               color=C['red'])
a.set_yscale('log'); a.set_xticks(xs)
a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('1σ, % of the fitted value')
a.set_title('(a) VALIDATION: two independent methods, one number.\n'
            'Agreement to 1–2% wherever the scan is trustworthy.', fontsize=9.4)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (b) the correlation matrix ----------------
a = ax[0, 1]
im = a.imshow(COR, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(5):
    for j in range(5):
        a.text(j, i, f'{COR[i,j]:.2f}', ha='center', va='center', fontsize=8.6,
               color='white' if abs(COR[i, j]) > .55 else C['ink'])
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=25, ha='right')
a.set_yticks(xs); a.set_yticklabels([PL[p] for p in P], fontsize=8)
a.grid(False)
fig.colorbar(im, ax=a, fraction=.046)
a.set_title('(b) THE NEW RESULT: A$_b$ and LIFETIME are 71% ANTI-\n'
            'correlated — both set how much charge arrives.', fontsize=9.4)

# ---------------- (c) what marginalising costs ----------------
a = ax[1, 0]
a.bar(xs - .19, cond, width=w, color=C['blue'], edgecolor='white',
      label='other calibration params FIXED')
a.bar(xs + .19, marg, width=w, color=C['red'], edgecolor='white',
      label='other calibration params FREE')
for i in range(5):
    a.text(i + .19, marg[i] * 1.3, f'×{marg[i]/cond[i]:.2f}', ha='center', fontsize=8.2,
           color=C['red'])
a.set_yscale('log'); a.set_xticks(xs)
a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('1σ, % of the fitted value')
a.set_title('(c) WHAT THE CORRELATION COSTS. Lifetime ×1.67 and A$_b$ ×1.66 —\n'
            'exactly the pair joined by −0.71. Honest lifetime 1σ = 0.43%, not 0.26%.', fontsize=9.4)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (d) achieved vs the floor, corrected ----------------
a = ax[1, 1]
ach = np.array([ACHIEVED[p] for p in P])
a.bar(xs - .19, marg, width=w, color=C['green'], edgecolor='white',
      label='1σ floor (this work, correlations included)')
a.bar(xs + .19, ach, width=w, color=C['red'], edgecolor='white',
      label='BT2 achieved |error| (n = 6)')
for i in range(5):
    a.text(i + .19, ach[i] * 1.3, f'×{ach[i]/marg[i]:.0f}', ha='center', fontsize=8.4,
           color=C['red'])
a.set_yscale('log'); a.set_xticks(xs)
a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('% of the fitted value')
a.set_title('(d) HOW FAR WE ARE FROM IT, corrected: 3–75×, not the 4–16×\n'
            'previously quoted. E field is 75× off, not 11×.', fontsize=9.4)
a.legend(fontsize=7.0, frameon=False, loc='upper left')

fig.suptitle('Fig 71 — the first real error bars. The calibration Hessian reproduces the '
             'independent fig-58 scan to 1–2% wherever that scan is\ntrustworthy, and shows the two '
             'places it is not. It then delivers what no scan could: the calibration correlation '
             'matrix, in which\nRECOMBINATION AND LIFETIME ARE 71% ANTICORRELATED — the degeneracy '
             'this campaign has been fighting, finally quantified.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig71_hessian.png'); plt.close(fig)
print('wrote', f'{OUT}/fig71_hessian.png')
print(f'\n{"param":11s} {"scan":>9s} {"H fixed":>9s} {"ratio":>6s} {"H free":>9s} {"infl":>6s} '
      f'{"achieved":>9s} {"x floor":>8s}')
for i, p in enumerate(P):
    print(f'{p:11s} {SCAN[p]:8.4f}% {cond[i]:8.4f}% {cond[i]/SCAN[p]:6.2f} {marg[i]:8.4f}% '
          f'{marg[i]/cond[i]:6.2f} {ACHIEVED[p]:8.3f}% {ACHIEVED[p]/marg[i]:8.1f}')
