"""Fig 72 — the uncertainty ladder: each nuisance level climbed, lifetime approaches its floor.

THE LADDER. Fig 71 gave the conditional covariance (everything but the probed parameter fixed) and
the calibration-marginalised one. This adds the dEdx marginalisation -- exact, via the block's
banded+rank-1 structure and a direct solve (no CG; the code documents three failed routes). Each
rung marginalises more nuisances, each 1-sigma can only grow, and the achieved error stays put:

    lifetime at BT2:  0.30% (conditional) -> 0.58% (calib corr) -> 0.9-1.1% (+dEdx)  achieved 1.65%

(the +dEdx floor is quoted as a band: 0.91% from the raw summed S, 1.11% after the PSD projection
that clips one small negative eigenvalue -- the difference IS the method systematic, and both ends
tell the same story). So lifetime sits x1.5-1.8 from the dEdx-marginalised floor, NOT the x6.4 the
first conditional estimate suggested, and geometry -- the one block still unmarginalised, and the
most information-bearing -- is still to come. The plausible endpoint is that the achieved lifetime error is close to its
statistical limit given the nuisance burden. That reframes "improve the optimiser" into "reduce the
nuisance burden or add external constraints".

THE CORRELATION DEEPENS ON 4 OF 5 JOBS after the PSD projection (-0.78 mean -> -0.83 mean; the
one exception, TS1 s0 at -0.70, is the seed whose clipped eigenvalue lies closest to the A_b/
lifetime subspace, so its post-projection correlation is the least reliable of the five). dEdx is the charge-normalisation intermediary: part of
the A_b<->lifetime trade routes THROUGH the dEdx field, and marginalising it hands that freedom to
the calibration pair.

HONESTY PANEL: 4 of 5 summed S matrices carry one small negative eigenvalue, living in the
diffusion subspace (FD band noise integrated over 100 batches). A_b, E field and lifetime -- whose
entries are stable across all 5 jobs -- are quoted; the two diffusions are marked unstable and are
not.
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from optimize.ranges import ranges

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
ACH = {'Ab': 0.269, 'eField': 0.030, 'tran_diff': 1.460, 'long_diff': 7.414, 'lifetime': 1.651}


def analyse(tag, s):
    h = pickle.load(open(glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')[0], 'rb'))
    Rs = h['truthpoint']; scale = float(h['normalization_scale_sigmoid'])
    Hc = np.zeros((5, 5)); D = np.zeros((5, 5))
    for r in Rs:
        if 'schur' not in r:
            continue
        d = r['points']['P0_fit']['hessian_fd_0.001']
        Hb = np.array([[d[kb][ka] for kb in P] for ka in P]); Hc += 0.5 * (Hb + Hb.T)
        Db = np.array([[r['schur']['delta'][P[j]][P[k]] for k in range(5)] for j in range(5)])
        D += 0.5 * (Db + Db.T)
    calib = Rs[0]['points']['P0_fit']['calib']
    J = np.array([ (ranges[k]['max']-ranges[k]['min']) * scale
                   * ((float(calib[k])-ranges[k]['min'])/(ranges[k]['max']-ranges[k]['min']))
                   * (1-(float(calib[k])-ranges[k]['min'])/(ranges[k]['max']-ranges[k]['min']))
                   / abs(float(calib[k])) for k in P])
    S = Hc - D
    # clip the small negative eigenvalue (diffusion subspace, FD noise): PSD projection
    w, V = np.linalg.eigh(S)
    Sc = V @ np.diag(np.maximum(w, 1e-6 * w.max())) @ V.T
    return Hc, Sc, J, w


def sig(M, J):
    return 100 * J * np.sqrt(np.diag(np.linalg.inv(M)))


def cor(M, a=0, b=4):
    Cv = np.linalg.inv(M); d = np.sqrt(np.diag(Cv))
    return Cv[a, b] / d[a] / d[b]


fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.4))

# ---------------- (a) the ladder, at BT2 ----------------
a = ax[0]
HB = [analyse('SCHBT2', s) for s in (0, 1)]
lev = {}
lev['conditional\n(fig 58 / 71)'] = np.mean([100 * J / np.sqrt(np.diag(H)) for H, S, J, w in HB], axis=0)
lev['+ calibration\ncorrelations'] = np.mean([sig(H, J) for H, S, J, w in HB], axis=0)
lev['+ dE/dx\nmarginalised'] = np.mean([sig(S, J) for H, S, J, w in HB], axis=0)
xs = np.arange(3)
show = [('Ab', C['blue'], 1), ('lifetime', C['red'], 1), ('eField', C['green'], 1)]
for p, col, _ in show:
    j = P.index(p)
    v = [lev[k][j] for k in lev]
    a.plot(xs, v, '-o', color=col, ms=9, lw=2.2, label=PL[p])
    a.axhline(ACH[p], color=col, lw=1.3, ls=':')
    a.annotate(f'  achieved {ACH[p]:.2f}%', (2.02, ACH[p]), fontsize=7, color=col, va='center')
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels(list(lev.keys()), fontsize=8)
a.set_ylabel('1σ floor, % of the fitted value')
a.set_title('(a) THE LADDER at BT2. Each marginalisation raises the floor;\n'
            'lifetime: 0.30 → 0.58 → 1.1% vs 1.65% achieved — now ×1.5 away.', fontsize=9.3)
a.legend(fontsize=7.6, frameon=False, loc='upper left')

# ---------------- (b) correlation deepening, 5/5 ----------------
a = ax[1]
labs = []
for i, (tag, seeds) in enumerate([('SCHBT2', [0, 1]), ('SCHTS1', [0, 1, 2])]):
    for s in seeds:
        H, S, J, w = analyse(tag, s)
        x = len(labs)
        a.plot([x, x], [cor(H), cor(S)], color=C['grey'], lw=1.4)
        a.plot(x, cor(H), 'o', ms=9, color=C['grey'], label='conditional' if not labs else None)
        a.plot(x, cor(S), 'o', ms=11, color=C['red'],
               label='dE/dx marginalised' if not labs else None)
        labs.append(f'{tag[3:]} s{s}')
a.set_xticks(range(len(labs))); a.set_xticklabels(labs, fontsize=8)
a.set_ylabel('A$_b$ ↔ lifetime correlation')
a.set_title('(b) THE DEGENERACY DEEPENS on 4/5 jobs (mean −0.78 → −0.83).\n'
            'dE/dx is the charge-normalisation intermediary.', fontsize=9.3)
a.legend(fontsize=7.6, frameon=False, loc='lower left')

# ---------------- (c) long_diff: the displaced axis ----------------
a = ax[2]
M = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
CB = [('gfit', 'dfit', 'nuisances at FIT', C['ink']), ('gtru', 'dtru', 'both at TRUTH', C['red'])]
for s, alpha in [(0, .95), (1, .5)]:
    h = pickle.load(open(glob.glob(f'fit_result/sci_full_LDSCAN/history_iter*seed{s}.pkl')[0], 'rb'))
    Rs = h['truthpoint']
    fitv = float(np.ravel(h['long_diff_iter'])[-1]); tgt = float(np.ravel(h['long_diff_target'])[0])
    for g, d, lab, col in CB:
        ys = np.array([sum(r['points'][f'Clong_diff_{g}_{d}_{m:g}']['loss'] for r in Rs) for m in M])
        xerr = [100 * (m * fitv / tgt - 1) for m in M]
        a.plot(xerr, ys - ys.min(), '-o', color=col, ms=5, lw=1.8, alpha=alpha,
               label=lab if s == 0 else None)
a.axvline(0, color=C['green'], lw=1.8)
a.annotate(' TRUTH', (0, 6000), fontsize=8, color=C['green'])
a.set_yscale('symlog', linthresh=100)
a.set_xlabel('longitudinal diffusion, % from truth')
a.set_ylabel('Δ loss from the minimum')
a.set_title('(c) LONG. DIFFUSION (±80% scan): the conditional optimum sits\n'
            '10–44% BELOW truth and moves AWAY as nuisances improve.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

fig.suptitle('Fig 72 — the uncertainty ladder and what it reframes. Marginalising dE/dx (exact, '
             'banded+rank-1 direct solve) raises the lifetime floor to 0.9–1.1%\n(band = the PSD-'
             'projection systematic): the achieved 1.65% is ×1.5–1.8 away, not ×6.4, with geometry '
             'still unmarginalised — lifetime is close to its\nstatistical limit given the nuisance '
             'burden. The A_b↔lifetime degeneracy deepens (4/5, mean −0.78 → −0.83), and '
             'longitudinal diffusion is a\ngenuinely displaced axis.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .85])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig72_uncertainty_ladder.png'); plt.close(fig)
print('wrote', f'{OUT}/fig72_uncertainty_ladder.png')
for tag, seeds in [('SCHBT2', [0, 1]), ('SCHTS1', [0, 1, 2])]:
    for s in seeds:
        H, S, J, w = analyse(tag, s)
        print(f'  {tag} s{s}: neg eig clipped: {int((w<0).sum())}   '
              f'life cond {100*J[4]/np.sqrt(H[4,4]):.3f} calib-m {sig(H,J)[4]:.3f} dEdx-m {sig(S,J)[4]:.3f}   '
              f'cor {cor(H):+.2f}->{cor(S):+.2f}')
