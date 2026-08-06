"""Fig 44: what the PPP objective is actually made of, and which term constrains each parameter.

THE DECOMPOSITION IS EXACT. From `strategies.py` (LUTProbabilisticSimulation branch):
    nll = -ll_hits - no_match_penalty + expected_total_hits
and the aux dict stores `log_likelihood_tick = -ll_hits` and `no_match_penalty` already negated, so

    total loss = log_likelihood_tick + no_match_penalty + expected_total_hits
                 + dedx_prior + dedx_mean_penalty + dedx_barrier + mcs_prior
                 + chain_drift_penalty + dedx_drift_penalty + spatial_moment

verified on a production run to 4.4e-3 absolute, 4.5e-7 relative (float32 noise).

THE KEY STORED AS `log_likelihood_tick` IS MISLABELLED -- it holds the JOINT (tick AND charge)
likelihood. `ProbabilisticLossStrategy` builds
    joint_window_log_probs = window_log_probs + log_time_weights + window_log_charge_intensity
where the last term is a Gaussian charge residual -0.5*((Q_tgt - Q_exp)/sigma_charge)^2 with
sigma_charge = 500 e-, and then marginalises with logsumexp. All three constructions of
`joint_hit_log_probs` in that class include it; there is no reachable path that omits it.

`aux["log_likelihood_charge"] = 0.0` is therefore a REPORTING STUB, not a disabled term.
PROVEN EMPIRICALLY: re-running an identical scan with --loss_sigma_charge 50 instead of 500 changes
the loss by 99.3% (3408.6 -> 7327.5 on the first grid point), and the entire change appears inside
the key stored as `log_likelihood_tick`, while the charge-independent `expected_total_hits` moves
by 1e-4. An earlier version of this figure claimed the loss had no charge term -- wrong.

Panels (b,c) decompose the 1-D likelihood scans, which answers the operational question: at the
minimum, WHICH term supplies the curvature that pins each parameter?
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_quality_ladder import ORDER, NOM

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
TERMS = [('log_likelihood_tick', C['blue'], 'joint hit log-intensity (tick AND charge)'),
         ('expected_total_hits', C['green'], 'expected hit count (PPP integral)'),
         ('no_match_penalty', C['red'], 'unmatched-target penalty'),
         ('dedx_prior', C['orange'], 'dE/dx student-t prior'),
         ('mcs_prior', C['purple'], 'MCS prior'),
         ('dedx_mean_penalty', C['grey'], 'dE/dx mean constraint')]


def aux(h, k):
    return np.array([e.get(k, 0.0) for e in h['aux_iter']], float)


def smooth(v, w=200):
    if len(v) < w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))

# ---- (a) production: components vs iteration
f = sorted(glob.glob('fit_result/sci_full_ANNEALLONG2/history_iter10000_*seed0.pkl'))[0]
h = pickle.load(open(f, 'rb'))
L = np.asarray(h['losses_iter'], float)
a = ax[0]
for k, col, lab in TERMS:
    v = aux(h, k)
    if np.abs(v).mean() < 1e-6:
        continue
    a.plot(smooth(v), color=col, lw=1.8, label=lab)
a.plot(smooth(L), color=C['ink'], lw=2.4, ls='--', label='TOTAL')
a.axhline(0, color=C['grey'], lw=1.0)
a.set_xlabel('iteration'); a.set_ylabel('contribution to the loss')
a.set_title('(a) production fit: 91% joint hit log-intensity\n(this term contains BOTH tick and charge)',
            fontsize=9.4)
a.legend(fontsize=7.0, frameon=False)

# ---- (b,c) 1-D scans: which term supplies the curvature?
def scan_terms(tag, param, nstep=21):
    g = sorted(glob.glob(f'fit_result/loss_profile/history_{ORDER[-1]}_batch*_prof_{tag}_seed0.pkl'))
    if not g:
        return None
    fpath = max(g, key=lambda p: int(p.split('_batch')[1].split('_')[0]))
    hh = pickle.load(open(fpath, 'rb'))
    Ls = np.asarray(hh['losses_iter'], float)
    nblock = len(Ls) // nstep
    x = np.asarray(hh[param + '_iter'], float)[1:]
    out = {}
    idx = [k for k in range(nblock) if ORDER[k % 5] == param]
    grid = x[idx[0] * nstep:(idx[0] + 1) * nstep]
    for key, _, _ in TERMS:
        vv = np.array([e.get(key, 0.0) for e in hh['aux_iter']], float)
        out[key] = np.sum([vv[k * nstep:(k + 1) * nstep] for k in idx], axis=0)
    out['TOTAL'] = np.sum([Ls[k * nstep:(k + 1) * nstep] for k in idx], axis=0)
    return grid, out


for j, param in enumerate(['lifetime', 'Ab']):
    r = scan_terms('true', param)
    a = ax[1 + j]
    if r is None:
        a.text(.5, .5, 'scan not found', ha='center'); continue
    grid, T = r
    xp = 100 * (grid - NOM[param]) / NOM[param]
    for k, col, lab in TERMS:
        v = T[k] - T[k].min()
        if np.ptp(T[k]) < 1e-6:
            continue
        a.plot(xp, v, color=col, lw=1.8, label=lab)
    a.plot(xp, T['TOTAL'] - T['TOTAL'].min(), color=C['ink'], lw=2.4, ls='--', label='TOTAL')
    a.axvline(0, color=C['grey'], lw=1.3)
    a.set_yscale('symlog', linthresh=1)
    a.set_xlabel(f'{param} offset from truth (%)')
    a.set_ylabel('Δ (term − its own minimum)')
    a.set_title(f'({chr(98+j)}) 1-D scan in {param}: which term\nsupplies the curvature?', fontsize=9.4)
    if j == 0:
        a.legend(fontsize=7.0, frameon=False)

fig.suptitle('Fig 44 — the PPP objective decomposed. The dominant term is the JOINT (tick AND charge) '
             'hit log-intensity;\nthe zero `log_likelihood_charge` field is a reporting stub, not a '
             'disabled term (proven by a σ$_Q$ scan).', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig44_loss_components.png'); plt.close(fig)
print('wrote', f'{OUT}/fig44_loss_components.png')

tot = sum(aux(h, k) for k, _, _ in TERMS) + aux(h, 'dedx_barrier') + aux(h, 'chain_drift_penalty') \
      + aux(h, 'dedx_drift_penalty') + aux(h, 'spatial_moment')
print(f'  decomposition closes to {np.abs(L - tot).max():.2e} absolute '
      f'({np.abs(L-tot).max()/np.abs(L).mean():.1e} relative)')
for param in ['lifetime', 'Ab']:
    r = scan_terms('true', param)
    if not r: continue
    grid, T = r
    print(f'  scan in {param}: curvature contributed by each term (range over the grid)')
    for k, _, _ in TERMS:
        if np.ptp(T[k]) > 1e-6:
            print(f'     {k:24s} range {np.ptp(T[k]):12.1f}')
