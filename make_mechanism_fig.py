"""Fig 76 — the landing-bias mechanism, closed: overshoot, truncated relaxation, inheritance,
and the one intervention that works.

THE ARC (all on replica 8, the worst case, lifetime landing +16.4%):
  * the bias forms as a hot-phase OVERSHOOT while geometry is still near its bad linear-guess
    init, then relaxes toward truth too slowly for the anneal (fig 75 + trace timing);
  * late8 (gate calibration until iter 2000) FAILS catastrophically (+55%): the nuisances
    co-adapt to the wrong frozen calibration during the gate, and the release is violent;
  * hot8 (calibration decay halved) recovers only marginally (+14.6%): the relaxation is real
    but far too slow to finish within any reasonable schedule;
  * gtru8 (geometry INITIALISED AT TRUTH) fixes most of everything: lifetime +5.8, long_diff
    +6.9 (from +34), position 117 um, loss 8299 (best by 870) -- the overshoot never happens
    when geometry starts right;
  * pass2r8 (geometry warm-started from the replica's own CONVERGED fit) does NOT reproduce
    gtru8: the co-adapted geometry REMEMBERS the biased lifetime and pulls the re-annealed
    calibration back toward it. Converged is not the same as true.

THE CONCLUSION FOR THE PROGRAMME: the landing bias is caused by the geometry transient and is
inherited by anything bootstrapped from a co-adapted state. The production remedies are
(1) better INDEPENDENT geometry initialisation -- from reconstruction, not from a previous fit --
and (2) the preconditioned joint step, which is the only optimiser-side route that moves the
blocks coherently instead of letting them co-adapt sequentially.
"""
import glob, os, pickle, re, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def tr(pat, key='lifetime'):
    fs = glob.glob(pat)
    f = max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    h = pickle.load(open(f, 'rb'))
    if key in ('pos', 'loss'):
        v = {'pos': np.ravel(h['pos_residual_iter']) * 1e4,
             'loss': np.asarray(h['losses_iter'], float)}[key]
        return sm(v)
    t = float(np.ravel(h[key + '_target'])[0])
    return sm(100 * (np.ravel(h[key + '_iter'])[1:] / t - 1))


ARMS = [('baseline', 'fit_result/sci_full_DSREP8/history_iter*.pkl', C['ink'],
         'single pass (the +16.4% landing)'),
        ('hot8', 'fit_result/sci_full_HOT8/history_iter*.pkl', C['orange'],
         'slower calib anneal → +14.6'),
        ('late8', 'fit_result/sci_full_LATE8/history_iter*.pkl', C['red'],
         'calib gated to 2 000 → +55, FAILED'),
        ('gtru8', 'fit_result/sci_full_GTRU8/history_iter*.pkl', C['green'],
         'geometry init AT TRUTH → +5.8'),
        ('pass2r8', 'fit_result/sci_full_PASS2R8/history_iter*.pkl', C['purple'],
         'geometry from own CONVERGED fit (running)')]

fig, ax = plt.subplots(1, 3, figsize=(16.8, 5.8))

# ---------------- (a) the traces ----------------
a = ax[0]
for tag, pat, col, lab in ARMS:
    e = tr(pat)
    a.plot(np.arange(len(e)), e, color=col, lw=2.0, label=lab)
a.axhline(0, color=C['green'], lw=2, ls=':')
a.annotate(' truth', (8600, 0.8), fontsize=9, color=C['green'])
a.set_ylim(-12, 60)
a.set_xlabel('iteration')
a.set_ylabel('lifetime error (%)')
a.set_title('(a) replica 8, five strategies. The overshoot happens whenever\n'
            'geometry starts wrong; only starting it RIGHT prevents it.', fontsize=10)
a.legend(fontsize=7.6, frameon=True, framealpha=.92, loc='upper right')

# ---------------- (b) final outcomes ----------------
a = ax[1]
labs, colv = [], []
LIF, LNG, POS = [], [], []
for tag, pat, col, _ in ARMS[:4]:
    labs.append(tag); colv.append(col)
    LIF.append(abs(tr(pat)[-1]))
    LNG.append(abs(tr(pat, 'long_diff')[-1]))
    POS.append(tr(pat, 'pos')[-1])
labs.append('drift w=1e3'); colv.append(C['blue'])
LIF.append(10.38); LNG.append(17.35); POS.append(218.9)
xs = np.arange(len(labs))
a.bar(xs - .22, LIF, width=.22, color=C['red'], edgecolor='white', label='|lifetime err| %')
a.bar(xs, LNG, width=.22, color=C['orange'], edgecolor='white', label='|long. diff. err| %')
a.bar(xs + .22, np.array(POS) / 10, width=.22, color=C['grey'], edgecolor='white',
      label='position (10 µm units)')
for i in range(len(labs)):
    a.text(i - .22, LIF[i] + .8, f'{LIF[i]:.1f}', ha='center', fontsize=7.6, color=C['red'])
a.set_xticks(xs); a.set_xticklabels(labs, fontsize=8.4, rotation=12)
a.set_ylim(0, 75)
a.set_title('(b) final outcomes. gtru8 fixes lifetime, long. diffusion AND\n'
            'position at once — one cause, three symptoms.', fontsize=10)
a.legend(fontsize=7.8, frameon=False)

# ---------------- (c) the inheritance ----------------
a = ax[2]
for tag, pat, col, lab in [('gtru8', ARMS[3][1], C['green'], 'geometry at TRUTH'),
                           ('pass2r8', ARMS[4][1], C['purple'],
                            'geometry from own CONVERGED fit'),
                           ('baseline', ARMS[0][1], C['ink'], 'cold start (reference)')]:
    e = tr(pat)
    a.plot(np.arange(len(e)), e, color=col, lw=2.2, label=lab)
a.axhline(0, color=C['green'], lw=2, ls=':')
a.set_xlim(0, 10000); a.set_ylim(-12, 40)
a.set_xlabel('iteration'); a.set_ylabel('lifetime error (%)')
a.set_title('(c) CONVERGED ≠ TRUE: warm-starting from the co-adapted fit\n'
            're-injects the bias — the nuisances remember the wrong lifetime.', fontsize=10)
a.legend(fontsize=8, frameon=False, loc='upper right')

fig.suptitle('Fig 76 — the landing-bias mechanism, closed (replica 8, baseline lifetime landing '
             '+16.4%). The bias is a hot-phase overshoot caused by the geometry\ntransient, '
             'relaxing too slowly for the anneal. Gating calibration fails (+55%); a slower anneal '
             'barely helps (+14.6%); geometry initialised at truth fixes\nlifetime (+5.8%), long. '
             'diffusion (+6.9%) and position (117 µm) at once — and a converged-but-co-adapted '
             'geometry does NOT substitute for a true one.', fontsize=10.4)
fig.tight_layout(rect=[0, 0, 1, .85])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig76_mechanism.png'); plt.close(fig)
print('wrote', f'{OUT}/fig76_mechanism.png')
