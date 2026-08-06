"""The standard lifetime measurement on SIMULATED HITS -- with the full front end.

S6k did this on truth segments and recovered tau to -1.2%, exactly scale-immune. But that bypasses
the electronics entirely. This redoes it on the hit lists the simulation actually produces
(adc, tick, pixel), so the discrimination threshold, the ADC response and the readout noise are
all in play.

THE EXPECTED BIAS, stated before looking: the FEE threshold (5000 e-) preferentially removes SMALL
hits. Small hits are the ones that drifted FURTHEST and lost the most charge. So the surviving
sample at large drift time is skewed high, the measured dQ/dx-vs-t slope FLATTENS, and the fitted
lifetime comes out TOO LONG. Noise adds a second, same-signed effect near threshold: upward
fluctuations get selected in, downward ones are lost (an Eddington/Malmquist bias).

Hit charge:  Q = adc2charge(adc) = (adc/ADC_COUNTS*(V_REF-V_CM) + V_CM - V_PEDESTAL)/GAIN * 1e-3  [ke]
Drift time:  t = tick * t_sampling   (t_sampling = 0.1 us)

Three estimators are compared, because the difference between them IS the result:
  mean    -- what a naive analysis does; maximally exposed to the threshold cut
  median  -- the usual robust choice
  trunc   -- mean of hits above a charge floor well clear of threshold, the standard mitigation
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T_SAMPLING = 0.1          # us per tick
TAU_TRUE = 2200.0         # us   (target lifetime; --scan_tgt_nom / nominal)
# adc2charge constants. Voltages are in mV (consts_jax.py:525-532) and module0.yaml OVERRIDES
# V_CM and V_REF, so both files must be consulted -- the defaults alone give the wrong pedestal.
# Only the additive (V_CM - V_PEDESTAL) term can bias tau; a pure scale factor cancels in the
# slope, so an error there would be invisible in the lifetime but wrong in the quoted charges.
ADC_COUNTS = 2 ** 8
V_REF = 1282.71484375     # module0.yaml (default 1300)
V_CM = 284.27734375       # module0.yaml (default 288)
V_PEDESTAL = 580          # consts_jax default, not overridden
GAIN = 4e-3
THRESHOLD_E = 5000        # module0.yaml DISCRIMINATION_THRESHOLD (default 7e3)
OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def adc2charge(adc):
    return (adc / ADC_COUNTS * (V_REF - V_CM) + V_CM - V_PEDESTAL) / GAIN * 1e-3   # ke


def load_hits(target_dir):
    Q, t = [], []
    for f in sorted(glob.glob(os.path.join(target_dir, 'batch*_target.npz'))):
        d = np.load(f)
        adc = np.asarray(d['adcs'], float).ravel()
        tick = np.asarray(d['ticks'], float).ravel()
        # The ADC pedestal sits at adc ~= 76 (Q=0), so `adc > 0` is NOT the right cut -- it would
        # admit sub-pedestal codes with negative reconstructed charge and poison the log fit.
        # Cut on reconstructed charge instead; real hits are above the 5000 e- FEE threshold.
        q = adc2charge(adc)
        m = np.isfinite(q) & np.isfinite(tick) & (q > 0) & (tick > 0)
        Q.append(q[m]); t.append(tick[m] * T_SAMPLING)
    if not Q:
        return None, None
    return np.concatenate(Q), np.concatenate(t)


def fit_exp(t, Q, nbin=18, est='median', qfloor=None, tmax=None):
    """Bin in drift time, take an estimator per bin, fit ln(Q) = ln A - t/tau."""
    if qfloor is not None:
        m = Q > qfloor
        t, Q = t[m], Q[m]
    if tmax is None:
        tmax = np.percentile(t, 99.5)
    m = t <= tmax
    t, Q = t[m], Q[m]
    edges = np.linspace(0, tmax, nbin + 1)
    idx = np.digitize(t, edges) - 1
    tc, q, qe = [], [], []
    for b in range(nbin):
        s = idx == b
        n = int(s.sum())
        if n < 100:
            continue
        v = Q[s]
        c = np.median(v) if est == 'median' else v.mean()
        sig = 1.4826 * np.median(np.abs(v - np.median(v))) if est == 'median' else v.std()
        tc.append(0.5 * (edges[b] + edges[b + 1])); q.append(c)
        qe.append((1.253 if est == 'median' else 1.0) * sig / np.sqrt(n))
    tc, q, qe = map(np.asarray, (tc, q, qe))
    if len(tc) < 3 or np.any(q <= 0):
        return tc, q, qe, np.nan, np.nan
    w = (q / qe) ** 2
    A = np.vstack([np.ones_like(tc), tc]).T
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    beta = cov @ (A.T @ np.diag(w) @ np.log(q))
    tau = -1.0 / beta[1]
    dtau = abs(tau) * np.sqrt(cov[1, 1]) / abs(beta[1])
    return tc, q, qe, tau, dtau


if __name__ == '__main__':
    import sys
    data = {}
    for tag in ['nonoise', 'noise']:
        ds = sorted(glob.glob(f'target_hitdump_hitdump_{tag}_*'))
        if not ds:
            print(f'{tag}: no dump yet'); continue
        Q, t = load_hits(ds[0])
        if Q is None or len(Q) == 0:
            print(f'{tag}: empty'); continue
        data[tag] = (Q, t)
        print(f'{tag:8s} {len(Q):>8d} hits | Q med {np.median(Q):.2f} ke (threshold {THRESHOLD_E/1e3:.0f} ke) '
              f'| t {t.min():.1f}-{t.max():.1f} us')
    if not data:
        sys.exit('nothing to analyse')

    TMINS = np.arange(0, 61, 5)
    print(f"\n{'condition':10s}{'t_min':>7s}{'tau (us)':>12s}{'err':>8s}{'bias':>10s}")
    scan = {}
    for tag, (Q, t) in data.items():
        rows = []
        for tmin in TMINS:
            m = t >= tmin
            if m.sum() < 5000:
                continue
            _, _, _, tau, dtau = fit_exp(t[m], Q[m], est='mean')
            if np.isfinite(tau):
                rows.append((tmin, tau, dtau))
                print(f'{tag:10s}{tmin:>7d}{tau:>12.1f}{dtau:>8.1f}{100*(tau/TAU_TRUE-1):>+9.1f}%')
        scan[tag] = np.array(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    ax = axes[0]
    for tag, col in [('nonoise', C['blue']), ('noise', C['red'])]:
        if tag not in data: continue
        Q, _ = data[tag]
        ax.hist(Q, bins=np.linspace(0, 40, 80), histtype='step', lw=1.8, color=col,
                label=f'noise {"ON" if tag=="noise" else "OFF"} ({len(Q)/1e3:.0f}k)')
    ax.axvline(THRESHOLD_E/1e3, color=C['ink'], ls='--', lw=1.6)
    ax.text(THRESHOLD_E/1e3, ax.get_ylim()[1]*.92, ' FEE threshold', fontsize=8, color=C['ink'])
    ax.set_xlabel('hit charge Q (ke)'); ax.set_ylabel('hits')
    ax.set_title('(a) the threshold sits only ~2x below\nthe median hit charge', fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for tag, col in [('nonoise', C['blue']), ('noise', C['red'])]:
        if tag not in data: continue
        Q, t = data[tag]
        e = np.linspace(0, 190, 20); i = np.digitize(t, e) - 1
        tc, q, qe = [], [], []
        for b in range(19):
            m = i == b
            if m.sum() < 100: continue
            tc.append(.5*(e[b]+e[b+1])); q.append(Q[m].mean()); qe.append(Q[m].std()/np.sqrt(m.sum()))
        ax.errorbar(tc, q, yerr=qe, color=col, marker='o', ms=4, lw=1.2,
                    label=f'noise {"ON" if tag=="noise" else "OFF"}')
    q0 = np.mean(data[list(data)[-1]][0][data[list(data)[-1]][1] < 20])
    tt = np.linspace(0, 190, 60)
    ax.plot(tt, q0*np.exp(-(tt-10)/TAU_TRUE), color=C['ink'], ls='--', lw=1.6,
            label=f'truth slope (τ={TAU_TRUE:.0f} µs)')
    ax.axvspan(0, 20, color=C['grey'], alpha=.15)
    ax.text(10, ax.get_ylim()[0], ' anode\n edge', fontsize=7.5, va='bottom', color=C['grey'])
    ax.set_xlabel('drift time (µs)'); ax.set_ylabel('mean hit Q (ke)')
    ax.set_title('(b) the shape is NOT a single exponential', fontsize=9.5)
    ax.legend(fontsize=7.6, frameon=False)

    ax = axes[2]
    for tag, col in [('nonoise', C['blue']), ('noise', C['red'])]:
        if tag not in scan or len(scan[tag]) == 0: continue
        r = scan[tag]
        ax.errorbar(r[:,0], 100*(r[:,1]/TAU_TRUE-1), yerr=100*r[:,2]/TAU_TRUE,
                    color=col, marker='o', ms=5, lw=1.8,
                    label=f'noise {"ON" if tag=="noise" else "OFF"}')
    ax.axhline(0, color=C['ink'], lw=1.4, label='truth')
    ax.axhline(-1.21, color=C['green'], lw=1.6, ls=':', label='truth-level fit (§6k): −1.2%')
    ax.set_xlabel('fit range start $t_{min}$ (µs)'); ax.set_ylabel('lifetime bias (%)')
    ax.set_title('(c) the answer depends on an\narbitrary analysis choice', fontsize=9.5)
    ax.legend(fontsize=7.6, frameon=False)
    fig.suptitle('Fig 35 — the same standard fit on SIMULATED HITS. Exact on truth (−1.2%), '
                 'uncontrolled once the front end is included.', fontsize=10.3)
    fig.tight_layout(rect=[0, 0, 1, .92])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/fig35_lifetime_hits.png'); plt.close(fig)
    print('\nwrote', f'{OUT}/fig35_lifetime_hits.png')
