"""The standard-method lifetime measurement on our own cosmic-muon sample, and why our fit
is ~24x levered on the dE/dx scale when in principle it should not be at all.

THE PHYSICS QUESTION (Pierre): electron lifetime is a pure charge-vs-drift-time effect. It is the
SLOPE of ln(dQ/dx) against drift time, so a global dE/dx scale error moves the INTERCEPT and must
leave the slope -- hence the lifetime -- untouched. Yet the S6h ladder measures -24.3 percentage
points of lifetime per 1% of dE/dx mean. Both cannot be right about the same estimator.

They are not about the same estimator. This script measures both on the same events:

 (1) STANDARD METHOD -- bin dQ/dx by drift time, fit ln(dQ/dx) = ln A - t/tau. Uses only the SHAPE.
     Repeated with a deliberate +/-2% charge scale applied, which must not move tau.

 (2) NORMALISATION METHOD -- what a 1-D likelihood slice with every other parameter FROZEN is
     forced to do: the only way it can absorb a global charge error is by changing the mean
     attenuation <exp(-t/tau)>, so it moves tau until the total charge matches.

Geometry (module0): drift along x, cathode at x=0, anodes at |x| = drift_length, so
drift_distance = drift_length - |x| and t = drift_distance / vdrift. With drift_length 30.27 cm
and vdrift 0.1587 cm/us the FULL detector is only 190.8 us deep against a 2200 us lifetime --
0.087 lifetimes, an 8.3% charge swing anode-to-cathode. That short lever arm is the whole story.
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRUE = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
L, VDRIFT, TAU = 30.27225, 0.1587, 2200.0
AB, KB, EFIELD, RHO = 0.8, 0.0486, 0.50, 1.38
OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def recomb(dedx):
    """Birks recombination, matching larndsim.quenching_jax.birks_model (the model this
    detector config actually uses -- it carries Ab/kb, not the Box model's alpha/beta).
    A function of dE/dx only, so it is drift-time independent and cannot affect the slope."""
    return AB / (1.0 + KB * dedx / (EFIELD * RHO))


def load():
    with h5py.File(TRUE, 'r') as h:
        s = h['segments'][:]
    dedx = s['dEdx'].astype(np.float64)
    dx = s['dx'].astype(np.float64)
    x = s['x'].astype(np.float64)
    ok = (dx > 0.05) & (dedx > 0.1) & (dedx < 10) & (np.abs(x) <= L)
    dedx, dx, x = dedx[ok], dx[ok], x[ok]
    t = (L - np.abs(x)) / VDRIFT                      # us
    dqdx = dedx * recomb(dedx) * np.exp(-t / TAU)     # arbitrary units; attenuation applied
    return t, dqdx


def fit_exp(t, dqdx, nbin=20, use='median'):
    """Standard method: bin in drift time, take the MPV-like statistic, fit ln(Q) vs t."""
    edges = np.linspace(t.min(), t.max(), nbin + 1)
    idx = np.digitize(t, edges) - 1
    tc, q, qe = [], [], []
    for b in range(nbin):
        m = idx == b
        if m.sum() < 50:
            continue
        v = dqdx[m]
        c = np.median(v) if use == 'median' else v.mean()
        # error on the median ~ 1.253 sigma/sqrt(N); use a robust sigma
        sig = 1.4826 * np.median(np.abs(v - np.median(v)))
        tc.append(0.5 * (edges[b] + edges[b + 1])); q.append(c)
        qe.append(1.253 * sig / np.sqrt(m.sum()))
    tc, q, qe = map(np.asarray, (tc, q, qe))
    w = (q / qe) ** 2                                  # weights for ln(q)
    A = np.vstack([np.ones_like(tc), tc]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    beta = cov @ (A.T @ W @ np.log(q))
    tau = -1.0 / beta[1]
    dtau = abs(tau) * np.sqrt(cov[1, 1]) / abs(beta[1])
    return tc, q, qe, tau, dtau, beta


if __name__ == '__main__':
    t, dqdx = load()
    print(f'{len(t)} segments | drift time  min {t.min():.1f}  max {t.max():.1f}  mean {t.mean():.1f} us')
    print(f'detector depth in lifetimes: {t.max() / TAU:.4f}   charge swing {100*(1-np.exp(-t.max()/TAU)):.2f}%')

    res = {}
    for lab, scale in [('nominal', 1.00), ('dE/dx +2%', 1.02), ('dE/dx −2%', 0.98)]:
        tc, q, qe, tau, dtau, beta = fit_exp(t, dqdx * scale)
        res[lab] = (tc, q, qe, tau, dtau)
        print(f'{lab:10s} charge scale {scale:.2f} -> fitted tau = {tau:8.1f} +/- {dtau:6.1f} us '
              f'({100*(tau/TAU-1):+6.2f}% vs truth {TAU:.0f})')

    # what a 1-D slice with everything else frozen is forced to do
    print()
    wq = np.exp(-t / TAU)
    lever = 1.0 / np.average(t / TAU, weights=wq)
    print(f'charge-weighted <t/tau> = {1/lever:.4f}  ->  normalisation-only leverage {lever:.1f}x')
    for s in (0.02, -0.02):
        # solve <exp(-t/tau')> = (1+s)*<exp(-t/tau)> for tau'
        # a +s dE/dx excess in the SIM must be cancelled by MORE attenuation -> shorter tau
        target = np.mean(np.exp(-t / TAU)) / (1 + s)
        lo, hi = 100.0, 200000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.mean(np.exp(-t / mid)) < target:
                lo = mid
            else:
                hi = mid
        print(f'  a {s*+100:+.0f}% charge error, absorbed ONLY by lifetime -> tau {mid:8.1f} us '
              f'({100*(mid/TAU-1):+7.2f}%)')

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    ax = axes[0]
    ax.hist(t, bins=60, color=C['blue'], edgecolor='white', linewidth=.4)
    ax.axvline(t.mean(), color=C['red'], lw=1.6, ls='--', label=f'mean {t.mean():.0f} µs')
    ax.set_xlabel('drift time (µs)'); ax.set_ylabel('segments')
    ax.set_title(f'(a) our lever arm: 0–{t.max():.0f} µs\nagainst a {TAU:.0f} µs lifetime',
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for (lab, col) in [('nominal', C['blue']), ('dE/dx +2%', C['orange']), ('dE/dx −2%', C['green'])]:
        tc, q, qe, tau, dtau = res[lab]
        ax.errorbar(tc, q, yerr=qe, color=col, marker='o', ms=4, lw=0, elinewidth=1.4, capsize=2)
        tt = np.linspace(0, t.max(), 100)
        b = np.polyfit(tc, np.log(q), 1)
        ax.plot(tt, np.exp(b[1] + b[0] * tt), color=col, lw=1.6,
                label=f'{lab}: τ = {tau:.0f} ± {dtau:.0f} µs')
    ax.set_xlabel('drift time (µs)'); ax.set_ylabel('median dQ/dx (arb.)')
    ax.set_title('(b) standard method: the SLOPE is\nimmune to the dE/dx scale', fontsize=9.5)
    ax.legend(fontsize=7.6, frameon=False)

    ax = axes[2]
    ss = np.linspace(-2.5, 2.5, 51)
    taus = []
    for s in ss:
        target = np.mean(np.exp(-t / TAU)) / (1 + s / 100)
        lo, hi = 100.0, 500000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.mean(np.exp(-t / mid)) < target:
                lo = mid
            else:
                hi = mid
        taus.append(100 * (mid / TAU - 1))
    ax.plot(ss, taus, color=C['red'], lw=2, label='normalisation-only (1-D slice)')
    ax.plot(ss, np.zeros_like(ss), color=C['blue'], lw=2, ls='--',
            label='standard method (slope)')
    ax.scatter([-2, 0, 2], [50.75, 2.18, -22.30], color=C['ink'], zorder=5, s=40,
               label='measured on the §6h scan')
    ax.axhline(0, color=C['grey'], lw=1.1); ax.axvline(0, color=C['grey'], lw=1.1)
    ax.set_xlabel('dE/dx scale error (%)'); ax.set_ylabel('induced lifetime error (%)')
    ax.set_title('(c) the two estimators disagree\nby construction', fontsize=9.5)
    ax.legend(fontsize=7.6, frameon=False)
    fig.suptitle('Fig 34 — why lifetime looks 24× levered on the dE/dx scale: it is only levered '
                 'when the fit\nmay not move anything else. The detector is 0.087 lifetimes deep.',
                 fontsize=10.2)
    fig.tight_layout(rect=[0, 0, 1, .88])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/fig34_lifetime_standard_method.png'); plt.close(fig)
    print('\nwrote', f'{OUT}/fig34_lifetime_standard_method.png')
