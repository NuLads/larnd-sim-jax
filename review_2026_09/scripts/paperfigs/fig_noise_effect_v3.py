"""Fig. noise_effect, version 3 (12 Oct 2026): why the noiseless simulation is not the expectation of the readout.
Same waveforms as Fig. prob_vs_stoch (fig3_census_cache.npz: one batch of real through-going muons, production parameters,
926 active pixels), so that this figure (the problem) and Fig. prob_vs_stoch (the solution) share their pixels.

  (a) one pixel: cumulative charge through threshold; the noiseless first hit vs the distribution of first-hit times
      over noise throws (and how often there is no hit / a second hit);
  (b) every pixel: mean number of hits under noise against the noiseless number;
  (c) every pixel with a hit: shift of the mean first-hit time under noise from the noiseless time, against pixel charge;
  (d) the operating point: relative bias of the noiseless hit yield against the discrimination threshold.
Noisy per-pixel statistics for (b), (c) from prob_vs_stoch_cache.npz (1e4 throws, production noise model,
fee_jax.get_adc_values); noiseless = the same function with RESET_NOISE_CHARGE = UNCORRELATED_NOISE_CHARGE = 0.

  --compute (GPU): noiseless output at the nominal threshold, per-throw hits on a few example pixels (2000 throws), and
                   the threshold scan (noiseless vs 1000 noise throws, 3-10 ke- in 0.5 ke- steps). Writes CACHE.
  (default) plots from CACHE + prob_vs_stoch_cache.npz and prints the numbers."""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, 'fig3_census_cache.npz')
PVS = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/prob_vs_stoch_cache.npz'
CACHE = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/noise_effect_v3_cache.npz'
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/noise_effect_v3'
THR = np.arange(3000, 10001, 500)
N_EX, N_SCAN, CHUNK = 2000, 1000, 50


def compute():
    sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
    import jax, jax.numpy as jnp
    import paper_common as pc
    from larndsim.fee_jax import get_adc_values, digitize
    truth, _ = pc.params()
    c = np.load(CENSUS); valid = c['valid']; wfs = jnp.asarray(c['wfs'][valid]); npx = wfs.shape[0]
    assert abs(float(truth.DISCRIMINATION_THRESHOLD) - float(c['Vth'])) < 1, (truth.DISCRIMINATION_THRESHOLD, c['Vth'])
    nf = lambda p: p.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
    q0, t0 = get_adc_values(nf(truth), wfs, jax.random.key(0))
    q0, t0 = np.asarray(q0), np.asarray(t0); a0 = np.asarray(digitize(truth, jnp.asarray(q0)))
    hit0 = q0 > 0
    qtot = np.asarray(wfs).sum(1) * float(truth.t_sampling)
    # example pixels: noiseless single hit, spread over charge
    cand = np.where(hit0.sum(1) == 1)[0]
    ex = cand[np.argsort(qtot[cand])][np.linspace(0, len(cand) - 1, 12).astype(int)]
    run = jax.jit(lambda w, k: get_adc_values(truth, w, k))
    ex_t, ex_n = [], []
    key = jax.random.key(7)
    for i in range(N_EX // CHUNK):
        key, sub = jax.random.split(key)
        q, tk = run(jnp.tile(wfs[ex], (CHUNK, 1)), sub)
        q = np.asarray(q).reshape(CHUNK, len(ex), -1); tk = np.asarray(tk).reshape(CHUNK, len(ex), -1)
        ex_n.append((q > 0).sum(2)); ex_t.append(np.where(q[:, :, 0] > 0, tk[:, :, 0], np.nan))
    ex_n, ex_t = np.concatenate(ex_n), np.concatenate(ex_t)
    # threshold scan
    scan = []
    for T in THR:
        p = truth.replace(DISCRIMINATION_THRESHOLD=float(T))
        qn, _ = get_adc_values(nf(p), wfs, jax.random.key(0)); n0 = int((np.asarray(qn) > 0).sum()); f0 = int((np.asarray(qn)[:, 0] > 0).sum())
        runT = jax.jit(lambda w, k, p=p: get_adc_values(p, w, k))
        tots, fires = [], []
        key = jax.random.key(int(T))
        for i in range(N_SCAN // CHUNK):
            key, sub = jax.random.split(key)
            q, _ = runT(jnp.tile(wfs, (CHUNK, 1)), sub)
            q = np.asarray(q).reshape(CHUNK, npx, -1)
            tots.append((q > 0).sum((1, 2))); fires.append((q[:, :, 0] > 0).sum(1))
        tots, fires = np.concatenate(tots), np.concatenate(fires)
        scan.append([T, n0, f0, tots.mean(), tots.std(ddof=1), fires.mean()])
        print(f'threshold {T}: noiseless {n0} hits on {f0} pixels; noisy mean {tots.mean():.1f} +- {tots.std(ddof=1)/np.sqrt(len(tots)):.1f}'
              f' (bias {100*(n0/tots.mean()-1):+.2f}%)', flush=True)
    np.savez_compressed(CACHE, hit0=hit0, t0=t0, a0=a0, qtot=qtot, ex=ex, ex_t=ex_t, ex_n=ex_n, cum_ex=np.cumsum(np.asarray(wfs)[ex], 1) * float(truth.t_sampling),
                        scan=np.array(scan), Vth=float(truth.DISCRIMINATION_THRESHOLD), ts=float(truth.t_sampling))
    print('wrote', CACHE)


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = dict(np.load(CACHE)); s = dict(np.load(PVS)); N = float(s['N'])
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
    C0, C1, CG = '#D55E00', '#0072B2', '#7F7F7F'      # noiseless, noisy, neutral
    n0 = z['hit0'].sum(1); nm = s['tot'] / N; q = z['qtot']
    hasfirst = z['hit0'][:, 0] & (s['cnt'][:, 0] > 0)
    tmean = np.where(s['cnt'][:, 0] > 0, s['st'][:, 0] / np.maximum(s['cnt'][:, 0], 1), np.nan)
    dt = tmean - z['t0'][:, 0]
    pfire = s['cnt'][:, 0] / N
    nums = dict(pixels=int(len(n0)),
                mean_hits_by_noiseless={int(k): float(nm[n0 == k].mean()) for k in range(0, 5) if (n0 == k).any()},
                n_pixels_by_noiseless={int(k): int((n0 == k).sum()) for k in range(0, 5)},
                noiseless_total=int(n0.sum()), noisy_total=float(nm.sum()),
                dt_median=float(np.nanmedian(dt[hasfirst])), dt_p10_p90=[float(np.nanpercentile(dt[hasfirst], p)) for p in (10, 90)],
                zero_hit_pixels_with_noise_hits=float(((n0 == 0) & (nm > 0.05)).sum()),
                scan=[[int(r[0]), float(100 * (r[1] / r[3] - 1))] for r in z['scan']])
    print(json.dumps(nums, indent=1)); json.dump(nums, open(CACHE.replace('.npz', '_numbers.json'), 'w'), indent=1)
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6)); (a, b), (c, d) = ax
    # (a) example pixel: a mid-charge one whose noisy first-hit time is spread
    ex = z['ex']; k = int(np.argmin(np.abs(z['qtot'][ex] - float(z['Vth']))))   # the example pixel nearest the threshold
    j = ex[k]; ts = float(z['ts']); cum = z['cum_ex'][k]; t_nl = z['t0'][j, 0]
    tt = np.arange(len(cum)); thr = float(z['Vth'])
    lo, hi = int(t_nl - 40), int(t_nl + 40)
    a.plot((tt[lo:hi] - t_nl) * ts, cum[lo:hi] / 1e3, color='k', lw=1.1, label='cumulative charge (no noise)')
    a.axhline(thr / 1e3, color=CG, ls=':', lw=0.7); a.text((hi - t_nl - 2) * ts, thr / 1e3 * 1.03, 'threshold', ha='right', fontsize=6, color=CG)
    a.axvline(0, color=C0, lw=1.2, label='hit without noise')
    et = z['ex_t'][:, k]; ok = np.isfinite(et)
    a2 = a.twinx(); a2.hist((et[ok] - t_nl) * ts, bins=np.arange(-4, 4.01, 0.2), color=C1, alpha=0.45, label='first hit with noise')
    a2.set_yticks([]); a2.set_ylim(0, a2.get_ylim()[1] * 2.2)
    a.axvline(np.nanmean(et - t_nl) * ts, color=C1, lw=1.2, ls='--', label='mean with noise')
    a.set_ylim(top=float(np.max(cum[lo:hi])) / 1e3 * 1.45); a.set_xlim(-4, 4); a.set_xlabel('time from the noiseless hit ($\\mu$s)'); a.set_ylabel('charge on the pixel (ke$^-$)')
    p_none = float(np.mean(~ok)); p_two = float(np.mean(z['ex_n'][:, k] >= 2))
    a.text(0.97, 0.06, f'no hit: {100*p_none:.0f}% of throws\ntwo hits or more: {100*p_two:.0f}%', transform=a.transAxes, ha='right', fontsize=6)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False, fontsize=6)
    a.text(0.97, 0.92, '(a)', transform=a.transAxes, ha='right')
    # (b) mean number of hits with noise vs noiseless number
    rng = np.random.default_rng(0)
    for kk in range(0, 5):
        m = n0 == kk
        if not m.any(): continue
        b.scatter(kk + rng.uniform(-0.18, 0.18, m.sum()), nm[m], s=4, color=C1, alpha=0.5, lw=0)
        b.plot([kk - 0.3, kk + 0.3], [nm[m].mean()] * 2, color=C1, lw=1.6)
    b.plot([-0.5, 4.5], [-0.5, 4.5], color=C0, lw=1.0, ls='--', label='no noise (identity)')
    b.plot([], [], color=C1, lw=1.6, label='mean with noise (pixels, average)')
    b.set_xlim(-0.5, 4.5); b.set_ylim(-0.2, 5); b.set_xticks(range(5))
    b.set_xlabel('hits on the pixel without noise'); b.set_ylabel('mean hits with noise'); b.legend(loc='upper left', frameon=False)
    b.text(0.97, 0.06, '(b)', transform=b.transAxes, ha='right')
    # (c) first-hit time shift vs pixel charge
    c.scatter(q[hasfirst] / 1e3, dt[hasfirst] * ts, s=4, color=C1, alpha=0.6, lw=0)
    c.axhline(0, color=C0, lw=1.0, ls='--')
    c.set_xscale('log'); c.set_xlabel('charge on the pixel (ke$^-$)'); c.set_ylabel('mean first hit with noise\n$-$ noiseless hit ($\\mu$s)')
    c.set_ylim(-1.6, 0.4); noff = int(np.sum((dt[hasfirst] * ts < -1.6) | (dt[hasfirst] * ts > 0.4)))
    if noff: c.text(0.03, 0.06, f'{noff} pixel' + ('s' if noff > 1 else '') + ' off scale', transform=c.transAxes, fontsize=6, color=CG)
    c.text(0.97, 0.06, '(c)', transform=c.transAxes, ha='right')
    # (d) operating point
    sc = z['scan']; bias = 100 * (sc[:, 1] / sc[:, 3] - 1); err = 100 * sc[:, 1] / sc[:, 3] ** 2 * sc[:, 4] / np.sqrt(N_SCAN)
    d.axhline(0, color=C0, lw=1.0, ls='--')
    d.errorbar(sc[:, 0] / 1e3, bias, yerr=err, color=C1, marker='o', ms=3, lw=1.0, capsize=0)
    d.axvline(thr / 1e3, color=CG, ls=':', lw=0.7); d.text(thr / 1e3 + 0.1, d.get_ylim()[1] * 0.85 if d.get_ylim()[1] > 0 else 1, 'nominal', fontsize=6, color=CG)
    d.set_xlabel('discrimination threshold (ke$^-$)'); d.set_ylabel('noiseless / mean noisy hits $-$ 1 (%)')
    d.text(0.97, 0.06, '(d)', transform=d.transAxes, ha='right')
    for x in (a, b, c, d): x.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.5, h_pad=1.4)
    fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight'); print('saved', OUT)


if __name__ == '__main__':
    compute() if '--compute' in sys.argv else plot()
