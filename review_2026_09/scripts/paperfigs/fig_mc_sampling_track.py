"""Fig. mc_sampling, version 2 (12 Oct 2026), rebuilt to match the rewritten Sec. II A: the reference simulation's
Monte Carlo packets at ITS step (10 um) against our analytic projection at OUR step (100 um), both measured against the
converged answer (our projection at 10 um), over every pixel of one muon track -- the same 48.5 cm of track as the
step-convergence numbers of the text (SAMPLING_STEP/input_49cm.h5, from LARNDSIM_CMP/tiny_input.h5).

Code path: the library's Monte Carlo cross-check path, where packets (detsim_jax.generate_electrons, params.mc_diff=True)
and the projection can be swapped on the SAME chopped pieces through the SAME current model (current_mc), without
induction on neighbouring pixels -- as the first version of this figure (fig_mc_sampling.py, whose helper functions and
the x0-aware projection `apply_tran_diff_x0` are copied here verbatim; see its docstring for why the shipped
apply_tran_diff cannot be used). Truth parameters and response from paper_common.

  --compute  (GPU job): projection at 0.01 and 0.001 cm, 40 Monte Carlo placements at 0.001 cm, 200 readout-noise throws
             on the converged waveform; per pixel: total charge, noiseless first threshold crossing; writes the cache.
  (default)  plot from the cache: (a) cumulative charge on the brightest pixel; (b) per-pixel charge deviation from the
             converged answer in reset-noise units vs the pixel charge; (c) the same for the threshold-crossing time, with
             the readout-noise timing spread for scale. Prints the numbers quoted in the text."""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
CACHE = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/mc_sampling_track_cache' + ('_nolong' if os.environ.get('NOLONG') == '1' else '') + '.npz'
SRC = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/input_49cm.h5'
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/mc_sampling'
N_SEEDS, N_NOISE = 40, (20 if os.environ.get('NOLONG') == '1' else 200)
RESET = 900.0


def compute():
    sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
    import paper_common as pc
    import jax, jax.numpy as jnp
    from optimize.dataio import TracksDataset
    from larndsim.sim_jax import simulate_drift, shift_tracks
    from larndsim.quenching_jax import quench
    from larndsim.drifting_jax import drift as drift_fn
    from larndsim.detsim_jax import (get_pixels, id2pixel, get_pixel_coordinates, current_mc,
                                      accumulate_signals_parametrized, gaussian_1d_integral_new)
    from larndsim.fee_jax import get_adc_values
    truth, response = pc.params()
    if os.environ.get('NOLONG') == '1':             # diagnostic: no longitudinal diffusion in either method
        truth = truth.replace(long_diff=1e-12)
    pitch, ts = float(truth.pixel_pitch), float(truth.t_sampling)
    EDGES = pitch * jnp.arange(-2.5, 3.0, 1.0)
    NT = int(truth.time_interval[1] / ts) + 1

    def load(r):
        ds = TracksDataset(filename=SRC, nevents=None, max_nbatch=None, swap_xz=True, random_nevents=False, data_seed=42,
                           max_batch_len=1000., print_input=False, chopped=True, pad=False, electron_sampling_resolution=r,
                           live_selection=False, use_dedx_density=False, dedx_density_mode='histogram')
        return ds.get_track_fields(), jnp.asarray(ds[0])

    fields, tr_c = load(0.01); fields_f, tr_f = load(0.001); assert fields == fields_f
    L = float(np.asarray(tr_c)[:, fields.index('dx')].sum())

    def make_params(mc, r, noise=False):
        p = truth.replace(mc_diff=mc, diffusion_in_current_sim=not mc, number_pix_neighbors=0, electron_sampling_resolution=r,
                          tran_diff_bin_edges=EDGES)
        return p if noise else p.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)

    def pieces(p, t):
        return drift_fn(p, quench(p, shift_tracks(p, t, fields), fields), fields)

    def proj(p, t):                                    # apply_tran_diff_x0 of fig_mc_sampling.py
        e = pieces(p, t)
        borders = p.tpc_borders[e[:, fields.index('pixel_plane')].astype(int)]
        bw = p.tran_diff_bin_edges[1] - p.tran_diff_bin_edges[0]
        x0 = (e[:, fields.index('x')] - borders[:, 0, 0]) % bw - bw / 2
        y0 = (e[:, fields.index('y')] - borders[:, 1, 0]) % bw - bw / 2
        sd = e[:, fields.index('tran_diff')]
        gx = gaussian_1d_integral_new(p.tran_diff_bin_edges[None, :], x0[:, None], sd[:, None])
        gy = gaussian_1d_integral_new(p.tran_diff_bin_edges[None, :], y0[:, None], sd[:, None])
        g2 = gx[:, :, None] * gy[:, None, :]
        nb = p.tran_diff_bin_edges.shape[0] / 2 - 1
        X, Y = jnp.mgrid[-nb:nb + 1, -nb:nb + 1]; sh = jnp.vstack([X.ravel(), Y.ravel()]).T
        ne = jnp.repeat(e[:, None, :], sh.shape[0], axis=1)
        ne = ne.at[:, :, fields.index('x')].add(sh[:, 0] * bw).at[:, :, fields.index('y')].add(sh[:, 1] * bw)
        ne = ne.at[:, :, fields.index('n_electrons')].multiply(g2.reshape(-1, sh.shape[0]))
        return ne.reshape(-1, e.shape[1])

    def wfs_of(p, el, up):                             # build_wfs of fig_mc_sampling.py
        pIDs = get_pixels(p, el, fields).ravel()
        xp, yp, plane, _ = id2pixel(p, pIDs)
        t0, sig = current_mc(p, el, get_pixel_coordinates(p, xp, yp, plane), fields)
        ren = jnp.searchsorted(up, pIDs); inr = jnp.clip(ren, 0, up.shape[0] - 1)
        ok = (ren < up.shape[0]) & (up[inr] == pIDs); ren = jnp.where(ok, ren, 0)
        return accumulate_signals_parametrized(jnp.zeros((up.shape[0], NT)), jnp.where(ok[:, None], sig, 0.), ren, t0 - sig.shape[1])

    pc_, pf_, mf_ = make_params(False, 0.01), make_params(False, 0.001), make_params(True, 0.001)
    e_pc, e_pf = proj(pc_, tr_c), proj(pf_, tr_f)
    plist = [jnp.unique(get_pixels(pf_, e_pf, fields).ravel())]
    for s in range(6):
        em, _ = simulate_drift(mf_, tr_f, fields, jax.random.key(500 + s)); plist.append(jnp.unique(get_pixels(mf_, em, fields).ravel()))
    up = jnp.unique(jnp.concatenate(plist)); up = up[up >= 0]
    def summarise(p, w):
        q = np.asarray(w).sum(1) * ts
        integ, ticks = get_adc_values(p, w[:, 1:], jax.random.key(0))
        hit = np.asarray(integ)[:, 0] > 0
        return q, np.where(hit, np.asarray(ticks)[:, 0], np.nan), np.cumsum(np.asarray(w), 1) * ts
    w_pc, w_pf = wfs_of(pc_, e_pc, up), wfs_of(pf_, e_pf, up)
    q_pc, t_pc, c_pc = summarise(pc_, w_pc); q_pf, t_pf, c_pf = summarise(pf_, w_pf)
    bright = int(np.argmax(q_pf))
    q_mc, t_mc, c_mc = [], [], []
    for s in range(N_SEEDS):
        em, _ = simulate_drift(mf_, tr_f, fields, jax.random.key(1000 + s))
        q, t, c = summarise(mf_, wfs_of(mf_, em, up)); q_mc.append(q); t_mc.append(t); c_mc.append(c[bright])
        print(f'MC seed {s}: total {q.sum():.0f} e- (projection {q_pf.sum():.0f})', flush=True)
    pn = make_params(False, 0.001, noise=True); t_noise = []
    for s in range(N_NOISE):
        integ, ticks = get_adc_values(pn, w_pf[:, 1:], jax.random.key(2000 + s))
        t_noise.append(np.where(np.asarray(integ)[:, 0] > 0, np.asarray(ticks)[:, 0], np.nan))
    np.savez_compressed(CACHE, up=np.asarray(up), q_pc=q_pc, q_pf=q_pf, t_pc=t_pc, t_pf=t_pf, q_mc=np.array(q_mc), t_mc=np.array(t_mc),
                        t_noise=np.array(t_noise), cum_pc=c_pc[bright], cum_pf=c_pf[bright], cum_mc=np.array(c_mc), bright=bright,
                        threshold=float(truth.DISCRIMINATION_THRESHOLD), length_cm=L, ts=ts)
    print('wrote', CACHE, f'{len(np.asarray(up))} pixels, {L:.1f} cm')


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = dict(np.load(CACHE))
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
    C_P, C_M, C_N = '#0072B2', '#D55E00', '#7F7F7F'
    qref = z['q_pf']; sel = qref > 2e3
    mc_sd = z['q_mc'].std(0, ddof=1); dproj = np.abs(z['q_pc'] - qref)
    hit = np.isfinite(z['t_pf'])
    with np.errstate(all='ignore'):
        mc_tsd = np.nanstd(z['t_mc'], 0, ddof=1); noise_tsd = np.nanstd(z['t_noise'], 0, ddof=1)
    dtp = np.abs(z['t_pc'] - z['t_pf'])
    # ---- numbers for the text
    n = dict(length_cm=float(z['length_cm']), pixels_gt2ke=int(sel.sum()), hit_pixels=int(hit.sum()),
             mc_rel_sd_median=float(np.median(mc_sd[sel] / qref[sel])), mc_rel_sd_range=[float(np.percentile(mc_sd[sel] / qref[sel], 10)), float(np.percentile(mc_sd[sel] / qref[sel], 90))],
             mc_sd_reset_median=float(np.median(mc_sd[sel] / RESET)), mc_sd_reset_max=float(np.max(mc_sd[sel] / RESET)),
             proj_rel_median=float(np.median(dproj[sel] / qref[sel])), proj_abs_max=float(dproj[sel].max()), proj_abs_median=float(np.median(dproj[sel])),
             mc_tsd_median=float(np.nanmedian(mc_tsd[hit])), mc_tsd_p90=float(np.nanpercentile(mc_tsd[hit], 90)),
             noise_tsd_median=float(np.nanmedian(noise_tsd[hit])), proj_tick_changed_frac=float(np.mean(dtp[hit] > 0)),
             mc_mean_bias_rel_median=float(np.median((z['q_mc'].mean(0)[sel] - qref[sel]) / qref[sel])))
    print(json.dumps(n, indent=1)); json.dump(n, open(CACHE.replace('.npz', '_numbers.json'), 'w'), indent=1)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ts = float(z['ts']); thr = float(z['threshold'])
    cum = z['cum_pf']; it = np.argmax(cum >= thr) if (cum >= thr).any() else len(cum) // 2
    tt = (np.arange(len(cum)) - it) * ts
    for k, cm in enumerate(z['cum_mc']):
        a.plot(tt, cm / 1e3, color=C_M, lw=0.4, alpha=0.35, label='Monte Carlo, 10 $\\mu$m (40 placements)' if k == 0 else None)
    a.plot(tt, z['cum_pc'] / 1e3, color=C_P, lw=1.3, label='projection, 100 $\\mu$m')
    a.plot(tt, cum / 1e3, color='k', lw=0.8, ls='--', label='projection, 10 $\\mu$m (converged)')
    a.axhline(thr / 1e3, color=C_N, ls=':', lw=0.7); a.text(1.0, thr / 1e3 * 0.80, 'threshold', fontsize=6, color=C_N)
    a.set_xlim(-1.5, 1.5); a.set_ylim(0, 3 * thr / 1e3); a.set_xlabel('time from threshold crossing ($\\mu$s)'); a.set_ylabel('cumulative charge on pixel (ke$^-$)')
    a.legend(loc='upper left', frameon=False, fontsize=5.6); a.text(0.95, 0.06, '(a)', transform=a.transAxes, ha='right')
    x = qref[sel] / 1e3
    b.scatter(x, mc_sd[sel] / RESET, s=7, color=C_M, lw=0, label='Monte Carlo, 10 $\\mu$m: spread')
    b.scatter(x, np.maximum(dproj[sel], 1e-1) / RESET, s=7, color=C_P, lw=0, label='projection, 100 $\\mu$m: deviation')
    b.axhline(1, color=C_N, ls=':', lw=0.7); b.text(x.min(), 1.15, 'reset noise', fontsize=6, color=C_N)
    b.set_xscale('log'); b.set_yscale('log'); b.set_ylim(1e-4, 3)
    b.set_xlabel('pixel charge (ke$^-$)'); b.set_ylabel('charge error / reset noise'); b.legend(loc='center right', frameon=False)
    b.text(0.05, 0.92, '(b)', transform=b.transAxes)
    xh = qref[hit] / 1e3
    c.scatter(xh, noise_tsd[hit], s=7, color=C_N, lw=0, label='readout noise: spread')
    c.scatter(xh, mc_tsd[hit], s=7, color=C_M, lw=0, label='Monte Carlo, 10 $\\mu$m: spread')
    c.scatter(xh, dtp[hit], s=9, color=C_P, lw=0, marker='s', label='projection, 100 $\\mu$m: change')
    c.set_xscale('log'); c.set_ylim(-0.05, None); c.set_xlabel('pixel charge (ke$^-$)'); c.set_ylabel('threshold crossing (ticks)')
    c.legend(loc='upper right', frameon=False); c.text(0.05, 0.92, '(c)', transform=c.transAxes)
    for ax in (a, b, c): ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight'); print('saved', OUT)


if __name__ == '__main__':
    compute() if '--compute' in sys.argv else plot()
