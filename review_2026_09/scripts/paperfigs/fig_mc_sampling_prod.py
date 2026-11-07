"""Fig. mc_sampling, version 3 (12 Oct 2026): Monte Carlo packets against the analytic projection, both in the PRODUCTION
simulation path (simulate_wfs: response_44 LUT, pre-convolved longitudinal templates, 4 neighbour rings, production
electronics), on the 48.5 cm of muon track of the step-convergence numbers (SAMPLING_STEP/input_49cm.h5).

Why not the cross-check path of versions 1-2 (fig_mc_sampling.py, fig_mc_sampling_track.py): there, packets go through
the old parametric current model (detsim_jax.current_model) and the projection through current_model_diff; the two
differ by ~0.9 ticks in threshold crossing even with no longitudinal diffusion (checked 12 Oct, NOLONG run), an artefact
of that path, not of sampling.

Monte Carlo emulation, as the reference simulation does it (one packet per 10 um step, one Gaussian draw per packet):
the track is chopped at 0.001 cm; each piece is displaced by (dx, dy, dz) ~ N(0, sigma_T, sigma_T, sigma_L), with
sigma = sqrt(2 D t_drift) from its own drift time (t_drift = |z - z_anode| / v_drift), and the displaced pieces are
simulated with diffusion switched off (D_T = D_L = 1e-12), so that each piece lands whole in one sub-pixel bin and
arrives at one time. Everything else -- LUT, neighbours, electronics -- is the production code.
Converged answer: the projection at 0.001 cm. Ours: the projection at 0.01 cm.

  --compute (GPU job) writes the cache; (default) plots and prints the numbers quoted in the text."""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
CACHE = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/mc_sampling_prod_cache.npz'
SRC = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/input_49cm.h5'
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/mc_sampling'
N_SEEDS, N_NOISE = 40, 200
RESET = 900.0


def compute():
    sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
    import paper_common as pc
    import jax, jax.numpy as jnp
    from optimize.dataio import TracksDataset
    from larndsim.sim_jax import simulate_wfs, simulate_stochastic
    from larndsim.consts_jax import get_vdrift
    truth, response = pc.params()
    ts = float(truth.t_sampling)

    def load(r):
        ds = TracksDataset(filename=SRC, nevents=None, max_nbatch=None, swap_xz=True, random_nevents=False, data_seed=42,
                           max_batch_len=1000., print_input=False, chopped=True, pad=False, electron_sampling_resolution=r,
                           live_selection=False, use_dedx_density=False, dedx_density_mode='histogram')
        return ds.get_track_fields(), np.asarray(ds[0])
    fields, tr_c = load(0.01); fields_f, tr_f = load(0.001); assert fields == fields_f
    L = float(tr_c[:, fields.index('dx')].sum())
    nf = truth.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)

    def run(p, tracks, r, rngseed=0, noise=False):
        p = p.replace(electron_sampling_resolution=r)
        wfs, up = simulate_wfs(p, response, jnp.asarray(tracks), fields)
        pe = p if noise else p.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
        adcs, x, y, z, ticks, hp, ev, hpix = simulate_stochastic(pe, wfs, up, rngseed=rngseed)
        up = np.asarray(up); w = np.asarray(wfs) * ts
        first = {}
        for pid, t in zip(np.asarray(hpix), np.asarray(ticks)):
            if pid >= 0 and (int(pid) not in first or t < first[int(pid)]): first[int(pid)] = float(t)
        return dict(up=up, q=w.sum(1), cum=np.cumsum(w, 1), first=first), (wfs, up)

    ref, (w_ref, up_ref) = run(truth, tr_f, 0.001)
    ours, _ = run(truth, tr_c, 0.01)
    pix = [int(u) for u in ref['up'] if u >= 0]
    idx = lambda d: {int(u): i for i, u in enumerate(d['up']) if u >= 0}
    iref, iours = idx(ref), idx(ours)
    bright = max(pix, key=lambda k: ref['q'][iref[k]])

    # Monte Carlo packets: displace each 10 um piece by its own Gaussian draw, then simulate without diffusion
    vd = float(get_vdrift(truth)); DT, DL = float(truth.tran_diff), float(truth.long_diff)
    zf = fields.index('z'); plane_f = fields.index('pixel_plane') if 'pixel_plane' in fields else None
    borders = np.asarray(truth.tpc_borders)
    # drift time of each piece: distance to the nearer anode of its TPC (the anodes are the z=borders[:,2,0] planes)
    zz = tr_f[:, zf]
    z_anodes = borders[:, 2, 0]
    t_drift = np.min(np.abs(zz[:, None] - z_anodes[None, :]), axis=1) / vd
    sT, sL = np.sqrt(2 * DT * t_drift), np.sqrt(2 * DL * t_drift)
    nodiff = truth.replace(tran_diff=1e-12, long_diff=1e-12)
    cols = {c: [fields.index(k) for k in (c, c + '_start', c + '_end') if k in fields] for c in ('x', 'y', 'z')}
    q_mc = np.full((N_SEEDS, len(pix)), 0.0); t_mc = np.full((N_SEEDS, len(pix)), np.nan); cum_mc = []
    rng = np.random.default_rng(12345)
    for s in range(N_SEEDS):
        t = tr_f.copy()
        d = rng.normal(size=(len(t), 3)) * np.stack([sT, sT, sL], 1)
        for k, c in enumerate(('x', 'y', 'z')):
            for j in cols[c]: t[:, j] += d[:, k]
        mc, _ = run(nodiff, t, 0.001)
        im = idx(mc)
        for n, k in enumerate(pix):
            if k in im: q_mc[s, n] = mc['q'][im[k]]
            if k in mc['first']: t_mc[s, n] = mc['first'][k]
        cum_mc.append(mc['cum'][im[bright]] if bright in im else np.zeros_like(ref['cum'][0]))
        print(f'MC seed {s}: total {mc["q"].sum():.0f} e- (projection {ref["q"].sum():.0f}); {len(mc["first"])} hit pixels '
              f'(projection {len(ref["first"])})', flush=True)
    # readout noise alone, on the converged waveform
    t_noise = np.full((N_NOISE, len(pix)), np.nan)
    for s in range(N_NOISE):
        adcs, x, y, z, ticks, hp, ev, hpix = simulate_stochastic(truth, w_ref, up_ref, rngseed=2000 + s)
        first = {}
        for pid, tk in zip(np.asarray(hpix), np.asarray(ticks)):
            if pid >= 0 and (int(pid) not in first or tk < first[int(pid)]): first[int(pid)] = float(tk)
        for n, k in enumerate(pix):
            if k in first: t_noise[s, n] = first[k]
    q_ref = np.array([ref['q'][iref[k]] for k in pix]); q_ours = np.array([ours['q'][iours[k]] if k in iours else 0. for k in pix])
    t_ref = np.array([ref['first'].get(k, np.nan) for k in pix]); t_ours = np.array([ours['first'].get(k, np.nan) for k in pix])
    np.savez_compressed(CACHE, pix=np.array(pix), q_ref=q_ref, q_ours=q_ours, t_ref=t_ref, t_ours=t_ours, q_mc=q_mc, t_mc=t_mc,
                        t_noise=t_noise, cum_ref=ref['cum'][iref[bright]], cum_ours=ours['cum'][iours[bright]], cum_mc=np.array(cum_mc),
                        threshold=float(truth.DISCRIMINATION_THRESHOLD), length_cm=L, ts=ts)
    print('wrote', CACHE, f'{len(pix)} pixels, {L:.1f} cm')


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = dict(np.load(CACHE))
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
    C_P, C_M, C_N = '#0072B2', '#D55E00', '#7F7F7F'
    qref = z['q_ref']; sel = qref > 2e3
    mc_sd = z['q_mc'].std(0, ddof=1); mc_bias = z['q_mc'].mean(0) - qref; dproj = np.abs(z['q_ours'] - qref)
    hit = np.isfinite(z['t_ref'])
    with np.errstate(all='ignore'):
        mc_tsd = np.nanstd(z['t_mc'], 0, ddof=1); noise_tsd = np.nanstd(z['t_noise'], 0, ddof=1)
        mc_tbias = np.nanmean(z['t_mc'], 0) - z['t_ref']
    dtp = np.abs(z['t_ours'] - z['t_ref'])
    n = dict(length_cm=float(z['length_cm']), pixels_gt2ke=int(sel.sum()), hit_pixels=int(hit.sum()),
             mc_rel_sd_median=float(np.median(mc_sd[sel] / qref[sel])),
             mc_rel_sd_p10_p90=[float(np.percentile(mc_sd[sel] / qref[sel], q)) for q in (10, 90)],
             mc_sd_reset_median=float(np.median(mc_sd[sel] / RESET)), mc_sd_reset_p90=float(np.percentile(mc_sd[sel] / RESET, 90)),
             mc_mean_bias_rel_median=float(np.median(mc_bias[sel] / qref[sel])),
             proj_rel_median=float(np.median(dproj[sel] / qref[sel])), proj_abs_max=float(dproj[sel].max()),
             proj_abs_median=float(np.median(dproj[sel])),
             mc_tsd_median=float(np.nanmedian(mc_tsd[hit])), mc_tsd_p90=float(np.nanpercentile(mc_tsd[hit], 90)),
             mc_tbias_median=float(np.nanmedian(mc_tbias[hit])),
             noise_tsd_median=float(np.nanmedian(noise_tsd[hit])),
             proj_tick_changed_frac=float(np.nanmean(dtp[hit] > 0)), proj_tick_changed_max=float(np.nanmax(dtp[hit])))
    print(json.dumps(n, indent=1)); json.dump(n, open(CACHE.replace('.npz', '_numbers.json'), 'w'), indent=1)
    # two panels (12 Oct): the single-pixel waveform panel was dropped -- on the brightest pixel the packets sit ~1.3%
    # above the projection, an effect of the projection's cloud-centre-pixel limitation (Appendix B), not of sampling
    fig, (b, c) = plt.subplots(1, 2, figsize=(6.4, 2.45))
    x = qref[sel] / 1e3
    b.scatter(x, mc_sd[sel] / RESET, s=8, color=C_M, lw=0, label='packets, 10 $\\mu$m: spread (40 draws)')
    b.scatter(x, np.maximum(dproj[sel], 0.1) / RESET, s=8, color=C_P, lw=0, label='projection: change 10 $\\to$ 100 $\\mu$m')
    b.axhline(1, color=C_N, ls=':', lw=0.7); b.text(x.max(), 1.25, 'reset noise', fontsize=6, color=C_N, ha='right')
    b.set_xscale('log'); b.set_yscale('log'); b.set_ylim(1e-4, 4)
    b.set_xlabel('charge collected on the pixel (ke$^-$)'); b.set_ylabel('charge error / reset noise')
    b.text(0.04, 0.92, '(a)', transform=b.transAxes)
    xh = qref[hit] / 1e3; ch = (dtp[hit] > 0)
    c.scatter(xh, noise_tsd[hit], s=8, color=C_N, lw=0, label='readout noise alone: spread')
    c.scatter(xh, mc_tsd[hit], s=8, color=C_M, lw=0, label='packets, 10 $\\mu$m: spread (40 draws)')
    c.scatter(xh[ch], dtp[hit][ch], s=12, color=C_P, lw=0, marker='s', label='projection: change 10 $\\to$ 100 $\\mu$m')
    c.text(0.97, 0.05, f'projection unchanged on {int((~ch).sum())} of {int(hit.sum())} pixels', transform=c.transAxes,
           ha='right', fontsize=6, color=C_P)
    c.set_xscale('log'); c.set_yscale('log'); c.set_ylim(0.05, 60)
    c.set_xlabel('charge collected on the pixel (ke$^-$)'); c.set_ylabel('threshold crossing (ticks)')
    c.text(0.04, 0.92, '(b)', transform=c.transAxes)
    h = [c.collections[1], c.collections[2], c.collections[0]]
    fig.legend(h, [x.get_label() for x in h], loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False,
               handletextpad=0.3, columnspacing=1.4)
    for ax in (b, c): ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight'); print('saved', OUT)


if __name__ == '__main__':
    compute() if '--compute' in sys.argv else plot()
