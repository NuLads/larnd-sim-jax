"""R089 job: how good is the coarse treatment of the induction on neighboring pixels? (21 Oct 2026)
Variants (neigh_exact.py): prod, P (re-assembled production, sanity), L (diffusion template on the center-point
induction), T (each transverse bin induces at its own position, no diffusion template), E (exact: both).
  --part readout : waveform and probabilistic-readout differences on the Fig. 2/4 batch (load(200,1,1), batch 0) and on
                   the 48.5 cm test track of Fig. mc_sampling; timing of each variant
  --part bias    : first-order calibration shift dtheta = -H^-1 sum_b [g_k(b) - g_prod(b)], g = gradient at the truth of the
                   production loss against stochastic targets simulated with variant k (same noise seed as the production
                   target, common random numbers); H = per-batch Hessians of PAPERJOBS/hessian.json (same batches, seeds)
usage: python run_neigh.py --part readout|bias --out /abs/path.json [--nb 60] [--chunk 512]"""
import argparse, json, os, sys, time
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH')
from paper_common import *                      # chdir to the repo: absolute output paths only
from neigh_exact import simulate_wfs_variant
from larndsim.fee_jax import get_adc_values_average_noise_chunked, digitize, get_average_hit_values
from larndsim.sim_jax import simulate_stochastic
ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--nb', type=int, default=60); ap.add_argument('--chunk', type=int, default=512); a = ap.parse_args()
truth, response = params(); ts = float(truth.t_sampling); SR = float(truth.RESET_NOISE_CHARGE)
KINDS = ['prod', 'P', 'L', 'T', 'E']
print('jax', jax.devices(), 'reset noise', SR, flush=True)


def wfs_all(tracks, fields, kinds=KINDS, timing=None):
    out = {}
    for k in kinds:
        for rep in range(2 if timing is not None else 1):     # second call timed (compiled)
            t0 = time.perf_counter(); w, up = simulate_wfs_variant(truth, response, tracks, fields, k, chunk=a.chunk)
            jax.block_until_ready(w); dt = time.perf_counter() - t0
        if timing is not None: timing[k] = dt
        out[k] = (w, up)
    return out


def readout(w):
    lp, ch = get_adc_values_average_noise_chunked(truth, w, chunk_size=128)
    Et, Eadc, lam = get_average_hit_values(lp, digitize(truth, ch))
    return np.asarray(lam), np.asarray(Et), np.asarray(Eadc)


def q50(x): return float(np.median(x)) if len(x) else None
def q68(x): lo, hi = np.percentile(x, [16, 84]) if len(x) else (np.nan, np.nan); return float(0.5 * (hi - lo))


def compare(tracks, fields, tag, res):
    tm = {}; W = wfs_all(tracks, fields, timing=tm)
    up = np.asarray(W['prod'][1]); valid = up >= 0
    w0 = np.asarray(W['prod'][0])[valid]; q0 = np.cumsum(w0, 1) * ts
    lam0, Et0, Ea0 = readout(jnp.asarray(w0))
    # first threshold crossing without noise (production), to evaluate dQ where it sets the hit time
    thr = float(truth.DISCRIMINATION_THRESHOLD)
    cross = np.where((q0 >= thr).any(1), (q0 >= thr).argmax(1), -1)
    r = dict(npix=int(valid.sum()), length_cm=track_length_cm(tracks, fields), time_s=tm, kinds={})
    for k in KINDS[1:]:
        wk = np.asarray(W[k][0])[valid]; dq = np.cumsum(wk, 1) * ts - q0
        mx = np.abs(dq).max(1); fin = dq[:, -1]
        atc = np.array([dq[i, c] if c >= 0 else np.nan for i, c in enumerate(cross)])
        lam, Et, Ea = readout(jnp.asarray(wk))
        d = dict(maxdq_e=dict(median=q50(mx), p90=float(np.percentile(mx, 90)), p99=float(np.percentile(mx, 99)), max=float(mx.max())),
                 dq_at_crossing_e=dict(median_abs=q50(np.abs(atc[np.isfinite(atc)])), p99_abs=float(np.nanpercentile(np.abs(atc), 99)),
                                       max_abs=float(np.nanmax(np.abs(atc)))),
                 final_dq_rel=dict(median=q50(fin / np.maximum(q0[:, -1], 1)), p99_abs=float(np.percentile(np.abs(fin / np.maximum(q0[:, -1], 1)), 99))),
                 expected_hits=dict(prod=float(lam0.sum()), variant=float(lam.sum()), rel=float(lam.sum() / lam0.sum() - 1)))
        for n in (0, 1):
            m = (lam0[:, n] > 0.5) & (lam[:, n] > 0.5)
            dl = lam[:, n] - lam0[:, n]; dt = (Et - Et0)[m, n]; da = (Ea - Ea0)[m, n]
            d[f'hit{n}'] = dict(dlam_max_abs=float(np.abs(dl).max()), dlam_p99_abs=float(np.percentile(np.abs(dl), 99)),
                                n_dlam_gt_0p01=int((np.abs(dl) > 0.01).sum()), dt_half68_ticks=q68(dt), dt_median_ticks=q50(dt),
                                dt_max_abs_ticks=float(np.abs(dt).max()) if m.any() else None, dadc_half68=q68(da),
                                dadc_median=q50(da), dadc_max_abs=float(np.abs(da).max()) if m.any() else None, n=int(m.sum()))
        r['kinds'][k] = d
        print(f'[{tag}] {k}: maxdQ median {d["maxdq_e"]["median"]:.0f} p99 {d["maxdq_e"]["p99"]:.0f} max {d["maxdq_e"]["max"]:.0f} e-; '
              f'dQ at crossing median {d["dq_at_crossing_e"]["median_abs"]:.0f} e-; hits {100*d["expected_hits"]["rel"]:+.3f}%; '
              f'first hit dt half68 {d["hit0"]["dt_half68_ticks"]:.3f} median {d["hit0"]["dt_median_ticks"]:+.3f} ticks, '
              f'dADC half68 {d["hit0"]["dadc_half68"]:.3f}; |dlam| max {d["hit0"]["dlam_max_abs"]:.3f}/{d["hit1"]["dlam_max_abs"]:.3f}', flush=True)
        np.savez_compressed(a.out.replace('.json', f'_{tag}_{k}.npz'), lam=lam, Et=Et, Ea=Ea, maxdq=mx, atc=atc, fin=fin)
    np.savez_compressed(a.out.replace('.json', f'_{tag}_prod.npz'), lam=lam0, Et=Et0, Ea=Ea0, q=q0[:, -1], cross=cross)
    print(f'[{tag}] timing (s, compiled): ' + ' '.join(f'{k} {v:.2f}' for k, v in tm.items()), flush=True)
    res[tag] = r


res = dict(part=a.part, chunk=a.chunk, git=os.popen(f'git -C {REPO} rev-parse --short HEAD').read().strip(),
           dirty=bool(os.popen(f'git -C {REPO} status --porcelain src/larndsim').read().strip()))
if a.part == 'readout':
    ds_sim, ds_tgt, fields = load(200., 1, 1)
    compare(jnp.asarray(ds_tgt[0]), fields, 'fig2batch', res); json.dump(res, open(a.out, 'w'), indent=1)
    from optimize.dataio import TracksDataset
    ds = TracksDataset(filename='/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/input_49cm.h5', nevents=None,
                       max_nbatch=None, swap_xz=True, random_nevents=False, data_seed=42, max_batch_len=1000., print_input=False,
                       chopped=True, pad=False, electron_sampling_resolution=0.01, live_selection=False,
                       use_dedx_density=False, dedx_density_mode='histogram')
    compare(jnp.asarray(np.asarray(ds[0])), ds.get_track_fields(), 'track49', res)
    json.dump(res, open(a.out, 'w'), indent=1)

elif a.part == 'bias':
    H = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian.json'))
    Hb = {b['batch']: b for b in H['batches'] if 'H' in b}
    ds_sim, ds_tgt, fields = load(200., 400, 1)
    sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
    res['batches'] = []
    for i in range(min(a.nb, len(ds_tgt))):
        t0 = time.perf_counter()
        tracks = jnp.asarray(ds_tgt[i]); Lb = track_length_cm(tracks, fields)
        if i not in Hb or abs(Hb[i]['length_cm'] - Lb) > 0.5:
            print(f'[batch {i}] skipped: no matching Hessian batch', flush=True); continue
        try:
            W = wfs_all(tracks, fields, kinds=['prod', 'L', 'T', 'E'])
            gs = {}
            for k, (w, up) in W.items():
                adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(truth, w, up, rngseed=i + 1)
                tgt = dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix)
                f = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
                g = jax.grad(f)(truth)
                gs[k] = [float(getattr(g, j)) * TRUTH[j] for j in PARAMS]
                gs[k + '_nhits'] = int((np.asarray(pix) >= 0).sum())
            res['batches'].append(dict(batch=i, length_cm=Lb, g=gs))
            dE = np.array(gs['E']) - np.array(gs['prod'])
            print(f'[batch {i}] {Lb:.0f} cm  g_prod ' + ' '.join(f'{x:+.2e}' for x in gs['prod']) + '  dg_E ' +
                  ' '.join(f'{x:+.2e}' for x in dE) + f'  hits prod/E {gs["prod_nhits"]}/{gs["E_nhits"]}  {time.perf_counter()-t0:.0f} s', flush=True)
        except Exception as e:
            print(f'[batch {i}] FAILED {repr(e)[:300]}', flush=True)
        json.dump(res, open(a.out, 'w'))
    ok = res['batches']; Hs = sum(np.array(Hb[b['batch']]['H']) for b in ok); Hs = 0.5 * (Hs + Hs.T); Hi = np.linalg.inv(Hs)
    summ = {}
    for k in ['L', 'T', 'E']:
        D = np.array([np.array(b['g'][k]) - np.array(b['g']['prod']) for b in ok])
        dth = -Hi @ D.sum(0); C = len(D) * np.cov(D.T); s = np.sqrt(np.diag(Hi @ C @ Hi))
        summ[k] = dict(shift_rel=dth.tolist(), shift_rel_err=s.tolist())
        print(f'{k}: first-order shift (%) ' + ' '.join(f'{p} {100*x:+.3f}+-{100*e:.3f}' for p, x, e in zip(PARAMS, dth, s)), flush=True)
    res['summary'] = dict(n_batches=len(ok), length_m=sum(b['length_cm'] for b in ok) / 100, shifts=summ,
                          floor_398m=json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian_summary.json'))['floor'])
    json.dump(res, open(a.out, 'w'), indent=1)
print('done', flush=True)
