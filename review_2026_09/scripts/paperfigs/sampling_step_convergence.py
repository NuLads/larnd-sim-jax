"""Convergence of the track sampling step (electron_sampling_resolution) of the analytic projection (9 Oct 2026).
Same real muon segments (first segments of LARNDSIM_CMP/tiny_input.h5), production parameters (paper_common.params():
TRUTH values, response_44 LUT, 4 neighbour rings, signal_length 150), chopped at 0.01 (production), 0.005, 0.002 and
0.001 cm. Reference = 0.001 cm. Compared per pixel: time-integrated charge, max |difference| of the cumulative charge
waveform (the quantity the discriminator sees), and noise-free hits (count, ADC, tick); scale = 900 e- reset noise.
Run: JAX_PLATFORMS=cpu python sampling_step_convergence.py [max_cm]"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np, h5py
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import paper_common as pc                      # chdirs to the repo
import jax, jax.numpy as jnp
from optimize.dataio import TracksDataset
from larndsim.sim_jax import simulate_wfs, simulate_stochastic
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP'
MAX_CM = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
src = '/sdf/scratch/neutrino/pgranger/review_2026_09/LARNDSIM_CMP/tiny_input.h5'
# take whole segments up to MAX_CM of track
with h5py.File(src, 'r') as f:
    seg = f['segments'][:]; traj = f['trajectories'][:] if 'trajectories' in f else None
keep = np.cumsum(seg['dx']) <= MAX_CM; keep[0] = True
sub = f'{OUT}/input_{MAX_CM:g}cm.h5'
with h5py.File(sub, 'w') as g:
    g.create_dataset('segments', data=seg[keep])
    if traj is not None: g.create_dataset('trajectories', data=traj)
truth, response = pc.params()
noisefree = truth.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
res = {}
for r in (0.01, 0.005, 0.002, 0.001):
    ds = TracksDataset(filename=sub, nevents=None, max_nbatch=None, swap_xz=True, random_nevents=False, data_seed=42,
                       max_batch_len=1000., print_input=False, chopped=True, pad=False, electron_sampling_resolution=r,
                       live_selection=False, use_dedx_density=False, dedx_density_mode='histogram')
    fields = ds.get_track_fields(); tracks = jnp.asarray(ds[0])
    p = truth.replace(electron_sampling_resolution=r)
    wfs, up = simulate_wfs(p, response, tracks, fields)
    adcs, x, y, z, ticks, hp, ev, hpix = simulate_stochastic(noisefree.replace(electron_sampling_resolution=r), wfs, up, rngseed=0)
    up = np.asarray(up); w = np.asarray(wfs) * float(truth.t_sampling)       # electrons per tick
    hits = {}
    for pid, a, t in zip(np.asarray(hpix), np.asarray(adcs), np.asarray(ticks)):
        if pid >= 0: hits.setdefault(int(pid), []).append((float(a), float(t)))
    res[r] = dict(up=up, cum=np.cumsum(w, 1), q=w.sum(1), hits=hits, rows=int(tracks.shape[0]),
                  length=float(np.asarray(tracks)[:, fields.index('dx')].sum()))
    print(f'r={r}: {res[r]["rows"]} pieces, {res[r]["length"]:.2f} cm, {int((up>=0).sum())} pixels, '
          f'{sum(len(v) for v in hits.values())} noise-free hits', flush=True)
ref = res[0.001]; rmap = {int(u): i for i, u in enumerate(ref['up']) if u >= 0}
summary = {}
for r in (0.01, 0.005, 0.002):
    c = res[r]; cmap = {int(u): i for i, u in enumerate(c['up']) if u >= 0}
    common = sorted(set(rmap) & set(cmap)); only = len(set(rmap) ^ set(cmap))
    dq = np.array([c['q'][cmap[k]] - ref['q'][rmap[k]] for k in common])
    qref = np.array([ref['q'][rmap[k]] for k in common])
    dcum = np.array([np.abs(c['cum'][cmap[k]] - ref['cum'][rmap[k]]).max() for k in common])
    big = qref > 2e3
    nh_c = sum(len(v) for v in c['hits'].values()); nh_r = sum(len(v) for v in ref['hits'].values())
    same_n = np.mean([len(c['hits'].get(k, [])) == len(ref['hits'].get(k, [])) for k in common])
    da, dt = [], []
    for k in common:
        hc, hr = c['hits'].get(k, []), ref['hits'].get(k, [])
        for (a1, t1), (a2, t2) in zip(hc, hr): da.append(a1 - a2); dt.append(t1 - t2)
    da, dt = np.array(da), np.array(dt)
    s = dict(pixels_common=len(common), pixels_only_one=only,
             charge_rel_diff_median=float(np.median(np.abs(dq[big] / qref[big]))), charge_rel_diff_max=float(np.max(np.abs(dq[big] / qref[big]))),
             charge_abs_diff_max_e=float(np.abs(dq).max()), cumwave_maxdiff_median_e=float(np.median(dcum)), cumwave_maxdiff_max_e=float(dcum.max()),
             hits=[nh_c, nh_r], same_hit_count_frac=float(same_n),
             adc_diff_max=float(np.abs(da).max()) if len(da) else 0.0, adc_diff_rms=float(np.sqrt(np.mean(da**2))) if len(da) else 0.0,
             tick_diff_nonzero_frac=float(np.mean(dt != 0)) if len(dt) else 0.0, tick_diff_max=float(np.abs(dt).max()) if len(dt) else 0.0)
    summary[r] = s
    print(f'step {r} cm vs 0.001 cm: {json.dumps(s)}', flush=True)
json.dump({str(k): v for k, v in summary.items()}, open(f'{OUT}/summary_{MAX_CM:g}cm.json', 'w'), indent=1)
print('reset noise 900 e-, threshold 5000 e-; one ADC count ~ 1 ke-')
