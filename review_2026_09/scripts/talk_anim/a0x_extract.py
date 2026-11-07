"""Run the production simulation chain (quench, drift, simulate_wfs) on the true segments of the hero event at truth parameters
and save what a01/a02/a03 need into data/a01_03_sim.npz. Prints the coordinate/drift checks."""
import os, sys, pickle, time, numpy as np
t00 = time.time()
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import paper_common as pc
import jax, jax.numpy as jnp
from larndsim.consts_jax import get_vdrift
from larndsim.sim_jax import shift_tracks, simulate_wfs, _signal_inputs, simulate_signals
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.detsim_jax import id2pixel, get_pixel_coordinates, get_hit_z
OUT = '/sdf/scratch/neutrino/pgranger/talk_animations/data/a01_03_sim.npz'
h = pickle.load(open('/sdf/scratch/neutrino/pgranger/talk_animations/data/hero.pkl', 'rb'))
truth, response = pc.params()
ct = h['calib_true']
P = truth.replace(**ct)
print('truth vs paper_common', {k: (ct[k], float(getattr(truth, k))) for k in ct})
print('rows params: nb_neigh', P.number_pix_neighbors, 'sig_len', P.signal_length, 't_samp', P.t_sampling, 'time_interval', P.time_interval, 'esr', P.electron_sampling_resolution,
      'nb_tran', P.nb_tran_diff_bins, 'nbins_per_pix', P.nb_sampling_bins_per_pixel, 'recomb', P.recombination_mode, 'density', P.lArDensity, 'kb', P.kb, 'MeV2e', P.MeVToElectrons, 'temp', P.temperature, 'shift', P.shift_x, P.shift_y, P.shift_z)
print('tpc_borders', np.asarray(P.tpc_borders), 'pitch', P.pixel_pitch, 'npix', P.n_pixels_x, P.n_pixels_y, 'drift_length', P.drift_length, 'size_margin', P.size_margin)
print('response shape', response.shape, 'long_diff_template', np.asarray(P.long_diff_template)[[0, 1, 2, -1]])
vd = float(get_vdrift(P)); print('vdrift cm/us', vd, 'vdrift_static', P.vdrift_static, 'init E vd', float(get_vdrift(P.replace(eField=h['calib_init']['eField']))))
f = tuple(h['tfields']); tr = jnp.asarray(h['true_rows'])
ix = {k: f.index(k) for k in f}
s0 = shift_tracks(P, tr, f); q1 = quench(P, s0, f); d1 = drift(P, q1, f)
q1n, d1n, s0n = map(np.asarray, (q1, d1, s0))
n_ion = s0n[:, ix['dE']] * float(P.MeVToElectrons)
nq = q1n[:, ix['n_electrons']]; nd = d1n[:, ix['n_electrons']]
print('R (n_e/n_ion): mean', (nq / n_ion).mean(), 'min', (nq / n_ion).min(), 'max', (nq / n_ion).max())
print('analytic Birks at mean dEdx', float(P.Ab) / (1 + float(P.kb) * s0n[:, ix['dEdx']].mean() / (float(P.eField) * float(P.lArDensity))))
plane = d1n[:, ix['pixel_plane']].astype(int)
print('planes', np.unique(plane, return_counts=True))
za = np.asarray(P.tpc_borders)[plane, 2, 0]; zc = np.asarray(P.tpc_borders)[plane, 2, 1]
print('anode z per plane', np.unique(za), 'cathode', np.unique(zc), 'sign(cathode-anode)', np.unique(np.sign(zc - za)))
tdrift = d1n[:, ix['t']] - d1n[:, ix['t0']]
sT, sL = d1n[:, ix['tran_diff']], d1n[:, ix['long_diff']]
print('tdrift us', tdrift.min(), tdrift.max(), 'attenuation', (nd / nq).min(), (nd / nq).max(), 'sT cm', sT.min(), sT.max(), 'sL', sL.min(), sL.max())
print('check tdrift=|z-za|/v', np.abs(tdrift - np.abs(d1n[:, ix['z']] - za) / vd).max(), 'check exp', np.abs(nd / nq - np.exp(-tdrift / float(P.lifetime))).max(), 'sum n ion/q/d', n_ion.sum(), nq.sum(), nd.sum())
for tid in np.unique(s0n[:, ix['trackID']]):
    m = s0n[:, ix['trackID']] == tid
    print('track', tid, 'rows', m.sum(), 'len', s0n[m, ix['dx']].sum(), 'dE', s0n[m, ix['dE']].sum(), 'x', s0n[m, ix['x']].min(), s0n[m, ix['x']].max(), 'y', s0n[m, ix['y']].min(), s0n[m, ix['y']].max(), 'z', s0n[m, ix['z']].min(), s0n[m, ix['z']].max(), 'tdrift', tdrift[m].min(), tdrift[m].max(), 'ne/cm', nq[m].sum() / s0n[m, ix['dx']].sum(), 'dedx', (s0n[m, ix['dE']].sum() / s0n[m, ix['dx']].sum()))
t1 = time.time()
wfs, up = simulate_wfs(P, response, tr, f); wfs = np.asarray(wfs); up = np.asarray(up); print('wfs', wfs.shape, up.shape, 'time', time.time() - t1, flush=True)
okp = up >= 0
# pixel mapping
px_, py_, pl_, ev_ = [np.asarray(a) for a in id2pixel(P, jnp.asarray(up))]
xy = np.asarray(get_pixel_coordinates(P, jnp.asarray(px_), jnp.asarray(py_), jnp.asarray(pl_)))
d = h['data']
print('hit plane/evt', np.unique(np.asarray(id2pixel(P, jnp.asarray(np.asarray(d['pixel_id'], dtype=np.int64)))[2])), np.unique(np.asarray(id2pixel(P, jnp.asarray(np.asarray(d['pixel_id'], dtype=np.int64)))[3])))
hx, hy, hpl, hev = [np.asarray(a) for a in id2pixel(P, jnp.asarray(np.asarray(d['pixel_id'], dtype=np.int64)))]
hxy = np.asarray(get_pixel_coordinates(P, jnp.asarray(hx), jnp.asarray(hy), jnp.asarray(hpl)))
print('hit xy check', np.abs(hxy[:, 0] - d['pixel_x']).max(), np.abs(hxy[:, 1] - d['pixel_y']).max())
pixkey = lambda a, b: a * 100000 + b
ukey = pixkey(px_[okp].astype(np.int64), py_[okp].astype(np.int64))
hkey = pixkey(hx.astype(np.int64), hy.astype(np.int64))
print('hit pixels', len(np.unique(hkey)), 'in sim pixels', np.isin(np.unique(hkey), ukey).sum(), 'sim pixels', okp.sum())
# segment -> main pixel
b = np.asarray(P.tpc_borders)[plane]
pitch = float(P.pixel_pitch)
sx = np.floor((d1n[:, ix['x']] - b[:, 0, 0]) / pitch).astype(np.int64); sy = np.floor((d1n[:, ix['y']] - b[:, 1, 0]) / pitch).astype(np.int64)
skey = pixkey(sx, sy)
# HIT TIMING CHECK: per pixel, hits against the drift ticks of the segments landing on it
rows = []
for k in np.unique(hkey):
    m = skey == k
    if m.sum() == 0: continue
    th = np.sort(d['ticks'][hkey == k])
    tseg = tdrift[m] / float(P.t_sampling)
    w = nd[m]
    rows.append((k, th[0], tseg.min(), np.average(tseg, weights=w), tseg.max(), m.sum(), np.abs(d['pixel_z'][hkey == k] - (za[m][0] - np.sign(zc[m][0] - za[m][0]) * 0 + np.sign(zc[m][0] - za[m][0]) * th * 0)).min()))
rows = np.array(rows, dtype=float)
res_min = rows[:, 1] - rows[:, 2]; res_mean = rows[:, 1] - rows[:, 3]
print('pixels matched', len(rows), 'first-hit tick minus MIN segment drift tick: median %.1f  p16 %.1f p84 %.1f' % (np.median(res_min), *np.percentile(res_min, [16, 84])), ' minus MEAN: median %.1f p16 %.1f p84 %.1f' % (np.median(res_mean), *np.percentile(res_mean, [16, 84])))
A = np.vstack([rows[:, 3], np.ones(len(rows))]).T
slope, icpt = np.linalg.lstsq(A, rows[:, 1], rcond=None)[0]; print('fit first-hit tick = %.4f * mean drift tick + %.2f' % (slope, icpt))
# hit pixel_z check
ztest = np.asarray(get_hit_z(P, jnp.asarray(d['ticks']), jnp.asarray(hpl)))
ztest2 = np.asarray(get_hit_z(P, jnp.asarray(d['ticks']), jnp.asarray(hpl), fixed_v=True))
print('pixel_z vs get_hit_z (fitted v)', np.abs(ztest - d['pixel_z']).max(), ' (static v)', np.abs(ztest2 - d['pixel_z']).max())
print('hit z (truth v) range', ztest.min(), ztest.max(), ' stored pixel_z range', d['pixel_z'].min(), d['pixel_z'].max())
# plain z comparison: hit z from truth v against the mean segment z landing on the pixel
zmean = np.array([np.average(d1n[skey == k, ix['z']], weights=nd[skey == k]) for k in rows[:, 0].astype(np.int64)])
zh = np.array([ztest[hkey == k].max() if np.sign(np.asarray(P.tpc_borders)[int(hpl[0]), 2, 1] - np.asarray(P.tpc_borders)[int(hpl[0]), 2, 0]) < 0 else ztest[hkey == k].min() for k in rows[:, 0].astype(np.int64)])
print('first-hit z(truth v) minus mean seg z on pixel: median %.3f cm, p16 %.3f p84 %.3f' % (np.median(zh - zmean), *np.percentile(zh - zmean, [16, 84])))
# peak of waveform vs drift time: waveform peak tick per sim pixel vs mean drift tick
wp = []
for iu in np.where(okp)[0]:
    k = pixkey(np.int64(px_[iu]), np.int64(py_[iu])); m = skey == k
    if m.sum() < 20: continue
    wp.append((wfs[iu].argmax(), np.average(tdrift[m], weights=nd[m]) / float(P.t_sampling), wfs[iu].max()))
wp = np.array(wp); print('waveform argmax minus mean drift tick (pixels with >=20 rows): median %.1f p16 %.1f p84 %.1f (N=%d)' % (np.median(wp[:, 0] - wp[:, 1]), *np.percentile(wp[:, 0] - wp[:, 1], [16, 84]), len(wp)))
# Raw table meaning
raw = np.load('src/larndsim/detector_properties/response_44.npy'); print('raw', raw.shape, raw.dtype, 'sum*0.1 at (0,0)', raw[0, 0].sum() * 0.1, 'at (4,4)', raw[4, 4].sum() * 0.1, '(5,0)', raw[5, 0].sum() * 0.1, '(10,0)', raw[10, 0].sum() * 0.1, '(14,0)', raw[14, 0].sum() * 0.1, '(44,44)', raw[44, 44].sum() * 0.1)
print('peak (0,0)', raw[0, 0].max(), 'argmax', raw[0, 0].argmax(), 'min', raw[0, 0].min(), '(10,0) max', raw[10, 0].max(), raw[10, 0].argmax(), 'min', raw[10, 0].min(), raw[10, 0].argmin())
print('non-zero first tick (0,0)', np.argmax(raw[0, 0] != 0), 'last', 1890 - np.argmax(raw[0, 0, ::-1] != 0), 'last 5', raw[0, 0, -5:])
print('integrated sum map first rows', np.round(raw.sum(-1)[:15, 0] * 0.1, 4))
print('template[0]==raw', np.abs(np.asarray(response[0]) - raw).max())
np.savez_compressed(OUT, raw_tail=raw[:, :, -400:].astype(np.float32), raw_sum=raw.sum(-1).astype(np.float32), raw_cum_full=np.cumsum(raw[[0, 4, 5, 10, 14, 15, 20], :][:, [0, 4, 5, 10]], axis=-1)[..., ::10].astype(np.float32),
                    fields=np.array(f), seg=np.stack([d1n[:, ix[k]] for k in ('x', 'y', 'z', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end', 'dx', 'dE', 'dEdx', 'trackID', 'segment_id')], 1).astype(np.float64),
                    n_ion=n_ion, nq=nq, nd=nd, tdrift=tdrift, sT=sT, sL=sL, plane=plane, sx=sx, sy=sy,
                    hit_x=d['pixel_x'], hit_y=d['pixel_y'], hit_z=d['pixel_z'], hit_ztruth=ztest, hit_t=d['ticks'], hit_adc=d['adcs'], hit_q=d['q'], hit_pid=d['pixel_id'], hix=hx, hiy=hy,
                    up=up[okp], upx=px_[okp], upy=py_[okp], upos=xy[okp], wfs=wfs[okp].astype(np.float32),
                    tpc_borders=np.asarray(P.tpc_borders), pitch=pitch, vd=vd, t_sampling=float(P.t_sampling), MeV2e=float(P.MeVToElectrons), Ab=float(P.Ab), kb=float(P.kb), rho=float(P.lArDensity), eField=float(P.eField),
                    lifetime=float(P.lifetime), DT=float(P.tran_diff), DL=float(P.long_diff), vd_static=float(P.vdrift_static), drift_length=float(P.drift_length))
print('saved', os.path.getsize(OUT) / 1e6, 'MB', time.time() - t00)
