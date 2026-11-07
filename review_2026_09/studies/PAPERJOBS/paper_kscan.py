"""Beam width K: analytic hit probability per (pixel, hit slot) against the frequency in N stochastic throws over the
same waveforms, binned in sampled frequency, for several K (fee_paths_scaling). Does the intermediate-probability
deficit shrink with K?"""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--nb', type=int, default=2); ap.add_argument('--nthrow', type=int, default=500)
ap.add_argument('--ks', default='10,25,50,100,200'); ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(400., a.nb); truth, response = params()
sim = LUTProbabilisticSimulation(response); KS = [int(x) for x in a.ks.split(',')]
EDGES = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
acc = {K: {s: [] for s in (0, 1)} for K in KS}
for i in range(min(a.nb, len(ds_tgt))):
    tracks = jnp.asarray(ds_tgt[i]); wfs, up = simulate_wfs(truth, response, tracks, fields)
    counts = {}
    for t in range(a.nthrow):
        _, _, _, _, ticks, hp, ev, pix = simulate_stochastic(truth, wfs, up, rngseed=1000003 * (i + 1) + t)
        pix = np.asarray(pix); ok = (pix >= 0) & (np.asarray(hp) > 0)
        u, c = np.unique(pix[ok], return_counts=True)
        for p_, c_ in zip(u.tolist(), c.tolist()): counts.setdefault(p_, []).append(c_)
    pids = np.array(sorted(counts)); nh = {p_: np.array(counts[p_] + [0] * (a.nthrow - len(counts[p_]))) for p_ in pids}
    freq = {s: np.array([np.mean(nh[p_] >= s + 1) for p_ in pids]) for s in (0, 1)}
    for K in KS:
        pred = sim.predict(truth.replace(fee_paths_scaling=K), tracks, fields, None)
        upx = np.asarray(pred['unique_pixels']); lam = np.exp(np.asarray(pred['hit_prob'])).sum(axis=2)   # (Npix, Nslots)
        idx = {p_: j for j, p_ in enumerate(upx.tolist()) if p_ >= 0}
        for s in (0, 1):
            for p_, f_ in zip(pids.tolist(), freq[s].tolist()):
                if p_ in idx: acc[K][s].append((f_, float(lam[idx[p_], s])))
        print(f'[batch {i}] K={K} done', flush=True)
summary = {}
for K in KS:
    for s in (0, 1):
        A = np.array(acc[K][s]); f, l = A[:, 0], A[:, 1]; rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (f >= lo) & (f < hi)
            if m.sum() > 20: rows.append((float(lo), float(hi), int(m.sum()), float((l[m] - f[m]).mean()), float((l[m] - f[m]).std() / np.sqrt(m.sum()))))
        summary[f'K{K}_slot{s}'] = rows
        mid = (f >= 0.2) & (f < 0.8)
        print(f'K={K:4d} slot {s}: mean (analytic - sampled) at sampled p in [0.2,0.8): {(l[mid]-f[mid]).mean():+.4f} +- {(l[mid]-f[mid]).std()/np.sqrt(mid.sum()):.4f}  (n={mid.sum()})', flush=True)
json.dump(summary, open(a.out, 'w'), indent=1)
