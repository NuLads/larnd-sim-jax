"""Origin of the K-independent intermediate-probability deficit (analytic hit probability below the sampled frequency).
Two acceptance-related candidates: (a) bookkeeping -- the analytic hit index counts threshold crossings including failed
digitisations, a sampled hit list only digitised ones, so a failed first crossing promotes the next one to sampled hit 1
(moves probability between indices, total unchanged); (b) re-arm noise -- after a failed digitisation the stochastic
electronics re-arm with the uncorrelated noise (500 e) as baseline, the analytic model always with the reset noise (900 e)
(changes the total). Zero uncorrelated noise in BOTH models removes failed digitisations, hence both (a) and (b).
Readout: per-index residuals AND total hits (sampled/analytic) per variant; (a) only => nominal total ratio ~1.
Same comparison as paper_kscan.py, at the production width K=100."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--nb', type=int, default=2); ap.add_argument('--nthrow', type=int, default=500)
ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(400., a.nb); truth, response = params()
sim = LUTProbabilisticSimulation(response)
VARIANTS = {'nominal': {}, 'nouncorr': dict(UNCORRELATED_NOISE_CHARGE=0.)}
EDGES = np.array([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
acc = {v: {s: [] for s in (0, 1)} for v in VARIANTS}; nhits = {v: [0., 0.] for v in VARIANTS}
for i in range(min(a.nb, len(ds_tgt))):
    tracks = jnp.asarray(ds_tgt[i]); wfs, up = simulate_wfs(truth, response, tracks, fields)
    for vname, over in VARIANTS.items():
        P = truth.replace(**over) if over else truth
        counts = {}
        for t in range(a.nthrow):
            _, _, _, _, ticks, hp, ev, pix = simulate_stochastic(P, wfs, up, rngseed=1000003 * (i + 1) + t)
            pix = np.asarray(pix); ok = (pix >= 0) & (np.asarray(hp) > 0)
            u, c = np.unique(pix[ok], return_counts=True)
            for p_, c_ in zip(u.tolist(), c.tolist()): counts.setdefault(p_, []).append(c_)
        pids = np.array(sorted(counts)); nh = {p_: np.array(counts[p_] + [0] * (a.nthrow - len(counts[p_]))) for p_ in pids}
        freq = {s: np.array([np.mean(nh[p_] >= s + 1) for p_ in pids]) for s in (0, 1)}
        pred = sim.predict(P, tracks, fields, None)
        upx = np.asarray(pred['unique_pixels']); lam = np.exp(np.asarray(pred['hit_prob'])).sum(axis=2)
        idx = {p_: j for j, p_ in enumerate(upx.tolist()) if p_ >= 0}
        nhits[vname][0] += float(sum(nh[p_].sum() for p_ in pids)) / a.nthrow          # sampled (digitised) hits per throw, all pixels
        nhits[vname][1] += float(lam[upx >= 0].sum())                                  # analytic expected hits, all pixels and hit indices
        for s in (0, 1):
            for p_, f_ in zip(pids.tolist(), freq[s].tolist()):
                if p_ in idx: acc[vname][s].append((f_, float(lam[idx[p_], s])))
        print(f'[batch {i}] {vname} done', flush=True)
summary = {}
for vname in VARIANTS:
    print(f'{vname}: total hits sampled/throw {nhits[vname][0]:.1f}  analytic {nhits[vname][1]:.1f}  ratio sampled/analytic {nhits[vname][0]/nhits[vname][1]:.5f}', flush=True)
    for s in (0, 1):
        A = np.array(acc[vname][s]); f, l = A[:, 0], A[:, 1]; rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (f >= lo) & (f < hi)
            if m.sum() > 20: rows.append((float(lo), float(hi), int(m.sum()), float((l[m] - f[m]).mean()), float((l[m] - f[m]).std() / np.sqrt(m.sum()))))
        summary[f'{vname}_slot{s}'] = rows
        for r in rows: print(f'  {vname} slot {s} [{r[0]:.2f},{r[1]:.2f}) n={r[2]:5d}  analytic-sampled {r[3]:+.4f} +- {r[4]:.4f}', flush=True)
summary['nhits'] = nhits
json.dump(summary, open(a.out, 'w'), indent=1)
