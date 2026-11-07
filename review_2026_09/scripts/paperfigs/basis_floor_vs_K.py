"""Representation floor of the trajectory basis vs the number of modes (12 Oct, Pierre: "is the position freedom enough?").

The fit's geometry per track: nominal straight line between the (true) endpoints + K sine modes per transverse direction,
displacement(alpha) = sum_k c_k sin(pi k alpha), alpha in [0, 1] along the track; e1 = pixel-plane direction, e2 = the
drift-carrying one (fit_params._chain_warp_spline, 'drift' frame). Production: K = clip(round(L / 40 cm), 3, 12), so K = 3
for every track shorter than 140 cm; 2K coefficients per track (K for tracks whose drift coordinate is held straight).

For K = 1..20, every true trajectory (true_positions of the chopped rows, matched by track id; GEANT4 steps, median 2.3 cm)
is least-squares projected onto the basis, separately along e1 and e2. Reported per track: mean 3-D distance between the
projection and the truth (the along-track component is not modelled by the basis and is included), rms along e2; plus the
residual profile against alpha (middle vs ends of the track). This is the best any fit with that basis could reach with
perfect information -- the representation floor, not a fit result. Track contexts and truth come from the production
checkpoints (independent of the fit). Also: the production fitted residual profile along alpha, for comparison."""
import os, sys, glob, json, pickle
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO)
for k, v in (('LARND_CHAIN_FRAME', 'drift'), ('LARND_SPLINE_KMIN', '3')): os.environ.setdefault(k, v)
import numpy as np
import optimize.fit_params as fp
SAMPLES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1,2,3,4').split(',')]
OUT = '/sdf/scratch/neutrino/pgranger/review_2026_09/BASISK'; os.makedirs(OUT, exist_ok=True)
KS = list(range(1, 21)); ABINS = np.linspace(0, 1, 11)

def parse_tp(tp):
    out = {}
    for k, v in tp.items():
        if isinstance(k, str): k = tuple(float(x) for x in k.strip('()').split(','))
        out[tuple(float(x) for x in k)] = np.asarray(v, float)
    return out

rows = []                       # per track: L, Kprod, frozen, d3[K], e2rms[K]
prof = {K: [[] for _ in range(len(ABINS) - 1)] for K in (1, 2, 3, 4, 6, 8, 12)}   # 3-D residual vs alpha
fitprof = [[] for _ in range(len(ABINS) - 1)]                                   # production fitted residual vs alpha
nparams = 0
for s in SAMPLES:
    ck = pickle.load(open(sorted(glob.glob(f'{REPO}/fit_result/sci_full_STRFRZ50{s}/history_iter10000_*.pkl'))[0], 'rb'))
    for b, ctxs in ck['chain_contexts'].items():
        tp = parse_tp(ck['true_positions'][b]); states = ck['chain_cache'].get(b, [])
        for c, st in zip(ctxs, states):
            tid = tuple(float(x) for x in c['track_id']); t = tp.get(tid)
            if t is None: continue
            L = float(c['total_len']); al = np.asarray(c['alpha_mid'], float); m = min(len(al), len(t)); al, t = al[:m], t[:m]
            u0 = np.asarray(fp._dir_from_angles(c['theta0_i'], c['phi0_i']), float); e1, e2 = (np.asarray(x, float) for x in fp._transverse_frame(u0))
            nom = np.asarray(c['x0_fixed'], float)[None, :] + al[:, None] * L * u0[None, :]
            d = t - nom; d1, d2, du = d @ e1, d @ e2, d @ u0
            ang = np.asarray(st['angles'], float); Kp = len(ang) // 2
            frozen = bool(np.abs(ang[Kp:]).max() == 0.0)
            nparams += Kp if frozen else 2 * Kp
            d3, r2 = [], []
            for K in KS:
                B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
                c1 = np.linalg.lstsq(B, d1, rcond=None)[0]; c2 = np.linalg.lstsq(B, d2, rcond=None)[0]
                res = np.stack([d1 - B @ c1, d2 - B @ c2, du], 1)
                dist = 1e4 * np.linalg.norm(res, axis=1)
                d3.append(float(dist.mean())); r2.append(float(1e4 * np.sqrt(np.mean((d2 - B @ c2) ** 2))))
                if K in prof:
                    idx = np.clip(np.digitize(al, ABINS) - 1, 0, len(ABINS) - 2)
                    for j in range(len(ABINS) - 1):
                        if np.any(idx == j): prof[K][j].append(float(dist[idx == j].mean()))
            # production fitted geometry: same basis with the fitted coefficients (Kp modes)
            B = np.sin(np.outer(al, np.pi * np.arange(1, Kp + 1)))
            fres = np.stack([d1 - B @ ang[:Kp], d2 - B @ ang[Kp:], du], 1)
            fd = 1e4 * np.linalg.norm(fres, axis=1); idx = np.clip(np.digitize(al, ABINS) - 1, 0, len(ABINS) - 2)
            for j in range(len(ABINS) - 1):
                if np.any(idx == j): fitprof[j].append(float(fd[idx == j].mean()))
            rows.append(dict(s=s, L=L, Kp=Kp, frozen=frozen, d3=d3, e2rms=r2, fit_d3=float(fd.mean()), du=float(1e4 * np.abs(du).mean())))
    print(f'sample {s}: {sum(1 for r in rows if r["s"] == s)} tracks', flush=True)

D = np.array([r['d3'] for r in rows]); E2 = np.array([r['e2rms'] for r in rows]); Lr = np.array([r['L'] for r in rows])
print(f'\n{len(rows)} tracks, {len(SAMPLES)} samples; production K per track: ' + str({int(k): int(v) for k, v in zip(*np.unique([r["Kp"] for r in rows], return_counts=True))})
      + f'; geometry parameters {nparams} ({nparams/len(SAMPLES):.0f} per sample, {nparams/len(rows):.2f} per track)')
print(f'track length: median {np.median(Lr):.1f} cm, mean {Lr.mean():.1f}; along-track |du| (not modelled) median {np.median([r["du"] for r in rows]):.1f} um')
print('K   mean3D(median track)  mean3D(mean)  length-weighted  p90    e2 rms median')
for i, K in enumerate(KS):
    print(f'{K:2d}  {np.median(D[:, i]):8.1f}  {D[:, i].mean():10.1f}  {np.average(D[:, i], weights=Lr):10.1f}  {np.percentile(D[:, i], 90):8.1f}  {np.median(E2[:, i]):8.1f}')
fd = np.array([r['fit_d3'] for r in rows]); print(f'production FIT (K=3, from data): median {np.median(fd):.1f}  mean {fd.mean():.1f}  length-weighted {np.average(fd, weights=Lr):.1f}')
for lo, hi in ((0, 30), (30, 60), (60, 100), (100, 200)):
    m = (Lr >= lo) & (Lr < hi)
    print(f'  L {lo}-{hi} cm (n={m.sum()}): floor median K=3 {np.median(D[m, 2]):.0f}  K=6 {np.median(D[m, 5]):.0f}  K=12 {np.median(D[m, 11]):.0f}  fit {np.median(fd[m]):.0f} um')
print('residual profile vs alpha (median over tracks of the bin-mean 3-D distance, um):')
print('  alpha bins: ' + ' '.join(f'{0.5*(a+b):.2f}' for a, b in zip(ABINS[:-1], ABINS[1:])))
for K in prof: print(f'  floor K={K:2d}: ' + ' '.join(f'{np.median(v):5.0f}' for v in prof[K]))
print('  prod. fit  : ' + ' '.join(f'{np.median(v):5.0f}' for v in fitprof))
json.dump(dict(samples=SAMPLES, Ks=KS, d3=D.tolist(), e2rms=E2.tolist(), L=Lr.tolist(), fit_d3=fd.tolist(),
               prof={str(K): [float(np.median(v)) for v in prof[K]] for K in prof}, fitprof=[float(np.median(v)) for v in fitprof],
               abins=ABINS.tolist(), nparams=nparams), open(f'{OUT}/basis_floor.json', 'w'))
print('saved', f'{OUT}/basis_floor.json')
