"""R089 follow-up: the actual calibration end point (not first order) when the targets carry the exact neighbor induction.
Five detector parameters only, track inputs held at their true values, production model and loss. Minimized by Newton
steps with the per-batch Hessians at the truth of PAPERJOBS/hessian.json (same batches and target seeds), the gradient
recomputed at each step, step halving if the loss rises. Targets: stochastic, seed batch+1, from the production waveforms
('prod') or from a variant of neigh_exact.py ('E', 'T', 'L'); the same noise draws for every kind (common random numbers).
usage: python run_fit.py --out /abs/fit.json [--nb 20] [--iters 12] [--kinds prod,E,T,L]"""
import argparse, json, sys, time
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/NEIGH')
from paper_common import *
from neigh_exact import simulate_wfs_variant
from larndsim.sim_jax import simulate_stochastic
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--nb', type=int, default=20)
ap.add_argument('--iters', type=int, default=12); ap.add_argument('--kinds', default='prod,E,T,L'); a = ap.parse_args()
kinds = a.kinds.split(',')
truth, response = params()
H = {b['batch']: b for b in json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS/hessian.json'))['batches'] if 'H' in b}
ds_sim, ds_tgt, fields = load(200., 400, 1)
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
batches, tg = [], {k: [] for k in kinds}
for i in range(len(ds_tgt)):
    if len(batches) >= a.nb: break
    tracks = jnp.asarray(ds_tgt[i]); Lb = track_length_cm(tracks, fields)
    if i not in H or abs(H[i]['length_cm'] - Lb) > 0.5: continue
    for k in kinds:
        w, up = simulate_wfs_variant(truth, response, tracks, fields, k, chunk=512)
        adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(truth, w, up, rngseed=i + 1)
        tg[k].append(dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix))
    batches.append((i, tracks, Lb)); print(f'targets batch {i} ({Lb:.0f} cm)', flush=True)
Hs = sum(np.array(H[i]['H']) for i, _, _ in batches); Hs = 0.5 * (Hs + Hs.T); Hi = np.linalg.inv(Hs)
L_m = sum(Lb for _, _, Lb in batches) / 100
vg = jax.value_and_grad(lambda P, tracks, tgt: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0])

def total(theta, k):
    P = truth.replace(**{p: jnp.asarray(TRUTH[p] * float(np.exp(theta[j])), dtype=jnp.asarray(getattr(truth, p)).dtype) for j, p in enumerate(PARAMS)})
    v, g = 0., np.zeros(5)
    for (i, tracks, Lb), tgt in zip(batches, tg[k]):
        vb, gb = vg(P, tracks, tgt); v += float(vb)
        g += np.array([float(getattr(gb, p)) * TRUTH[p] * float(np.exp(theta[j])) for j, p in enumerate(PARAMS)])
    return v, g

res = dict(n_batches=len(batches), length_m=L_m, kinds={})
for k in kinds:
    th = np.zeros(5); v, g = total(th, k); hist = [dict(theta=th.tolist(), loss=v, g=g.tolist())]
    t0 = time.perf_counter()
    for it in range(a.iters):
        step = -Hi @ g; lam = 1.0
        while True:
            v2, g2 = total(th + lam * step, k)
            if v2 <= v + 1e-6 or lam < 1 / 16: break
            lam /= 2
        th = th + lam * step; v, g = v2, g2
        hist.append(dict(theta=th.tolist(), loss=v, g=g.tolist(), lam=lam))
        print(f'[{k}] it {it+1} lam {lam:.3g} loss {v:.3f} shift % ' + ' '.join(f'{p} {100*(np.exp(x)-1):+.3f}' for p, x in zip(PARAMS, th)) +
              f'  ({time.perf_counter()-t0:.0f} s)', flush=True)
        if np.abs(lam * step).max() < 1e-5: break
    res['kinds'][k] = dict(history=hist, end_shift_rel=[float(np.exp(x) - 1) for x in th])
    json.dump(res, open(a.out, 'w'), indent=1)
e0 = np.array(res['kinds']['prod']['end_shift_rel']) if 'prod' in res['kinds'] else np.zeros(5)
for k in kinds:
    if k == 'prod': continue
    d = np.array(res['kinds'][k]['end_shift_rel']) - e0
    print(f'{k} - prod end point (%): ' + ' '.join(f'{p} {100*x:+.3f}' for p, x in zip(PARAMS, d)), flush=True)
print('done', flush=True)
