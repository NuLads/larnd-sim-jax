"""Statistical floor of the five calibration parameters: Hessian of the production loss (probabilistic model + PPP
likelihood) at the true parameter values, with the true track segments and no nuisances, summed over every batch of a
production-sized sample. Built from five forward-over-reverse Hessian-vector products per batch, in log-parameter units
(H_jk = theta_j theta_k d2L/dtheta_j dtheta_k), because the dense jacfwd(jacrev) Hessian of a 400 cm batch needs 63 GB.
Also stores each batch's score (gradient at truth) for the sandwich (outer-product) estimate."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--len', type=float, default=200.); ap.add_argument('--nb', type=int, default=400)
ap.add_argument('--data_seed', type=int, default=1); ap.add_argument('--out', required=True); a = ap.parse_args()
_, ds400, f400 = load(400., 100, a.data_seed)          # production batching, for the reference track length only
L_prod = sum(track_length_cm(jnp.asarray(ds400[i]), f400) for i in range(len(ds400)))
print(f'production sample (100 x 400 cm, data_seed {a.data_seed}): {len(ds400)} batches, {L_prod/100:.1f} m of track', flush=True)
del ds400
ds_sim, ds_tgt, fields = load(a.len, a.nb, a.data_seed); truth, response = params()
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
zero = jax.tree_util.tree_map(jnp.zeros_like, truth)
th = np.array([TRUTH[k] for k in PARAMS])
res = dict(params=PARAMS, truth=TRUTH, len=a.len, data_seed=a.data_seed, L_prod_cm=L_prod, batches=[])
for i in range(min(a.nb, len(ds_tgt))):
    t0 = time.perf_counter()
    tracks = jnp.asarray(ds_tgt[i]); Lb = track_length_cm(tracks, fields)
    try:
        tgt = target_hits(truth, response, tracks, fields, i + 1)
        f = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
        grad_f = jax.grad(f); Hb = np.zeros((5, 5)); gb = None
        for c, k in enumerate(PARAMS):
            v = zero.replace(**{k: jnp.ones_like(getattr(truth, k)) * TRUTH[k]})       # tangent = d ln theta_k
            gv, hv = jax.jvp(grad_f, (truth,), (v,))
            Hb[:, c] = [float(getattr(hv, j)) * TRUTH[j] for j in PARAMS]
            if gb is None: gb = np.array([float(getattr(gv, j)) * TRUTH[j] for j in PARAMS])
        asym = float(np.abs(Hb - Hb.T).max() / np.abs(Hb).max())
        res['batches'].append(dict(batch=i, length_cm=Lb, H=Hb.tolist(), g=gb.tolist(), asym=asym))
        print(f'[batch {i}] {Lb:.0f} cm  diag(H) ' + ' '.join(f'{x:.3e}' for x in np.diag(Hb)) + f'  asym {asym:.1e}  {time.perf_counter()-t0:.1f} s', flush=True)
    except Exception as e:                                # e.g. out of memory on an unusually large batch
        res['batches'].append(dict(batch=i, length_cm=Lb, error=repr(e)[:300]))
        print(f'[batch {i}] {Lb:.0f} cm  FAILED {repr(e)[:200]}', flush=True)
    if i % 10 == 0: json.dump(res, open(a.out, 'w'))
json.dump(res, open(a.out, 'w'))
ok = [b for b in res['batches'] if 'H' in b]
H = sum(np.array(b['H']) for b in ok); Hs = 0.5 * (H + H.T); Lok = sum(b['length_cm'] for b in ok)
C = np.linalg.inv(Hs) * (Lok / L_prod)                    # scaled to the production track length
G = np.array([b['g'] for b in ok]); V = G.T @ G; Csw = np.linalg.inv(Hs) @ V @ np.linalg.inv(Hs) * (Lok / L_prod)
print(f'{len(ok)}/{len(res["batches"])} batches, {Lok/100:.1f} m; eigenvalues of H (ln units): ' + ' '.join(f'{x:.3e}' for x in np.linalg.eigvalsh(Hs)), flush=True)
for j, k in enumerate(PARAMS):
    print(f'  {k:10s} sigma (production length) Hessian {100*np.sqrt(C[j,j]):.3f}%  sandwich {100*np.sqrt(Csw[j,j]):.3f}%   '
          f'conditional (others fixed) {100/np.sqrt(Hs[j,j]*L_prod/Lok):.3f}%', flush=True)
print('correlation:\n' + np.array2string(C / np.sqrt(np.outer(np.diag(C), np.diag(C))), precision=2), flush=True)
