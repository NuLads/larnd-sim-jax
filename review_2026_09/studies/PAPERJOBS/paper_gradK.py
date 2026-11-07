"""Is the AD/FD gap of the drift-field derivative on batch 1 (gradcheck.json: AD 15% above FD) the derivative discarded
at the beam search's top-K selection? Repeat the check at several beam widths K: the rank-K candidates that swap carry
less weight at larger K, so a top-K origin predicts a gap that shrinks with K; any other origin predicts no change."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=int, default=1); ap.add_argument('--ks', default='50,100,200')
ap.add_argument('--params', default='eField,tran_diff,Ab'); ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(400., a.batch + 1); truth, response = params()
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
off = dict(Ab=+0.01, eField=-0.01, tran_diff=+0.03, long_diff=-0.03, lifetime=+0.05)     # same point as gradcheck.json
i = a.batch; tracks = jnp.asarray(ds_tgt[i]); tgt = target_hits(truth, response, tracks, fields, i + 1)
out = []
for K in [int(x) for x in a.ks.split(',')]:
    P0 = truth.replace(fee_paths_scaling=K).replace(**{k: TRUTH[k] * (1 + off[k]) for k in PARAMS})
    L = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
    (v, _), g = jax.value_and_grad(lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt), has_aux=True)(P0)
    for k in a.params.split(','):
        ad = float(getattr(g, k)); x0 = float(getattr(P0, k)); row = dict(K=K, batch=i, param=k, ad=ad, loss=float(v), fd={})
        for h in (1e-2, 3e-3, 1e-3, 3e-4):
            lp = float(L(P0.replace(**{k: x0 * (1 + h)}))); lm = float(L(P0.replace(**{k: x0 * (1 - h)})))
            row['fd'][h] = (lp - lm) / (2 * h * x0)
        fds = np.array(list(row['fd'].values()))
        print(f'K={K:4d} batch {i} {k:10s} loss {float(v):.3f}  AD {ad*x0:+.5e}  FD ' + ' '.join(f'{h:g}:{fd*x0:+.5e}' for h, fd in row['fd'].items())
              + f'   AD/FD(h<=1e-3) = {ad/np.mean(fds[2:]):.4f}', flush=True)
        out.append(row)
    json.dump(out, open(a.out, 'w'), indent=1)
