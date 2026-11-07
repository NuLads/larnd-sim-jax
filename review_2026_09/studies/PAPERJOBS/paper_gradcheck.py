"""Gradient correctness: autodiff gradient of the production loss (probabilistic model + PPP likelihood) w.r.t. the
five calibration parameters, against central finite differences at several step sizes, at a point near truth."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--nb', type=int, default=2); ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(400., a.nb); truth, response = params()
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
off = dict(Ab=+0.01, eField=-0.01, tran_diff=+0.03, long_diff=-0.03, lifetime=+0.05)     # evaluate away from the optimum
P0 = truth.replace(**{k: TRUTH[k] * (1 + off[k]) for k in PARAMS})
out = []
for i in range(min(a.nb, len(ds_tgt))):
    tracks = jnp.asarray(ds_tgt[i]); tgt = target_hits(truth, response, tracks, fields, i + 1)
    L = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
    (v, _), g = jax.value_and_grad(lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt), has_aux=True)(P0)
    for k in PARAMS:
        ad = float(getattr(g, k)); x0 = float(getattr(P0, k)); row = dict(batch=i, param=k, ad=ad, loss=float(v), fd={})
        for h in (1e-2, 3e-3, 1e-3, 3e-4):
            lp = float(L(P0.replace(**{k: x0 * (1 + h)}))); lm = float(L(P0.replace(**{k: x0 * (1 - h)})))
            fd = (lp - lm) / (2 * h * x0); row['fd'][h] = fd
        best = min(row['fd'].items(), key=lambda kv: abs(kv[1] - ad))
        print(f'batch {i} {k:10s} AD {ad*x0:+.5e}  FD(h) ' + ' '.join(f'{h:g}:{fd*x0:+.5e}' for h, fd in row['fd'].items())
              + f'   best rel diff {abs(best[1]-ad)/max(abs(ad),1e-30):.2e} at h={best[0]:g}   (derivatives x param value)', flush=True)
        out.append(row)
json.dump(out, open(a.out, 'w'), indent=1)
