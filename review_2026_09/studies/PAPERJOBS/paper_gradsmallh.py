"""Hypothesis for the batch-1 drift-field AD/FD gap (AD/FD = 1.17 at relative steps 3e-4..1e-2; the gap is entirely in
the observed-hit term, paper_gradloc): the arrival time is linearly interpolated between 0.1-us ticks, so the likelihood
is smooth only WITHIN a tick; AD returns the exact local slope, while a relative step h in E shifts far-drift charge by
~h * 2000 ticks (h = 1e-3 -> ~2 ticks) and the central difference averages the slope over the kinks. Prediction: for
h << 1/2000 (shifts << 1 tick) the central difference converges to AD. float32 rounding of the ~2e4 loss limits h to
~1e-6 for this derivative (d L / d ln E ~ 1e6)."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=int, default=1); ap.add_argument('--out', required=True); a = ap.parse_args()
ds_sim, ds_tgt, fields = load(400., a.batch + 1); truth, response = params()
sim = LUTProbabilisticSimulation(response); loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
off = dict(Ab=+0.01, eField=-0.01, tran_diff=+0.03, long_diff=-0.03, lifetime=+0.05)      # same point as gradcheck.json
P0 = truth.replace(**{k: TRUTH[k] * (1 + off[k]) for k in PARAMS})
i = a.batch; tracks = jnp.asarray(ds_tgt[i]); tgt = target_hits(truth, response, tracks, fields, i + 1)
L = lambda P: float(loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0])
(v, _), g = jax.value_and_grad(lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt), has_aux=True)(P0)
x0 = float(P0.eField); ad = float(g.eField) * x0; out = dict(batch=i, ad=ad, loss=float(v), fd={})
print(f'batch {i} eField: loss {float(v):.4f}  AD dL/dlnE = {ad:+.5e}', flush=True)
for h in (1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
    lp, lm = L(P0.replace(eField=x0 * (1 + h))), L(P0.replace(eField=x0 * (1 - h)))
    fd = (lp - lm) / (2 * h); out['fd'][h] = fd
    print(f'  h = {h:.0e}  (far-drift shift ~{h * 1900:.3f} ticks)  FD = {fd:+.5e}   AD/FD = {ad / fd:.4f}   dL = {lp - lm:+.3f}', flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
