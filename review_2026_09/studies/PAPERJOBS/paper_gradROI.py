"""Is the drift-field AD/FD gap on batch 1 (AD/FD = 1.17 at K = 50/100/200, gradK.json) caused by the region-of-interest
window? Pixels whose signal span is shorter than roi_split_length are simulated in a window that starts at the first
tick above roi_threshold (an integer that moves with the hit timing); the sub-threshold charge before it is dropped, so
the loss carries a sawtooth in the timing that the gradient does not see. roi_split_length = 0 sends every pixel through
the full-waveform path (no window start). Prediction if the window is the cause: the gap vanishes in 'all_full'."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--batches', default='1,0'); ap.add_argument('--params', default='eField,Ab,lifetime')
ap.add_argument('--out', required=True); a = ap.parse_args()
bl = [int(x) for x in a.batches.split(',')]
ds_sim, ds_tgt, fields = load(400., max(bl) + 1); truth, response = params()
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
off = dict(Ab=+0.01, eField=-0.01, tran_diff=+0.03, long_diff=-0.03, lifetime=+0.05)     # same point as gradcheck.json
VARIANTS = {'nominal': {}, 'all_full': dict(roi_split_length=0)}
out = []
for i in bl:
    tracks = jnp.asarray(ds_tgt[i]); tgt = target_hits(truth, response, tracks, fields, i + 1)
    for vname, over in VARIANTS.items():
        sim = LUTProbabilisticSimulation(response)                       # fresh: no cached ROI padding across variants
        P0 = (truth.replace(**over) if over else truth).replace(**{k: TRUTH[k] * (1 + off[k]) for k in PARAMS})
        L = lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt)[0]
        (v, _), g = jax.value_and_grad(lambda P: loss.compute(P, sim.predict(P, tracks, fields, None), tgt), has_aux=True)(P0)
        for k in a.params.split(','):
            ad = float(getattr(g, k)); x0 = float(getattr(P0, k)); row = dict(variant=vname, batch=i, param=k, ad=ad, loss=float(v), fd={})
            for h in (3e-3, 1e-3, 3e-4):
                lp = float(L(P0.replace(**{k: x0 * (1 + h)}))); lm = float(L(P0.replace(**{k: x0 * (1 - h)})))
                row['fd'][h] = (lp - lm) / (2 * h * x0)
            fds = np.array(list(row['fd'].values()))
            print(f'{vname:9s} batch {i} {k:9s} loss {float(v):.3f}  AD {ad*x0:+.5e}  FD ' + ' '.join(f'{h:g}:{fd*x0:+.5e}' for h, fd in row['fd'].items())
                  + f'   AD/FD(h<=1e-3) = {ad/np.mean(fds[1:]):.4f}', flush=True)
            out.append(row)
        json.dump(out, open(a.out, 'w'), indent=1)
