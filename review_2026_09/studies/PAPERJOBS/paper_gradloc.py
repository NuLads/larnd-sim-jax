"""Localise the drift-field AD/FD gap of batch 1 (AD/FD = 1.17, K-independent; batch 0 agrees to 0.7%).
Step 1, by likelihood term: nll = -ll_hits - no_match_penalty + expected_total_hits. AD and central FD of each term
w.r.t. eField at the gradcheck point (aux['log_likelihood_tick'] = -ll_hits, aux['no_match_penalty'],
aux['expected_total_hits']). no_match_penalty is a count (piecewise constant): its AD is 0 by construction.
Step 2, by pixel, for the expected-hit term: per-pixel lambda_p = sum_{n,t} exp(hit_prob), forward-mode derivative
(one jvp) against per-pixel central FD; the pixels carrying the gap are listed with their properties."""
import argparse, json
from paper_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--batches', default='1,0'); ap.add_argument('--h', type=float, default=1e-3)
ap.add_argument('--out', required=True); a = ap.parse_args()
bl = [int(x) for x in a.batches.split(',')]
ds_sim, ds_tgt, fields = load(400., max(bl) + 1); truth, response = params()
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)
off = dict(Ab=+0.01, eField=-0.01, tran_diff=+0.03, long_diff=-0.03, lifetime=+0.05)     # same point as gradcheck.json
P0 = truth.replace(**{k: TRUTH[k] * (1 + off[k]) for k in PARAMS}); e0 = float(P0.eField); h = a.h
out = {}
for i in bl:
    tracks = jnp.asarray(ds_tgt[i]); tgt = target_hits(truth, response, tracks, fields, i + 1)
    sim = LUTProbabilisticSimulation(response)
    def terms(e):
        P = P0.replace(eField=e); nll, aux = loss.compute(P, sim.predict(P, tracks, fields, None), tgt)
        return jnp.stack([nll, aux['log_likelihood_tick'], aux['no_match_penalty'], aux['expected_total_hits']])
    names = ['nll', '-ll_hits', 'no_match_penalty', 'expected_total_hits']
    v0, ad = jax.jvp(terms, (jnp.asarray(e0),), (jnp.asarray(1.0),))
    tp, tm = terms(jnp.asarray(e0 * (1 + h))), terms(jnp.asarray(e0 * (1 - h)))
    fd = (np.asarray(tp) - np.asarray(tm)) / (2 * h * e0)
    rec = {}
    for k, nm in enumerate(names):
        rec[nm] = dict(value=float(v0[k]), ad=float(ad[k]) * e0, fd=float(fd[k]) * e0)
        print(f'batch {i} {nm:20s} value {float(v0[k]):12.3f}  dL/dlnE  AD {float(ad[k])*e0:+.5e}  FD {fd[k]*e0:+.5e}  diff {(float(ad[k])-fd[k])*e0:+.4e}', flush=True)
    # per-pixel expected hits
    def lam_vec(e):
        P = P0.replace(eField=e); pred = sim.predict(P, tracks, fields, None)
        lam = jnp.exp(pred['hit_prob']).sum(axis=(1, 2)); return jnp.where(pred['unique_pixels'] >= 0, lam, 0.0), pred
    l0 = np.asarray(lam_vec(jnp.asarray(e0))[0])
    _, dlam = jax.jvp(lambda e: lam_vec(e)[0], (jnp.asarray(e0),), (jnp.asarray(1.0),))
    lp, lm = np.asarray(lam_vec(jnp.asarray(e0 * (1 + h)))[0]), np.asarray(lam_vec(jnp.asarray(e0 * (1 - h)))[0])
    dad = np.asarray(dlam) * e0; dfd = (lp - lm) / (2 * h * e0) * e0; diff = dad - dfd
    pred = sim.predict(P0, tracks, fields, None); upx = np.asarray(pred['unique_pixels'])
    hp = np.exp(np.asarray(pred['hit_prob'])); tt = np.arange(hp.shape[2]); w0 = hp[:, 0, :]; l0s = np.maximum(w0.sum(1), 1e-12)
    t0 = (w0 * tt).sum(1) / l0s
    print(f'batch {i} per-pixel lambda: sum AD {dad.sum():+.4e}  sum FD {dfd.sum():+.4e}  (Npix {int((upx >= 0).sum())}); '
          f'|diff| top-20 share {np.sort(np.abs(diff))[::-1][:20].sum() / max(np.abs(diff).sum(), 1e-30):.2f}', flush=True)
    top = np.argsort(-np.abs(diff))[:20]
    for j in top:
        print(f'   pix {int(upx[j]):8d}  lambda {l0[j]:7.3f}  AD {dad[j]:+.4e}  FD {dfd[j]:+.4e}  diff {diff[j]:+.4e}  E[t0] {t0[j]:7.1f}  lam0 {l0s[j]:.3f}', flush=True)
    rec['pixels_top'] = [dict(pix=int(upx[j]), lam=float(l0[j]), ad=float(dad[j]), fd=float(dfd[j]), t0=float(t0[j])) for j in top]
    rec['lambda_sum'] = dict(ad=float(dad.sum()), fd=float(dfd.sum()))
    out[i] = rec
    json.dump(out, open(a.out, 'w'), indent=1)
