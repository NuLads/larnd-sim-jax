"""One-at-a-time sweep of a calibration parameter on the hero event: true segments, truth point for the other four,
production simulation (LUTProbabilisticSimulation.predict) + production PPP loss (ProbabilisticLossStrategy.compute),
value and jax.grad at every point. Writes one small npz per point, so a killed run keeps what it has.
usage: a06_sweep.py --param lifetime --n 25 [--tag main] | --fd"""
import argparse, os, sys, time, numpy as np
from a06_hero_common import *
ap = argparse.ArgumentParser(); ap.add_argument('--param'); ap.add_argument('--tag', default='main'); ap.add_argument('--points', default=''); ap.add_argument('--n', type=int, default=0); ap.add_argument('--nfd', type=int, default=7)
ap.add_argument('--grad', type=int, default=1); ap.add_argument('--rows', default='true'); a = ap.parse_args()
S = setup(); jax, jnp = S['jax'], S['jnp']; truth, f, tgt, loss, sim, z = S['truth'], S['fields'], S['tgt'], S['loss'], S['sim'], S['z']
adc2charge = S['adc2charge']
tracks = jnp.asarray(z['true_rows'] if a.rows == 'true' else z[f'{a.rows}_rows'])
tk = np.asarray(z['data_ticks']).astype(int); pidt = np.asarray(z['data_pixel_id']); Qt = np.asarray(z['data_q'])
OUT = f'{T}/data/a06_sweep'; os.makedirs(OUT, exist_ok=True)


def hit_scores(hp_log, adc, up, P):
    """Per-hit log I_i and its time part, same formula as ProbabilisticLossStrategy._compute_fast (checked against aux below)."""
    W, st, sq = 2, 1.0, 0.5
    d = np.arange(-W, W + 1); lw = -0.5 * (d / st) ** 2; lw = lw - np.log(np.exp(lw).sum())
    ok = up >= 0; upv = up[ok]; rows = np.where(ok)[0]
    idx = np.searchsorted(upv, pidt); idx = np.clip(idx, 0, len(upv) - 1); match = upv[idx] == pidt
    Nt = hp_log.shape[-1]
    win = np.clip(tk[:, None] + d[None, :], 0, Nt - 1)           # (Nhit, 5)
    r = rows[idx]
    lp = hp_log[r[:, None], :, win]                               # (Nhit, 5, Nslot)
    qq = np.asarray(adc2charge(jnp.asarray(adc[r[:, None], :, win]), P))
    lq = -0.5 * ((Qt[:, None, None] - qq) / sq) ** 2 - 0.5 * np.log(2 * np.pi * sq ** 2)
    joint = lp + lw[None, :, None] + lq
    timep = np.exp(lp + lw[None, :, None]).sum((1, 2))
    logI = np.logaddexp.reduce(joint.reshape(len(pidt), -1).astype(np.float64), axis=1)
    return np.where(match, np.maximum(logI, -18.420681), 0.0), np.log(np.maximum(timep, 1e-300)), match


def fn(x):
    P = truth.replace(**{a.param: x}) if a.rows == 'true' else truth.replace(**x)
    pred = sim.predict(P, tracks, f, None)
    nll, aux = loss.compute(P, pred, tgt)
    return nll, (aux, pred)


def pack(v, aux, pred, g, P):
    up = np.asarray(pred['unique_pixels']); ok = up >= 0
    lp = np.asarray(pred['hit_prob']); adc = np.asarray(pred['adcs_distrib'])
    logI, logT, match = hit_scores(lp, adc, up, P)
    hp = np.exp(lp); q = np.asarray(adc2charge(pred['adcs_distrib'], P)); tt = np.arange(hp.shape[2], dtype=float)
    hp = hp[ok]; q = q[ok]; hp[hp < 1e-7] = 0.0
    lam = hp.sum(2); ns = 3; L = np.maximum(lam[:, :ns], 1e-12)
    tm = (hp[:, :ns] * tt).sum(2) / L; ts = np.sqrt(np.maximum((hp[:, :ns] * tt ** 2).sum(2) / L - tm ** 2, 0)); qm = (hp[:, :ns] * q[:, :ns]).sum(2) / L
    Lam = hp.sum(1); Qtick = (hp * q).sum(1)
    return dict(nll=float(v), grad=g, ll=float(aux['log_likelihood_tick']), nomatch=float(aux['no_match_penalty']), expct=float(aux['expected_total_hits']), matched=int(aux['matched_hits']),
                pid=up[ok], x=np.asarray(pred['pixel_x'])[ok], y=np.asarray(pred['pixel_y'])[ok], lam=lam.astype(np.float32), tm=tm.astype(np.float32), ts=ts.astype(np.float32), qm=qm.astype(np.float32),
                Lam=Lam.astype(np.float32), Qtick=Qtick.astype(np.float32), logI=logI.astype(np.float32), logT=logT.astype(np.float32), match=match)


if a.rows == 'true':
    from a06_grid import F
    t_ = float(z['true_' + a.param])
    pts = [float(s) for s in a.points.split(',')] if a.points else [t_ * f_ for f_ in F[a.param][:a.n or None]]
    vg = jax.value_and_grad(fn, has_aux=True)
    def fd_stage():
        import json
        from a06_grid import F as F_
        h = dict(Ab=1e-3, eField=3e-4, tran_diff=3e-3, long_diff=3e-3, lifetime=3e-3)[a.param]; fj = f'{OUT}/fd_{a.param}.json'
        res = json.load(open(fj)) if os.path.exists(fj) else []
        for c in (F_[a.param][0], F_[a.param][3]):
            if any(abs(r['c'] - c) < 1e-9 for r in res): continue
            x = t_ * c; r0 = np.load(f'{OUT}/{a.tag}_{a.param}_{x:.8g}.npz')
            up = float(fn(x * (1 + h))[0]); dn = float(fn(x * (1 - h))[0]); fd = (up - dn) / (2 * x * h); ad = float(r0['grad'])
            res.append(dict(param=a.param, c=c, h=h, val=x, ad=ad, fd=fd, up=up, dn=dn, L=float(r0['nll'])))
            with open(fj + '.tmp', 'w') as g_: json.dump(res, g_, indent=1)
            os.replace(fj + '.tmp', fj)
            print('FD', a.param, c, h, 'AD', ad, 'FD', fd, 'rel', (ad - fd) / fd, 'x*(AD-FD)', x * (ad - fd), flush=True)
    for i, x in enumerate(pts):
        if i == a.nfd: fd_stage()
        fo = f'{OUT}/{a.tag}_{a.param}_{x:.8g}.npz'
        if os.path.exists(fo): continue
        t0 = time.time()
        if a.grad:
            (v, (aux, pred)), g = vg(x); g = float(g)
        else:
            v, (aux, pred) = fn(x); g = np.nan
        P = truth.replace(**{a.param: x})
        o = pack(v, aux, pred, g, P)
        o['val'] = x; o['tsec'] = time.time() - t0
        chk = float(o['logI'].sum()) + float(aux['log_likelihood_tick'])
        np.savez_compressed(fo + '.tmp.npz', **o); os.replace(fo + '.tmp.npz', fo)
        print(a.param, i, x, 'nll', round(float(v), 3), 'g', g, 'n_ok', o['matched'], 'expct', round(o['expct'], 2), 'sumcheck', round(chk, 3), 'tsec', round(o['tsec'], 1), flush=True)
else:                                     # a saved fit state: gradient with respect to all five calibration parameters
    x0 = {p: float(z[f'{a.rows}_{p}']) for p in PARAMS}
    (v, (aux, pred)), g = jax.value_and_grad(fn, has_aux=True)(x0)
    P = truth.replace(**x0); o = pack(v, aux, pred, 0.0, P)
    for p in PARAMS: o['grad_' + p] = float(g[p]); o['val_' + p] = x0[p]
    o['Lamslots'] = np.exp(np.asarray(pred['hit_prob']))[np.asarray(pred['unique_pixels']) >= 0].astype(np.float32)
    o['qslots'] = np.asarray(adc2charge(pred['adcs_distrib'], P))[np.asarray(pred['unique_pixels']) >= 0].astype(np.float32)
    o['Lamslots'][o['Lamslots'] < 1e-7] = 0
    np.savez_compressed(f'{OUT}/state_{a.rows}.npz', **o)
    print(a.rows, 'nll', float(v), {k: float(vv) for k, vv in aux.items()}, 'grad', {p: float(g[p]) for p in PARAMS}, 'x*g', {p: float(g[p]) * x0[p] for p in PARAMS}, 'sumcheck', float(o['logI'].sum()) + float(aux['log_likelihood_tick']))
