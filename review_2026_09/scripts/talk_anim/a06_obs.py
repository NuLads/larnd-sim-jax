"""Reduce the a06 sweep to what the animation draws: for each parameter the fixed data series, the model series at every swept value,
and the loss with its jax.grad slope. Only real evaluations go in; display interpolation happens in a06_five_knobs.py."""
import json, os, numpy as np
from a06_load import *
from a06_grid import F

D = data(); z = D['z']
O = {}
upd, inv = np.unique(D['pid'], return_inverse=True)
xyd = np.stack([np.bincount(inv, D['px']) / np.bincount(inv), np.bincount(inv, D['py']) / np.bincount(inv)], 1)
TICK_US = 0.1
CAP = dict(Ab='Fewer electrons survive recombination: every charge scales together',
           lifetime='Electrons are lost as they drift: the far end droops',
           eField='Times here count from the cathode: lower E moves every hit earlier',
           tran_diff='Wider diffusion blurs the pixel edge: charge spills to the neighbour',
           long_diff='Wider diffusion smears the arrival time: the second hit broadens')
UNIT = dict(Ab=('', 1, '%.3f'), eField=('kV/cm', 1, '%.3f'), tran_diff=('×10⁻⁶ cm²/µs', 1e6, '%.1f'), long_diff=('×10⁻⁶ cm²/µs', 1e6, '%.1f'), lifetime=('µs', 1, '%.0f'))


def geom(R):
    xy = {}
    for r in R: xy.update(zip(r['pid'].tolist(), zip(r['x'].tolist(), r['y'].tolist())))
    u = np.unique(np.concatenate([r['pid'] for r in R]))
    X = np.array([xy[int(k)][0] for k in u]); Y = np.array([xy[int(k)][1] for k in u])
    return u, X, Y, track_geom(D, X, Y)


def common(p, R):
    tv = float(z['true_' + p]); vals = np.array([float(r['val']) for r in R]); o = {}
    i0 = int(np.argmin(np.abs(vals / tv - 1)))
    o.update(vals=vals, nll=np.array([float(r['nll']) for r in R]), grad=np.array([float(r['grad']) for r in R]), true=tv, lo=vals[0], hi=vals[-1], L0=float(R[i0]['nll']))
    o.update(cap=CAP[p], unit=UNIT[p][0], scale=UNIT[p][1], fmt=UNIT[p][2], endc=F[p][3], ldpos=(0.03, 0.95), ms=15)   # closing tangent: the off-truth FD point
    return o, i0


def binned(Rq, u, xp, edges):
    """Model: summed expected charge of the pixels in each bin, for every swept value."""
    ib = np.digitize(xp, edges) - 1; n = len(edges) - 1
    return np.array([[row[ib == b].sum() for b in range(n)] for row in Rq])


def panel_charge_s(p, R):
    """Charge along the main track."""
    o, i0 = common(p, R); u, X, Y, G = geom(R); _, Q = align(R, 'Qtick'); Qj = Q.sum(2); m = G['track'] == 4
    Gd = track_geom(D, D['px'], D['py']); md = Gd['track'] == 4
    e = np.linspace(0, 64, 14); mid = .5 * (e[1:] + e[:-1])
    yd = np.histogram(Gd['s'][md], e, weights=D['q'][md])[0]; ym = binned(Qj[:, m], u[m], G['s'][m], e)
    k = (yd > 200) & (ym[i0] > 200)                       # bins fully on the track: the partial last bin is dropped
    o.update(xd=mid[k], yd=yd[k], xm=mid[k], ym=ym[:, k], xlabel='position along the track (cm)', ylabel='charge per 4.9 cm of track (ke)', xlim=(0, 60), ylim=(200, 350))
    return o


def panel_charge_t(p, R):
    """Charge against drift time on the main track."""
    o, i0 = common(p, R); u, X, Y, G = geom(R); _, Q = align(R, 'Qtick'); m = G['track'] == 4
    e = np.linspace(100, 900, 9); mid = .5 * (e[1:] + e[:-1])
    Gd = track_geom(D, D['px'], D['py']); md = Gd['track'] == 4
    yd = np.histogram(D['t'][md], e, weights=D['q'][md])[0] / 1e3
    ym = np.array([[Q[i][m][:, int(a):int(b)].sum() for a, b in zip(e[:-1], e[1:])] for i in range(len(R))]) / 1e3
    o.update(xd=mid * TICK_US, yd=yd, xm=mid * TICK_US, ym=ym, xlabel='drift time (µs)', ylabel='charge per 8 µs of drift (Me)', xlim=(5, 95), ylim=(0.6 * ym.min(), 1.1 * ym.max()))
    return o


def panel_time(p, R):
    """Hit time against the truth-model time of the same pixel: the lever arm of the drift velocity."""
    o, i0 = common(p, R); u, X, Y, G = geom(R); _, tm = align(R, 'tm'); _, lam = align(R, 'lam')
    good = lam[i0, :, 0] > .9
    first = np.full(len(u), np.nan); ip = np.searchsorted(u, D['pid']); okp = (ip < len(u)) & (u[np.clip(ip, 0, len(u) - 1)] == D['pid'])
    for j in np.unique(ip[okp]): first[j] = D['t'][okp & (ip == j)].min()
    k = good & np.isfinite(first); t0 = tm[i0, :, 0]
    e = np.linspace(0, 1100, 23); mid = .5 * (e[1:] + e[:-1]); ib = np.digitize(t0, e) - 1
    ym = np.array([[np.nanmean(np.where(ib[k] == b, (tm[i, :, 0] - t0)[k], np.nan)) if (ib[k] == b).any() else np.nan for b in range(len(mid))] for i in range(len(R))])
    ok = np.isfinite(ym[i0])
    o.update(xd=t0[k] * TICK_US, yd=(first - t0)[k], xm=mid[ok] * TICK_US, ym=ym[:, ok], xlabel='drift time (µs)', ylabel='hit time minus model at truth (ticks)', xlim=(0, 106), ylim=(-32, 32), ms=9)
    return o


PANEL = dict(Ab=panel_charge_s, lifetime=panel_charge_t, eField=panel_time)


if __name__ == '__main__':
    import sys
    from a06_obs_extra import PANEL_EXTRA
    PANEL.update(PANEL_EXTRA)
    fd = []
    for p in PARAMS:
        f = f'{SW}/fd_{p}.json'
        if os.path.exists(f): fd += json.load(open(f))
    for p in PARAMS:
        R = sweep(p); print(p, 'points', len(R), np.round([float(r['val']) / float(z['true_' + p]) for r in R], 3))
        o = PANEL[p](p, R)
        for k, v in o.items(): O[f'{p}_{k}'] = v
    O['end_caption'] = 'Each tangent is jax.grad through the production simulation'; O['end_sub'] = 'Finite-difference check not available yet'
    # value-only nodes: every real loss evaluation of the sweep, including the forward-only evaluations of the finite-difference runs near the E minimum
    for p in PARAMS:
        xv, xL = list(O[f'{p}_vals']), list(O[f'{p}_nll'])
        if p == 'eField' and os.path.exists(f'{SW}/fd_e_steps.json'):
            for r in json.load(open(f'{SW}/fd_e_steps.json')):
                if abs(r['c'] - 1) < 1e-9 and r['h'] >= 1e-3:
                    xv += [r['x'] * (1 + r['h']), r['x'] * (1 - r['h'])]; xL += [r['up'], r['dn']]
        k = np.argsort(xv); O[f'{p}_xv'] = np.array(xv)[k]; O[f'{p}_xL'] = np.array(xL)[k]
    if fd:
        O['fd'] = json.dumps(fd)
        for r in fd:
            print('FD', r['param'], 'c', round(r['c'], 4), 'h', r['h'], 'x*AD', round(r['ad'] * r['val'], 2), 'x*FD', round(r['fd'] * r['val'], 2), 'rel', round((r['ad'] - r['fd']) / r['fd'], 5), 'dL', round(r['up'] - r['dn'], 4))
        rel = lambda r: abs(r['ad'] - r['fd']) / abs(r['fd'])
        oth = [r for r in fd if r['param'] != 'eField']; e = {round(r['c'], 4): rel(r) for r in fd if r['param'] == 'eField'}
        strong = [r for r in oth if abs(r['up'] - r['dn']) > 0.1]      # steps well above the float32 rounding of a loss near 3000
        print('largest rel, non-E, all', max(rel(r) for r in oth), 'with |dL|>0.1', max(rel(r) for r in strong), 'n', len(oth), len(strong), 'E', e)
        # E: the finite difference depends on the step (sub-tick structure on a very narrow well), so quote the smallest step measured at each point
        ad = {round(float(v) / float(z['true_eField']), 4): float(g) for v, g in zip(O['eField_vals'], O['eField_grad'])}; st = []    # jax.grad at every real E point
        for fn in ('fd_e_steps.json', 'fd_e_steps2.json'):
            if os.path.exists(f'{SW}/{fn}'): st += json.load(open(f'{SW}/{fn}'))
        for r in sorted(st, key=lambda r: (r['c'], r['h'])): print('E step c', r['c'], 'h', r['h'], 'FD', round(r['fd']), 'AD', ad.get(round(r['c'], 4)))
        sm = [abs(ad[round(r['c'], 4)] - r['fd']) / abs(r['fd']) for r in st if r['h'] == 1e-4 and round(r['c'], 4) in ad]
        print('E rel at h=1e-4', sm)
        O["end_sub"] = (f"Finite differences agree within {100 * max(rel(r) for r in oth):.1f}% for $A_b$, $\\tau$, $D_T$, $D_L$. "
                        f"For $E$, whose well is far narrower, a 0.01% step agrees to {100 * max(sm):.0f}%")
    np.savez(f'{T}/data/a06_obs.npz', **O); print('wrote obs')
