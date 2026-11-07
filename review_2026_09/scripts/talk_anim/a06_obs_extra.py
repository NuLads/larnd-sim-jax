"""D_L, D_T and lifetime panels for a06, chosen from what the real sweep shows (see the report)."""
import numpy as np
from a06_load import *
import a06_obs as A
TICK_US = 0.1


def panel_second_hit(p, R):
    """D_L: stacked time profile of the second hit on far-drift pixels. The first hit is dominated by noise (sd ~7.5 ticks, unchanged by D_L)."""
    o, i0 = A.common(p, R); u, X, Y, G = A.geom(R); _, Lam = align(R, 'Lam'); _, tm = align(R, 'tm'); _, lam = align(R, 'lam')
    D = A.D; far = (G['z'] < 22) & (lam[i0, :, 0] > .9) & (lam[i0, :, 1] > .9)
    tb = np.rint(tm[i0, :, 1]).astype(int); off = np.arange(-14, 15)
    ym = np.array([np.mean([Lam[i, j, np.clip(tb[j] + off, 0, Lam.shape[2] - 1)] for j in np.where(far)[0]], 0) for i in range(len(R))])
    # data: for each of these pixels, its later hit, relative to the same truth-model time
    ip = np.searchsorted(u, D['pid']); ok = (ip < len(u)) & (u[np.clip(ip, 0, len(u) - 1)] == D['pid']); dt = []
    for j in np.where(far)[0]:
        h = D['t'][ok & (ip == j)]
        if len(h) == 2: dt.append(h.max() - tb[j])
    e = np.arange(-15, 16, 3.0); c = .5 * (e[1:] + e[:-1]); yd = np.histogram(dt, e)[0] / (len(dt) * 3.0)
    o.update(xd=c, yd=yd, xm=off.astype(float), ym=ym, xlabel='second hit time, relative to model at truth (ticks)', ylabel='hit probability per tick', xlim=(-14, 14), ylim=(0, 1.3 * ym.max()), npix=int(far.sum()), nhit=len(dt))
    return o


def panel_edge(p, R):
    """D_T: mean charge of a hit pixel against how far the track passes from the pixel centre. Hit pixels end at half a pitch."""
    o, i0 = A.common(p, R); u, X, Y, G = A.geom(R); _, Q = align(R, 'Qtick'); Qj = Q.sum(2); d = np.abs(G['d']); D = A.D
    q = np.zeros(len(u)); ip = np.searchsorted(u, D['pid']); ok = (ip < len(u)) & (u[np.clip(ip, 0, len(u) - 1)] == D['pid']); np.add.at(q, ip[ok], D['q'][ok])
    e = np.array([0, .04, .08, .12, .15, .18, .20, .22, .24, .26]); c = .5 * (e[1:] + e[:-1]); ib = np.digitize(d, e) - 1; n = len(c)
    has = q > 0; yd = np.array([q[(ib == b) & has].mean() if ((ib == b) & has).any() else np.nan for b in range(n)])
    ym = np.array([[Qj[i][(ib == b) & has].mean() if ((ib == b) & has).any() else np.nan for b in range(n)] for i in range(len(R))])
    k = np.isfinite(yd)
    o.update(xd=c[k], yd=yd[k], xm=c[k], ym=ym[:, k], xlabel='distance from pixel centre to track (cm)', ylabel='charge per pixel (ke)', xlim=(0, .26), ylim=(0, 1.2 * np.nanmax(ym)))
    return o


def panel_droop(p, R):
    """Lifetime: charge against drift time, each bin relative to the model at truth, so that the dE/dx texture of the track drops out."""
    o, i0 = A.common(p, R); u, X, Y, G = A.geom(R); _, Q = align(R, 'Qtick'); Qj = Q.sum(2); _, tm = align(R, 'tm'); m = G['track'] == 4; D = A.D
    q = np.zeros(len(u)); ip = np.searchsorted(u, D['pid']); ok = (ip < len(u)) & (u[np.clip(ip, 0, len(u) - 1)] == D['pid']); np.add.at(q, ip[ok], D['q'][ok])
    e = np.linspace(0, 64, 14); ib = np.digitize(G['s'], e) - 1; bins = [b for b in range(len(e) - 1) if ((ib == b) & m).sum() >= 6]
    ref = np.array([Qj[i0][(ib == b) & m].sum() for b in bins]); yd = np.array([q[(ib == b) & m].sum() for b in bins]) / ref
    tt = np.array([tm[i0][(ib == b) & m, 0].mean() for b in bins]) * TICK_US
    ym = np.array([[Qj[i][(ib == b) & m].sum() for b in bins] for i in range(len(R))]) / ref
    s = np.argsort(tt)
    o.update(xd=tt[s], yd=yd[s], xm=tt[s], ym=ym[:, s], xlabel='drift time (µs)', ylabel='charge, relative to model at truth', xlim=(0, 100), ylim=(0.86, 1.08))
    return o


PANEL_EXTRA = dict(long_diff=panel_second_hit, tran_diff=panel_edge, lifetime=panel_droop)
