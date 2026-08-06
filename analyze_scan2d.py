"""2-D likelihood scans: turning 1-D SLICES into PROFILE likelihoods.

THE POINT. Every scan in the campaign so far is a 1-D slice with the other four parameters frozen.
That is what manufactures the ~25x sensitivity of lifetime to any charge-normalisation error:
lifetime becomes the only knob able to change total charge, so it absorbs the whole thing. In a
real fit A_b is free and absorbs it instead -- confirmed directly by the fit-side test, where a 4%
dE/dx mean swing moved lifetime by 0.8 +/- 4.7 points against a slice prediction of 97.

A 2-D scan over (lifetime, A_b) fixes this WITHOUT needing a minimiser: minimising the loss over
A_b at each lifetime IS the profile likelihood in lifetime. That is the quantity a displaced
minimum has to be measured on before it can be called a bias.

FILE LAYOUT. `history_scan2d_batch<N>_<label>.pkl`, written per batch by the LARND_SCAN_2D branch.
Rows are ordered (pA outer, pB inner) within each batch, so the flat loss array reshapes to
(nbatch, nsteps, nsteps). The `<param>_iter` arrays carry one extra leading element (the init), so
they are offset by exactly one relative to `losses_iter`.
"""
import glob, os, pickle
import numpy as np

NOM = {'Ab': 0.8, 'eField': 0.5, 'tran_diff': 8.8e-6, 'long_diff': 4.0e-6, 'lifetime': 2200.0}


def load2d(path, pA='lifetime', pB='Ab'):
    """-> (gridA, gridB, loss[nbatch, nA, nB])"""
    h = pickle.load(open(path, 'rb'))
    L = np.asarray(h['losses_iter'], float)
    c = h.get('config'); c = c.item() if hasattr(c, 'item') else c
    n = int(getattr(c, 'iterations'))
    per = n * n
    nb = len(L) // per
    if nb == 0:
        return None
    L = L[:nb * per].reshape(nb, n, n)
    off = len(np.asarray(h[pA + '_iter'])) - len(np.asarray(h['losses_iter'], float))
    a = np.asarray(h[pA + '_iter'], float)[off:off + per].reshape(n, n)[:, 0]
    b = np.asarray(h[pB + '_iter'], float)[off:off + per].reshape(n, n)[0, :]
    return a, b, L


def vertex(x, y):
    """3-point parabola vertex around the argmin; falls back to the grid point at an edge."""
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return x[i], True
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    return (-B / (2 * A), False) if A > 0 else (x[i], True)


def summarise(a, b, L, pA='lifetime', pB='Ab'):
    """slice vs profile minimum in pA, plus the joint minimum."""
    S = L.sum(0)                                    # total NLL over batches
    jb = np.unravel_index(np.argmin(S), S.shape)
    # SLICE: pB frozen at its nominal grid point
    kB = int(np.argmin(np.abs(b - NOM[pB])))
    xs, _ = vertex(a, S[:, kB])
    # PROFILE: minimise over pB at each pA
    prof = S.min(axis=1)
    xp, _ = vertex(a, prof)
    # where pB sits along the profile (the degeneracy direction)
    bhat = b[np.argmin(S, axis=1)]
    return dict(slice_pct=100 * (xs - NOM[pA]) / NOM[pA],
                prof_pct=100 * (xp - NOM[pA]) / NOM[pA],
                joint=(a[jb[0]], b[jb[1]]),
                prof_curve=prof, bhat=bhat, S=S,
                bmin_pct=100 * (b[jb[1]] - NOM[pB]) / NOM[pB])


def find(tag):
    g = sorted(glob.glob(f'fit_result/loss_profile/history_scan2d_batch*_prof_{tag}_seed0.pkl'))
    return max(g, key=lambda p: int(p.split('batch')[1].split('_')[0])) if g else None


if __name__ == '__main__':
    import sys
    tags = sys.argv[1:] or ['true_s2dsmoke']
    print(f"{'condition':22s}{'nbatch':>7s}{'grid':>7s}{'SLICE':>10s}{'PROFILE':>10s}{'shrink':>9s}{'Ab at min':>11s}")
    print('-' * 76)
    for t in tags:
        f = find(t)
        if not f:
            print(f'{t:22s} not found'); continue
        r = load2d(f)
        if r is None:
            print(f'{t:22s} no complete batch'); continue
        a, b, L = r
        s = summarise(a, b, L)
        shrink = abs(s['prof_pct']) / abs(s['slice_pct']) if s['slice_pct'] else np.nan
        print(f"{t:22s}{L.shape[0]:>7d}{L.shape[1]:>4d}x{L.shape[2]:<2d}"
              f"{s['slice_pct']:>+9.2f}%{s['prof_pct']:>+9.2f}%{shrink:>8.2f}x{s['bmin_pct']:>+10.2f}%")
