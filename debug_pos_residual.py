"""Recompute the INITIAL position residual per batch, independently of the fitter.

With zero spline coefficients the fitted chain is exactly the nominal (linear-guess) track, so

    pos_nodes[k] = x0_fixed + k * step_len * u0,   u0 = dir(theta0_i, phi0_i)

and the residual is a pure function of the stored `chain_contexts` + `true_positions`. Both are
serialised per batch into every checkpoint, so this reproduces what `_compute_pos_residual_batch`
would have returned at each batch's FIRST visit -- with no dependence on visit order, learning
rate, or calibration state.

That makes it the referee for the SHUFFLE-vs-ANNEALLONG discrepancy: if the two runs' stored
contexts give the SAME initial residual per batch, the 3.4x gap came from the run dynamics; if
they give DIFFERENT values, the metric or its inputs changed between the two runs and the
comparison was never valid.
"""
import pickle
import numpy as np
import robust_convergence as R


def dir_from_angles(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], dtype=np.float64)


def initial_residual(ctx, true_pts):
    """Residual of the nominal straight chain against the true points (cm)."""
    u0 = dir_from_angles(ctx['theta0_i'], ctx['phi0_i'])
    nc = int(ctx['n_chain'])
    ks = np.arange(nc + 1)
    nodes = np.asarray(ctx['x0_fixed'], float)[None, :] + (ks[:, None] * float(ctx['step_len'])) * u0[None, :]
    out = []
    am = np.asarray(ctx['alpha_mid'], float)
    tp = np.asarray(true_pts, float)
    n = min(len(am), len(tp))
    for j in range(n):
        a = am[j]
        k = int(np.clip(int(np.floor(a * nc)), 0, nc - 1))
        frac = float(a * nc - k)
        fitted = nodes[k] + frac * (nodes[k + 1] - nodes[k])
        out.append(np.linalg.norm(fitted - tp[j]))
    return out


def per_batch(path):
    h = pickle.load(open(path, 'rb'))
    cx, tp = h['chain_contexts'], h['true_positions']
    res, ntr = {}, {}
    for b, ctxs in cx.items():
        d = tp.get(b, {})
        dists = []
        for c in ctxs:
            tid = c['track_id'] if isinstance(c, dict) else c.track_id
            cc = c if isinstance(c, dict) else {k: getattr(c, k) for k in
                                               ('theta0_i', 'phi0_i', 'n_chain', 'x0_fixed', 'step_len', 'alpha_mid')}
            t = d.get(tid) if not isinstance(tid, str) else d.get(tid)
            if t is None:
                # keys may be tuples vs strings depending on serialisation
                for k2, v2 in d.items():
                    if str(k2) == str(tid):
                        t = v2
                        break
            if t is None:
                continue
            dists += initial_residual(cc, t)
        if dists:
            res[b] = float(np.mean(dists)) * 1e4
            ntr[b] = len(ctxs)
    return res, ntr


if __name__ == '__main__':
    A, nA = per_batch(R.seed_files('fit_result/sci_full_ANNEALLONG', True)[0])
    S, nS = per_batch(R.seed_files('fit_result/sci_full_SHUFFLE', True)[0])
    common = sorted(set(A) & set(S))
    print(f'batches with a recomputable residual: ANNEALLONG {len(A)}, SHUFFLE {len(S)}, common {len(common)}')
    if not common:
        raise SystemExit('no common batches -- check the serialisation format')
    a = np.array([A[b] for b in common]); s = np.array([S[b] for b in common])
    print(f'median initial residual   ANNEALLONG {np.median(a):7.0f} um   SHUFFLE {np.median(s):7.0f} um')
    print(f'max |per-batch difference| {np.abs(a - s).max():.1f} um   (identical if the inputs match)')
    print(f'tracks per batch          ANNEALLONG {np.mean([nA[b] for b in common]):.1f}   '
          f'SHUFFLE {np.mean([nS[b] for b in common]):.1f}')
    print('\nfirst 8 batches:')
    for b in common[:8]:
        print(f'  batch {b:3d}: ANNEALLONG {A[b]:7.0f}  SHUFFLE {S[b]:7.0f}  ntracks {nA[b]:3d}/{nS[b]:3d}')
