#!/usr/bin/env python3
"""Unit tests for the structured dEdx Newton solve (no GPU, no simulation).

Validates, against dense linear algebra:
  1. mod-P probing recovers a bandwidth-2 matrix EXACTLY from 5 mat-vecs
  2. the probing is not fooled by an added dense rank-1 (which must be subtracted first)
  3. banded + Sherman-Morrison reproduces a dense solve of (B + lam I + q q^T)
  4. it still works when B has a near-zero diagonal entry (the short-segment case)
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from optimize.fit_params import (_dedx_probe_matrix, _dedx_band_from_probes,
                                 _dedx_banded_sm_solve, _DEDX_GN_BW)

rng = np.random.default_rng(0)
BW = _DEDX_GN_BW
FAIL = 0


def check(name, cond, detail=''):
    global FAIL
    print('  %-58s %s  %s' % (name, 'PASS' if cond else '**FAIL**', detail))
    if not cond:
        FAIL += 1


def make_banded(N, bw=BW, jitter=0.0):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(max(0, i - bw), min(N, i + bw + 1)):
            if j >= i:
                v = rng.normal()
                A[i, j] = A[j, i] = v
    A += np.diag(np.abs(rng.normal(size=N)) * 20 + 5 + jitter)
    return A


print('1. mod-P probing recovers a bandwidth-2 matrix exactly')
for N in (7, 13, 40, 377):
    A = make_banded(N)
    V = _dedx_probe_matrix(N, BW)
    probes = np.stack([A @ V[c] for c in range(2 * BW + 1)])
    ab = _dedx_band_from_probes(probes, N, BW)
    # rebuild dense from ab
    R = np.zeros((N, N))
    i = np.arange(N)
    for d in range(-BW, BW + 1):
        j = i + d
        ok = (j >= 0) & (j < N)
        R[i[ok], j[ok]] = ab[BW - d, j[ok]]
    check('N=%-4d exact band recovery from %d mat-vecs' % (N, 2 * BW + 1),
          np.allclose(R, A, atol=1e-10), 'max err %.2e' % np.abs(R - A).max())

print('\n2. a dense rank-1 BREAKS probing unless subtracted (sanity: the trap is real)')
N = 40
A = make_banded(N)
q = np.abs(rng.normal(size=N)) + 0.5
H = A + np.outer(q, q)
V = _dedx_probe_matrix(N, BW)
probes_raw = np.stack([H @ V[c] for c in range(2 * BW + 1)])
probes_fix = np.stack([H @ V[c] - q * (q @ V[c]) for c in range(2 * BW + 1)])
ab_raw = _dedx_band_from_probes(probes_raw, N, BW)
ab_fix = _dedx_band_from_probes(probes_fix, N, BW)
ab_true = _dedx_band_from_probes(np.stack([A @ V[c] for c in range(2 * BW + 1)]), N, BW)
check('raw probes are WRONG (rank-1 contamination)', not np.allclose(ab_raw, ab_true, atol=1e-8),
      'max err %.2e' % np.abs(ab_raw - ab_true).max())
check('rank-1-subtracted probes are exact', np.allclose(ab_fix, ab_true, atol=1e-10),
      'max err %.2e' % np.abs(ab_fix - ab_true).max())

print('\n2b. vectorised rank-1 subtraction == per-probe form (shape-order regression)')
N = 11
q = np.abs(rng.normal(size=N)) + 0.5
V = _dedx_probe_matrix(N, BW)
raw = rng.normal(size=(V.shape[0], N))
vec = raw - np.outer(V @ q, q)
per = np.stack([raw[c] - q * float(q @ V[c]) for c in range(V.shape[0])])
check('outer(V@q, q) matches per-probe loop', vec.shape == per.shape and np.allclose(vec, per),
      'shape %s vs %s' % (vec.shape, per.shape))
check('outer(q, V@q) is the WRONG order (would crash/corrupt)',
      np.outer(q, V @ q).shape != per.shape, 'wrong shape %s' % (np.outer(q, V @ q).shape,))

print('\n3. banded + Sherman-Morrison == dense solve of (B + lam I + q q^T)')
for N in (13, 40, 377):
    for lam in (0.0, 1.0, 50.0):
        A = make_banded(N)
        q = np.abs(rng.normal(size=N)) + 0.5
        rhs = rng.normal(size=N)
        V = _dedx_probe_matrix(N, BW)
        probes = np.stack([(A + np.outer(q, q)) @ V[c] - q * (q @ V[c])
                           for c in range(2 * BW + 1)])
        ab = _dedx_band_from_probes(probes, N, BW)
        x, ok = _dedx_banded_sm_solve(ab, q, rhs, lam, BW)
        xd = np.linalg.solve(A + lam * np.eye(N) + np.outer(q, q), rhs)
        rel = np.abs(x - xd).max() / max(np.abs(xd).max(), 1e-30)
        check('N=%-4d lam=%-5.1f SM matches dense' % (N, lam), ok and rel < 1e-8,
              'rel err %.2e' % rel)

print('\n4. short-segment case: one near-zero diagonal (the q_i^2 + c q_i cancellation)')
N = 40
A = make_banded(N)
A[7, 7] = 1e-3                      # a segment whose curvature was cancelled
q = np.abs(rng.normal(size=N)) + 0.5
rhs = rng.normal(size=N)
V = _dedx_probe_matrix(N, BW)
probes = np.stack([(A + np.outer(q, q)) @ V[c] - q * (q @ V[c]) for c in range(2 * BW + 1)])
ab = _dedx_band_from_probes(probes, N, BW)
for lam in (0.0, 1.0):
    x, ok = _dedx_banded_sm_solve(ab, q, rhs, lam, BW)
    xd = np.linalg.solve(A + lam * np.eye(N) + np.outer(q, q), rhs)
    rel = np.abs(x - xd).max() / max(np.abs(xd).max(), 1e-30) if ok else np.inf
    check('near-zero diagonal, lam=%.1f' % lam, ok and rel < 1e-6, 'rel err %.2e' % rel)

A2 = A.copy(); A2[7, 7] = -5.0      # driven NEGATIVE
probes = np.stack([(A2 + np.outer(q, q)) @ V[c] - q * (q @ V[c]) for c in range(2 * BW + 1)])
ab2 = _dedx_band_from_probes(probes, N, BW)
x, ok = _dedx_banded_sm_solve(ab2, q, rhs, 0.0, BW)
xd = np.linalg.solve(A2 + np.outer(q, q), rhs)
rel = np.abs(x - xd).max() / max(np.abs(xd).max(), 1e-30) if ok else np.inf
check('NEGATIVE diagonal entry still solved correctly', ok and rel < 1e-6, 'rel err %.2e' % rel)

print('\n%s' % ('ALL TESTS PASSED' if FAIL == 0 else '%d TEST(S) FAILED' % FAIL))
sys.exit(1 if FAIL else 0)
