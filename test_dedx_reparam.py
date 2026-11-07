#!/usr/bin/env python3
"""Unit tests for the (scale, shape) dEdx reparametrisation. CPU only, no simulation.

  1. mean_dedx == exp(mu) EXACTLY, whatever the shape a is
  2. the analytic gradient map matches JAX autodiff through the forward
  3. the gauge direction (a -> a + c) is exactly flat, and gets exactly zero gradient
  4. the mean constraint's Hessian in (mu, a) has NO a-a coupling -- the rank-1 term is gone
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '.')
from optimize.fit_params import _dedx_scale_shape_forward, _dedx_scale_shape_grads

rng = np.random.default_rng(0)
FAIL = 0


def check(name, cond, detail=''):
    global FAIL
    print('  %-56s %s  %s' % (name, 'PASS' if cond else '**FAIL**', detail))
    if not cond:
        FAIL += 1


N = 13
l = np.abs(rng.normal(size=N)) * 3 + 0.03      # segment lengths incl. one very short
l[5] = 0.0355
a = rng.normal(size=N) * 0.15
mu = np.log(1.874)
L = l.sum()

print('1. mean_dedx == exp(mu) exactly, independent of the shape')
for trial in range(4):
    aa = rng.normal(size=N) * (0.05 + trial)
    ld = np.asarray(_dedx_scale_shape_forward(jnp.array(mu), jnp.array(aa), jnp.array(l)))
    m = (l * np.exp(ld)).sum() / L
    check('trial %d  mean=%.10f vs exp(mu)=%.10f' % (trial, m, np.exp(mu)),
          abs(m - np.exp(mu)) / np.exp(mu) < 1e-12,
          'rel %.2e' % (abs(m - np.exp(mu)) / np.exp(mu)))

print('\n2. analytic gradient map == JAX autodiff through the forward')
for trial in range(4):
    aa = rng.normal(size=N) * 0.2
    mm = np.log(1.8 + 0.1 * trial)
    g = rng.normal(size=N)                      # an arbitrary dL/dlog_dedx

    def scalar(mu_, a_):
        return jnp.sum(_dedx_scale_shape_forward(mu_, a_, jnp.array(l)) * jnp.array(g))

    gmu_ad, ga_ad = jax.grad(scalar, argnums=(0, 1))(jnp.array(mm), jnp.array(aa))
    gmu, ga = _dedx_scale_shape_grads(g, aa, l)
    check('trial %d  dL/dmu' % trial, abs(gmu - float(gmu_ad)) < 1e-10,
          'analytic %.6f vs AD %.6f' % (gmu, float(gmu_ad)))
    check('trial %d  dL/da' % trial, np.abs(ga - np.asarray(ga_ad)).max() < 1e-10,
          'max diff %.2e' % np.abs(ga - np.asarray(ga_ad)).max())

print('\n3. gauge direction a -> a + c is exactly flat and gets zero gradient')
ld0 = np.asarray(_dedx_scale_shape_forward(jnp.array(mu), jnp.array(a), jnp.array(l)))
for c in (0.1, 1.0, -5.0):
    ld1 = np.asarray(_dedx_scale_shape_forward(jnp.array(mu), jnp.array(a + c), jnp.array(l)))
    check('shift c=%+.1f leaves log_dedx unchanged' % c, np.abs(ld1 - ld0).max() < 1e-12,
          'max diff %.2e' % np.abs(ld1 - ld0).max())
g = rng.normal(size=N)
_, ga = _dedx_scale_shape_grads(g, a, l)
check('sum(dL/da) == 0 (no drift along the gauge)', abs(ga.sum()) < 1e-10, 'sum %.2e' % ga.sum())

print('\n4. the mean constraint no longer couples the shape parameters')
w, t = 3000.0, 1.887


def mean_penalty(mu_, a_):
    ld = _dedx_scale_shape_forward(mu_, a_, jnp.array(l))
    m = jnp.sum(jnp.array(l) * jnp.exp(ld)) / L
    return w * (m - t) ** 2


H = jax.hessian(mean_penalty, argnums=(0, 1))(jnp.array(mu), jnp.array(a))
Haa = np.asarray(H[1][1])
Hmm = float(np.asarray(H[0][0]))
check('H_aa == 0 (rank-1 coupling GONE)', np.abs(Haa).max() < 1e-6,
      'max|H_aa| = %.2e  vs H_mumu = %.4g' % (np.abs(Haa).max(), Hmm))

# same penalty in the OLD per-segment parametrisation, for contrast
def mean_penalty_old(ld):
    m = jnp.sum(jnp.array(l) * jnp.exp(ld)) / L
    return w * (m - t) ** 2


Hold = np.asarray(jax.hessian(mean_penalty_old)(jnp.array(ld0)))
off = Hold - np.diag(np.diag(Hold))
check('OLD parametrisation DOES couple (contrast)', np.abs(off).max() > 1.0,
      'max|off-diag| = %.4g' % np.abs(off).max())
ev = np.linalg.eigvalsh(0.5 * (Hold + Hold.T))
# NB the old form is rank-1 PLUS a diagonal 2w(m-t)diag(g) term, so the sub-leading eigenvalues
# are small but NOT zero (and are negative when m < t). Assert dominance, not exact rank 1.
check('OLD is dominated by ONE direction', ev[-1] / max(abs(ev[-2]), 1e-12) > 1e3,
      'top two eig: %.4g , %.4g  (ratio %.0f)' % (ev[-1], ev[-2], ev[-1] / max(abs(ev[-2]), 1e-12)))
# (Do NOT assert that `off` itself is rank-1: a rank-1 matrix with its diagonal REMOVED is not
# rank-1. The factorisation H_ij = q_i q_j for i != j is tested properly in
# test_dedx_gn_numerics.py; here the two checks above already establish the contrast.)
i_, j_ = 0, 3
k_, m_ = 5, 9
check('OLD off-diagonals factorise: H_ij H_kl == H_il H_kj',
      abs(off[i_, j_] * off[k_, m_] - off[i_, m_] * off[k_, j_]) /
      max(abs(off[i_, j_] * off[k_, m_]), 1e-30) < 1e-6,
      'minor rel %.2e' % (abs(off[i_, j_] * off[k_, m_] - off[i_, m_] * off[k_, j_]) /
                          max(abs(off[i_, j_] * off[k_, m_]), 1e-30)))

print('\n%s' % ('ALL TESTS PASSED' if FAIL == 0 else '%d TEST(S) FAILED' % FAIL))
sys.exit(1 if FAIL else 0)
