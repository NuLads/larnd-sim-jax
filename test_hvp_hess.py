"""Unit test: _hess_by_hvp must reproduce jacfwd(jacrev(.)) exactly on a toy problem
with the same pytree shape as the real norm_params."""
import numpy as np, jax, jax.numpy as jnp
from flax import struct
from optimize.fit_params import _hess_by_hvp

KEYS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

@struct.dataclass
class Toy:
    Ab: jnp.ndarray
    eField: jnp.ndarray
    tran_diff: jnp.ndarray
    long_diff: jnp.ndarray
    lifetime: jnp.ndarray
    tag: str = struct.field(pytree_node=False, default='static')

rng = np.random.default_rng(3)
A = rng.standard_normal((5, 5)); A = A @ A.T + 5*np.eye(5)     # SPD, non-trivial cross terms
p = Toy(**{k: jnp.asarray(float(v)) for k, v in zip(KEYS, rng.standard_normal(5))})

def f(pp):
    x = jnp.stack([getattr(pp, k) for k in KEYS])
    # non-quadratic so 3rd derivatives are nonzero -> a real test, not a linear special case
    return 0.5 * x @ jnp.asarray(A) @ x + jnp.sum(jnp.sin(x)**3), jnp.sum(x)

ref, _ = jax.jacfwd(jax.jacrev(f, has_aux=True), has_aux=True)(p)
H_ref = np.array([[float(getattr(getattr(ref, ka), kb)) for kb in KEYS] for ka in KEYS])
hv, aux = _hess_by_hvp(f, p, KEYS)
H_hvp = np.array([[float(getattr(getattr(hv, ka), kb)) for kb in KEYS] for ka in KEYS])

print('max |H_hvp - H_jacfwd| =', np.abs(H_hvp - H_ref).max())
print('symmetric?             ', np.abs(H_hvp - H_hvp.T).max() < 1e-5)
print('aux preserved?         ', float(aux))
print('PASS' if np.allclose(H_hvp, H_ref, atol=1e-4) else 'FAIL')
