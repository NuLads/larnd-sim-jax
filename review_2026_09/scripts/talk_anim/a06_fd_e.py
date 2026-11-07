"""Is the E gradient disagreement a step-size effect or a real difference? Central differences of the production loss at several steps, plus jax.grad, at several E."""
import json, time, numpy as np
from a06_hero_common import *
S = setup(); jax, jnp = S['jax'], S['jnp']; truth, f, tgt, loss, sim, z = S['truth'], S['fields'], S['tgt'], S['loss'], S['sim'], S['z']
tracks = jnp.asarray(z['true_rows']); tv = float(z['true_eField']); fo = f'{T}/data/a06_sweep/fd_e_steps.json'; res = json.load(open(fo)) if os.path.exists(fo) else []
def L(x):
    P = truth.replace(eField=x); return loss.compute(P, sim.predict(P, tracks, f, None), tgt)[0]
c0 = {}
def val(x):
    k = round(x, 10)
    if k not in c0: c0[k] = float(L(x))
    return c0[k]
for c in (1.0, 1.0075):
    x = tv * c; ad = None
    for h in (1e-4, 1e-3, 3e-3):
        t0 = time.time(); fd = (val(x * (1 + h)) - val(x * (1 - h))) / (2 * x * h)
        res.append(dict(c=c, h=h, fd=fd, up=val(x * (1 + h)), dn=val(x * (1 - h)), x=x))
        print('FDE c', c, 'h', h, 'FD', fd, 'x*FD', fd * x, 'tsec', round(time.time() - t0), flush=True)
        with open(fo + '.tmp', 'w') as g: json.dump(res, g, indent=1)
        os.replace(fo + '.tmp', fo)
