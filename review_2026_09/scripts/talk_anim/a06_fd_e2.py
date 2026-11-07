"""Second step size for the E finite difference at -1.5%, where only h=3e-4 was measured."""
import json, time, numpy as np
from a06_hero_common import *
S = setup(); jax, jnp = S['jax'], S['jnp']; truth, f, tgt, loss, sim, z = S['truth'], S['fields'], S['tgt'], S['loss'], S['sim'], S['z']
tracks = jnp.asarray(z['true_rows']); tv = float(z['true_eField']); fo = f'{T}/data/a06_sweep/fd_e_steps2.json'
def L(x):
    P = truth.replace(eField=x); return float(loss.compute(P, sim.predict(P, tracks, f, None), tgt)[0])
res = []
for c, h in ((0.985, 1e-4),):
    x = tv * c; up, dn = L(x * (1 + h)), L(x * (1 - h)); fd = (up - dn) / (2 * x * h); res.append(dict(c=c, h=h, fd=fd, up=up, dn=dn, x=x))
    print('FDE2 c', c, 'h', h, 'FD', fd, 'x*FD', fd * x, flush=True); json.dump(res, open(fo, 'w'), indent=1)
