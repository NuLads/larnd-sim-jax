"""Verify the padded-cell mask: (a) shapes broadcast correctly, (b) with no padding the
result is unchanged, (c) with padding the phantom contribution is removed."""
import jax.numpy as jnp, numpy as np
Npix, Nval, Ntick = 6, 3, 40
rng = np.random.default_rng(0)
tp = jnp.asarray(rng.normal(-3, 1, (Npix, Nval, Ntick)).astype(np.float32))

def old(ticks_prob, first=False):
    return float(jnp.sum(jnp.exp(ticks_prob[:, 0, :]))) if first else float(jnp.sum(jnp.exp(ticks_prob)))
def new(ticks_prob, upix, first=False):
    v = (upix >= 0)
    if first:
        return float(jnp.sum(jnp.where(v[:, None], jnp.exp(ticks_prob[:, 0, :]), 0.0)))
    return float(jnp.sum(jnp.where(v[:, None, None], jnp.exp(ticks_prob), 0.0)))

no_pad = jnp.asarray(np.arange(Npix))                       # all real
padded = jnp.asarray([10, 11, 12, -1, -1, -1])              # last 3 are padding
for first in (False, True):
    tag = 'first_hit_only' if first else 'all hits'
    o, n_nopad = old(tp, first), new(tp, no_pad, first)
    n_pad = new(tp, padded, first)
    direct = old(tp[:3], first)                              # ground truth: real rows only
    print(f'{tag:15s} unpadded: old {o:.5f} new {n_nopad:.5f}  identical={abs(o-n_nopad)<1e-6}')
    print(f'{tag:15s} padded  : old {o:.5f} new {n_pad:.5f}  matches real-rows-only {direct:.5f} = {abs(n_pad-direct)<1e-6}')
    print(f'{tag:15s} phantom removed: {o-n_pad:.5f} ({(o-n_pad)/o*100:.0f}% of the old term)')
