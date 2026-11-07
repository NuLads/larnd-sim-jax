"""Shared loader for a06/a07 sweeps: production simulation + production loss on the hero event (CPU). Not an animation."""
import os, sys, time, pickle, numpy as np
PJ = '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS'
if PJ not in sys.path: sys.path.insert(0, PJ)
T = '/sdf/scratch/neutrino/pgranger/talk_animations'
SMALL = f'{T}/data/a06_hero_small.npz'
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']


def make_small():
    """Extract what a06/a07 need from hero.pkl (249 MB) once."""
    h = pickle.load(open(f'{T}/data/hero.pkl', 'rb'))
    f = h['fields']; dx = f.index('dx')
    tr = h['true_rows']; tr = tr[tr[:, dx] > 0]
    o = dict(fields=np.array(f), true_rows=tr, main=np.array(h['main']), total_len=h['main_ctx']['total_len'])
    for k, v in h['data'].items(): o['data_' + k] = v
    for p in PARAMS: o['true_' + p] = h['calib_true'][p]; o['init_' + p] = h['calib_init'][p]
    for k in ('x0', 'u0', 'e1', 'e2'): o['main_' + k] = h['main_ctx'][k]
    for s in ('start', 'iter20000'):
        S = h['states'][s]; P = S['pred']
        o[f'{s}_rows'] = S['rows_tracks']
        for p in PARAMS: o[f'{s}_{p}'] = S['calib'][p]
        for k in ('pid', 'x', 'y', 'nhit', 'q', 'lam0', 't0', 't0sd'): o[f'{s}_{k}'] = P[k]
    np.savez_compressed(SMALL, **o)
    return SMALL


def setup():
    if not os.path.exists(SMALL): make_small()
    z = np.load(SMALL, allow_pickle=False)
    import paper_common as pc                      # chdir(REPO), production detector config
    import jax, jax.numpy as jnp
    from larndsim.losses_jax import llhd_loss, adc2charge
    from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
    truth, resp = pc.params()
    truth = truth.replace(**{p: float(z['true_' + p]) for p in PARAMS})
    fields = tuple(str(s) for s in z['fields'])
    tgt = {k: jnp.asarray(z['data_' + k]) for k in ('pixel_id', 'ticks', 'adcs')}
    loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)     # production: no loss_fn_kw in argv -> defaults sigma_charge=500, eps=1e-8, W=2, sigma_t=1
    sim = LUTProbabilisticSimulation(resp)
    return dict(z=z, truth=truth, resp=resp, fields=fields, tgt=tgt, loss=loss, sim=sim, jax=jax, jnp=jnp, adc2charge=adc2charge)
