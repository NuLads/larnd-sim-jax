"""Derived cache for a04/a05. Takes the cached real pixel of fig3_probabilistic_method_cache.npz and re-runs the PRODUCTION electronics
(src/larndsim/fee_jax.py) on it with the SAME RNG key, recording what the cache does not keep: the noise draws of each throw.
find_hit below is the body of fee_jax.get_adc_values, copied line by line with extra outputs; it is checked bitwise against the
cached adc_s/ticks_s, so the noise shown in the animations is the noise that made those hits.
Also evaluates the production probabilistic model on the waveform scaled by (1+s), for the last beat of a05."""
import os, sys, time
import numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import jax, jax.numpy as jnp
from jax import random, vmap, lax
import paper_common as pc
from larndsim.fee_jax import get_adc_values, get_adc_values_average_noise_vmap, digitize, get_average_hit_values
HERE = os.path.dirname(os.path.abspath(__file__))
d = dict(np.load('review_2026_09/scripts/paperfigs/fig3_probabilistic_method_cache.npz'))
params, _ = pc.params()
SEED = 12345                        # throw_seed of fig3_probabilistic_method.compute_detail (its default; cache made with --n_throws 10000)
wfs_pixel = jnp.asarray(d['wfs_pixel']); N = int(d['n_throws'])
wfs = jnp.tile(wfs_pixel[None, :], (N, 1))
t0 = time.time()
a_prod, t_prod = get_adc_values(params, wfs, random.key(SEED)); a_prod, t_prod = np.asarray(a_prod), np.asarray(t_prod)
print('production rerun', time.time() - t0, 's; identical to cache:', np.array_equal(a_prod, d['adc_s']), np.array_equal(t_prod, d['ticks_s']), flush=True)


def traced(params, pixels_signals, noise_rng_key, nslots=5):
    q_sum_base = random.normal(noise_rng_key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE
    q = pixels_signals * params.t_sampling
    q_cumsum = q.cumsum(axis=-1)
    q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum
    first_base = q_sum_base

    def find_hit(carry, it):
        key, q_sum, q_cumsum = carry
        selec_func = lambda x: jnp.where((x[1:] >= params.DISCRIMINATION_THRESHOLD) & (x[:-1] <= params.DISCRIMINATION_THRESHOLD), size=1, fill_value=q_sum.shape[1] - 2)
        idx_t, = vmap(selec_func, 0, 0)(q_sum)
        idx_t = idx_t.ravel()
        ic = jnp.zeros((q_sum.shape[0],)); idx_pix = jnp.arange(0, q_sum.shape[0]); ic = ic.at[idx_pix].set(idx_t)
        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        integrate_end = idx_t + 1 + interval
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1] - 1, integrate_end)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end]))
        q_vals = q_sum[end2d_idx]; q_vals_no_noise = q_cumsum[end2d_idx]
        key, = random.split(key, 1)
        extra_noise = random.normal(key, (pixels_signals.shape[0],)) * params.UNCORRELATED_NOISE_CHARGE
        adc = jnp.where(q_vals_no_noise != 0, q_vals + extra_noise, q_vals_no_noise)
        cond_adc = adc < params.DISCRIMINATION_THRESHOLD
        cond_adc = jnp.logical_or(cond_adc, idx_t == q_sum.shape[1] - 2)
        adc = jnp.where(cond_adc, 0, adc)
        ic = jnp.where(cond_adc, q_sum.shape[1] - 2, ic)
        key, = random.split(key, 1)
        q_adc_pass = random.normal(key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE
        key, = random.split(key, 1)
        q_adc_fail = random.normal(key, (pixels_signals.shape[0],)) * params.UNCORRELATED_NOISE_CHARGE
        q_sum_base = jnp.where(cond_adc, q_adc_fail, q_adc_pass)
        integrate_end2 = idx_t + 1 + interval + 1
        integrate_end2 = jnp.where(integrate_end2 >= q_sum.shape[1], q_sum.shape[1] - 1, integrate_end2)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end2]))
        q_vals_no_noise2 = q_cumsum[end2d_idx]
        q_cumsum = q_cumsum - q_vals_no_noise2[..., jnp.newaxis]
        q_cumsum = jnp.where(q_cumsum < 0, 0, q_cumsum)
        q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum
        return (key, q_sum, q_cumsum), (adc, ic, idx_t, extra_noise, q_adc_pass, q_adc_fail, q_sum_base, cond_adc, q_vals_no_noise2)

    init_loop = (random.split(noise_rng_key, 1)[0], q_sum, q_cumsum)
    _, out = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    return first_base, out


t0 = time.time()
first_base, out = jax.jit(traced)(params, wfs, random.key(SEED))   # jit as production: same float32 cumsum order
adc, ic, idx_t, eps_u, q_pass, q_fail, base_next, failed, q_removed = [np.asarray(x) for x in out]
print('traced', time.time() - t0, 's; shape', adc.shape, flush=True)
print('diff', np.abs(adc.T-d['adc_s']).max(), np.abs(ic.T-d['ticks_s']).max(), (adc.T!=d['adc_s']).sum(), (ic.T!=d['ticks_s']).sum(), flush=True)
assert np.array_equal(adc.T, d['adc_s']) and np.array_equal(ic.T, d['ticks_s']), 'traced replay differs from the cached throws'
print('traced replay bitwise identical to cached adc_s, ticks_s', flush=True)
adc_digit = np.asarray(digitize(params, jnp.asarray(d['adc_s'])))
adc_pedestal = float(digitize(params, jnp.zeros(())))
gain = dict(GAIN=float(params.GAIN), V_PEDESTAL=float(params.V_PEDESTAL), V_CM=float(params.V_CM), V_REF=float(params.V_REF), ADC_COUNTS=float(params.ADC_COUNTS),
            CLOCK_CYCLE=float(params.CLOCK_CYCLE), ADC_HOLD_DELAY=float(params.ADC_HOLD_DELAY))
print(gain, 'pedestal ADC', adc_pedestal, flush=True)
M = 12                               # keep the noise of the first throws in full; for all throws keep the slots a05 needs
out_d = dict(first_base=np.asarray(first_base), idx_t=idx_t[:5], eps_u=eps_u[:5], base_next=base_next[:5], failed=failed[:5], q_pass=q_pass[:5], q_fail=q_fail[:5],
             q_removed=q_removed[:5], adc_digit=adc_digit[:, :5], adc_pedestal=adc_pedestal, **gain)

# probabilistic model on the scaled waveform, production function, one call per scale
scales = np.round(np.linspace(-0.04, 0.04, 81), 4)
t0 = time.time()
f = lambda w: get_adc_values_average_noise_vmap(params, w)
lp0, ch0 = f(wfs_pixel[None, :])
print('model rerun at s=0: max |log p - cached| on p>1e-4:', float(np.abs(np.asarray(lp0[0]) - d['log_prob_pixel'])[np.exp(d['log_prob_pixel']) > 1e-4].max()), flush=True)
LP = []; ET = []; LAM = []
for s in scales:
    lp, ch = f(wfs_pixel[None, :] * (1 + float(s)))
    et, _, lam = get_average_hit_values(lp, digitize(params, ch))
    LP.append(np.asarray(lp[0, :3, 150:500])); ET.append(np.asarray(et[0, :3])); LAM.append(np.asarray(lam[0, :3]))
print('scale scan', time.time() - t0, 's', flush=True)
out_d.update(scales=scales, scan_logp=np.stack(LP), scan_Et=np.stack(ET), scan_lam=np.stack(LAM), scan_tick0=150)

# stochastic at the same scales, same key (same noise draws), production function, all throws
TK = []
t0 = time.time()
for s in scales:
    a_s, t_s = get_adc_values(params, wfs * (1 + float(s)), random.key(SEED))
    TK.append(np.asarray(t_s[:M, :3]))
print('stochastic scan', time.time() - t0, 's', flush=True)
out_d.update(scan_ticks=np.stack(TK))
np.savez_compressed(f'{HERE}/a04_replay.npz', **out_d)
print('wrote', f'{HERE}/a04_replay.npz', os.path.getsize(f'{HERE}/a04_replay.npz') / 1e6, 'MB')
for m in range(6):
    print(m, 'eps_r0', round(float(first_base[m])), 'ticks', d['ticks_s'][m, :4], 'q', np.round(d['adc_s'][m, :4]), 'adc', np.round(adc_digit[m, :4], 1), 'eps_u', np.round(eps_u[:3, m]), 'base_next', np.round(base_next[:3, m]))
