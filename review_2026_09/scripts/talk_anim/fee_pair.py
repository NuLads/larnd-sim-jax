"""Shared data and geometry for the electronics pair a04_current_to_hit / a05_random_to_differentiable.
Waveform, throws, analytic P(t): paperfigs/fig3_probabilistic_method_cache.npz. Noise draws of each throw: a04_replay.npz (a04_build_replay.py,
production electronics re-run with the same key, bitwise equal to the cached throws)."""
import os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_d = np.load(f'{_HERE}/../paperfigs/fig3_probabilistic_method_cache.npz')
_r = np.load(f'{_HERE}/a04_replay.npz')
I_E = np.asarray(_d['wfs_pixel'], float)            # induced current, e-/us (Q = dt * cumsum I)
Q = np.asarray(_d['Q'], float)                      # noiseless cumulative charge, e-
VTH = float(_d['Vth']); SIG_R = float(_d['sigma_r']); SIG_U = float(_d['sigma_u'])
DT = float(_d['t_sampling']); HOLD = int(_d['interval'])
NT = I_E.size; SENT = NT - 2                        # production sentinel tick: no hit in this slot
P_LOG = np.asarray(_d['log_prob_pixel'], float)
PROB = np.exp(np.clip(P_LOG, -40, 0))               # P(hit n at tick i), unconditional: sums to lambda_n, not to 1
TICKS = np.asarray(_d['ticks_s']).astype(int); CHG = np.asarray(_d['adc_s'], float); ADC = np.asarray(_r['adc_digit'], float)
NTHROW = TICKS.shape[0]
EPS0 = np.asarray(_r['first_base'], float); IDX = np.asarray(_r['idx_t']).astype(int); EPSU = np.asarray(_r['eps_u'], float)
BASE = np.asarray(_r['base_next'], float); FAILED = np.asarray(_r['failed']).astype(bool)
PED = float(_r['adc_pedestal'])
PIX_ID = int(_d['meta_unique_pixel_id']); PIX_X = float(_d['meta_x_cm']); PIX_Y = float(_d['meta_y_cm'])
T0, T1 = 240, 350                                   # ticks shown: the close double hit of the first strike


def throw(m, scale=1.0):
    """Integrator history of throw m, with the production arithmetic of fee_jax.get_adc_values (float64 here, for display).
    Returns q[t] (chip state, with noise, NaN never), q0[t] (same without the baseline noise) and the list of hit dicts."""
    qc = Q * scale; base = EPS0[m]
    q = np.empty(NT); q0 = np.empty(NT); hits = []; start = 0
    for n in range(IDX.shape[0]):
        s = base + qc
        idx = int(IDX[n, m])
        if idx >= NT - 2:
            q[start:] = s[start:]; q0[start:] = qc[start:]; start = NT; break
        end = idx + 1 + HOLD; reset = end + 1
        frac = (VTH - s[idx]) / (s[idx + 1] - s[idx])
        q[start:reset] = s[start:reset]; q0[start:reset] = qc[start:reset]
        hits.append(dict(n=n, tick=idx, cross=idx + frac, end=end, reset=reset, eps_r=base, eps_u=EPSU[n, m], q_pre=s[end], q_meas=s[end] + EPSU[n, m], failed=bool(FAILED[n, m]),
                         adc=float(ADC[m, n]), base_next=BASE[n, m]))
        qc = qc - qc[reset]; qc = np.where(qc < 0, 0.0, qc); base = BASE[n, m]; start = reset
    if start < NT:
        q[start:] = (base + qc)[start:]; q0[start:] = qc[start:]
    return q, q0, hits


def check(m):
    """Display replay vs the cached hit of the same throw."""
    _, _, h = throw(m)
    for x in h:
        if not x['failed']:
            assert TICKS[m, x['n']] == x['tick'], (m, x)
            assert abs(CHG[m, x['n']] - x['q_meas']) < 30, (m, x, CHG[m, x['n']])     # float32 vs float64 cumsum


def tomu(tick):
    return np.asarray(tick, float) * DT
