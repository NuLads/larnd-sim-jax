#!/usr/bin/env python3
"""Analyse the dEdx mode-vs-mean bias toy (start_dedx_bias_toy.sh).

For each arm reports:
  * the fitted dEdx mean vs the TRUE mean of the same segments  -> THE BIAS under test
  * fitted dispersion vs true dispersion                        -> the over-concentration
  * dEdx MAE
  * the 5 calibration parameter errors                          -> does dropping the mean
                                                                   constraint actually hurt?
Arms are keyed by (loc, jacobian, mean_weight).
"""
import glob
import os
import pickle
import re
import sys

import numpy as np

TRUTH = {'Ab': 0.83488, 'eField': 0.52152, 'tran_diff': 1.0028e-05,
         'long_diff': 5.8142e-06, 'lifetime': 2406.4}
PRIOR_MODE, PRIOR_MEAN = 1.8740, 1.8847


def collect(pkl):
    with open(pkl, 'rb') as f:
        h = pickle.load(f)
    dedx_cache = h.get('dedx_cache') or {}
    true_cache = h.get('true_dedx_cache') or {}
    pid_cache = h.get('batch_parent_ids') or {}
    fit_all, true_all = [], []
    for b, entry in dedx_cache.items():
        ld = np.asarray(entry['log_dedx'])
        td = np.asarray(true_cache.get(b, []))
        if td.size == 0:
            continue
        n = min(ld.shape[0], td.shape[0])
        # only segments that actually carry a fitted parameter
        pid = np.asarray(pid_cache.get(b, []))
        if pid.size:
            used = np.unique(pid[pid >= 0])
            used = used[used < n]
        else:
            used = np.arange(n)
        fit_all.append(np.exp(ld[used]))
        true_all.append(td[used])
    if not fit_all:
        return None
    fit = np.concatenate(fit_all)
    tru = np.concatenate(true_all)
    ok = np.isfinite(fit) & np.isfinite(tru) & (tru > 0)
    fit, tru = fit[ok], tru[ok]

    cur = h.get('current_params_state') or {}
    params = {}
    for k in TRUTH:
        v = cur.get(k)
        if v is None:
            continue
        params[k] = 100.0 * (float(np.asarray(v).ravel()[0]) - TRUTH[k]) / TRUTH[k]
    return dict(n=fit.size, fit=fit, true=tru, params=params,
                loc=h.get('dedx_student_loc'), iters=h.get('total_iter'))


def main(pattern='fit_result/dedx_bias/history_iter*_dedxbias_*.pkl'):
    files = sorted(glob.glob(pattern))
    if not files:
        print('no checkpoints matching', pattern)
        return 1
    # keep the highest-iteration checkpoint per arm
    best = {}
    for f in files:
        m = re.search(r'history_iter(\d+)_dedxbias_(.+)\.pkl$', os.path.basename(f))
        if not m:
            continue
        it, tag = int(m.group(1)), m.group(2)
        if tag not in best or it > best[tag][0]:
            best[tag] = (it, f)

    rows = []
    for tag, (it, f) in sorted(best.items()):
        r = collect(f)
        if r is None:
            print('  (no dedx cache in %s)' % f)
            continue
        m = re.match(r'loc([\d.]+)_jac(\d)_mw([\d.]+)_off([\d.]+)', tag)
        loc, jac, mw, off = (float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))) if m else (0,0,0,0)
        fit, tru = r['fit'], r['true']
        rows.append(dict(tag=tag, it=it, loc=loc, jac=jac, mw=mw, off=off, n=r['n'],
                         fit_mean=fit.mean(), true_mean=tru.mean(),
                         bias=100 * (fit.mean() - tru.mean()) / tru.mean(),
                         fit_sd=fit.std(), true_sd=tru.std(),
                         disp=100 * (fit.std() / tru.std() - 1),
                         mae=np.abs(fit - tru).mean(), params=r['params']))

    print('=' * 118)
    print('dEdx MODE-vs-MEAN BIAS TOY   (prior mode %.4f, prior mean %.4f, mean-constraint target 1.887)'
          % (PRIOR_MODE, PRIOR_MEAN))
    print('=' * 118)
    hdr = ('%-8s %-4s %-9s | %5s | %8s %8s %8s | %8s %8s | %7s'
           % ('loc', 'jac', 'mean_w', 'nseg', 'fit_mean', 'true_mn', 'BIAS %', 'fit_sd', 'DISP %', 'MAE'))
    print(hdr); print('-' * 118)
    for r in sorted(rows, key=lambda r: (r['loc'], r['jac'], -r['mw'])):
        print('%-8.4f %-4d %-9.0f | %5d | %8.4f %8.4f %+8.3f | %8.4f %+8.1f | %7.4f'
              % (r['loc'], r['jac'], r['mw'], r['n'], r['fit_mean'], r['true_mean'],
                 r['bias'], r['fit_sd'], r['disp'], r['mae']))
    print()
    print('CALIBRATION PARAMETER ERRORS (%)  -- does removing the mean constraint hurt them?')
    print('-' * 118)
    print('%-8s %-4s %-9s | %8s %8s %10s %10s %9s' %
          ('loc', 'jac', 'mean_w', 'Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime'))
    print('-' * 118)
    for r in sorted(rows, key=lambda r: (r['loc'], r['jac'], -r['mw'])):
        p = r['params']
        print('%-8.4f %-4d %-9.0f | %+8.3f %+8.3f %+10.3f %+10.3f %+9.3f'
              % (r['loc'], r['jac'], r['mw'],
                 p.get('Ab', float('nan')), p.get('eField', float('nan')),
                 p.get('tran_diff', float('nan')), p.get('long_diff', float('nan')),
                 p.get('lifetime', float('nan'))))
    print()
    print('BIAS   = 100*(mean(fitted dEdx)/mean(true dEdx) - 1)   -> 0 is unbiased')
    print('DISP % = 100*(sd(fitted)/sd(true) - 1)                 -> negative = over-concentrated (MAP shrinkage)')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
