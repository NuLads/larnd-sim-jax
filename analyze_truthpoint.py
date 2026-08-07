"""Study 2: is the loss minimum misplaced, or merely unreachable?

Reads the truth-point evaluations (6 seeds x 100 batches x 5 points) and answers three
questions, in this order:

  1. Sign of loss(truth) - loss(fit), per batch.
       < 0 : truth is a BETTER point than the fit found -> the minimum is where it should be
             and the optimiser fails to reach it.
       > 0 : the fit found a genuinely better point than truth -> the minimum is DISPLACED and
             no optimiser change can fix it.
  2. Does that verdict survive splitting the objective into LIKELIHOOD and PRIORS? If truth
     loses on the total but wins on the likelihood, the defect is in our priors, not in the
     physics model -- a completely different fix. The decomposition is exact (4.5e-7).
  3. Which BLOCK does the objective disagree about? P2/P3/P4 move exactly one block to truth.

Every number is per batch. Batches differ by ~4x in total loss (585 to several thousand matched
hits), so absolute deltas are also reported normalised by the fit loss -- an absolute mean over
batches is dominated by the biggest ones.
"""
import glob, pickle, re
import numpy as np

RUN = 'fit_result/sci_full_TP'
POINTS = ['P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true']
LABEL = {'P0_fit': 'fit', 'P1_truth': 'truth (all blocks)', 'P2_calib_true': 'calibration -> truth',
         'P3_dedx_true': 'dE/dx -> truth', 'P4_geom_true': 'geometry -> truth'}
# total loss = likelihood + priors, exactly (verified 4.5e-7 relative in fig 44)
LIK = ['log_likelihood_tick', 'no_match_penalty', 'expected_total_hits']
PRI = ['dedx_prior', 'dedx_mean_penalty', 'dedx_barrier', 'mcs_prior',
       'chain_drift_penalty', 'dedx_drift_penalty', 'spatial_moment']


def load():
    rows = []
    for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
        seed = int(re.search(r'seed(\d+)', f).group(1))
        for rec in pickle.load(open(f, 'rb'))['truthpoint']:
            if not all(p in rec['points'] for p in POINTS):
                continue
            d = {'seed': seed, 'batch': rec['batch'], 'floor': rec['repr_floor_um'],
                 'n_tracks': rec['n_tracks']}
            for p in POINTS:
                a = rec['points'][p]['aux']
                d[p] = rec['points'][p]['loss']
                d[p + '_lik'] = sum(float(a.get(k, 0.0)) for k in LIK)
                d[p + '_pri'] = sum(float(a.get(k, 0.0)) for k in PRI)
                for k in ('matched_hits', 'mean_dedx', 'n_active_pixels'):
                    d[f'{p}_{k}'] = float(a.get(k, np.nan))
            rows.append(d)
    return rows


R = load()
S = sorted({r['seed'] for r in R})
print(f'{len(R)} batch evaluations over {len(S)} seeds {S}\n')

# ---- closure check: does likelihood + priors reproduce the stored total?
clo = np.array([abs(r[p] - (r[p + '_lik'] + r[p + '_pri'])) for r in R for p in POINTS])
tot = np.array([abs(r[p]) for r in R for p in POINTS])
print(f'decomposition closure: max |total - (lik+pri)| = {clo.max():.2e}  '
      f'({clo.max()/tot.mean():.1e} relative)\n')

g = lambda k: np.array([r[k] for r in R], float)
p0, p1 = g('P0_fit'), g('P1_truth')

print('=' * 78)
print('Q1  TRUTH vs FIT, total objective')
print('=' * 78)
d = p1 - p0
rel = 100 * d / np.abs(p0)
print(f'  truth is BETTER (lower loss) in {(d < 0).sum():4d} / {len(d)} batches '
      f'({100*(d<0).mean():.1f}%)')
print(f'  delta = truth - fit :  median {np.median(d):+10.1f}   mean {d.mean():+10.1f}')
print(f'  as % of the fit loss:  median {np.median(rel):+9.2f}%  mean {rel.mean():+9.2f}%')
print('  per seed:')
for s in S:
    m = g('seed') == s
    print(f'    seed {s}: truth better in {(d[m]<0).sum():3d}/{m.sum():3d} batches, '
          f'median {np.median(rel[m]):+7.2f}%')

print()
print('=' * 78)
print('Q2  DOES THE VERDICT SURVIVE THE LIKELIHOOD / PRIOR SPLIT?')
print('=' * 78)
for nm, suf in [('LIKELIHOOD only', '_lik'), ('PRIORS only', '_pri')]:
    a, b = g('P0_fit' + suf), g('P1_truth' + suf)
    dd = b - a
    print(f'  {nm:16s}: truth better in {(dd<0).sum():4d}/{len(dd)} '
          f'({100*(dd<0).mean():5.1f}%)   median delta {np.median(dd):+9.1f}')

print()
print('=' * 78)
print('Q3  WHICH BLOCK DOES THE OBJECTIVE DISAGREE ABOUT?')
print('    (exactly one block moved to truth; the other two stay at the fit)')
print('=' * 78)
print(f'  {"point":22s} {"median delta":>13s} {"median %":>10s} {"better in":>12s}  '
      f'{"lik":>9s} {"pri":>9s}')
for p in POINTS[1:]:
    dd = g(p) - p0
    rr = 100 * dd / np.abs(p0)
    dl = g(p + '_lik') - g('P0_fit_lik')
    dp = g(p + '_pri') - g('P0_fit_pri')
    print(f'  {LABEL[p]:22s} {np.median(dd):+13.1f} {np.median(rr):+9.2f}% '
          f'{(dd<0).sum():6d}/{len(dd):<5d} {np.median(dl):+9.1f} {np.median(dp):+9.1f}')

print()
print('=' * 78)
print('Q4  THE REPRESENTATION FLOOR (how well the spline basis can fit truth AT ALL)')
print('=' * 78)
fl = np.array([r['floor'] for r in R if r['floor'] is not None], float)
print(f'  n = {len(fl)} batches;  median {np.median(fl):.1f} um   mean {fl.mean():.1f} um   '
      f'IQR {np.percentile(fl,25):.1f}-{np.percentile(fl,75):.1f}   max {fl.max():.1f}')
print(f'  production achieves 159 um  ->  headroom factor {159/np.median(fl):.2f}x')

print()
print('=' * 78)
print('Q5  IS THE GAP DATA-DEPENDENT?  (delta vs batch size / hit count)')
print('=' * 78)
mh = g('P0_fit_matched_hits')
print(f'  correlation of relative delta with matched hits : {np.corrcoef(rel, mh)[0,1]:+.3f}')
print(f'  correlation of relative delta with n_tracks     : {np.corrcoef(rel, g("n_tracks"))[0,1]:+.3f}')
print(f'  correlation of relative delta with repr floor   : '
      f'{np.corrcoef(rel[[r["floor"] is not None for r in R]], fl)[0,1]:+.3f}')

print()
print('=' * 78)
print('Q6  MEAN dEdx AT EACH POINT (the mean constraint pins this to 1.887)')
print('=' * 78)
for p in POINTS:
    v = g(p + '_mean_dedx')
    print(f'  {LABEL[p]:22s} median {np.median(v):.4f}')
np.save('/tmp/tp_rows.npy', np.array([0]))  # placeholder; plotting script reloads via load()

print()
print('=' * 78)
print('Q7  WHICH PRIOR TERM CHARGES FOR MOVING dEdx TO TRUTH?')
print('=' * 78)
for term in PRI + LIK:
    a = np.array([r['points_raw'][0] if False else 0 for r in []]) if False else None
# reload with per-term detail
def term_series(point, term):
    out = []
    for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
        for rec in pickle.load(open(f, 'rb'))['truthpoint']:
            if point in rec['points']:
                out.append(float(rec['points'][point]['aux'].get(term, 0.0)))
    return np.array(out)

print(f'  {"term":24s} {"at fit":>12s} {"at dEdx-truth":>14s} {"delta":>12s}')
for term in ['dedx_prior', 'dedx_mean_penalty', 'dedx_barrier', 'mcs_prior',
             'log_likelihood_tick', 'no_match_penalty', 'expected_total_hits']:
    a, b = term_series('P0_fit', term), term_series('P3_dedx_true', term)
    if np.abs(a).max() < 1e-9 and np.abs(b).max() < 1e-9:
        continue
    print(f'  {term:24s} {np.median(a):+12.1f} {np.median(b):+14.1f} {np.median(b-a):+12.1f}')

print()
print('=' * 78)
print('Q8  WHAT-IF: the same evaluation with the dEdx prior at its DEFAULT weight 0.5')
print('=' * 78)
print('  The aux stores the WEIGHTED prior, so scaling it by 0.1 is exactly what')
print('  dedx_prior_weight 5 -> 0.5 does to this term. Everything else is held fixed:')
print('  this re-scores the SAME points, it does not re-fit, so it is an indication of')
print('  which way the objective would tilt, not a prediction of where a new fit lands.')
for w, lab in [(1.0, 'prior weight 5   (production)'), (0.1, 'prior weight 0.5 (default)')]:
    tot0 = g('P0_fit') - (1 - w) * term_series('P0_fit', 'dedx_prior')
    tot1 = g('P1_truth') - (1 - w) * term_series('P1_truth', 'dedx_prior')
    dd = tot1 - tot0
    print(f'  {lab:32s}: truth better in {(dd<0).sum():3d}/{len(dd)} '
          f'({100*(dd<0).mean():5.1f}%)   median delta {np.median(dd):+9.1f} '
          f'({100*np.median(dd/np.abs(tot0)):+6.2f}%)')
