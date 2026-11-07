"""PRODUCTION POLISH estimate (6 Nov, Pierre: "Lets adopt POLISH"; REVIEW.md "POLISH RESULT"). The final calibration is the
conditional minimum m = the end-point iterate (H @ 20000) + the calibration-only E1 Newton step of one held epoch
(nuisances held at the fit), from the dump written by run_polish.sbatch. The step d is in ln(physical) units
(jac = d ln(phys) / d(norm)); it is applied exactly in the fit's own coordinates: m = map(norm + dn), dn = d / jac, with the
production sigmoid map phys = lo + (hi - lo) sigmoid(s norm) (optimize/ranges.py), so s dn = d phys / ((hi - lo) u (1 - u)),
u = (phys - lo) / (hi - lo). (The validated readout used e + d, the same to first order; see REVIEW.md POLISH adoption.)
Uncertainty to quote: the ensemble SD of m over the 30 production end points x sqrt(1 + 1/30) (UQ/polish_exact_check.py) -- valid
for this configuration only; sigma_E1 is a floor (3-96x too small) and is reported only as a diagnostic, with z = d / sigma_E1.
usage (container): python3 polish_estimate.py <dump.npz> <H checkpoint.pkl>   -> JSON on stdout"""
import sys, json, pickle, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import uq_lib as Q

ENSEMBLE_SD_PCT = dict(Ab=0.156, eField=0.043, tran_diff=1.321, long_diff=5.132, lifetime=3.675)   # n = 30, exact form: UQ/polish_exact_check.txt
N_ENS = 30
dump, ck = sys.argv[1:3]
d = Q.load(dump); P = [str(p) for p in d['params']]
assert bool(d['calib_only']), 'not a calibration-only dump'
E = Q.estimators(d)['E1']
h = pickle.load(open(ck, 'rb'))
it = int(h['total_iter']); assert it <= int(d['iteration']) <= it + 101, (int(d['iteration']), it)   # held epoch: Adam held
f1 = lambda x: float(np.mean(x))
it_phys = np.array([f1(h[f'{p}_iter'][-1]) for p in P])
assert np.allclose(d['phys'], it_phys, rtol=1e-5), 'dump is not at the checkpoint iterate'
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.ranges import ranges
lo = np.array([float(ranges[p]['min']) for p in P]); hi = np.array([float(ranges[p]['max']) for p in P])
u = (it_phys - lo) / (hi - lo); assert np.all((u > 0) & (u < 1)), u
assert str(d['scheme']) == 'sigmoid', str(d['scheme'])
s_imp = d['jac'] * it_phys / ((hi - lo) * u * (1 - u))                # the sigmoid scale the dump's jac implies: one value
assert np.allclose(s_imp, s_imp[0], rtol=1e-3), s_imp
sdn = E['d'] * it_phys / ((hi - lo) * u * (1 - u))                   # s * dn
m = lo + (hi - lo) / (1 + np.exp(-(np.log(u / (1 - u)) + sdn)))
end = np.array([np.mean([f1(x) for x in h[f'{p}_iter'][-1000:]]) for p in P])
out = dict(checkpoint=ck, dump=dump, iteration=it,
           polished={p: float(v) for p, v in zip(P, m)},
           uncertainty_pct={p: ENSEMBLE_SD_PCT[p] * np.sqrt(1 + 1 / N_ENS) for p in P},
           uncertainty_note='ensemble SD of m over 30 production end points x sqrt(1 + 1/30); this configuration only',
           end_point_last1000={p: float(v) for p, v in zip(P, end)},
           newton_step_pct={p: float(100 * v) for p, v in zip(P, E['d'])},
           diag_sigma_E1_pct={p: float(100 * v) for p, v in zip(P, E['sig'])}, diag_z={p: float(v) for p, v in zip(P, E['z'])})
if all(f'{p}_target' in h for p in P):          # simulation: errors against the truth, for the record
    t = np.array([f1(h[f'{p}_target'][0]) for p in P])
    out['sim_error_pct'] = dict(polished={p: float(100 * (a / b - 1)) for p, a, b in zip(P, m, t)},
                                end_point={p: float(100 * (a / b - 1)) for p, a, b in zip(P, end, t)})
print(json.dumps(out, indent=1))
