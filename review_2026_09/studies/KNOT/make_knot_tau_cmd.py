"""Lifetime-regime reruns at a 10 cm knot (15 Oct): the run STRFRZ50T<tau>D<s> (alternative true lifetime, 1e4
iterations) re-run from scratch with --chain_spline_knot_cm 10 and otherwise its OWN launch argv and LARND_* env
(checkpoint provenance; fresh runs, so the provenance is the run's own). usage: python make_knot_tau_cmd.py <src_run> <test_name>"""
import sys, glob, pickle, shlex
src, name = sys.argv[1], sys.argv[2]
ck = sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_{src}/history_iter10000_*.pkl'))[0]
d = pickle.load(open(ck, 'rb')); argv = list(d['provenance']['argv'][1:])
for k, v in (('--chain_spline_knot_cm', '10'), ('--test_name', name)):
    i = argv.index(k); assert k != '--chain_spline_knot_cm' or argv[i + 1] == '40', argv[i + 1]; argv[i + 1] = v
assert '--resume_from' not in argv and argv[argv.index('--iterations') + 1] == '10000'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(d['env_overrides'].items()): print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
