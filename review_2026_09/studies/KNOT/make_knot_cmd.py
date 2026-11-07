"""Basis fit test (12 Oct): the production fit of sample s (sci_full_STRFRZ50<s>) re-run from scratch with K scaled with track
length -- --chain_spline_knot_cm 10 (K = clip(round(L/10 cm), 3, 12), ~10 coefficients per track instead of 6) -- and
otherwise the run's OWN launch argv and LARND_* env (checkpoint provenance). usage: python make_knot_cmd.py <s> <test_name>"""
import sys, glob, pickle, shlex
s, name = sys.argv[1], sys.argv[2]
ck = sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_STRFRZ50{s}/history_iter10000_*.pkl'))[0]
d = pickle.load(open(ck, 'rb')); argv = list(d['provenance']['argv'][1:])
for k, v in (('--chain_spline_knot_cm', '10'), ('--test_name', name)):
    i = argv.index(k); assert k != '--chain_spline_knot_cm' or argv[i + 1] == '40', argv[i + 1]; argv[i + 1] = v
assert '--resume_from' not in argv and argv[argv.index('--iterations') + 1] == '10000'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(d['env_overrides'].items()): print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
