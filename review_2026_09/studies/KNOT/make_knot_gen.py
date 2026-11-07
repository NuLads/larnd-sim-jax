"""Generic knot rerun (18 Oct): the FRESH run <src_run> (own provenance) re-run from scratch with --chain_spline_knot_cm <knot>
and otherwise its own launch argv and LARND_* env. usage: python make_knot_gen.py <src_run> <test_name> <knot_cm>"""
import sys, glob, pickle, shlex
src, name, knot = sys.argv[1], sys.argv[2], sys.argv[3]
ck = sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_{src}/history_iter10000_*.pkl'))[0]
d = pickle.load(open(ck, 'rb')); argv = list(d['provenance']['argv'][1:])
assert '--resume_from' not in argv and argv[argv.index('--iterations') + 1] == '10000'
for k, v in (('--chain_spline_knot_cm', knot), ('--test_name', name)):
    i = argv.index(k); assert k != '--chain_spline_knot_cm' or argv[i + 1] == '40', argv[i + 1]; argv[i + 1] = v
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(d['env_overrides'].items()): print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
