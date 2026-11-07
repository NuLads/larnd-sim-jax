"""Pinning test at 10 cm (19 Oct): the production 10 cm fit of sample s (sci_full_KN10S<s>) re-run from scratch with the drift
coordinate of straight-looking tracks NOT held (no LARND_CHAIN_FREEZE_DRIFT, no LARND_FREEZE_STRAIGHT_CL): the 10 cm version of
the three-mode 'hit seed (free)' arm HITSEED of Fig. recipe, whose provenance differs from STRFRZ50 only by these two env vars.
Otherwise the run's OWN launch argv and LARND_* env (checkpoint provenance). usage: python make_knot_hit_cmd.py <s> <test_name>"""
import sys, glob, pickle, shlex
s, name = sys.argv[1], sys.argv[2]
ck = sorted(glob.glob(f'/sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_KN10S{s}/history_iter10000_*.pkl'))[0]
d = pickle.load(open(ck, 'rb')); argv = list(d['provenance']['argv'][1:])
assert argv[argv.index('--chain_spline_knot_cm') + 1] == '10'
assert '--resume_from' not in argv and argv[argv.index('--iterations') + 1] == '10000'
argv[argv.index('--test_name') + 1] = name
DROP = ('LARND_CHAIN_FREEZE_DRIFT', 'LARND_FREEZE_STRAIGHT_CL')
env = d['env_overrides']; assert env['LARND_CHAIN_FREEZE_DRIFT'] == '1' and env['LARND_FREEZE_STRAIGHT_CL'] == '0.50'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
print('unset ' + ' '.join(DROP))
for k, v in sorted(env.items()):
    if k not in DROP: print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
