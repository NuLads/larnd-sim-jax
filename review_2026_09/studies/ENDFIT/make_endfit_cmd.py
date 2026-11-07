"""ENDFIT (12 Oct 2026): track endpoints not taken from truth but estimated and then FITTED (LARND_FIT_ENDPOINTS=1,
see optimize/fit_params.py). Prints the bash lines of a fresh 1e4-iteration fit with the launch argv and LARND_* env of
the production STRFRZ50<s> run (checkpoint provenance), changed only by --test_name and the env overrides given.
usage: python make_endfit_cmd.py <sample> <new_test_name> KEY=VAL [KEY=VAL ...] [--argv-set FLAG=VAL ...]"""
import sys, pickle, shlex, glob
s, name = sys.argv[1], sys.argv[2]
ck = sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_STRFRZ50{s}/history_iter10000_*.pkl'))[0]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
env = dict(d['env_overrides']); setargs = {}
mode = 'env'
for a in sys.argv[3:]:
    if a == '--argv-set': mode = 'argv'; continue
    k, v = a.split('=', 1)
    (env if mode == 'env' else setargs)[k] = v
setargs['--test_name'] = name
for k, v in setargs.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
