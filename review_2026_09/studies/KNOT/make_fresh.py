"""22 Oct: print the bash lines that re-run a production fit FROM SCRATCH with its own launch argv and LARND_* env
(checkpoint provenance), changing only --test_name and the env overrides given as KEY=VALUE.
usage: python make_fresh.py <checkpoint.pkl> <new_test_name> [KEY=VALUE ...]"""
import sys, pickle, shlex
ck, name, sets = sys.argv[1], sys.argv[2], dict(kv.split('=', 1) for kv in sys.argv[3:])
d = pickle.load(open(ck, 'rb')); argv = list(d['provenance']['argv'][1:]); env = dict(d['env_overrides'])
assert '--resume_from' not in argv; i = argv.index('--test_name'); argv[i + 1] = name
for k, v in sets.items():
    assert k in env, k; env[k] = v
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()): print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
