"""ENDFIT (h2), 26 Oct: the (h) FAST continuation with the chain-geometry LR budget matched to the long run's (single
factor vs (h)). (h) gave lifetime PARTIAL; the geometry of H had moved only ~40 % of the long run's way, and its chain
LR integrated to ~7 against the long run's ~35. Same tool as make_fast_cont.py, changed only here:
  - chain LR 1e-2 (the 1e4 fit's own initial value) held 14000-17000 (--chain_decay_start 17000), then x0.9990626 per
    iteration -> 6.0e-4 at 20000 (the (h) / long end value); integrated ~40.
  - unchanged from (h): calibration peak LR 1e-2 held to 16500 then x0.805 per epoch; dE/dx 1e-2 constant, never
    frozen; LARND_CHAIN_FREEZE_DRIFT=0; 6000 iterations; --save_freq 2000.
usage: python make_fast2_cont.py <history_iter14000.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/param_list_cont_1e-2.yaml', '--iterations': '6000',
       '--test_name': name, '--lr': '1e-2', '--dedx_freeze_iter': '100000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 16000}',
       '--chain_lr': '1e-2', '--chain_decay_rate': '0.9990626', '--save_freq': '2000'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', '17000']
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
