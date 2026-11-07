"""28 Oct, "faster convergence" idea 1 (one schedule instead of three stages; REVIEW.md "ENDFIT (m)"): the production
fast continuation (CONT/make_prod_fast.py = the (h) recipe) generalised to any start: resume <checkpoint> (total_iter S)
for --iters N iterations with the freeze released, calibration peak LR 1e-2 held for --hold H iterations (to S + H), then
x0.805 per epoch; chain LR re-heat 2e-3 decaying 0.9998 per iteration from S; dE/dx never frozen. Extra LARND_*=v pairs
are exported. With S = 14000, N = 6000, H = 2500 it reproduces CONT/make_prod_fast.py byte for byte (checked).
usage: python make_sched.py <checkpoint.pkl> <new_test_name> [--iters N] [--hold H] [KEY=VALUE ...]"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
extra, iters, hold, a = {}, 6000, 2500, sys.argv[3:]
while a:
    x = a.pop(0)
    if x == '--iters': iters = int(a.pop(0))
    elif x == '--hold': hold = int(a.pop(0))
    else:
        k, v = x.split('=', 1); assert k.startswith('LARND_'), k; extra[k] = v
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
S = int(d['total_iter'])
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/param_list_cont_1e-2.yaml', '--iterations': str(iters),
       '--test_name': name, '--lr': '1e-2', '--dedx_freeze_iter': '100000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": %d}' % (S + hold - 500),
       '--chain_lr': '2e-3', '--chain_decay_rate': '0.9998', '--save_freq': '2000'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', str(S)]
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
env.update(extra)
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
