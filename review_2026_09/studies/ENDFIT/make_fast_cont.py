"""ENDFIT (h), 26 Oct (Pierre asked why 44000 iterations are needed; "Do what you think is best"): FAST continuation of a
finished production run from its <NAME>FR @ 14000 checkpoint, 6000 more iterations (-> 20000), calibration LR 10x the
long run's. Diagnosis (REVIEW.md "Why 44000"): the calibration gradient per iteration is ~99 % noise, Adam drifts at
~LR x signal/noise, so the long run's 28000 iterations at 1e-3 should be matched by ~2500 at 1e-2. Same tool as
make_long_cont.py, changed only here:
  - calibration: peak LR 1e-2 (ENDFIT/param_list_cont_1e-2.yaml -- with a YAML dict the per-parameter values ARE the
    peak LRs; --lr is ignored), held to 16500, then x0.805 per epoch (lr_kw transition_begin 16000: the decay starts
    after warmup_steps + transition_begin global iterations) -> ~5e-6 at 20000. Integrated LR ~30 (long run ~28.5);
  - chain LR re-heat 2e-3 from 14000, decay 0.9998 per iteration -> 6e-4 at 20000 (long run: the same start and end
    over 30000 iterations);
  - --dedx_freeze_iter 100000 (dE/dx never frozen); LARND_CHAIN_FREEZE_DRIFT=0; --save_freq 2000.
usage: python make_fast_cont.py <history_iter14000.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/param_list_cont_1e-2.yaml', '--iterations': '6000',
       '--test_name': name, '--lr': '1e-2', '--dedx_freeze_iter': '100000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 16000}',
       '--chain_lr': '2e-3', '--chain_decay_rate': '0.9998', '--save_freq': '2000'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', '14000']
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
