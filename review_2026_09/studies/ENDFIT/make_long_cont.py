"""ENDFIT (e), 25 Oct (Pierre: "Launch (e)"): LONG continuation of a finished production run from its <NAME>FR @ 14000
checkpoint, 30000 more iterations (-> 44000, ~300 more Adam updates per track instead of 40), to see whether endpoints
fitted from a 1 mm smear converge and whether converged ends recover the calibration. Same idea as make_prod_cont.py
(the validated PRF recipe), re-applied over the longer window; command = the run's OWN launch argv and LARND_* env
(checkpoint provenance = the original fit's), changed only here:
  - calibration: peak LR 1e-3 (CONT/param_list_cont.yaml), held, then x0.805 per epoch from 42000 (lr_kw
    transition_begin 41500; the schedule counts GLOBAL iterations, as in the CONT recipe) -> ~1.3e-5 at 44000, the same
    end-of-run annealing as the production continuation (12000 -> 14000);
  - chain LR re-heat 2e-3 from 14000, decay 0.99996 per iteration -> 6e-4 at 44000 (production continuation: 2e-3 ->
    6e-4 over 4000);
  - --dedx_freeze_iter 100000 (dE/dx never frozen, as in production);
  - LARND_CHAIN_FREEZE_DRIFT=0 (freeze released, as in the production continuation);
  - --save_freq 2000 (I/O only).
usage: python make_long_cont.py <history_iter14000.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--iterations': '30000',
       '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': '100000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 41500}',
       '--chain_lr': '2e-3', '--chain_decay_rate': '0.99996', '--save_freq': '2000'}
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
