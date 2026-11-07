"""PRODUCTION continuation (adopted 24 Oct, Pierre's OK; REVIEW.md "FB block = CONFIRMED"): resume a finished production
fit at 10000 iterations for 4000 more with the straight-track freeze RELEASED. This is exactly the validated PRF<s> recipe:
  - the CONT protocol of make_cmd.py (10 Oct): calibration peak LR 1e-3 via CONT/param_list_cont.yaml, lr_kw
    transition_begin 11500 / decay 0.805, --dedx_freeze_iter 20000 (dE/dx never frozen), --iterations 4000 (additional);
  - chain LR re-heat: --chain_lr 2e-3 --chain_decay_start 10000 (2e-3 -> 6e-4 over 10000-14000);
  - LARND_CHAIN_FREEZE_DRIFT=0 (the tracks called straight are no longer frozen in the drift direction).
Command = the run's OWN launch argv and LARND_* env (checkpoint provenance), changed only there.
Checked: on KN10S2 @ 1e4 with name sci_full_PRF2 it reproduces KNOT/inner_sci_full_PRF2.sh.
usage: python make_prod_cont.py <checkpoint_iter10000.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 10000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--iterations': '4000',
       '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': '20000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 11500}',
       '--chain_lr': '2e-3'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', '10000']
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
