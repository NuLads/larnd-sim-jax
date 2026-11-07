"""PRODUCTION fast continuation (adopted 28 Oct, Pierre: "Let's make this part of standard production now"; REVIEW.md
"(j) RESULT"): resume the freeze-release end point <NAME>FR at 14000 iterations for 6000 more, i.e. the validated (h)
recipe (frozen copy of ENDFIT/make_opt_cont.py without options):
  - calibration peak LR 1e-2 via ENDFIT/param_list_cont_1e-2.yaml, lr_kw transition_begin 16000 / decay 0.805 per epoch
    (held to 16500), --dedx_freeze_iter 100000 (dE/dx never frozen), --iterations 6000 (additional);
  - chain LR re-heat: --chain_lr 2e-3 --chain_decay_start 14000 --chain_decay_rate 0.9998;
  - LARND_CHAIN_FREEZE_DRIFT=0 (freeze stays released).
With realistic (independent per-track 1 mm) endpoint errors it reaches the 44000-iteration state to within the true-end
runs' own fast-vs-long distance. Command = the run's OWN launch argv and LARND_* env (checkpoint provenance), changed
only there. Checked: on KN10N1FR @ 14000 with name sci_full_KN10N1H it reproduces KNOT/inner_sci_full_KN10N1H.sh.
usage: python make_prod_fast.py <checkpoint_iter14000.pkl> <new_test_name>"""
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
