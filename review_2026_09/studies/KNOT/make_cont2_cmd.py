"""Second continuation (18 Oct): resume a CONTINUED fit (total_iter 14000, e.g. KN10C<s>) for 4000 more iterations with
the CONT protocol shifted by 4000: calibration LR 1e-3 (CONT/param_list_cont.yaml) from 14000 to 16000, then x0.805 per
epoch to ~1.3e-5 at 18000; nuisance schedules unchanged (dE/dx never frozen). The resumed checkpoint's provenance is the
ORIGINAL fresh run's (resume copies the history), so the CONT changes are re-applied here.
usage: python make_cont2_cmd.py <checkpoint_iter14000.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
assert argv[argv.index('--chain_spline_knot_cm') + 1] == '10'
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--iterations': '4000',
       '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': '30000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 15500}'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv
argv += ['--resume_from', ck]
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax'); print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(d['env_overrides'].items()): print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
