"""Continuation test (10 Oct): print the bash lines that resume a finished production fit for 4000 more iterations
with a re-heated, then re-annealed schedule. The command is the run's OWN launch argv (checkpoint provenance) and
LARND_* env (env_overrides), changed only where the test needs it:
  --resume_from <checkpoint>, --iterations 4000 (additional: fit() sets target_iter = start_iter + iterations), --test_name <new>
  calibration: per-parameter peak LR 1e-3 via --params CONT/param_list_cont.yaml (the calibration LR is taken from the
               param YAML, 0.1 in production; --lr is IGNORED when the YAML gives values -- the first submission set
               only --lr and re-heated the calibration to the full peak 0.1: D_L -1 -> -35% in 25 iterations, aborted)
               + lr_kw transition_begin 11500, decay 0.805 -> 1e-3 constant over 10000-12000 (the production LR near
               iteration 5200), then x0.805 per epoch to 1.6e-5 at 14000 (production end: 1.3e-5)
  trajectories: production schedule unchanged (x0.9997/iteration, 5e-4 -> 2e-4); a restored geometry gets a FRESH Adam
               state, so re-heating this block kicks every coefficient by a full LR step (not done)
  dE/dx: unchanged (constant 1e-2); --dedx_freeze_iter 20000 keeps it never frozen, as in production (10200 > 10000)
usage: python make_cmd.py <checkpoint.pkl> <new_test_name>"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 10000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--iterations': '4000',   # on resume --iterations counts ADDITIONAL iterations (target = start + iterations)
        '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': '20000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 11500}'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
argv += ['--resume_from', ck]
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(d['env_overrides'].items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
