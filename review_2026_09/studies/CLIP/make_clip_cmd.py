"""Clipping test (12 Oct 2026): does the global-norm-100 clip of the calibration gradient move the end point?
The clip is active on ~99% of production iterations (pre-clip norm ~9 400, dominated by eField), so each batch enters
the calibration update with weight 1/|g_b|. At the end of CONTR1-16 the clipped gradient averages to zero but the raw one
does not (eField pooled z -3.8, 16/16; long_diff +2.6, 14/16).
Two arms resumed from the SAME continued fit (CONTR<s>, iteration 14000), 2000 additional iterations, identical except
--max_clip_norm_val: CLIPON (100, as production) vs CLIPOFF (1e12; the clip stays in the optax chain so the restored
state structure matches). Both arms: LARND_CALIB_ADAM_RESET=1 (fresh calibration Adam moments; the restored nu holds
clipped squared gradients, ~100x below raw, which would make the first raw steps several x the LR), calibration LR 1e-3
(CONT/param_list_cont.yaml) held to iteration 15000 then x0.63 per epoch to 1.6e-5 at 16000 (production end 1.3e-5).
Nuisance schedules as in the continuation (dE/dx constant 1e-2, never frozen; trajectories x0.9997/iteration).
The command is the run's own launch argv (checkpoint provenance) and LARND_* env, changed only where listed.
usage: python make_clip_cmd.py <checkpoint.pkl> <new_test_name> <clip_norm>"""
import sys, pickle, shlex
ck, name, clip = sys.argv[1], sys.argv[2], sys.argv[3]
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--iterations': '2000',
       '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': '20000', '--max_clip_norm_val': clip,
       '--lr_kw': '{"decay_rate": 0.63, "init_value": 0, "warmup_steps": 500, "transition_begin": 14500}'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CALIB_ADAM_RESET'] = '1'
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
