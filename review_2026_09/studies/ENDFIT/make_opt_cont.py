"""27 Oct, follow-up of the "why 44000" thread (Pierre: "try options 1 and 2"): the (h) FAST continuation
(make_fast_cont.py: <NAME>FR @ 14000 -> 20000, calibration peak LR 1e-2 held to 16500 then x0.805 per epoch, chain LR
re-heat 2e-3 decaying 0.9998 per iteration) with extra environment flags and nothing else changed:
  option 2 = LARND_CALIB_SAGA=1 (SAGA calibration gradient), option 1 = LARND_GEOM_SWEEP=<iterations> (per-track Newton
  sweeps of the geometry); see optimize/fit_params.py.
With --long: the LONG continuation's schedule instead (make_long_cont.py: calibration peak LR 1e-3 held, decay only from
42000; chain LR re-heat 2e-3 decaying 0.99996 per iteration), for --iters iterations (27 Oct SAGA arm SL: SAGA works
only where a batch's gradient changes little between its visits, i.e. at 1e-3, not 1e-2).
With --hold N (fast schedule only): the calibration LR 1e-2 is held to iteration N instead of 16500, then x0.805 per epoch
(27 Oct arm HH: is the fast run's lifetime lag the early anneal?).
usage: python make_opt_cont.py <history_iter14000.pkl> <new_test_name> [KEY=VALUE ...] [--iters N] [--save N] [--long] [--hold N]"""
import sys, pickle, shlex
ck, name = sys.argv[1], sys.argv[2]
extra, iters, save, long_, hold, a = {}, '6000', '2000', False, None, sys.argv[3:]
while a:
    x = a.pop(0)
    if x == '--iters': iters = a.pop(0)
    elif x == '--save': save = a.pop(0)
    elif x == '--long': long_ = True
    elif x == '--hold': hold = int(a.pop(0))
    else:
        k, v = x.split('=', 1); assert k.startswith('LARND_'), k; extra[k] = v
d = pickle.load(open(ck, 'rb'))
argv = list(d['provenance']['argv'][1:])
assert int(d['total_iter']) == 14000, d['total_iter']
SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/param_list_cont_1e-2.yaml', '--iterations': iters,
       '--test_name': name, '--lr': '1e-2', '--dedx_freeze_iter': '100000',
       '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 16000}',
       '--chain_lr': '2e-3', '--chain_decay_rate': '0.9998', '--save_freq': save}
if hold is not None:
    assert not long_
    SET['--lr_kw'] = SET['--lr_kw'].replace('"transition_begin": 16000', f'"transition_begin": {hold - 500}')
if long_:
    SET.update({'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml', '--lr': '1e-3',
                '--lr_kw': '{"decay_rate": 0.805, "init_value": 0, "warmup_steps": 500, "transition_begin": 41500}',
                '--chain_decay_rate': '0.99996'})
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
assert '--resume_from' not in argv and '--chain_decay_start' not in argv
i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', '14000']
argv += ['--resume_from', ck]
env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
env.update(extra)
print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
print('export PYTHONPATH=$PWD/src:$PWD')
for k, v in sorted(env.items()):
    print(f'export {k}={shlex.quote(str(v))}')
print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))
