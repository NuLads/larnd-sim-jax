"""DATASZ (3 Nov; Pierre: "how the fit accuracy evolves with the quantity of data ... if the batch size, independently of
the number of tracks has an impact"): the production pipeline (C75 fit -> FR continuation -> H fast continuation, realistic
ends) with the data volume and/or the batch length changed and EVERY schedule element held fixed IN EPOCHS.
r = N_b / 100 (N_b = --max_nbatch; production 100 batches of 400 cm). Changed, and only these:
  stage 1 (template = the Set-A inner KNOT/inner_sci_full_EXP3C75.sh with --data_seed / --test_name as given):
    --max_nbatch N_b, --max_batch_len L, --data_sz D, --iterations 10000 r, lr_kw warmup_steps 500 r, --dedx_freeze_iter
    10200 r, --chain_decay_rate 0.9997^(1/r) (per-iteration decay), --save_freq 200 r, LARND_CHAIN_COARSE_ITER 7500 r,
    --dedx_mean_constraint_weight 2500 L/400 (the per-batch mean constraint kept honest: true batch-mean SD ~ L^-1/2,
    dedx_mean_constraint.md; = 2500 for L = 400);
  FR (= CONT/make_prod_cont.py with every iteration index x r): --iterations 4000 r, warmup 500 r, transition_begin 11500 r,
    --dedx_freeze_iter 20000 r, --chain_decay_start 10000 r;
  H (= CONT/make_prod_fast.py x r): --iterations 6000 r, warmup 500 r, transition_begin 16000 r, --dedx_freeze_iter 100000 r,
    --chain_decay_start 14000 r, --chain_decay_rate 0.9998^(1/r), --save_freq 2000 r.
Epoch-indexed elements (the calibration decay per epoch, transition_steps = epoch size; dE/dx and geometry Adam counts, one
step per visit) are unchanged. With r = 1 and L = 400 each mode reproduces the production command (checked: check_r1.sh).
usage: make_ds.py fit <name> <data_seed> <data_sz> <N_b> <L>     (prints the stage-1 inner)
       make_ds.py cont <checkpoint @ 10000 r> <name>             (FR inner)
       make_ds.py fast <checkpoint @ 14000 r> <name>             (H inner)"""
import sys, pickle, shlex, json, re
TEMPLATE = '/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_EXP3C75.sh'


def fmt(x):
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else repr(x)


def iscale(v, r):
    x = v * r
    assert abs(x - round(x)) < 1e-6, (v, r)
    return str(int(round(x)))


def setarg(argv, k, v):
    i = argv.index(k); argv[i + 1] = v


def emit(env, argv):
    print('cd /sdf/group/neutrino/pgranger/larnd-sim-jax')
    print('export PYTHONPATH=$PWD/src:$PWD')
    for k, v in env:
        print(f'export {k}={shlex.quote(str(v))}')
    print('python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv))


def fit(name, seed, dsz, nb, L):
    r = int(nb) / 100; L = float(L)
    lines = open(TEMPLATE).read().rstrip('\n').split('\n')
    assert lines[0].startswith('cd ') and lines[1].startswith('export PYTHONPATH')
    env = [tuple(l[len('export '):].split('=', 1)) for l in lines[2:-1]]
    assert all(l.startswith('export LARND_') for l in lines[2:-1])
    env = [(k, shlex.split(v)[0] if v not in ("''", '') else '') for k, v in env]
    env = [(k, iscale(7500, r) if k == 'LARND_CHAIN_COARSE_ITER' else v) for k, v in env]
    assert dict(env)['LARND_CHAIN_COARSE_ITER'] == iscale(7500, r)
    argv = shlex.split(lines[-1])[3:]
    assert argv[argv.index('--iterations') + 1] == '10000' and argv[argv.index('--max_nbatch') + 1] == '100'
    assert argv[argv.index('--max_batch_len') + 1] == '400' and argv[argv.index('--chain_decay_rate') + 1] == '0.9997'
    assert argv[argv.index('--dedx_mean_constraint_weight') + 1] == '2500.0'
    kw = json.loads(argv[argv.index('--lr_kw') + 1]); assert kw == {'decay_rate': 0.91, 'init_value': 0, 'warmup_steps': 500}
    kw['warmup_steps'] = int(iscale(500, r))
    SET = {'--max_nbatch': str(nb), '--max_batch_len': fmt(L), '--data_sz': str(dsz), '--data_seed': str(seed),
           '--iterations': iscale(10000, r), '--dedx_freeze_iter': iscale(10200, r), '--save_freq': iscale(200, r),
           '--chain_decay_rate': '0.9997' if r == 1 else repr(0.9997 ** (1 / r)),
           '--dedx_mean_constraint_weight': '2500.0' if L == 400 else repr(2500.0 * L / 400),
           '--lr_kw': '{"decay_rate" : 0.91, "init_value" : 0, "warmup_steps": %d}' % kw['warmup_steps'],
           '--test_name': f'sci_full_{name}'}
    for k, v in SET.items():
        setarg(argv, k, v)
    emit(env, argv)


def stage(ck, name, which):
    d = pickle.load(open(ck, 'rb'))
    argv = list(d['provenance']['argv'][1:])       # = the STAGE-1 argv (a resumed run keeps the original provenance)
    r = int(argv[argv.index('--max_nbatch') + 1]) / 100
    base = 10000 if which == 'cont' else 14000
    assert int(d['total_iter']) == int(iscale(base, r)), (d['total_iter'], base, r)
    assert '--resume_from' not in argv and '--chain_decay_start' not in argv
    if which == 'cont':
        SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/param_list_cont.yaml',
               '--iterations': iscale(4000, r), '--test_name': name, '--lr': '1e-3', '--dedx_freeze_iter': iscale(20000, r),
               '--lr_kw': json.dumps({"decay_rate": 0.805, "init_value": 0, "warmup_steps": int(iscale(500, r)),
                                      "transition_begin": int(iscale(11500, r))}),
               '--chain_lr': '2e-3'}
        start = iscale(10000, r)
    else:
        SET = {'--params': '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/param_list_cont_1e-2.yaml',
               '--iterations': iscale(6000, r), '--test_name': name, '--lr': '1e-2', '--dedx_freeze_iter': iscale(100000, r),
               '--lr_kw': json.dumps({"decay_rate": 0.805, "init_value": 0, "warmup_steps": int(iscale(500, r)),
                                      "transition_begin": int(iscale(16000, r))}),
               '--chain_lr': '2e-3', '--chain_decay_rate': '0.9998' if r == 1 else repr(0.9998 ** (1 / r)),
               '--save_freq': iscale(2000, r)}
        start = iscale(14000, r)
    for k, v in SET.items():
        setarg(argv, k, v)
    i = argv.index('--chain_lr'); argv[i + 2:i + 2] = ['--chain_decay_start', start]
    argv += ['--resume_from', ck]
    env = dict(d['env_overrides']); env['LARND_CHAIN_FREEZE_DRIFT'] = '0'
    emit(sorted(env.items()), argv)


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit(*sys.argv[2:7])
    else:
        stage(sys.argv[2], sys.argv[3], sys.argv[1])
