"""UQ (31 Oct): command for a profile-Hessian DIAGNOSTIC of a fit end point. Base = the inner script of the STAGE that wrote
the checkpoint (KNOT/inner_sci_full_<NAME>{,FR,H}.sh as run_knot wrote it: its own argv and LARND_* env; the checkpoint
provenance keeps only the stage-1 command). Changed only by: --iterations 101 (one held assembly epoch + 1 iteration),
--test_name, --save_freq 1000000, --resume_from <checkpoint> (replaced or added), and the exports LARND_JOINT_NEWTON=<checkpoint
iteration>, LARND_JOINT_NEWTON_DUMP=<dump.npz> (fit_params._jn_dump), plus any extra LARND_X=value (3 Nov: an extra that
the stage script already exports replaces that export, e.g. LARND_CHAIN_FREEZE_DRIFT=0 for a stage-1 checkpoint, as the
FR stage that resumes from it; the joint Newton refuses the drift freeze, which only masks geometry updates).
Adam is held during the assembly epoch, so calibration and nuisances stay those of the checkpoint.
usage: python make_jn_dump.py <stage_inner.sh> <checkpoint.pkl> <new_test_name> <dump.npz> [LARND_X=value ...]"""
import sys, pickle, re, shlex
inner, ck, name, dump = sys.argv[1:5]
extra = dict(a.split('=', 1) for a in sys.argv[5:])
it = int(pickle.load(open(ck, 'rb'))['total_iter'])
lines = open(inner).read().rstrip('\n').split('\n')
rep = [k for k in extra if any(re.match(rf'export {k}=', l) for l in lines)]
assert not any(k.startswith('LARND_JOINT_NEWTON') for k in rep), rep
lines = [l for l in lines if not any(re.match(rf'export {k}=', l) for k in rep)]
py = [k for k, l in enumerate(lines) if l.startswith('python3 -m optimize.example_run ')]
assert len(py) == 1, inner
argv = shlex.split(lines[py[0]])[3:]
SET = {'--iterations': '101', '--test_name': name, '--save_freq': '1000000'}
for k, v in SET.items():
    i = argv.index(k); argv[i + 1] = v
if '--resume_from' in argv:
    argv[argv.index('--resume_from') + 1] = ck
else:
    argv += ['--resume_from', ck]
env = {'LARND_JOINT_NEWTON': str(it), 'LARND_JOINT_NEWTON_DUMP': dump, **extra}
assert not any(re.match(rf'export {k}=', l) for l in lines for k in env), 'inner already sets a dump variable'
assert not any('LARND_JOINT_NEWTON' in l for l in lines), 'inner already uses the joint Newton'
out = lines[:py[0]] + [f'export {k}={shlex.quote(v)}' for k, v in env.items()] + \
      ['python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv)] + lines[py[0] + 1:]
print('\n'.join(out))
