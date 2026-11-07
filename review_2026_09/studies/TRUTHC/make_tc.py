"""TRUTHC (4 Nov; REVIEW.md "TRUTHC"): command for a truth-point evaluation (LARND_TRUTHPOINT, no fitting step) of a fit end
point. Base = the inner of the STAGE that wrote the checkpoint (KNOT/inner_sci_full_<RUN>.sh: its argv and LARND_* env, so
the objective is the fit's own). Changed only by: cd + PYTHONPATH -> the review copy TRUTHC/tree (= the live tree + the
gradient/jac storage in the truth-point Hessian block, 26 added lines); --resume_from removed (a fresh run, one epoch);
--iterations 100; --test_name; --save_freq 1000000; exports LARND_TRUTHPOINT=<checkpoint>,
LARND_TP_HESS=P0_fit:P1_truth:P2_calib_true:P3_dedx_true:P4_geom_true, LARND_KEEP_TARGETS='' (targets deleted at the end).
usage: python make_tc.py <stage_inner.sh> <checkpoint.pkl> <new_test_name>"""
import sys, re, shlex
inner, ck, name = sys.argv[1:4]
TREE = '/sdf/scratch/neutrino/pgranger/review_2026_09/TRUTHC/tree'
lines = open(inner).read().rstrip('\n').split('\n')
assert lines[0] == 'cd /sdf/group/neutrino/pgranger/larnd-sim-jax' and lines[1] == 'export PYTHONPATH=$PWD/src:$PWD', lines[:2]
lines[0] = f'cd {TREE}'
SETENV = {'LARND_TRUTHPOINT': ck, 'LARND_TP_HESS': 'P0_fit:P1_truth:P2_calib_true:P3_dedx_true:P4_geom_true',
          'LARND_KEEP_TARGETS': ''}
for k, v in SETENV.items():
    hit = [j for j, l in enumerate(lines) if re.match(rf'export {k}=', l)]
    assert len(hit) == 1, (k, hit)
    lines[hit[0]] = f'export {k}={shlex.quote(v)}'
assert any(l == "export LARND_TP_HESS_STEPS=1e-3:3e-3" for l in lines)
py = [k for k, l in enumerate(lines) if l.startswith('python3 -m optimize.example_run ')]
assert len(py) == 1
argv = shlex.split(lines[py[0]])[3:]
i = argv.index('--resume_from'); del argv[i:i + 2]
for k, v in {'--iterations': '100', '--test_name': name, '--save_freq': '1000000'}.items():
    j = argv.index(k); argv[j + 1] = v
lines[py[0]] = 'python3 -m optimize.example_run ' + ' '.join(shlex.quote(a) for a in argv)
print('\n'.join(lines))
