"""NOCLIP (4 Nov; REVIEW.md "NOCLIP"): the production H stage (CONT/make_prod_fast.py, unchanged) of <NAME>FR @ 14000, with
the calibration gradient clip off (--max_clip_norm_val 100 -> 1e12) and LARND_CALIB_ADAM_RESET=1 (fresh calibration Adam
moments, schedule count kept: the restored nu holds CLIPPED squared gradients). Nothing else changes.
usage: python make_noclip.py <checkpoint_iter14000.pkl> <new_test_name>"""
import sys, subprocess
out = subprocess.run([sys.executable, '/sdf/scratch/neutrino/pgranger/review_2026_09/CONT/make_prod_fast.py'] + sys.argv[1:3],
                     stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.rstrip('\n').split('\n')
py = [k for k, l in enumerate(out) if l.startswith('python3 -m optimize.example_run ')]
assert len(py) == 1 and out[py[0]].count(' --max_clip_norm_val 100 ') == 1, 'expected exactly one --max_clip_norm_val 100'
assert not any('LARND_CALIB_ADAM_RESET' in l for l in out)
out[py[0]] = out[py[0]].replace(' --max_clip_norm_val 100 ', ' --max_clip_norm_val 1e12 ')
out.insert(py[0], 'export LARND_CALIB_ADAM_RESET=1')
print('\n'.join(out))
