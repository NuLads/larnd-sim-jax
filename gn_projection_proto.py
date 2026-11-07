"""Prototype: would projecting the GN step onto the STIFF subspace have prevented the runaway?

Replays the 60 accepted steps of job 33123555. At each step we know where GN actually went.
We decompose each step increment into (soft plane, stiff subspace) using the measured
eigenbasis, and ask what the trajectory would have looked like if the soft component had
been suppressed.
"""
import numpy as np, json, re

P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
H = json.load(open('fit_result/hessian_analysis/hessian_results.json'))


def vec(e):
    d = dict((m.group(1), float(m.group(2)))
             for m in re.finditer(r'([A-Za-z_]+):([+-]?[\d.]+)', e['composition']))
    v = np.array([d.get(n, 0.0) for n in P])
    return v / np.linalg.norm(v)


V = np.stack([vec(e) for e in H['calib5x5']['eigen']]).T     # columns = eigenvectors
lam = np.array([e['value'] for e in H['calib5x5']['eigen']])
SOFT = [0, 1]                                                 # the two flat modes
STIFF = [2, 3, 4]

steps = []
for line in open('logs/gn_proto/job-33123555.out'):
    m = re.match(r'\[GN\s+(\d+)\] loss ([\d.]+) \| Ab([+-][\d.]+)% eField([+-][\d.]+)% '
                 r'tran_diff([+-][\d.]+)% long_diff([+-][\d.]+)% lifetime([+-][\d.]+)%', line)
    if m:
        steps.append([float(m.group(i)) for i in range(3, 8)])
X = np.array(steps)                       # (nstep, 5) parameter error in % points
D = np.diff(X, axis=0)                    # per-step increments

# decompose each increment in the eigenbasis
C = D @ V                                 # coefficients along each eigenvector
soft_frac = np.linalg.norm(C[:, SOFT], axis=1) / np.linalg.norm(C, axis=1)
print(f'{len(D)} GN increments from job 33123555')
print(f'  soft-plane fraction of each step: median {np.median(soft_frac):.3f}  '
      f'min {soft_frac.min():.3f}  max {soft_frac.max():.3f}')
print()

# counterfactual: zero out the soft components, keep the stiff ones
C_proj = C.copy(); C_proj[:, SOFT] = 0.0
D_proj = C_proj @ V.T
X_proj = np.vstack([X[0], X[0] + np.cumsum(D_proj, axis=0)])

print('                          ACTUAL GN                 STIFF-PROJECTED GN')
print(f"{'param':11s}{'start':>8s}{'end':>9s}{'|drift|':>9s}   {'end':>9s}{'|drift|':>9s}")
for j, p in enumerate(P):
    print(f'{p:11s}{X[0,j]:+8.1f}{X[-1,j]:+9.1f}{abs(X[-1,j]-X[0,j]):9.1f}   '
          f'{X_proj[-1,j]:+9.1f}{abs(X_proj[-1,j]-X[0,j]):9.1f}')
print()
print(f'  total |displacement|  actual {np.linalg.norm(X[-1]-X[0]):8.1f} pts'
      f'   projected {np.linalg.norm(X_proj[-1]-X[0]):8.1f} pts')
print(f'  lifetime            : {X[-1,4]:+.1f}%  ->  {X_proj[-1,4]:+.1f}%')
print(f'  long_diff           : {X[-1,3]:+.1f}%  ->  {X_proj[-1,3]:+.1f}%')
print()
print('Caveat: this replays the ACTUAL step directions with their soft component removed.')
print('A real projected run would recompute H,g at each new point, so the trajectory would')
print('differ. This bounds the mechanism, it does not simulate the alternative optimizer.')
