"""CPU-only GN diagnostics. No forward simulation required."""
import numpy as np, json, re

print('='*78)
print('TEST 1 — does float32 assembly actually destroy the Hessian inverse?')
print('  (synthetic matrix with the MEASURED eigenvalue spectrum; no reconstruction error)')
print('='*78)
ev = np.array([3.287e3, 6.134e3, 3.880e5, 7.953e6, 1.285e10])
rng = np.random.default_rng(0)
errs32, errs64 = [], []
for trial in range(200):
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    H64 = (Q * ev) @ Q.T                       # exact, float64
    s64 = np.sqrt(np.diag(np.linalg.inv(H64)))
    H32 = H64.astype(np.float32).astype(np.float64)   # round-trip through float32 storage
    try:
        s32 = np.sqrt(np.abs(np.diag(np.linalg.inv(H32))))
    except np.linalg.LinAlgError:
        continue
    errs32.append(np.abs(s32 / s64 - 1))
e = np.array(errs32)
print(f'  condition number            : {ev.max()/ev.min():.3e}')
print(f'  float32 eps                 : {np.finfo(np.float32).eps:.3e}')
print(f'  predicted relative error ~ cond*eps = {ev.max()/ev.min()*np.finfo(np.float32).eps:.3f}')
print(f'  MEASURED |sigma32/sigma64 - 1| over {len(e)} random rotations:')
print(f'    median {np.median(e)*100:7.2f}%   p90 {np.percentile(e,90)*100:7.2f}%   max {e.max()*100:7.2f}%')
print(f'    per-mode median: ' + '  '.join(f'{v*100:.2f}%' for v in np.median(e,axis=0)))

print()
print('='*78)
print('TEST 2 — did the 60-step runaway ride the soft eigendirection?')
print('='*78)
steps = []
for line in open('logs/gn_proto/job-33123555.out'):
    m = re.match(r'\[GN\s+(\d+)\] loss ([\d.]+) \| Ab([+-][\d.]+)% eField([+-][\d.]+)% '
                 r'tran_diff([+-][\d.]+)% long_diff([+-][\d.]+)% lifetime([+-][\d.]+)%', line)
    if m:
        steps.append([float(m.group(i)) for i in range(2, 8)])
S = np.array(steps)
print(f'  parsed {len(S)} accepted steps; loss {S[0,0]:.1f} -> {S[-1,0]:.1f} (monotone: {np.all(np.diff(S[:,0])<0)})')
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
disp = S[-1, 1:] - S[0, 1:]
u = disp / np.linalg.norm(disp)
print(f'  net displacement (% pts): ' + '  '.join(f'{p} {d:+.1f}' for p, d in zip(P, disp)))
H = json.load(open('fit_result/hessian_analysis/hessian_results.json'))
def vec(e, names):
    d = dict((m.group(1), float(m.group(2)))
             for m in re.finditer(r'([A-Za-z_]+):([+-]?[\d.]+)', e['composition']))
    v = np.array([d.get(n, 0.0) for n in names]); return v/np.linalg.norm(v)
for k, e in enumerate(H['calib5x5']['eigen']):
    v = vec(e, P)
    print(f"    |cos| with mode {k+1} (lambda={e['value']:.3e}): {abs(float(u@v)):.3f}")
soft = np.stack([vec(H['calib5x5']['eigen'][0], P), vec(H['calib5x5']['eigen'][1], P)]).T
Psoft = soft @ np.linalg.pinv(soft)
print(f'  fraction of the walk inside the 2-D SOFT plane: {np.linalg.norm(Psoft@u):.3f}')
