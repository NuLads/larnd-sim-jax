"""Read hessian.json (paper_hessian.py) and derive, for one production-sized sample (L_prod of track):
  - the statistical floor: sigma_j from the inverse of the summed Hessian, all five parameters profiled;
  - the conditional width (others fixed) and the lifetime width with only A_b profiled (cross-check against the
    independent Laplace harness: 0.75% on ~380 m, review section 3.3b);
  - the sandwich width H^-1 V H^-1, V from the scatter of the per-batch scores around their length-proportional mean;
  - the model-at-truth bias: delta = -H^-1 g_mean, the shift of the likelihood optimum from the true values that the
    mean score at truth implies (relative units). All quantities in log-parameter units, i.e. relative errors."""
import json, sys, numpy as np
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'hessian.json'))
P = d['params']; Lp = d['L_prod_cm']
ok = [b for b in d['batches'] if 'H' in b]; bad = [b for b in d['batches'] if 'H' not in b]
L = np.array([b['length_cm'] for b in ok]); Ltot = L.sum()
H = sum(np.array(b['H']) for b in ok); H = 0.5 * (H + H.T)
G = np.array([b['g'] for b in ok])
s = Lp / Ltot                                               # scale everything to the production track length
Hp = H * s; C = np.linalg.inv(Hp)
mu = G.sum(0) / Ltot                                        # mean score per cm
R = G - np.outer(L, mu); n = len(ok)
V = (R.T @ R) * n / (n - 1) * s                             # variance of the total score at production length
Csw = C @ V @ C
delta = -C @ (mu * Lp)                                      # optimum shift implied by the mean score at truth
print(f'{n} batches used ({len(bad)} failed: {[b["batch"] for b in bad]}), {Ltot/100:.1f} m; scaled to production {Lp/100:.1f} m')
ev = np.linalg.eigvalsh(Hp); print('eigenvalues of H (ln units, production length):', ' '.join(f'{x:.3g}' for x in ev), f' condition {ev[-1]/ev[0]:.2g}')
print(f'{"param":10s} {"floor%":>8s} {"cond%":>8s} {"sandw%":>8s} {"bias%":>8s} {"bias/floor":>10s}')
for j, k in enumerate(P):
    print(f'{k:10s} {100*np.sqrt(C[j,j]):8.4f} {100/np.sqrt(Hp[j,j]):8.4f} {100*np.sqrt(Csw[j,j]):8.4f} {100*delta[j]:+8.4f} {delta[j]/np.sqrt(C[j,j]):+10.2f}')
iA, iT = P.index('Ab'), P.index('lifetime'); sub = np.linalg.inv(Hp[np.ix_([iA, iT], [iA, iT])])
print(f'lifetime with only A_b profiled: {100*np.sqrt(sub[1,1]):.3f}%  (Laplace harness: 0.75% on ~380 m -> {0.75*np.sqrt(380/(Lp/100)):.3f}% at {Lp/100:.0f} m)')
corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
print('correlation (floor):'); print('            ' + ' '.join(f'{k[:8]:>8s}' for k in P))
for j, k in enumerate(P): print(f'{k:10s}  ' + ' '.join(f'{x:+8.2f}' for x in corr[j]))
# stability of the floor across halves of the sample (information per metre fluctuates from batch to batch)
for half, sel in (('even', slice(0, None, 2)), ('odd', slice(1, None, 2))):
    Hh = sum(np.array(b['H']) for b in ok[sel]); Lh = sum(b['length_cm'] for b in ok[sel]); Ch = np.linalg.inv(0.5 * (Hh + Hh.T) * Lp / Lh)
    print(f'{half:4s} half ({Lh/100:.0f} m): floor% ' + ' '.join(f'{k}={100*np.sqrt(Ch[j,j]):.4f}' for j, k in enumerate(P)))
json.dump(dict(params=P, L_prod_m=Lp / 100, L_used_m=Ltot / 100, floor=np.sqrt(np.diag(C)).tolist(), cond=(1 / np.sqrt(np.diag(Hp))).tolist(),
               sandwich=np.sqrt(np.diag(Csw)).tolist(), bias=delta.tolist(), corr=corr.tolist(), lifetime_Ab_only=float(np.sqrt(sub[1, 1]))),
          open('hessian_summary.json', 'w'), indent=1)
