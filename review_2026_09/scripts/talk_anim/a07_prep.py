"""Per-hit and per-pixel loss of the hero event at the two saved states (start, iter20000), from the production prediction
(a06_sweep.py --rows <state>), reproducing the production PPP terms. Chooses the showcase pixel and writes a small npz."""
import numpy as np, json
from a06_load import *
D = data(); z = D['z']
OUT = f'{T}/data/a07_score.npz'
EPS = 18.420681
St = {s: dict(np.load(f'{SW}/state_{s}.npz')) for s in ('start', 'iter20000')}
hp_ = {}
pid_h, t_h, q_h = D['pid'], D['t'].astype(int), D['q']
up = np.unique(pid_h)
res = {}
for s, S in St.items():
    logI, logT, match = S['logI'].astype(float), S['logT'].astype(float), S['match']
    nh = dict(zip(z[f'{s}_pid'].tolist(), z[f'{s}_nhit'].astype(float).tolist()))
    pids = np.union1d(up, S['pid'])
    Lj = np.zeros(len(pids)); hitpart = np.zeros(len(pids)); exppart = np.zeros(len(pids)); fnpart = np.zeros(len(pids))
    ij = np.searchsorted(pids, pid_h)
    # hits on simulated pixels: -log I (floored); hits on pixels the model did not simulate: -log eps
    np.add.at(hitpart, ij[match], -logI[match]); np.add.at(fnpart, ij[~match], EPS)
    for k, p in enumerate(pids): exppart[k] = nh.get(int(p), 0.0)
    Lj = hitpart + fnpart + exppart
    tot = float(Lj.sum()); prod = float(S['nll'])
    floored = int((logI[match] <= -EPS + 1e-3).sum())
    print(s, 'sum pixels', round(tot, 2), 'production nll', round(prod, 2), 'diff', round(tot - prod, 3), '| hits term', round(hitpart.sum(), 2), 'prod', round(float(S['ll']), 2),
          '| unmatched', int((~match).sum()), round(fnpart.sum(), 2), 'prod', round(float(S['nomatch']), 2), '| expected', round(exppart.sum(), 2), 'prod', round(float(S['expct']), 2), '| floored matched hits', floored)
    res[s] = dict(pids=pids, Lj=Lj, hitpart=hitpart, exppart=exppart, fnpart=fnpart, logI=logI, logT=logT, match=match, nll=prod, floored=floored)
# common pixel table: union over states
allp = np.union1d(res['start']['pids'], res['iter20000']['pids'])
xy = {}
for s, S in St.items():
    for a, b, c in zip(S['pid'].tolist(), S['x'].tolist(), S['y'].tolist()): xy[a] = (b, c)
for a, b, c in zip(pid_h.tolist(), D['px'].tolist(), D['py'].tolist()): xy[a] = (b, c)
X = np.array([xy[int(p)][0] for p in allp]); Y = np.array([xy[int(p)][1] for p in allp])
G = track_geom(D, X, Y)
def onto(s, k):
    a = np.zeros(len(allp)); a[np.searchsorted(allp, res[s]['pids'])] = res[s][k]; return a
L0, L1 = onto('start', 'Lj'), onto('iter20000', 'Lj')
main = G['track'] == 4
nhd = np.bincount(np.searchsorted(allp, pid_h), minlength=len(allp))
print('main pixels', main.sum(), 'with data hits', ((nhd > 0) & main).sum(), 'L main start/fit', L0[main].sum(), L1[main].sum())
print('hit change d logI all', np.percentile(res['iter20000']['logI'] - res['start']['logI'], [10, 25, 50, 75, 90]), 'mean', (res['iter20000']['logI'] - res['start']['logI']).mean())
im = main[np.searchsorted(allp, pid_h)]
dh = res['iter20000']['logI'] - res['start']['logI']
print('main track hits: dlogI median', np.median(dh[im]), 'mean', dh[im].mean(), 'pctl', np.percentile(dh[im], [10, 25, 50, 75, 90]))
# showcase pixel: 2 data hits (the common case), simulated at both states, median per-pixel improvement within main-track pixels of that class
in0 = np.isin(allp, res['start']['pids'][onto('start', 'exppart')[np.searchsorted(allp, res['start']['pids'])] > 0]); 
cand = main & (nhd == 2) & np.isin(allp, St['start']['pid']) & np.isin(allp, St['iter20000']['pid'])
dL = L0 - L1
med = np.median(dL[cand]); order = np.where(cand)[0][np.argsort(np.abs(dL[cand] - med))]
print('candidates', cand.sum(), 'dL pctl', np.percentile(dL[cand], [10, 25, 50, 75, 90]), 'all main', np.percentile(dL[main], [10, 25, 50, 75, 90]))
for k in order[:8]:
    h = np.where(pid_h == allp[k])[0]
    print('pix', allp[k], 'x,y', round(X[k], 2), round(Y[k], 2), 's', round(G['s'][k], 1), 'd', round(G['d'][k], 2), 'z', round(G['z'][k], 1), 'L', round(L0[k], 2), round(L1[k], 2), 'dL', round(dL[k], 2), 'pct', round((dL[cand] < dL[k]).mean() * 100, 1), 'ticks', t_h[h], 'q', np.round(q_h[h], 2),
          'logI', np.round(res['start']['logI'][h], 2), np.round(res['iter20000']['logI'][h], 2))
np.savez(OUT, allp=allp, X=X, Y=Y, s=G['s'], d=G['d'], ztr=G['z'], track=G['track'], L0=L0, L1=L1, nhd=nhd, h0=onto('start', 'hitpart'), h1=onto('iter20000', 'hitpart'), e0=onto('start', 'exppart'), e1=onto('iter20000', 'exppart'),
         f0=onto('start', 'fnpart'), f1=onto('iter20000', 'fnpart'), logI0=res['start']['logI'], logI1=res['iter20000']['logI'], logT0=res['start']['logT'], logT1=res['iter20000']['logT'],
         match0=res['start']['match'], match1=res['iter20000']['match'], nll0=res['start']['nll'], nll1=res['iter20000']['nll'], cand=cand, floored0=res['start']['floored'], floored1=res['iter20000']['floored'])

# showcase pixel: order[0] is the 2-hit main-track pixel closest to the median per-pixel improvement of that class
k = order[0]; pix = int(allp[k]); h = np.where(pid_h == pix)[0]
sc = dict(pix=pix, x=X[k], y=Y[k], s_cm=G['s'][k], ticks=t_h[h], q=q_h[h], pct=(dL[cand] < dL[k]).mean() * 100, dL=dL[k], dL_med=med, ncand=int(cand.sum()),
          dlogI_main_med=np.median(dh[im]), dlogI_main_q25=np.percentile(dh[im], 25), dlogI_main_q75=np.percentile(dh[im], 75), dlogI_all_med=np.median(dh))
for s, S in St.items():
    j = int(np.where(S['pid'] == pix)[0][0])
    sc[f'{s}_P'] = S['Lamslots'][j]; sc[f'{s}_Q'] = S['qslots'][j]; sc[f'{s}_L'] = float(onto(s, 'Lj')[k]); sc[f'{s}_hit'] = float(onto(s, 'hitpart')[k]); sc[f'{s}_exp'] = float(onto(s, 'exppart')[k])
    sc[f'{s}_logI'] = res[s]['logI'][h]; sc[f'{s}_logT'] = res[s]['logT'][h]
    sc[f'{s}_grad'] = np.array([float(S['grad_' + p]) for p in PARAMS]); sc[f'{s}_val'] = np.array([float(S['val_' + p]) for p in PARAMS])
np.savez_compressed(f'{T}/data/a07_pixel.npz', **sc)
print('SHOWCASE', pix, 'pct', round(sc['pct'], 1), 'ticks', sc['ticks'], 'q', sc['q'], 'dL', sc['dL'], 'median', med, 'L', sc['start_L'], sc['iter20000_L'])
