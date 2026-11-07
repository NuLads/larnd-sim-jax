"""3.1 — per-seed plateau detection. Re-reads every campaign variant at each seed's OWN
plateau instead of a shared iteration count, to check whether past rankings survive."""
import pickle, glob, numpy as np, re, os, json
P=['Ab','eField','lifetime','tran_diff','long_diff']

def plateau_idx(v, frac_win=0.10, tol=0.25):
    """First index after which the trajectory's rolling std (relative to its own rolling
    mean) stays below `tol` x the early-phase std for the rest of the run."""
    n=len(v)
    if n<200: return n-1
    w=max(20,int(n*frac_win))
    rs=np.array([v[max(0,i-w):i+1].std() for i in range(n)])
    early=rs[int(n*0.1):int(n*0.3)].mean()
    if early<=0: return n-1
    ok=rs<tol*early
    idx=n-1
    for i in range(int(n*0.2),n):
        if ok[i:].mean()>0.9: idx=i; break
    return idx

rows=[]
for d in sorted(glob.glob('fit_result/sci_*')):
    seeds={}
    for f in glob.glob(d+'/history_iter*.pkl'):
        m=re.search(r'seed(\d+)',f)
        if not m: continue
        s=m.group(1); it=int(re.search(r'iter(\d+)',f).group(1))
        if s not in seeds or it>seeds[s][0]: seeds[s]=(it,f)
    for s,(it,f) in sorted(seeds.items()):
        try: h=pickle.load(open(f,'rb'))
        except Exception: continue
        if 'lifetime_iter' not in h: continue
        v=np.ravel(np.array(h['lifetime_iter'])); t=np.ravel(h['lifetime_target'])[0]
        pi=plateau_idx(v)
        r={'variant':os.path.basename(d),'seed':s,'iters':it,'plateau_it':int(pi),
           'frac':round(pi/len(v),2)}
        for p in P:
            if p+'_iter' not in h: continue
            vv=np.ravel(np.array(h[p+'_iter'])); tt=np.ravel(h[p+'_target'])[0]
            r[p+'_end']=round((vv[-1]/tt-1)*100,1)
            r[p+'_plat']=round((vv[min(pi,len(vv)-1)]/tt-1)*100,1)
        rows.append(r)
json.dump(rows,open('plateau_audit.json','w'),indent=1)
print(f"{'variant':34s} {'sd':>2s} {'iters':>6s} {'plat':>6s} {'tau_end':>8s} {'tau_plat':>9s} {'long_end':>9s} {'long_plat':>10s}")
for r in rows:
    print(f"{r['variant'][:34]:34s} {r['seed']:>2s} {r['iters']:6d} {r['plateau_it']:6d} "
          f"{r.get('lifetime_end',float('nan')):8.1f} {r.get('lifetime_plat',float('nan')):9.1f} "
          f"{r.get('long_diff_end',float('nan')):9.1f} {r.get('long_diff_plat',float('nan')):10.1f}")
