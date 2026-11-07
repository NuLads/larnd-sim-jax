import pickle, glob, re, sys, numpy as np
P=['Ab','eField','tran_diff','long_diff','lifetime']
ZH=np.array([0.,0.,1.],np.float32)
def frame(u0, mode):
    if mode=='drift':
        e1=np.cross(ZH,u0); n=np.linalg.norm(e1)
        if n>1e-3:
            e1=e1/n; e2=np.cross(u0,e1); e2/=np.linalg.norm(e2)+1e-12
            return e1,e2
    ref=ZH.copy()
    if abs(float(np.dot(u0,ref)))>0.9: ref=np.array([1.,0.,0.],np.float32)
    e1=np.cross(u0,ref); e1/=np.linalg.norm(e1)+1e-12
    e2=np.cross(u0,e1);  e2/=np.linalg.norm(e2)+1e-12
    return e1,e2
def load(d):
    o={}
    for f in glob.glob(d+'/history_iter*.pkl'):
        m=re.search(r'seed(\d+)',f)
        if m: o[int(m.group(1))]=pickle.load(open(f,'rb'))
    return o
def report(tag,d,mode):
    hs=load(d)
    if not hs: print(f'{tag}: no data'); return
    print(f'=== {tag}  (n={len(hs)} seeds, frame={mode}) ===')
    agg={p:[] for p in P}; gap={p:[] for p in P}; pos=[]
    for s,h in sorted(hs.items()):
        row=[]
        for p in P:
            v=np.ravel(np.array(h[p+'_iter'])); t=np.ravel(h[p+'_target'])[0]; ini=np.ravel(h[p+'_init'])[0]
            ie=(ini/t-1)*100; fe=(v[-1]/t-1)*100
            agg[p].append(fe); gap[p].append(100*(1-abs(fe)/max(abs(ie),1e-12)))
            row.append(f'{p[:4]} {fe:+6.1f}%({100*(1-abs(fe)/max(abs(ie),1e-12)):3.0f}%)')
        pr=np.ravel(h['pos_residual_iter'])*1e4
        pt=np.median(pr[int(len(pr)*.8):]); pos.append(pt)
        print(f'  seed{s}: '+' '.join(row)+f' | pos {pt:5.0f}um')
    print('  MEAN  : '+' '.join(f'{p[:4]} {np.mean(agg[p]):+6.1f}±{np.std(agg[p]):4.1f}(gap {np.mean(gap[p]):3.0f}%)' for p in P)
          +f' | pos {np.mean(pos):.0f}um')
    # drift decomposition of the fitted geometry
    dr,wp=[],[]
    h=hs[sorted(hs)[0]]
    cache,ctxs=h.get('chain_cache'),h.get('chain_contexts')
    if cache and ctxs:
        for b in ctxs:
            for ci,c in enumerate(ctxs[b]):
                try: co=np.asarray(cache[b][ci]['angles'],float)
                except Exception: continue
                K=len(co)//2; b1,b2=co[:K],co[K:2*K]
                th_,ph=float(c['theta0_i']),float(c['phi0_i']); st=np.sin(th_)
                u0=np.array([st*np.cos(ph),st*np.sin(ph),np.cos(th_)],np.float32)
                e1,e2=frame(u0,mode)
                nc=int(c['n_chain']); al=np.arange(nc+1)/nc
                B=np.sin(np.outer(al,np.pi*np.arange(1,K+1)))
                disp=(B@b1)[:,None]*e1[None,:]+(B@b2)[:,None]*e2[None,:]
                dz=disp[:,2]; dp=np.linalg.norm(disp-dz[:,None]*ZH[None,:],axis=1)
                dr.append(np.sqrt(np.mean(dz**2))*1e4); wp.append(np.sqrt(np.mean(dp**2))*1e4)
        dr,wp=np.array(dr),np.array(wp)
        f=dr/np.maximum(np.sqrt(dr**2+wp**2),1e-9)
        print(f'  GEOM (seed{sorted(hs)[0]}, {len(dr)} tracks): drift {np.median(dr):.0f}um | '
              f'wire-plane {np.median(wp):.0f}um | drift fraction {np.median(f)*100:.0f}%')
    print()
report('DRIFTW6b  (drift frame, penalty w=1e6)','fit_result/sci_full_DRIFTW6b','drift')
report('ANNEAL    (legacy frame, no penalty)','fit_result/sci_full_noise_s4_anneal','legacy')
