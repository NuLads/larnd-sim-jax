import numpy as np
z = np.load('/sdf/scratch/neutrino/pgranger/talk_animations/data/a01_03_sim.npz')
seg=z['seg']; X,Y,Z,DX,DE,DEDX,TID=seg[:,0],seg[:,1],seg[:,2],seg[:,9],seg[:,10],seg[:,11],seg[:,12]
nq,nd,td,sT,sL,sx,sy=z['nq'],z['nd'],z['tdrift'],z['sT'],z['sL'],z['sx'],z['sy']
wfs=z['wfs']; upx,upy=z['upx'],z['upy']; hx,hy=z['hix'],z['hiy']; ht=z['hit_t']; hq=z['hit_q']
TS=float(z['t_sampling']); vd=float(z['vd']); THR=5000.
print('table time offset: 1884 - full-drift ticks', 1884 - float(z['drift_length'])/vd/TS, 'dl', float(z['drift_length']))
# noiseless crossing
key=lambda a,b:a*100000+b
ukey=key(upx.astype(np.int64),upy.astype(np.int64)); hkey=key(hx.astype(np.int64),hy.astype(np.int64)); skey=key(sx,sy)
Q=np.cumsum(wfs*TS,axis=1)
rows=[]
for k in np.unique(hkey):
    iu=np.where(ukey==k)[0][0]
    c=np.argmax(Q[iu]>=THR) if (Q[iu]>=THR).any() else -1
    hts=np.sort(ht[hkey==k])
    m=skey==k
    if m.sum()==0: continue
    o=np.argsort(td[m]); cq=np.cumsum(nd[m][o]); 
    # drift tick at which landing charge reaches THR
    j=np.searchsorted(cq,THR); j=min(j,len(o)-1)
    tthr=td[m][o][j]/TS
    tid=TID[m][0]
    rows.append((k,hts[0],c,tthr,td[m].min()/TS,np.average(td[m],weights=nd[m])/TS,Q[iu].max(),len(hts),m.sum(),tid,wfs[iu].argmax(),wfs[iu].max()))
r=np.array(rows)
ok=r[:,2]>0
print('N',len(r),'with noiseless crossing',ok.sum())
d=r[ok,1]-r[ok,2]; print('first-hit tick - noiseless 5ke crossing tick: median %.1f p16 %.1f p84 %.1f'%(np.median(d),*np.percentile(d,[16,84])))
d=r[ok,2]-r[ok,3]; print('noiseless crossing - drift tick where landing charge reaches 5ke: median %.1f p16 %.1f p84 %.1f'%(np.median(d),*np.percentile(d,[16,84])))
d=r[ok,1]-r[ok,3]; print('first hit - drift tick where landing charge reaches 5ke: median %.1f p16 %.1f p84 %.1f'%(np.median(d),*np.percentile(d,[16,84])))
A=np.vstack([r[ok,3],np.ones(ok.sum())]).T; print('fit hit tick vs drift tick@5ke', np.linalg.lstsq(A,r[ok,1],rcond=None)[0])
A=np.vstack([r[ok,3],np.ones(ok.sum())]).T; print('fit noiseless crossing vs drift tick@5ke', np.linalg.lstsq(A,r[ok,2],rcond=None)[0])
for t in np.unique(r[:,9]):
    m=ok&(r[:,9]==t); A=np.vstack([r[m,3],np.ones(m.sum())]).T; print('track',t,m.sum(),'tick range',r[m,3].min(),r[m,3].max(),'fit',np.linalg.lstsq(A,r[m,1],rcond=None)[0], 'mean diff', (r[m,1]-r[m,3]).mean())
# candidates for a03: main track pixels
m=(r[:,9]==4)
s=r[m]; o=np.argsort(-s[:,6])[:15]
print('pix(x,y) hits rows Qtot wfmax wfargmax meandrift firsthit')
for i in o:
    k=int(s[i,0]); print(k//100000,k%100000,int(s[i,7]),int(s[i,8]),int(s[i,6]),int(s[i,11]),int(s[i,10]),round(s[i,5],1),s[i,1])
print('hits per pixel hist', np.unique(r[:,7],return_counts=True))
print('main track, mean drift>700 tick:')
s2=s[s[:,5]>700]; o=np.argsort(-s2[:,6])[:12]
for i in o:
    k=int(s2[i,0]); print(k//100000,k%100000,int(s2[i,7]),int(s2[i,8]),int(s2[i,6]),int(s2[i,11]),int(s2[i,10]),round(s2[i,5],1),s2[i,1])
print('Q frac neighbours (0,0) raw peaks')
raw=z['raw_tail']; print(raw.shape)
for ij in [(0,0),(2,2),(4,4),(5,0),(7,0),(10,0),(10,10),(15,0),(20,0)]:
    w=raw[ij]; print(ij,'max %.4f at %d  min %.4f at %d'%(w.max(),w.argmax()-399,w.min(),w.argmin()-399),' sum*0.1 %.4f'%(w.sum()*0.1), 'cum raw sum', round(float(z['raw_sum'][ij])*0.1,4))
print('nq per row', nq[TID==4][:5], DE[TID==4][:5], 'dt', td[TID==4][:3], 'longest', td[TID==4].max(), np.argmax(td*(TID==4)), 'sT', sT[np.argmax(td*(TID==4))], 'sL', sL[np.argmax(td*(TID==4))], X[np.argmax(td*(TID==4))],Y[np.argmax(td*(TID==4))],Z[np.argmax(td*(TID==4))])
