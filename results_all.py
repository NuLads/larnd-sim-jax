import pickle, glob, numpy as np, re, os
P=['Ab','eField','lifetime','tran_diff','long_diff']
def rep(tag,d):
    seeds={}
    for f in glob.glob(d+'/history_iter*.pkl'):
        m=re.search(r'seed(\d+)',f)
        if not m: continue
        s=m.group(1); it=int(re.search(r'iter(\d+)',f).group(1))
        if s not in seeds or it>seeds[s][0]: seeds[s]=(it,f)
    agg={p:[] for p in P}; pos=[]
    for s,(it,f) in sorted(seeds.items()):
        h=pickle.load(open(f,'rb')); out=[]
        for p in P:
            v=np.ravel(np.array(h[p+'_iter'])); t=np.ravel(h[p+'_target'])[0]
            e=(v[-1]/t-1)*100; agg[p].append(e); out.append(f"{p[:4]} {e:+6.1f}%")
            if p=='lifetime':
                n=len(v); a=(v[int(n*0.7)]/t-1)*100; drift=e-a   # last-30% movement
        pr=h.get('pos_residual_iter'); ps=''
        if pr is not None and len(np.ravel(pr)): 
            pv=np.ravel(pr)[-1]*1e4; pos.append(pv); ps=f" pos{pv:4.0f}um"
        print(f"  {tag:12s} sd{s} {it:5d}it: "+" ".join(out)+ps+f"  [tau last30%: {drift:+.1f}pt]")
    if agg['lifetime']:
        m=lambda p: f"{np.mean(agg[p]):+.1f}+-{np.std(agg[p]):.1f}"
        print(f"  {tag:12s} MEAN(n={len(agg['lifetime'])}): Ab {m('Ab')} eF {m('eField')} "
              f"TAU {m('lifetime')} tran {m('tran_diff')} LONG {m('long_diff')}"
              + (f" pos {np.mean(pos):.0f}um" if pos else ""))
    print()
for tag,d in [('ANNEAL-S4','fit_result/sci_full_noise_s4_anneal'),
              ('ANNEAL-S2','fit_result/sci_ceiling_noise_s2_anneal'),
              ('CHSTART1000','fit_result/sci_full_CHSTART1000'),
              ('CHSTART2500','fit_result/sci_full_CHSTART2500'),
              ('FRAMEONLY','fit_result/sci_full_FRAMEONLY'),
              ('DRIFTW6','fit_result/sci_full_DRIFTW6'),
              ('DRIFTW7','fit_result/sci_full_DRIFTW7')]:
    if os.path.isdir(d): rep(tag,d)
