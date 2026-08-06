"""Schedule-invariance: does the answer depend on the LR schedule, or only on the data?"""
import pickle, glob, re, os
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, optax
OUT='plots/noise_report'; os.makedirs(OUT,exist_ok=True)
P=['Ab','eField','tran_diff','long_diff','lifetime']
PL={'Ab':'A$_b$','eField':'E field','tran_diff':'tran. diff.','long_diff':'long. diff.','lifetime':'lifetime'}
C=['#0072B2','#E69F00','#009E73']; GREY,RED='#666666','#D55E00'
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
RUNS=[('ANNEAL 5k\n(decay 0.91)','fit_result/sci_full_noise_s4_anneal',0.91,5000),
      ('ANNEALLONG 10k\n(decay 0.91)','fit_result/sci_full_ANNEALLONG',0.91,10000),
      ('SLOWANNEAL 10k\n(decay 0.9539)','fit_result/sci_full_SLOWANNEAL',0.9539,10000)]
def load(d):
    o={}
    for f in glob.glob(d+'/history_iter*.pkl'):
        if 'len400' not in f: continue
        m=re.search(r'seed(\d+)',f)
        if m: o[int(m.group(1))]=pickle.load(open(f,'rb'))
    return o
def robust(v,t):
    e=(np.asarray(v,float)/t-1)*100; return float(np.median(e[int(len(e)*.8):]))

fig,axes=plt.subplots(1,3,figsize=(14.4,4.0),gridspec_kw={'width_ratios':[1.35,1,1]})
# (a) per-parameter across the three schedules
ax=axes[0]; w=.26
for j,(lab,d,dr,ni) in enumerate(RUNS):
    hs=load(d)
    for k,p in enumerate(P):
        vals=[robust(np.ravel(np.array(h[p+'_iter'])),np.ravel(h[p+'_target'])[0]) for h in hs.values()]
        ax.errorbar(k+(j-1)*w, np.mean(vals), yerr=np.std(vals), fmt='o', ms=6, capsize=3,
                    color=C[j], label=lab.replace('\n',' ') if k==0 else None)
ax.axhline(0,color='k',lw=.9); ax.axhspan(-5,5,color=GREY,alpha=.16,zorder=0)
ax.set_xticks(range(len(P))); ax.set_xticklabels([PL[p] for p in P],fontsize=8)
ax.set_ylabel('error (%)'); ax.legend(fontsize=7.5,frameon=False)
ax.set_title('(a) Same answer across schedules — except lifetime\nshaded = ±5%',loc='left',fontsize=9)
# (b) lifetime vs total travel
ax=axes[1]
tot=[]
for j,(lab,d,dr,ni) in enumerate(RUNS):
    s=optax.warmup_exponential_decay_schedule(init_value=0.,peak_value=1e-1,warmup_steps=500,
        transition_steps=100,decay_rate=dr,staircase=True)
    travel=float(np.sum([s(t) for t in range(ni)]))
    hs=load(d)
    vals=[robust(np.ravel(np.array(h['lifetime_iter'])),np.ravel(h['lifetime_target'])[0]) for h in hs.values()]
    tot.append((travel,np.mean(vals),np.std(vals),lab))
    ax.errorbar(travel,np.mean(vals),yerr=np.std(vals),fmt='o',ms=9,capsize=4,color=C[j])
    ax.annotate(lab.split('\n')[0],(travel,np.mean(vals)),textcoords='offset points',
                xytext=(6,8),fontsize=7.5,color=C[j])
ax.axhline(0,color='k',lw=.9); ax.axhspan(-5,5,color=GREY,alpha=.16,zorder=0)
ax.set_xlabel('total optimizer travel  $\\sum$LR'); ax.set_ylabel('lifetime error (%)')
ax.set_title('(b) Lifetime keeps drifting with more travel\n→ not converged; value is a LOWER bound on bias',loc='left',fontsize=9)
# (c) position
ax=axes[2]
for j,(lab,d,dr,ni) in enumerate(RUNS):
    hs=load(d); pv=[]
    for h in hs.values():
        pr=np.ravel(h['pos_residual_iter'])*1e4; pv.append(np.median(pr[int(len(pr)*.8):]))
    ax.errorbar(j,np.mean(pv),yerr=np.std(pv),fmt='s',ms=8,capsize=4,color=C[j])
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in RUNS],fontsize=7.5)
ax.set_ylabel('position residual (µm)')
ax.set_title('(c) Position: 10k clearly beats 5k,\nthen converges',loc='left',fontsize=9)
fig.suptitle('Fig 20 — Schedule invariance test. SLOWANNEAL gives the same final LR as ANNEAL but 1.78x more '
             'total travel.\nEverything lands within the 1.25-point run-to-run noise floor EXCEPT lifetime.',
             x=.02,ha='left',y=1.05)
fig.tight_layout(); fig.savefig(f'{OUT}/fig20_schedule_invariance.png'); plt.close(fig)
print('travel / lifetime:'); [print(f'  {l.split(chr(10))[0]:22s} travel {t:7.1f}  lifetime {m:+6.2f} ± {s:.2f}') for t,m,s,l in tot]
