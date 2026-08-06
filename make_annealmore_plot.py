"""Fig 15 — reproducibility of the LR-anneal result at n=6 seeds (adds ANNEALMORE seeds 3-5)."""
import pickle, glob, re, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='plots/noise_report'; os.makedirs(OUT,exist_ok=True)
P=['Ab','eField','tran_diff','long_diff','lifetime']
PL={'Ab':'A$_b$','eField':'E field','tran_diff':'tran. diff.','long_diff':'long. diff.','lifetime':'lifetime'}
C=dict(blue='#0072B2',orange='#E69F00',green='#009E73',red='#D55E00',grey='#666666')
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
def load(d):
    o={}
    for f in glob.glob(d+'/history_iter*.pkl'):
        m=re.search(r'seed(\d+)',f)
        if m: o[int(m.group(1))]=pickle.load(open(f,'rb'))
    return o
runs={}
for s,h in load('fit_result/sci_full_noise_s4_anneal').items(): runs[s]=('orig',h)
for s,h in load('fit_result/sci_full_ANNEALMORE').items():      runs[s]=('new',h)
seeds=sorted(runs)
def stat(h,p):
    v=np.ravel(np.array(h[p+'_iter'])); t=np.ravel(h[p+'_target'])[0]; ini=np.ravel(h[p+'_init'])[0]
    ie=(ini/t-1)*100; fe=(v[-1]/t-1)*100
    return fe, 100*(1-abs(fe)/max(abs(ie),1e-12)), ie

fig,axes=plt.subplots(1,3,figsize=(14.4,3.9),gridspec_kw={'width_ratios':[1.25,1.15,1.1]})
# (a) per-parameter final error, all seeds
ax=axes[0]
for j,p in enumerate(P):
    for s in seeds:
        grp,h=runs[s]; fe,_,_=stat(h,p)
        ax.scatter(j+(0.16 if grp=='new' else -0.16), fe, s=44, zorder=3,
                   color=C['blue'] if grp=='orig' else C['orange'],
                   marker='o' if grp=='orig' else 's', edgecolor='white', linewidth=.7)
    m=np.mean([stat(runs[s][1],p)[0] for s in seeds]); sd=np.std([stat(runs[s][1],p)[0] for s in seeds])
    ax.plot([j-.32,j+.32],[m,m],color='k',lw=1.6)
    ax.text(j, 15.5, f'{m:+.2f}\n±{sd:.2f}', ha='center', fontsize=7)
ax.axhline(0,color='k',lw=.9); ax.axhspan(-5,5,color=C['grey'],alpha=.16,zorder=0)
ax.set_xticks(range(len(P))); ax.set_xticklabels([PL[p] for p in P],fontsize=8)
ax.set_ylim(-22,22); ax.set_ylabel('final error (%)')
ax.scatter([],[],color=C['blue'],marker='o',label='seeds 0–2 (original)')
ax.scatter([],[],color=C['orange'],marker='s',label='seeds 3–5 (ANNEALMORE)')
ax.legend(fontsize=7.5,frameon=False,loc='lower left')
ax.set_title('(a) All five parameters, 6 independent seeds\nblack bar = mean; shaded = ±5%',loc='left',fontsize=9)
# (b) lifetime trajectories
ax=axes[1]
for s in seeds:
    grp,h=runs[s]
    v=np.ravel(np.array(h['lifetime_iter'])); t=np.ravel(h['lifetime_target'])[0]
    ax.plot((v/t-1)*100, lw=1.0, alpha=.85,
            color=C['blue'] if grp=='orig' else C['orange'],
            ls='-' if grp=='orig' else '--')
ax.axhline(0,color='k',lw=.9); ax.axhspan(-5,5,color=C['grey'],alpha=.16,zorder=0)
ax.set_ylim(-40,120); ax.set_xlabel('iteration'); ax.set_ylabel('lifetime error (%)')
ax.set_title('(b) Lifetime trajectories — the new seeds (dashed)\nfollow the same convergence pattern',loc='left',fontsize=9)
# (c) gap-closed for the two soft parameters
ax=axes[2]
w=.36
for k,p in enumerate(['lifetime','long_diff']):
    vals=[stat(runs[s][1],p)[1] for s in seeds]
    ax.bar(np.arange(len(seeds))+(k-.5)*w, np.clip(vals,-20,None), w,
           color=[C['red'],C['green']][k], label=PL[p], edgecolor='white', linewidth=.6)
ax.axhline(0,color='k',lw=.9); ax.axhline(50,ls=':',color=C['grey'])
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f'seed {s}' for s in seeds],fontsize=7.5,rotation=30)
ax.set_ylabel('% of initial offset removed'); ax.set_ylim(-22,112)
ax.legend(fontsize=8,frameon=False)
ax.set_title('(c) How much work each seed did\n5 of 6 seeds close ≥64% of the lifetime gap',loc='left',fontsize=9)
fig.suptitle('Fig 15 — Reproducibility of the LR-anneal result at n=6 seeds. Adding three fresh seeds '
             'reproduces the\nrecovery and widens the honest error bar: lifetime −2.10 ± 4.49%, position 255 ± 13 µm.',
             x=.02,ha='left',y=1.07)
fig.savefig(f'{OUT}/fig15_annealmore_n6.png'); plt.close(fig)
print('wrote fig15')
