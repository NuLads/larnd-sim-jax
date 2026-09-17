"""Fig 91 — the calibration-init replication: at-truth initialisation, six replicas, paired.

Fig 84 was n=1 per arm and suggested (a) the start matters a lot and (b) rep 3's monotone
'start lower -> land higher' ordering. With four more replicas:
  * the MONOTONE ordering does NOT replicate (2/6 seeds);
  * at-truth init still helps lifetime on 5/6 and lowers the loss on 5/6, but the lifetime gain
    is NOT significant at n=6 (-2.4 +- 2.8, permutation p=0.16) while the LOSS gain IS (p=0.047);
  * seed 13 is the informative failure: it was the one replica already AT truth (-0.6%), and
    at-truth init pushed it to +8.4%. Initialising at truth does not pin you there.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT='plots/noise_report'
BLUE,ORANGE,GREEN,RED,GREY,INK='#0072B2','#E69F00','#009E73','#D55E00','#888888','#222222'
plt.rcParams.update({'font.size':9.5,'axes.grid':True,'grid.alpha':.25,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.bbox':'tight'})
def ep(tag,k='lifetime'):
    fs=glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs: return None
    h=pickle.load(open(fs[0],'rb'))
    if k=='loss': return float(np.median(np.asarray(h['losses_iter'],float)[-500:]))
    v=np.ravel(h[k+'_iter']); t=float(np.ravel(h[k+'_target'])[0])
    return 100*(np.median(v[int(len(v)*.9):])/t-1)

S=[1,3,7,8,12,13]
fig,ax=plt.subplots(1,3,figsize=(13.6,4.3))

# (a) slopegraph nominal -> at truth
a=ax[0]
for s in S:
    n_,t_=ep(f'DSREP{s}'),ep(f'CINITT{s}')
    col=RED if abs(t_)>abs(n_) else BLUE
    a.plot([0,1],[n_,t_],'o-',color=col,lw=1.6,ms=6)
    a.annotate(f'{s}',(1,t_),textcoords='offset points',xytext=(8,-3),fontsize=8,color=col)
a.axhline(0,color=GREEN,lw=1.8)
a.annotate('truth',(.02,0),xycoords=('axes fraction','data'),textcoords='offset points',
           xytext=(0,5),color=GREEN,fontweight='bold',fontsize=9)
a.set_xticks([0,1],['nominal init\n(−8.6%)','init AT TRUTH'])
a.set_xlim(-.18,1.35)
a.set_ylabel('lifetime error (%)')
a.set_title('(a) helps on 5/6 — but not significantly\n(−2.4 ± 2.8, permutation p = 0.16)',fontsize=9.5)

# (b) loss, the one that IS significant
a=ax[1]
d=[ep(f'CINITT{s}','loss')-ep(f'DSREP{s}','loss') for s in S]
cols=[BLUE if x<0 else RED for x in d]
a.bar(range(len(S)),d,color=cols)
a.axhline(0,color=INK,lw=1)
a.axhline(np.mean(d),color=ORANGE,lw=1.6,ls='--',label=f'mean {np.mean(d):+.0f} (p = 0.047)')
a.set_xticks(range(len(S)),[f'seed {s}' for s in S],fontsize=8)
a.set_ylabel('loss − nominal-init loss')
a.set_title('(b) the LOSS gain replicates: lower on 5/6\nthe nominal init is genuinely sub-optimal',fontsize=9.5)
a.legend(frameon=False,fontsize=8.5)

# (c) the monotone claim, tested
a=ax[2]
for s in S:
    xs=[-31,-8.6,0]; ys=[ep(f'CINITL{s}'),ep(f'DSREP{s}'),ep(f'CINITT{s}')]
    mono = ys[0]>ys[1]>ys[2]
    a.plot(xs,ys,'o-',color=(BLUE if mono else GREY),lw=1.8 if mono else 1.1,
           ms=5,alpha=1 if mono else .75,zorder=3 if mono else 2)
    a.annotate(f'{s}',(0,ys[2]),textcoords='offset points',xytext=(7,-3),fontsize=8,
               color=(BLUE if mono else GREY))
a.axhline(0,color=GREEN,lw=1.8)
a.set_xlabel('calibration START: lifetime error (%)')
a.set_ylabel('LANDING: lifetime error (%)')
a.set_title('(c) fig 84\'s monotone ordering does NOT replicate:\nonly 2/6 seeds (blue) show start-lower → land-higher',fontsize=9.5)

fig.suptitle('Fig 91 — the calibration-init test replicated on six replicas (paired, same data, only the calibration START differs).\n'
             'The loss gain from initialising at truth is real (p = 0.047); the lifetime gain is suggestive but not established at n = 6;\n'
             'the monotone start→landing law of fig 84 was a single-seed accident.',
             fontsize=10.5,y=1.09)
fig.tight_layout()
fig.savefig(f'{OUT}/fig91_cinit_replication.png')
print('wrote fig91')
