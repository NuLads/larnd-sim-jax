"""Fig 16 — what LR annealing actually does to each parameter block."""
import numpy as np, optax, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=dict(blue='#0072B2',orange='#E69F00',green='#009E73',red='#D55E00',grey='#666666',purple='#CC79A7')
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
E,N=100,10000
def calib(dr,n=N):
    s=optax.warmup_exponential_decay_schedule(init_value=0.,peak_value=1e-1,warmup_steps=500,
        transition_steps=E,decay_rate=dr,staircase=True)
    return np.array([float(s(t)) for t in range(n)])
t=np.arange(N)
old,new=calib(0.999),calib(0.91)
dedx=np.full(N,1e-2)                       # optax.adam(1e-2): NO schedule
geom=1e-2*0.9997**t                        # chain_lr x chain_decay_rate**t
fig,axes=plt.subplots(1,3,figsize=(14.0,3.8))
ax=axes[0]
ax.plot(t[:5000],old[:5000],lw=2,color=C['red'],label='calibration, decay 0.999 (OLD)')
ax.plot(t[:5000],new[:5000],lw=2,color=C['blue'],label='calibration, decay 0.91 (ANNEAL)')
ax.axvline(500,ls=':',color=C['grey']); ax.text(560,6e-2,'end of warmup',fontsize=7,color=C['grey'])
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('learning rate')
ax.legend(fontsize=7.5,frameon=False,loc='lower left')
ax.set_title('(a) The only thing that changed.\nIdentical for 500 warmup steps, then diverge',loc='left',fontsize=9)
ax=axes[1]
for y,lab,c,ls in [(old[:5000]/1e-1,'calibration OLD (decay 0.999)',C['red'],'-'),
                   (new[:5000]/1e-1,'calibration ANNEAL (decay 0.91)',C['blue'],'-'),
                   (dedx[:5000]/1e-2,'dE/dx nuisance (~4000 params) — NO schedule',C['orange'],'--'),
                   (geom[:5000]/1e-2,'geometry (chain) — decay 0.9997/step',C['green'],'-.')]:
    ax.plot(t[:5000],y*100,lw=1.9,color=c,ls=ls,label=lab)
ax.set_yscale('log'); ax.set_ylim(.5,200); ax.set_xlabel('iteration')
ax.set_ylabel('% of that block\'s own peak LR')
ax.legend(fontsize=7,frameon=False,loc='lower left')
ax.set_title('(b) Relative "temperature" of each block.\nOLD: physics params stay as hot as the nuisances',loc='left',fontsize=9)
ax=axes[2]
ax.plot(t,new/1e-1*100,lw=2,color=C['blue'])
ax.axvspan(5000,10000,color=C['grey'],alpha=.16)
ax.text(5250,3,'ANNEALLONG\nregion:\nLR < 0.02% of peak\n→ quenched, not\n   converging',fontsize=7.5,color=C['grey'])
ax.axvline(5000,ls=':',color='k')
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('% of peak LR')
ax.set_title('(c) Caveat for the 10k run: with decay 0.91 the\ncalibration LR is ~0 well before 10000 iterations',loc='left',fontsize=9)
fig.suptitle('Fig 16 — What "annealing the calibration LR" actually does. Decay is applied once per EPOCH '
             '(100 iterations,\nstaircase), so over 50 epochs 0.999 leaves 95.7% of the peak while 0.91 leaves 1.6%.',
             x=.02,ha='left',y=1.06)
fig.savefig('plots/noise_report/fig16_lr_schedules.png'); plt.close(fig)
print('wrote fig16')
