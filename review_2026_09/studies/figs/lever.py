import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
C=['#2a78d6','#eb6834','#1baf7a']
vd=0.1587
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
tau=np.linspace(500,6000,300)
for c,(L,lab) in zip(C,[(30.27,'module-0 / 2x2 (30 cm drift)'),(50.4,'ND-LAr module (50 cm drift)')]):
    t=L/vd
    ax[0].plot(tau, 100*(1-np.exp(-t/tau)), color=c, lw=2, label=lab)
    # a fractional charge error eps at max drift (linear in t) -> dtau/tau = tau*eps/t
    ax[1].plot(tau, 100*tau*0.01/t, color=c, lw=2, label=lab)
ax[0].set_xlabel('electron lifetime τ (µs)'); ax[0].set_ylabel('charge lost at full drift (%)')
ax[0].set_title('(a) the lifetime signal is small',loc='left',fontsize=10)
ax[1].set_xlabel('electron lifetime τ (µs)'); ax[1].set_ylabel('lifetime error (%)')
ax[1].set_title('(b) lifetime error from a 1% drift-correlated\n     charge mis-model (at full drift)',loc='left',fontsize=10)
for a in ax:
    a.axvline(2200,color='#8a8983',ls='--',lw=1); a.grid(axis='y',color='#e5e4df',lw=0.8); a.legend(frameon=False,fontsize=8)
ax[0].text(2250,ax[0].get_ylim()[1]*0.9,'simulated truth\n2200 µs',fontsize=8,color='#52514e')
fig.tight_layout(); fig.savefig('R1_lever_arm.png',dpi=150)
t=30.27/vd; print('tmax',t,'loss@2200',1-np.exp(-t/2200),'amp 1%->',2200*0.01/t)
