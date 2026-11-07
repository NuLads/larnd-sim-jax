import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
r=np.load('scale_rows.npy')*100
C=['#2a78d6','#eb6834','#1baf7a','#4a3aa7']
fig,ax=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw=dict(width_ratios=[1,1.15]))
groups=[('production fits, 16 independent samples',slice(0,16),C[0],'o'),('both nuisances pinned at truth',slice(16,19),C[1],'s'),('dE/dx pinned at truth',slice(19,22),C[2],'D'),('dE/dx frozen at its own fit',slice(22,25),C[3],'^')]
for lab,sl,c,m in groups:
    ax[0].scatter(r[sl,0],r[sl,1],color=c,marker=m,s=48,label=lab,edgecolor='white',lw=1,zorder=3)
x=np.linspace(-0.2,0.25,10)
ax[0].plot(x,1.46-20.3*x,color='#8a8983',ls='--',lw=1.5,label='NLL valley at truth (GPU diagnostic),\nslope −20 %/%',zorder=2)
ax[0].axhline(0,color='#c3c2b7',lw=0.8); ax[0].axvline(0,color='#c3c2b7',lw=0.8)
ax[0].set_xlabel('Ab error (%)'); ax[0].set_ylabel('lifetime error (%)')
ax[0].set_title('(a) lifetime error follows the charge-scale (Ab) error',loc='left',fontsize=10)
ax[0].legend(frameon=False,fontsize=7.5,loc='upper right')
labs=['model @truth\nAb profiled\n(GPU diag.)','model @truth\nAb fixed\n(GPU diag.)','fit: both\nnuisances\n@truth','fit: dE/dx\n@truth','fit: dE/dx\nfrozen @own','production\nfit (n=16)']
val=[0.51,1.46,-3.74,3.01,10.6,8.85]; err=[0.27,0.15,0.39,2.55,1.13,1.18]
cols=['#8a8983','#8a8983',C[1],C[2],C[3],C[0]]
ax[1].bar(range(6),val,yerr=err,color=cols,width=0.62,capsize=3,error_kw=dict(ecolor='#52514e',lw=1))
for i,v in enumerate(val):
    y = v+err[i]+0.4 if v>=0 else v+1.0
    ax[1].text(i,y,f'{v:+.1f}',ha='center',fontsize=8.5,color='#0b0b0b' if v>=0 else 'white',va='bottom')
ax[1].axhline(0,color='#52514e',lw=0.8); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(labs,fontsize=7.8)
ax[1].set_ylabel('lifetime error (%)'); ax[1].set_title('(b) where the lifetime bias appears (mean ± SE)',loc='left',fontsize=10)
ax[1].grid(axis='y',color='#e5e4df',lw=0.8); ax[1].set_axisbelow(True)
fig.tight_layout(); fig.savefig('R2_lifetime_bias_anatomy.png',dpi=150)
