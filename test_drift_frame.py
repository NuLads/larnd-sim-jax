import numpy as np, os
os.environ.setdefault('LARND_CHAIN_FRAME','drift')
import jax.numpy as jnp
from optimize.fit_params import _transverse_frame, _dir_from_angles, _chain_drift_penalty, _DRIFT_AXIS
rng=np.random.default_rng(0); ok=True
print("theta(deg)  e1.z      |e2.z|    sin(th)   orthonorm_err")
for th_deg in [5,30,60,89,90,120,175,179.9]:
    th=np.radians(th_deg); ph=rng.uniform(0,2*np.pi)
    u0=_dir_from_angles(th,ph)
    e1,e2=_transverse_frame(u0)
    err=max(abs(np.dot(e1,e2)),abs(np.dot(e1,u0)),abs(np.dot(e2,u0)),
            abs(np.linalg.norm(e1)-1),abs(np.linalg.norm(e2)-1))
    print(f"{th_deg:9.1f} {np.dot(e1,_DRIFT_AXIS):+9.2e} {abs(np.dot(e2,_DRIFT_AXIS)):9.4f} {np.sin(th):9.4f} {err:9.2e}")
    if err>1e-5: ok=False
    parallel = np.sin(th) <= 1e-3   # matches the code's fallback threshold
    if not parallel:
        if abs(np.dot(e1,_DRIFT_AXIS))>1e-6: ok=False; print("   !! e1 not in wire plane")
        if abs(abs(np.dot(e2,_DRIFT_AXIS))-abs(np.sin(th)))>1e-5: ok=False; print("   !! |e2.z| != sin(theta)")
    else:
        # track along drift: BOTH transverse dirs must lie in the wire plane
        if abs(np.dot(e1,_DRIFT_AXIS))>1e-3 or abs(np.dot(e2,_DRIFT_AXIS))>1e-3:
            ok=False; print("   !! parallel-track fallback leaked drift component")

# penalty: purely wire-plane displacement must cost 0; pure-e2 displacement must cost sin^2
T,nc=2,4
th=np.radians(90.0); u0=_dir_from_angles(th,0.3)
e1,e2=_transverse_frame(u0)
meta={'T':T,'max_nc':nc,'ncs':jnp.array([nc]*T),
      'e1z':jnp.array([e1[2]]*T,dtype=jnp.float32),'e2z':jnp.array([e2[2]]*T,dtype=jnp.float32)}
d_wire=jnp.zeros((T,nc+1,2)).at[:,:,0].set(1.0)     # 1 cm along e1 only
d_e2  =jnp.zeros((T,nc+1,2)).at[:,:,1].set(1.0)     # 1 cm along e2 only
pw=float(_chain_drift_penalty(d_wire,meta,1.0)); pe=float(_chain_drift_penalty(d_e2,meta,1.0))
exp_e2=(nc+1)*np.sin(th)**2
print(f"\npenalty(pure wire-plane) = {pw:.3e}   (expect 0)")
print(f"penalty(pure e2)         = {pe:.4f}     (expect {exp_e2:.4f} = (nc+1)*sin^2)")
print(f"penalty(weight=0)        = {float(_chain_drift_penalty(d_e2,meta,0.0)):.3e}   (expect 0)")
if pw>1e-9 or abs(pe-exp_e2)>1e-3: ok=False
print("\n", "PASS" if ok else "FAIL")
