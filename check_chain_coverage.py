"""
Diagnostic: chain arc-length coverage and fixed_pixel coverage.
"""
import pickle, numpy as np, sys, os, dataclasses, types
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

import jax.numpy as jnp

# Re-define TrackContext so pickle can deserialise it
@dataclasses.dataclass
class TrackContext:
    x0_fixed: object; step_len: float; total_len: float
    sigma_mcs: float; n_chain: int; theta0_i: float; phi0_i: float
    dtheta_i: object; dphi_i: object; x0_chain: object
    bounds_chain: object; n_chain_params: int; target_pixels: object
    target_adc: object; fixed_pixels: object; psn: int; pln: int

@dataclasses.dataclass
class OptResult:
    x: object
    fun: float

# Register under the module name used when pickling
fake_mod = types.ModuleType("run_chain_fit")
fake_mod.TrackContext = TrackContext
fake_mod.OptResult = OptResult
sys.modules["run_chain_fit"] = fake_mod

with open('chain_fit_checkpoint.pkl', 'rb') as f:
    ckpt = pickle.load(f)

PIXEL_PITCH_CM = 0.4434

print(f"Pixel pitch: {PIXEL_PITCH_CM*10:.2f} mm  ({PIXEL_PITCH_CM:.4f} cm)\n")
print(f"{'Track':<16} {'L':>6} {'N':>4} {'step':>6} {'σ(mrad)':>8} "
      f"{'3σ√N(mrad)':>11} {'shortfall(cm)':>14} {'shortfall/px':>13} "
      f"{'endpt_Δ(cm)':>12} {'θ_fit(mrad)':>12} {'tgt_px_missed':>14}")
print("-"*120)

def local_basis(u):
    ref = np.array([0.,0.,1.]) if abs(u[2]) < 0.99 else np.array([1.,0.,0.])
    v = ref - np.dot(ref,u)*u; v /= np.linalg.norm(v)
    return v, np.cross(u,v)

def shortfall(L, theta):
    if theta < 1e-8: return 0.
    return L * (1. - np.sin(theta/2.) / (theta/2.))

for key, r in sorted(ckpt.items()):
    ctx = r['ctx']
    r3  = r['res_s3']

    sigma = ctx.sigma_mcs; N = ctx.n_chain; L = ctx.total_len; step = ctx.step_len
    theta_3s = 3. * np.sqrt(N) * sigma
    sf = shortfall(L, theta_3s)

    # Reconstruct fitted endpoint
    th = float(r3.x[0]); ph = float(r3.x[1])
    dt = r3.x[2:2+N-1]; dp = r3.x[2+N-1:ctx.n_chain_params]
    u = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    pos = np.array(ctx.x0_fixed, dtype=float)
    for dth, dph in zip(dt, dp):
        pos = pos + step*u
        v, w = local_basis(u)
        u = np.cos(dth)*u + np.sin(dth)*(np.cos(dph)*v + np.sin(dph)*w)
        u /= np.linalg.norm(u)
    endpoint_fit = pos + step*u

    # Straight-line exit
    phi0 = ctx.phi0_i % (2*np.pi)
    exit_straight = np.array(ctx.x0_fixed, dtype=float) + L * np.array([
        np.sin(ctx.theta0_i)*np.cos(phi0),
        np.sin(ctx.theta0_i)*np.sin(phi0),
        np.cos(ctx.theta0_i)])
    endpoint_delta = float(np.linalg.norm(endpoint_fit - exit_straight))

    # Total fitted deflection
    theta_fit = float(np.sum(np.abs(dt))) * 1e3  # mrad

    # Target pixels outside fixed_pixels
    fixed_set  = set(int(x) for x in np.array(ctx.fixed_pixels) if x >= 0)
    target_set = set(int(x) for x in np.array(ctx.target_pixels) if x >= 0)
    missed     = len(target_set - fixed_set)

    print(f"b={key[0]} ev={key[1]} tr={key[2]:<4}"
          f"{L:6.1f} {N:4d} {step:6.3f} {sigma*1e3:8.3f} "
          f"{theta_3s*1e3:11.1f} {sf:14.4f} {sf/PIXEL_PITCH_CM:13.3f} "
          f"{endpoint_delta:12.4f} {theta_fit:12.1f} {missed:>9d}/{len(target_set)}")

print()
print("shortfall = arc-length shortfall at 3σ_MCS*sqrt(N) total deflection")
print("shortfall/px > 1 would mean the curved chain misses >= 1 pixel at the far end")
print("endpt_Δ = displacement of fitted chain endpoint from straight-line exit (cm)")
print("θ_fit = sum(|dtheta|) of all fitted joints (crude total deflection estimate, mrad)")
print("tgt_px_missed = target pixels initially outside fixed_pixel set (should be 0)")
