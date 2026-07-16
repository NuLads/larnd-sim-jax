"""Stage 2 — input-segment marginalization.

Applies a Gaussian perturbation to the position columns of each segment before the
segment enters the differentiable simulator. Purpose: correctly propagate the reconstruction
noise (characterized by sigma_hat, derived once from a closure-test residual) through the
nonlinear simulator so the diffusion parameter fits the physical signal instead of silently
absorbing the input spread. Details / physics motivation live in the Stage 1 FINDINGS.md.

Design contract:
  * Same ε is added to (x_start, y_start, z_start), (x_end, y_end, z_end), and (x, y, z)
    of one segment. That preserves segment length, direction, and dQ/dx while shifting
    the whole segment by ε.
  * sigma_hat.npz is derived in the *file* frame (z = drift). TracksDataset applies
    swap_xz when loading, so at the point this function runs, tracks live in the *code*
    frame where x = drift. We swap the sigma components accordingly at load time.
  * The RNG stream is independent of the sim's own rngkey — it is derived from the
    training iteration index with a separate offset so noise for input and internal
    simulator noise never coincide.

Usage (from fit_params.compute_loss):
    from larndsim.marginalization import SigmaHatPerturbation
    self.marginalization = SigmaHatPerturbation.from_npz(sigma_hat_npz, apply_swap=True)
    ...
    if self.marginalization is not None:
        tracks = self.marginalization.perturb(tracks, self.sim_track_fields, i)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


POSITION_FIELDS = ('x_start', 'y_start', 'z_start',
                   'x_end',   'y_end',   'z_end',
                   'x',       'y',       'z')


@dataclass(frozen=True)
class SigmaHatPerturbation:
    """Immutable container of the perturbation configuration.

    Attributes:
      sigma_code : (3,) float32 array — (sigma_x, sigma_y, sigma_z) in the *code* frame
                   (i.e. after swap_xz has been applied by the loader).
      seed_offset: int — added to the iteration index to derive an independent PRNGKey
                   stream that does not collide with the sim's own rngkey.

    Note: K > 1 sample averaging (marginalize over multiple ε per step) is handled at the
    compute_loss level: it calls perturb() K times with distinct seeds, runs the loss/grad
    for each, and averages. This module only ever produces one ε draw per call.
    """
    sigma_code: jnp.ndarray
    seed_offset: int = 1_000_003  # large prime, no overlap with typical rngkey magnitudes

    @classmethod
    def from_npz(cls, path: str, apply_swap: bool = True,
                 seed_offset: int = 1_000_003) -> 'SigmaHatPerturbation':
        """Load sigma_hat.npz (file frame) and convert to code frame if apply_swap.

        The file frame has (x, y, z) = (transverse1, transverse2, drift).
        After TracksDataset(swap_xz=True), the code frame has (x, y, z) = (drift, transverse2, transverse1).
        So the swap is: sigma_x_code = sigma_z_file, sigma_z_code = sigma_x_file.
        """
        d = np.load(path)
        sigma_file = np.asarray(d['sigma_diag'], dtype=np.float32)  # (sig_x, sig_y, sig_z) file frame
        assert sigma_file.shape == (3,), f"sigma_diag shape {sigma_file.shape}"
        if apply_swap:
            sigma_code = np.array([sigma_file[2], sigma_file[1], sigma_file[0]], dtype=np.float32)
        else:
            sigma_code = sigma_file.astype(np.float32)
        return cls(sigma_code=jnp.asarray(sigma_code), seed_offset=int(seed_offset))

    def perturb(self, tracks: jnp.ndarray, fields, iteration: int) -> jnp.ndarray:
        """Return tracks with each segment's position fields shifted by ε ~ N(0, diag(sigma_code)^2).

        Same ε per segment applied to start / end / midpoint. Fields is the tuple returned by
        TracksDataset.track_fields (already renamed so 'eventID', 'trackID' — but position
        field names are unchanged).
        """
        xs, ys, zs = fields.index('x_start'), fields.index('y_start'), fields.index('z_start')
        xe, ye, ze = fields.index('x_end'),   fields.index('y_end'),   fields.index('z_end')
        xm, ym, zm = fields.index('x'),       fields.index('y'),       fields.index('z')

        return _perturb_impl(
            tracks, self.sigma_code, int(iteration) + self.seed_offset,
            xs, ys, zs, xe, ye, ze, xm, ym, zm,
        )


@partial(jax.jit, static_argnums=range(3, 12))
def _perturb_impl(tracks, sigma_code, seed,
                  xs, ys, zs, xe, ye, ze, xm, ym, zm):
    key = jax.random.PRNGKey(seed)
    n = tracks.shape[0]
    eps = jax.random.normal(key, (n, 3)) * sigma_code[None, :]  # (N, 3) in code frame

    tracks = tracks.at[:, xs].add(eps[:, 0])
    tracks = tracks.at[:, xe].add(eps[:, 0])
    tracks = tracks.at[:, xm].add(eps[:, 0])

    tracks = tracks.at[:, ys].add(eps[:, 1])
    tracks = tracks.at[:, ye].add(eps[:, 1])
    tracks = tracks.at[:, ym].add(eps[:, 1])

    tracks = tracks.at[:, zs].add(eps[:, 2])
    tracks = tracks.at[:, ze].add(eps[:, 2])
    tracks = tracks.at[:, zm].add(eps[:, 2])
    return tracks


if __name__ == '__main__':
    # Smoke test: build a fake tracks array, perturb, and confirm the shift is applied.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sigma_hat', required=True)
    args = p.parse_args()

    fields = ('eventID', 'trackID', 'x_start', 'y_start', 'z_start',
              'x_end', 'y_end', 'z_end', 'x', 'y', 'z', 'dE')
    tracks = jnp.array([
        [0., 1.,  0.10, 0.20, 0.30,  0.11, 0.21, 0.31,  0.105, 0.205, 0.305, 1.0],
        [0., 1.,  0.11, 0.21, 0.31,  0.12, 0.22, 0.32,  0.115, 0.215, 0.315, 1.0],
    ])
    marg = SigmaHatPerturbation.from_npz(args.sigma_hat, apply_swap=True)
    print(f"sigma_code (cm) = {marg.sigma_code}")
    out = marg.perturb(tracks, fields, iteration=0)
    print("before → after start_x for seg 0:", float(tracks[0, 2]), "→", float(out[0, 2]))
    print("delta_start_x - delta_end_x - delta_midpoint_x (should all be ~equal per segment):")
    for i in (0, 1):
        d_s = float(out[i, 2]) - float(tracks[i, 2])
        d_e = float(out[i, 5]) - float(tracks[i, 5])
        d_m = float(out[i, 8]) - float(tracks[i, 8])
        print(f"  seg {i}: dxs={d_s:+.5e}  dxe={d_e:+.5e}  dxm={d_m:+.5e}  identical={d_s == d_e == d_m}")
    # segment length preserved?
    def seg_len(t):
        return jnp.sqrt((t[:, 5]-t[:, 2])**2 + (t[:, 6]-t[:, 3])**2 + (t[:, 7]-t[:, 4])**2)
    print(f"segment lengths before: {seg_len(tracks)}")
    print(f"segment lengths after:  {seg_len(out)}")
