"""Exact treatment of the induction on neighboring pixels, for review item R089 (21 Oct 2026).

Production (larndsim.sim_jax.simulate_wfs) splits each cloud into the 5x5 transverse bins of Eq. (tran_diff_weights) only
for the pixel on which each bin lands (collection, with the longitudinal-diffusion template). The induction on the
other pixels of the 9x9 block around the cloud's main pixel uses the whole cloud as a point at its center bin and the
template without longitudinal diffusion. This module builds, from the same drift and the same pixel set, the variants

  'prod'  production (simulate_wfs itself)
  'P'     production re-assembled with the functions below (sanity check: must equal 'prod')
  'L'     production, but the center-point induction uses the longitudinal-diffusion template
  'T'     every bin induces on every pixel of the block at its own position, template without diffusion off its own pixel
  'E'     exact: every bin induces on every pixel of the block at its own position, with the diffusion template

All variants keep the production pixel set (unique main pixels; review item R091 is a separate question) and the
production footprint (9x9 block around the main pixel). Bin-pixel offsets beyond the response table (4.5 pixel pitches)
are dropped, as the table has no entry there. Everything is passed through larndsim.sim_jax.simulate_signals, the
production accumulation (sub-tick time shift, quadratic template interpolation, early-charge restoration), in chunks of
clouds. Read-only use of the repository code; nothing in it is modified."""
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from jax import jit
from larndsim.sim_jax import simulate_signals, simulate_drift_new, pad_to_closest_multiple, simulate_wfs
from larndsim.detsim_jax import get_bin_shifts, pixel2id
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.sim_jax import shift_tracks
from larndsim.consts_jax import get_vdrift


@partial(jit, static_argnames=['fields'])
def drift_geometry(params, tracks, fields):
    """The per-cloud quantities of simulate_drift_new needed to rebuild the bins: absolute sub-pixel bin index of each
    of the 5x5 bins, their electron counts, drift time t0 and longitudinal width (ticks) of the cloud, main pixel
    (pitch indices), plane and event. Same code as simulate_drift_new."""
    new_tracks = drift(params, quench(params, shift_tracks(params, tracks, fields), fields), fields)
    xi, yi, zi = fields.index('x'), fields.index('y'), fields.index('z')
    ne, pp, ev = fields.index('n_electrons'), fields.index('pixel_plane'), fields.index('eventID')
    ld, td = fields.index('long_diff'), fields.index('tran_diff')
    bins_pitches = get_bin_shifts(params, new_tracks, fields)                       # (Nc, 2) center bin
    nb = params.nb_tran_diff_bins; nbs = (nb - 1) // 2
    width = params.pixel_pitch / params.nb_sampling_bins_per_pixel
    edges = jnp.linspace(-nbs * width, (nbs + 1) * width, nb + 1)
    from larndsim.detsim_jax import density_2d
    borders = jnp.take(params.tpc_borders, new_tracks[:, pp].astype(int), axis=0)
    x0 = (new_tracks[:, xi] - borders[:, 0, 0]) % width
    y0 = (new_tracks[:, yi] - borders[:, 1, 0]) % width
    w = density_2d(edges, x0, y0, new_tracks[:, td])                                 # (Nc, 5, 5)
    nel = w * new_tracks[:, ne][:, None, None]
    z_cath = borders[..., 2, 1]
    t0 = jnp.abs(new_tracks[:, zi] - z_cath) / get_vdrift(params)
    long_diff = new_tracks[:, ld] / get_vdrift(params) / params.t_sampling
    shifts = jnp.mgrid[-nbs:nbs + 1, -nbs:nbs + 1]
    B = jnp.moveaxis(bins_pitches[..., None, None] + shifts, 1, -1)                   # (Nc, 5, 5, 2)
    main = B[:, nbs, nbs, :] // params.nb_sampling_bins_per_pixel                       # (Nc, 2)
    return dict(B=B.reshape(-1, nb * nb, 2), nel=nel.reshape(-1, nb * nb), t0=t0, long_diff=long_diff, main=main,
                Bc=bins_pitches, n=new_tracks[:, ne], plane=new_tracks[:, pp].astype(int), event=new_tracks[:, ev].astype(int))


def _entries_bins(params, g, sel, Nx, sigma_off):
    """Family A: every bin of the clouds `sel` on every pixel of the block. sigma_off: 'diff' (long diffusion off its own
    pixel too) or 'none' (template without diffusion off its own pixel). Returns flat main-path inputs."""
    n_sub = params.nb_sampling_bins_per_pixel; nn = params.number_pix_neighbors
    grid = jnp.moveaxis(jnp.mgrid[-nn:nn + 1, -nn:nn + 1], 0, -1).reshape(-1, 2)       # (81, 2)
    B = g['B'][sel]; P = g['main'][sel][:, None, :] + grid[None]                         # (nc,25,2), (nc,81,2)
    off = B[:, :, None, :] - P[:, None, :, :] * n_sub - n_sub // 2 + 0.5                 # (nc,25,81,2)
    idx = jnp.abs(off).astype(int)
    valid = jnp.all(idx <= Nx - 1, axis=-1)
    own = jnp.all((B[:, :, None, :] // n_sub) == P[:, None, :, :], axis=-1)
    pid = pixel2id(params, P[..., 0], P[..., 1], g['plane'][sel][:, None], g['event'][sel][:, None])   # (nc,81)
    pid = jnp.broadcast_to(pid[:, None, :], idx.shape[:-1])
    nel = jnp.where(valid, g['nel'][sel][:, :, None], 0.)
    nel = jnp.broadcast_to(nel, idx.shape[:-1])
    t0 = jnp.broadcast_to(g['t0'][sel][:, None, None], idx.shape[:-1])
    ld = jnp.broadcast_to(g['long_diff'][sel][:, None, None], idx.shape[:-1])
    if sigma_off == 'none':
        ld = jnp.where(own, ld, params.long_diff_template[0])
    idx = jnp.clip(idx, 0, Nx - 1)
    return pid.ravel(), t0.ravel(), nel.ravel(), ld.ravel(), idx.reshape(-1, 2)


def _entries_center(params, g, sel, Nx, with_diff):
    """Family B: the whole cloud at its center bin, on every pixel of the block except the main one (production
    neighbor path), with or without the longitudinal-diffusion template."""
    n_sub = params.nb_sampling_bins_per_pixel; nn = params.number_pix_neighbors
    grid = jnp.moveaxis(jnp.mgrid[-nn:nn + 1, -nn:nn + 1], 0, -1).reshape(-1, 2)
    Bc = g['Bc'][sel]; P = g['main'][sel][:, None, :] + grid[None]                       # (nc,81,2)
    off = Bc[:, None, :] - P * n_sub - n_sub // 2 + 0.5
    idx = jnp.abs(off).astype(int)
    notmain = jnp.any(grid != 0, axis=-1)[None, :]
    pid = pixel2id(params, P[..., 0], P[..., 1], g['plane'][sel][:, None], g['event'][sel][:, None])
    nel = jnp.where(notmain, g['n'][sel][:, None], 0.)
    t0 = jnp.broadcast_to(g['t0'][sel][:, None], idx.shape[:-1])
    ld = jnp.broadcast_to(g['long_diff'][sel][:, None], idx.shape[:-1])
    if not with_diff:
        ld = jnp.full_like(ld, params.long_diff_template[0])
    idx = jnp.clip(idx, 0, Nx - 1)
    return pid.ravel(), t0.ravel(), nel.ravel(), ld.ravel(), idx.reshape(-1, 2)


def _entries_own(params, g, sel):
    """Production main path: each bin on its own pixel, with the diffusion template."""
    n_sub = params.nb_sampling_bins_per_pixel
    B = g['B'][sel]; own = B // n_sub
    idx = jnp.abs(B % n_sub - n_sub // 2 + 0.5).astype(int)
    pid = pixel2id(params, own[..., 0], own[..., 1], g['plane'][sel][:, None], g['event'][sel][:, None])
    t0 = jnp.broadcast_to(g['t0'][sel][:, None], idx.shape[:-1])
    ld = jnp.broadcast_to(g['long_diff'][sel][:, None], idx.shape[:-1])
    return pid.ravel(), t0.ravel(), g['nel'][sel].ravel(), ld.ravel(), idx.reshape(-1, 2)


@partial(jit, static_argnames=['kind'])
def _chunk_wfs(params, response, unique_pixels, g, sel, kind):
    Nx = response.shape[1]
    parts = []
    if kind in ('P', 'L'):
        parts.append(_entries_own(params, g, sel))
        parts.append(_entries_center(params, g, sel, Nx, with_diff=(kind == 'L')))
    elif kind == 'T':
        parts.append(_entries_bins(params, g, sel, Nx, 'none'))
    elif kind == 'E':
        parts.append(_entries_bins(params, g, sel, Nx, 'diff'))
    pid, t0, nel, ld, idx = (jnp.concatenate(x) for x in zip(*parts))
    z1 = jnp.zeros((1,)); nnb = (2 * params.number_pix_neighbors + 1) ** 2
    return simulate_signals(params, unique_pixels, pid, t0, response, nel, ld, idx,
                            z1, jnp.zeros((nnb,), dtype=int), z1, jnp.zeros((nnb, 2), dtype=int))


def simulate_wfs_variant(params, response, tracks, fields, kind, chunk=256):
    """Waveforms (Npixels, Nticks-1) and unique pixels for one of the variants above. Same pixel set as production."""
    wfs_prod, up = simulate_wfs(params, response, tracks, fields)
    if kind == 'prod':
        return wfs_prod, up
    g = drift_geometry(params, tracks, fields)
    Nc = g['t0'].shape[0]
    pad = (-Nc) % chunk
    gp = {k: jnp.concatenate([v, jnp.zeros((pad,) + v.shape[1:], v.dtype)]) for k, v in g.items()}
    wfs = None
    for s in range(0, Nc + pad, chunk):
        sel = jnp.arange(s, s + chunk)
        w = _chunk_wfs(params, response, up, gp, sel, kind)
        wfs = w if wfs is None else wfs + w
    wfs = wfs[:, 1:]
    # rows of the padding pixel (-1) collect the out-of-set contributions in production: keep them identical, so that the
    # stochastic electronics draw the same noise on every variant (common random numbers)
    wfs = jnp.where((up >= 0)[:, None], wfs, wfs_prod)
    return wfs, up
