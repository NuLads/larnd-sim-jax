"""Helpers shared by a09/a10/a11 only (fit traces, learning-rate schedule, min/max decimation, fading axis ticks)."""
import glob
import numpy as np
import optax
from common import *

_TR = np.load(f'{DATA}/traces.npz', allow_pickle=True)
NAMES = [str(n) for n in _TR['names']]
EXP = np.array([i for i, n in enumerate(NAMES) if n.startswith('EXP') and n.endswith('C75H')])
KN = np.array([i for i, n in enumerate(NAMES) if n.startswith('KN10N') and n.endswith('C75H')])
assert len(EXP) == 36 and len(KN) == 12
SAMPLE = _TR['data_seed']
TARGET = {p: float(_TR[p + '_target'][0]) for p in PARAMS}
INIT = {p: float(_TR[p + '_init'][0]) for p in PARAMS}
for _p in PARAMS:   # every run shares the same truth and the same start
    assert np.ptp(_TR[_p + '_target']) == 0 and np.ptp(_TR[_p + '_init']) == 0
START = {p: 100 * (INIT[p] / TARGET[p] - 1) for p in PARAMS}


def errs(p, idx=EXP):
    """% from truth, runs x 20001."""
    return (100 * (_TR[p][idx] / TARGET[p] - 1)).astype(np.float32)


def endpoint(p, idx=EXP):
    """REVIEW convention (KNOT/read_ens.py): mean of the last 1000 iterations / target - 1, in %."""
    return 100 * (_TR[p][idx][:, -1000:].mean(1) / TARGET[p] - 1)


def loss_cycles(idx=EXP):
    """Loss per batch averaged over each full 100-batch cycle: runs x 200; the x position is the cycle's last iteration."""
    L = _TR['losses_iter'][idx]
    return L.reshape(len(idx), 200, 100).mean(2), np.arange(1, 201) * 100


def ensemble(v):
    v = np.asarray(v, float)
    return dict(mean=v.mean(), sd=v.std(ddof=1), se=v.std(ddof=1) / np.sqrt(len(v)), lo=v.min(), hi=v.max(), n=len(v))


# Learning rates, from the same optax schedules as the production commands (verified against KNOT/inner_sci_full_EXP1C75{,FR,H}.sh
# and the optimiser counts 10000/14000/20000 in the checkpoints). The schedule count is restored on resume, so it keeps running.
def calib_lr():
    it = np.arange(20001)
    kw = dict(init_value=0., warmup_steps=500, transition_steps=100, staircase=True)
    s1 = optax.warmup_exponential_decay_schedule(peak_value=1e-1, decay_rate=0.91, **kw)
    s2 = optax.warmup_exponential_decay_schedule(peak_value=1e-3, decay_rate=0.805, transition_begin=11500, **kw)
    s3 = optax.warmup_exponential_decay_schedule(peak_value=1e-2, decay_rate=0.805, transition_begin=16000, **kw)
    return it, np.where(it < 10000, np.asarray(s1(it)), np.where(it < 14000, np.asarray(s2(it)), np.asarray(s3(it)))).astype(float)


class Pyramid:
    """Min/max decimation: keeps both extremes of every bin, in time order, so no swing is lost when bins are sub-pixel."""

    def __init__(self, Y, x=None, levels=7):
        Y = np.atleast_2d(np.asarray(Y, np.float32)); N = Y.shape[1]
        x = np.arange(N, dtype=np.float32) if x is None else np.asarray(x, np.float32)
        self.lv = [(x, Y)]
        nr = Y.shape[0]
        for k in range(1, levels):
            b = 2 ** k; n = N // b
            S = Y[:, :n * b].reshape(nr, n, b); Xb = x[:n * b].reshape(n, b)
            i0, i1 = S.argmin(2), S.argmax(2); a, c = np.minimum(i0, i1), np.maximum(i0, i1)
            rr, cc = np.arange(nr)[:, None], np.arange(n)[None, :]
            Yk = np.stack([S[rr, cc, a], S[rr, cc, c]], 2).reshape(nr, 2 * n)
            Xk = np.stack([Xb[:, 0], Xb[:, -1]], 1).reshape(2 * n)
            self.lv.append((Xk, Yk))

    def level(self, its_per_bin):
        return self.lv[int(np.clip(np.floor(np.log2(max(its_per_bin, 1.0))), 0, len(self.lv) - 1))]


class EnvPyramid:
    """Raw ensemble envelope per level: filled polygon of the min and max inside each bin, up to the current iteration."""

    def __init__(self, lo, hi, levels=7):
        self.lv = []
        for k in range(levels):
            b = 2 ** k; n = len(lo) // b
            self.lv.append((np.arange(n) * b + b - 1.0, lo[:n * b].reshape(n, b).min(1), hi[:n * b].reshape(n, b).max(1)))

    def poly(self, its_per_bin, it):
        x, a, c = self.lv[int(np.clip(np.floor(np.log2(max(its_per_bin, 1.0))), 0, len(self.lv) - 1))]
        n = int(np.searchsorted(x, it, side='right'))
        if n < 2:
            return np.zeros((3, 2))
        return np.vstack([np.c_[x[:n], a[:n]], np.c_[x[:n][::-1], c[:n][::-1]]])


def widening_mean(Y, i0=1000, i1=3000, wmax=100):
    """Trailing mean of width 1 (raw) up to i0, widening linearly to wmax iterations (one pass over the 100 batches) by i1."""
    Y = np.asarray(Y, float); N = Y.shape[1]; it = np.arange(N)
    w = np.rint(1 + (wmax - 1) * clamp((it - i0) / (i1 - i0))).astype(int)
    S = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Y, 1)], 1)
    return ((S[:, it + 1] - S[:, it + 1 - w]) / w).astype(np.float32)


def time_map(knots):
    """iteration(t) from (t, iter) knots, eased in each segment: the fit pauses at every knot, and the iteration rate differs per segment."""
    T = np.array([k[0] for k in knots], float); I = np.array([k[1] for k in knots], float)

    def f(t):
        j = int(np.clip(np.searchsorted(T, t, side='right') - 1, 0, len(T) - 2))
        return float(lerp(I[j], I[j + 1], ease(seg(t, T[j], T[j + 1]))))
    return f


def caption_at(t, caps, dur=0.4):
    """(text, alpha) of the single caption active at t; caps = [(t_on, t_off, text), ...]."""
    for a, b, s in caps:
        if a <= t < b:
            on = 1.0 if a <= 0 else ease(seg(t, a, a + dur))      # the opening caption is already there on frame 0: beat 0 must read as a still
            return s, float(on * (1 - ease(seg(t, b - dur, b))))
    return '', 0.0


_LADDER = [m * 10.0 ** e for e in range(-4, 7) for m in (1, 2, 5)]


def fmt_pct(v):
    v = round(float(v), 6)
    return '0' if v == 0 else ('+' if v > 0 else '−') + f'{abs(v):g}%'


class SoftTicks:
    """Round-number ticks that cross-fade between 1-2-5 steps while an axis zooms. `length` is the axis length in design px."""

    def __init__(self, ax, axis, length, gap, fmt, grid=True, labels=True, n=16, offset=10, zero_grid=True, size=22):
        self.ax, self.axis, self.L, self.gap, self.fmt, self.zero_grid = ax, axis, float(length), float(gap), fmt, zero_grid
        tr = ax.get_yaxis_transform() if axis == 'y' else ax.get_xaxis_transform()
        self.lines, self.txt = [], []
        for _ in range(n):
            if grid:
                xy = ([0, 1], [0, 0]) if axis == 'y' else ([0, 0], [0, 1])
                ln, = ax.plot(*xy, transform=tr, color=C.grid, lw=px(1.2), zorder=0.5, clip_on=True); ln.set_visible(False)
                self.lines.append(ln)
            if labels:
                if axis == 'y':
                    tx = ax.annotate('', xy=(0, 0), xycoords=tr, xytext=(-px(offset), 0), textcoords='offset points', ha='right', va='center', fontsize=px(size), color=C.ink3, annotation_clip=False)
                else:
                    tx = ax.annotate('', xy=(0, 0), xycoords=tr, xytext=(0, -px(offset)), textcoords='offset points', ha='center', va='top', fontsize=px(size), color=C.ink3, annotation_clip=False)
                tx.set_visible(False); self.txt.append(tx)

    def update(self, lo, hi):
        rng = hi - lo
        sp = np.array(_LADDER) / rng * self.L
        i = int(np.argmax(sp >= self.gap))
        f, g = _LADDER[i], _LADDER[i - 1]
        a = float(ease(seg(sp[i - 1], self.gap * 0.93, self.gap)))
        vals = {}
        for step, w in ((f, 1 - a), (g, a)):
            for k in range(int(np.ceil(lo / step)), int(np.floor(hi / step)) + 1):
                v = round(k * step, 8); vals[v] = min(1.0, vals.get(v, 0.0) + w)
        edge = 14.0 / self.L * rng
        items = [(v, w * float(clamp(min(v - lo, hi - v) / edge))) for v, w in sorted(vals.items())]
        items = [(v, w) for v, w in items if w > 0.01 and (self.zero_grid or v != 0 or self.txt)]
        for j in range(max(len(self.lines), len(self.txt))):
            on = j < len(items)
            if j < len(self.lines):
                ln = self.lines[j]; ln.set_visible(on and (self.zero_grid or items[j][0] != 0))
                if on:
                    v, w = items[j]; ln.set_alpha(w)
                    (ln.set_ydata if self.axis == 'y' else ln.set_xdata)([v, v])
            if j < len(self.txt):
                tx = self.txt[j]; tx.set_visible(on)
                if on:
                    v, w = items[j]; tx.set_text(self.fmt(v)); tx.set_alpha(w); tx.xy = (0, v) if self.axis == 'y' else (v, 0)
