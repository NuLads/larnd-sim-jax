"""Shared toolkit for the talk animations (matplotlib frames piped straight into ffmpeg/libx264).

Every animation script defines two functions and calls render():

    from common import *
    args = parse_args()                # --preview 3,12.5  --draft  --final  --theme dark|light  --nproc 12

    def setup():                       # runs once per worker: build the figure + artists, return a state object
        fig = new_figure()
        ax = fig.add_axes(box_px(100, 120, 1720, 800))
        style_axes(ax)
        return dict(fig=fig, ax=ax, ...)

    def draw(s, i, t):                 # PURE function of the frame index: frames are rendered out of order in parallel
        ...                            # update artists from i / t only; never accumulate state between calls

    render(args, 'a05_random_to_differentiable', duration=30, setup=setup, draw=draw, beats=[(0, 'intro'), ...])

Modes: --preview t1,t2 writes 1920x1080 PNGs to preview/<name>/; --draft writes a 960x540 15 fps MP4 to preview/<name>/;
--final writes the 3840x2160 30 fps H.264 MP4 + poster PNG + beats JSON to output/. Without a mode flag --final is assumed.
Figure geometry is always 1920x1080 "design pixels": use px() for sizes and box_px() for axes, whatever the output scale."""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba

ROOT = '/sdf/scratch/neutrino/pgranger/talk_animations'
FFMPEG = f'{ROOT}/tools/ffmpeg'
DATA = f'{ROOT}/data'
W, H = 1920, 1080                       # design pixels
FPS = 30

for _f in sorted(os.listdir(f'{ROOT}/tools/fonts')):
    if _f.endswith('.ttf'):
        fm.fontManager.addfont(f'{ROOT}/tools/fonts/{_f}')
FONT = 'Source Sans 3'


class Tok(dict):
    __getattr__ = dict.__getitem__


# Colour has one meaning across the whole talk:
#   data = orange, model = blue, truth = primary ink (dashed), start/guess = grey.
# The five calibration parameters keep their own hue in every animation.
THEMES = dict(
    dark=Tok(bg='#111214', panel='#191a1d', grid='#2b2d31', spine='#4a4c52', ink='#f0efec', ink2='#c3c2b7', ink3='#8b8a82',
             data='#f08a4b', model='#5aa2f0', truth='#f0efec', start='#8b8a82', good='#2fbf8f', bad='#e66767',
             Ab='#2fbf8f', eField='#e0a526', tran_diff='#e56b98', long_diff='#a79cf0', lifetime='#e66767'),
    light=Tok(bg='#fcfcfb', panel='#f4f3f0', grid='#e4e2dc', spine='#b5b3ac', ink='#0b0b0b', ink2='#52514e', ink3='#7d7c76',
              data='#d9601f', model='#2a78d6', truth='#0b0b0b', start='#7d7c76', good='#0f8a5f', bad='#c93a39',
              Ab='#0f8a5f', eField='#b67d00', tran_diff='#c93f78', long_diff='#6250d6', lifetime='#c93a39'),
)
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PARAM_LABEL = dict(Ab=r'$A_b$', eField=r'$E$', tran_diff=r'$D_T$', long_diff=r'$D_L$', lifetime=r'$\tau$')
PARAM_NAME = dict(Ab='recombination', eField='electric field', tran_diff='transverse diffusion',
                  long_diff='longitudinal diffusion', lifetime='electron lifetime')
C = Tok(THEMES['dark'])


def apply_theme(name='dark'):
    C.clear(); C.update(THEMES[name])
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': FONT, 'font.size': px(26), 'text.color': C.ink, 'axes.labelcolor': C.ink2, 'axes.edgecolor': C.spine,
        'axes.facecolor': 'none', 'figure.facecolor': C.bg, 'savefig.facecolor': C.bg, 'xtick.color': C.ink3,
        'ytick.color': C.ink3, 'xtick.labelsize': px(22), 'ytick.labelsize': px(22), 'axes.labelsize': px(26),
        'axes.linewidth': px(1.5), 'xtick.major.width': px(1.5), 'ytick.major.width': px(1.5), 'xtick.major.size': px(7),
        'ytick.major.size': px(7), 'lines.linewidth': px(3), 'lines.solid_capstyle': 'round', 'lines.antialiased': True,
        'axes.grid': False, 'grid.color': C.grid, 'grid.linewidth': px(1.2), 'legend.frameon': False, 'legend.fontsize': px(22),
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold',
        'mathtext.sf': FONT, 'mathtext.cal': FONT, 'mathtext.fallback': 'stixsans', 'axes.unicode_minus': True,
        'path.simplify': False, 'axes.axisbelow': True,
    })


def px(n):
    """Design pixels -> points (the figure is 19.2x10.8 in at 100 design dpi)."""
    return n * 72.0 / 100.0


def box_px(x, y, w, h):
    """Axes rectangle from design pixels, origin BOTTOM-left, for fig.add_axes()."""
    return [x / W, y / H, w / W, h / H]


def new_figure(scale=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * (scale if scale is not None else _STATE['scale']))
    fig.patch.set_facecolor(C.bg)
    return fig


def style_axes(ax, grid='y', spines=('left', 'bottom')):
    for k, sp in ax.spines.items():
        sp.set_visible(k in spines)
    if grid:
        ax.grid(True, axis=grid if grid in ('x', 'y') else 'both', color=C.grid, lw=px(1.2))
    ax.tick_params(colors=C.ink3)
    return ax


def title(fig, text, sub=None):
    """Top-left heading. Returns the text artists so they can be changed per frame."""
    a = fig.text(60 / W, 1012 / H, text, fontsize=px(46), fontweight='semibold', color=C.ink, va='baseline')
    b = fig.text(60 / W, 968 / H, sub or '', fontsize=px(28), color=C.ink2, va='baseline')
    return a, b


def footer(fig, text='Simulated data: noisy simulation at known truth', right=None):
    """Small bottom strip: every animation states what the data is. `right` names the source run or file."""
    a = fig.text(60 / W, 26 / H, text, fontsize=px(20), color=C.ink3, va='baseline')
    b = fig.text(1860 / W, 26 / H, right or '', fontsize=px(20), color=C.ink3, va='baseline', ha='right')
    return a, b


def clamp(x, a=0.0, b=1.0):
    return np.minimum(np.maximum(x, a), b)


def seg(t, t0, t1):
    """Linear progress 0->1 of t across [t0, t1], clamped."""
    return float(clamp((t - t0) / max(t1 - t0, 1e-9)))


def ease(x):
    """Smooth in-out (cubic smoothstep) on a progress value in [0, 1]."""
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def lerp(a, b, x):
    return a + (b - a) * x


def fade(t, t0, t1, dur=0.5):
    """Opacity envelope: fades in at t0, out at t1 (use t1=inf to stay on)."""
    return float(ease(seg(t, t0, t0 + dur)) * (1 - ease(seg(t, t1, t1 + dur))))


def cmap_to(color, name='seq'):
    """Sequential map, one hue: surface -> colour -> slightly lighter tint."""
    rgb = np.array(to_rgb(color)); bg = np.array(to_rgb(C.bg)); top = rgb + (1 - rgb) * 0.45
    return LinearSegmentedColormap.from_list(name, [bg * 0.6 + rgb * 0.4 * 0.35, rgb, top])


def cmap_diverging():
    """Data excess = orange, model excess = blue, neutral midpoint."""
    return LinearSegmentedColormap.from_list('div', [C.model, C.panel if C.bg == THEMES['dark'].bg else '#eeeeea', C.data])


def caption(fig_or_ax, *a, **k):
    raise NotImplementedError('use fig.text directly')


def param_err(x, target):
    return 100.0 * (np.asarray(x, float) / target - 1.0)


# ---- render driver ---------------------------------------------------------------------------------------------------
_STATE = dict(scale=1.0)


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', default=None, help='comma list of times (s): write 1080p PNGs')
    ap.add_argument('--draft', action='store_true', help='960x540, 15 fps MP4 in preview/')
    ap.add_argument('--final', action='store_true', help='3840x2160, 30 fps MP4 + poster in output/')
    ap.add_argument('--theme', default='dark', choices=['dark', 'light'])
    ap.add_argument('--nproc', type=int, default=12)
    if extra:
        extra(ap)
    a = ap.parse_args()
    apply_theme(a.theme)
    return a


def _enc_cmd(w, h, fps, out, crf):
    return [FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
            '-profile:v', 'high', '-threads', '2', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
            '-color_range', 'tv', '-movflags', '+faststart', '-r', str(fps), out]


def _frame_bytes(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).tobytes()


def _worker(job):
    scale, theme, fps, n, lo, hi, out, crf, setup, draw = job
    apply_theme(theme); _STATE['scale'] = scale
    st = setup(); fig = st['fig']
    w, h = int(round(W * scale)), int(round(H * scale))
    p = subprocess.Popen(_enc_cmd(w, h, fps, out, crf), stdin=subprocess.PIPE)
    for i in range(lo, hi):
        draw(st, i, i / fps)
        p.stdin.write(_frame_bytes(fig))
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    plt.close(fig)
    return out


_JOB = {}


def _run(args):
    return _worker(args + (_JOB['setup'], _JOB['draw']))


def render(args, name, duration, setup, draw, beats=None, fps=None):
    """duration is in seconds and includes any end hold. Keep at least 2 s of static end hold for Keynote."""
    t0 = time.time()
    pdir = f'{ROOT}/preview/{name}'; os.makedirs(pdir, exist_ok=True); os.makedirs(f'{ROOT}/output', exist_ok=True)
    if args.preview:
        _STATE['scale'] = 1.0
        st = setup(); fig = st['fig']
        for s in args.preview.split(','):
            t = float(s); i = int(round(t * FPS))
            draw(st, i, i / FPS)
            fig.savefig(f'{pdir}/t{t:06.2f}.png', dpi=100)
            print(f'{pdir}/t{t:06.2f}.png', flush=True)
        return
    final = args.final or not args.draft
    scale = 2.0 if final else 0.5
    fps = fps or (FPS if final else 15)
    crf = 17 if final else 24
    n = int(round(duration * fps))
    tmp = tempfile.mkdtemp(prefix=f'{name}_', dir=f'{ROOT}/tmp' if os.path.isdir(f'{ROOT}/tmp') else None)
    nw = min(args.nproc, max(1, n // 8))
    edges = np.linspace(0, n, nw + 1).astype(int)
    jobs = [(scale, args.theme, fps, n, int(edges[k]), int(edges[k + 1]), f'{tmp}/seg{k:03d}.mp4', crf) for k in range(nw)]
    _JOB.update(setup=setup, draw=draw)
    with mp.get_context('fork').Pool(nw) as pool:
        segs = pool.map(_run, jobs, chunksize=1)
    lst = f'{tmp}/list.txt'
    with open(lst, 'w') as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    suffix = '' if args.theme == 'dark' else '_light'
    out = f'{ROOT}/output/{name}{suffix}.mp4' if final else f'{pdir}/{name}_draft{suffix}.mp4'
    subprocess.check_call([FFMPEG, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', '-movflags', '+faststart', out])
    shutil.rmtree(tmp, ignore_errors=True)
    if final:
        apply_theme(args.theme); _STATE['scale'] = 1.0
        st = setup(); draw(st, n - 1, (n - 1) / fps)
        st['fig'].savefig(f'{ROOT}/output/{name}{suffix}_poster.png', dpi=100)
        if beats:
            json.dump([dict(t=float(a), label=b) for a, b in beats], open(f'{ROOT}/output/{name}_beats.json', 'w'), indent=1)
    print(f'{out}: {n} frames, {os.path.getsize(out) / 1e6:.1f} MB, {time.time() - t0:.0f} s', flush=True)
