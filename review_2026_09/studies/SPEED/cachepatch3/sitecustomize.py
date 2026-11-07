"""Speed bench round 3 only (20 Oct): as cachepatch/ (enlarge JAX's in-memory weakref LRU caches to LARND_JAX_LRU=N; the
jax._src.util source is rewritten by an import hook, site-packages untouched), plus, every LARND_JAX_LRU_LOG s: process RSS,
cumulative time in Python's cyclic GC by generation, and the per-cache size / hits / misses of the 6 caches with the most
misses since the last report, named by the cached function. Put this directory first on PYTHONPATH."""
import os, sys, time
_N = int(os.environ.get('LARND_JAX_LRU', '0') or 0)
if _N:
    import importlib.abc, importlib.machinery
    _OLD = 'def weakref_lru_cache(call: Callable, maxsize=2048,'
    _ADD = '  _weakref_lru_caches.add(cached_call)\n'

    class _Loader(importlib.abc.Loader):
        def __init__(self, origin): self.origin = origin
        def create_module(self, spec): return None
        def exec_module(self, module):
            src = open(self.origin).read(); assert _OLD in src and src.count(_ADD) == 1, 'jax._src.util changed: patch does not apply'
            src = src.replace(_OLD, f'def weakref_lru_cache(call: Callable, maxsize={_N},').replace(
                _ADD, _ADD + "  _larnd_names[id(cached_call)] = getattr(call, '__module__', '?') + '.' + getattr(call, '__qualname__', repr(call))\n")
            module.__dict__['_larnd_names'] = {}
            exec(compile(src, self.origin, 'exec'), module.__dict__)
            print(f'[LARND_JAX_LRU] weakref_lru_cache default maxsize 2048 -> {_N}', file=sys.stderr, flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != 'jax._src.util': return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None: spec.loader = _Loader(spec.origin)
            return spec
    sys.meta_path.insert(0, _Finder())

_LOG = int(os.environ.get('LARND_JAX_LRU_LOG', '0') or 0)
if _N and _LOG:
    import threading, gc
    _gc = {'t': [0.0, 0.0, 0.0], 'n': [0, 0, 0], 't0': None}

    def _gccb(phase, info):
        if phase == 'start': _gc['t0'] = time.perf_counter()
        elif _gc['t0'] is not None:
            g = info['generation']; _gc['t'][g] += time.perf_counter() - _gc['t0']; _gc['n'][g] += 1
    gc.callbacks.append(_gccb)

    def _rss_gb():
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'): return int(line.split()[1]) / 2**20
        return float('nan')

    def _report():
        prev = {}
        while True:
            time.sleep(_LOG)
            try:
                import jax._src.util as u
                cs = list(u._weakref_lru_caches); infos = [(c, c.cache_info()) for c in cs]
                full = [i for _, i in infos if i.currsize >= i.maxsize]
                print(f'[LARND_JAX_LRU] t={time.time():.0f} rss_gb {_rss_gb():.2f} gc_s {_gc["t"][0]:.0f}/{_gc["t"][1]:.0f}/{_gc["t"][2]:.0f} '
                      f'gc_n {_gc["n"][0]}/{_gc["n"][1]}/{_gc["n"][2]} caches {len(infos)} entries {sum(i.currsize for _, i in infos)} '
                      f'at-capacity {len(full)} hits {sum(i.hits for _, i in infos)} misses {sum(i.misses for _, i in infos)} '
                      f'misses-in-full {sum(i.misses for i in full)}', file=sys.stderr, flush=True)
                dm = sorted(((i.misses - prev.get(id(c), (0, 0))[1], c, i) for c, i in infos), key=lambda x: -x[0])[:6]
                for d, c, i in dm:
                    if d > 0:
                        print(f'[LARND_JAX_LRU]   {u._larnd_names.get(id(c), "?")}: size {i.currsize}/{i.maxsize} '
                              f'+hits {i.hits - prev.get(id(c), (0, 0))[0]} +misses {d}', file=sys.stderr, flush=True)
                prev = {id(c): (i.hits, i.misses) for c, i in infos}
            except Exception as e:
                print(f'[LARND_JAX_LRU] report failed {e!r}', file=sys.stderr, flush=True)
    threading.Thread(target=_report, daemon=True).start()
