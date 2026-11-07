"""Speed bench only (20 Oct): enlarge JAX's in-memory weakref LRU caches (trace / jvp / lowering / compilation; default
maxsize 2048) when LARND_JAX_LRU=N is set. jax._src.util is loaded through an import hook that rewrites the default before
any jax module applies the decorator; nothing in site-packages is modified. Put this directory first on PYTHONPATH."""
import os, sys
_N = int(os.environ.get('LARND_JAX_LRU', '0') or 0)
if _N:
    import importlib.abc, importlib.machinery
    _OLD = 'def weakref_lru_cache(call: Callable, maxsize=2048,'

    class _Loader(importlib.abc.Loader):
        def __init__(self, origin): self.origin = origin
        def create_module(self, spec): return None
        def exec_module(self, module):
            src = open(self.origin).read(); assert _OLD in src, 'jax._src.util changed: patch does not apply'
            exec(compile(src.replace(_OLD, f'def weakref_lru_cache(call: Callable, maxsize={_N},'), self.origin, 'exec'), module.__dict__)
            print(f'[LARND_JAX_LRU] weakref_lru_cache default maxsize 2048 -> {_N}', file=sys.stderr, flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != 'jax._src.util': return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None: spec.loader = _Loader(spec.origin)
            return spec
    sys.meta_path.insert(0, _Finder())

_LOG = int(os.environ.get('LARND_JAX_LRU_LOG', '0') or 0)
if _N and _LOG:                                   # every _LOG s: fill and hit counts of all weakref LRU caches
    import threading, time

    def _report():
        while True:
            time.sleep(_LOG)
            try:
                import jax._src.util as u
                infos = [c.cache_info() for c in list(u._weakref_lru_caches)]
                full = [i for i in infos if i.currsize >= i.maxsize]
                print(f'[LARND_JAX_LRU] t={time.time():.0f} caches {len(infos)} entries {sum(i.currsize for i in infos)} '
                      f'at-capacity {len(full)} (entries {sum(i.currsize for i in full)}) hits {sum(i.hits for i in infos)} '
                      f'misses {sum(i.misses for i in infos)} misses-in-full {sum(i.misses for i in full)}', file=sys.stderr, flush=True)
            except Exception as e:
                print(f'[LARND_JAX_LRU] report failed {e!r}', file=sys.stderr, flush=True)
    threading.Thread(target=_report, daemon=True).start()
