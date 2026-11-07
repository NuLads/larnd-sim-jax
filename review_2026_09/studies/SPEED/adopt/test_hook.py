# Test the example_run LRU hook block in isolation: exec it, import jax, check the maxsize of JAX's own caches.
import os, sys, re
src = open('/sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/adopt/example_run.new.py').read()
block = src[src.index('_JAX_LRU = int('):src.index('    sys.meta_path.insert(0, _LRUFinder())') + len('    sys.meta_path.insert(0, _LRUFinder())')]
exec(block)
import jax, jax._src.interpreters.partial_eval as pe, jax._src.pjit as pj
print('LARND_JAX_LRU', os.environ.get('LARND_JAX_LRU'), '| _dce_jaxpr maxsize', pe._dce_jaxpr.cache_info().maxsize,
      '| _pjit_lower_cached maxsize', pj._pjit_lower_cached.cache_info().maxsize if hasattr(pj, '_pjit_lower_cached') else '?')
print(jax.jit(lambda x: x * 2)(3.0))
