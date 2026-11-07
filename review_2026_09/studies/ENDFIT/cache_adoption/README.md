# Per-pipeline JAX cache: adoption proposal (29 Oct; NOT applied -- Pierre's decision)

Evidence: review_2026_09/REVIEW.md "Per-pipeline cache RESULT" (KN10N3PC vs KN10N3P: FR/H first epoch 9:50 -> 3:20 /
2:15, pipeline 3:07:16 -> 2:49:27, 45 MB; CPU bitwise).

1. submit_prod.sh.patch -- export LARND_JAX_PERSISTENT_CACHE=shared and
   JAX_COMPILATION_CACHE_DIR=/sdf/scratch/neutrino/$USER/jax_job_cache/pipe_<NAME> (both overridable) before the three
   sbatch calls. This is exactly the tested path (the test passed both on the submit_prod.sh command line).
2. example_run.py.patch -- needed with 1: the mode is a LARND_* variable, so it is stored in every checkpoint and all 10
   continuation builders (CONT/make_*.py, ENDFIT/make_*.py) re-export it. Today, mode=shared without a usable dir
   DISABLES the cache (~3x slower), so e.g. a 44000-iteration long continuation of an adopted pipeline submitted by hand
   would run uncached. The patch falls back to the per-job cache (with a warning) instead. Logic test (all 7 env
   combinations, orig vs new, sweeper root redirected): only "shared + no dir" and "shared + /sdf/group dir" change
   ('' -> per-job dir); nothing is ever written under /sdf/group.

Apply: patch -p0 / cp *.new over the live files (live example_run.py == example_run.py.orig at 29 Oct 17:39).
