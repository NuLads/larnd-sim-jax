cd /sdf/group/neutrino/pgranger/larnd-sim-jax
export PYTHONPATH=$PWD/src:$PWD
echo "inner sees LRU=$LARND_JAX_LRU UB=$LARND_UNIQUE_BUCKET PV=$LARND_POSRES_VEC"
python3 /sdf/scratch/neutrino/pgranger/review_2026_09/SPEED/adopt/test_hook_live.py 2>&1 | grep -E "LARND_JAX_LRU"
