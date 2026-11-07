# ENDFIT (g), 25 Oct: long continuation of sci_full_EXP1FR @ 14000 -> 44000, built at job start with
# ENDFIT/make_long_cont.py (the (e) / (f) tool; runs inside the container), then run.
CK=$(ls /sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_EXP1FR/history_iter14000_*.pkl) || { echo "no 14000 checkpoint"; exit 1; }
B=/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_EXP1L.built.sh
python3 /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/make_long_cont.py $CK sci_full_EXP1L > $B || exit 1
cat $B
bash $B
