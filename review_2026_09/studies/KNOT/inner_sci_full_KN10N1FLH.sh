# ENDFIT (m) FLH: built at job start from KN10N1FL @ 10000 (make_sched.py --iters 4000 --hold 1500), then run.
CK=$(ls /sdf/scratch/neutrino/pgranger/fit_result_staging/sci_full_KN10N1FL/history_iter10000_*.pkl) || { echo "no 10000 checkpoint for KN10N1FL"; exit 1; }
python3 /sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/make_sched.py $CK sci_full_KN10N1FLH --iters 4000 --hold 1500 > /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_KN10N1FLH.built.sh || exit 1
cat /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_KN10N1FLH.built.sh
bash /sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_KN10N1FLH.built.sh
