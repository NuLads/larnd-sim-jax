"""28 Oct batch-size probe (Pierre: "probe again the memory and speed for different batch sizes now that we optimized the
code. Maybe making bigger batches might help"): the production 1e4 fit (KNOT/inner_sci_full_KN10N1.sh, live tree) with
--max_batch_len LEN and --max_nbatch NB (same total track length, NB x LEN = 40000 cm), 5 epochs, checkpoint at the end
(training_history step_time / memory per iteration). XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 so the pool is not the limit.
usage: python3 make_bs.py LEN NB > inner.sh"""
import sys, shlex
L, NB = int(sys.argv[1]), int(sys.argv[2]); it = 5 * NB
src = open('/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/inner_sci_full_KN10N1.sh').read().splitlines()
out = []
for line in src:
    if line.startswith('python3 -m optimize.example_run'):
        a = shlex.split(line)
        for k, v in (('--iterations', it), ('--max_nbatch', NB), ('--max_batch_len', L), ('--save_freq', it),
                     ('--test_name', f'sci_full_BS{L}'), ('--out_label', f'sci_full_b{NB}_len{L}_mw2500.0_refit_seed0')):
            i = a.index(k); a[i + 1] = str(v)
        out.append('export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95')
        line = ' '.join(shlex.quote(x) for x in a)
    out.append(line)
print('\n'.join(out))
