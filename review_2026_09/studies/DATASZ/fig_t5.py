"""DATASZ T5 figure (descriptive; numbers = T5_result.txt, read_ds.py t5 of 4 Nov 09:17): spread, bias and sigma_E1 vs data volume."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
m = np.array([100, 199, 400, 799]); n = np.array([8, 8, 12, 8])          # track length used (m), samples per arm
mean = np.array([[-0.081, 0.135, 1.562, 6.043, 5.733], [-0.073, 0.118, 0.671, 3.634, 4.258],
                 [-0.094, 0.127, 0.633, 4.615, 2.841], [0.093, 0.128, 1.191, 5.497, 0.080]])
sem = np.array([[0.119, 0.031, 0.908, 6.125, 3.534], [0.042, 0.029, 0.491, 2.784, 1.814],
                [0.058, 0.017, 0.385, 1.992, 1.479], [0.027, 0.008, 0.551, 1.483, 0.793]])
sd = np.array([[0.336, 0.088, 2.567, 17.324, 9.995], [0.119, 0.083, 1.389, 7.874, 5.130],
               [0.202, 0.061, 1.335, 6.902, 5.123], [0.077, 0.022, 1.559, 4.195, 2.243]])
e1 = np.array([[0.06035, 0.0008585, 0.7837, 1.6, 1.625], [0.04336, 0.0006123, 0.5199, 1.108, 1.094],
               [0.03054, 0.0004419, 0.3882, 0.8167, 0.7959], [0.02163, 0.0003136, 0.2751, 0.5606, 0.5692]])
alpha = ['0.53 [0.19, 0.86]', '0.63 [0.30, 0.96]', '0.23 [-0.10, 0.56]', '0.63 [0.29, 0.96]', '0.63 [0.29, 0.96]']
fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
for j, p in enumerate(P):
    a = ax[j]
    a.errorbar(m, sd[:, j], yerr=sd[:, j] / np.sqrt(2 * (n - 1)), fmt='o-', c='C0', label='spread (SD over samples)')
    a.errorbar(m * 1.03, np.abs(mean[:, j]), yerr=sem[:, j], fmt='s--', c='C3', label='|bias| (mean error)')
    a.plot(m, e1[:, j], '^:', c='C2', label='Hessian sigma (E1)')
    a.plot(m, sd[2, j] * (m / 400) ** -0.5, c='0.6', lw=1, label='1/sqrt(N) through 400 m')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(m); a.set_xticklabels([f'{x}' for x in m])
    a.set_title(f'{p}   alpha = {alpha[j]}', fontsize=10); a.set_xlabel('track length in the fit (m)')
    if j == 0: a.set_ylabel('error (%)'); a.legend(fontsize=7)
fig.suptitle('DATASZ T5: calibration error vs data volume (8 samples per arm, 12 at 400 m = production); ~215 / 430 / 880 / 1770 tracks', fontsize=11)
fig.tight_layout(); fig.savefig('fig_t5.png', dpi=110); print('fig_t5.png')
