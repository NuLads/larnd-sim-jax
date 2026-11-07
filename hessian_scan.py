"""sigma(param) vs number of batches, + inversion-conditioning check.

Answers two reviewer questions:
  (1) is the quoted sigma(lifetime)=1.61% a true floor, or does it fall as 1/sqrt(N)?
      -> data-limited vs information-limited.
  (2) how sensitive are the quoted sigmas (and the 119x eField inflation) to the
      float32 assembly / ridge regularisation, given cond(H) ~ 4e6?

NOTE: evaluates at truth via map_phys_to_norm (the CURRENT sigmoid parametrization).
An earlier analysis used log(truth/init), which was correct only for the older
unbounded-log parametrization and would place the evaluation point ~38% off in lifetime.
"""
import sys, numpy as np
NB = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [5, 10, 20, 40]
MAXB = max(NB)
sys.argv = ['x', '--history',
            'fit_result/sci_ceiling_noise_s2/history_iter5000_sci_ceiling_b100_len400_seed0.pkl',
            '--objective', 'ppp', '--steps', '0', '--max_batches', str(MAXB)]
src = open('optimize/prototype_gn_calib.py').read()
src = src.replace("for s in range(args.steps):", """
    from optimize.fit_params import map_phys_to_norm as _p2n
    th_truth = jnp.asarray([float(_p2n(truth[p], p, scheme='sigmoid', scale=1.0)) for p in relevant])
    print('[SCAN] theta_truth =', np.asarray(th_truth))
    print('[SCAN] round-trip check (phys from theta vs truth):',
          {p: f"{float(map_norm_to_phys(th_truth[k], p, scheme='sigmoid', scale=1.0))/truth[p]-1:+.2e}"
           for k, p in enumerate(relevant)})
    Hs = np.zeros((P, P)); gs = np.zeros(P)
    NBLIST = __NBLIST__
    for i in range(nb):
        Hs += np.asarray(jac_fns[i](th_truth)); gs += np.asarray(res_fns[i](th_truth))
        n = i + 1
        if n in NBLIST:
            ev = np.linalg.eigvalsh(Hs)
            cond = ev.max() / max(ev.min(), 1e-30)
            out = [f'N={n:3d} cond={cond:.3e} |grad|={np.linalg.norm(gs):.4g}']
            for ridge in [0.0, 1e-9, 1e-6]:
                try:
                    Hi = np.linalg.inv(Hs.astype(np.float64) + ridge * np.eye(P))
                    sig = {p: np.sqrt(max(Hi[j, j], 0)) * 100 for j, p in enumerate(relevant)}
                    out.append('  ridge=%-6g ' % ridge + ' '.join(f'{p}={sig[p]:.3f}%' for p in relevant))
                except Exception as e:
                    out.append(f'  ridge={ridge} FAILED {e}')
            print('[SCAN] ' + '\\n[SCAN] '.join(out), flush=True)
            print('[SCAN] N=%d eigenvalues: ' % n + ' '.join(f'{v:.4e}' for v in ev), flush=True)
    raise SystemExit
    for s in range(args.steps):""".replace('__NBLIST__', repr(NB)))
exec(compile(src, 'scan', 'exec'))
