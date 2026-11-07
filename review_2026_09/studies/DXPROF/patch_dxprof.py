"""DXPROF (6 Nov): patch of the review copy DXPROF/tree (copied from TRUTHC/tree) -- the dE/dx-exact Schur block of the
truth-point evaluator, generalised: (1) at any point (LARND_TP_SCHUR_POINT, default P0_fit = unchanged), (2) H_cc from the
same calibration passes, (3) per dE/dx probe step (LARND_TP_SCHUR_DSTEPS) and ridge (LARND_TP_SCHUR_LAMS) the curvature
correction Delta = H_cd D^-1 H_dc AND the profiled score r = g_c - H_cd D^-1 g_d. Inert unless LARND_TP_SCHUR=1."""
p = '/sdf/scratch/neutrino/pgranger/review_2026_09/DXPROF/tree/optimize/fit_params.py'
t = open(p).read()

def rep(a, b):
    global t
    assert t.count(a) == 1, ('not unique', a[:80])
    t = t.replace(a, b)

rep("""            if self._tp_schur and i < self._tp_schur_nb and 'P0_fit' in rec['points']:
              try:
                _t0 = time()
""", """            # DXPROF (6 Nov, review copy only): profile at any point; default P0_fit = unchanged.
            _sp = os.environ.get('LARND_TP_SCHUR_POINT', 'P0_fit')
            _spt = {_n: (_c, _l, _g) for _n, _c, _l, _g in POINTS}
            if self._tp_schur and i < self._tp_schur_nb and _sp in rec['points'] and _sp in _spt:
              try:
                _t0 = time()
                # the block below reads calib_fit / d_fit_j / g_fit; rebind them to the chosen point (nothing after
                # this block uses them: the function only summarises and appends rec)
                calib_fit, d_fit_j, g_fit = _spt[_sp]
""")
rep("""                HNC = np.zeros((N, 5))
                for j, k in enumerate(P5):
                    _, gp = _grads(dcal={k: +_h})
                    _, gm = _grads(dcal={k: -_h})
                    HNC[:, j] = (gp - gm) / (2 * _h)
""", """                HNC = np.zeros((N, 5)); HCC = np.zeros((5, 5))
                for j, k in enumerate(P5):
                    cp, gp = _grads(dcal={k: +_h})
                    cm, gm = _grads(dcal={k: -_h})
                    HNC[:, j] = (gp - gm) / (2 * _h)
                    HCC[:, j] = (cp - cm) / (2 * _h)
""")
old_start = """                    _hp = 3e-3
                    probes = np.empty((Vp.shape[0], Nd))"""
old_end = """                    else:
                        logger.warning(f"[SCHUR] batch {i:3d}: banded solve failed, no record")
                    # restore and skip the joint-CG path entirely
                    self.norm_params = _np_base
                    self.update_params()
                    raise _SchurDone()
"""
a = t.index(old_start); b = t.index(old_end) + len(old_end)
assert t.count(old_start) == 1 and t.count(old_end) == 1 and a < b
new = """                    # DXPROF (6 Nov): every probe step x ridge combination; per combination the curvature correction
                    # Delta = H_cd D^-1 H_dc and the PROFILED SCORE r = g_c - H_cd D^-1 g_d (the efficient score for the
                    # dE/dx block), which the sandwich and the one-step profiled estimate need.
                    _dsteps = [float(x) for x in os.environ.get('LARND_TP_SCHUR_DSTEPS', '3e-3').split(':') if x.strip()]
                    _lams = [float(x) for x in os.environ.get('LARND_TP_SCHUR_LAMS', repr(self._tp_schur_lam)).split(':') if x.strip()]
                    HDC = HNC[:Nd, :]; _gd0 = _gn0[:Nd]
                    _res = {}
                    for _hp in _dsteps:
                        probes = np.empty((Vp.shape[0], Nd))
                        for c in range(Vp.shape[0]):
                            vfull = np.zeros(N); vfull[:Nd] = Vp[c]
                            _, gp = _grads(vn=+_hp * vfull)
                            _, gm = _grads(vn=-_hp * vfull)
                            probes[c] = (gp[:Nd] - gm[:Nd]) / (2 * _hp)
                        if q is not None:
                            probes = probes - np.outer(Vp @ q, q)
                        ab = _dedx_band_from_probes(probes, Nd)
                        _mdiag = float(np.median(np.abs(ab[_DEDX_GN_BW])))
                        _nneg = int((ab[_DEDX_GN_BW] <= 0).sum())
                        for _lr in _lams:
                            lam = _lr * _mdiag
                            X = np.zeros((Nd, 6)); solve_ok = True
                            for j, rhs in enumerate(list(HDC.T) + [_gd0]):
                                x, solve_ok = _dedx_banded_sm_solve(ab, q, rhs, lam)
                                if not solve_ok:
                                    break
                                X[:, j] = x
                            key = f'{_hp:g}_{_lr:g}'
                            if not solve_ok:
                                _res[key] = {'ok': False, 'n_diag_nonpos': _nneg}
                                continue
                            DELTA = HDC.T @ X[:, :5]
                            _res[key] = {'ok': True, 'lam': float(lam), 'n_diag_nonpos': _nneg,
                                         'delta': DELTA.tolist(), 'r_prof': (_gc0 - HDC.T @ X[:, 5]).tolist(),
                                         'asym': float(np.max(np.abs(DELTA - DELTA.T)) / max(np.max(np.abs(DELTA)), 1e-30)),
                                         'dinv_gd_max': float(np.abs(X[:, 5]).max())}
                    rec['schur'] = {'mode': 'dedx', 'point': _sp, 'n_nuisance': int(Nd), 'params': list(P5), 'h_cal': _h,
                                    'gc': _gc0.tolist(), 'hcc': (0.5 * (HCC + HCC.T)).tolist(),
                                    'hcc_asym': float(np.abs(HCC - HCC.T).max() / max(np.abs(HCC).max(), 1e-30)),
                                    'mean_rank1': q is not None, 'res': _res, 'wall_s': round(time() - _t0, 1)}
                    logger.info(f"[SCHUR] batch {i:3d}: dEdx-exact at {_sp}, N={Nd}, "
                                + ', '.join(f"{k_}: {'ok' if v_['ok'] else 'FAILED'}" for k_, v_ in _res.items())
                                + f", {rec['schur']['wall_s']} s")
                    # restore and skip the joint-CG path entirely
                    self.norm_params = _np_base
                    self.update_params()
                    raise _SchurDone()
"""
t = t[:a] + new + t[b:]
open(p, 'w').write(t)
print('patched')
