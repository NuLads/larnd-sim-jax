"""GNPROF (6 Nov): patch of the review copy GNPROF/tree (copied from DXPROF/tree) -- the Gauss-Newton profile of the
calibration over each batch's dE/dx AND geometry, added to the dE/dx-exact Schur block of the truth-point evaluator.
Inert unless LARND_TP_SCHUR_GN=1 (and LARND_TP_SCHUR=1, LARND_TP_SCHUR_MODE=dedx). Live tree untouched.
(1) loss_wrapper_combined: self._gnp_mode 'fields' / 'fields_up' returns the predicted fields (log-intensity hit_prob,
    adcs_distrib [, unique_pixels]); 'prior' skips the simulation (hit term 0); 'lof' takes the fields from
    self._gnp_override (the objective as a function of the fields). None = unchanged.
(2) compute_loss(return_loss_closure='gnp') returns loss_wrapper_combined itself.
(3) ParamFitter._gnp_block: the GN columns, prior Hessian, assembly; called from the dE/dx Schur block."""
p = '/sdf/scratch/neutrino/pgranger/review_2026_09/GNPROF/tree/optimize/fit_params.py'
t = open(p).read()


def rep(a, b):
    global t
    assert t.count(a) == 1, ('not unique', t.count(a), a[:100])
    t = t.replace(a, b)


rep("""                prediction = self.sim_strategy.predict(physical_params, t, self.sim_track_fields, rngkey, unique_size=unique_size, roi_override=roi_override)
                _target_use = target_data if data_over is None else data_over['target']
                hit_loss, hit_aux = self.loss_strategy.compute(physical_params, prediction, _target_use)
""", """                # GNPROF (6 Nov, review copy only): _gnp_mode 'fields' / 'fields_up' returns the predicted fields for the
                # Gauss-Newton profile, 'prior' skips the simulation (hit term 0), 'lof' reads the fields from
                # self._gnp_override (objective as a function of the fields). None (default): unchanged.
                _gm = self.__dict__.get('_gnp_mode')
                if _gm == 'prior':
                    prediction = None; hit_loss = jnp.float32(0.); hit_aux = {}
                else:
                    if _gm == 'lof':
                        _ov = self._gnp_override
                        prediction = {'hit_prob': _ov[0], 'adcs_distrib': _ov[1], 'unique_pixels': _ov[2]}
                    else:
                        prediction = self.sim_strategy.predict(physical_params, t, self.sim_track_fields, rngkey, unique_size=unique_size, roi_override=roi_override)
                    if _gm == 'fields':
                        return (prediction['hit_prob'], prediction['adcs_distrib']), {}
                    if _gm == 'fields_up':
                        return (prediction['hit_prob'], prediction['adcs_distrib'], prediction['unique_pixels']), {}
                    _target_use = target_data if data_over is None else data_over['target']
                    hit_loss, hit_aux = self.loss_strategy.compute(physical_params, prediction, _target_use)
""")

rep("""                aux_out['n_active_pixels'] = jnp.sum(prediction['unique_pixels'] >= 0)
""", """                aux_out['n_active_pixels'] = (jnp.sum(prediction['unique_pixels'] >= 0) if prediction is not None
                                              else jnp.int32(0))
""")

rep("""            if return_loss_closure == 'joint' and _chain_angles_active and _dedx_active:
""", """            if return_loss_closure == 'gnp':                  # GNPROF (6 Nov, review copy only)
                return loss_wrapper_combined
            if return_loss_closure == 'joint' and _chain_angles_active and _dedx_active:
""")

rep("""                                    'mean_rank1': q is not None, 'res': _res, 'wall_s': round(time() - _t0, 1)}
""", """                                    'mean_rank1': q is not None, 'res': _res, 'wall_s': round(time() - _t0, 1)}
                    # GNPROF (6 Nov, review copy only; LARND_TP_SCHUR_GN=1): Gauss-Newton profile over dE/dx AND geometry
                    if os.environ.get('LARND_TP_SCHUR_GN', '0') == '1':
                        rec['schur']['gn'] = self._gnp_block(i, tracks, ref, epoch, parent_ids, _spt, _sp, Vp, _gc0, _gn0)
                        rec['schur']['wall_s'] = round(time() - _t0, 1)
""")

rep("""    def _set_calib_physical(self, phys):
""", '''    def _gnp_block(self, i, tracks, ref, epoch, parent_ids, spt, sp, Vp, gc0, gn0):
        """GNPROF (6 Nov, review copy only; REVIEW.md "GNPROF"): Gauss-Newton profile of the calibration over this batch's
        dE/dx AND geometry at the point sp. The curvature uses FIRST derivatives of the predicted fields only (one jvp +
        one vjp through the simulation per column), so the non-smooth geometry (discrete pixel / tick selections) never
        enters a second derivative. Two curvature models from the same jvp tangents (t_z, t_a) of the fields
        (z = log-intensity, a = ADC distribution):
          A: expected Fisher of the marked Poisson process, sum lam t_z^2 + lam (kappa t_a)^2 / sigma^2 (PSD; ignores the
             +-2 tick matching window and the hit-index mixture of the loss);
          C: exact generalised Gauss-Newton J^T (d2 loss / d fields2) J (the loss's own curvature, window included =
             the Hessian of the objective with the simulation linearised; not PSD by construction).
        Columns: 5 calibration, 5 dE/dx band probes (data block banded, bw 2), every geometry coefficient. Priors (dE/dx
        split-t + barrier + mean constraint, MCS + endpoint) enter with their exact Hessian (no simulation). Also the
        calibration Fisher at P1_truth (validation against the finite-difference Hessian stored there)."""
        from jax.flatten_util import ravel_pytree
        P5 = list(self.relevant_params_list)
        sig = float(self.loss_strategy.sigma_charge) / 1000.0
        kap = float(adc2charge(jnp.float32(1.0), self.ref_params) - adc2charge(jnp.float32(0.0), self.ref_params))
        lams = [float(x) for x in os.environ.get('LARND_TP_SCHUR_GN_LAMS', '0:1e-3:1e-2').split(':') if x.strip()]
        T = {}; out = {'variants': ['A', 'C'], 'kappa': kap, 'sigma': sig}
        nonfin = {'A': 0, 'C': 0}

        def _mode(m, fn):
            self._gnp_mode = m
            try:
                return fn()
            finally:
                self._gnp_mode = None

        def _clean(a, key):
            bad = ~jnp.isfinite(a)
            nonfin[key] += int(jnp.sum(bad))
            return jnp.where(bad, 0.0, a)

        def _setup(name):
            cal, ld, gg = spt[name]
            gg = list(gg)
            self._set_calib_physical(cal)
            npb = self.norm_params
            lwc = self.compute_loss(tracks, i, *ref, epoch=epoch, use_physical_params=False, log_dedx=ld,
                                    parent_ids=parent_ids, chain_angles_per_track=gg, return_loss_closure='gnp')
            x0 = jnp.asarray([float(getattr(npb, k)) for k in P5], jnp.float32)
            shp = {k: jnp.shape(getattr(npb, k)) for k in P5}

            def f(x, d, g):
                return lwc(npb.replace(**{k: jnp.reshape(x[j], shp[k]) for j, k in enumerate(P5)}), d, g)[0]
            z0, a0, up0 = _mode('fields_up', lambda: f(x0, ld, gg))
            valid = (jnp.asarray(up0) >= 0).reshape((-1,) + (1,) * (z0.ndim - 1))
            lam = jnp.where(valid, jnp.exp(jnp.where(valid, z0, -jnp.inf)), 0.0)

            def lof(z, a):
                self._gnp_override = (z, a, up0)
                try:
                    return _mode('lof', lambda: lwc(npb, ld, gg)[0])
                finally:
                    self._gnp_override = None
            glof = jax.grad(lof, argnums=(0, 1))

            def tang(vx, vd, vg):
                tz, ta = _mode('fields', lambda: jax.jvp(f, (x0, ld, gg), (vx, vd, vg))[1])
                return jnp.where(valid, _clean(tz, 'A'), 0.0), jnp.where(valid, _clean(ta, 'A'), 0.0)

            def cot(tz, ta):
                ca = (lam * tz, lam * (kap * kap / sig ** 2) * ta)
                hz, ha = jax.jvp(glof, (z0, a0), (tz, ta))[1]
                return ca, (_clean(hz, 'C'), _clean(ha, 'C'))
            return dict(f=f, x0=x0, ld=ld, gg=gg, tang=tang, cot=cot, npb=npb, lwc=lwc,
                        ncell=int(z0.size), shape=list(z0.shape), nvalid=int(jnp.sum(valid)))

        def _gram(tc):
            g = {}
            for v, key in (('A', 2), ('C', 3)):
                g[v] = np.array([[float(jnp.sum(a[0] * b[key][0]) + jnp.sum(a[1] * b[key][1])) for b in tc] for a in tc])
            return g

        # ---- P0 (the profile point): jvp columns, both cotangents, vjp rows ----
        _t = time()
        s0 = _setup(sp)
        Nd = int(np.asarray(s0['ld']).size)
        flat_g, unflat_g = ravel_pytree(s0['gg']); Ng = int(flat_g.size); Ntot = 5 + Nd + Ng
        out.update(n_d=Nd, n_g=Ng, ncell=s0['ncell'], field_shape=s0['shape'], nvalid_pix=s0['nvalid'])
        _, vjp_fn = _mode('fields', lambda: jax.vjp(s0['f'], s0['x0'], s0['ld'], s0['gg']))

        def _vjp(c):
            gx, gd, gg_ = vjp_fn(c)
            return np.concatenate([np.asarray(gx, np.float64), np.ravel(np.asarray(gd, np.float64)),
                                   np.asarray(ravel_pytree(list(gg_))[0], np.float64)])
        T['setup'] = time() - _t
        zx = jnp.zeros(5, jnp.float32); zd = jnp.zeros_like(s0['ld']); zg = [jnp.zeros_like(c) for c in s0['gg']]
        tans = [(zx.at[j].set(1.0), zd, zg) for j in range(5)]
        tans += [(zx, jnp.asarray(Vp[c], jnp.float32).reshape(zd.shape), zg) for c in range(Vp.shape[0])]
        eye = np.eye(Ng, dtype=np.float32)
        tans += [(zx, zd, unflat_g(jnp.asarray(eye[m]))) for m in range(Ng)]
        cols = {'A': np.zeros((Ntot, len(tans))), 'C': np.zeros((Ntot, len(tans)))}
        tc0 = []; tj = tv = 0.0
        for k, (vx, vd, vg) in enumerate(tans):
            _a = time()
            tz, ta = s0['tang'](vx, vd, vg); ca, cc = s0['cot'](tz, ta); jax.block_until_ready(cc)
            tj += time() - _a
            if k < 5:
                tc0.append((tz, ta, ca, cc))
            _a = time(); cols['A'][:, k] = _vjp(ca); cols['C'][:, k] = _vjp(cc); tv += time() - _a
        T['jvp'] = tj; T['vjp'] = tv
        g0 = _gram(tc0)
        del vjp_fn, tc0
        out['mem_gib_after_cols'] = float(_gn_mem_peak())

        # ---- exact prior Hessian over (dE/dx, geometry) (no simulation) ----
        _t = time()
        npb0 = s0['npb']; lwc0 = s0['lwc']; dsh = zd.shape

        def prf(v):
            return _mode('prior', lambda: lwc0(npb0, v[:Nd].reshape(dsh), unflat_g(v[Nd:]))[0])
        v0 = jnp.concatenate([jnp.ravel(s0['ld']).astype(jnp.float32), flat_g.astype(jnp.float32)])
        gpr = jax.grad(prf)
        hv = jax.vmap(lambda u: jax.jvp(gpr, (v0,), (u,))[1])
        E = jnp.eye(Nd + Ng, dtype=jnp.float32)
        HP = np.concatenate([np.asarray(hv(E[a:a + 64]), np.float64) for a in range(0, Nd + Ng, 64)], axis=0)
        out['prior_asym'] = float(np.abs(HP - HP.T).max() / max(np.abs(HP).max(), 1e-30))
        HP = 0.5 * (HP + HP.T)
        out['prior_dg_max'] = float(np.abs(HP[:Nd, Nd:]).max()) if Ng else 0.0
        T['prior'] = time() - _t

        # ---- P1_truth: calibration Fisher / GGN for the validation against the FD Hessian there ----
        _t = time()
        if 'P1_truth' in spt and spt['P1_truth'][1] is not None:
            s1 = _setup('P1_truth')
            zd1 = jnp.zeros_like(s1['ld']); zg1 = [jnp.zeros_like(c) for c in s1['gg']]
            tc1 = []
            for j in range(5):
                tz, ta = s1['tang'](zx.at[j].set(1.0), zd1, zg1); ca, cc = s1['cot'](tz, ta); tc1.append((tz, ta, ca, cc))
            g1 = _gram(tc1); del tc1
            out['p1'] = {v: g1[v].tolist() for v in g1}
        T['p1'] = time() - _t

        # ---- assembly (float64, host) ----
        _t = time()
        bw = _DEDX_GN_BW
        for v in ('A', 'C'):
            Cm = cols[v]; Fc = Cm[:, :5]; Fp = Cm[:, 5:5 + Vp.shape[0]]; Fg = Cm[:, 5 + Vp.shape[0]:]
            Fcc = 0.5 * (Fc[:5] + Fc[:5].T)
            Fdc = Fc[5:5 + Nd]
            Fgc_c = Fc[5 + Nd:]; Fcg_g = Fg[:5].T
            Fgc = 0.5 * (Fgc_c + Fcg_g)
            Fdg = Fg[5:5 + Nd]; Fgg = Fg[5 + Nd:]
            rec = {'fcc': Fcc.tolist(), 'fcc_gram': g0[v].tolist(),
                   'asym_cc': float(np.abs(Fc[:5] - Fc[:5].T).max() / max(np.abs(Fc[:5]).max(), 1e-30)),
                   'gram_vs_vjp': float(np.abs(g0[v] - Fcc).max() / max(np.abs(Fcc).max(), 1e-30)),
                   'asym_cg': float(np.abs(Fgc_c - Fcg_g).max() / max(np.abs(Fgc_c).max(), 1e-30)) if Ng else 0.0,
                   'asym_gg': float(np.abs(Fgg - Fgg.T).max() / max(np.abs(Fgg).max(), 1e-30)) if Ng else 0.0}
            Fgg = 0.5 * (Fgg + Fgg.T)
            ab = _dedx_band_from_probes(Fp[5:5 + Nd].T, Nd)
            Fdd = np.zeros((Nd, Nd)); ii = np.arange(Nd)
            for dd in range(-bw, bw + 1):
                jj = ii + dd; ok = (jj >= 0) & (jj < Nd)
                Fdd[ii[ok], jj[ok]] = ab[bw - dd, jj[ok]]
            rec['dd_asym'] = float(np.abs(Fdd - Fdd.T).max() / max(np.abs(Fdd).max(), 1e-30))
            Fdd = 0.5 * (Fdd + Fdd.T)
            Ann = np.block([[Fdd + HP[:Nd, :Nd], Fdg + HP[:Nd, Nd:]], [Fdg.T + HP[Nd:, :Nd], Fgg + HP[Nd:, Nd:]]])
            Ann = 0.5 * (Ann + Ann.T)
            Fnc = np.vstack([Fdc, Fgc])
            md = float(np.median(np.abs(np.diag(Ann))))
            ev = np.linalg.eigvalsh(Ann)
            evd = np.linalg.eigvalsh(Ann[:Nd, :Nd]); evg = np.linalg.eigvalsh(Ann[Nd:, Nd:]) if Ng else np.array([1.0])
            rec.update(ann_nneg=int((ev <= 0).sum()), ann_mineig_rel=float(ev.min() / md),
                       add_nneg=int((evd <= 0).sum()), agg_nneg=int((evg <= 0).sum()), med_diag=md,
                       data_diag_frac=float(np.median(np.diag(np.block([[Fdd, Fdg], [Fdg.T, Fgg]])) / np.diag(Ann))))
            res = {}
            for lr in lams:
                key = f'{lr:g}'
                try:
                    Am = Ann + lr * md * np.eye(Nd + Ng)
                    X = np.linalg.solve(Am, np.hstack([Fnc, gn0[:, None]]))
                    Xd = np.linalg.solve(Am[:Nd, :Nd], np.hstack([Fdc, gn0[:Nd, None]]))
                    Xg = (np.linalg.solve(Am[Nd:, Nd:], np.hstack([Fgc, gn0[Nd:, None]])) if Ng
                          else np.zeros((0, 6)))
                    D = Fnc.T @ X[:, :5]; Dd = Fdc.T @ Xd[:, :5]; Dg = Fgc.T @ Xg[:, :5]
                    ok = bool(np.all(np.isfinite(D)) and np.all(np.isfinite(Dd)) and np.all(np.isfinite(Dg)))
                    res[key] = {'ok': ok, 'lam': lr * md, 'delta': D.tolist(), 'delta_d': Dd.tolist(), 'delta_g': Dg.tolist(),
                                'ceff': (Fnc.T @ X[:, 5]).tolist(), 'ceff_d': (Fdc.T @ Xd[:, 5]).tolist(),
                                'ceff_g': (Fgc.T @ Xg[:, 5]).tolist(),
                                'asym': float(np.abs(D - D.T).max() / max(np.abs(D).max(), 1e-30)),
                                'minS': float(np.linalg.eigvalsh(Fcc - 0.5 * (D + D.T)).min())}
                except np.linalg.LinAlgError:
                    res[key] = {'ok': False}
            rec['res'] = res
            out[v] = rec
        out['nonfinite'] = nonfin
        T['assembly'] = time() - _t
        out['t'] = {k: round(float(x), 2) for k, x in T.items()}
        out['mem_gib_peak'] = float(_gn_mem_peak())
        logger.info(f"[GNPROF] batch {i:3d} at {sp}: Nd={Nd} Ng={Ng} cells={out['ncell']} | times {out['t']} | "
                    f"mem {out['mem_gib_peak']:.2f} GiB | A: nneg {out['A']['ann_nneg']} gram/vjp {out['A']['gram_vs_vjp']:.1e} "
                    f"asym_gg {out['A']['asym_gg']:.1e} | C: nneg {out['C']['ann_nneg']} asym_gg {out['C']['asym_gg']:.1e} | "
                    f"nonfinite {nonfin}")
        return out

    def _set_calib_physical(self, phys):
''')

open(p, 'w').write(t)
print('patched', p)
