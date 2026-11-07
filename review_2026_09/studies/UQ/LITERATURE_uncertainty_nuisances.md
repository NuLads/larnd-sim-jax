# Literature: uncertainty on θ with many fitted nuisances, a non-optimal end point, and a non-smooth objective

*6 Nov 2026. Every reference below was found by web search and its bibliographic details were checked. Remarks marked "(ours)" are standard algebra, not quotations.*

**Framing (ours).** Each nuisance belongs to one batch, so θ̂ is an M-estimator over independent groups with profiled score ψ_b = ∂_θ ℓ_b(θ, η̂_b(θ)). The target is V = H_p⁻¹(Σ_b ψ_bψ_bᵀ)H_p⁻¹, where H_p = H_θθ − H_θη H_ηη⁻¹ H_ηθ. Because H_θθ ⪰ H_p, the conditional sandwich must under-predict; your 2.3–3.3× is the expected direction. V is the variance about the pseudo-true value. The incidental bias is a separate problem (§8).

## 1. Efficient information and the profile sandwich
Ĩ = I_θθ − I_θη I_ηη⁻¹ I_ηθ, with efficient scores s̃ = s_θ − I_θη I_ηη⁻¹ s_η. It needs an invertible per-track I_ηη, which is cheap because the matrix is block-diagonal.
- **(a) Yes.** van der Vaart Thm 5.21 needs only Ψ_n(θ̂) = o_P(n^-1/2) and a differentiable *population* score.
- **(b) Yes.**
- **(c) No**, if built from the observed FD Hessian.
- The Murphy–Topel two-step correction is the same algebra.

References:
- Murphy & van der Vaart 2000, JASA 95:449. doi:10.1080/01621459.2000.10474219
- van der Vaart 1998, *Asymptotic Statistics*, CUP. https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D
- Huber 1967, Proc. 5th Berkeley Symp. 1:221. https://bibbase.org/network/publication/huber-thebehaviorofmaximumlikelihoodestimatesundernonstandardconditions-1967
- White 1982, Econometrica 50:1. doi:10.2307/1912526
- Stefanski & Boos 2002, Am. Stat. 56:29. doi:10.1198/000313002753631330
- Murphy & Topel 1985, JBES 3:370. doi:10.2307/1391724
- Newey & McFadden 1994, Handbook of Econometrics 4 ch. 36 (two-step and nonsmooth cases). https://ideas.repec.org/h/eee/ecochp/4-36.html

## 2. Expected information / Gauss–Newton
For a Poisson NLL, F = Jᵀdiag(1/μ)J (plus the dE/dx prior), which equals the GGN. It is PSD, needs no stationarity, and needs only JVPs; its Schur complement is PSD.
- Efron–Hinkley favour observed information. That is moot here, since the observed Hessian is unusable.
- Do not use the empirical Fisher as the bread (Kunstner et al.).
- **(c) Partial.** It omits curvature from the pixel/tick selection jumps.

References:
- Efron & Hinkley 1978, Biometrika 65:457. doi:10.1093/biomet/65.3.457
- Schraudolph 2002, Neural Comput. 14:1723. doi:10.1162/08997660260028683
- Martens 2020, JMLR 21(146). https://jmlr.org/papers/v21/17-678.html
- Kunstner, Hennig & Balles 2019, NeurIPS. https://proceedings.neurips.cc/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html

## 3. HEP: Asimov data and Barlow–Beeston
- **Asimov.** Cowan et al. (Eq. 28) take V⁻¹ from the Hessian on Asimov data, which is §2.
  - **Flag:** the Asimov set is built at *hypothesised* parameters. Using it at the fitted point is an extension.
- **Barlow–Beeston** (one nuisance per bin) is also a growing-nuisance problem. Dembinski & Abdelmotteleb report good toy coverage.
  - **Flag:** Alexe et al. find *systematic under-coverage of Hessian intervals even with the full BB likelihood*, because fitted nuisances absorb fluctuations. This is the closest HEP analogue to your case.

References:
- Cowan, Cranmer, Gross & Vitells 2011, EPJC 71:1554. arXiv:1007.1727
- Barlow & Beeston 1993, CPC 77:219. doi:10.1016/0010-4655(93)90005-W
- Conway 2011. arXiv:1103.0354
- Argüelles, Schneider & Yuan 2019, JHEP 06:030. doi:10.1007/JHEP06(2019)030
- Dembinski & Abdelmotteleb 2022, EPJC 82:1043. doi:10.1140/epjc/s10052-022-11019-z
- Alexe et al. 2024/2026. arXiv:2401.10542 (NIM A doi:10.1016/j.nima.2026.171360, per arXiv)

## 4. One-step estimators
From a √n-consistent start, θ₁ = θ̂ + Ĩ⁻¹Σ_b s̃_b is asymptotically efficient.
- **(a) Yes.** It removes the optimiser residual, and the step size relative to σ is a diagnostic.
- The efficient score is Neyman-orthogonal, so it tolerates unconverged η̂. The partial score does not.
- It does not remove the incidental bias, and it has the same (c) problem as §1.

References:
- Le Cam 1956, Proc. 3rd Berkeley Symp. 1 (sources disagree on pages). https://projecteuclid.org/euclid.bsmsp/1200501652
- Bickel 1975, JASA 70:428. doi:10.1080/01621459.1975.10479884
- van der Vaart 1998 §5.7
- Chernozhukov et al. 2018, Econometrics J. 21:C1. doi:10.1111/ectj.12097

## 5. SGD-trajectory covariance (flag: does not apply)
The theory assumes one pass over fresh draws, step η·i^-α, and a strongly convex loss. Your multi-epoch Adam fit on fixed data, with an annealed LR, falls outside it. The trajectory spread would then measure optimiser noise (your 20–40 % component), not sampling variance. Mandt et al. need a tuned constant LR.

References:
- Polyak & Juditsky 1992, SIAM J. Control Optim. 30:838. doi:10.1137/0330046
- Chen, Lee, Tong & Zhang 2020, AoS 48:251. doi:10.1214/18-AOS1801
- Zhu, Chen & Wu 2023, JASA 118:393. doi:10.1080/01621459.2021.1933498
- Mandt, Hoffman & Blei 2017, JMLR 18(134). https://jmlr.org/papers/v18/17-214.html

## 6. Infinitesimal jackknife
- **(ours)** With batch weights and a block-arrow H, dθ̂/dw_b = −H_p⁻¹s̃_b. The IJ variance is therefore exactly the §1 sandwich, including nuisance propagation.
- Giordano et al.'s error bounds assume smoothness. Stephenson & Broderick's bounds degrade as the dimension grows with N.
- The exact delete-a-batch jackknife needs no derivatives.

References:
- Jaeckel 1972, Bell Labs MM 72-1215-11. https://faculty.washington.edu/fscholz/Jaeckel.html
- Efron 1982, SIAM CBMS-NSF 38. doi:10.1137/1.9781611970319
- Giordano et al. 2019, PMLR 89:1139. https://proceedings.mlr.press/v89/giordano19a.html
- Stephenson & Broderick 2020, PMLR 108:2424. https://proceedings.mlr.press/v108/stephenson20a.html

## 7. IFT profiling
- dη̂/dθ = −H_ηη⁻¹H_ηθ for each track (Ramsay et al.; in JAX, jaxopt `custom_root`). It needs an exact inner optimum and an invertible H_ηη.
- **Derivative-free variant.** Re-optimise η only at θ̂ ± h·e_k, then difference Σ_b∂_θℓ_b to get H_p.
- On the step size: "for a discontinuous objective function, the step size should not converge to zero too rapidly" (Hong et al.). Only the derivative of the expected criterion matters (Pakes–Pollard).

References:
- Ramsay, Hooker, Campbell & Cao 2007, JRSSB 69:741. doi:10.1111/j.1467-9868.2007.00610.x
- Blondel et al. 2022, NeurIPS. arXiv:2105.15183
- Hong, Mahajan & Nekipelov 2015, J. Econometrics 188:250. doi:10.1016/j.jeconom.2014.05.019
- Pakes & Pollard 1989, Econometrica 57:1027. doi:10.2307/1913622

## 8. Incidental-parameter corrections
- **The problem.** With fixed observations per nuisance, the MLE is inconsistent (Neyman–Scott). This matches "bias does not shrink with data".
- **Adjusted likelihoods.** Barndorff-Nielsen's modified profile; Cox–Reid ℓ_p − ½log|j_ηη| (needs orthogonality); McCullagh–Tibshirani; integrated (Laplace) likelihood.
- **Growth conditions.** Sartori: profile inference is standard only if q/m → 0; the modified profile needs only q/m³ → 0.
- **Panel corrections.** Hahn–Newey, Dhaene–Jochmans (split-panel 2θ̂ − mean θ̂_half), Fernández-Val–Weidner.
- **Flag: these do not fix the variance.** "Bias correction does not increase the asymptotic variance" (Hahn–Newey). Dhaene–Jochmans compute SEs by bootstrap or jackknife over units. All are large-T results, so they reduce fixed-T bias but do not remove it.
- **Truth-free bias options:**
  - Split-panel: 2 fits, but it needs two equivalent halves per track that share the nuisances, which is unestablished for pixel data.
  - Parametric-bootstrap bias at (θ̂, η̂) (Kosmidis §4.2).
  - Analytical corrections need third derivatives, which is infeasible.

References:
- Neyman & Scott 1948, Econometrica 16:1. doi:10.2307/1914288
- Lancaster 2000, J. Econometrics 95:391. doi:10.1016/S0304-4076(99)00044-5
- Barndorff-Nielsen 1983, Biometrika 70:343. doi:10.1093/biomet/70.2.343
- Cox & Reid 1987, JRSSB 49:1. doi:10.1111/j.2517-6161.1987.tb01422.x
- McCullagh & Tibshirani 1990, JRSSB 52:325. doi:10.1111/j.2517-6161.1990.tb01790.x
- Berger, Liseo & Wolpert 1999, Stat. Sci. 14:1. doi:10.1214/ss/1009211804
- Sartori 2003, Biometrika 90:533. doi:10.1093/biomet/90.3.533
- Hahn & Newey 2004, Econometrica 72:1295. doi:10.1111/j.1468-0262.2004.00533.x
- Dhaene & Jochmans 2015, REStud 82:991. doi:10.1093/restud/rdv007
- Fernández-Val & Weidner 2016, J. Econometrics 192:291. doi:10.1016/j.jeconom.2015.12.014
- Kosmidis 2014, WIREs Comput. Stat. 6:185. doi:10.1002/wics.1296

## 9. Non-smooth objectives
- Asymptotic normality needs only a smooth *expected* criterion (§1, §7). The bread can come from the GGN, a numerical profile, or smoothing the selection (stochastic AD, Kagan & Heinrich).
- **Bootstrap.** Parametric toys are the gold standard. A cluster bootstrap with exchangeable weights is valid with nuisances (Cheng & Huang).
- The k-step bootstrap (Davidson–MacKinnon; Andrews) assumes smoothness, so warm-started short refits are only a heuristic analogue for you.
- **Flag:** a batch bootstrap cannot see the incidental bias.

References:
- Kagan & Heinrich 2023. arXiv:2308.16680
- Cheng & Huang 2010, AoS 38:2884. doi:10.1214/10-AOS809
- Davidson & MacKinnon 1999, IER 40:487. doi:10.1111/1468-2354.00026
- Andrews 2002, Econometrica 70:119. doi:10.1111/1468-0262.00271

## Ranked recommendation
1. **GGN-Schur profile sandwich:** Ĩ = F_θθ − Σ_t F_θη,t F_ηη,t⁻¹ F_ηθ,t, with the meat from the per-batch s̃_b. The one-step correction comes free.
   - **Caveat:** it omits the selection-jump curvature, and many-nuisance Hessians can still under-cover (Alexe). Check it against the n = 24 replicas.
2. **Numerical profile bread** (10 η-only refits at θ̂ ± h·e_k) with the same meat. This tests #1 under non-smoothness.
   - **Caveat:** re-fit noise and the choice of h. Use identical warm starts and schedules.
3. **Replicas or a warm-started batch bootstrap** to calibrate #1 and #2, plus the parametric-bootstrap bias at (θ̂, η̂).
   - **Caveat:** cost. The batch bootstrap cannot see the incidental bias, and toys generated at η̂ inherit its roughness.
