# reverse slope b~a: = 1 for a calibrated posterior mean (MAP with the matched 1 mm prior), = 1/c for an unfinished fit a = c b
exec(open("/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py").read().split("\nH = {")[0])
def ab(d):
    A, B = {1: [], 2: []}, {1: [], 2: []}
    for bi in d["chain_cache"]:
        for st, c in zip(d["chain_cache"][bi], d["chain_contexts"][bi]):
            a = np.asarray(st["angles"], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c["theta0_i"], c["phi0_i"]); e1, e2 = _transverse_frame(u0)
            g = np.random.default_rng(zlib.crc32((str(tuple(c["track_id"])) + "|0").encode()))
            sm = (g.normal(0., .1, 3), g.normal(0., .1, 3))
            for k in (0, 1):
                for ax, e, j in ((1, e1, K - 2 + k), (2, e2, 2 * K - 2 + k)):
                    B[ax].append(-sm[k] @ e); A[ax].append(a[j])
    return {ax: (np.array(A[ax]), np.array(B[ax])) for ax in (1, 2)}
for s in S:
    for it in (14000, 24000, 36000, 44000):
        p = (glob.glob(f"{FR}/sci_full_EZS{s}FR/history_iter14000_*.pkl") if it == 14000 else
             glob.glob(f"{LK}/sci_full_EZS{s}L/history_iter{it}_*.pkl") if it < END else
             glob.glob(f"{FR}/sci_full_EZS{s}L/history_iter{END}_*.pkl"))
        o = ab(pickle.load(open(sorted(p)[0], "rb")))
        print(f"EZS{s} {it:6d} " + "   ".join(f"e{ax}: fwd {a @ b / (b @ b):.3f} rev {a @ b / (a @ a):.3f} rms(b) {1e4 * np.sqrt(np.mean(b * b)):4.0f}" for ax, (a, b) in o.items()))
