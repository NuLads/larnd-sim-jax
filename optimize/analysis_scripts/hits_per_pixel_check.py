"""Print hits-per-pixel diagnostics from a scan pickle's aux_iter.

Reports max_hits_per_pixel and frac_hits_over_slot_cap across scan values,
so you can see whether the current dataset has pixels above D2's per-pixel
slot cap (MAX_ADC_VALUES, default 10) that would trigger the silent
occurrence-index clamping bug.

Usage:
    python3 -m optimize.analysis_scripts.hits_per_pixel_check <pickle_or_glob> [...]

Any pickle produced *after* the diagnostic aux keys were added will emit these
values; older pickles print "keys missing" and are skipped.
"""

import glob
import os
import pickle
import sys

import numpy as np


DIAG_KEYS = ("max_hits_per_pixel", "frac_hits_over_slot_cap", "slot_cap")


def summarize(pkl_path):
    with open(pkl_path, "rb") as fh:
        r = pickle.load(fh)
    aux = r.get("aux_iter") or []
    if not aux or aux[0] is None:
        print(f"  no aux_iter in {os.path.basename(pkl_path)}")
        return
    if any(k not in aux[0] for k in DIAG_KEYS):
        missing = [k for k in DIAG_KEYS if k not in aux[0]]
        print(f"  {os.path.basename(pkl_path)}: missing keys {missing} "
              "(pre-diagnostic pickle; re-run to populate)")
        return

    # Extract per-iteration series.
    param_keys = [k.replace("_grad", "") for k in r.keys() if k.endswith("_grad")]
    if not param_keys:
        print(f"  {os.path.basename(pkl_path)}: no scan-param axis")
        return
    param = param_keys[0]
    piter = np.asarray(r[f"{param}_iter"])
    pvals = np.sort(np.unique(piter[1:]))
    nbt = int(np.count_nonzero(piter == pvals[0]))
    n_scan = len(aux) // nbt

    def series(key):
        raw = np.array([float(a.get(key, np.nan)) if a is not None else np.nan
                        for a in aux[: nbt * n_scan]])
        return raw.reshape(nbt, n_scan).T.mean(axis=1)

    max_hits = series("max_hits_per_pixel")
    frac_over = series("frac_hits_over_slot_cap")
    slot_cap = int(round(float(aux[0]["slot_cap"])))

    tgt = float(r[f"{param}_target"][0])

    print(f"\n== {os.path.basename(pkl_path)}")
    print(f"   scan param: {param} (target={tgt:.4g}, n_scan={n_scan})")
    print(f"   MAX_ADC_VALUES (slot cap): {slot_cap}")
    print(f"   max_hits_per_pixel: min={np.nanmin(max_hits):.1f}  max={np.nanmax(max_hits):.1f}  mean={np.nanmean(max_hits):.1f}")
    print(f"   frac_hits_over_slot_cap: min={np.nanmin(frac_over):.4f}  max={np.nanmax(frac_over):.4f}  mean={np.nanmean(frac_over):.4f}")
    if np.nanmax(max_hits) > slot_cap:
        n_over = int((max_hits > slot_cap).sum())
        print(f"   ⚠  max_hits_per_pixel EXCEEDS slot cap on {n_over}/{n_scan} scan values")
        print(f"      → D2 (full LLHD) would silently clamp overflow slots to slot {slot_cap - 1}")
    else:
        print(f"   ✓ max_hits_per_pixel ≤ slot_cap on all scan points → D2 slot-collision bug would NOT trigger")


def main(patterns):
    files = []
    for pat in patterns:
        matches = sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
        if not matches:
            print(f"no match for {pat}", file=sys.stderr)
        files.extend(matches)
    if not files:
        print("no pickles to inspect", file=sys.stderr)
        sys.exit(1)
    for f in files:
        summarize(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
