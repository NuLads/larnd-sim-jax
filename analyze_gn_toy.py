#!/usr/bin/env python3
"""analyze_gn_toy.py — quantitative comparison of GN-toy calibration fit arms.

The "GN toy" is a pure 5-parameter calibration fit (Ab, eField, tran_diff,
long_diff, lifetime) run via `optimize/scripts/start_gn_toy.sh` (and the
gnA/gnC forks of it) using `--fit_type gn_calib`. Each SLURM array task runs
one "offset" (fractional distance of the initial guess from the true values);
an "arm" is a particular knob configuration (soft_modes, damping mode,
batching, curvature, ...) that may be run at several offsets.

This script parses the *.out SLURM logs under logs/gn_toy/ (and, where
available, cross-checks against the *.pkl fit-history checkpoints under
fit_result/gn_toy/) to build one row per (arm, offset) with the final
parameter values, their signed percent error vs. the known truth, loss
trajectory, GN-iteration/termination bookkeeping, and the initial Hessian
diagnostics — instead of eyeballing log tails.

CRITICAL CORRECTNESS POINT: when a GN step is rejected (norm-space
saturation -> REVERTING, or "no decrease after backtracking" -> stalled),
the last "GN iter N: ... Ab=... " log line shows the *tentative* (pre-revert)
parameters, not the final state, which reverts to the previous accepted
iterate (or to init, if rejected at iter 0). This script tracks the accepted
trajectory explicitly rather than trusting the last GN-iter line's params;
rows where the naive "last log line" value would have been wrong are
flagged (see the `reverted_last_line_wrong` note / --csv column).

Usage:
    python3 analyze_gn_toy.py [--arm SUBSTRING] [--csv PATH] [--logdir DIR]
                               [--pkldir DIR] [--no-pkl-check]

Read-only: this script never submits jobs and never touches any file other
than the ones it reads (plus the optional --csv output it writes itself).
"""
import argparse
import glob
import os
import pickle
import re
import sys
from datetime import datetime

# ----------------------------------------------------------------------------
# Ground truth (hardcoded in optimize/scripts/start_gn_toy.sh and forks).
# ----------------------------------------------------------------------------
TRUTH = {
    "Ab": 0.83488,
    "eField": 0.52152,
    "tran_diff": 1.0028e-05,
    "long_diff": 5.8142e-06,
    "lifetime": 2406.4,
}
PARAMS = list(TRUTH.keys())

DEFAULT_LOGDIR = "logs/gn_toy"
DEFAULT_PKLDIR = "fit_result/gn_toy"

# ----------------------------------------------------------------------------
# Regexes for the various log lines we care about.
# ----------------------------------------------------------------------------
RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
RE_OFFSET = re.compile(r"^=== offset ([0-9.eE+-]+) ===")
RE_TARGET = re.compile(r"^target:\s+(.*)$")
RE_INIT = re.compile(r"^init\s*:\s+(.*)$")
RE_TOYENV = re.compile(r"\[TOY-ENV(?:\s+(\S+))?\]\s*(.*)")
RE_FITLABEL = re.compile(r"fit label:\s*(\S+)")

RE_GN_INIT = re.compile(
    r"GN init:\s+loss\s+([\-0-9.eE+]+)\s+\|g\|\s+([\-0-9.eE+]+)\s+curvature=(\S+)"
)
RE_GN_INIT_HESS = re.compile(
    r"GN init Hessian:\s+eig \[(.*?)\]\s+diag \[(.*?)\]\s+cond\s+([\-0-9.eE+]+)"
)
RE_GN_STEP = re.compile(
    r"GN iter (\d+):\s+loss\s+([\-0-9.eE+]+)\s+lambda\s+([\-0-9.eE+]+)\s+"
    r"\|step\|\s+([\-0-9.eE+]+)\s+"
    r"Ab=(\S+)\s+eField=(\S+)\s+tran_diff=(\S+)\s+long_diff=(\S+)\s+lifetime=(\S+)"
)
RE_GN_REVERT = re.compile(
    r"GN iter (\d+):\s+norm-space saturation ([\-0-9.eE+]+)\s*>\s*([\-0-9.eE+]+).*REVERTING"
)
RE_GN_FLOOR = re.compile(
    r"GN iter (\d+):\s+no decrease after backtracking with tiny last step.*numerical floor"
)
RE_GN_STALL = re.compile(
    r"GN iter (\d+):\s+no decrease after backtracking \(lambda"
)
RE_GN_CONVERGED = re.compile(r"GN converged at iter (\d+)")
RE_FIT_OK = re.compile(r"Fitting successful")
RE_FIT_FAIL = re.compile(r"Error: Fitting failed")
RE_OOM = re.compile(r"RESOURCE_EXHAUSTED: Failed to allocate request for (\S+)")
RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
RE_PYERR = re.compile(r"^(?:jax\.errors\.\w+|jaxlib\._jax\.\w+Error|[A-Za-z_.]*Error|[A-Za-z_.]*Exception):\s*(.*)$")
RE_LOGFNAME = re.compile(r"^(?:job|gnA|gnC)-(\d+)_(\d+)\.out$")


def parse_kv_params(s):
    """Parse 'Ab 0.834 eField 0.521 ...' -> {'Ab': 0.834, ...}"""
    toks = s.split()
    out = {}
    for k, v in zip(toks[0::2], toks[1::2]):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def parse_float_list(s):
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def strip_offset_suffix(label):
    """'gnA_dmdiag_ls0_sc0_off0.01' -> 'gnA_dmdiag_ls0_sc0'"""
    return re.sub(r"_off[0-9.eE+-]+$", "", label)


# ----------------------------------------------------------------------------
# Per-log-file parsing
# ----------------------------------------------------------------------------
class LogRecord:
    def __init__(self, path):
        self.path = path
        self.basename = os.path.basename(path)
        self.offset = None
        self.target = {}
        self.init = {}
        self.toyenv_tag = None
        self.toyenv_str = None
        self.fit_label = None
        self.arm = None
        self.gn_init_loss = None
        self.gn_init_gnorm = None
        self.gn_curvature = None
        self.hess_eig = []
        self.hess_diag = []
        self.hess_cond = None
        self.n_step_lines = 0
        self.n_proj_lines = 0
        self.is_validate_only = False
        self.accepted_params = None
        self.accepted_iter = -1  # -1 means "still at init"
        self.terminal_status = None  # converged/reverted/stalled/floor/budget/crashed/incomplete
        self.terminal_detail = ""
        self.crashed = False
        self.crash_reason = ""
        self.fit_ok = False
        self.losses = []  # (iter, loss) accepted-trajectory losses, incl. init as iter=-1
        self.first_ts = None
        self.last_ts = None
        self.reverted_last_line_wrong = False  # True if naive "last GN iter line" would mislead
        self.last_step_line_params = None  # raw params of the very last GN-iter line seen
        m = RE_LOGFNAME.match(self.basename)
        self.slurm_job_id = m.group(1) if m else None
        self.slurm_task_id = m.group(2) if m else None

    def wall_time_s(self):
        if self.first_ts is not None and self.last_ts is not None:
            return (self.last_ts - self.first_ts).total_seconds()
        return None


def parse_log_file(path):
    rec = LogRecord(path)
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        rec.terminal_status = "unreadable"
        rec.terminal_detail = str(e)
        return rec

    candidate = None  # (iter_n, params, loss) pending acceptance
    accepted_params = None
    accepted_iter = -1
    accepted_loss = None

    for raw in lines:
        line = raw.rstrip("\n")

        m = RE_TS.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if rec.first_ts is None:
                    rec.first_ts = ts
                rec.last_ts = ts
            except ValueError:
                pass

        m = RE_OFFSET.match(line)
        if m:
            try:
                rec.offset = float(m.group(1))
            except ValueError:
                pass
            continue

        m = RE_TARGET.match(line)
        if m:
            rec.target = parse_kv_params(m.group(1))
            continue

        m = RE_INIT.match(line)
        if m:
            rec.init = parse_kv_params(m.group(1))
            if accepted_params is None:
                accepted_params = dict(rec.init)
            continue

        m = RE_TOYENV.search(line)
        if m:
            rec.toyenv_tag = m.group(1)
            rec.toyenv_str = m.group(2).strip()
            continue

        m = RE_FITLABEL.search(line)
        if m:
            rec.fit_label = m.group(1)
            continue

        m = RE_GN_INIT.search(line)
        if m:
            rec.gn_init_loss = float(m.group(1))
            rec.gn_init_gnorm = float(m.group(2))
            rec.gn_curvature = m.group(3)
            rec.losses.append((-1, rec.gn_init_loss))
            continue

        m = RE_GN_INIT_HESS.search(line)
        if m:
            rec.hess_eig = parse_float_list(m.group(1))
            rec.hess_diag = parse_float_list(m.group(2))
            try:
                rec.hess_cond = float(m.group(3))
            except ValueError:
                pass
            continue

        if "[GN-PROJ]" in line:
            rec.n_proj_lines += 1
            continue

        if "[GN-VALIDATE]" in line:
            rec.is_validate_only = True
            continue

        m = RE_GN_STEP.search(line)
        if m:
            # A new step line: whatever the previous candidate was, since no
            # revert/stall/floor line intervened before this new step line,
            # it must have been implicitly accepted.
            if candidate is not None:
                accepted_iter, accepted_params, accepted_loss = candidate
            iter_n = int(m.group(1))
            loss = float(m.group(2))
            params = {
                "Ab": float(m.group(5)),
                "eField": float(m.group(6)),
                "tran_diff": float(m.group(7)),
                "long_diff": float(m.group(8)),
                "lifetime": float(m.group(9)),
            }
            candidate = (iter_n, params, loss)
            rec.last_step_line_params = params
            rec.n_step_lines += 1
            continue

        m = RE_GN_REVERT.search(line)
        if m:
            iter_n = int(m.group(1))
            sat = m.group(2)
            if candidate is not None and candidate[0] == iter_n:
                rec.reverted_last_line_wrong = True
                candidate = None  # rejected, do not accept
            rec.terminal_status = "reverted"
            rec.terminal_detail = f"iter {iter_n}: norm-space saturation {sat} > bound"
            continue

        m = RE_GN_FLOOR.search(line)
        if m:
            iter_n = int(m.group(1))
            if candidate is not None and candidate[0] == iter_n:
                accepted_iter, accepted_params, accepted_loss = candidate
                candidate = None
            rec.terminal_status = "floor"
            rec.terminal_detail = f"iter {iter_n}: numerical-floor stall"
            continue

        m = RE_GN_STALL.search(line)
        if m:
            iter_n = int(m.group(1))
            if candidate is not None and candidate[0] == iter_n:
                # step itself did not decrease loss -> not accepted
                rec.reverted_last_line_wrong = True
                candidate = None
            rec.terminal_status = "stalled"
            rec.terminal_detail = f"iter {iter_n}: no decrease after backtracking"
            continue

        m = RE_GN_CONVERGED.search(line)
        if m:
            iter_n = int(m.group(1))
            if candidate is not None and candidate[0] == iter_n:
                accepted_iter, accepted_params, accepted_loss = candidate
                candidate = None
            rec.terminal_status = "converged"
            rec.terminal_detail = f"GN converged at iter {iter_n}"
            continue

        if RE_FIT_OK.search(line):
            rec.fit_ok = True
            continue

        if RE_FIT_FAIL.search(line):
            rec.crashed = True
            continue

        m = RE_OOM.search(line)
        if m:
            rec.crashed = True
            rec.crash_reason = f"OOM ({m.group(1)})"
            continue

        if RE_TRACEBACK.search(line):
            rec.crashed = True
            if not rec.crash_reason:
                rec.crash_reason = "exception"
            continue

        m = RE_PYERR.match(line.strip())
        if m and rec.crashed and rec.crash_reason in ("", "exception"):
            rec.crash_reason = line.strip()[:160]

    # End of file: resolve any still-pending candidate.
    if candidate is not None:
        accepted_iter, accepted_params, accepted_loss = candidate
        if rec.terminal_status is None:
            if rec.fit_ok:
                rec.terminal_status = "budget"
                rec.terminal_detail = f"reached iter {accepted_iter} (no explicit convergence msg)"
            elif rec.crashed:
                rec.terminal_status = "crashed"
            else:
                rec.terminal_status = "incomplete"
                rec.terminal_detail = "log ends mid-run (no Fitting successful/failed)"

    if rec.terminal_status is None:
        if rec.crashed:
            rec.terminal_status = "crashed"
        elif accepted_params is None and rec.init:
            rec.terminal_status = "incomplete"
            rec.terminal_detail = "no GN iterations logged"
        elif not rec.fit_ok:
            rec.terminal_status = "incomplete"
            rec.terminal_detail = "log ends mid-run"

    if accepted_params is None:
        accepted_params = dict(rec.init) if rec.init else None

    if rec.is_validate_only and rec.n_step_lines == 0:
        # --gn_validate: dumps exact-vs-GGN Hessian diagnostics at the *init*
        # point and exits without running the GN loop at all. There is no
        # "fit result" here -- final == init trivially -- so don't let this
        # masquerade as a converged/attempted fit.
        rec.terminal_status = "validate_only"
        rec.terminal_detail = "--gn_validate run: Hessian/Fisher diagnostics only, no GN steps taken"

    rec.accepted_params = accepted_params
    rec.accepted_iter = accepted_iter
    if accepted_loss is not None:
        rec.losses.append((accepted_iter, accepted_loss))

    # Determine arm identity.
    if rec.fit_label:
        rec.arm = strip_offset_suffix(rec.fit_label)
    elif rec.toyenv_tag or rec.toyenv_str:
        tag = rec.toyenv_tag or "toy"
        rec.arm = f"legacy:{tag}:{rec.toyenv_str}".strip(":")
    else:
        # header-only truncation (e.g. log got clipped by the script's
        # `| tail -120`, or the run crashed before printing fit label) —
        # fall back to the filename family so runs are still grouped
        # sensibly instead of being silently dropped.
        prefix = re.sub(r"[-_]\d+(_\d+)?\.out$", "", rec.basename)
        rec.arm = f"legacy:{prefix}"

    return rec


# ----------------------------------------------------------------------------
# Pickle cross-check
# ----------------------------------------------------------------------------
def _load_pkl_final(path):
    """Load one history pkl and extract (final_params_dict, n_entries,
    provenance_job_id, provenance_task_id) or raise."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    final = {}
    n_entries = None
    for p in PARAMS:
        key = f"{p}_iter"
        if key in d and len(d[key]) > 0:
            final[p] = float(d[key][-1])
            n_entries = len(d[key])
        elif p in d and len(d[p]) > 0:
            final[p] = float(d[p][-1])
            n_entries = len(d[p])
    prov = d.get("provenance", {}) or {}
    return final, n_entries, str(prov.get("slurm_job_id")), str(prov.get("slurm_array_task_id"))


def find_pickle_final(pkldir, out_label, want_job_id=None, want_task_id=None):
    """Locate the fit-history checkpoint(s) for `out_label` and return a dict:
        {'path', 'final_params', 'n_entries', 'verified', 'note'}
    or None if nothing matches at all.

    IMPORTANT: multiple distinct runs can share the same --out_label (e.g. two
    manual resubmissions of the same arm/offset), and since the checkpoint
    filename is keyed purely by out_label, a later run silently overwrites an
    earlier run's file on disk. We only trust a checkpoint as this log's own
    final state ('verified') if its embedded provenance (slurm_job_id +
    slurm_array_task_id) matches the log's filename; otherwise we still
    surface it as a hint but mark it unverified so callers don't silently
    substitute another run's numbers.
    """
    if not out_label:
        return None
    pattern = os.path.join(pkldir, "history_iter*_" + out_label + ".pkl")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def iter_num(p):
        m = re.search(r"history_iter(\d+)_", os.path.basename(p))
        return int(m.group(1)) if m else -1

    candidates.sort(key=iter_num, reverse=True)  # most-complete checkpoint first

    # If we know which job/task this log is, prefer the highest-iterN
    # checkpoint whose provenance actually matches it.
    if want_job_id is not None and want_task_id is not None:
        for path in candidates:
            try:
                final, n_entries, job_id, task_id = _load_pkl_final(path)
            except Exception as e:
                continue
            if job_id == want_job_id and task_id == want_task_id:
                return {
                    "path": path, "final_params": final or None, "n_entries": n_entries,
                    "verified": True,
                    "note": None if final else "no *_iter keys found",
                }
        # No provenance match among any candidate -> the on-disk checkpoint(s)
        # for this out_label belong to a *different* run that overwrote it.
        best = candidates[0]
        try:
            final, n_entries, job_id, task_id = _load_pkl_final(best)
            note = (f"out_label collision: checkpoint on disk belongs to "
                    f"slurm {job_id}_{task_id}, not this run ({want_job_id}_{want_task_id})")
            return {"path": best, "final_params": final or None, "n_entries": n_entries,
                    "verified": False, "note": note}
        except Exception as e:
            return {"path": best, "final_params": None, "n_entries": None,
                    "verified": False, "note": f"load error: {e}"}

    # Can't identify this log's own job/task id (unexpected filename) -> best
    # effort, unverified.
    best = candidates[0]
    try:
        final, n_entries, _, _ = _load_pkl_final(best)
        return {"path": best, "final_params": final or None, "n_entries": n_entries,
                "verified": False, "note": "could not parse slurm job/task id from log filename"}
    except Exception as e:
        return {"path": best, "final_params": None, "n_entries": None,
                "verified": False, "note": f"load error: {e}"}


# ----------------------------------------------------------------------------
# Discovery / assembly
# ----------------------------------------------------------------------------
def discover(logdir):
    paths = sorted(glob.glob(os.path.join(logdir, "*.out")))
    recs = []
    for p in paths:
        recs.append(parse_log_file(p))
    return recs


def pct_err(final, truth):
    if final is None:
        return None
    return (final - truth) / truth * 100.0


def mean_abs_pct_err(row):
    vals = [row["pct_err"][p] for p in PARAMS if row["pct_err"].get(p) is not None]
    if not vals:
        return None
    return sum(abs(v) for v in vals) / len(vals)


def build_rows(recs, pkldir, use_pkl):
    rows = []
    for rec in recs:
        row = {
            "arm": rec.arm,
            "offset": rec.offset,
            "log": rec.basename,
            "fit_label": rec.fit_label,
            "status": rec.terminal_status,
            "detail": rec.terminal_detail,
            "crash_reason": rec.crash_reason,
            "n_step_lines": rec.n_step_lines,
            "accepted_iter": rec.accepted_iter,
            "n_proj_lines": rec.n_proj_lines,
            "init_loss": rec.gn_init_loss,
            "final_loss": rec.losses[-1][1] if rec.losses else None,
            "curvature": rec.gn_curvature,
            "hess_eig": rec.hess_eig,
            "hess_cond": rec.hess_cond,
            "wall_time_s": rec.wall_time_s(),
            "reverted_last_line_wrong": rec.reverted_last_line_wrong,
            "final_params_log": rec.accepted_params,
            "final_params": rec.accepted_params,
            "final_source": "log",
            "pkl_path": None,
            "pkl_verified": False,
            "pkl_final_params": None,
            "pkl_note": None,
            "pkl_mismatch": None,
        }

        if use_pkl and rec.fit_label:
            found = find_pickle_final(pkldir, rec.fit_label, rec.slurm_job_id, rec.slurm_task_id)
            if found:
                row["pkl_path"] = found["path"]
                row["pkl_verified"] = found["verified"]
                row["pkl_final_params"] = found["final_params"]
                row["pkl_note"] = found["note"]
                pkl_final = found["final_params"]
                if row["final_params_log"] and pkl_final:
                    mism = {}
                    for p in PARAMS:
                        a = row["final_params_log"].get(p)
                        b = pkl_final.get(p)
                        if a is not None and b is not None and b != 0:
                            rel = abs(a - b) / abs(b) * 100.0
                            if rel > 0.05:  # >0.05% relative diff flagged
                                mism[p] = (a, b, rel)
                    row["pkl_mismatch"] = mism if mism else None
                # A *verified* checkpoint (provenance-matched to this exact
                # log) is ground truth for the final state — it correctly
                # captures both GN-revert-to-loop-entry AND bound clipping of
                # the initial guess, neither of which is fully recoverable
                # from the log text alone. Prefer it.
                if found["verified"] and pkl_final:
                    row["final_params"] = pkl_final
                    row["final_source"] = "pkl(verified)"

        # A run that never actually attempted/completed any optimization
        # step yields "final == init" trivially; reporting that as a percent
        # error would misleadingly rank it as if the fit had run and simply
        # not moved. Treat these as no-data rather than a (spurious) result.
        no_data = rec.terminal_status in ("crashed", "incomplete", "validate_only", "unreadable")
        fp = {} if no_data else (row["final_params"] or {})
        row["pct_err"] = {p: pct_err(fp.get(p), TRUTH[p]) for p in PARAMS}
        row["mape"] = mean_abs_pct_err(row) if not no_data else None
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def fmt_pct(v):
    if v is None:
        return "   n/a "
    return f"{v:+7.2f}%"


def fmt_num(v, nd=3):
    if v is None:
        return "n/a"
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.{nd}e}"
    return f"{v:.{nd}f}"


def status_flag(row):
    tags = []
    if row["status"] == "reverted":
        tags.append("REVERTED")
    if row["status"] == "stalled":
        tags.append("STALLED")
    if row["status"] == "floor":
        tags.append("FLOOR")
    if row["status"] == "converged":
        tags.append("CONVERGED")
    if row["status"] == "budget":
        tags.append("BUDGET")
    if row["status"] == "crashed":
        tags.append("CRASHED")
    if row["status"] == "incomplete":
        tags.append("INCOMPLETE")
    if row["reverted_last_line_wrong"]:
        tags.append("!last-line!=final")
    if row["final_source"] == "pkl(verified)":
        tags.append("pkl-verified")
    if row["pkl_mismatch"] and row["final_source"] == "pkl(verified)":
        # log heuristic disagreed with the (trusted) pickle -> log fallback
        # would have been wrong; this is the case #3 in the task asked us to
        # catch explicitly.
        tags.append("!log-heuristic-wrong!")
    elif row["pkl_mismatch"] and not row["pkl_verified"]:
        tags.append("pkl-on-disk-differs(unverified/overwritten)")
    if row["pkl_path"] and not row["pkl_verified"]:
        tags.append("PKL-UNVERIFIED")
    return ",".join(tags) if tags else row["status"] or "?"


def print_grouped_table(rows):
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    print("=" * 160)
    print("GN TOY ARM COMPARISON — grouped by arm")
    print("=" * 160)
    for arm in sorted(by_arm):
        arm_rows = sorted(by_arm[arm], key=lambda r: (r["offset"] if r["offset"] is not None else -1))
        print(f"\n### arm: {arm}  ({len(arm_rows)} offset(s))")
        header = (
            f"{'offset':>7} {'status':>26} {'iters':>6} {'loss0':>10} {'loss_f':>10} "
            f"{'t(s)':>7} {'cond':>10} "
            + " ".join(f"{p:>11}%err" for p in PARAMS)
            + f" {'MAPE%':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in arm_rows:
            off = f"{r['offset']:.2f}" if r["offset"] is not None else "?"
            iters = f"{r['n_step_lines']}"
            if r["reverted_last_line_wrong"]:
                iters += f"(acc={r['accepted_iter']+1})"
            loss0 = fmt_num(r["init_loss"], 1)
            lossf = fmt_num(r["final_loss"], 1)
            t = fmt_num(r["wall_time_s"], 0) if r["wall_time_s"] is not None else "n/a"
            cond = fmt_num(r["hess_cond"], 2) if r["hess_cond"] is not None else "n/a"
            pcts = " ".join(fmt_pct(r["pct_err"][p]).rjust(12) for p in PARAMS)
            mape = f"{r['mape']:.2f}" if r["mape"] is not None else "n/a"
            flag = status_flag(r)
            print(
                f"{off:>7} {flag:>26} {iters:>6} {loss0:>10} {lossf:>10} "
                f"{t:>7} {cond:>10} {pcts} {mape:>7}"
            )
            if r["pkl_mismatch"]:
                tag = "log-vs-pkl(TRUSTED)" if r["pkl_verified"] else "log-vs-pkl(unverified, likely different run)"
                for p, (a, b, rel) in r["pkl_mismatch"].items():
                    print(f"    ! {tag} [{p}]: log-derived={a:.6g}  pkl={b:.6g}  (rel diff {rel:.3f}%)")
            if r["pkl_note"] and not r["pkl_final_params"]:
                print(f"    ! pkl note: {r['pkl_path']}: {r['pkl_note']}")
            elif r["pkl_note"] and not r["pkl_verified"]:
                print(f"    ! pkl note: {r['pkl_note']}")
            if r["status"] == "crashed":
                print(f"    ! crash: {r['crash_reason'] or 'unknown'}  (log: {r['log']})")


def print_ranked_table(rows):
    ranked = [r for r in rows if r["mape"] is not None]
    ranked.sort(key=lambda r: r["mape"])
    print("\n" + "=" * 160)
    print("RANKED BY MEAN ABSOLUTE PERCENT ERROR (5-param average) — best first")
    print("=" * 160)
    header = (
        f"{'rank':>4} {'MAPE%':>7} {'arm':<38} {'offset':>7} {'status':>14} "
        + " ".join(f"{p:>11}%err" for p in PARAMS)
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked, 1):
        off = f"{r['offset']:.2f}" if r["offset"] is not None else "?"
        pcts = " ".join(fmt_pct(r["pct_err"][p]).rjust(12) for p in PARAMS)
        print(f"{i:>4} {r['mape']:>7.2f} {r['arm']:<38} {off:>7} {r['status']:>14} {pcts}")

    skipped = [r for r in rows if r["mape"] is None]
    if skipped:
        print(f"\n({len(skipped)} row(s) excluded from ranking — no usable final params: "
              + ", ".join(f"{r['arm']}/off{r['offset']}({r['status']})" for r in skipped) + ")")


def write_csv(rows, path):
    import csv

    fields = (
        ["arm", "offset", "status", "detail", "crash_reason", "log", "fit_label",
         "n_step_lines", "accepted_iter", "reverted_last_line_wrong", "final_source",
         "init_loss", "final_loss", "wall_time_s", "hess_cond", "curvature",
         "pkl_path", "pkl_verified", "pkl_note", "pkl_mismatch_vs_log", "mape"]
        + [f"{p}_final" for p in PARAMS]
        + [f"{p}_pct_err" for p in PARAMS]
    )
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            fp = r["final_params"] or {}
            row = [
                r["arm"], r["offset"], r["status"], r["detail"], r["crash_reason"],
                r["log"], r["fit_label"], r["n_step_lines"], r["accepted_iter"],
                r["reverted_last_line_wrong"], r["final_source"],
                r["init_loss"], r["final_loss"],
                r["wall_time_s"], r["hess_cond"], r["curvature"], r["pkl_path"],
                r["pkl_verified"], r["pkl_note"],
                bool(r["pkl_mismatch"]), r["mape"],
            ] + [fp.get(p) for p in PARAMS] + [r["pct_err"][p] for p in PARAMS]
            w.writerow(row)
    print(f"\nCSV written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, help="only show arms whose name contains this substring")
    ap.add_argument("--csv", default=None, help="write full per-row table to this CSV path")
    ap.add_argument("--logdir", default=DEFAULT_LOGDIR)
    ap.add_argument("--pkldir", default=DEFAULT_PKLDIR)
    ap.add_argument("--no-pkl-check", action="store_true", help="skip pickle cross-check (log-only, faster)")
    args = ap.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"error: log dir {args.logdir} not found", file=sys.stderr)
        sys.exit(1)

    recs = discover(args.logdir)
    if not recs:
        print(f"no *.out files found under {args.logdir}")
        sys.exit(0)

    rows = build_rows(recs, args.pkldir, use_pkl=not args.no_pkl_check)

    if args.arm:
        rows = [r for r in rows if args.arm in (r["arm"] or "")]
        if not rows:
            print(f"no arms matching substring {args.arm!r}")
            sys.exit(0)

    print(f"Discovered {len(recs)} log file(s) under {args.logdir}, "
          f"{len(set(r['arm'] for r in rows))} distinct arm(s) after filtering.\n")

    print_grouped_table(rows)
    print_ranked_table(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
