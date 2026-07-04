"""
Benchmark helpers: summarize per-step runtime and peak GPU memory from a
ParamFitter training_history and print / save a report.

Usage inside code:
    from optimize.benchmark import summarize_history, print_summary
    summary = summarize_history(param_fit.training_history)
    print_summary(summary)

CLI usage:
    python -m optimize.benchmark path/to/history.pkl
"""
import argparse
import json
import pickle
import statistics
import sys


_MEM_KEYS = ("bytes_in_use", "peak_bytes_in_use", "largest_alloc_size")


def _percentiles(values, ps=(0.5, 0.9, 0.99)):
    if not values:
        return {p: float("nan") for p in ps}
    sorted_vals = sorted(values)
    out = {}
    for p in ps:
        idx = min(int(p * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
        out[p] = sorted_vals[idx]
    return out


def summarize_history(history, warmup_steps=1):
    """Extract runtime & memory statistics from a ParamFitter training_history.

    Args:
        history: dict as produced by ParamFitter (must contain 'step_time';
            optionally 'memory').
        warmup_steps: skip this many initial steps for the runtime aggregates
            (they include JIT compile time). Compile step is reported
            separately as 'compile_step_time_s'.

    Returns:
        dict with runtime and memory summary fields.
    """
    step_times = list(history.get("step_time", []))
    memory_records = list(history.get("memory", []))

    total_steps = len(step_times)
    compile_time = step_times[0] if step_times else float("nan")
    warm = step_times[warmup_steps:] if total_steps > warmup_steps else []

    runtime_pct = _percentiles(warm)

    peak_bytes = float("nan")
    final_bytes = float("nan")
    if memory_records:
        peak_bytes = max(
            (rec.get("peak_bytes_in_use", 0) for rec in memory_records), default=0
        )
        last = memory_records[-1]
        final_bytes = last.get("bytes_in_use", float("nan"))

    return {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "compile_step_time_s": compile_time,
        "median_step_time_s": runtime_pct[0.5],
        "p90_step_time_s": runtime_pct[0.9],
        "p99_step_time_s": runtime_pct[0.99],
        "min_step_time_s": min(warm) if warm else float("nan"),
        "max_step_time_s": max(warm) if warm else float("nan"),
        "mean_step_time_s": statistics.mean(warm) if warm else float("nan"),
        "peak_gpu_bytes": peak_bytes,
        "peak_gpu_gb": peak_bytes / 1e9 if peak_bytes == peak_bytes else float("nan"),
        "final_gpu_bytes_in_use": final_bytes,
        "final_gpu_gb_in_use": final_bytes / 1e9 if final_bytes == final_bytes else float("nan"),
    }


def print_summary(summary, header=None):
    """Human-readable one-block report of a summary dict."""
    if header:
        print(f"=== {header} ===")
    print(f"steps recorded          : {summary['total_steps']}")
    print(f"warmup steps skipped    : {summary['warmup_steps']}")
    print(f"compile step time       : {summary['compile_step_time_s']*1000:8.1f} ms")
    print(f"step time (post-warmup) : median {summary['median_step_time_s']*1000:8.1f} ms | "
          f"p90 {summary['p90_step_time_s']*1000:8.1f} ms | "
          f"min {summary['min_step_time_s']*1000:8.1f} ms")
    print(f"peak GPU memory         : {summary['peak_gpu_gb']:8.3f} GB")
    print(f"final in-use GPU memory : {summary['final_gpu_gb_in_use']:8.3f} GB")


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _cli():
    parser = argparse.ArgumentParser(
        description="Summarize runtime and peak GPU memory from a ParamFitter history pickle."
    )
    parser.add_argument("history_path", help="Path to training_history .pkl")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Skip this many initial steps when aggregating step times.")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Optional path to write a JSON summary.")
    args = parser.parse_args()

    with open(args.history_path, "rb") as f:
        history = pickle.load(f)

    summary = summarize_history(history, warmup_steps=args.warmup)
    print_summary(summary, header=args.history_path)
    if args.json_out:
        save_summary(summary, args.json_out)
        print(f"\nsummary written to {args.json_out}")


if __name__ == "__main__":
    _cli()
