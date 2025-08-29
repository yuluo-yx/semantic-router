import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

parser = argparse.ArgumentParser()
parser.add_argument(
    "--summary",
    type=Path,
    required=True,
    help="Path to summary.json produced by the bench",
)
parser.add_argument(
    "--router-summary",
    type=Path,
    required=False,
    help="Optional path to router summary.json to overlay",
)
parser.add_argument(
    "--metrics",
    type=str,
    nargs="+",
    default=["accuracy", "avg_response_time", "avg_total_tokens"],
    choices=["accuracy", "avg_response_time", "avg_total_tokens"],
    help="One or more metrics to plot (default: all)",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("."),
    help="Directory to save plots (default: current directory)",
)
args = parser.parse_args()
summary_path = args.summary

with open(summary_path) as f:
    s = json.load(f)

s_router = None
if args.router_summary:
    with open(args.router_summary) as f:
        s_router = json.load(f)


def derive_metrics(summary_json: dict, summary_path: Path):
    cat_by_mode = summary_json.get("category_by_mode")
    cat_ranges = summary_json.get("category_ranges")
    if cat_by_mode is not None and cat_ranges is not None:
        return cat_by_mode, cat_ranges

    csv_path = summary_path.parent / "detailed_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing fields in summary and CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df.get("success", True) == True]
    if "mode_label" not in df.columns:
        raise SystemExit(
            "detailed_results.csv lacks 'mode_label' column; cannot compute per-mode stats"
        )

    grouped = (
        df.groupby(["category", "mode_label"]).agg(
            accuracy=("is_correct", "mean"),
            avg_response_time=("response_time", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
        )
    ).reset_index()

    cat_by_mode = {}
    cat_ranges = {}
    for cat in grouped["category"].unique():
        sub = grouped[grouped["category"] == cat]
        modes = {}
        for _, row in sub.iterrows():
            modes[str(row["mode_label"])] = {
                "accuracy": (
                    float(row["accuracy"]) if pd.notna(row["accuracy"]) else 0.0
                ),
                "avg_response_time": (
                    float(row["avg_response_time"])
                    if pd.notna(row["avg_response_time"])
                    else 0.0
                ),
                "avg_prompt_tokens": (
                    float(row["avg_prompt_tokens"])
                    if pd.notna(row["avg_prompt_tokens"])
                    else None
                ),
                "avg_completion_tokens": (
                    float(row["avg_completion_tokens"])
                    if pd.notna(row["avg_completion_tokens"])
                    else None
                ),
                "avg_total_tokens": (
                    float(row["avg_total_tokens"])
                    if pd.notna(row["avg_total_tokens"])
                    else None
                ),
            }
        cat_by_mode[cat] = modes

        # ranges
        def _mm(values):
            values = [v for v in values if v is not None]
            if not values:
                return {"min": 0.0, "max": 0.0}
            return {"min": float(min(values)), "max": float(max(values))}

        acc_vals = [v.get("accuracy") for v in modes.values()]
        lat_vals = [v.get("avg_response_time") for v in modes.values()]
        tok_vals = [v.get("avg_total_tokens") for v in modes.values()]
        cat_ranges[cat] = {
            "accuracy": _mm(acc_vals),
            "avg_response_time": _mm(lat_vals),
            "avg_total_tokens": _mm(tok_vals),
        }
    return cat_by_mode, cat_ranges


cat_by_mode, cat_ranges = derive_metrics(s, summary_path)

cats = sorted(cat_ranges.keys())


def plot_metric(metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(cats))

    # Overlay each mode as points
    all_modes = sorted({m for c in cats for m in cat_by_mode.get(c, {}).keys()})
    if len(all_modes) > 0:
        palette = colormaps.get_cmap("tab10").resampled(len(all_modes))
        for i, mode in enumerate(all_modes):
            ys = []
            for c in cats:
                ys.append(cat_by_mode.get(c, {}).get(mode, {}).get(metric))
            ax.scatter(x, ys, s=20, color=palette.colors[i], label=mode, alpha=0.8)

    # Overlay router per-category metric as diamonds, if provided
    if s_router is not None:
        router_cat = s_router.get("category_metrics", {})
        router_vals = []
        router_x = []
        for idx, c in enumerate(cats):
            v = router_cat.get(c, {}).get(metric)
            if v is not None:
                router_x.append(idx)
                router_vals.append(v)
        if router_vals:
            ax.scatter(
                router_x,
                router_vals,
                s=50,
                color="tab:red",
                marker="D",
                label="router",
                zorder=4,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=60, ha="right")
    ylabel = metric.replace("_", " ")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-category {ylabel} per-mode values")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


args.out_dir.mkdir(parents=True, exist_ok=True)
for metric in args.metrics:
    out_path = args.out_dir / f"bench_plot_{metric}.png"
    plot_metric(metric, out_path)
