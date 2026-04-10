#!/usr/bin/env python3
"""
Create charts from evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib/numpy missing. Install with: make install-dev")


PALETTE = {
    "baseline": "#4C72B0",
    "tuned": "#DD8452",
    "good": "#55A868",
    "warn": "#C44E52",
    "neutral": "#8172B2",
    "bg": "#F8F8F8",
    "grid": "#E0E0E0",
}

FIGURE_DIR = Path("outputs/figures")


def _setup() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["bg"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def plot_radar(baseline_agg: dict, tuned_agg: Optional[dict] = None) -> Path:
    labels = ["SCR ↑", "F1 ↑", "1−CR ↑", "1−TVE ↑", "1−CDEE ↑"]
    keys_raw = ["avg_scr", "avg_f1", "avg_cr", "avg_tve", "avg_cdee"]
    invert = [False, False, True, True, True]

    def extract(agg):
        vals = []
        for k, inv in zip(keys_raw, invert):
            v = agg.get(k, 0.0)
            vals.append(1.0 - v if inv else v)
        return vals

    b_vals = extract(baseline_agg)
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    b_vals_c = b_vals + b_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)

    ax.plot(angles, b_vals_c, "o-", linewidth=2, color=PALETTE["baseline"], label="Baseline")
    ax.fill(angles, b_vals_c, alpha=0.15, color=PALETTE["baseline"])

    if tuned_agg:
        t_vals = extract(tuned_agg)
        t_vals_c = t_vals + t_vals[:1]
        ax.plot(angles, t_vals_c, "s-", linewidth=2, color=PALETTE["tuned"], label="Fine-tuned")
        ax.fill(angles, t_vals_c, alpha=0.15, color=PALETTE["tuned"])

    ax.set_title("Model Performance Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    path = FIGURE_DIR / "metrics_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_chs_distribution(chs_values: List[float], model_tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    _, bins, patches = ax.hist(chs_values, bins=20, edgecolor="white", alpha=0.8, density=False)
    for patch, left in zip(patches, bins):
        if left < 0.3:
            patch.set_facecolor(PALETTE["good"])
        elif left < 0.6:
            patch.set_facecolor(PALETTE["warn"])
        else:
            patch.set_facecolor("#CC3333")

    mean_val = sum(chs_values) / max(len(chs_values), 1)
    ax.axvline(x=mean_val, color="black", linewidth=2, label=f"Mean CHS = {mean_val:.3f}")
    green_p = mpatches.Patch(color=PALETTE["good"], label=f"Low (≤0.3): {sum(1 for c in chs_values if c<=0.3)}")
    orange_p = mpatches.Patch(color=PALETTE["warn"], label=f"Mid (0.3–0.6): {sum(1 for c in chs_values if 0.3<c<=0.6)}")
    red_p = mpatches.Patch(color="#CC3333", label=f"High (>0.6): {sum(1 for c in chs_values if c>0.6)}")
    ax.legend(handles=[green_p, orange_p, red_p], loc="upper right")
    ax.set_xlabel("Composite Hallucination Score (CHS)")
    ax.set_ylabel("Count")
    ax.set_title(f"CHS Distribution — {model_tag}")

    path = FIGURE_DIR / f"chs_distribution_{model_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_claim_breakdown(sample_records: List[dict], max_queries: int = 20) -> Path:
    records = sample_records[:max_queries]
    queries_short = [f"Q{i+1}" for i in range(len(records))]
    supported = [r["supported_claims"] for r in records]
    contradicted = [r["contradicted_claims"] for r in records]
    neutral = [
        r.get("neutral_claims", r["total_claims"] - r["supported_claims"] - r["contradicted_claims"])
        for r in records
    ]

    x = np.arange(len(queries_short))
    width = 0.65
    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.6), 5))
    ax.bar(x, supported, width, label="Supported", color=PALETTE["good"], alpha=0.9)
    ax.bar(x, neutral, width, bottom=supported, label="Neutral", color=PALETTE["neutral"], alpha=0.9)
    bot = [s + n for s, n in zip(supported, neutral)]
    ax.bar(x, contradicted, width, bottom=bot, label="Contradicted", color=PALETTE["warn"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(queries_short, rotation=45, ha="right")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Claim Verification Breakdown per Query")
    ax.legend()

    path = FIGURE_DIR / "claim_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_chs_vs_f1(sample_records: List[dict], model_tag: str) -> Path:
    chs = [r.get("chs", 0.0) for r in sample_records]
    f1 = [r.get("f1", 0.0) for r in sample_records]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(chs, f1, alpha=0.6, c=chs, cmap="RdYlGn_r", s=50, edgecolors="white")
    plt.colorbar(scatter, ax=ax, label="CHS (lower=better)")

    if len(chs) > 2:
        z = np.polyfit(chs, f1, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(chs), max(chs), 100)
        ax.plot(xs, p(xs), "--", color="grey", alpha=0.7, label=f"Trend (slope={z[0]:.2f})")
        corr = np.corrcoef(chs, f1)[0, 1]
        ax.set_title(f"CHS vs F1 — {model_tag} (r={corr:.3f})")
        ax.legend()
    else:
        ax.set_title(f"CHS vs F1 — {model_tag}")

    ax.set_xlabel("CHS ↓")
    ax.set_ylabel("F1 ↑")
    path = FIGURE_DIR / f"chs_vs_f1_{model_tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(baseline: dict, tuned: dict) -> Path:
    metrics = {
        "F1 ↑": ("avg_f1", False),
        "SCR ↑": ("avg_scr", False),
        "CR ↓": ("avg_cr", True),
        "TVE ↓": ("avg_tve", True),
        "CDEE ↓": ("avg_cdee", True),
        "CHS ↓": ("avg_chs", True),
    }
    labels = list(metrics.keys())
    b_vals = [baseline.get(k, 0) for k, _ in metrics.values()]
    t_vals = [tuned.get(k, 0) for k, _ in metrics.values()]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, b_vals, width, label="Baseline", color=PALETTE["baseline"], alpha=0.85)
    ax.bar(x + width / 2, t_vals, width, label="Fine-tuned", color=PALETTE["tuned"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Fine-tuned")
    ax.legend()

    path = FIGURE_DIR / "metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_chs_curve(chs_values: List[float]) -> Path:
    xs = list(range(1, len(chs_values) + 1))
    window = max(3, len(chs_values) // 10)
    rolling = [
        sum(chs_values[max(0, i - window) : i + 1]) / len(chs_values[max(0, i - window) : i + 1])
        for i in range(len(chs_values))
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(xs, chs_values, alpha=0.4, s=25, color=PALETTE["baseline"], label="Per-sample CHS")
    ax.plot(xs, rolling, color=PALETTE["warn"], linewidth=2.5, label=f"Rolling mean (w={window})")
    ax.axhline(y=0.3, color=PALETTE["good"], linestyle="--", alpha=0.7, label="Low CHS threshold (0.3)")
    ax.set_xlabel("Training Sample Index")
    ax.set_ylabel("CHS")
    y_max = max(chs_values + rolling + [0.3, 1.0])
    ax.set_ylim(0, max(1.05, y_max * 1.05))
    ax.set_title("CHS Values Across Training Corpus")
    ax.legend()

    path = FIGURE_DIR / "training_chs_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/eval")
    parser.add_argument("--samples", default=None)
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument(
        "--charts",
        nargs="+",
        choices=["radar", "chs", "claims", "scatter", "comparison", "training", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    if not HAS_MPL:
        sys.exit(1)

    global FIGURE_DIR
    FIGURE_DIR = Path(args.figures_dir)
    _setup()
    charts = set(args.charts)
    all_charts = "all" in charts

    baseline_agg, tuned_agg = None, None
    for tag in ["baseline", "tuned"]:
        p = Path(args.results_dir) / f"{tag}_aggregate.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            if tag == "baseline":
                baseline_agg = d
            else:
                tuned_agg = d

    sample_records = []
    for tag in ["baseline", "tuned"]:
        p = Path(args.results_dir) / f"{tag}_samples.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    r["model_tag"] = tag
                    sample_records.append(r)

    if (all_charts or "radar" in charts) and baseline_agg:
        plot_radar(baseline_agg, tuned_agg)
    if all_charts or "chs" in charts:
        for tag in ["baseline", "tuned"]:
            recs = [r for r in sample_records if r.get("model_tag") == tag]
            if recs:
                plot_chs_distribution([r.get("chs", 0.0) for r in recs], model_tag=tag)
    if (all_charts or "claims" in charts) and sample_records:
        b_recs = [r for r in sample_records if r.get("model_tag") == "baseline"]
        if b_recs:
            plot_claim_breakdown(b_recs)
    if all_charts or "scatter" in charts:
        for tag in ["baseline", "tuned"]:
            recs = [r for r in sample_records if r.get("model_tag") == tag]
            if len(recs) >= 3:
                plot_chs_vs_f1(recs, model_tag=tag)
    if (all_charts or "comparison" in charts) and baseline_agg and tuned_agg:
        plot_metric_comparison(baseline_agg, tuned_agg)
    if (all_charts or "training" in charts) and args.samples and Path(args.samples).exists():
        samp_chs = []
        with open(args.samples) as f:
            content = f.read().strip()
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    samp_chs = [float(r.get("chs", 0.0)) for r in parsed if isinstance(r, dict)]
            except json.JSONDecodeError:
                with open(args.samples) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict):
                            samp_chs.append(float(row.get("chs", 0.0)))
        if samp_chs:
            plot_training_chs_curve(samp_chs)

    figs = list(FIGURE_DIR.glob("*.png"))
    print(f"Saved {len(figs)} figure(s) to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
