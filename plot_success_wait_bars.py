from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot success-rate and waiting-time bar charts from method_compare_metrics.json")
    p.add_argument("--metrics", type=str, required=True, help="Path to method_compare_metrics.json")
    p.add_argument("--outdir", type=str, default="", help="Output dir (default: metrics parent)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    outdir = Path(args.outdir) if args.outdir else metrics_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    methods = list(data.keys())
    success = [float(data[m].get("success_rate", 0.0)) for m in methods]
    wait = [float(data[m].get("avg_wait_minutes", 0.0)) for m in methods]
    x = np.arange(len(methods))

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x, success, color="#2a9d8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=10)
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate by Method")
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    p1 = outdir / "success_rate_bar.png"
    fig1.savefig(p1, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x, wait, color="#e76f51")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=10)
    ax2.set_ylabel("Average Wait (minutes)")
    ax2.set_title("Average Waiting Time by Method")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    p2 = outdir / "avg_wait_minutes_bar.png"
    fig2.savefig(p2, dpi=180)
    plt.close(fig2)

    print(f"saved: {p1}")
    print(f"saved: {p2}")


if __name__ == "__main__":
    main()
