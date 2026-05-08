from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export all method results into one table.")
    p.add_argument("--metrics", type=str, required=True, help="Path to method_compare_metrics.json")
    p.add_argument("--outdir", type=str, default="", help="Output directory (default: metrics parent)")
    return p.parse_args()


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    outdir = Path(args.outdir) if args.outdir else metrics_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cols = [
        "method",
        "success_rate",
        "mcs_success_rate",
        "avg_wait_minutes",
        "timeout_events_total",
        "mcs_avg_income",
        "business_score",
        "avg_total_agent_reward_per_ep",
    ]

    rows = []
    for method, m in data.items():
        rows.append(
            {
                "method": method,
                "success_rate": _fmt(m.get("success_rate", 0.0)),
                "mcs_success_rate": _fmt(m.get("mcs_success_rate", 0.0)),
                "avg_wait_minutes": _fmt(m.get("avg_wait_minutes", 0.0)),
                "timeout_events_total": _fmt(m.get("timeout_events_total", 0.0)),
                "mcs_avg_income": _fmt(m.get("mcs_avg_income", 0.0)),
                "business_score": _fmt(m.get("business_score", 0.0)),
                "avg_total_agent_reward_per_ep": _fmt(m.get("avg_total_agent_reward_per_ep", 0.0)),
            }
        )

    csv_path = outdir / "all_results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_path = outdir / "all_results_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r[c] for c in cols) + " |\n")

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
