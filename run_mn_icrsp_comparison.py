from __future__ import annotations

"""
MN-ICRSP-style Transformer policy-gradient comparison experiment runner.

This script trains a Transformer actor-critic policy-gradient baseline without
LSTM/DT components as an MN-ICRSP-style comparison policy under this project's
environment and action space. It starts from random network initialization and
uses only online policy-gradient updates.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MN-ICRSP-style Transformer policy-gradient comparison experiment.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--outdir", type=str, default="result/mn_icrsp_transformer_pg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--episodes-per-epoch", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-epoch-interval", type=int, default=0)
    p.add_argument("--candidate-requests", type=int, default=8)
    p.add_argument("--candidate-hotspots", type=int, default=8)
    p.add_argument("--use-lstm-summary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")
    return p.parse_args()


def run(cmd: List[str]) -> None:
    print("[mn_icrsp] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def _best_by(rows: List[Dict[str, float]], key: str, reverse: bool = True) -> Dict[str, float]:
    if not rows:
        return {}
    return max(rows, key=lambda r: r.get(key, float("-inf"))) if reverse else min(rows, key=lambda r: r.get(key, float("inf")))


def build_summary(rows: List[Dict[str, float]]) -> Dict[str, object]:
    best_success = _best_by(rows, "eval_success_rate", reverse=True)
    best_wait = _best_by(rows, "eval_ev_avg_wait_minutes", reverse=False)
    best_business = _best_by(rows, "eval_business_score", reverse=True)
    return {
        "method": "MN-ICRSP-style Transformer policy-gradient comparison",
        "objective_note": "Transformer actor-critic with masked stochastic policy-gradient training",
        "best_by_success": {
            "epoch": int(best_success.get("epoch", 0)),
            "eval_success_rate": float(best_success.get("eval_success_rate", float("nan"))),
            "eval_mcs_avg_income": float(best_success.get("eval_mcs_avg_income", float("nan"))),
            "eval_wait_minutes": float(best_success.get("eval_ev_avg_wait_minutes", float("nan"))),
            "eval_reward": float(best_success.get("eval_reward", float("nan"))),
            "eval_business_score": float(best_success.get("eval_business_score", float("nan"))),
        },
        "best_by_wait": {
            "epoch": int(best_wait.get("epoch", 0)),
            "eval_wait_minutes": float(best_wait.get("eval_ev_avg_wait_minutes", float("nan"))),
            "eval_success_rate": float(best_wait.get("eval_success_rate", float("nan"))),
            "eval_reward": float(best_wait.get("eval_reward", float("nan"))),
            "eval_business_score": float(best_wait.get("eval_business_score", float("nan"))),
        },
        "best_by_business": {
            "epoch": int(best_business.get("epoch", 0)),
            "eval_business_score": float(best_business.get("eval_business_score", float("nan"))),
            "eval_success_rate": float(best_business.get("eval_success_rate", float("nan"))),
            "eval_wait_minutes": float(best_business.get("eval_ev_avg_wait_minutes", float("nan"))),
            "eval_reward": float(best_business.get("eval_reward", float("nan"))),
        },
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "train_mn_icrsp_transformer_pg.py",
        "--outdir",
        str(outdir),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--epochs",
        str(args.epochs),
        "--episodes-per-epoch",
        str(args.episodes_per_epoch),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--candidate-requests",
        str(args.candidate_requests),
        "--candidate-hotspots",
        str(args.candidate_hotspots),
    ]
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if int(args.save_epoch_interval) > 0:
        cmd.extend(["--save-epoch-interval", str(args.save_epoch_interval)])
    if bool(args.use_lstm_summary):
        cmd.append("--use-lstm-summary")
        if args.lstm_predictor_ckpt:
            cmd.extend(["--lstm-predictor-ckpt", str(args.lstm_predictor_ckpt)])

    run(cmd)

    metrics_csv = outdir / "business_metrics.csv"
    if not metrics_csv.exists():
        raise RuntimeError(f"business_metrics.csv not found: {metrics_csv}")
    rows = _load_rows(metrics_csv)
    summary = build_summary(rows)
    summary_path = outdir / "mn_icrsp_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
