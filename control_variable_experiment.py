from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import torch

from config import CONFIG
from plot_method_comparison import (
    _evaluate_actor_ckpt,
    _evaluate_dt_ckpt,
    _evaluate_greedy_2opt_one_to_many,
    _evaluate_random,
    _evaluate_transformer_pg_ckpt,
)


DEFAULT_EV_COUNTS = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
DEFAULT_MCS_COUNTS = [10, 20, 30, 40, 50]


def _int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run controlled EV-count and MCS-count sensitivity experiments.")
    p.add_argument("--outdir", type=str, default="result/control_variable")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--ev-counts", type=str, default=",".join(str(x) for x in DEFAULT_EV_COUNTS))
    p.add_argument("--mcs-counts", type=str, default=",".join(str(x) for x in DEFAULT_MCS_COUNTS))
    p.add_argument("--fixed-ev-count", type=int, default=int(CONFIG.get("ev_count", 1000)))
    p.add_argument("--fixed-mcs-count", type=int, default=int(CONFIG.get("mcs_num", 20)))
    p.add_argument("--methods", type=str, default="RWS,MN-ICRSP,Greedy+2Opt+One-to-Many,Ours-NoLSTM,Ours-NoDT,Ours")
    p.add_argument("--ppo-ckpt", type=str, default="result/ppo_for_offline/best_by_business.pt")
    p.add_argument("--dt-ckpt", type=str, default="result/dt_ppo_ft/best_by_business.pt")
    p.add_argument("--mn-ckpt", type=str, default="result/mn_icrsp_transformer_pg/best_by_business.pt")
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _method_names(raw: str) -> List[str]:
    return [m.strip() for m in str(raw).split(",") if m.strip()]


def _base_cfg(ev_count: int, mcs_count: int, use_lstm_summary: bool = True) -> dict:
    cfg = dict(CONFIG)
    cfg["ev_count"] = int(ev_count)
    cfg["mcs_num"] = int(mcs_count)
    cfg["use_lstm_summary"] = bool(use_lstm_summary)
    return cfg


def _one_to_many_cfg(ev_count: int, logical_mcs_count: int) -> dict:
    cfg = _base_cfg(ev_count=ev_count, mcs_count=max(1, int(logical_mcs_count) // 2), use_lstm_summary=False)
    cfg["mcs_service_parallel_capacity"] = 2
    return cfg


def _evaluate_method(
    method: str,
    ev_count: int,
    mcs_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    episodes = int(args.episodes)
    seed = int(args.seed)
    deterministic = not bool(args.eval_stochastic)
    if method == "RWS":
        return _evaluate_random(
            _base_cfg(ev_count, mcs_count, use_lstm_summary=False),
            episodes=episodes,
            seed=seed,
            max_steps=args.max_steps,
            verbose=bool(args.verbose),
        )
    if method == "Greedy+2Opt+One-to-Many":
        return _evaluate_greedy_2opt_one_to_many(
            _one_to_many_cfg(ev_count, mcs_count),
            episodes=episodes,
            seed=seed,
            max_steps=args.max_steps,
            verbose=bool(args.verbose),
        )
    if method in {"Ours-NoDT", "Ours-NoDTTune"}:
        return _evaluate_actor_ckpt(
            Path(args.ppo_ckpt),
            _base_cfg(ev_count, mcs_count, use_lstm_summary=True),
            episodes=episodes,
            seed=seed,
            device=device,
            max_steps=args.max_steps,
            deterministic=deterministic,
            verbose=bool(args.verbose),
            tag=method,
        )
    if method == "Ours":
        return _evaluate_dt_ckpt(
            Path(args.dt_ckpt),
            _base_cfg(ev_count, mcs_count, use_lstm_summary=True),
            episodes=episodes,
            seed=seed,
            device=device,
            max_steps=args.max_steps,
            deterministic=deterministic,
            verbose=bool(args.verbose),
            tag=method,
        )
    if method == "Ours-NoLSTM":
        return _evaluate_dt_ckpt(
            Path(args.dt_ckpt),
            _base_cfg(ev_count, mcs_count, use_lstm_summary=False),
            episodes=episodes,
            seed=seed,
            device=device,
            max_steps=args.max_steps,
            deterministic=deterministic,
            verbose=bool(args.verbose),
            tag=method,
        )
    if method in {"MN-ICRSP", "MN-ICRSP-style Transformer PG"}:
        return _evaluate_transformer_pg_ckpt(
            Path(args.mn_ckpt),
            _base_cfg(ev_count, mcs_count, use_lstm_summary=False),
            episodes=episodes,
            seed=seed,
            device=device,
            max_steps=args.max_steps,
            deterministic=deterministic,
        )
    raise ValueError(f"Unknown method: {method}")


def _run_sweep(
    sweep_name: str,
    values: Iterable[int],
    methods: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for value in values:
        ev_count = int(value) if sweep_name == "ev_count" else int(args.fixed_ev_count)
        mcs_count = int(args.fixed_mcs_count) if sweep_name == "ev_count" else int(value)
        for method in methods:
            print(f"[{sweep_name}] value={value} ev={ev_count} mcs={mcs_count} method={method}", flush=True)
            stats = _evaluate_method(method, ev_count, mcs_count, args, device)
            row: Dict[str, object] = {
                "sweep": sweep_name,
                "value": int(value),
                "ev_count": ev_count,
                "mcs_count": mcs_count,
                "method": method,
            }
            row.update({k: float(v) for k, v in stats.items()})
            rows.append(row)
            print(
                f"  success={float(row.get('success_rate', 0.0)):.4f} "
                f"mcs_success={float(row.get('mcs_success_rate', 0.0)):.4f} "
                f"wait={float(row.get('avg_wait_minutes', 0.0)):.2f}min "
                f"income={float(row.get('mcs_avg_income', 0.0)):.2f} "
                f"biz={float(row.get('business_score', 0.0)):.2f}",
                flush=True,
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(path: Path, rows: List[Dict[str, object]], sweep_name: str, metric: str, ylabel: str) -> None:
    methods = sorted({str(r["method"]) for r in rows if str(r["sweep"]) == sweep_name})
    if not methods:
        return
    plt.figure(figsize=(8.8, 5.2))
    for method in methods:
        pts = [r for r in rows if str(r["sweep"]) == sweep_name and str(r["method"]) == method]
        pts.sort(key=lambda r: int(r["value"]))
        plt.plot([int(r["value"]) for r in pts], [float(r.get(metric, 0.0)) for r in pts], marker="o", label=method)
    plt.xlabel("EV count" if sweep_name == "ev_count" else "MCS count")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_all(outdir: Path, rows: List[Dict[str, object]]) -> None:
    specs = [
        ("success_rate", "Success rate"),
        ("mcs_success_rate", "MCS success rate"),
        ("avg_wait_minutes", "Average wait (min)"),
        ("mcs_avg_income", "MCS average income"),
        ("business_score", "Business score"),
    ]
    for sweep_name in ["ev_count", "mcs_count"]:
        for metric, ylabel in specs:
            _plot_metric(outdir / f"{sweep_name}_{metric}.png", rows, sweep_name, metric, ylabel)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    methods = _method_names(args.methods)
    ev_counts = _int_list(args.ev_counts)
    mcs_counts = _int_list(args.mcs_counts)

    rows: List[Dict[str, object]] = []
    rows.extend(_run_sweep("ev_count", ev_counts, methods, args, device))
    rows.extend(_run_sweep("mcs_count", mcs_counts, methods, args, device))

    _write_csv(outdir / "control_variable_results.csv", rows)
    with (outdir / "control_variable_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    _plot_all(outdir, rows)
    print(f"saved: {outdir / 'control_variable_results.csv'}")
    print(f"saved: {outdir / 'control_variable_results.json'}")


if __name__ == "__main__":
    main()
