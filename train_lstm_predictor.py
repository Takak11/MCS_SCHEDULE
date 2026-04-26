from __future__ import annotations

"""
Train LSTM predictor for:
1) future regional request demand
2) future FCS congestion
and export a compressed summary encoder for environment state augmentation.

Example:
  python train_lstm_predictor.py --episodes 200 --seq-len 12 --future-horizon 6
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import CONFIG
from env import Environment
from predictive_summary import LSTMSummaryPredictor, assign_points_to_centers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM demand+congestion predictor and save summary encoder.")
    p.add_argument("--outdir", type=str, default="result/predictor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--policy", type=str, default="random", choices=["random", "stay"])

    p.add_argument("--region-k", type=int, default=15)
    p.add_argument("--kmeans-max-iter", type=int, default=30)
    p.add_argument("--kmeans-sample-size", type=int, default=50000)

    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--future-horizon", type=int, default=6)
    p.add_argument("--summary-dim", type=int, default=16)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--congestion-loss-coef", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=30, help="<=0 disables early stopping.")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    return p.parse_args()


def _sample_actions(env: Environment, policy: str, rng: np.random.Generator) -> Dict[str, int]:
    if policy == "stay":
        return {agent: 4 for agent in env.agents}

    # random over valid actions
    mask = env.get_action_mask()
    actions: Dict[str, int] = {}
    for agent in env.agents:
        valid = np.flatnonzero(np.asarray(mask[agent], dtype=np.int8) > 0)
        if len(valid) == 0:
            act_idx = 3
        else:
            act_idx = int(rng.choice(valid))
        actions[agent] = int(act_idx + 1)
    return actions


def _queue_ratio_vec(step_fcs_states: Dict[int, dict], env: Environment) -> np.ndarray:
    vals = []
    for fcs in env.fcs_list:
        st = step_fcs_states.get(fcs.fcs_id)
        q = float(st["queue"]) if st is not None else float(env.fcs_queue.get(fcs.fcs_id, 0))
        cap = float(st["capacity"]) if st is not None else float(fcs.capacity)
        vals.append(q / max(1.0, cap))
    return np.asarray(vals, dtype=np.float32)


def collect_step_series(
    episodes: int,
    max_steps: Optional[int],
    policy: str,
    seed: int,
) -> Tuple[List[List[List[Tuple[float, float]]]], List[np.ndarray], np.ndarray]:
    cfg = dict(CONFIG)
    cfg["use_lstm_summary"] = False
    env = Environment(config=cfg, seed=seed)
    rng = np.random.default_rng(seed + 17)

    all_episode_request_locs: List[List[List[Tuple[float, float]]]] = []
    all_episode_queue_ratio: List[np.ndarray] = []
    region_centers: Optional[np.ndarray] = None

    for ep in range(int(episodes)):
        env.reset(seed=int(seed + (ep + 1) * 9973))
        if region_centers is None and env.relocate_hotspots:
            region_centers = np.asarray(env.relocate_hotspots, dtype=np.float32)
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)

        ep_request_locs_steps: List[List[Tuple[float, float]]] = []
        ep_queue_ratio_steps: List[np.ndarray] = []
        for _ in range(horizon):
            actions = _sample_actions(env, policy=policy, rng=rng)
            step_result = env.step_parallel(actions)

            locs_step: List[Tuple[float, float]] = []
            for req in step_result.requests:
                loc = req.get("location")
                if loc is None:
                    continue
                lat, lon = float(loc[0]), float(loc[1])
                locs_step.append((lat, lon))
            ep_request_locs_steps.append(locs_step)
            ep_queue_ratio_steps.append(_queue_ratio_vec(step_result.fcs_states, env))

            if step_result.done:
                break

        all_episode_request_locs.append(ep_request_locs_steps)
        all_episode_queue_ratio.append(np.asarray(ep_queue_ratio_steps, dtype=np.float32))
        print(
            f"[collect] ep={ep + 1}/{int(episodes)} steps={len(ep_queue_ratio_steps)} "
            f"requests={sum(len(x) for x in ep_request_locs_steps)}"
        )

    if region_centers is None or region_centers.shape[0] == 0:
        region_centers = np.asarray([f.lat_lon for f in env.fcs_list], dtype=np.float32)
    return all_episode_request_locs, all_episode_queue_ratio, region_centers


def compute_region_counts(
    episode_request_locs: List[List[List[Tuple[float, float]]]],
    centers: np.ndarray,
) -> np.ndarray:
    region_k = int(centers.shape[0])
    counts = np.zeros((region_k,), dtype=np.int64)
    for ep_locs_steps in episode_request_locs:
        for locs in ep_locs_steps:
            if not locs:
                continue
            loc_arr = np.asarray(locs, dtype=np.float32)
            idx = assign_points_to_centers(loc_arr, centers)
            np.add.at(counts, idx, 1)
    if counts.sum() <= 0:
        counts[:] = 1
    return counts


def build_supervised_samples(
    episode_request_locs: List[List[List[Tuple[float, float]]]],
    episode_queue_ratio: List[np.ndarray],
    centers: np.ndarray,
    seq_len: int,
    future_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    region_k = int(centers.shape[0])
    fcs_n = int(episode_queue_ratio[0].shape[1]) if episode_queue_ratio else 0

    xs: List[np.ndarray] = []
    y_req: List[np.ndarray] = []
    y_cong: List[np.ndarray] = []

    for ep_locs_steps, ep_queue in zip(episode_request_locs, episode_queue_ratio):
        t_steps = int(ep_queue.shape[0])
        if t_steps <= seq_len + future_horizon:
            continue

        region_counts = np.zeros((t_steps, region_k), dtype=np.float32)
        for t in range(t_steps):
            locs = ep_locs_steps[t]
            if not locs:
                continue
            loc_arr = np.asarray(locs, dtype=np.float32)
            idx = assign_points_to_centers(loc_arr, centers)
            np.add.at(region_counts[t], idx, 1.0)

        x_feat = np.concatenate([region_counts, ep_queue], axis=1)
        for t in range(seq_len - 1, t_steps - future_horizon):
            xs.append(x_feat[t - seq_len + 1 : t + 1].astype(np.float32))
            y_req.append(region_counts[t + 1 : t + 1 + future_horizon].sum(axis=0).astype(np.float32))
            y_cong.append(ep_queue[t + 1 : t + 1 + future_horizon].mean(axis=0).astype(np.float32))

    if not xs:
        raise RuntimeError("No supervised samples built. Increase episodes or reduce seq_len/future_horizon.")

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(y_req, dtype=np.float32),
        np.asarray(y_cong, dtype=np.float32),
    )


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = np.asarray(x, dtype=np.float32)
    s = np.where(np.abs(s) < 1e-6, 1.0, s)
    return s


def evaluate(
    model: LSTMSummaryPredictor,
    x: np.ndarray,
    y_req: np.ndarray,
    y_cong: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    req_mean: np.ndarray,
    req_std: np.ndarray,
    cong_mean: np.ndarray,
    cong_std: np.ndarray,
    batch_size: int,
    congestion_loss_coef: float,
    device: torch.device,
) -> Dict[str, float]:
    if len(x) == 0:
        return {"loss": 0.0, "req_loss": 0.0, "cong_loss": 0.0}

    model.eval()
    losses: List[float] = []
    req_losses: List[float] = []
    cong_losses: List[float] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = (x[start : start + int(batch_size)] - input_mean) / input_std
            yb_req = (y_req[start : start + int(batch_size)] - req_mean) / req_std
            yb_cong = (y_cong[start : start + int(batch_size)] - cong_mean) / cong_std

            xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yb_req_t = torch.as_tensor(yb_req, dtype=torch.float32, device=device)
            yb_cong_t = torch.as_tensor(yb_cong, dtype=torch.float32, device=device)
            pred_req, pred_cong, _ = model(xb_t)

            req_loss = F.mse_loss(pred_req, yb_req_t)
            cong_loss = F.mse_loss(pred_cong, yb_cong_t)
            loss = req_loss + float(congestion_loss_coef) * cong_loss

            losses.append(float(loss.item()))
            req_losses.append(float(req_loss.item()))
            cong_losses.append(float(cong_loss.item()))
    return {
        "loss": float(np.mean(losses)),
        "req_loss": float(np.mean(req_losses)),
        "cong_loss": float(np.mean(cong_losses)),
    }


def _plot_training_curves(path: Path, log_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skip plotting LSTM curves: {e}")
        return

    rows: List[dict] = []
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    epochs = np.asarray([float(r["epoch"]) for r in rows], dtype=np.float32)
    train_loss = np.asarray([float(r["train"]["loss"]) for r in rows], dtype=np.float32)
    val_loss = np.asarray([float(r["val"]["loss"]) for r in rows], dtype=np.float32)
    train_req = np.asarray([float(r["train"]["req_loss"]) for r in rows], dtype=np.float32)
    val_req = np.asarray([float(r["val"]["req_loss"]) for r in rows], dtype=np.float32)
    train_cong = np.asarray([float(r["train"]["cong_loss"]) for r in rows], dtype=np.float32)
    val_cong = np.asarray([float(r["val"]["cong_loss"]) for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plots = [
        (axes[0], train_loss, val_loss, "Total Loss"),
        (axes[1], train_req, val_req, "Request Loss"),
        (axes[2], train_cong, val_cong, "Congestion Loss"),
    ]
    for ax, tr, va, title in plots:
        ax.plot(epochs, tr, label="train", linewidth=2.0)
        ax.plot(epochs, va, "--", label="val", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ep_locs, ep_q, centers = collect_step_series(
        episodes=int(args.episodes),
        max_steps=args.max_steps,
        policy=args.policy,
        seed=int(args.seed),
    )
    if centers.shape[0] == 0:
        raise RuntimeError("Environment hotspots are empty.")
    counts = compute_region_counts(episode_request_locs=ep_locs, centers=centers)
    region_k = int(centers.shape[0])

    x_all, y_req_all, y_cong_all = build_supervised_samples(
        episode_request_locs=ep_locs,
        episode_queue_ratio=ep_q,
        centers=centers,
        seq_len=int(args.seq_len),
        future_horizon=int(args.future_horizon),
    )

    n = len(x_all)
    perm = rng.permutation(n)
    n_val = int(round(float(args.val_ratio) * n))
    n_val = min(max(n_val, 1), n - 1) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx

    x_train, x_val = x_all[train_idx], x_all[val_idx]
    y_req_train, y_req_val = y_req_all[train_idx], y_req_all[val_idx]
    y_cong_train, y_cong_val = y_cong_all[train_idx], y_cong_all[val_idx]

    input_mean = x_train.mean(axis=(0, 1), keepdims=False).astype(np.float32)
    input_std = _safe_std(x_train.std(axis=(0, 1), keepdims=False).astype(np.float32))
    req_mean = y_req_train.mean(axis=0, keepdims=False).astype(np.float32)
    req_std = _safe_std(y_req_train.std(axis=0, keepdims=False).astype(np.float32))
    cong_mean = y_cong_train.mean(axis=0, keepdims=False).astype(np.float32)
    cong_std = _safe_std(y_cong_train.std(axis=0, keepdims=False).astype(np.float32))

    input_dim = int(x_all.shape[-1])
    fcs_n = int(y_cong_all.shape[-1])
    model = LSTMSummaryPredictor(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        region_k=int(region_k),
        fcs_n=fcs_n,
        summary_dim=int(args.summary_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopped = False
    best_path = outdir / "lstm_predictor.pt"
    log_path = outdir / "lstm_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = rng.permutation(len(x_train))
        tr_losses: List[float] = []
        tr_req_losses: List[float] = []
        tr_cong_losses: List[float] = []

        for start in range(0, len(x_train), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = (x_train[idx] - input_mean) / input_std
            yb_req = (y_req_train[idx] - req_mean) / req_std
            yb_cong = (y_cong_train[idx] - cong_mean) / cong_std

            xb_t = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yb_req_t = torch.as_tensor(yb_req, dtype=torch.float32, device=device)
            yb_cong_t = torch.as_tensor(yb_cong, dtype=torch.float32, device=device)
            pred_req, pred_cong, _ = model(xb_t)

            req_loss = F.mse_loss(pred_req, yb_req_t)
            cong_loss = F.mse_loss(pred_cong, yb_cong_t)
            loss = req_loss + float(args.congestion_loss_coef) * cong_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            tr_losses.append(float(loss.item()))
            tr_req_losses.append(float(req_loss.item()))
            tr_cong_losses.append(float(cong_loss.item()))

        train_metrics = {
            "loss": float(np.mean(tr_losses)),
            "req_loss": float(np.mean(tr_req_losses)),
            "cong_loss": float(np.mean(tr_cong_losses)),
        }
        val_metrics = evaluate(
            model=model,
            x=x_val,
            y_req=y_req_val,
            y_cong=y_cong_val,
            input_mean=input_mean,
            input_std=input_std,
            req_mean=req_mean,
            req_std=req_std,
            cong_mean=cong_mean,
            cong_std=cong_std,
            batch_size=int(args.batch_size),
            congestion_loss_coef=float(args.congestion_loss_coef),
            device=device,
        )

        line = (
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_req={train_metrics['req_loss']:.4f} train_cong={train_metrics['cong_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_req={val_metrics['req_loss']:.4f} val_cong={val_metrics['cong_loss']:.4f}"
        )
        print(line)

        payload = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        improved = float(val_metrics["loss"]) < best_val - float(args.early_stop_min_delta)
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = int(epoch)
            stale_epochs = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "seq_len": int(args.seq_len),
                "future_horizon": int(args.future_horizon),
                "region_k": int(region_k),
                "fcs_n": int(fcs_n),
                "summary_dim": int(args.summary_dim),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "region_centers": centers.astype(np.float32).tolist(),
                "cluster_counts": counts.astype(np.int64).tolist(),
                "input_mean": input_mean.astype(np.float32).tolist(),
                "input_std": input_std.astype(np.float32).tolist(),
                "req_mean": req_mean.astype(np.float32).tolist(),
                "req_std": req_std.astype(np.float32).tolist(),
                "cong_mean": cong_mean.astype(np.float32).tolist(),
                "cong_std": cong_std.astype(np.float32).tolist(),
                "args": vars(args),
            }
            torch.save(ckpt, best_path)
            print(f"best updated: epoch={epoch:03d} val_loss={best_val:.4f}")
        else:
            stale_epochs += 1
            if int(args.early_stop_patience) > 0 and stale_epochs >= int(args.early_stop_patience):
                early_stopped = True
                print(
                    f"early stop: epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_val_loss={best_val:.4f}"
                )
                break

    meta = {
        "best_ckpt": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "early_stopped": bool(early_stopped),
        "trained_epochs": int(epoch),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "input_dim": int(input_dim),
        "region_k": int(region_k),
        "fcs_n": int(fcs_n),
        "summary_dim": int(args.summary_dim),
        "seq_len": int(args.seq_len),
        "future_horizon": int(args.future_horizon),
    }
    with (outdir / "lstm_predictor_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _plot_training_curves(path=outdir / "lstm_training_curves.png", log_path=log_path)
    print(f"done: predictor saved to {best_path}")


if __name__ == "__main__":
    main()
