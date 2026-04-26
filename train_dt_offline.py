from __future__ import annotations

"""
Offline Decision Transformer training (discrete actions, BC/NLL).

Example:
  python train_dt_offline.py --dataset dataset/offline_ppo_traj.npz --outdir result/dt_offline
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from decision_transformer.models.decision_transformer import DecisionTransformer


@dataclass
class Trajectory:
    observations: np.ndarray
    action_indices: np.ndarray
    returns_to_go: np.ndarray
    steps: np.ndarray
    action_masks: np.ndarray
    length: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Decision Transformer with offline trajectories.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--outdir", type=str, default="result/dt_offline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--max-trajs",
        type=int,
        default=8000,
        help=">0 means randomly sample at most this many trajectories from dataset before training; <=0 keeps all.",
    )

    p.add_argument("--context-len", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-ep-len", type=int, default=216)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rtg-scale", type=float, default=0.0, help="<=0 means auto from dataset.")
    p.add_argument("--early-stop-patience", type=int, default=15, help="<=0 disables early stopping.")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    return p.parse_args()


def load_trajectories(path: Path, max_trajs: int = 0, rng: Optional[np.random.Generator] = None) -> tuple[List[Trajectory], int]:
    d = np.load(path, allow_pickle=True)
    n = int(len(d["observations"]))
    if int(max_trajs) > 0 and int(max_trajs) < n:
        if rng is None:
            rng = np.random.default_rng(0)
        picked = np.asarray(rng.choice(n, size=int(max_trajs), replace=False), dtype=np.int64)
        picked.sort()
    else:
        picked = np.arange(n, dtype=np.int64)

    out: List[Trajectory] = []
    for i in picked.tolist():
        obs = np.asarray(d["observations"][i], dtype=np.float32)
        act_idx = np.asarray(d["action_indices"][i], dtype=np.int64).reshape(-1)
        rtg = np.asarray(d["returns_to_go"][i], dtype=np.float32).reshape(-1, 1)
        steps = np.asarray(d["steps"][i], dtype=np.int64).reshape(-1, 1)
        masks = np.asarray(d["action_masks"][i], dtype=np.int8)
        k = int(obs.shape[0])
        out.append(
            Trajectory(
                observations=obs,
                action_indices=act_idx,
                returns_to_go=rtg,
                steps=steps,
                action_masks=masks,
                length=k,
            )
        )
    return out, n


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = np.asarray(x, dtype=np.float32)
    s = np.where(np.abs(s) < 1e-6, 1.0, s)
    return s


class BatchSampler:
    def __init__(
        self,
        trajectories: List[Trajectory],
        obs_dim: int,
        action_dim: int,
        context_len: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        rtg_scale: float,
        max_ep_len: int,
    ) -> None:
        self.trajs = trajectories
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.context_len = int(context_len)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.rtg_scale = float(max(1e-6, rtg_scale))
        self.max_ep_len = int(max_ep_len)
        lengths = np.array([max(1, t.length) for t in self.trajs], dtype=np.float64)
        self.sample_prob = lengths / lengths.sum()

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.choice(len(self.trajs), size=int(batch_size), p=self.sample_prob)

        states = np.zeros((batch_size, self.context_len, self.obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.context_len, self.action_dim), dtype=np.float32)
        returns = np.zeros((batch_size, self.context_len, 1), dtype=np.float32)
        timesteps = np.zeros((batch_size, self.context_len), dtype=np.int64)
        attention = np.zeros((batch_size, self.context_len), dtype=np.int64)
        labels = np.zeros((batch_size, self.context_len), dtype=np.int64)
        action_masks = np.zeros((batch_size, self.context_len, self.action_dim), dtype=np.bool_)

        eye = np.eye(self.action_dim, dtype=np.float32)
        for b, ti in enumerate(idx):
            tr = self.trajs[int(ti)]
            k = int(tr.length)
            if k <= self.context_len:
                s = 0
                e = k
            else:
                s = int(rng.integers(0, k - self.context_len + 1))
                e = s + self.context_len
            seg = e - s
            pad = self.context_len - seg

            obs = (tr.observations[s:e] - self.state_mean) / self.state_std
            act_idx = tr.action_indices[s:e]
            rtg = tr.returns_to_go[s:e] / self.rtg_scale
            tsteps = np.clip(tr.steps[s:e, 0], 0, self.max_ep_len - 1)
            am = tr.action_masks[s:e].astype(np.bool_)

            states[b, pad:] = obs
            actions[b, pad:] = eye[act_idx]
            returns[b, pad:] = rtg
            timesteps[b, pad:] = tsteps
            attention[b, pad:] = 1
            labels[b, pad:] = act_idx
            action_masks[b, pad:] = am

        return {
            "states": states,
            "actions": actions,
            "returns": returns,
            "timesteps": timesteps,
            "attention": attention,
            "labels": labels,
            "action_masks": action_masks,
        }


def _compute_loss_and_acc(model: DecisionTransformer, batch: dict, device: torch.device) -> tuple[torch.Tensor, float]:
    states_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    timesteps_t = torch.as_tensor(batch["timesteps"], dtype=torch.long, device=device)
    attention_t = torch.as_tensor(batch["attention"], dtype=torch.long, device=device)
    labels_t = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
    action_masks_t = torch.as_tensor(batch["action_masks"], dtype=torch.bool, device=device)

    _, action_logits, _ = model(
        states=states_t,
        actions=actions_t,
        rewards=None,
        returns_to_go=returns_t,
        timesteps=timesteps_t,
        attention_mask=attention_t,
    )
    masked_logits = action_logits.masked_fill(~action_masks_t, -1e9)

    valid = attention_t > 0
    logits_flat = masked_logits[valid]
    labels_flat = labels_t[valid]
    loss = F.cross_entropy(logits_flat, labels_flat)

    with torch.no_grad():
        pred = torch.argmax(logits_flat, dim=-1)
        acc = float((pred == labels_flat).float().mean().item()) if logits_flat.numel() > 0 else 0.0
    return loss, acc


def evaluate(model: DecisionTransformer, sampler: BatchSampler, steps: int, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    with torch.no_grad():
        for _ in range(int(steps)):
            b = sampler.sample(batch_size=int(batch_size), rng=rng)
            loss, acc = _compute_loss_and_acc(model=model, batch=b, device=device)
            losses.append(float(loss.item()))
            accs.append(float(acc))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(np.mean(accs)) if accs else 0.0,
    }


def _plot_training_curves(path: Path, log_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skip plotting DT curves: {e}")
        return

    rows = []
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
    train_loss = np.asarray([float(r["train_loss"]) for r in rows], dtype=np.float32)
    val_loss = np.asarray([float(r["val_loss"]) for r in rows], dtype=np.float32)
    train_acc = np.asarray([float(r["train_acc"]) for r in rows], dtype=np.float32)
    val_acc = np.asarray([float(r["val_acc"]) for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train", linewidth=2.0)
    axes[0].plot(epochs, val_loss, "--", label="val", linewidth=1.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train", linewidth=2.0)
    axes[1].plot(epochs, val_acc, "--", label="val", linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "dt_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    trajs, total_traj = load_trajectories(Path(args.dataset), max_trajs=int(args.max_trajs), rng=rng)
    if not trajs:
        raise RuntimeError("Empty trajectory dataset.")

    obs_dim = int(trajs[0].observations.shape[1])
    action_dim = int(trajs[0].action_masks.shape[1])

    all_obs = np.concatenate([t.observations for t in trajs], axis=0).astype(np.float32)
    all_rtg = np.concatenate([t.returns_to_go for t in trajs], axis=0).astype(np.float32).reshape(-1)
    state_mean = all_obs.mean(axis=0, keepdims=False).astype(np.float32)
    state_std = _safe_std(all_obs.std(axis=0, keepdims=False).astype(np.float32))
    dataset_max_return = float(np.max(all_rtg)) if all_rtg.size > 0 else 0.0
    rtg_abs_max = float(np.max(np.abs(all_rtg))) if all_rtg.size > 0 else 1.0
    rtg_scale = float(args.rtg_scale) if float(args.rtg_scale) > 0 else max(1.0, rtg_abs_max)

    perm = rng.permutation(len(trajs))
    n_val = int(round(float(args.val_ratio) * len(trajs)))
    n_val = min(max(n_val, 1), len(trajs) - 1) if len(trajs) > 1 else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx = val_idx
    train_trajs = [trajs[int(i)] for i in tr_idx]
    val_trajs = [trajs[int(i)] for i in val_idx] if len(val_idx) > 0 else train_trajs

    train_sampler = BatchSampler(
        trajectories=train_trajs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_len=int(args.context_len),
        state_mean=state_mean,
        state_std=state_std,
        rtg_scale=rtg_scale,
        max_ep_len=int(args.max_ep_len),
    )
    val_sampler = BatchSampler(
        trajectories=val_trajs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_len=int(args.context_len),
        state_mean=state_mean,
        state_std=state_std,
        rtg_scale=rtg_scale,
        max_ep_len=int(args.max_ep_len),
    )

    model = DecisionTransformer(
        state_dim=obs_dim,
        act_dim=action_dim,
        hidden_size=int(args.hidden_size),
        max_length=int(args.context_len),
        max_ep_len=int(args.max_ep_len),
        action_tanh=False,
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_inner=4 * int(args.hidden_size),
        resid_pdrop=float(args.dropout),
        attn_pdrop=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopped = False
    best_path = outdir / "dt_best.pt"
    best_acc_path = outdir / "dt_best_acc.pt"
    last_path = outdir / "dt_last.pt"

    print(
        f"dt_init traj={len(trajs)}/{total_traj} train={len(train_trajs)} val={len(val_trajs)} "
        f"obs_dim={obs_dim} action_dim={action_dim} ctx={int(args.context_len)} rtg_scale={rtg_scale:.4f}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: List[float] = []
        train_accs: List[float] = []
        for _ in range(int(args.steps_per_epoch)):
            batch = train_sampler.sample(batch_size=int(args.batch_size), rng=rng)
            loss, acc = _compute_loss_and_acc(model=model, batch=batch, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_accs.append(float(acc))

        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        tr_acc = float(np.mean(train_accs)) if train_accs else 0.0
        val = evaluate(
            model=model,
            sampler=val_sampler,
            steps=int(args.eval_steps),
            batch_size=int(args.batch_size),
            device=device,
            rng=rng,
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "context_len": int(args.context_len),
            "max_ep_len": int(args.max_ep_len),
            "hidden_size": int(args.hidden_size),
            "n_layer": int(args.n_layer),
            "n_head": int(args.n_head),
            "dropout": float(args.dropout),
            "state_mean": state_mean.astype(np.float32).tolist(),
            "state_std": state_std.astype(np.float32).tolist(),
            "rtg_scale": float(rtg_scale),
            "dataset_max_return": float(dataset_max_return),
            "args": vars(args),
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(val["loss"]),
            "val_acc": float(val["acc"]),
        }
        torch.save(payload, last_path)
        improved_loss = float(val["loss"]) < best_val_loss - float(args.early_stop_min_delta)
        if improved_loss:
            best_val_loss = float(val["loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            torch.save(payload, best_path)
        else:
            stale_epochs += 1
        if float(val["acc"]) > best_val_acc:
            best_val_acc = float(val["acc"])
            torch.save(payload, best_acc_path)

        line = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(val["loss"]),
            "val_acc": float(val["acc"]),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}"
        )

        if int(args.early_stop_patience) > 0 and stale_epochs >= int(args.early_stop_patience):
            early_stopped = True
            print(
                f"early stop: epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_val_loss={best_val_loss:.4f} best_val_acc={best_val_acc:.4f}"
            )
            break

    summary = {
        "best_ckpt": str(best_path),
        "best_acc_ckpt": str(best_acc_path),
        "last_ckpt": str(last_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "early_stopped": bool(early_stopped),
        "trained_epochs": int(epoch),
    }
    with (outdir / "dt_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _plot_training_curves(path=outdir / "dt_training_curves.png", log_path=log_path)
    print(f"done: best={best_path} best_acc={best_acc_path} last={last_path}")


if __name__ == "__main__":
    main()
