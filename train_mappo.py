from __future__ import annotations

"""
Train MAPPO (CTDE) on the custom multi-agent MCS environment.

MAPPO here uses:
  - shared actor pi(a_i | o_i)
  - centralized critic V(s) where s = concat(o_1 ... o_n)
  - team reward (mean/sum of per-agent rewards) + GAE

Example:
  python train_mappo.py --epochs 500 --episodes-per-epoch 4 --device cuda
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import CONFIG
from env import Environment


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MAPPO policy for MCS action learning.")
    p.add_argument("--outdir", type=str, default="result/mappo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-lstm-summary", action="store_true")
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--episodes-per-epoch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None, help="Optional per-episode step cap.")

    p.add_argument("--actor-hidden-dim", type=int, default=128)
    p.add_argument("--critic-hidden-dim", type=int, default=256)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)

    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--mini-batch-size", type=int, default=256, help="Mini-batch size in time-steps (not agent transitions).")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--team-reward-mode", type=str, choices=["mean", "sum"], default="mean")
    p.add_argument("--stage-snapshot-all", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--eval-max-steps", type=int, default=None)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--save-epoch-interval", type=int, default=0, help=">0 to save epoch_{k}.pt snapshots every K epochs.")

    p.add_argument("--rollout-log-interval", type=int, default=0, help="Print rollout progress every N env steps (<=0 disables).")
    p.add_argument("--update-log-interval", type=int, default=0, help="Print PPO update progress every N mini-batches (<=0 disables).")
    p.add_argument("--log-flush", action="store_true", help="Flush stdout immediately for real-time logs.")
    return p.parse_args()


def _build_stage_epoch_targets(total_epochs: int) -> Dict[str, int]:
    total = max(1, int(total_epochs))
    return {
        "early": max(1, int(np.ceil(total / 3.0))),
        "middle": max(1, int(np.ceil(total * 2.0 / 3.0))),
        "best": total,
    }


def _build_pending_req_info(req: dict, fallback_step: int) -> Dict[str, float]:
    return {
        "step": float(int(req.get("step", fallback_step))),
        "required_kwh": float(req.get("required_kwh", 0.0)),
    }


def _row_metric_business_score(row: Dict[str, float]) -> float:
    eval_score = float(row.get("eval_business_score", np.nan))
    if np.isfinite(eval_score):
        return eval_score
    return float(row.get("business_score", np.nan))


def _smooth_curve(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values.astype(np.float64, copy=True)
    window = int(min(11, max(3, values.size // 30)))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _convergence_midpoint_row(valid_rows: List[Dict[str, float]]) -> Dict[str, float]:
    scores = np.asarray([_row_metric_business_score(r) for r in valid_rows], dtype=np.float64)
    smooth = _smooth_curve(scores)
    best_score = float(np.max(smooth))
    worst_idx = int(np.argmin(smooth))
    worst_score = float(smooth[worst_idx])
    band = max(1e-6, best_score - worst_score)
    threshold = best_score - 0.05 * band

    converge_idx = int(np.argmax(smooth >= threshold))
    if converge_idx <= worst_idx:
        later = np.where(smooth[worst_idx:] >= threshold)[0]
        if later.size > 0:
            converge_idx = int(worst_idx + later[0])
        else:
            converge_idx = int(np.argmax(smooth))

    start_idx = int(worst_idx)
    end_idx = int(converge_idx)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    start_epoch = float(valid_rows[start_idx]["epoch"])
    end_epoch = float(valid_rows[end_idx]["epoch"])
    mid_epoch = (start_epoch + end_epoch) / 2.0
    candidates = valid_rows[start_idx : end_idx + 1] if end_idx >= start_idx else valid_rows
    row = min(candidates, key=lambda r: abs(float(r["epoch"]) - mid_epoch))
    return {
        **row,
        "stage_role": "worst_to_convergence_midpoint",
        "convergence_start_epoch": float(start_epoch),
        "convergence_end_epoch": float(end_epoch),
        "convergence_start_score": float(scores[start_idx]),
        "convergence_end_score": float(scores[end_idx]),
        "convergence_threshold_score": float(threshold),
        "stage_metric_business_score": float(_row_metric_business_score(row)),
    }


def _rapid_rise_midpoint_row(valid_rows: List[Dict[str, float]]) -> Dict[str, float]:
    # Kept for backwards-compatible summaries; middle now uses convergence midpoint.
    scores = np.asarray([_row_metric_business_score(r) for r in valid_rows], dtype=np.float64)
    smooth = _smooth_curve(scores)
    slopes = np.diff(smooth)
    if slopes.size <= 0:
        start_idx = end_idx = 0
    else:
        max_slope = float(np.max(slopes))
        if max_slope <= 0.0:
            start_idx = end_idx = int(np.argmax(scores))
        else:
            peak_idx = int(np.argmax(slopes))
            threshold = 0.30 * max_slope
            start_idx = peak_idx
            end_idx = min(peak_idx + 1, len(valid_rows) - 1)
            while start_idx > 0 and float(slopes[start_idx - 1]) >= threshold:
                start_idx -= 1
            while end_idx < int(slopes.size) and float(slopes[end_idx]) >= threshold:
                end_idx += 1

    start_epoch = float(valid_rows[start_idx]["epoch"])
    end_epoch = float(valid_rows[end_idx]["epoch"])
    mid_epoch = (start_epoch + end_epoch) / 2.0
    lo, hi = sorted((start_idx, end_idx))
    candidates = valid_rows[lo : hi + 1] if hi >= lo else valid_rows
    row = min(candidates, key=lambda r: abs(float(r["epoch"]) - mid_epoch))
    return {
        **row,
        "stage_role": "rapid_rise_midpoint",
        "rapid_rise_start_epoch": float(start_epoch),
        "rapid_rise_end_epoch": float(end_epoch),
        "rapid_rise_start_score": float(scores[start_idx]),
        "rapid_rise_end_score": float(scores[end_idx]),
        "stage_metric_business_score": float(_row_metric_business_score(row)),
    }


def _select_stage_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    valid = sorted(
        [r for r in rows if np.isfinite(_row_metric_business_score(r))],
        key=lambda r: float(r["epoch"]),
    )
    if not valid:
        return {}
    random_row = {
        **valid[0],
        "stage_role": "epoch_001_random",
        "stage_metric_business_score": float(_row_metric_business_score(valid[0])),
    }
    return {"early": random_row, "middle": _convergence_midpoint_row(valid)}


def _event_mcs_income(env: Environment, event: dict, req_info: Optional[Dict[str, float]] = None) -> float:
    mcs_id = int(event.get("mcs_id", -1))
    mcs = env.mcs_by_id.get(mcs_id)
    if mcs is None:
        return 0.0

    distance_km = float(event.get("distance_km", 0.0))
    income = -distance_km * float(mcs.cost_per_km)
    if str(event.get("action", "")) == "serve_request":
        required_kwh = float((req_info or {}).get("required_kwh", 0.0))
        income += required_kwh * float(mcs.price_per_kwh)
    return float(income)


def _business_score(stats: Dict[str, float], args: argparse.Namespace) -> float:
    success = float(stats.get("success_rate", 0.0))
    mcs_success = float(stats.get("mcs_success_rate", 0.0))
    wait = float(stats.get("avg_wait_minutes", 0.0))
    timeouts = float(stats.get("timeout_events_total", 0.0))
    return float(1000.0 * success + 300.0 * mcs_success - 25.0 * wait - 0.05 * timeouts)


def _save_business_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fields = [
        "epoch",
        "reward",
        "success_rate",
        "mcs_avg_income",
        "ev_avg_wait_minutes",
        "business_score",
        "eval_reward",
        "eval_success_rate",
        "eval_mcs_avg_income",
        "eval_ev_avg_wait_minutes",
        "eval_business_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, float("nan")) for k in fields})


def _plot_business_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skip plotting business metrics: {e}")
        return

    epochs = np.asarray([float(r["epoch"]) for r in rows], dtype=np.float32)
    reward = np.asarray([float(r["reward"]) for r in rows], dtype=np.float32)
    success = np.asarray([float(r["success_rate"]) for r in rows], dtype=np.float32)
    income = np.asarray([float(r["mcs_avg_income"]) for r in rows], dtype=np.float32)
    wait_min = np.asarray([float(r["ev_avg_wait_minutes"]) for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        (axes[0, 0], reward, "Reward", "Reward"),
        (axes[0, 1], success, "Success Rate", "Rate"),
        (axes[1, 0], income, "MCS Avg Income", "Income"),
        (axes[1, 1], wait_min, "EV Avg Wait (min)", "Minutes"),
    ]
    for ax, y_train, title, ylabel in plots:
        ax.plot(epochs, y_train, label="train", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _obs_dict_to_matrix(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in agents], axis=0)


def _mask_dict_to_matrix(mask_dict: Dict[str, np.ndarray], agents: List[str], action_dim: int) -> np.ndarray:
    mat = np.stack([np.asarray(mask_dict[a], dtype=np.bool_) for a in agents], axis=0)
    if mat.shape != (len(agents), int(action_dim)):
        raise RuntimeError(f"Invalid mask shape: expected {(len(agents), int(action_dim))}, got {tuple(mat.shape)}")
    return mat


def _build_state(obs_mat: np.ndarray) -> np.ndarray:
    return np.asarray(obs_mat, dtype=np.float32).reshape(-1)


def _aggregate_team_reward(agent_rewards: Dict[str, float], mode: str) -> float:
    vals = [float(v) for v in agent_rewards.values()]
    if not vals:
        return 0.0
    if mode == "sum":
        return float(np.sum(vals))
    return float(np.mean(vals))


def _act_actor(
    actor: ActorNet,
    obs_mat: np.ndarray,
    action_masks: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = actor(obs_t).masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        logps = dist.log_prob(actions)
    return actions.cpu().numpy().astype(np.int64), logps.cpu().numpy().astype(np.float32)


def _value_critic(critic: CriticNet, state: np.ndarray, device: torch.device) -> float:
    st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        v = critic(st)[0]
    return float(v.item())


def save_ckpt(
    path: Path,
    actor: ActorNet,
    critic: CriticNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    state_dim: int,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "n_agents": int(n_agents),
        "state_dim": int(state_dim),
        "metrics": metrics,
    }
    torch.save(payload, path)


def collect_rollouts(
    env: Environment,
    actor: ActorNet,
    critic: CriticNet,
    episodes: int,
    gamma: float,
    gae_lambda: float,
    team_reward_mode: str,
    device: torch.device,
    rng: np.random.Generator,
    action_dim: int,
    max_steps: Optional[int],
    epoch: int = 0,
    rollout_log_interval: int = 0,
    log_flush: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    agents = list(env.agents)
    step_minutes = float(env.config.get("sim_step_minutes", 5))

    obs_seq: List[np.ndarray] = []
    state_seq: List[np.ndarray] = []
    mask_seq: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    old_logp_seq: List[np.ndarray] = []
    value_seq: List[float] = []
    reward_seq: List[float] = []
    adv_seq: List[float] = []
    ret_seq: List[float] = []

    total_steps = 0
    total_requests = 0
    mcs_requests = 0
    success_requests = 0
    mcs_served = 0
    unresolved_mcs_total = 0
    timeout_events_total = 0
    wait_steps_sum = 0.0
    wait_count = 0
    total_agent_reward = 0.0
    total_mcs_income = 0.0
    total_horizon_target = 0

    for ep_idx in range(int(episodes)):
        obs_dict = env.reset(seed=int(rng.integers(1_000_000_000)))
        mcs_pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        episode_done = False
        ep_requests = 0
        ep_mcs_requests = 0
        ep_success_requests = 0
        ep_mcs_served = 0
        ep_timeout_events = 0
        ep_wait_steps_sum = 0.0
        ep_wait_count = 0
        ep_agent_reward = 0.0
        ep_mcs_income = 0.0

        ep_values: List[float] = []
        ep_rewards: List[float] = []
        ep_dones: List[float] = []
        last_next_state: Optional[np.ndarray] = None

        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        total_horizon_target += int(horizon)

        for _step in range(horizon):
            action_mask_dict = env.get_action_mask()
            obs_mat = _obs_dict_to_matrix(obs_dict, agents=agents)
            action_masks = _mask_dict_to_matrix(action_mask_dict, agents=agents, action_dim=action_dim)
            state = _build_state(obs_mat)

            act_idx, logps = _act_actor(
                actor=actor,
                obs_mat=obs_mat,
                action_masks=action_masks,
                device=device,
                deterministic=False,
            )
            value = _value_critic(critic=critic, state=state, device=device)
            env_actions = {agents[i]: int(act_idx[i] + 1) for i in range(len(agents))}

            step_result = env.step_parallel(env_actions)
            executed_steps += 1
            episode_done = bool(step_result.done)

            team_reward = _aggregate_team_reward(step_result.agent_rewards, mode=team_reward_mode)

            obs_seq.append(obs_mat)
            state_seq.append(state)
            mask_seq.append(action_masks)
            action_seq.append(np.asarray(act_idx, dtype=np.int64))
            old_logp_seq.append(np.asarray(logps, dtype=np.float32))
            value_seq.append(float(value))
            reward_seq.append(float(team_reward))
            ep_values.append(float(value))
            ep_rewards.append(float(team_reward))
            ep_dones.append(1.0 if episode_done else 0.0)

            if not episode_done:
                next_obs_dict = env.get_agent_observations()
                next_obs_mat = _obs_dict_to_matrix(next_obs_dict, agents=agents)
                last_next_state = _build_state(next_obs_mat)
            else:
                last_next_state = None

            for req in step_result.requests:
                total_requests += 1
                ep_requests += 1
                if req.get("service_mode") == "fcs":
                    success_requests += 1
                    ep_success_requests += 1
                else:
                    mcs_requests += 1
                    ep_mcs_requests += 1
                    mcs_pending_by_ev[int(req["ev_id"])] = _build_pending_req_info(req=req, fallback_step=int(step_result.step))

            for event in step_result.mcs_events:
                action = str(event.get("action", ""))
                req_info: Optional[Dict[str, float]] = None
                if action == "serve_request":
                    ev_id = int(event["ev_id"])
                    req_info = mcs_pending_by_ev.pop(ev_id, {"step": float(step_result.step), "required_kwh": 0.0})
                income = _event_mcs_income(env=env, event=event, req_info=req_info)
                total_mcs_income += income
                ep_mcs_income += income

                if event.get("action") != "serve_request":
                    continue
                req_step = int((req_info or {}).get("step", float(step_result.step)))
                wait_steps = max(0, int(step_result.step) - int(req_step))
                mcs_served += 1
                ep_mcs_served += 1
                success_requests += 1
                ep_success_requests += 1
                wait_steps_sum += float(wait_steps)
                wait_count += 1
                ep_wait_steps_sum += float(wait_steps)
                ep_wait_count += 1

            for timeout_event in step_result.timeout_events:
                timeout_events_total += 1
                ep_timeout_events += 1
                ev_id = int(timeout_event.get("ev_id", -1))
                req_step = int(timeout_event.get("request_step", step_result.step))
                wait_steps = int(timeout_event.get("wait_steps", max(0, int(step_result.step) - req_step)))
                req_info = mcs_pending_by_ev.pop(ev_id, None)
                if req_info is not None:
                    req_step = int(req_info.get("step", float(req_step)))
                wait_steps_sum += float(wait_steps)
                wait_count += 1
                ep_wait_steps_sum += float(wait_steps)
                ep_wait_count += 1

            step_agent_reward = float(np.sum([float(v) for v in step_result.agent_rewards.values()]))
            total_agent_reward += step_agent_reward
            ep_agent_reward += step_agent_reward

            if rollout_log_interval > 0 and (executed_steps % int(rollout_log_interval) == 0):
                success_rate_now = float(ep_success_requests / max(1, ep_requests))
                avg_wait_now = float(ep_wait_steps_sum / ep_wait_count) if ep_wait_count > 0 else 0.0
                print(
                    f"[rollout][epoch={epoch:03d}] "
                    f"ep={ep_idx + 1}/{int(episodes)} step={executed_steps}/{horizon} "
                    f"succ={success_rate_now:.3f} wait={avg_wait_now * step_minutes:.2f}min "
                    f"pending={len(env.pending_ev_requests)} timeout={ep_timeout_events}",
                    flush=log_flush,
                )

            if episode_done:
                break
            obs_dict = env.get_agent_observations()

        for req_info in mcs_pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            wait_steps = max(0, int(executed_steps - 1) - req_step)
            wait_steps_sum += float(wait_steps)
            wait_count += 1
            ep_wait_steps_sum += float(wait_steps)
            ep_wait_count += 1
        unresolved_mcs_total += len(mcs_pending_by_ev)
        total_steps += executed_steps

        # GAE on team reward with centralized value
        if episode_done or last_next_state is None:
            bootstrap_value = 0.0
        else:
            bootstrap_value = _value_critic(critic=critic, state=last_next_state, device=device)

        ep_adv = np.zeros((len(ep_rewards),), dtype=np.float32)
        ep_ret = np.zeros((len(ep_rewards),), dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in range(len(ep_rewards) - 1, -1, -1):
            delta = ep_rewards[t] + gamma * (1.0 - ep_dones[t]) * next_value - ep_values[t]
            gae = delta + gamma * gae_lambda * (1.0 - ep_dones[t]) * gae
            ep_adv[t] = float(gae)
            ep_ret[t] = float(gae + ep_values[t])
            next_value = float(ep_values[t])

        for t in range(len(ep_rewards)):
            adv_seq.append(float(ep_adv[t]))
            ret_seq.append(float(ep_ret[t]))

        if len(adv_seq) != len(obs_seq):
            raise RuntimeError(
                f"adv/obs length mismatch at episode {ep_idx}: adv={len(adv_seq)} obs={len(obs_seq)}"
            )

        ep_success_rate = float(ep_success_requests / max(1, ep_requests))
        ep_mcs_success_rate = float(ep_mcs_served / max(1, ep_mcs_requests))
        ep_avg_wait_steps = float(ep_wait_steps_sum / ep_wait_count) if ep_wait_count > 0 else 0.0
        if rollout_log_interval > 0:
            print(
                f"[rollout][epoch={epoch:03d}] "
                f"ep={ep_idx + 1}/{int(episodes)} done "
                f"steps={executed_steps}/{horizon} "
                f"succ={ep_success_rate:.3f} mcs_succ={ep_mcs_success_rate:.3f} "
                f"wait={ep_avg_wait_steps * step_minutes:.2f}min timeout={ep_timeout_events} "
                f"reward={ep_agent_reward:.2f} mcs_income={ep_mcs_income:.2f}",
                flush=log_flush,
            )

    if len(obs_seq) == 0:
        raise RuntimeError("No rollout transitions collected.")

    avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
    rollout_stats = {
        "episodes": float(episodes),
        "transitions": float(len(obs_seq) * len(agents)),
        "steps": float(total_steps),
        "steps_target": float(total_horizon_target),
        "requests": float(total_requests),
        "success_rate": float(success_requests / max(1, total_requests)),
        "mcs_success_rate": float(mcs_served / max(1, mcs_requests)),
        "avg_wait_steps": float(avg_wait_steps),
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "unresolved_mcs_total": float(unresolved_mcs_total),
        "timeout_events_total": float(timeout_events_total),
        "avg_total_agent_reward_per_ep": float(total_agent_reward / max(1, int(episodes))),
        "avg_reward_per_transition": float(total_agent_reward / max(1, (len(obs_seq) * len(agents)))),
        "mcs_total_income": float(total_mcs_income),
        "mcs_avg_income": float(total_mcs_income / max(1.0, float(int(episodes) * len(env.mcs_list)))),
    }
    batch = {
        "obs": np.asarray(obs_seq, dtype=np.float32),
        "states": np.asarray(state_seq, dtype=np.float32),
        "masks": np.asarray(mask_seq, dtype=np.bool_),
        "actions": np.asarray(action_seq, dtype=np.int64),
        "old_logps": np.asarray(old_logp_seq, dtype=np.float32),
        "values": np.asarray(value_seq, dtype=np.float32),
        "returns": np.asarray(ret_seq, dtype=np.float32),
        "advantages": np.asarray(adv_seq, dtype=np.float32),
    }
    return batch, rollout_stats


def mappo_update(
    actor: ActorNet,
    critic: CriticNet,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    epoch: int = 0,
    update_log_interval: int = 0,
    log_flush: bool = False,
) -> Dict[str, float]:
    obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    states_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logps_t = torch.as_tensor(batch["old_logps"], dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)

    t_steps, n_agents, obs_dim = obs_t.shape
    action_dim = int(masks_t.shape[-1])
    if int(args.normalize_adv):
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    p_loss_all: List[float] = []
    v_loss_all: List[float] = []
    ent_all: List[float] = []
    kl_all: List[float] = []
    clipfrac_all: List[float] = []
    total_loss_all: List[float] = []
    mb_counter = 0
    mb_total = int(args.update_epochs) * int(np.ceil(float(t_steps) / max(1, int(args.mini_batch_size))))

    for _ in range(int(args.update_epochs)):
        perm = rng.permutation(t_steps)
        for start in range(0, t_steps, int(args.mini_batch_size)):
            idx_np = perm[start : start + int(args.mini_batch_size)]
            idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=device)

            mb_states = states_t.index_select(0, idx_t)
            mb_returns = returns_t.index_select(0, idx_t)
            mb_adv = adv_t.index_select(0, idx_t)

            mb_obs = obs_t.index_select(0, idx_t).reshape(-1, obs_dim)
            mb_masks = masks_t.index_select(0, idx_t).reshape(-1, action_dim)
            mb_actions = actions_t.index_select(0, idx_t).reshape(-1)
            mb_old_logps = old_logps_t.index_select(0, idx_t).reshape(-1)
            mb_adv_flat = mb_adv.repeat_interleave(n_agents)

            logits = actor(mb_obs).masked_fill(~mb_masks, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logps - mb_old_logps)
            surr1 = ratio * mb_adv_flat
            surr2 = torch.clamp(ratio, 1.0 - float(args.ppo_clip), 1.0 + float(args.ppo_clip)) * mb_adv_flat
            p_loss = -torch.min(surr1, surr2).mean()

            values = critic(mb_states)
            v_loss = F.mse_loss(values, mb_returns)
            total_loss = p_loss + float(args.vf_coef) * v_loss - float(args.ent_coef) * entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), float(args.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_logps - new_logps).mean()
                clip_frac = (torch.abs(ratio - 1.0) > float(args.ppo_clip)).float().mean()

            p_loss_all.append(float(p_loss.item()))
            v_loss_all.append(float(v_loss.item()))
            ent_all.append(float(entropy.item()))
            kl_all.append(float(approx_kl.item()))
            clipfrac_all.append(float(clip_frac.item()))
            total_loss_all.append(float(total_loss.item()))
            mb_counter += 1

            if update_log_interval > 0 and (mb_counter % int(update_log_interval) == 0 or mb_counter == mb_total):
                print(
                    f"[update][epoch={epoch:03d}] mb={mb_counter}/{mb_total} "
                    f"loss={float(np.mean(total_loss_all)):.4f} "
                    f"policy={float(np.mean(p_loss_all)):.4f} "
                    f"value={float(np.mean(v_loss_all)):.4f} "
                    f"ent={float(np.mean(ent_all)):.4f} "
                    f"kl={float(np.mean(kl_all)):.5f} "
                    f"clip={float(np.mean(clipfrac_all)):.4f}",
                    flush=log_flush,
                )

    return {
        "loss": float(np.mean(total_loss_all)),
        "policy_loss": float(np.mean(p_loss_all)),
        "value_loss": float(np.mean(v_loss_all)),
        "entropy": float(np.mean(ent_all)),
        "approx_kl": float(np.mean(kl_all)),
        "clipfrac": float(np.mean(clipfrac_all)),
    }


def evaluate_policy(
    actor: ActorNet,
    env_config: dict,
    episodes: int,
    seed: int,
    device: torch.device,
    action_dim: int,
    deterministic: bool,
    max_steps: Optional[int],
) -> Dict[str, float]:
    if episodes <= 0:
        return {}

    env = Environment(config=env_config, seed=seed)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    agents = list(env.agents)

    total_steps = 0
    total_requests = 0
    mcs_requests = 0
    success_requests = 0
    mcs_served = 0
    unresolved_mcs_total = 0
    timeout_events_total = 0
    wait_steps_sum = 0.0
    wait_count = 0
    total_agent_reward = 0.0
    total_mcs_income = 0.0

    for ep in range(int(episodes)):
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        mcs_pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)

        for _ in range(horizon):
            action_mask = env.get_action_mask()
            obs_mat = _obs_dict_to_matrix(obs_dict, agents=agents)
            action_masks = _mask_dict_to_matrix(action_mask, agents=agents, action_dim=action_dim)
            act_idx, _ = _act_actor(
                actor=actor,
                obs_mat=obs_mat,
                action_masks=action_masks,
                device=device,
                deterministic=deterministic,
            )
            env_actions = {agents[i]: int(act_idx[i] + 1) for i in range(len(agents))}

            step_result = env.step_parallel(env_actions)
            executed_steps += 1

            for req in step_result.requests:
                total_requests += 1
                if req.get("service_mode") == "fcs":
                    success_requests += 1
                else:
                    mcs_requests += 1
                    mcs_pending_by_ev[int(req["ev_id"])] = _build_pending_req_info(req=req, fallback_step=int(step_result.step))

            for event in step_result.mcs_events:
                action = str(event.get("action", ""))
                req_info: Optional[Dict[str, float]] = None
                if action == "serve_request":
                    ev_id = int(event["ev_id"])
                    req_info = mcs_pending_by_ev.pop(ev_id, {"step": float(step_result.step), "required_kwh": 0.0})
                total_mcs_income += _event_mcs_income(env=env, event=event, req_info=req_info)

                if event.get("action") != "serve_request":
                    continue
                req_step = int((req_info or {}).get("step", float(step_result.step)))
                wait_steps = max(0, int(step_result.step) - int(req_step))
                mcs_served += 1
                success_requests += 1
                wait_steps_sum += float(wait_steps)
                wait_count += 1

            for timeout_event in step_result.timeout_events:
                timeout_events_total += 1
                ev_id = int(timeout_event.get("ev_id", -1))
                req_step = int(timeout_event.get("request_step", step_result.step))
                wait_steps = int(timeout_event.get("wait_steps", max(0, int(step_result.step) - req_step)))
                req_info = mcs_pending_by_ev.pop(ev_id, None)
                if req_info is not None:
                    req_step = int(req_info.get("step", float(req_step)))
                wait_steps_sum += float(wait_steps)
                wait_count += 1

            for agent in env.agents:
                total_agent_reward += float(step_result.agent_rewards[agent])

            if step_result.done:
                break
            obs_dict = env.get_agent_observations()

        for req_info in mcs_pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            wait_steps = max(0, int(executed_steps - 1) - req_step)
            wait_steps_sum += float(wait_steps)
            wait_count += 1
        unresolved_mcs_total += len(mcs_pending_by_ev)
        total_steps += executed_steps

    avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
    return {
        "episodes": float(episodes),
        "steps": float(total_steps),
        "requests": float(total_requests),
        "success_rate": float(success_requests / max(1, total_requests)),
        "mcs_success_rate": float(mcs_served / max(1, mcs_requests)),
        "avg_wait_steps": float(avg_wait_steps),
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "unresolved_mcs_total": float(unresolved_mcs_total),
        "timeout_events_total": float(timeout_events_total),
        "avg_total_agent_reward_per_ep": float(total_agent_reward / max(1, int(episodes))),
        "mcs_total_income": float(total_mcs_income),
        "mcs_avg_income": float(total_mcs_income / max(1.0, float(int(episodes) * len(env.mcs_list)))),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mappo_cfg = dict(CONFIG)
    mappo_cfg["use_lstm_summary"] = bool(args.use_lstm_summary)
    if args.lstm_predictor_ckpt:
        mappo_cfg["lstm_predictor_ckpt"] = str(args.lstm_predictor_ckpt)

    env = Environment(config=mappo_cfg, seed=args.seed)
    obs_dict = env.reset(seed=args.seed)
    obs_dim = int(next(iter(obs_dict.values())).shape[0])
    n_agents = int(len(env.agents))
    action_dim = int(len(env.ACTION_SPACE))
    state_dim = int(obs_dim * n_agents)

    actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(args.actor_hidden_dim)).to(device)
    critic = CriticNet(state_dim=state_dim, hidden_dim=int(args.critic_hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": float(args.lr_actor)},
            {"params": critic.parameters(), "lr": float(args.lr_critic)},
        ],
        weight_decay=float(args.weight_decay),
    )

    total_epochs = int(args.epochs)
    _build_stage_epoch_targets(total_epochs=total_epochs)

    best_sim_business = float("-inf")
    best_sim_success = float("-inf")
    best_sim_reward = float("-inf")
    best_sim_wait = float("inf")
    log_path = outdir / "mappo_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    stage_snapshot_dir = outdir / "stage_snapshots"
    if bool(args.stage_snapshot_all):
        stage_snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"mappo_init obs_dim={obs_dim} state_dim={state_dim} action_dim={action_dim} n_agents={n_agents} "
        f"device={device} epochs={args.epochs} use_lstm_summary={bool(args.use_lstm_summary)}"
    )
    print(
        "stage_ckpt_targets early=epoch_001 middle=worst_to_convergence_midpoint business=best_by_business",
        flush=bool(args.log_flush),
    )
    business_history: List[Dict[str, float]] = []

    for epoch in range(1, total_epochs + 1):
        actor.train()
        critic.train()
        rollout_batch, rollout_stats = collect_rollouts(
            env=env,
            actor=actor,
            critic=critic,
            episodes=int(args.episodes_per_epoch),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            team_reward_mode=str(args.team_reward_mode),
            device=device,
            rng=rng,
            action_dim=action_dim,
            max_steps=args.max_steps,
            epoch=epoch,
            rollout_log_interval=int(args.rollout_log_interval),
            log_flush=bool(args.log_flush),
        )
        update_stats = mappo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            batch=rollout_batch,
            args=args,
            device=device,
            rng=rng,
            epoch=epoch,
            update_log_interval=int(args.update_log_interval),
            log_flush=bool(args.log_flush),
        )

        eval_stats: Dict[str, float] = {}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            actor.eval()
            eval_stats = evaluate_policy(
                actor=actor,
                env_config=mappo_cfg,
                episodes=int(args.eval_episodes),
                seed=int(args.eval_seed + epoch),
                device=device,
                action_dim=action_dim,
                deterministic=not bool(args.eval_stochastic),
                max_steps=args.eval_max_steps,
            )

        collect_business_score = _business_score(rollout_stats, args)
        eval_business_score = _business_score(eval_stats, args) if eval_stats else float("nan")
        metric_stats = eval_stats if eval_stats else rollout_stats
        metric_business_score = float(eval_business_score) if eval_stats else float(collect_business_score)
        merged_for_ckpt = {**rollout_stats, **update_stats, **eval_stats, "business_score": float(metric_business_score)}
        save_ckpt(
            path=outdir / "last.pt",
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            state_dim=state_dim,
            metrics=merged_for_ckpt,
        )
        if int(args.save_epoch_interval) > 0 and epoch % int(args.save_epoch_interval) == 0:
            save_ckpt(
                path=outdir / f"epoch_{epoch:03d}.pt",
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                obs_dim=obs_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                state_dim=state_dim,
                metrics=merged_for_ckpt,
            )
        if bool(args.stage_snapshot_all):
            save_ckpt(
                path=stage_snapshot_dir / f"epoch_{epoch:03d}.pt",
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                obs_dim=obs_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                state_dim=state_dim,
                metrics=merged_for_ckpt,
            )
        if eval_stats:
            sim_business = float(metric_business_score)
            if sim_business > best_sim_business:
                best_sim_business = sim_business
                save_ckpt(
                    path=outdir / "best.pt",
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    state_dim=state_dim,
                    metrics=merged_for_ckpt,
                )
                save_ckpt(
                    path=outdir / "best_by_business.pt",
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    state_dim=state_dim,
                    metrics=merged_for_ckpt,
                )
            sim_success = float(eval_stats.get("success_rate", -1.0))
            if sim_success > best_sim_success:
                best_sim_success = sim_success
                save_ckpt(
                    path=outdir / "best_by_success.pt",
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    state_dim=state_dim,
                    metrics=merged_for_ckpt,
                )
            sim_reward = float(eval_stats.get("avg_total_agent_reward_per_ep", float("-inf")))
            if sim_reward > best_sim_reward:
                best_sim_reward = sim_reward
                save_ckpt(
                    path=outdir / "best_by_reward.pt",
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    state_dim=state_dim,
                    metrics=merged_for_ckpt,
                )
            sim_wait = float(eval_stats.get("avg_wait_minutes", float("inf")))
            if sim_wait < best_sim_wait:
                best_sim_wait = sim_wait
                save_ckpt(
                    path=outdir / "best_by_wait.pt",
                    actor=actor,
                    critic=critic,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    state_dim=state_dim,
                    metrics=merged_for_ckpt,
                )

        summary = {
            "epoch": int(epoch),
            "collect": rollout_stats,
            "update": update_stats,
            "eval": eval_stats,
            "lr_actor": float(optimizer.param_groups[0]["lr"]),
            "lr_critic": float(optimizer.param_groups[1]["lr"]),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        biz_row = {
            "epoch": float(epoch),
            "reward": float(rollout_stats.get("avg_total_agent_reward_per_ep", 0.0)),
            "success_rate": float(rollout_stats.get("success_rate", 0.0)),
            "mcs_avg_income": float(rollout_stats.get("mcs_avg_income", 0.0)),
            "ev_avg_wait_minutes": float(rollout_stats.get("avg_wait_minutes", 0.0)),
            "business_score": float(collect_business_score),
            "eval_reward": float(eval_stats.get("avg_total_agent_reward_per_ep", np.nan)) if eval_stats else float("nan"),
            "eval_success_rate": float(eval_stats.get("success_rate", np.nan)) if eval_stats else float("nan"),
            "eval_mcs_avg_income": float(eval_stats.get("mcs_avg_income", np.nan)) if eval_stats else float("nan"),
            "eval_ev_avg_wait_minutes": float(eval_stats.get("avg_wait_minutes", np.nan)) if eval_stats else float("nan"),
            "eval_business_score": float(eval_business_score),
        }
        business_history.append(biz_row)
        print(
            f"epoch={epoch:03d} "
            f"collect_success={rollout_stats['success_rate']:.3f} "
            f"collect_wait={rollout_stats['avg_wait_minutes']:.2f}min "
            f"reward={biz_row['reward']:.3f} "
            f"biz={biz_row['business_score']:.2f}",
            flush=bool(args.log_flush),
        )

    biz_csv = outdir / "business_metrics.csv"
    biz_png = outdir / "business_metrics.png"
    _save_business_metrics_csv(path=biz_csv, rows=business_history)
    _plot_business_metrics(path=biz_png, rows=business_history)
    selected_stage_rows = _select_stage_rows(business_history)
    if bool(args.stage_snapshot_all):
        for stage_name, row in selected_stage_rows.items():
            epoch_num = int(float(row["epoch"]))
            src = stage_snapshot_dir / f"epoch_{epoch_num:03d}.pt"
            dst = outdir / f"stage_{stage_name}.pt"
            if src.exists():
                shutil.copy2(src, dst)
                ckpt = torch.load(dst, map_location="cpu")
                ckpt["metrics"] = {**ckpt.get("metrics", {}), **row}
                torch.save(ckpt, dst)
                print(
                    f"selected stage_{stage_name}.pt epoch={epoch_num:03d} "
                    f"score={row['stage_metric_business_score']:.2f} role={row.get('stage_role', stage_name)}",
                    flush=bool(args.log_flush),
                )
    best_summary = {
        "best_by_business": float(best_sim_business),
        "best_by_success": float(best_sim_success),
        "best_by_reward": float(best_sim_reward),
        "best_by_wait": float(best_sim_wait),
        "stage_selection": selected_stage_rows,
    }
    with (outdir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)
    print(f"saved business metrics: {biz_csv}")
    print(f"saved business plot: {biz_png}")
    print(f"saved best summary: {outdir / 'best_summary.json'}")
    print(f"done: checkpoints/log in {outdir}")


if __name__ == "__main__":
    main()
