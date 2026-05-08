from __future__ import annotations

"""
Online fine-tuning: initialize policy from DT backbone, then optimize with PPO + KL to reference DT.
"""

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import CONFIG
from decision_transformer.models.decision_transformer import DecisionTransformer
from env import Environment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DT backbone online with PPO.")
    p.add_argument("--dt-ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, default="result/dt_ppo_ft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-lstm-summary", action="store_true")
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")
    p.add_argument("--target-return", type=float, default=0.0, help="<=0 means auto from offline dataset max RTG * scale.")
    p.add_argument("--target-return-scale", type=float, default=1.0)
    p.add_argument("--offline-dataset", type=str, default="", help="Path to offline dataset .npz for auto target return.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--episodes-per-epoch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--context-len", type=int, default=0, help="<=0 uses DT checkpoint context length.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-clip", type=float, default=0.05)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--mini-batch-size", type=int, default=1024)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--kl-coef", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--policy-warmup-epochs", type=int, default=10, help="Freeze the DT policy trunk and train only value head during early epochs.")
    p.add_argument("--warmup-kl-mult", type=float, default=4.0)
    p.add_argument("--reward-profile", type=str, default="config", choices=["business", "config"])
    p.add_argument("--stability-guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--guard-until-epoch", type=int, default=80)
    p.add_argument("--guard-success-drop", type=float, default=0.08)
    p.add_argument("--guard-wait-increase-minutes", type=float, default=5.0)
    p.add_argument("--guard-lr-decay", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--eval-max-steps", type=int, default=None)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--log-flush", action="store_true")
    return p.parse_args()


def _load_dt_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _compute_dataset_max_return(dataset_path: Path) -> float:
    d = np.load(dataset_path, allow_pickle=True)
    if "returns_to_go" not in d:
        raise RuntimeError(f"`returns_to_go` not found in dataset: {dataset_path}")
    vals: List[float] = []
    for arr in d["returns_to_go"]:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.size > 0:
            vals.append(float(np.max(a)))
    if not vals:
        raise RuntimeError(f"No return values found in dataset: {dataset_path}")
    return float(np.max(vals))


def _build_pending_req_info(req: dict, fallback_step: int) -> Dict[str, float]:
    return {
        "step": float(int(req.get("step", fallback_step))),
        "required_kwh": float(req.get("required_kwh", 0.0)),
    }


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
        "ref_kl",
        "approx_kl",
        "guard_rollback",
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

    eval_reward = np.asarray([float(r.get("eval_reward", np.nan)) for r in rows], dtype=np.float32)
    eval_success = np.asarray([float(r.get("eval_success_rate", np.nan)) for r in rows], dtype=np.float32)
    eval_income = np.asarray([float(r.get("eval_mcs_avg_income", np.nan)) for r in rows], dtype=np.float32)
    eval_wait = np.asarray([float(r.get("eval_ev_avg_wait_minutes", np.nan)) for r in rows], dtype=np.float32)
    business_score = np.asarray([float(r.get("business_score", np.nan)) for r in rows], dtype=np.float32)
    eval_business_score = np.asarray([float(r.get("eval_business_score", np.nan)) for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        (axes[0, 0], reward, eval_reward, "Reward", "Reward"),
        (axes[0, 1], success, eval_success, "Success Rate", "Rate"),
        (axes[1, 0], income, eval_income, "MCS Avg Income", "Income"),
        (axes[1, 1], wait_min, eval_wait, "EV Avg Wait (min)", "Minutes"),
    ]
    for ax, y_train, y_eval, title, ylabel in plots:
        ax.plot(epochs, y_train, label="train", linewidth=2.0)
        if np.any(np.isfinite(y_eval)):
            ax.plot(epochs, y_eval, "--", label="eval", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    if np.any(np.isfinite(business_score)) or np.any(np.isfinite(eval_business_score)):
        fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
        if np.any(np.isfinite(business_score)):
            ax.plot(epochs, business_score, label="train", linewidth=2.0)
        if np.any(np.isfinite(eval_business_score)):
            ax.plot(epochs, eval_business_score, "--", label="eval", linewidth=1.8)
        ax.set_title("Business Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(path.with_name("business_score.png"), dpi=180)
        plt.close(fig2)


def _business_score(stats: Dict[str, float], args: argparse.Namespace) -> float:
    success = float(stats.get("success_rate", 0.0))
    mcs_success = float(stats.get("mcs_success_rate", 0.0))
    wait = float(stats.get("avg_wait_minutes", 0.0))
    timeouts = float(stats.get("timeout_events_total", 0.0))
    return float(1000.0 * success + 300.0 * mcs_success - 25.0 * wait - 0.05 * timeouts)


def _apply_reward_profile(env_cfg: dict, profile: str) -> dict:
    if str(profile) != "business":
        return env_cfg
    env_cfg.update(
        {
            "reward_service_reward": 6.0,
            "reward_fast_service_bonus": 3.0,
            "reward_waiting_penalty": 0.18,
            "reward_serve_wait_penalty": 0.10,
            "reward_timeout_penalty": 6.0,
            "reward_timeout_wait_penalty": 0.12,
            "reward_pending_count_penalty": 0.025,
            "reward_empty_drive_penalty": 0.012,
            "reward_fcs_overload_penalty": 0.10,
            "reward_crowd_penalty": 0.03,
            "reward_invalid_action_penalty": 3.0,
            "reward_success_rate_bonus": 2.0,
            "reward_mcs_success_rate_bonus": 1.0,
            "reward_wait_improvement_bonus": 0.8,
            "reward_income_scale": 0.02,
            "reward_shape_relocate_scale": 0.08,
            "reward_shape_reinforce_scale": 0.10,
            "reward_shape_stay_scale": 0.05,
            "reward_shape_clip": 0.15,
            "reward_clip_abs": 12.0,
        }
    )
    return env_cfg


def _build_dt_from_ckpt(ckpt: dict, device: torch.device) -> DecisionTransformer:
    model = DecisionTransformer(
        state_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["action_dim"]),
        hidden_size=int(ckpt["hidden_size"]),
        max_length=int(ckpt["context_len"]),
        max_ep_len=int(ckpt["max_ep_len"]),
        action_tanh=False,
        n_layer=int(ckpt["n_layer"]),
        n_head=int(ckpt["n_head"]),
        n_inner=4 * int(ckpt["hidden_size"]),
        resid_pdrop=float(ckpt["dropout"]),
        attn_pdrop=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _forward_dt_logits(
    dt: DecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_length = states.shape[0], states.shape[1]
    state_embeddings = dt.embed_state(states)
    action_embeddings = dt.embed_action(actions)
    returns_embeddings = dt.embed_return(returns_to_go)
    time_embeddings = dt.embed_timestep(timesteps)

    state_embeddings = state_embeddings + time_embeddings
    action_embeddings = action_embeddings + time_embeddings
    returns_embeddings = returns_embeddings + time_embeddings

    stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
    stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, dt.hidden_size)
    stacked_inputs = dt.embed_ln(stacked_inputs)

    stacked_attention = torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
    stacked_attention = stacked_attention.permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

    x = dt.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention)["last_hidden_state"]
    x = x.reshape(batch_size, seq_length, 3, dt.hidden_size).permute(0, 2, 1, 3)
    state_token = x[:, 1]
    return dt.predict_action(state_token)


class DTPolicyWithValue(nn.Module):
    def __init__(self, dt_model: DecisionTransformer, state_mean: np.ndarray, state_std: np.ndarray, rtg_scale: float) -> None:
        super().__init__()
        self.dt = dt_model
        self.value_head = nn.Linear(int(self.dt.hidden_size), 1)
        sm = np.asarray(state_mean, dtype=np.float32).reshape(1, 1, -1)
        ss = np.asarray(state_std, dtype=np.float32).reshape(1, 1, -1)
        self.register_buffer("state_mean", torch.as_tensor(sm, dtype=torch.float32), persistent=False)
        self.register_buffer("state_std", torch.as_tensor(ss, dtype=torch.float32), persistent=False)
        self.rtg_scale = float(max(1e-6, rtg_scale))

    def _normalize(self, states: torch.Tensor, returns_to_go: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (states - self.state_mean) / self.state_std, returns_to_go / self.rtg_scale

    def forward_seq(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states_n, returns_n = self._normalize(states, returns_to_go)
        batch_size, seq_length = states_n.shape[0], states_n.shape[1]
        state_embeddings = self.dt.embed_state(states_n)
        action_embeddings = self.dt.embed_action(actions)
        returns_embeddings = self.dt.embed_return(returns_n)
        time_embeddings = self.dt.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.dt.hidden_size)
        stacked_inputs = self.dt.embed_ln(stacked_inputs)

        stacked_attention = torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
        stacked_attention = stacked_attention.permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        x = self.dt.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention)["last_hidden_state"]
        x = x.reshape(batch_size, seq_length, 3, self.dt.hidden_size).permute(0, 2, 1, 3)
        state_token = x[:, 1]
        logits = self.dt.predict_action(state_token)
        values = self.value_head(state_token).squeeze(-1)
        return logits, values

    def act(
        self,
        ctx_states: np.ndarray,
        ctx_actions: np.ndarray,
        ctx_returns: np.ndarray,
        ctx_steps: np.ndarray,
        ctx_attention: np.ndarray,
        action_mask: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        st = torch.as_tensor(ctx_states, dtype=torch.float32, device=device).unsqueeze(0)
        ac = torch.as_tensor(ctx_actions, dtype=torch.float32, device=device).unsqueeze(0)
        rt = torch.as_tensor(ctx_returns, dtype=torch.float32, device=device).unsqueeze(0)
        ts = torch.as_tensor(ctx_steps, dtype=torch.long, device=device).unsqueeze(0)
        at = torch.as_tensor(ctx_attention, dtype=torch.long, device=device).unsqueeze(0)
        am = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits_seq, values_seq = self.forward_seq(st, ac, rt, ts, at)
            li = int(torch.sum(at, dim=1).item()) - 1
            logits = logits_seq[:, li, :].masked_fill(~am, -1e9)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            value = values_seq[:, li]
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())


def _build_context(
    history: dict,
    obs_now: np.ndarray,
    step_now: int,
    rtg_now: float,
    context_len: int,
    action_dim: int,
    max_ep_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero_a = np.zeros((action_dim,), dtype=np.float32)
    s_seq = history["states"] + [np.asarray(obs_now, dtype=np.float32)]
    a_seq = history["actions"] + [zero_a]
    r_seq = history["returns"] + [np.asarray([rtg_now], dtype=np.float32)]
    t_seq = history["steps"] + [int(np.clip(step_now, 0, max_ep_len - 1))]

    if len(s_seq) > context_len:
        s_seq = s_seq[-context_len:]
        a_seq = a_seq[-context_len:]
        r_seq = r_seq[-context_len:]
        t_seq = t_seq[-context_len:]

    seq = len(s_seq)
    pad = context_len - seq
    obs_dim = int(np.asarray(obs_now).shape[0])
    ctx_states = np.zeros((context_len, obs_dim), dtype=np.float32)
    ctx_actions = np.zeros((context_len, action_dim), dtype=np.float32)
    ctx_returns = np.zeros((context_len, 1), dtype=np.float32)
    ctx_steps = np.zeros((context_len,), dtype=np.int64)
    ctx_attention = np.zeros((context_len,), dtype=np.int64)
    ctx_states[pad:] = np.asarray(s_seq, dtype=np.float32)
    ctx_actions[pad:] = np.asarray(a_seq, dtype=np.float32)
    ctx_returns[pad:] = np.asarray(r_seq, dtype=np.float32)
    ctx_steps[pad:] = np.asarray(t_seq, dtype=np.int64)
    ctx_attention[pad:] = 1
    return ctx_states, ctx_actions, ctx_returns, ctx_steps, ctx_attention


def _init_agent_histories(agents: List[str]) -> Dict[str, dict]:
    return {a: {"states": [], "actions": [], "returns": [], "steps": [], "cum_reward": 0.0} for a in agents}


def _init_episode_buffers(agents: List[str]) -> Dict[str, dict]:
    return {
        a: {
            "ctx_states": [],
            "ctx_actions": [],
            "ctx_returns": [],
            "ctx_steps": [],
            "ctx_attention": [],
            "action_mask": [],
            "action": [],
            "logp": [],
            "value": [],
            "reward": [],
            "done": [],
        }
        for a in agents
    }


def collect_rollouts(
    env: Environment,
    policy: DTPolicyWithValue,
    episodes: int,
    target_return: float,
    gamma: float,
    gae_lambda: float,
    context_len: int,
    max_ep_len: int,
    action_dim: int,
    device: torch.device,
    rng: np.random.Generator,
    max_steps: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    ctx_states_all: List[np.ndarray] = []
    ctx_actions_all: List[np.ndarray] = []
    ctx_returns_all: List[np.ndarray] = []
    ctx_steps_all: List[np.ndarray] = []
    ctx_attention_all: List[np.ndarray] = []
    action_masks_all: List[np.ndarray] = []
    actions_all: List[int] = []
    old_logps_all: List[float] = []
    returns_all: List[float] = []
    adv_all: List[float] = []
    values_all: List[float] = []

    step_minutes = float(env.config.get("sim_step_minutes", 5))
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

    for _ep in range(int(episodes)):
        obs_dict = env.reset(seed=int(rng.integers(1_000_000_000)))
        agents = env.agents
        histories = _init_agent_histories(agents)
        per_agent = _init_episode_buffers(agents)
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        mcs_pending_by_ev: Dict[int, Dict[str, float]] = {}
        obs_snapshot: Dict[str, np.ndarray] = {}
        rtg_snapshot: Dict[str, float] = {}
        executed_steps = 0
        episode_done = False
        next_obs_dict: Optional[Dict[str, np.ndarray]] = None

        for _ in range(horizon):
            mask_dict = env.get_action_mask()
            env_actions: Dict[str, int] = {}
            obs_snapshot.clear()
            rtg_snapshot.clear()

            for a in agents:
                obs_now = np.asarray(obs_dict[a], dtype=np.float32)
                rtg_now = float(target_return - histories[a]["cum_reward"])
                ctx = _build_context(
                    history=histories[a],
                    obs_now=obs_now,
                    step_now=int(env.current_step),
                    rtg_now=rtg_now,
                    context_len=context_len,
                    action_dim=action_dim,
                    max_ep_len=max_ep_len,
                )
                am = np.asarray(mask_dict[a], dtype=np.bool_)
                act_idx, logp, value = policy.act(
                    ctx_states=ctx[0],
                    ctx_actions=ctx[1],
                    ctx_returns=ctx[2],
                    ctx_steps=ctx[3],
                    ctx_attention=ctx[4],
                    action_mask=am,
                    device=device,
                    deterministic=False,
                )
                env_actions[a] = int(act_idx + 1)
                obs_snapshot[a] = obs_now
                rtg_snapshot[a] = rtg_now

                per_agent[a]["ctx_states"].append(ctx[0])
                per_agent[a]["ctx_actions"].append(ctx[1])
                per_agent[a]["ctx_returns"].append(ctx[2])
                per_agent[a]["ctx_steps"].append(ctx[3])
                per_agent[a]["ctx_attention"].append(ctx[4])
                per_agent[a]["action_mask"].append(am)
                per_agent[a]["action"].append(int(act_idx))
                per_agent[a]["logp"].append(float(logp))
                per_agent[a]["value"].append(float(value))

            step_result = env.step_parallel(env_actions)
            executed_steps += 1
            episode_done = bool(step_result.done)

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

            for a in agents:
                rew = float(step_result.agent_rewards[a])
                per_agent[a]["reward"].append(rew)
                per_agent[a]["done"].append(1.0 if episode_done else 0.0)
                histories[a]["cum_reward"] += rew
                total_agent_reward += rew

                onehot = np.zeros((action_dim,), dtype=np.float32)
                onehot[int(per_agent[a]["action"][-1])] = 1.0
                histories[a]["states"].append(obs_snapshot[a])
                histories[a]["actions"].append(onehot)
                histories[a]["returns"].append(np.asarray([rtg_snapshot[a]], dtype=np.float32))
                histories[a]["steps"].append(int(np.clip(step_result.step, 0, max_ep_len - 1)))

            if episode_done:
                next_obs_dict = None
                break
            obs_dict = env.get_agent_observations()
            next_obs_dict = obs_dict

        for req_info in mcs_pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            wait_steps = max(0, int(executed_steps - 1) - req_step)
            wait_steps_sum += float(wait_steps)
            wait_count += 1
        unresolved_mcs_total += len(mcs_pending_by_ev)
        total_steps += executed_steps

        for a in agents:
            rew_np = np.asarray(per_agent[a]["reward"], dtype=np.float32)
            done_np = np.asarray(per_agent[a]["done"], dtype=np.float32)
            val_np = np.asarray(per_agent[a]["value"], dtype=np.float32)
            if rew_np.size == 0:
                continue

            if episode_done or next_obs_dict is None:
                bootstrap_value = 0.0
            else:
                obs_now = np.asarray(next_obs_dict[a], dtype=np.float32)
                rtg_now = float(target_return - histories[a]["cum_reward"])
                ctx_next = _build_context(
                    history=histories[a],
                    obs_now=obs_now,
                    step_now=int(env.current_step),
                    rtg_now=rtg_now,
                    context_len=context_len,
                    action_dim=action_dim,
                    max_ep_len=max_ep_len,
                )
                with torch.no_grad():
                    st = torch.as_tensor(ctx_next[0], dtype=torch.float32, device=device).unsqueeze(0)
                    ac = torch.as_tensor(ctx_next[1], dtype=torch.float32, device=device).unsqueeze(0)
                    rt = torch.as_tensor(ctx_next[2], dtype=torch.float32, device=device).unsqueeze(0)
                    ts = torch.as_tensor(ctx_next[3], dtype=torch.long, device=device).unsqueeze(0)
                    at = torch.as_tensor(ctx_next[4], dtype=torch.long, device=device).unsqueeze(0)
                    _, v_seq = policy.forward_seq(st, ac, rt, ts, at)
                    li = int(torch.sum(at, dim=1).item()) - 1
                    bootstrap_value = float(v_seq[0, li].item())

            adv_np = np.zeros_like(rew_np, dtype=np.float32)
            ret_np = np.zeros_like(rew_np, dtype=np.float32)
            gae = 0.0
            next_value = float(bootstrap_value)
            for t in range(len(rew_np) - 1, -1, -1):
                delta = rew_np[t] + gamma * (1.0 - done_np[t]) * next_value - val_np[t]
                gae = delta + gamma * gae_lambda * (1.0 - done_np[t]) * gae
                adv_np[t] = gae
                ret_np[t] = gae + val_np[t]
                next_value = float(val_np[t])

            ctx_states_all.extend(per_agent[a]["ctx_states"])
            ctx_actions_all.extend(per_agent[a]["ctx_actions"])
            ctx_returns_all.extend(per_agent[a]["ctx_returns"])
            ctx_steps_all.extend(per_agent[a]["ctx_steps"])
            ctx_attention_all.extend(per_agent[a]["ctx_attention"])
            action_masks_all.extend(per_agent[a]["action_mask"])
            actions_all.extend(per_agent[a]["action"])
            old_logps_all.extend(per_agent[a]["logp"])
            values_all.extend(per_agent[a]["value"])
            returns_all.extend(ret_np.tolist())
            adv_all.extend(adv_np.tolist())

    if len(actions_all) == 0:
        raise RuntimeError("No transitions collected.")

    avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
    batch = {
        "ctx_states": np.asarray(ctx_states_all, dtype=np.float32),
        "ctx_actions": np.asarray(ctx_actions_all, dtype=np.float32),
        "ctx_returns": np.asarray(ctx_returns_all, dtype=np.float32),
        "ctx_steps": np.asarray(ctx_steps_all, dtype=np.int64),
        "ctx_attention": np.asarray(ctx_attention_all, dtype=np.int64),
        "action_masks": np.asarray(action_masks_all, dtype=np.bool_),
        "actions": np.asarray(actions_all, dtype=np.int64),
        "old_logps": np.asarray(old_logps_all, dtype=np.float32),
        "returns": np.asarray(returns_all, dtype=np.float32),
        "advantages": np.asarray(adv_all, dtype=np.float32),
        "values": np.asarray(values_all, dtype=np.float32),
    }
    stats = {
        "episodes": float(episodes),
        "transitions": float(len(actions_all)),
        "steps": float(total_steps),
        "requests": float(total_requests),
        "success_rate": float(success_requests / max(1, total_requests)),
        "mcs_success_rate": float(mcs_served / max(1, mcs_requests)),
        "avg_wait_steps": float(avg_wait_steps),
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "unresolved_mcs_total": float(unresolved_mcs_total),
        "timeout_events_total": float(timeout_events_total),
        "avg_total_agent_reward_per_ep": float(total_agent_reward / max(1, int(episodes))),
        "avg_reward_per_transition": float(total_agent_reward / max(1, len(actions_all))),
        "mcs_total_income": float(total_mcs_income),
        "mcs_avg_income": float(total_mcs_income / max(1.0, float(int(episodes) * len(env.mcs_list)))),
    }
    return batch, stats


def ppo_update_with_kl(
    policy: DTPolicyWithValue,
    ref_dt: DecisionTransformer,
    batch: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    st = torch.as_tensor(batch["ctx_states"], dtype=torch.float32, device=device)
    ac = torch.as_tensor(batch["ctx_actions"], dtype=torch.float32, device=device)
    rt = torch.as_tensor(batch["ctx_returns"], dtype=torch.float32, device=device)
    ts = torch.as_tensor(batch["ctx_steps"], dtype=torch.long, device=device)
    at = torch.as_tensor(batch["ctx_attention"], dtype=torch.long, device=device)
    am = torch.as_tensor(batch["action_masks"], dtype=torch.bool, device=device)
    act = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch["old_logps"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    n = int(st.shape[0])
    loss_all: List[float] = []
    p_all: List[float] = []
    v_all: List[float] = []
    e_all: List[float] = []
    klppo_all: List[float] = []
    klref_all: List[float] = []
    clip_all: List[float] = []

    for _ in range(int(args.update_epochs)):
        perm = rng.permutation(n)
        for s in range(0, n, int(args.mini_batch_size)):
            idx = perm[s : s + int(args.mini_batch_size)]
            mb_st = st[idx]
            mb_ac = ac[idx]
            mb_rt = rt[idx]
            mb_ts = ts[idx]
            mb_at = at[idx]
            mb_am = am[idx]
            mb_act = act[idx]
            mb_old = old_logp[idx]
            mb_ret = ret[idx]
            mb_adv = adv[idx]

            logits_seq, values_seq = policy.forward_seq(mb_st, mb_ac, mb_rt, mb_ts, mb_at)
            li = mb_at.sum(dim=1) - 1
            row = torch.arange(logits_seq.shape[0], device=device)
            logits = logits_seq[row, li, :].masked_fill(~mb_am, -1e9)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            values = values_seq[row, li]
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - mb_old)
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1.0 - float(args.ppo_clip), 1.0 + float(args.ppo_clip)) * mb_adv
            p_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(values, mb_ret)

            with torch.no_grad():
                st_n, rt_n = policy._normalize(mb_st, mb_rt)
                ref_logits_seq = _forward_dt_logits(ref_dt, st_n, mb_ac, rt_n, mb_ts, mb_at)
                ref_logits = ref_logits_seq[row, li, :].masked_fill(~mb_am, -1e9)
                ref_logits = torch.nan_to_num(ref_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            p_log = F.log_softmax(logits, dim=-1)
            q_log = F.log_softmax(ref_logits, dim=-1)
            p_prob = torch.exp(p_log)
            kl_ref = torch.sum(p_prob * (p_log - q_log), dim=-1).mean()

            kl_coef = float(getattr(args, "_effective_kl_coef", args.kl_coef))
            total = p_loss + float(args.vf_coef) * v_loss - float(args.ent_coef) * entropy + kl_coef * kl_ref
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(args.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old - new_logp).mean()
                clipfrac = (torch.abs(ratio - 1.0) > float(args.ppo_clip)).float().mean()
            loss_all.append(float(total.item()))
            p_all.append(float(p_loss.item()))
            v_all.append(float(v_loss.item()))
            e_all.append(float(entropy.item()))
            klppo_all.append(float(approx_kl.item()))
            klref_all.append(float(kl_ref.item()))
            clip_all.append(float(clipfrac.item()))

    return {
        "loss": float(np.mean(loss_all)),
        "policy_loss": float(np.mean(p_all)),
        "value_loss": float(np.mean(v_all)),
        "entropy": float(np.mean(e_all)),
        "approx_kl": float(np.mean(klppo_all)),
        "ref_kl": float(np.mean(klref_all)),
        "clipfrac": float(np.mean(clip_all)),
    }


def evaluate_policy(
    policy: DTPolicyWithValue,
    env_cfg: dict,
    episodes: int,
    seed: int,
    target_return: float,
    context_len: int,
    max_ep_len: int,
    action_dim: int,
    deterministic: bool,
    device: torch.device,
    max_steps: Optional[int],
) -> Dict[str, float]:
    env = Environment(config=env_cfg, seed=seed)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
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
        histories = _init_agent_histories(env.agents)
        mcs_pending_by_ev: Dict[int, Dict[str, float]] = {}
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        executed_steps = 0
        for _ in range(horizon):
            mask_dict = env.get_action_mask()
            env_actions: Dict[str, int] = {}
            obs_snapshot: Dict[str, np.ndarray] = {}
            rtg_snapshot: Dict[str, float] = {}
            for a in env.agents:
                obs_now = np.asarray(obs_dict[a], dtype=np.float32)
                rtg_now = float(target_return - histories[a]["cum_reward"])
                ctx = _build_context(histories[a], obs_now, int(env.current_step), rtg_now, context_len, action_dim, max_ep_len)
                am = np.asarray(mask_dict[a], dtype=np.bool_)
                act_idx, _, _ = policy.act(*ctx, action_mask=am, device=device, deterministic=deterministic)
                env_actions[a] = int(act_idx + 1)
                obs_snapshot[a] = obs_now
                rtg_snapshot[a] = rtg_now

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
            for a in env.agents:
                rew = float(step_result.agent_rewards[a])
                histories[a]["cum_reward"] += rew
                total_agent_reward += rew
                onehot = np.zeros((action_dim,), dtype=np.float32)
                onehot[int(env_actions[a] - 1)] = 1.0
                histories[a]["states"].append(obs_snapshot[a])
                histories[a]["actions"].append(onehot)
                histories[a]["returns"].append(np.asarray([rtg_snapshot[a]], dtype=np.float32))
                histories[a]["steps"].append(int(np.clip(step_result.step, 0, max_ep_len - 1)))
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


def _save_ckpt(path: Path, policy: DTPolicyWithValue, optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int, metrics: Dict[str, float], meta: Dict[str, float]) -> None:
    torch.save(
        {
            "dt_model_state_dict": policy.dt.state_dict(),
            "value_head_state_dict": policy.value_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "metrics": metrics,
            "meta": meta,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "dt_ppo_ft_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dt_ckpt = _load_dt_checkpoint(Path(args.dt_ckpt))
    dt_model = _build_dt_from_ckpt(dt_ckpt, device=device)
    state_mean = np.asarray(dt_ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(dt_ckpt["state_std"], dtype=np.float32)
    rtg_scale = float(dt_ckpt["rtg_scale"])
    context_len = int(args.context_len) if int(args.context_len) > 0 else int(dt_ckpt["context_len"])
    max_ep_len = int(dt_ckpt["max_ep_len"])
    obs_dim = int(dt_ckpt["obs_dim"])
    action_dim = int(dt_ckpt["action_dim"])

    policy = DTPolicyWithValue(dt_model=dt_model, state_mean=state_mean, state_std=state_std, rtg_scale=rtg_scale).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ref_dt = _build_dt_from_ckpt(dt_ckpt, device=device)
    ref_dt.eval()
    for p in ref_dt.parameters():
        p.requires_grad = False

    env_cfg = _apply_reward_profile(dict(CONFIG), str(args.reward_profile))
    env_cfg["use_lstm_summary"] = bool(args.use_lstm_summary)
    if args.lstm_predictor_ckpt:
        env_cfg["lstm_predictor_ckpt"] = str(args.lstm_predictor_ckpt)
    env = Environment(config=env_cfg, seed=int(args.seed))

    dataset_max_return: Optional[float] = None
    if float(args.target_return) > 0:
        target_return = float(args.target_return)
        target_mode = "manual"
    else:
        if args.offline_dataset:
            ds_path = Path(args.offline_dataset)
            if not ds_path.exists():
                raise RuntimeError(f"Offline dataset not found: {ds_path}")
            dataset_max_return = _compute_dataset_max_return(ds_path)
        elif "dataset_max_return" in dt_ckpt:
            dataset_max_return = float(dt_ckpt["dataset_max_return"])
        else:
            raise RuntimeError("Auto target_return requires --offline-dataset or dt_ckpt['dataset_max_return'].")
        target_return = float(dataset_max_return) * float(args.target_return_scale)
        target_mode = "auto"

    best_metric = {
        "success": float("-inf"),
        "reward": float("-inf"),
        "business": float("-inf"),
        "wait": float("inf"),
    }
    best_metric_epoch = {k: 0 for k in best_metric}
    best_guard_success = float("-inf")
    best_guard_wait = float("inf")
    best_guard_business = float("-inf")
    best_guard_epoch = 0
    best_guard_policy_state: Optional[dict] = None
    best_guard_optimizer_state: Optional[dict] = None
    print(
        f"dt_ppo_init obs_dim={obs_dim} action_dim={action_dim} ctx={context_len} "
        f"target_return={target_return:.4f} mode={target_mode} "
        f"dataset_max={float(dataset_max_return) if dataset_max_return is not None else -1.0:.4f} "
        f"scale={float(args.target_return_scale):.3f} reward_profile={args.reward_profile} device={device}"
    )
    business_history: List[Dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        freeze_policy = int(args.policy_warmup_epochs) > 0 and epoch <= int(args.policy_warmup_epochs)
        for p in policy.dt.parameters():
            p.requires_grad = not freeze_policy
        for p in policy.value_head.parameters():
            p.requires_grad = True
        args._effective_kl_coef = float(args.kl_coef) * (float(args.warmup_kl_mult) if freeze_policy else 1.0)

        policy.train()
        batch, collect_stats = collect_rollouts(
            env=env,
            policy=policy,
            episodes=int(args.episodes_per_epoch),
            target_return=float(target_return),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            context_len=context_len,
            max_ep_len=max_ep_len,
            action_dim=action_dim,
            device=device,
            rng=rng,
            max_steps=args.max_steps,
        )
        update_stats = ppo_update_with_kl(policy, ref_dt, batch, args, device, rng, optimizer)
        update_stats["effective_kl_coef"] = float(args._effective_kl_coef)
        update_stats["policy_frozen"] = float(1 if freeze_policy else 0)
        eval_stats: Dict[str, float] = {}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            policy.eval()
            eval_stats = evaluate_policy(
                policy=policy,
                env_cfg=env_cfg,
                episodes=int(args.eval_episodes),
                seed=int(args.eval_seed + epoch),
                target_return=float(target_return),
                context_len=context_len,
                max_ep_len=max_ep_len,
                action_dim=action_dim,
                deterministic=not bool(args.eval_stochastic),
                device=device,
                max_steps=args.eval_max_steps,
            )

        collect_business_score = _business_score(collect_stats, args)
        eval_business_score = _business_score(eval_stats, args) if eval_stats else float("nan")
        metric_stats = eval_stats if eval_stats else collect_stats
        metric_business_score = float(eval_business_score) if eval_stats else float(collect_business_score)

        guard_rollback = False
        guard_reason = ""
        if (
            bool(args.stability_guard)
            and eval_stats
            and best_guard_policy_state is not None
            and epoch <= int(args.guard_until_epoch)
        ):
            eval_success = float(eval_stats.get("success_rate", 0.0))
            eval_wait = float(eval_stats.get("avg_wait_minutes", 0.0))
            success_collapsed = eval_success < best_guard_success - float(args.guard_success_drop)
            wait_collapsed = eval_wait > best_guard_wait + float(args.guard_wait_increase_minutes)
            if success_collapsed or wait_collapsed:
                guard_rollback = True
                guard_reason = "success_drop" if success_collapsed else "wait_increase"
                policy.load_state_dict(best_guard_policy_state)
                optimizer.load_state_dict(best_guard_optimizer_state or optimizer.state_dict())
                for group in optimizer.param_groups:
                    group["lr"] = max(1e-8, float(group["lr"]) * float(args.guard_lr_decay))
                update_stats["guard_rollback"] = 1.0
                update_stats["guard_reason"] = guard_reason
                update_stats["guard_restored_epoch"] = float(best_guard_epoch)
                print(
                    f"[guard] rollback epoch={epoch:03d} reason={guard_reason} "
                    f"restore_epoch={best_guard_epoch:03d} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            else:
                update_stats["guard_rollback"] = 0.0
        else:
            update_stats["guard_rollback"] = 0.0

        merged = {**collect_stats, **update_stats, **eval_stats}
        merged["business_score"] = float(metric_business_score)
        meta = {
            "obs_dim": float(obs_dim),
            "action_dim": float(action_dim),
            "context_len": float(context_len),
            "max_ep_len": float(max_ep_len),
            "rtg_scale": float(rtg_scale),
            "target_return": float(target_return),
            "target_mode_auto": float(1 if target_mode == "auto" else 0),
            "target_return_scale": float(args.target_return_scale),
            "reward_profile_business": float(1 if str(args.reward_profile) == "business" else 0),
            "effective_kl_coef": float(args._effective_kl_coef),
        }
        _save_ckpt(outdir / "last.pt", policy, optimizer, args, epoch, merged, meta)

        if not guard_rollback:
            metric_success = float(metric_stats.get("success_rate", -1.0))
            metric_reward = float(metric_stats.get("avg_total_agent_reward_per_ep", float("-inf")))
            metric_wait = float(metric_stats.get("avg_wait_minutes", float("inf")))

            if metric_success > best_metric["success"]:
                best_metric["success"] = metric_success
                best_metric_epoch["success"] = int(epoch)
                _save_ckpt(outdir / "best.pt", policy, optimizer, args, epoch, merged, meta)
                _save_ckpt(outdir / "best_by_success.pt", policy, optimizer, args, epoch, merged, meta)
            if metric_reward > best_metric["reward"]:
                best_metric["reward"] = metric_reward
                best_metric_epoch["reward"] = int(epoch)
                _save_ckpt(outdir / "best_by_reward.pt", policy, optimizer, args, epoch, merged, meta)
            if metric_wait < best_metric["wait"]:
                best_metric["wait"] = metric_wait
                best_metric_epoch["wait"] = int(epoch)
                _save_ckpt(outdir / "best_by_wait.pt", policy, optimizer, args, epoch, merged, meta)
            if metric_business_score > best_metric["business"]:
                best_metric["business"] = metric_business_score
                best_metric_epoch["business"] = int(epoch)
                _save_ckpt(outdir / "best_by_business.pt", policy, optimizer, args, epoch, merged, meta)

            if eval_stats and metric_business_score >= best_guard_business:
                best_guard_business = metric_business_score
                best_guard_success = float(eval_stats.get("success_rate", 0.0))
                best_guard_wait = float(eval_stats.get("avg_wait_minutes", float("inf")))
                best_guard_epoch = int(epoch)
                best_guard_policy_state = copy.deepcopy(policy.state_dict())
                best_guard_optimizer_state = copy.deepcopy(optimizer.state_dict())

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "collect": collect_stats,
                        "update": update_stats,
                        "eval": eval_stats,
                        "business": {
                            "collect_business_score": float(collect_business_score),
                            "eval_business_score": float(eval_business_score),
                            "best_guard_epoch": int(best_guard_epoch),
                            "guard_rollback": bool(guard_rollback),
                            "guard_reason": guard_reason,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        biz_row = {
            "epoch": float(epoch),
            "reward": float(collect_stats.get("avg_total_agent_reward_per_ep", 0.0)),
            "success_rate": float(collect_stats.get("success_rate", 0.0)),
            "mcs_avg_income": float(collect_stats.get("mcs_avg_income", 0.0)),
            "ev_avg_wait_minutes": float(collect_stats.get("avg_wait_minutes", 0.0)),
            "business_score": float(collect_business_score),
            "eval_reward": float(eval_stats.get("avg_total_agent_reward_per_ep", np.nan)) if eval_stats else float("nan"),
            "eval_success_rate": float(eval_stats.get("success_rate", np.nan)) if eval_stats else float("nan"),
            "eval_mcs_avg_income": float(eval_stats.get("mcs_avg_income", np.nan)) if eval_stats else float("nan"),
            "eval_ev_avg_wait_minutes": float(eval_stats.get("avg_wait_minutes", np.nan)) if eval_stats else float("nan"),
            "eval_business_score": float(eval_business_score),
            "ref_kl": float(update_stats.get("ref_kl", np.nan)),
            "approx_kl": float(update_stats.get("approx_kl", np.nan)),
            "guard_rollback": float(1 if guard_rollback else 0),
        }
        business_history.append(biz_row)
        print(
            f"epoch={epoch:03d} "
            f"collect_success={collect_stats['success_rate']:.3f} "
            f"collect_wait={collect_stats['avg_wait_minutes']:.2f}min "
            f"reward={biz_row['reward']:.3f} "
            f"biz={biz_row['business_score']:.2f}",
            flush=bool(args.log_flush),
        )

    biz_csv = outdir / "business_metrics.csv"
    biz_png = outdir / "business_metrics.png"
    _save_business_metrics_csv(path=biz_csv, rows=business_history)
    _plot_business_metrics(path=biz_png, rows=business_history)
    best_summary = {
        "best_by_success": {"epoch": int(best_metric_epoch["success"]), "value": float(best_metric["success"])},
        "best_by_reward": {"epoch": int(best_metric_epoch["reward"]), "value": float(best_metric["reward"])},
        "best_by_wait": {"epoch": int(best_metric_epoch["wait"]), "value": float(best_metric["wait"])},
        "best_by_business": {"epoch": int(best_metric_epoch["business"]), "value": float(best_metric["business"])},
        "reward_profile": str(args.reward_profile),
        "stability_guard": bool(args.stability_guard),
    }
    with (outdir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)
    print(f"saved business metrics: {biz_csv}")
    print(f"saved business plot: {biz_png}")
    print(f"saved best summary: {outdir / 'best_summary.json'}")
    print(f"done: checkpoints/log in {outdir}")


if __name__ == "__main__":
    main()
