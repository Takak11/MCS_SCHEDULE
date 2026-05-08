from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import CONFIG
from env import Environment
from eval_dt_uncertainty_gate import _finalize_local, _fused_action, load_dt_prior
from plot_method_comparison import _event_income
from train_mappo import (
    ActorNet,
    CriticNet,
    _aggregate_team_reward,
    _build_state,
    _business_score,
    _mask_dict_to_matrix,
    _obs_dict_to_matrix,
    _value_critic,
    save_ckpt,
)


class ActorAdapter:
    def __init__(self, actor: ActorNet) -> None:
        self.actor = actor

    def act(self, obs: np.ndarray, action_mask: np.ndarray, device: torch.device, deterministic: bool = False) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t).masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MAPPO with frozen uncertainty-gated DT prior.")
    p.add_argument("--outdir", type=str, default="result/mappo_dt_gate")
    p.add_argument("--init-mappo-ckpt", type=str, default="result/ppo_for_offline/best_by_business.pt")
    p.add_argument("--dt-ckpt", type=str, default="result/dt_ppo_ft/best_by_business.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-lstm-summary", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lstm-predictor-ckpt", type=str, default="result/predictor/lstm_predictor.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--episodes-per-epoch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-clip", type=float, default=0.15)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--mini-batch-size", type=int, default=256)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--team-reward-mode", type=str, choices=["mean", "sum"], default="mean")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--soft-gate", action="store_true")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--target-return", type=float, default=0.0, help="Override DT target return. <=0 uses checkpoint/default scale.")
    p.add_argument("--target-return-scale", type=float, default=1.2, help="Used with offline DT checkpoints when --target-return<=0.")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--eval-max-steps", type=int, default=None)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--save-epoch-interval", type=int, default=0)
    return p.parse_args()


def _init_histories(agents: List[str]) -> Dict[str, dict]:
    return {a: {"states": [], "actions": [], "returns": [], "steps": [], "cum_reward": 0.0} for a in agents}


def collect_rollouts_fused(
    env: Environment,
    actor: ActorNet,
    critic: CriticNet,
    dt_policy,
    episodes: int,
    gamma: float,
    gae_lambda: float,
    team_reward_mode: str,
    device: torch.device,
    rng: np.random.Generator,
    action_dim: int,
    context_len: int,
    max_ep_len: int,
    target_return: float,
    threshold: float,
    alpha: float,
    soft_gate: bool,
    temperature: float,
    max_steps: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    agents = list(env.agents)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    adapter = ActorAdapter(actor)

    obs_seq: List[np.ndarray] = []
    state_seq: List[np.ndarray] = []
    mask_seq: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    old_logp_seq: List[np.ndarray] = []
    dt_ctx_states: List[np.ndarray] = []
    dt_ctx_actions: List[np.ndarray] = []
    dt_ctx_returns: List[np.ndarray] = []
    dt_ctx_steps: List[np.ndarray] = []
    dt_ctx_attention: List[np.ndarray] = []
    gate_seq: List[np.ndarray] = []
    uncertainty_seq: List[np.ndarray] = []
    value_seq: List[float] = []
    adv_seq: List[float] = []
    ret_seq: List[float] = []

    stats = {
        "steps": 0.0,
        "requests": 0.0,
        "mcs_requests": 0.0,
        "success_requests": 0.0,
        "mcs_served": 0.0,
        "wait_steps_sum": 0.0,
        "wait_count": 0.0,
        "timeout_events_total": 0.0,
        "unresolved_mcs_total": 0.0,
        "total_agent_reward": 0.0,
        "total_mcs_income": 0.0,
    }

    for _ep in range(int(episodes)):
        obs_dict = env.reset(seed=int(rng.integers(1_000_000_000)))
        histories = _init_histories(agents)
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        ep_values: List[float] = []
        ep_rewards: List[float] = []
        ep_dones: List[float] = []
        last_next_state: Optional[np.ndarray] = None
        executed_steps = 0
        episode_done = False
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)

        for _ in range(horizon):
            masks = env.get_action_mask()
            obs_mat = _obs_dict_to_matrix(obs_dict, agents=agents)
            mask_mat = _mask_dict_to_matrix(masks, agents=agents, action_dim=action_dim)
            state = _build_state(obs_mat)
            value = _value_critic(critic=critic, state=state, device=device)

            act_idx = np.zeros((len(agents),), dtype=np.int64)
            logps = np.zeros((len(agents),), dtype=np.float32)
            gates = np.zeros((len(agents),), dtype=np.float32)
            uncs = np.zeros((len(agents),), dtype=np.float32)
            ctx_s = np.zeros((len(agents), context_len, obs_mat.shape[-1]), dtype=np.float32)
            ctx_a = np.zeros((len(agents), context_len, action_dim), dtype=np.float32)
            ctx_r = np.zeros((len(agents), context_len, 1), dtype=np.float32)
            ctx_t = np.zeros((len(agents), context_len), dtype=np.int64)
            ctx_att = np.zeros((len(agents), context_len), dtype=np.int64)
            rtg_snapshot: Dict[str, float] = {}

            for i, a in enumerate(agents):
                obs_now = np.asarray(obs_dict[a], dtype=np.float32)
                rtg_snapshot[a] = float(target_return - histories[a]["cum_reward"])
                action_i, logp_i, unc_i, gate_i = _fused_action(
                    actor_like=adapter,
                    dt_policy=dt_policy,
                    history=histories[a],
                    obs_now=obs_now,
                    action_mask=np.asarray(masks[a], dtype=np.bool_),
                    step_now=int(env.current_step),
                    target_return=target_return,
                    context_len=context_len,
                    action_dim=action_dim,
                    max_ep_len=max_ep_len,
                    device=device,
                    threshold=threshold,
                    alpha=alpha,
                    soft_gate=soft_gate,
                    temperature=temperature,
                    deterministic=False,
                )
                from eval_dt_uncertainty_gate import _build_dt_context

                ctx = _build_dt_context(histories[a], obs_now, int(env.current_step), rtg_snapshot[a], context_len, action_dim, max_ep_len)
                ctx_s[i], ctx_a[i], ctx_r[i], ctx_t[i], ctx_att[i] = ctx
                act_idx[i] = int(action_i)
                logps[i] = float(logp_i)
                gates[i] = float(gate_i)
                uncs[i] = float(unc_i)

            step_result = env.step_parallel({agents[i]: int(act_idx[i] + 1) for i in range(len(agents))})
            executed_steps += 1
            episode_done = bool(step_result.done)
            team_reward = _aggregate_team_reward(step_result.agent_rewards, mode=team_reward_mode)

            obs_seq.append(obs_mat)
            state_seq.append(state)
            mask_seq.append(mask_mat)
            action_seq.append(act_idx.copy())
            old_logp_seq.append(logps.copy())
            dt_ctx_states.append(ctx_s)
            dt_ctx_actions.append(ctx_a)
            dt_ctx_returns.append(ctx_r)
            dt_ctx_steps.append(ctx_t)
            dt_ctx_attention.append(ctx_att)
            gate_seq.append(gates.copy())
            uncertainty_seq.append(uncs.copy())
            value_seq.append(float(value))
            ep_values.append(float(value))
            ep_rewards.append(float(team_reward))
            ep_dones.append(1.0 if episode_done else 0.0)

            if not episode_done:
                next_obs = env.get_agent_observations()
                last_next_state = _build_state(_obs_dict_to_matrix(next_obs, agents=agents))
            else:
                last_next_state = None

            for req in step_result.requests:
                stats["requests"] += 1.0
                if req.get("service_mode") == "fcs":
                    stats["success_requests"] += 1.0
                else:
                    stats["mcs_requests"] += 1.0
                    pending_by_ev[int(req["ev_id"])] = {"step": float(int(req.get("step", step_result.step))), "required_kwh": float(req.get("required_kwh", 0.0))}
            for event in step_result.mcs_events:
                req_info = None
                if str(event.get("action", "")) == "serve_request":
                    req_info = pending_by_ev.pop(int(event["ev_id"]), {"step": float(step_result.step), "required_kwh": float(event.get("required_kwh", 0.0))})
                stats["total_mcs_income"] += _event_income(env, event, req_info)
                if str(event.get("action", "")) == "serve_request":
                    req_step = int((req_info or {}).get("step", float(step_result.step)))
                    wait_steps = max(0, int(step_result.step) - req_step)
                    stats["mcs_served"] += 1.0
                    stats["success_requests"] += 1.0
                    stats["wait_steps_sum"] += float(wait_steps)
                    stats["wait_count"] += 1.0
            for to in step_result.timeout_events:
                stats["timeout_events_total"] += 1.0
                ev_id = int(to.get("ev_id", -1))
                req_step = int(to.get("request_step", step_result.step))
                wait_steps = int(to.get("wait_steps", max(0, int(step_result.step) - req_step)))
                pending_by_ev.pop(ev_id, None)
                stats["wait_steps_sum"] += float(wait_steps)
                stats["wait_count"] += 1.0
            for a in agents:
                rew = float(step_result.agent_rewards[a])
                histories[a]["cum_reward"] += rew
                onehot = np.zeros((action_dim,), dtype=np.float32)
                onehot[int(act_idx[agents.index(a)])] = 1.0
                histories[a]["states"].append(np.asarray(obs_dict[a], dtype=np.float32))
                histories[a]["actions"].append(onehot)
                histories[a]["returns"].append(np.asarray([rtg_snapshot[a]], dtype=np.float32))
                histories[a]["steps"].append(int(np.clip(step_result.step, 0, max_ep_len - 1)))
                stats["total_agent_reward"] += rew

            if episode_done:
                break
            obs_dict = env.get_agent_observations()

        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)

        bootstrap = 0.0 if episode_done or last_next_state is None else _value_critic(critic=critic, state=last_next_state, device=device)
        ep_adv = np.zeros((len(ep_rewards),), dtype=np.float32)
        ep_ret = np.zeros((len(ep_rewards),), dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap)
        for t in range(len(ep_rewards) - 1, -1, -1):
            delta = ep_rewards[t] + gamma * (1.0 - ep_dones[t]) * next_value - ep_values[t]
            gae = delta + gamma * gae_lambda * (1.0 - ep_dones[t]) * gae
            ep_adv[t] = float(gae)
            ep_ret[t] = float(gae + ep_values[t])
            next_value = float(ep_values[t])
        adv_seq.extend(ep_adv.tolist())
        ret_seq.extend(ep_ret.tolist())

    rollout_stats = _finalize_local(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)
    rollout_stats["transitions"] = float(len(obs_seq) * len(agents))
    rollout_stats["gate_rate"] = float(np.mean(np.asarray(gate_seq, dtype=np.float32))) if gate_seq else 0.0
    rollout_stats["avg_uncertainty"] = float(np.mean(np.asarray(uncertainty_seq, dtype=np.float32))) if uncertainty_seq else 0.0
    batch = {
        "obs": np.asarray(obs_seq, dtype=np.float32),
        "states": np.asarray(state_seq, dtype=np.float32),
        "masks": np.asarray(mask_seq, dtype=np.bool_),
        "actions": np.asarray(action_seq, dtype=np.int64),
        "old_logps": np.asarray(old_logp_seq, dtype=np.float32),
        "returns": np.asarray(ret_seq, dtype=np.float32),
        "advantages": np.asarray(adv_seq, dtype=np.float32),
        "ctx_states": np.asarray(dt_ctx_states, dtype=np.float32),
        "ctx_actions": np.asarray(dt_ctx_actions, dtype=np.float32),
        "ctx_returns": np.asarray(dt_ctx_returns, dtype=np.float32),
        "ctx_steps": np.asarray(dt_ctx_steps, dtype=np.int64),
        "ctx_attention": np.asarray(dt_ctx_attention, dtype=np.int64),
        "gates": np.asarray(gate_seq, dtype=np.float32),
    }
    return batch, rollout_stats


def update_fused(
    actor: ActorNet,
    critic: CriticNet,
    dt_policy,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    states_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logps_t = torch.as_tensor(batch["old_logps"], dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    gates_t = torch.as_tensor(batch["gates"], dtype=torch.float32, device=device)
    ctx_states_t = torch.as_tensor(batch["ctx_states"], dtype=torch.float32, device=device)
    ctx_actions_t = torch.as_tensor(batch["ctx_actions"], dtype=torch.float32, device=device)
    ctx_returns_t = torch.as_tensor(batch["ctx_returns"], dtype=torch.float32, device=device)
    ctx_steps_t = torch.as_tensor(batch["ctx_steps"], dtype=torch.long, device=device)
    ctx_attention_t = torch.as_tensor(batch["ctx_attention"], dtype=torch.long, device=device)

    t_steps, n_agents, obs_dim = obs_t.shape
    action_dim = int(masks_t.shape[-1])
    if bool(args.normalize_adv):
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    losses: List[float] = []
    p_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    kls: List[float] = []
    clipfracs: List[float] = []

    for _ in range(int(args.update_epochs)):
        perm = rng.permutation(t_steps)
        for start in range(0, t_steps, int(args.mini_batch_size)):
            idx = torch.as_tensor(perm[start : start + int(args.mini_batch_size)], dtype=torch.long, device=device)
            mb_obs = obs_t.index_select(0, idx).reshape(-1, obs_dim)
            mb_masks = masks_t.index_select(0, idx).reshape(-1, action_dim)
            mb_actions = actions_t.index_select(0, idx).reshape(-1)
            mb_old_logps = old_logps_t.index_select(0, idx).reshape(-1)
            mb_adv = adv_t.index_select(0, idx).repeat_interleave(n_agents)
            mb_gates = gates_t.index_select(0, idx).reshape(-1, 1)

            logits_m = actor(mb_obs).masked_fill(~mb_masks, -1e9)
            mb_ctx_s = ctx_states_t.index_select(0, idx).reshape(-1, ctx_states_t.shape[2], ctx_states_t.shape[3])
            mb_ctx_a = ctx_actions_t.index_select(0, idx).reshape(-1, ctx_actions_t.shape[2], ctx_actions_t.shape[3])
            mb_ctx_r = ctx_returns_t.index_select(0, idx).reshape(-1, ctx_returns_t.shape[2], 1)
            mb_ctx_t = ctx_steps_t.index_select(0, idx).reshape(-1, ctx_steps_t.shape[2])
            mb_ctx_att = ctx_attention_t.index_select(0, idx).reshape(-1, ctx_attention_t.shape[2])
            with torch.no_grad():
                logits_dt_seq, _ = dt_policy.forward_seq(mb_ctx_s, mb_ctx_a, mb_ctx_r, mb_ctx_t, mb_ctx_att)
                lengths = mb_ctx_att.sum(dim=1).clamp_min(1) - 1
                row = torch.arange(logits_dt_seq.shape[0], device=device)
                logits_dt = logits_dt_seq[row, lengths, :].masked_fill(~mb_masks, -1e9)
                dt_logprob = torch.log_softmax(logits_dt, dim=-1)

            fused_logits = logits_m + mb_gates * float(args.alpha) * dt_logprob
            fused_logits = fused_logits.masked_fill(~mb_masks, -1e9)
            dist = torch.distributions.Categorical(logits=fused_logits)
            new_logps = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logps - mb_old_logps)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - float(args.ppo_clip), 1.0 + float(args.ppo_clip)) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()

            values = critic(states_t.index_select(0, idx))
            v_loss = F.mse_loss(values, returns_t.index_select(0, idx))
            total = p_loss + float(args.vf_coef) * v_loss - float(args.ent_coef) * entropy

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), float(args.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_logps - new_logps).mean()
                clipfrac = (torch.abs(ratio - 1.0) > float(args.ppo_clip)).float().mean()
            losses.append(float(total.item()))
            p_losses.append(float(p_loss.item()))
            v_losses.append(float(v_loss.item()))
            entropies.append(float(entropy.item()))
            kls.append(float(approx_kl.item()))
            clipfracs.append(float(clipfrac.item()))

    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(p_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(kls)),
        "clipfrac": float(np.mean(clipfracs)),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env_cfg = dict(CONFIG)
    env_cfg["use_lstm_summary"] = bool(args.use_lstm_summary)
    if args.lstm_predictor_ckpt:
        env_cfg["lstm_predictor_ckpt"] = str(args.lstm_predictor_ckpt)
    env = Environment(config=env_cfg, seed=int(args.seed))
    obs_dict = env.reset(seed=int(args.seed))
    obs_dim = int(next(iter(obs_dict.values())).shape[0])
    n_agents = int(len(env.agents))
    action_dim = int(len(env.ACTION_SPACE))
    state_dim = int(obs_dim * n_agents)

    actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    critic = CriticNet(state_dim=state_dim, hidden_dim=256).to(device)
    init = torch.load(Path(args.init_mappo_ckpt), map_location="cpu")
    init_args = init.get("args", {}) if isinstance(init.get("args", {}), dict) else {}
    actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(init_args.get("actor_hidden_dim", 128))).to(device)
    critic = CriticNet(state_dim=state_dim, hidden_dim=int(init_args.get("critic_hidden_dim", 256))).to(device)
    actor.load_state_dict(init["actor_state_dict"])
    if "critic_state_dict" in init:
        critic.load_state_dict(init["critic_state_dict"], strict=False)

    dt_policy, dt_meta, _dt_ck = load_dt_prior(
        Path(args.dt_ckpt),
        device=device,
        target_return=float(args.target_return),
        target_return_scale=float(args.target_return_scale),
    )
    context_len = int(dt_meta["context_len"])
    max_ep_len = int(dt_meta["max_ep_len"])
    dt_action_dim = int(dt_meta["action_dim"])
    target_return = float(dt_meta["target_return"])
    if int(dt_action_dim) != int(action_dim):
        raise RuntimeError(f"DT action_dim={dt_action_dim} does not match env action_dim={action_dim}")

    optimizer = torch.optim.AdamW(
        [{"params": actor.parameters(), "lr": float(args.lr_actor)}, {"params": critic.parameters(), "lr": float(args.lr_critic)}],
        weight_decay=float(args.weight_decay),
    )

    log_path = outdir / "mappo_dt_gate_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    csv_path = outdir / "business_metrics.csv"
    rows: List[Dict[str, float]] = []
    best_business = float("-inf")
    best_success = float("-inf")

    print(
        f"mappo_dt_gate_init obs_dim={obs_dim} state_dim={state_dim} action_dim={action_dim} "
        f"ctx={context_len} threshold={args.threshold} alpha={args.alpha} target_return={target_return:.3f} device={device}",
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        critic.train()
        batch, rollout_stats = collect_rollouts_fused(
            env=env,
            actor=actor,
            critic=critic,
            dt_policy=dt_policy,
            episodes=int(args.episodes_per_epoch),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            team_reward_mode=str(args.team_reward_mode),
            device=device,
            rng=rng,
            action_dim=action_dim,
            context_len=context_len,
            max_ep_len=max_ep_len,
            target_return=target_return,
            threshold=float(args.threshold),
            alpha=float(args.alpha),
            soft_gate=bool(args.soft_gate),
            temperature=float(args.temperature),
            max_steps=args.max_steps,
        )
        update_stats = update_fused(actor, critic, dt_policy, optimizer, batch, args, device, rng)

        eval_stats: Dict[str, float] = {}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            from eval_dt_uncertainty_gate import evaluate as evaluate_gated

            actor.eval()
            eval_stats = evaluate_gated(
                actor_like=ActorAdapter(actor),
                dt_policy=dt_policy,
                dt_meta=dt_meta,
                env_cfg=env_cfg,
                episodes=int(args.eval_episodes),
                seed=int(args.eval_seed + epoch),
                device=device,
                max_steps=args.eval_max_steps,
                threshold=float(args.threshold),
                alpha=float(args.alpha),
                soft_gate=bool(args.soft_gate),
                temperature=float(args.temperature),
                deterministic=not bool(args.eval_stochastic),
                use_gate=True,
                verbose=False,
            )

        collect_business = _business_score(rollout_stats, args)
        eval_business = _business_score(eval_stats, args) if eval_stats else float("nan")
        metric_business = float(eval_business) if eval_stats else float(collect_business)
        merged = {**rollout_stats, **update_stats, **eval_stats, "business_score": metric_business}
        save_ckpt(outdir / "last.pt", actor, critic, optimizer, epoch, args, obs_dim, action_dim, n_agents, state_dim, merged)
        if int(args.save_epoch_interval) > 0 and epoch % int(args.save_epoch_interval) == 0:
            save_ckpt(outdir / f"epoch_{epoch:03d}.pt", actor, critic, optimizer, epoch, args, obs_dim, action_dim, n_agents, state_dim, merged)
        if metric_business > best_business:
            best_business = metric_business
            save_ckpt(outdir / "best_by_business.pt", actor, critic, optimizer, epoch, args, obs_dim, action_dim, n_agents, state_dim, merged)
        metric_success = float((eval_stats or rollout_stats).get("success_rate", 0.0))
        if metric_success > best_success:
            best_success = metric_success
            save_ckpt(outdir / "best_by_success.pt", actor, critic, optimizer, epoch, args, obs_dim, action_dim, n_agents, state_dim, merged)

        row = {
            "epoch": float(epoch),
            "success_rate": float(rollout_stats.get("success_rate", 0.0)),
            "ev_avg_wait_minutes": float(rollout_stats.get("avg_wait_minutes", 0.0)),
            "business_score": float(collect_business),
            "eval_success_rate": float(eval_stats.get("success_rate", np.nan)) if eval_stats else float("nan"),
            "eval_ev_avg_wait_minutes": float(eval_stats.get("avg_wait_minutes", np.nan)) if eval_stats else float("nan"),
            "eval_business_score": float(eval_business),
            "gate_rate": float(rollout_stats.get("gate_rate", 0.0)),
            "avg_uncertainty": float(rollout_stats.get("avg_uncertainty", 0.0)),
            "approx_kl": float(update_stats.get("approx_kl", 0.0)),
        }
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "collect": rollout_stats, "update": update_stats, "eval": eval_stats}, ensure_ascii=False) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"epoch={epoch:03d} collect_succ={row['success_rate']:.3f} wait={row['ev_avg_wait_minutes']:.2f} "
            f"gate={row['gate_rate']:.3f} eval_succ={row['eval_success_rate']:.3f} "
            f"eval_biz={row['eval_business_score']:.2f}",
            flush=True,
        )

    summary = {"best_by_business": {"value": float(best_business)}, "best_by_success": {"value": float(best_success)}}
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
