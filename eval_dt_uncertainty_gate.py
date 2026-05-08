from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from build_offline_dataset_ppo import _load_policy_actor
from config import CONFIG
from env import Environment
from plot_method_comparison import _event_income, _init_stats, _load_dt_policy
from train_dt_ppo_finetune import DTPolicyWithValue, _build_dt_from_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MAPPO with uncertainty-gated DT action prior.")
    p.add_argument("--mappo-ckpt", type=str, default="result/ppo_for_offline/best_by_business.pt")
    p.add_argument("--dt-ckpt", type=str, default="result/dt_ppo_ft/best_by_business.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.70, help="Normalized MAPPO entropy above this value enables DT prior.")
    p.add_argument("--alpha", type=float, default=0.10, help="DT log-prob prior strength.")
    p.add_argument("--soft-gate", action="store_true")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--outdir", type=str, default="result/dt_uncertainty_gate")
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--target-return", type=float, default=0.0, help="Override DT target return. <=0 uses checkpoint/default scale.")
    p.add_argument("--target-return-scale", type=float, default=1.2, help="Used with offline DT checkpoints when --target-return<=0.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_dt_prior(ckpt_path: Path, device: torch.device, target_return: float = 0.0, target_return_scale: float = 1.2):
    ck = torch.load(ckpt_path, map_location="cpu")
    if "dt_model_state_dict" in ck:
        policy = _load_dt_policy(ckpt_path, device=device)
        base = torch.load(Path(ck["args"]["dt_ckpt"]), map_location="cpu")
        context_len = int(ck["args"].get("context_len", 0)) or int(base["context_len"])
        chosen_target = float(ck.get("meta", {}).get("target_return", ck["args"].get("target_return", 0.0)))
        source = "finetuned_dt"
    elif "model_state_dict" in ck:
        dt = _build_dt_from_ckpt(ck, device=device)
        dt.load_state_dict(ck["model_state_dict"])
        policy = DTPolicyWithValue(
            dt_model=dt,
            state_mean=np.asarray(ck["state_mean"], dtype=np.float32),
            state_std=np.asarray(ck["state_std"], dtype=np.float32),
            rtg_scale=float(ck["rtg_scale"]),
        ).to(device)
        context_len = int(ck["context_len"])
        chosen_target = float(ck.get("dataset_max_return", 0.0)) * float(target_return_scale)
        base = ck
        source = "offline_dt"
    else:
        raise RuntimeError(f"Unsupported DT checkpoint format: {ckpt_path}")

    if float(target_return) > 0:
        chosen_target = float(target_return)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    meta = {
        "checkpoint": str(ckpt_path),
        "source": source,
        "context_len": int(context_len),
        "max_ep_len": int(base["max_ep_len"]),
        "action_dim": int(base["action_dim"]),
        "target_return": float(chosen_target),
        "dataset_max_return": float(base.get("dataset_max_return", 0.0)),
        "rtg_scale": float(base.get("rtg_scale", 1.0)),
    }
    return policy, meta, ck


def _finalize_local(stats: Dict[str, float], episodes: int, n_mcs: int, step_minutes: float) -> Dict[str, float]:
    avg_wait_steps = float(stats["wait_steps_sum"] / max(1.0, stats["wait_count"]))
    out = {
        "episodes": float(episodes),
        "steps": float(stats["steps"]),
        "requests": float(stats["requests"]),
        "success_rate": float(stats["success_requests"] / max(1.0, stats["requests"])),
        "mcs_success_rate": float(stats["mcs_served"] / max(1.0, stats["mcs_requests"])),
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "timeout_events_total": float(stats["timeout_events_total"]),
        "unresolved_mcs_total": float(stats["unresolved_mcs_total"]),
        "avg_total_agent_reward_per_ep": float(stats["total_agent_reward"] / max(1.0, float(episodes))),
        "mcs_avg_income": float(stats["total_mcs_income"] / max(1.0, float(episodes * n_mcs))),
    }
    out["business_score"] = float(
        1000.0 * out["success_rate"]
        + 300.0 * out["mcs_success_rate"]
        - 25.0 * out["avg_wait_minutes"]
        - 0.05 * out["timeout_events_total"]
    )
    return out


def _actor_logits(actor_like, obs: np.ndarray, action_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        if hasattr(actor_like, "actor"):
            logits = actor_like.actor(obs_t)
        else:
            out = actor_like(obs_t)
            logits = out[0] if isinstance(out, tuple) else out
        logits = logits.masked_fill(~mask_t, -1e9)
    return logits


def _build_dt_context(
    history: dict,
    obs_now: np.ndarray,
    step_now: int,
    rtg_now: float,
    context_len: int,
    action_dim: int,
    max_ep_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero_a = np.zeros((action_dim,), dtype=np.float32)
    s_seq = history["states"] + [np.asarray(obs_now, dtype=np.float32)]
    a_seq = history["actions"] + [zero_a]
    r_seq = history["returns"] + [np.asarray([rtg_now], dtype=np.float32)]
    t_seq = history["steps"] + [int(np.clip(step_now, 0, max_ep_len - 1))]
    if len(s_seq) > context_len:
        s_seq, a_seq, r_seq, t_seq = s_seq[-context_len:], a_seq[-context_len:], r_seq[-context_len:], t_seq[-context_len:]

    pad = context_len - len(s_seq)
    ctx_states = np.zeros((context_len, int(obs_now.shape[0])), dtype=np.float32)
    ctx_actions = np.zeros((context_len, action_dim), dtype=np.float32)
    ctx_returns = np.zeros((context_len, 1), dtype=np.float32)
    ctx_steps = np.zeros((context_len,), dtype=np.int64)
    ctx_attention = np.zeros((context_len,), dtype=np.int64)
    ctx_states[pad:] = np.asarray(s_seq, dtype=np.float32)
    ctx_actions[pad:] = np.asarray(a_seq, dtype=np.float32)
    ctx_returns[pad:] = np.asarray(r_seq, dtype=np.float32).reshape(-1, 1)
    ctx_steps[pad:] = np.asarray(t_seq, dtype=np.int64)
    ctx_attention[pad:] = 1
    return ctx_states, ctx_actions, ctx_returns, ctx_steps, ctx_attention


def _dt_logits(policy, ctx: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], action_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    st, ac, rt, ts, at = ctx
    st_t = torch.as_tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
    ac_t = torch.as_tensor(ac, dtype=torch.float32, device=device).unsqueeze(0)
    rt_t = torch.as_tensor(rt, dtype=torch.float32, device=device).unsqueeze(0)
    ts_t = torch.as_tensor(ts, dtype=torch.long, device=device).unsqueeze(0)
    at_t = torch.as_tensor(at, dtype=torch.long, device=device).unsqueeze(0)
    am_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits_seq, _ = policy.forward_seq(st_t, ac_t, rt_t, ts_t, at_t)
        li = int(torch.sum(at_t, dim=1).item()) - 1
        logits = logits_seq[:, li, :].masked_fill(~am_t, -1e9)
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return logits


def _fused_action(
    actor_like,
    dt_policy,
    history: dict,
    obs_now: np.ndarray,
    action_mask: np.ndarray,
    step_now: int,
    target_return: float,
    context_len: int,
    action_dim: int,
    max_ep_len: int,
    device: torch.device,
    threshold: float,
    alpha: float,
    soft_gate: bool,
    temperature: float,
    deterministic: bool,
) -> Tuple[int, float, float, float]:
    logits_mappo = _actor_logits(actor_like, obs_now, action_mask, device=device)
    dist_mappo = torch.distributions.Categorical(logits=logits_mappo)
    entropy = dist_mappo.entropy()
    valid_count = torch.as_tensor(action_mask, dtype=torch.bool, device=device).sum().clamp_min(1).float()
    max_entropy = torch.log(valid_count).clamp_min(1e-6)
    uncertainty = entropy / max_entropy

    rtg_now = float(target_return - history["cum_reward"])
    ctx = _build_dt_context(
        history=history,
        obs_now=obs_now,
        step_now=step_now,
        rtg_now=rtg_now,
        context_len=context_len,
        action_dim=action_dim,
        max_ep_len=max_ep_len,
    )
    logits_dt = _dt_logits(dt_policy, ctx, action_mask, device=device)
    dt_logprob = torch.log_softmax(logits_dt, dim=-1)
    if soft_gate:
        gate = torch.sigmoid((uncertainty - float(threshold)) / max(float(temperature), 1e-6))
    else:
        gate = (uncertainty > float(threshold)).float()
    final_logits = logits_mappo + gate.unsqueeze(-1) * float(alpha) * dt_logprob
    final_logits = final_logits.masked_fill(~torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0), -1e9)
    dist = torch.distributions.Categorical(logits=final_logits)
    action = final_logits.argmax(dim=-1) if deterministic else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(uncertainty.item()), float(gate.item())


def evaluate(
    actor_like,
    dt_policy,
    dt_meta: dict,
    env_cfg: dict,
    episodes: int,
    seed: int,
    device: torch.device,
    max_steps: Optional[int],
    threshold: float,
    alpha: float,
    soft_gate: bool,
    temperature: float,
    deterministic: bool,
    use_gate: bool,
    verbose: bool,
) -> Dict[str, float]:
    context_len = int(dt_meta["context_len"])
    max_ep_len = int(dt_meta["max_ep_len"])
    action_dim = int(dt_meta["action_dim"])
    target_return = float(dt_meta["target_return"])

    env = Environment(config=env_cfg, seed=seed)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    stats = _init_stats()
    gate_sum = 0.0
    uncertainty_sum = 0.0
    action_count = 0.0

    for ep in range(int(episodes)):
        if verbose:
            print(f"[{'gated' if use_gate else 'mappo'}] episode {ep + 1}/{int(episodes)}", flush=True)
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        histories = {a: {"states": [], "actions": [], "returns": [], "steps": [], "cum_reward": 0.0} for a in env.agents}
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        executed_steps = 0

        for _ in range(horizon):
            masks = env.get_action_mask()
            actions_env: Dict[str, int] = {}
            obs_snapshot: Dict[str, np.ndarray] = {}
            rtg_snapshot: Dict[str, float] = {}
            for a in env.agents:
                obs_now = np.asarray(obs_dict[a], dtype=np.float32)
                am = np.asarray(masks[a], dtype=np.bool_)
                if use_gate:
                    act_idx, _, unc, gate = _fused_action(
                        actor_like=actor_like,
                        dt_policy=dt_policy,
                        history=histories[a],
                        obs_now=obs_now,
                        action_mask=am,
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
                        deterministic=deterministic,
                    )
                    gate_sum += gate
                    uncertainty_sum += unc
                    action_count += 1.0
                else:
                    act_idx, _, _ = actor_like.act(obs=obs_now, action_mask=am, device=device, deterministic=deterministic)
                actions_env[a] = int(act_idx + 1)
                obs_snapshot[a] = obs_now
                rtg_snapshot[a] = float(target_return - histories[a]["cum_reward"])

            sr = env.step_parallel(actions_env)
            executed_steps += 1
            for req in sr.requests:
                stats["requests"] += 1.0
                if req.get("service_mode") == "fcs":
                    stats["success_requests"] += 1.0
                else:
                    stats["mcs_requests"] += 1.0
                    pending_by_ev[int(req["ev_id"])] = {"step": float(int(req.get("step", sr.step))), "required_kwh": float(req.get("required_kwh", 0.0))}
            for event in sr.mcs_events:
                req_info = None
                if str(event.get("action", "")) == "serve_request":
                    req_info = pending_by_ev.pop(int(event["ev_id"]), {"step": float(sr.step), "required_kwh": float(event.get("required_kwh", 0.0))})
                stats["total_mcs_income"] += _event_income(env, event, req_info)
                if str(event.get("action", "")) == "serve_request":
                    req_step = int((req_info or {}).get("step", float(sr.step)))
                    wait_steps = max(0, int(sr.step) - req_step)
                    stats["mcs_served"] += 1.0
                    stats["success_requests"] += 1.0
                    stats["wait_steps_sum"] += float(wait_steps)
                    stats["wait_count"] += 1.0
            for to in sr.timeout_events:
                stats["timeout_events_total"] += 1.0
                ev_id = int(to.get("ev_id", -1))
                req_step = int(to.get("request_step", sr.step))
                wait_steps = int(to.get("wait_steps", max(0, int(sr.step) - req_step)))
                pending_by_ev.pop(ev_id, None)
                stats["wait_steps_sum"] += float(wait_steps)
                stats["wait_count"] += 1.0
            for a in env.agents:
                rew = float(sr.agent_rewards[a])
                histories[a]["cum_reward"] += rew
                onehot = np.zeros((action_dim,), dtype=np.float32)
                onehot[int(actions_env[a] - 1)] = 1.0
                histories[a]["states"].append(obs_snapshot[a])
                histories[a]["actions"].append(onehot)
                histories[a]["returns"].append(np.asarray([rtg_snapshot[a]], dtype=np.float32))
                histories[a]["steps"].append(int(np.clip(sr.step, 0, max_ep_len - 1)))
                stats["total_agent_reward"] += rew
            if sr.done:
                break
            obs_dict = env.get_agent_observations()

        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)

    out = _finalize_local(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)
    if use_gate:
        out["gate_rate"] = float(gate_sum / max(1.0, action_count))
        out["avg_uncertainty"] = float(uncertainty_sum / max(1.0, action_count))
        out["threshold"] = float(threshold)
        out["alpha"] = float(alpha)
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    env_cfg = dict(CONFIG)
    env_cfg["use_lstm_summary"] = not bool(args.no_lstm)
    if env_cfg["use_lstm_summary"]:
        env_cfg["lstm_predictor_ckpt"] = str(env_cfg.get("lstm_predictor_ckpt", "result/predictor/lstm_predictor.pt"))

    actor_like = _load_policy_actor(Path(args.mappo_ckpt), device=device)
    dt_policy, dt_meta, _ = load_dt_prior(
        Path(args.dt_ckpt),
        device=device,
        target_return=float(args.target_return),
        target_return_scale=float(args.target_return_scale),
    )

    deterministic = not bool(args.eval_stochastic)
    baseline = evaluate(
        actor_like=actor_like,
        dt_policy=dt_policy,
        dt_meta=dt_meta,
        env_cfg=env_cfg,
        episodes=int(args.episodes),
        seed=int(args.seed),
        device=device,
        max_steps=args.max_steps,
        threshold=float(args.threshold),
        alpha=float(args.alpha),
        soft_gate=bool(args.soft_gate),
        temperature=float(args.temperature),
        deterministic=deterministic,
        use_gate=False,
        verbose=bool(args.verbose),
    )
    gated = evaluate(
        actor_like=actor_like,
        dt_policy=dt_policy,
        dt_meta=dt_meta,
        env_cfg=env_cfg,
        episodes=int(args.episodes),
        seed=int(args.seed),
        device=device,
        max_steps=args.max_steps,
        threshold=float(args.threshold),
        alpha=float(args.alpha),
        soft_gate=bool(args.soft_gate),
        temperature=float(args.temperature),
        deterministic=deterministic,
        use_gate=True,
        verbose=bool(args.verbose),
    )
    results = {"MAPPO": baseline, "MAPPO+DT-UncertaintyGate": gated}
    path = outdir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved metrics: {path}")


if __name__ == "__main__":
    main()
