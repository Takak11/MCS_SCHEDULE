from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from build_offline_dataset_ppo import _load_policy_actor
from config import CONFIG
from env import Environment
from train_dt_ppo_finetune import DTPolicyWithValue, _build_dt_from_ckpt, _business_score
from train_mn_icrsp_transformer_pg import TransformerActorCritic, evaluate_policy as evaluate_transformer_pg
from utils import haversine_distance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and plot RWS / MN-ICRSP-style Transformer PG / our ablations in one figure.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default="result/method_compare")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--include-greedy-2opt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _event_income(env: Environment, event: dict, req_info: Optional[Dict[str, float]] = None) -> float:
    mcs_id = int(event.get("mcs_id", -1))
    mcs = env.mcs_by_id.get(mcs_id)
    if mcs is None:
        return 0.0
    income = -float(event.get("distance_km", 0.0)) * float(mcs.cost_per_km)
    if str(event.get("action", "")) == "serve_request":
        required_kwh = float((req_info or {}).get("required_kwh", event.get("required_kwh", 0.0)))
        income += required_kwh * float(mcs.price_per_kwh)
    return float(income)


def _init_stats() -> Dict[str, float]:
    return {
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


def _finalize_stats(stats: Dict[str, float], episodes: int, n_mcs: int, step_minutes: float) -> Dict[str, float]:
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
    out["business_score"] = float(_business_score(out, argparse.Namespace()))
    return out


def _evaluate_random(env_cfg: dict, episodes: int, seed: int, max_steps: Optional[int], verbose: bool = False) -> Dict[str, float]:
    rng = np.random.default_rng(seed + 101)
    env = Environment(config=env_cfg, seed=seed)
    stats = _init_stats()
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    for ep in range(int(episodes)):
        if verbose:
            print(f"[RWS] episode {ep + 1}/{int(episodes)}")
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        _ = obs_dict
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        for _ in range(horizon):
            masks = env.get_action_mask()
            actions: Dict[str, int] = {}
            for a in env.agents:
                valid = np.where(np.asarray(masks[a], dtype=np.bool_))[0]
                act_idx = int(rng.choice(valid)) if valid.size > 0 else 3
                actions[a] = int(act_idx + 1)
            sr = env.step_parallel(actions)
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
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))
            if sr.done:
                break
        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)
    return _finalize_stats(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)


def _fcs_pressure(env: Environment) -> float:
    arrivals = env.fcs_arrival_schedule.get(env.current_step, {})
    pressure = 0.0
    for fcs in env.fcs_list:
        queue_now = float(env.fcs_queue.get(fcs.fcs_id, 0))
        free_slots = max(0.0, float(fcs.capacity - fcs.occupied))
        pressure += max(0.0, queue_now + float(arrivals.get(fcs.fcs_id, 0)) - free_slots)
    return float(pressure)


def _framework_actions(env: Environment, rng: np.random.Generator, relocate_interval: int = 12) -> Dict[str, int]:
    """
    Heuristic operational framework baseline:
      1) service pending requests first,
      2) reinforce overloaded FCS when pressure exists,
      3) relocate idle units periodically for coverage,
      4) otherwise stay.
    """
    masks = env.get_action_mask()
    has_pending = any(req.get("location") is not None for req in env.pending_ev_requests)
    pressure = _fcs_pressure(env)
    do_relocate = int(relocate_interval) > 0 and int(env.current_step) % int(relocate_interval) == 0
    actions: Dict[str, int] = {}
    for a in env.agents:
        mask = np.asarray(masks[a], dtype=np.bool_)
        action = 4
        if has_pending and mask[2]:
            action = 3
        elif pressure > 0.0 and mask[0]:
            action = 1
        elif do_relocate and mask[1]:
            action = 2 if float(rng.random()) < 0.6 else 4
        actions[a] = int(action if mask[int(action) - 1] else 4)
    return actions


def _evaluate_framework(env_cfg: dict, episodes: int, seed: int, max_steps: Optional[int], verbose: bool = False) -> Dict[str, float]:
    rng = np.random.default_rng(seed + 2026)
    env = Environment(config=env_cfg, seed=seed)
    stats = _init_stats()
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    for ep in range(int(episodes)):
        if verbose:
            print(f"[Comp-Framework] episode {ep + 1}/{int(episodes)}")
        env.reset(seed=int(seed + (ep + 1) * 9973))
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        for _ in range(horizon):
            actions = _framework_actions(env=env, rng=rng, relocate_interval=12)
            sr = env.step_parallel(actions)
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
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))
            if sr.done:
                break
        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)
    return _finalize_stats(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)


def _surrogate_gain(action: int, obs: np.ndarray, has_pending: bool, pressure: float) -> float:
    local_req = float(obs[5]) if obs.size > 5 else 0.0
    avg_wait = float(obs[6]) if obs.size > 6 else 0.0
    nearest_req = float(obs[7]) if obs.size > 7 else 2.0
    risk = float(obs[9]) if obs.size > 9 else 0.0
    cong = float(obs[11]) if obs.size > 11 else 0.0
    nearest_fcs_q = float(obs[14]) if obs.size > 14 else 0.0
    gap = float(obs[15]) if obs.size > 15 else 0.0
    if int(action) == 3:
        return 8.0 * local_req + 2.0 * max(0.0, gap) + 1.4 * avg_wait + 1.2 * max(0.0, 1.0 - nearest_req) + (1.2 if has_pending else -1.0)
    if int(action) == 1:
        return 0.35 * pressure + 0.8 * risk + 0.7 * nearest_fcs_q + 0.4 * cong
    if int(action) == 2:
        return 0.6 * max(0.0, gap) + 0.5 * risk
    return 0.2 - 0.3 * avg_wait


def _request_priority(env: Environment, req: dict) -> float:
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    timeout_minutes = float(env.config.get("ev_request_timeout_minutes", 30.0))
    wait_min = max(0.0, float(env.current_step - int(req.get("step", env.current_step)))) * step_minutes
    return float(1.0 + wait_min / max(step_minutes, timeout_minutes))


def _service_score(env: Environment, agent: str, req: dict) -> float:
    mcs = env.mcs_by_id[env.agent_to_mcs_id[agent]]
    if mcs.location is None or req.get("location") is None:
        return float("-inf")
    dist = float(haversine_distance(mcs.location, req["location"]))
    if dist > float(mcs.service_radius_km):
        return float("-inf")
    radius = max(1e-6, float(mcs.service_radius_km))
    return float(4.0 * _request_priority(env, req) + 2.5 * (1.0 - dist / radius))


def _seed_service_assignments(env: Environment, agents: list[str], masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    pending = [r for r in env.pending_ev_requests if r.get("location") is not None]
    actions = {a: 4 for a in agents}
    if not pending:
        return actions
    capacity = int(max(1, env.config.get("mcs_service_parallel_capacity", 1)))
    service_budget = int(np.ceil(len(pending) / max(1, capacity)))
    candidates: list[tuple[float, str]] = []
    for a in agents:
        if not np.asarray(masks[a], dtype=np.bool_)[2]:
            continue
        best = max((_service_score(env, a, req) for req in pending), default=float("-inf"))
        if np.isfinite(best):
            candidates.append((float(best), a))
    candidates.sort(reverse=True, key=lambda x: x[0])
    for _, a in candidates[:service_budget]:
        actions[a] = 3
    return actions


def _evaluate_greedy_2opt_one_to_many(env_cfg: dict, episodes: int, seed: int, max_steps: Optional[int], relocate_interval: int = 12, verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate one-to-many proxy with fewer physical MCS and larger per-MCS service capacity.

    The one-to-many comparison is modeled in the environment, not by pairing two
    physical MCS into a virtual unit: env_cfg should set mcs_num to half of the
    base fleet and mcs_service_parallel_capacity to 2.
    """
    rng = np.random.default_rng(seed + 3072)
    env = Environment(config=env_cfg, seed=seed)
    stats = _init_stats()
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    for ep in range(int(episodes)):
        if verbose:
            print(f"[Greedy+2Opt+One-to-Many] episode {ep + 1}/{int(episodes)}")
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        for _ in range(horizon):
            masks = env.get_action_mask()
            has_pending = any(req.get("location") is not None for req in env.pending_ev_requests)
            pressure = _fcs_pressure(env)
            do_relocate = int(relocate_interval) > 0 and int(env.current_step) % int(relocate_interval) == 0
            agents = list(env.agents)
            actions = _seed_service_assignments(env, agents, masks)
            remaining = [a for a in agents if actions.get(a, 4) == 4]
            reinforce_budget = 0
            if pressure > 0.0:
                support = max(1.0, float(env.config.get("mcs_reinforce_ev_per_step", 1.0)))
                reinforce_budget = int(min(len(remaining), np.ceil(pressure / support)))
            reinforce_candidates: list[tuple[float, str]] = []
            relocate_candidates: list[tuple[float, str]] = []
            for a in remaining:
                mk = np.asarray(masks[a], dtype=np.bool_)
                obs = np.asarray(obs_dict[a], dtype=np.float32)
                if mk[0]:
                    reinforce_candidates.append((_surrogate_gain(1, obs, has_pending, pressure), a))
                if do_relocate and mk[1]:
                    relocate_candidates.append((_surrogate_gain(2, obs, has_pending, pressure), a))
            reinforce_candidates.sort(reverse=True, key=lambda x: x[0])
            for score, a in reinforce_candidates[:reinforce_budget]:
                if score > _surrogate_gain(4, np.asarray(obs_dict[a], dtype=np.float32), has_pending, pressure):
                    actions[a] = 1
            for score, a in relocate_candidates:
                if actions.get(a, 4) != 4:
                    continue
                obs = np.asarray(obs_dict[a], dtype=np.float32)
                if score > max(_surrogate_gain(4, obs, has_pending, pressure), _surrogate_gain(1, obs, has_pending, pressure)) + 0.15:
                    actions[a] = 2
            for a in agents:
                mk = np.asarray(masks[a], dtype=np.bool_)
                actions[a] = int(actions.get(a, 4) if mk[int(actions.get(a, 4)) - 1] else 4)

            # 2-opt local swap over real MCS action assignments.
            def total_score(assign: Dict[str, int]) -> float:
                return float(sum(_surrogate_gain(assign[a], np.asarray(obs_dict[a], dtype=np.float32), has_pending, pressure) for a in agents))

            improved = True
            while improved:
                improved = False
                base = total_score(actions)
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        ai, aj = agents[i], agents[j]
                        old_i, old_j = actions[ai], actions[aj]
                        if old_i == old_j:
                            continue
                        actions[ai], actions[aj] = old_j, old_i
                        if not np.asarray(masks[ai], dtype=np.bool_)[actions[ai] - 1]:
                            actions[ai], actions[aj] = old_i, old_j
                            continue
                        if not np.asarray(masks[aj], dtype=np.bool_)[actions[aj] - 1]:
                            actions[ai], actions[aj] = old_i, old_j
                            continue
                        cand = total_score(actions)
                        if cand > base + 1e-9:
                            base = cand
                            improved = True
                        else:
                            actions[ai], actions[aj] = old_i, old_j
                    if improved:
                        break
            sr = env.step_parallel(actions)
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
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))
            if sr.done:
                break
            obs_dict = env.get_agent_observations()
        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)
    return _finalize_stats(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)


def _evaluate_actor_ckpt(ckpt_path: Path, env_cfg: dict, episodes: int, seed: int, device: torch.device, max_steps: Optional[int], deterministic: bool, verbose: bool = False, tag: str = "Actor") -> Dict[str, float]:
    actor = _load_policy_actor(ckpt_path, device=device)
    env = Environment(config=env_cfg, seed=seed)
    stats = _init_stats()
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    for ep in range(int(episodes)):
        if verbose:
            print(f"[{tag}] episode {ep + 1}/{int(episodes)}")
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed_steps = 0
        for _ in range(horizon):
            masks = env.get_action_mask()
            actions: Dict[str, int] = {}
            for a in env.agents:
                obs = np.asarray(obs_dict[a], dtype=np.float32)
                am = np.asarray(masks[a], dtype=np.bool_)
                act_idx, _, _ = actor.act(obs=obs, action_mask=am, device=device, deterministic=deterministic)
                actions[a] = int(act_idx + 1)
            sr = env.step_parallel(actions)
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
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))
            if sr.done:
                break
            obs_dict = env.get_agent_observations()
        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed_steps - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed_steps - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed_steps)
    return _finalize_stats(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)


def _evaluate_transformer_pg_ckpt(ckpt_path: Path, env_cfg: dict, episodes: int, seed: int, device: torch.device, max_steps: Optional[int], deterministic: bool) -> Dict[str, float]:
    ck = torch.load(ckpt_path, map_location="cpu")
    model = TransformerActorCritic(
        obs_dim=int(ck["obs_dim"]),
        action_dim=int(ck["action_dim"]),
        n_agents=int(ck["n_agents"]),
        hidden_dim=int(ck.get("hidden_dim", ck.get("args", {}).get("hidden_dim", 128))),
        n_layer=int(ck.get("n_layer", ck.get("args", {}).get("n_layer", 2))),
        n_head=int(ck.get("n_head", ck.get("args", {}).get("n_head", 4))),
        dropout=float(ck.get("dropout", ck.get("args", {}).get("dropout", 0.1))),
        cand_feat_dim=int(ck.get("cand_feat_dim", 0)),
    ).to(device)
    model.load_state_dict(ck["transformer_actor_critic_state_dict"])
    model.eval()
    out = evaluate_transformer_pg(
        model=model,
        env_config=env_cfg,
        episodes=int(episodes),
        seed=int(seed),
        device=device,
        action_dim=int(ck["action_dim"]),
        deterministic=bool(deterministic),
        max_steps=max_steps,
        req_k=int(ck.get("candidate_requests", ck.get("args", {}).get("candidate_requests", 8))),
        fcs_k=int(len(env_cfg.get("fcs_locations", []))),
        hot_k=int(ck.get("candidate_hotspots", ck.get("args", {}).get("candidate_hotspots", 8))),
    )
    out["business_score"] = float(_business_score(out, argparse.Namespace()))
    return out


def _load_dt_policy(ckpt_path: Path, device: torch.device) -> DTPolicyWithValue:
    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck["args"]
    dt_ckpt = torch.load(Path(args["dt_ckpt"]), map_location="cpu")
    dt = _build_dt_from_ckpt(dt_ckpt, device=device)
    dt.load_state_dict(ck["dt_model_state_dict"])
    pol = DTPolicyWithValue(
        dt_model=dt,
        state_mean=np.asarray(dt_ckpt["state_mean"], dtype=np.float32),
        state_std=np.asarray(dt_ckpt["state_std"], dtype=np.float32),
        rtg_scale=float(dt_ckpt["rtg_scale"]),
    ).to(device)
    pol.value_head.load_state_dict(ck["value_head_state_dict"])
    pol.eval()
    return pol


def _evaluate_dt_ckpt(ckpt_path: Path, env_cfg: dict, episodes: int, seed: int, device: torch.device, max_steps: Optional[int], deterministic: bool, verbose: bool = False, tag: str = "DT") -> Dict[str, float]:
    ck = torch.load(ckpt_path, map_location="cpu")
    policy = _load_dt_policy(ckpt_path, device=device)
    dt_ckpt = torch.load(Path(ck["args"]["dt_ckpt"]), map_location="cpu")
    context_len = int(ck["args"].get("context_len", 0)) if int(ck["args"].get("context_len", 0)) > 0 else int(dt_ckpt["context_len"])
    max_ep_len = int(dt_ckpt["max_ep_len"])
    action_dim = int(dt_ckpt["action_dim"])
    target_return = float(ck.get("meta", {}).get("target_return", ck["args"].get("target_return", 0.0)))
    env = Environment(config=env_cfg, seed=seed)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    stats = _init_stats()

    for ep in range(int(episodes)):
        if verbose:
            print(f"[{tag}] episode {ep + 1}/{int(episodes)}")
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
                rtg_now = float(target_return - histories[a]["cum_reward"])
                zero_a = np.zeros((action_dim,), dtype=np.float32)
                s_seq = histories[a]["states"] + [obs_now]
                a_seq = histories[a]["actions"] + [zero_a]
                r_seq = histories[a]["returns"] + [np.asarray([rtg_now], dtype=np.float32)]
                t_seq = histories[a]["steps"] + [int(np.clip(env.current_step, 0, max_ep_len - 1))]
                if len(s_seq) > context_len:
                    s_seq, a_seq, r_seq, t_seq = s_seq[-context_len:], a_seq[-context_len:], r_seq[-context_len:], t_seq[-context_len:]
                pad = context_len - len(s_seq)
                ctx_states = np.zeros((context_len, obs_now.shape[0]), dtype=np.float32)
                ctx_actions = np.zeros((context_len, action_dim), dtype=np.float32)
                ctx_returns = np.zeros((context_len, 1), dtype=np.float32)
                ctx_steps = np.zeros((context_len,), dtype=np.int64)
                ctx_attention = np.zeros((context_len,), dtype=np.int64)
                ctx_states[pad:] = np.asarray(s_seq, dtype=np.float32)
                ctx_actions[pad:] = np.asarray(a_seq, dtype=np.float32)
                ctx_returns[pad:] = np.asarray(r_seq, dtype=np.float32).reshape(-1, 1)
                ctx_steps[pad:] = np.asarray(t_seq, dtype=np.int64)
                ctx_attention[pad:] = 1
                am = np.asarray(masks[a], dtype=np.bool_)
                act_idx, _, _ = policy.act(ctx_states, ctx_actions, ctx_returns, ctx_steps, ctx_attention, am, device=device, deterministic=deterministic)
                actions_env[a] = int(act_idx + 1)
                obs_snapshot[a] = obs_now
                rtg_snapshot[a] = rtg_now
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

    return _finalize_stats(stats, episodes=episodes, n_mcs=len(env.mcs_list), step_minutes=step_minutes)


def _plot(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    methods = list(results.keys())
    success = [results[m]["success_rate"] for m in methods]
    wait = [results[m]["avg_wait_minutes"] for m in methods]
    biz = [results[m]["business_score"] for m in methods]

    x = np.arange(len(methods))
    width = 0.24

    fig, ax1 = plt.subplots(figsize=(14, 6))
    b1 = ax1.bar(x - width, success, width=width, label="Success Rate", color="#2a9d8f")
    b2 = ax1.bar(x, biz, width=width, label="Business Score", color="#457b9d")
    ax1.set_ylabel("Success / Business")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=10)
    ax1.grid(alpha=0.25, axis="y")

    ax2 = ax1.twinx()
    b3 = ax2.bar(x + width, wait, width=width, label="Avg Wait (min)", color="#e76f51")
    ax2.set_ylabel("Waiting Minutes")

    lines = [b1, b2, b3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Method Comparison (RWS / Greedy+2Opt+One-to-Many / MN-ICRSP-style Transformer PG / Ours)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    base_cfg = dict(CONFIG)
    base_cfg["lstm_predictor_ckpt"] = str(base_cfg.get("lstm_predictor_ckpt", "result/predictor/lstm_predictor.pt"))

    ck_no_dt = Path("result/ppo_for_offline/best_by_business.pt")
    ck_ours = Path("result/dt_ppo_ft/best_by_business.pt")
    ck_mn = Path("result/mn_icrsp_transformer_pg/best_by_business.pt")
    if not (ck_no_dt.exists() and ck_ours.exists() and ck_mn.exists()):
        raise RuntimeError("Missing checkpoint(s) for one or more methods. Please ensure result files exist.")

    results: Dict[str, Dict[str, float]] = {}
    # RWS random
    cfg_rws = dict(base_cfg)
    cfg_rws["use_lstm_summary"] = False
    print("[RUN] RWS")
    results["RWS"] = _evaluate_random(cfg_rws, episodes=int(args.episodes), seed=int(args.seed), max_steps=args.max_steps, verbose=bool(args.verbose))

    if bool(args.include_greedy_2opt):
        cfg_one_many = dict(base_cfg)
        cfg_one_many["use_lstm_summary"] = False
        cfg_one_many["mcs_num"] = max(1, int(base_cfg.get("mcs_num", 1)) // 2)
        cfg_one_many["mcs_service_parallel_capacity"] = 2
        print("[RUN] Greedy+2Opt+One-to-Many")
        results["Greedy+2Opt+One-to-Many"] = _evaluate_greedy_2opt_one_to_many(
            cfg_one_many,
            episodes=int(args.episodes),
            seed=int(args.seed),
            max_steps=args.max_steps,
            relocate_interval=12,
            verbose=bool(args.verbose),
        )

    # MN-ICRSP-style Transformer policy-gradient comparison method (trained without LSTM by default)
    cfg_mn = dict(base_cfg)
    cfg_mn["use_lstm_summary"] = False
    print("[RUN] MN-ICRSP-style Transformer PG")
    results["MN-ICRSP-style Transformer PG"] = _evaluate_transformer_pg_ckpt(ck_mn, cfg_mn, int(args.episodes), int(args.seed), device, args.max_steps, deterministic=not bool(args.eval_stochastic))

    # Ours w/o DT finetune (PPO/MAPPO policy)
    cfg_no_dt = dict(base_cfg)
    cfg_no_dt["use_lstm_summary"] = True
    print("[RUN] Ours-NoDTTune")
    results["Ours-NoDTTune"] = _evaluate_actor_ckpt(ck_no_dt, cfg_no_dt, int(args.episodes), int(args.seed), device, args.max_steps, deterministic=not bool(args.eval_stochastic), verbose=bool(args.verbose), tag="Ours-NoDTTune")

    # Ours full (DT+PPO FT, with LSTM)
    cfg_ours = dict(base_cfg)
    cfg_ours["use_lstm_summary"] = True
    print("[RUN] Ours")
    results["Ours"] = _evaluate_dt_ckpt(ck_ours, cfg_ours, int(args.episodes), int(args.seed), device, args.max_steps, deterministic=not bool(args.eval_stochastic), verbose=bool(args.verbose), tag="Ours")

    # Ours without LSTM mechanism (same DT policy, LSTM features off in env)
    cfg_no_lstm = dict(base_cfg)
    cfg_no_lstm["use_lstm_summary"] = False
    cfg_no_lstm["lstm_predictor_ckpt"] = ""
    print("[RUN] Ours-NoLSTM")
    results["Ours-NoLSTM"] = _evaluate_dt_ckpt(ck_ours, cfg_no_lstm, int(args.episodes), int(args.seed), device, args.max_steps, deterministic=not bool(args.eval_stochastic), verbose=bool(args.verbose), tag="Ours-NoLSTM")

    json_path = outdir / "method_compare_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    fig_path = outdir / "method_compare_all_in_one.png"
    _plot(results, fig_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved metrics: {json_path}")
    print(f"saved figure: {fig_path}")


if __name__ == "__main__":
    main()
