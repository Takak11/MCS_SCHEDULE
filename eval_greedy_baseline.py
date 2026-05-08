from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import CONFIG
from env import Environment
from utils import haversine_distance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate non-learning greedy baselines without LSTM/predictive mechanisms.")
    p.add_argument("--outdir", type=str, default="result/greedy_baseline")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--policy",
        type=str,
        default="service",
        choices=["stay", "service", "service_reinforce", "service_relocate", "local_greedy", "greedy_2opt"],
        help="greedy_2opt is the unified Greedy+2Opt+One-to-Many paper baseline.",
    )
    p.add_argument("--relocate-interval", type=int, default=12)
    p.add_argument(
        "--service-prob",
        type=float,
        default=1.0,
        help="Probability of taking a greedy service action when service is selected. Values below 1 create a weaker greedy baseline.",
    )
    p.add_argument("--log-episodes", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _event_mcs_income(env: Environment, event: dict, req_info: Optional[Dict[str, float]] = None) -> float:
    mcs_id = int(event.get("mcs_id", -1))
    mcs = env.mcs_by_id.get(mcs_id)
    if mcs is None:
        return 0.0
    distance_km = float(event.get("distance_km", 0.0))
    income = -distance_km * float(mcs.cost_per_km)
    if str(event.get("action", "")) == "serve_request":
        required_kwh = float((req_info or {}).get("required_kwh", event.get("required_kwh", 0.0)))
        income += required_kwh * float(mcs.price_per_kwh)
    return float(income)


def _fcs_pressure(env: Environment) -> float:
    arrivals = env.fcs_arrival_schedule.get(env.current_step, {})
    pressure = 0.0
    for fcs in env.fcs_list:
        queue_now = float(env.fcs_queue.get(fcs.fcs_id, 0))
        free_slots = max(0.0, float(fcs.capacity - fcs.occupied))
        pressure += max(0.0, queue_now + float(arrivals.get(fcs.fcs_id, 0)) - free_slots)
    return float(pressure)


def _local_greedy_action(obs: np.ndarray, mask: np.ndarray) -> int:
    local_req = float(obs[5]) if obs.size > 5 else 0.0
    nearest_req = float(obs[7]) if obs.size > 7 else 2.0
    high_risk_fcs = float(obs[9]) if obs.size > 9 else 0.0
    nearest_fcs_queue = float(obs[14]) if obs.size > 14 else 0.0
    local_gap = float(obs[15]) if obs.size > 15 else 0.0

    if mask[2] and (local_req > 0.0 or nearest_req <= 1.0 or local_gap > 0.0):
        return 3
    if mask[0] and (nearest_fcs_queue > 1.0 or high_risk_fcs > 0.0):
        return 1
    return 4


def _surrogate_gain(action: int, obs: np.ndarray, has_pending: bool, pressure: float) -> float:
    local_req = float(obs[5]) if obs.size > 5 else 0.0
    avg_wait = float(obs[6]) if obs.size > 6 else 0.0
    nearest_req = float(obs[7]) if obs.size > 7 else 2.0
    risk = float(obs[9]) if obs.size > 9 else 0.0
    cong = float(obs[11]) if obs.size > 11 else 0.0
    nearest_fcs_q = float(obs[14]) if obs.size > 14 else 0.0
    gap = float(obs[15]) if obs.size > 15 else 0.0

    if int(action) == 3:  # service
        return 8.0 * local_req + 2.0 * max(0.0, gap) + 1.4 * avg_wait + 1.2 * max(0.0, 1.0 - nearest_req) + (1.2 if has_pending else -1.0)
    if int(action) == 1:  # reinforce
        return 0.35 * pressure + 0.8 * risk + 0.7 * nearest_fcs_q + 0.4 * cong
    if int(action) == 2:  # relocate
        return 0.6 * max(0.0, gap) + 0.5 * risk
    return 0.2 - 0.3 * avg_wait  # stay


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


def _seed_service_assignments(env: Environment, agents: List[str], masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    pending = [r for r in env.pending_ev_requests if r.get("location") is not None]
    actions = {a: 4 for a in agents}
    if not pending:
        return actions

    capacity = int(max(1, env.config.get("mcs_service_parallel_capacity", 1)))
    service_budget = int(np.ceil(len(pending) / max(1, capacity)))
    candidates: List[Tuple[float, str]] = []
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


def _greedy_2opt_actions(env: Environment, obs_dict: Dict[str, np.ndarray], rng: np.random.Generator, relocate_interval: int) -> Dict[str, int]:
    mask_dict = env.get_action_mask()
    has_pending = any(req.get("location") is not None for req in env.pending_ev_requests)
    pressure = _fcs_pressure(env)
    do_relocate = int(relocate_interval) > 0 and int(env.current_step) % int(relocate_interval) == 0

    agents = list(env.agents)
    actions: Dict[str, int] = _seed_service_assignments(env, agents, mask_dict)
    remaining = [a for a in agents if actions.get(a, 4) == 4]

    reinforce_budget = 0
    if pressure > 0.0:
        support = max(1.0, float(env.config.get("mcs_reinforce_ev_per_step", 1.0)))
        reinforce_budget = int(min(len(remaining), np.ceil(pressure / support)))
    reinforce_candidates: List[Tuple[float, str]] = []
    relocate_candidates: List[Tuple[float, str]] = []
    stay_candidates: List[str] = []
    for a in remaining:
        mask = np.asarray(mask_dict[a], dtype=np.bool_)
        obs = np.asarray(obs_dict[a], dtype=np.float32)
        if mask[0]:
            reinforce_candidates.append((_surrogate_gain(1, obs, has_pending, pressure), a))
        if do_relocate and mask[1]:
            relocate_candidates.append((_surrogate_gain(2, obs, has_pending, pressure), a))
        stay_candidates.append(a)
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
        mask = np.asarray(mask_dict[a], dtype=np.bool_)
        if actions.get(a, 4) != 4:
            actions[a] = int(actions[a] if mask[int(actions[a]) - 1] else 4)
            continue
        actions[a] = 4

    # 2-opt style local swap improvement over action assignment (pairwise exchanges).
    def total_score(assign: Dict[str, int]) -> float:
        return float(sum(_surrogate_gain(assign[a], np.asarray(obs_dict[a], dtype=np.float32), has_pending, pressure) for a in agents))

    improved = True
    while improved:
        improved = False
        base = total_score(actions)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                ci, cj = actions[ai], actions[aj]
                if ci == cj:
                    continue
                # swap two assignment "segments"
                actions[ai], actions[aj] = cj, ci
                # keep feasibility
                if not np.asarray(mask_dict[ai], dtype=np.bool_)[actions[ai] - 1]:
                    actions[ai], actions[aj] = ci, cj
                    continue
                if not np.asarray(mask_dict[aj], dtype=np.bool_)[actions[aj] - 1]:
                    actions[ai], actions[aj] = ci, cj
                    continue
                cand = total_score(actions)
                if cand > base + 1e-9:
                    base = cand
                    improved = True
                else:
                    actions[ai], actions[aj] = ci, cj
            if improved:
                break
    # tiny randomized tie-break to avoid deterministic lock-in
    if float(rng.random()) < 0.02:
        k = agents[int(rng.integers(0, len(agents)))]
        if np.asarray(mask_dict[k], dtype=np.bool_)[3]:
            actions[k] = 4
    return actions

def _maybe_throttle_service(action: int, service_prob: float, rng: np.random.Generator) -> int:
    if int(action) != 3:
        return int(action)
    p = float(np.clip(service_prob, 0.0, 1.0))
    if p >= 1.0:
        return 3
    return 3 if float(rng.random()) < p else 4


def greedy_actions(
    env: Environment,
    policy: str,
    relocate_interval: int,
    service_prob: float,
    rng: np.random.Generator,
    obs_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, int]:
    mask_dict = env.get_action_mask()
    has_pending = any(req.get("location") is not None for req in env.pending_ev_requests)
    pressure = _fcs_pressure(env)
    do_relocate = int(relocate_interval) > 0 and int(env.current_step) % int(relocate_interval) == 0

    actions: Dict[str, int] = {}
    if policy == "greedy_2opt":
        return _greedy_2opt_actions(
            env=env,
            obs_dict=obs_dict or env.get_agent_observations(),
            rng=rng,
            relocate_interval=relocate_interval,
        )
    for agent in env.agents:
        mask = np.asarray(mask_dict[agent], dtype=np.bool_)
        action = 4
        if policy == "local_greedy":
            obs = np.asarray((obs_dict or {}).get(agent, np.zeros((16,), dtype=np.float32)), dtype=np.float32)
            action = _local_greedy_action(obs=obs, mask=mask)
        elif policy == "stay":
            action = 4
        elif has_pending and mask[2]:
            action = 3
        elif policy in {"service_reinforce", "service_relocate"} and pressure > 0.0 and mask[0]:
            action = 1
        elif policy == "service_relocate" and do_relocate and mask[1]:
            action = 2
        action = _maybe_throttle_service(action=action, service_prob=float(service_prob), rng=rng)
        actions[agent] = int(action if mask[int(action) - 1] else 4)
    return actions


def _business_score(stats: Dict[str, float], args: argparse.Namespace) -> float:
    return float(
        1000.0 * float(stats.get("success_rate", 0.0))
        + 300.0 * float(stats.get("mcs_success_rate", 0.0))
        - 25.0 * float(stats.get("avg_wait_minutes", 0.0))
        - 0.05 * float(stats.get("timeout_events_total", 0.0))
    )


def run_episode(env: Environment, seed: int, args: argparse.Namespace) -> Dict[str, float]:
    obs_dict = env.reset(seed=int(seed))
    policy_rng = np.random.default_rng(int(seed) + 7919)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    horizon = env.total_steps if args.max_steps is None else min(int(args.max_steps), env.total_steps)

    total_requests = 0
    mcs_requests = 0
    success_requests = 0
    mcs_served = 0
    timeout_events_total = 0
    unresolved_mcs_total = 0
    wait_steps_sum = 0.0
    wait_count = 0
    total_agent_reward = 0.0
    total_mcs_income = 0.0
    action_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    mcs_pending_by_ev: Dict[int, Dict[str, float]] = {}
    executed_steps = 0

    for _ in range(horizon):
        actions = greedy_actions(
            env=env,
            policy=str(args.policy),
            relocate_interval=int(args.relocate_interval),
            service_prob=float(args.service_prob),
            rng=policy_rng,
            obs_dict=obs_dict,
        )
        for a in actions.values():
            action_counts[int(a)] += 1
        step_result = env.step_parallel(actions)
        executed_steps += 1

        for req in step_result.requests:
            total_requests += 1
            if req.get("service_mode") == "fcs":
                success_requests += 1
            else:
                mcs_requests += 1
                mcs_pending_by_ev[int(req["ev_id"])] = {
                    "step": float(int(req.get("step", step_result.step))),
                    "required_kwh": float(req.get("required_kwh", 0.0)),
                }

        for event in step_result.mcs_events:
            action = str(event.get("action", ""))
            req_info: Optional[Dict[str, float]] = None
            if action == "serve_request":
                ev_id = int(event["ev_id"])
                req_info = mcs_pending_by_ev.pop(
                    ev_id,
                    {"step": float(step_result.step), "required_kwh": float(event.get("required_kwh", 0.0))},
                )
            total_mcs_income += _event_mcs_income(env=env, event=event, req_info=req_info)
            if action != "serve_request":
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

        total_agent_reward += float(sum(step_result.agent_rewards.values()))
        if step_result.done:
            break
        obs_dict = env.get_agent_observations()

    for req_info in mcs_pending_by_ev.values():
        req_step = int(req_info.get("step", float(executed_steps - 1)))
        wait_steps = max(0, int(executed_steps - 1) - req_step)
        wait_steps_sum += float(wait_steps)
        wait_count += 1
    unresolved_mcs_total += len(mcs_pending_by_ev)

    avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
    stats = {
        "steps": float(executed_steps),
        "requests": float(total_requests),
        "mcs_requests": float(mcs_requests),
        "success_rate": float(success_requests / max(1, total_requests)),
        "mcs_success_rate": float(mcs_served / max(1, mcs_requests)),
        "avg_wait_steps": float(avg_wait_steps),
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "timeout_events_total": float(timeout_events_total),
        "unresolved_mcs_total": float(unresolved_mcs_total),
        "avg_total_agent_reward_per_ep": float(total_agent_reward),
        "mcs_total_income": float(total_mcs_income),
        "mcs_avg_income": float(total_mcs_income / max(1.0, float(len(env.mcs_list)))),
        "action_reinforce": float(action_counts[1]),
        "action_relocate": float(action_counts[2]),
        "action_service": float(action_counts[3]),
        "action_stay": float(action_counts[4]),
    }
    stats["business_score"] = _business_score(stats, args)
    return stats


def _aggregate(rows: List[Dict[str, float]], args: argparse.Namespace) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    out: Dict[str, float] = {"episodes": float(len(rows))}
    for key in keys:
        vals = np.asarray([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    out["business_score"] = _business_score(out, args)
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env_cfg = dict(CONFIG)
    env_cfg["use_lstm_summary"] = False
    env_cfg["lstm_predictor_ckpt"] = ""
    if str(args.policy) == "greedy_2opt":
        env_cfg["mcs_num"] = max(1, int(CONFIG.get("mcs_num", 1)) // 2)
        env_cfg["mcs_service_parallel_capacity"] = 2

    rows: List[Dict[str, float]] = []
    env = Environment(config=env_cfg, seed=int(args.seed))
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed + (ep + 1) * 9973)
        stats = run_episode(env=env, seed=ep_seed, args=args)
        stats["episode"] = float(ep + 1)
        stats["seed"] = float(ep_seed)
        rows.append(stats)
        if bool(args.log_episodes):
            print(
                f"episode={ep + 1:03d} success={stats['success_rate']:.4f} "
                f"mcs_success={stats['mcs_success_rate']:.4f} wait={stats['avg_wait_minutes']:.2f}min "
                f"reward={stats['avg_total_agent_reward_per_ep']:.2f} biz={stats['business_score']:.2f}",
                flush=True,
            )

    summary = _aggregate(rows, args)
    summary.update(
        {
            "policy": "greedy_2opt_one_to_many" if str(args.policy) == "greedy_2opt" else str(args.policy),
            "seed": float(args.seed),
            "use_lstm_summary": 0.0,
            "relocate_interval": float(args.relocate_interval),
        }
    )

    csv_path = outdir / f"{args.policy}_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = outdir / f"{args.policy}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved episodes: {csv_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
