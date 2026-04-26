from __future__ import annotations

"""
Example:
  python build_offline_dataset.py --episodes 2000 --mix 0.5,0.35,0.15 --output dataset/offline_mcs_odt_traj.npz

Saved (trajectory-level global buffer, loaded with allow_pickle=True):
  observations[i]:   [K, obs_dim]
  actions[i]:        [K, action_dim] (one-hot)
  action_indices[i]: [K] in 0..action_dim-1
  returns_to_go[i]:  [K, 1]
  steps[i]:          [K, 1]
  action_masks[i]:   [K, action_dim]
  lengths[i]:        scalar K
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from env import Environment
from utils import haversine_distance


STRATEGY_NAMES = ("expert", "mid", "random")
ACTION_DIM = 4


@dataclass
class AgentTrajectory:
    strategy_id: int
    observations: np.ndarray    # [K, obs_dim]
    actions: np.ndarray         # [K, action_dim] one-hot
    action_indices: np.ndarray  # [K] in 0..action_dim-1
    returns_to_go: np.ndarray   # [K, 1]
    steps: np.ndarray           # [K, 1]
    action_masks: np.ndarray    # [K, action_dim]
    length: int


@dataclass
class EpisodeStats:
    total_requests: int = 0
    fcs_requests: int = 0
    mcs_requests: int = 0
    mcs_served: int = 0
    unresolved_mcs: int = 0
    success_requests: int = 0
    served_wait_steps_sum: float = 0.0
    served_wait_count: int = 0


@dataclass
class StatsAccumulator:
    episodes: int = 0
    total_requests: int = 0
    fcs_requests: int = 0
    mcs_requests: int = 0
    mcs_served: int = 0
    unresolved_mcs: int = 0
    success_requests: int = 0
    served_wait_steps_sum: float = 0.0
    served_wait_count: int = 0

    def add(self, ep: EpisodeStats) -> None:
        self.episodes += 1
        self.total_requests += int(ep.total_requests)
        self.fcs_requests += int(ep.fcs_requests)
        self.mcs_requests += int(ep.mcs_requests)
        self.mcs_served += int(ep.mcs_served)
        self.unresolved_mcs += int(ep.unresolved_mcs)
        self.success_requests += int(ep.success_requests)
        self.served_wait_steps_sum += float(ep.served_wait_steps_sum)
        self.served_wait_count += int(ep.served_wait_count)

    def summary(self, step_minutes: float) -> Dict[str, float]:
        req_total = max(1, self.total_requests)
        mcs_total = max(1, self.mcs_requests)
        wait_count = max(1, self.served_wait_count)
        avg_wait_steps = self.served_wait_steps_sum / wait_count if self.served_wait_count > 0 else 0.0
        return {
            "episodes": float(self.episodes),
            "total_requests": float(self.total_requests),
            "fcs_requests": float(self.fcs_requests),
            "mcs_requests": float(self.mcs_requests),
            "mcs_served": float(self.mcs_served),
            "unresolved_mcs": float(self.unresolved_mcs),
            "success_rate": float(self.success_requests / req_total),
            "mcs_success_rate": float(self.mcs_served / mcs_total),
            "avg_wait_steps": float(avg_wait_steps),
            "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline MARL dataset as per-agent trajectories.")
    p.add_argument("--episodes", type=int, default=2000, help="Total episodes to generate.")
    p.add_argument("--mix", type=str, default="0.4,0.4,0.2", help="expert,mid,random ratios.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="dataset/offline_mcs_odt_traj.npz")
    p.add_argument("--max-steps", type=int, default=None, help="Override episode horizon.")
    p.add_argument("--random-weights", type=str, default="0.2,0.2,0.3,0.3", help="Action probs for random policy (1..4).")
    p.add_argument(
        "--mid-random-mix",
        type=float,
        default=0.7,
        help="For mid strategy: probability to use random action instead of greedy action (0..1).",
    )
    p.add_argument("--log-interval", type=int, default=0, help="Progress print interval in episodes (0 means auto).")
    return p.parse_args()


def parse_ratio(text: str, n: int) -> np.ndarray:
    vals = np.array([float(x.strip()) for x in text.split(",")], dtype=float)
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}")
    if np.any(vals < 0):
        raise ValueError("Ratios must be non-negative.")
    s = float(vals.sum())
    if s <= 0:
        raise ValueError("Ratio sum must be > 0.")
    return vals / s


def split_counts(total: int, ratios: np.ndarray) -> np.ndarray:
    raw = ratios * total
    base = np.floor(raw).astype(int)
    remain = total - int(base.sum())
    if remain > 0:
        frac_idx = np.argsort(-(raw - base))
        base[frac_idx[:remain]] += 1
    return base


def nearest_pending_distance(env: Environment, mcs_id: int) -> float:
    mcs = env.mcs_by_id[mcs_id]
    if mcs.location is None:
        return float("inf")
    d = float("inf")
    for req in env.pending_ev_requests:
        loc = req.get("location")
        if loc is None:
            continue
        d = min(d, haversine_distance(mcs.location, loc))
    return d


def has_serviceable_request(env: Environment, mcs_id: int) -> bool:
    mcs = env.mcs_by_id[mcs_id]
    if mcs.location is None:
        return False
    for req in env.pending_ev_requests:
        loc = req.get("location")
        if loc is None:
            continue
        if haversine_distance(mcs.location, loc) <= mcs.service_radius_km:
            return True
    return False


def current_fcs_deficit(env: Environment) -> float:
    arrivals = env.fcs_arrival_schedule.get(env.current_step, {})
    return max((arrivals.get(f.fcs_id, 0) - f.capacity for f in env.fcs_list), default=0.0)


def future_fcs_pressure(env: Environment, horizon: int = 12) -> float:
    score = 0.0
    end = min(env.total_steps, env.current_step + 1 + horizon)
    for t in range(env.current_step + 1, end):
        arr = env.fcs_arrival_schedule.get(t, {})
        if not arr:
            continue
        score += max((arr.get(f.fcs_id, 0) - f.capacity for f in env.fcs_list), default=0.0)
    return score


def expert_policy(env: Environment, mask: Dict[str, np.ndarray]) -> Dict[str, int]:
    decisions: Dict[str, int] = {}
    cur_deficit = current_fcs_deficit(env)
    fut_pressure = future_fcs_pressure(env, horizon=int(env.config.get("mcs_relocate_horizon_steps", 12)))

    for agent in env.agents:
        valid = mask[agent]
        mcs_id = env.agent_to_mcs_id[agent]

        if valid[2] and has_serviceable_request(env, mcs_id):
            decisions[agent] = 3
        elif valid[0] and cur_deficit > 0:
            decisions[agent] = 1
        elif valid[1] and fut_pressure > 0:
            decisions[agent] = 2
        else:
            decisions[agent] = 4
    return decisions


def greedy_policy(env: Environment, mask: Dict[str, np.ndarray]) -> Dict[str, int]:
    decisions: Dict[str, int] = {}
    cur_deficit = current_fcs_deficit(env)
    fut_pressure = future_fcs_pressure(env, horizon=int(env.config.get("mcs_relocate_horizon_steps", 12)))

    for agent in env.agents:
        valid = mask[agent]
        mcs_id = env.agent_to_mcs_id[agent]
        d_req = nearest_pending_distance(env, mcs_id)

        score1 = 1.0 * cur_deficit
        score2 = 0.8 * fut_pressure
        score3 = (2.0 if np.isfinite(d_req) else -1e6) - 0.25 * (d_req if np.isfinite(d_req) else 1000.0)
        score4 = 0.0
        scores = np.array([score1, score2, score3, score4], dtype=float)

        for i in range(ACTION_DIM):
            if not valid[i]:
                scores[i] = -1e9
        decisions[agent] = int(np.argmax(scores)) + 1
    return decisions


def mid_policy(
    env: Environment,
    mask: Dict[str, np.ndarray],
    random_weights: np.ndarray,
    random_mix: float,
) -> Dict[str, int]:
    mix = float(np.clip(random_mix, 0.0, 1.0))
    greedy_cmd = greedy_policy(env, mask)
    random_cmd = random_policy(env, mask, random_weights)

    decisions: Dict[str, int] = {}
    for agent in env.agents:
        decisions[agent] = random_cmd[agent] if float(env.rng.random()) < mix else greedy_cmd[agent]
    return decisions


def random_policy(env: Environment, mask: Dict[str, np.ndarray], weights_1_to_4: np.ndarray) -> Dict[str, int]:
    decisions: Dict[str, int] = {}
    for agent in env.agents:
        valid = mask[agent].astype(bool)
        probs = np.where(valid, weights_1_to_4, 0.0)
        if probs.sum() <= 0:
            decisions[agent] = 4
            continue
        probs = probs / probs.sum()
        decisions[agent] = int(env.rng.choice(np.array([1, 2, 3, 4]), p=probs))
    return decisions


def _compute_rtg_1d(rewards_1d: np.ndarray) -> np.ndarray:
    # rewards_1d: [K]
    rtg = np.zeros((len(rewards_1d), 1), dtype=np.float32)
    running = 0.0
    for t in range(len(rewards_1d) - 1, -1, -1):
        running += float(rewards_1d[t])
        rtg[t, 0] = running
    return rtg


def collect_episode_trajectories(
    env: Environment,
    strategy_id: int,
    random_weights: np.ndarray,
    mid_random_mix: float,
    max_steps: int | None,
) -> tuple[List[AgentTrajectory], EpisodeStats]:
    obs_dict = env.reset(seed=int(env.rng.integers(1_000_000_000)))
    agents = env.agents
    horizon = env.total_steps if max_steps is None else min(max_steps, env.total_steps)
    stats = EpisodeStats()
    mcs_pending_step_by_ev: Dict[int, int] = {}
    last_step = -1

    per_agent = {
        a: {"obs": [], "act_idx": [], "rew": [], "mask": []}
        for a in agents
    }

    for _ in range(horizon):
        mask = env.get_action_mask()
        if strategy_id == 0:
            cmd = expert_policy(env, mask)
        elif strategy_id == 1:
            cmd = mid_policy(env, mask, random_weights=random_weights, random_mix=mid_random_mix)
        else:
            cmd = random_policy(env, mask, random_weights)

        step_result = env.step_parallel(cmd)
        last_step = int(step_result.step)

        for req in step_result.requests:
            stats.total_requests += 1
            if req.get("service_mode") == "fcs":
                stats.fcs_requests += 1
                stats.success_requests += 1
            else:
                stats.mcs_requests += 1
                mcs_pending_step_by_ev[int(req["ev_id"])] = int(req.get("step", step_result.step))

        for event in step_result.mcs_events:
            if event.get("action") != "serve_request":
                continue
            ev_id = int(event["ev_id"])
            req_step = mcs_pending_step_by_ev.pop(ev_id, step_result.step)
            wait_steps = max(0, int(step_result.step) - int(req_step))
            stats.mcs_served += 1
            stats.success_requests += 1
            stats.served_wait_steps_sum += float(wait_steps)
            stats.served_wait_count += 1

        for timeout_event in step_result.timeout_events:
            ev_id = int(timeout_event.get("ev_id", -1))
            req_step = int(timeout_event.get("request_step", step_result.step))
            wait_steps = int(timeout_event.get("wait_steps", max(0, int(step_result.step) - req_step)))
            if ev_id in mcs_pending_step_by_ev:
                mcs_pending_step_by_ev.pop(ev_id, None)
            stats.served_wait_steps_sum += float(wait_steps)
            stats.served_wait_count += 1

        for a in agents:
            per_agent[a]["obs"].append(np.asarray(obs_dict[a], dtype=np.float32))
            # executed action 1..4 -> 0..3
            per_agent[a]["act_idx"].append(int(step_result.mcs_decisions[a]) - 1)
            per_agent[a]["rew"].append(float(step_result.agent_rewards[a]))
            per_agent[a]["mask"].append(np.asarray(mask[a], dtype=np.int8))

        if step_result.done:
            break
        obs_dict = env.get_agent_observations()

    trajectories: List[AgentTrajectory] = []
    for a in agents:
        obs = np.asarray(per_agent[a]["obs"], dtype=np.float32)
        act_idx = np.asarray(per_agent[a]["act_idx"], dtype=np.int64)
        rew = np.asarray(per_agent[a]["rew"], dtype=np.float32)
        amask = np.asarray(per_agent[a]["mask"], dtype=np.int8)

        k = obs.shape[0]
        actions_one_hot = np.eye(ACTION_DIM, dtype=np.float32)[act_idx]
        rtg = _compute_rtg_1d(rew)
        steps = np.arange(k, dtype=np.int32).reshape(-1, 1)

        trajectories.append(
            AgentTrajectory(
                strategy_id=strategy_id,
                observations=obs,            # [K, obs_dim]
                actions=actions_one_hot,     # [K, action_dim]
                action_indices=act_idx,      # [K]
                returns_to_go=rtg,           # [K, 1]
                steps=steps,                 # [K, 1]
                action_masks=amask,          # [K, action_dim]
                length=k,
            )
        )

    stats.unresolved_mcs = len(mcs_pending_step_by_ev)
    # Include wait time for unresolved (failed/offline) MCS requests until episode end.
    for req_step in mcs_pending_step_by_ev.values():
        wait_steps = max(0, int(last_step) - int(req_step))
        stats.served_wait_steps_sum += float(wait_steps)
        stats.served_wait_count += 1
    return trajectories, stats


def main() -> None:
    args = parse_args()
    mix = parse_ratio(args.mix, 3)
    counts = split_counts(args.episodes, mix)
    random_weights = parse_ratio(args.random_weights, ACTION_DIM)
    total_episodes = int(counts.sum())
    log_interval = int(args.log_interval) if int(args.log_interval) > 0 else max(1, total_episodes // 20)

    env = Environment(seed=args.seed)
    step_minutes = float(env.config.get("sim_step_minutes", 5))

    print(
        f"[build] start episodes={total_episodes} "
        f"mix(expert,mid,random)={counts.tolist()} "
        f"max_steps={args.max_steps if args.max_steps is not None else 'full'}"
    )
    print(f"[build] random_weights(1..4)={random_weights.tolist()}")
    print(f"[build] mid_random_mix={float(np.clip(args.mid_random_mix, 0.0, 1.0)):.2f}")

    global_buffer: List[AgentTrajectory] = []
    overall_stats = StatsAccumulator()
    strategy_stats = {sid: StatsAccumulator() for sid in range(len(STRATEGY_NAMES))}
    done_total = 0
    for sid, num in enumerate(counts):
        print(f"[build] strategy={STRATEGY_NAMES[sid]} target_episodes={int(num)}")
        for _ in range(int(num)):
            trajectories, ep_stats = collect_episode_trajectories(
                env=env,
                strategy_id=sid,
                random_weights=random_weights,
                mid_random_mix=args.mid_random_mix,
                max_steps=args.max_steps,
            )
            global_buffer.extend(trajectories)
            strategy_stats[sid].add(ep_stats)
            overall_stats.add(ep_stats)
            done_total += 1
            if done_total == 1 or done_total % log_interval == 0 or done_total == total_episodes:
                pct = 100.0 * done_total / max(1, total_episodes)
                sum_now = overall_stats.summary(step_minutes)
                print(
                    f"[build] progress {done_total}/{total_episodes} ({pct:.1f}%) "
                    f"current_strategy={STRATEGY_NAMES[sid]} buffer_traj={len(global_buffer)} "
                    f"success={sum_now['success_rate']:.3f} wait={sum_now['avg_wait_minutes']:.2f}min"
                )
        print(f"[build] strategy={STRATEGY_NAMES[sid]} done")
        s = strategy_stats[sid].summary(step_minutes)
        print(
            f"[metrics][{STRATEGY_NAMES[sid]}] "
            f"episodes={int(s['episodes'])} req={int(s['total_requests'])} "
            f"success_rate={s['success_rate']:.3f} "
            f"mcs_success_rate={s['mcs_success_rate']:.3f} "
            f"avg_wait={s['avg_wait_steps']:.2f}step/{s['avg_wait_minutes']:.2f}min"
        )

    perm = np.arange(len(global_buffer))
    rng = np.random.default_rng(args.seed + 7)
    rng.shuffle(perm)
    global_buffer = [global_buffer[i] for i in perm]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        observations=np.array([x.observations for x in global_buffer], dtype=object),
        actions=np.array([x.actions for x in global_buffer], dtype=object),
        action_indices=np.array([x.action_indices for x in global_buffer], dtype=object),
        returns_to_go=np.array([x.returns_to_go for x in global_buffer], dtype=object),
        steps=np.array([x.steps for x in global_buffer], dtype=object),
        action_masks=np.array([x.action_masks for x in global_buffer], dtype=object),
        lengths=np.array([x.length for x in global_buffer], dtype=np.int32),
        strategy_ids=np.array([x.strategy_id for x in global_buffer], dtype=np.int8),
        strategy_names=np.array(STRATEGY_NAMES),
    )

    print(f"Saved: {output}")
    print(f"Episodes: {int(counts.sum())}  Mix(expert/mid/random): {counts.tolist()}")
    print(f"Trajectories in global buffer: {len(global_buffer)}")
    final_m = overall_stats.summary(step_minutes)
    print(
        "[metrics][overall] "
        f"episodes={int(final_m['episodes'])} req={int(final_m['total_requests'])} "
        f"success_rate={final_m['success_rate']:.3f} "
        f"mcs_success_rate={final_m['mcs_success_rate']:.3f} "
        f"avg_wait={final_m['avg_wait_steps']:.2f}step/{final_m['avg_wait_minutes']:.2f}min "
        f"mcs_unresolved_total={int(final_m['unresolved_mcs'])}"
    )
    if global_buffer:
        sample = global_buffer[0]
        print(
            "Sample shapes "
            f"obs={sample.observations.shape} actions={sample.actions.shape} "
            f"rtg={sample.returns_to_go.shape} steps={sample.steps.shape} mask={sample.action_masks.shape}"
        )


if __name__ == "__main__":
    main()
