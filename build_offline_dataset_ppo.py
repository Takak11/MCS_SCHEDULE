from __future__ import annotations

"""
Build offline trajectory dataset from PPO policies with broad coverage.

Supports:
1) Multi-checkpoint collection (stage binning by training progress, e.g. early/middle/best)
2) Multi-seed rollout collection
3) Fixed quota per source (checkpoint x seed)
4) Stratified mixing by (stage, return-bin)
5) Event-level, single-MCS, windowed dataset construction with balanced sampling
6) Stochastic rollout by default (deterministic only if --deterministic)

Examples:
  python build_offline_dataset_ppo.py --ppo-ckpt result/ppo/best.pt --episodes 2000 --output dataset/offline_ppo_traj.npz
  python build_offline_dataset_ppo.py --ppo-ckpts "result/ppo_seed1/epoch_*.pt,result/ppo_seed2/epoch_*.pt" --env-seeds "42,43,44" --episodes 3000 --stratified-mix
"""

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

from config import CONFIG
from env import Environment
from train_ppo import PolicyValueNet
from train_mappo import ActorNet


ACTION_DIM = 4


class ActorLike(Protocol):
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        ...


class MappoActorAdapter:
    def __init__(self, actor: ActorNet) -> None:
        self.actor = actor

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t).masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), 0.0


@dataclass
class AgentTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    action_indices: np.ndarray
    returns_to_go: np.ndarray
    steps: np.ndarray
    action_masks: np.ndarray
    length: int


@dataclass
class EpisodeStats:
    total_requests: int = 0
    mcs_requests: int = 0
    success_requests: int = 0
    mcs_served: int = 0
    unresolved_mcs: int = 0
    wait_steps_sum: float = 0.0
    wait_count: int = 0


@dataclass
class SourcePlan:
    source_id: int
    ckpt_path: Path
    env_seed: int
    stage_id: int
    stage_name: str
    episodes: int


@dataclass
class TrajectoryRecord:
    traj: AgentTrajectory
    source_id: int
    stage_id: int
    return0: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline trajectories with PPO policies.")
    p.add_argument("--ppo-ckpt", type=str, default="", help="Single PPO checkpoint path (fallback when --ppo-ckpts is empty).")
    p.add_argument("--ppo-ckpts", type=str, default="", help="Comma-separated checkpoint paths or glob patterns.")
    p.add_argument("--output", type=str, default="dataset/offline_ppo_traj.npz")
    p.add_argument("--episodes", type=int, default=2000, help="Total episodes when --per-source-episodes<=0.")
    p.add_argument("--per-source-episodes", type=int, default=0, help=">0 means fixed episodes for each (ckpt, seed) source.")
    p.add_argument("--env-seeds", type=str, default="", help='Comma-separated rollout seeds (e.g., "42,43,44"). Default uses --seed.')
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-lstm-summary", action="store_true")
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")
    p.add_argument("--deterministic", action="store_true", help="Use argmax action. Default is stochastic sampling.")
    p.add_argument("--stratified-mix", action="store_true", help="Mix trajectories by (stage, return-bin) in round-robin.")
    p.add_argument("--return-bins", type=int, default=3, help="Return bins used by --stratified-mix.")
    p.add_argument(
        "--stage-bins",
        type=int,
        default=3,
        help="Stage bins by training progress (epoch/total_epochs). When ==3: early/middle/best.",
    )
    p.add_argument("--max-trajs", type=int, default=12000, help="Final sample cap. In event-level mode this means max event windows.")
    p.add_argument("--event-level", action=argparse.BooleanOptionalAction, default=True, help="Build event-level samples (single MCS) instead of full trajectories.")
    p.add_argument("--window-len", type=int, default=20, help="Context window length used in event-level mode.")
    p.add_argument(
        "--balance-by",
        type=str,
        default="stage_action",
        choices=["none", "action", "stage", "stage_action"],
        help="Balanced sampling key in event-level mode.",
    )
    p.add_argument("--log-interval", type=int, default=0)
    return p.parse_args()


def _split_csv_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    for tok in _split_csv_tokens(text):
        vals.append(int(tok))
    return vals


def _resolve_ckpt_paths(args: argparse.Namespace) -> List[Path]:
    tokens = _split_csv_tokens(args.ppo_ckpts)
    if not tokens and args.ppo_ckpt:
        tokens = [str(args.ppo_ckpt)]
    if not tokens:
        raise RuntimeError("Please provide --ppo-ckpt or --ppo-ckpts.")

    paths: List[Path] = []
    for tok in tokens:
        if any(ch in tok for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(tok))
            for m in matches:
                paths.append(Path(m))
        else:
            paths.append(Path(tok))

    uniq: List[Path] = []
    seen = set()
    for p in paths:
        key = str(Path(p))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(Path(p))

    if not uniq:
        raise RuntimeError("No checkpoint matched the provided --ppo-ckpts pattern(s).")
    for p in uniq:
        if not p.exists():
            raise RuntimeError(f"PPO checkpoint not found: {p}")
    return uniq


def _resolve_env_seeds(args: argparse.Namespace) -> List[int]:
    seeds = _parse_int_list(args.env_seeds)
    if not seeds:
        seeds = [int(args.seed)]
    return seeds


def _allocate_even(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    total = max(0, int(total))
    base = total // parts
    rem = total % parts
    out = [base] * parts
    for i in range(rem):
        out[i] += 1
    return out


def _parse_epoch_from_name(path: Path) -> int:
    m = re.search(r"epoch[_-]?(\d+)", str(path.stem), flags=re.IGNORECASE)
    if m is None:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _load_ckpt_epoch_and_total(path: Path) -> Tuple[int, int]:
    epoch = -1
    total_epochs = -1
    try:
        payload = torch.load(path, map_location="cpu")
        epoch = int(payload.get("epoch", -1))
        args_obj = payload.get("args", {})
        if isinstance(args_obj, dict):
            total_epochs = int(args_obj.get("epochs", -1))
    except Exception:
        pass
    if epoch < 0:
        epoch = _parse_epoch_from_name(path)
    return int(epoch), int(total_epochs)


def _assign_stage_ids(ckpt_paths: List[Path], stage_bins: int) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    n = len(ckpt_paths)
    if n <= 0:
        return {}, {0: "all"}, {}

    stage_bins = max(1, int(stage_bins))
    epoch_total = {str(p): _load_ckpt_epoch_and_total(p) for p in ckpt_paths}
    epochs = {k: int(v[0]) for k, v in epoch_total.items()}
    total_epochs = {k: int(v[1]) for k, v in epoch_total.items()}
    stage_by_ckpt: Dict[str, int] = {}

    valid_epochs = [int(v) for v in epochs.values() if int(v) >= 0]
    fallback_total = max(valid_epochs) if len(valid_epochs) > 0 else -1

    for p in ckpt_paths:
        key = str(p)
        ep = int(epochs.get(key, -1))
        denom = int(total_epochs.get(key, -1))
        if denom <= 0:
            denom = int(fallback_total)
        if ep >= 0 and denom > 0:
            # Stage assignment strictly follows training progress.
            progress = float(np.clip(float(ep) / float(max(1, denom)), 0.0, 1.0))
            stage = min(stage_bins - 1, int(progress * stage_bins))
        else:
            stage = 0
        stage_by_ckpt[key] = int(stage)

    if stage_bins == 3:
        stage_name = {0: "early", 1: "middle", 2: "best"}
    elif stage_bins == 2:
        stage_name = {0: "early", 1: "best"}
    elif stage_bins == 1:
        stage_name = {0: "best"}
    else:
        stage_name = {i: f"stage_{i}" for i in range(stage_bins)}
    return stage_by_ckpt, stage_name, epochs


def _build_source_plans(
    ckpt_paths: List[Path],
    env_seeds: List[int],
    episodes: int,
    per_source_episodes: int,
    stage_bins: int,
) -> Tuple[List[SourcePlan], int, Dict[str, int]]:
    stage_by_ckpt, stage_name, epochs = _assign_stage_ids(ckpt_paths=ckpt_paths, stage_bins=stage_bins)
    raw_sources: List[Tuple[Path, int]] = []
    for ckpt in ckpt_paths:
        for s in env_seeds:
            raw_sources.append((ckpt, int(s)))

    if per_source_episodes > 0:
        ep_alloc = [int(per_source_episodes)] * len(raw_sources)
    else:
        ep_alloc = _allocate_even(total=int(episodes), parts=len(raw_sources))

    plans: List[SourcePlan] = []
    for i, ((ckpt, s), ep) in enumerate(zip(raw_sources, ep_alloc)):
        sid = int(stage_by_ckpt.get(str(ckpt), 0))
        plans.append(
            SourcePlan(
                source_id=i,
                ckpt_path=ckpt,
                env_seed=int(s),
                stage_id=sid,
                stage_name=stage_name.get(sid, f"stage_{sid}"),
                episodes=int(ep),
            )
        )
    total_eps = int(sum(ep_alloc))
    return plans, total_eps, epochs


def _compute_rtg_1d(rewards_1d: np.ndarray) -> np.ndarray:
    rtg = np.zeros((len(rewards_1d), 1), dtype=np.float32)
    running = 0.0
    for t in range(len(rewards_1d) - 1, -1, -1):
        running += float(rewards_1d[t])
        rtg[t, 0] = running
    return rtg


def _load_policy_actor(ckpt_path: Path, device: torch.device) -> ActorLike:
    payload = torch.load(ckpt_path, map_location="cpu")
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload["action_dim"])
    args_obj = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}

    if "model_state_dict" in payload:
        hidden_dim = int(args_obj.get("hidden_dim", 128))
        model = PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    if "actor_state_dict" in payload:
        hidden_dim = int(args_obj.get("actor_hidden_dim", 128))
        actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        actor.load_state_dict(payload["actor_state_dict"])
        actor.eval()
        return MappoActorAdapter(actor=actor)

    raise RuntimeError(
        f"Unsupported checkpoint format: {ckpt_path}. "
        "Expected PPO `model_state_dict` or MAPPO `actor_state_dict`."
    )


def _collect_one_episode(
    env: Environment,
    model: ActorLike,
    device: torch.device,
    deterministic: bool,
    max_steps: Optional[int],
) -> Tuple[List[AgentTrajectory], EpisodeStats]:
    obs_dict = env.reset(seed=int(env.rng.integers(1_000_000_000)))
    agents = env.agents
    horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)

    per_agent = {a: {"obs": [], "act_idx": [], "rew": [], "mask": []} for a in agents}
    stats = EpisodeStats()
    mcs_pending_step_by_ev: Dict[int, int] = {}
    last_step = -1

    for _ in range(horizon):
        mask_dict = env.get_action_mask()
        env_actions: Dict[str, int] = {}
        for a in agents:
            obs = np.asarray(obs_dict[a], dtype=np.float32)
            am = np.asarray(mask_dict[a], dtype=np.bool_)
            act_idx, _, _ = model.act(obs=obs, action_mask=am, device=device, deterministic=deterministic)
            env_actions[a] = int(act_idx + 1)

        step_result = env.step_parallel(env_actions)
        last_step = int(step_result.step)

        for req in step_result.requests:
            stats.total_requests += 1
            if req.get("service_mode") == "fcs":
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
            stats.wait_steps_sum += float(wait_steps)
            stats.wait_count += 1

        for timeout_event in step_result.timeout_events:
            ev_id = int(timeout_event.get("ev_id", -1))
            req_step = int(timeout_event.get("request_step", step_result.step))
            wait_steps = int(timeout_event.get("wait_steps", max(0, int(step_result.step) - req_step)))
            if ev_id in mcs_pending_step_by_ev:
                mcs_pending_step_by_ev.pop(ev_id, None)
            stats.wait_steps_sum += float(wait_steps)
            stats.wait_count += 1

        for a in agents:
            per_agent[a]["obs"].append(np.asarray(obs_dict[a], dtype=np.float32))
            per_agent[a]["act_idx"].append(int(step_result.mcs_decisions[a]) - 1)
            per_agent[a]["rew"].append(float(step_result.agent_rewards[a]))
            per_agent[a]["mask"].append(np.asarray(mask_dict[a], dtype=np.int8))

        if step_result.done:
            break
        obs_dict = env.get_agent_observations()

    stats.unresolved_mcs = len(mcs_pending_step_by_ev)
    for req_step in mcs_pending_step_by_ev.values():
        wait_steps = max(0, int(last_step) - int(req_step))
        stats.wait_steps_sum += float(wait_steps)
        stats.wait_count += 1

    trajectories: List[AgentTrajectory] = []
    for a in agents:
        obs = np.asarray(per_agent[a]["obs"], dtype=np.float32)
        act_idx = np.asarray(per_agent[a]["act_idx"], dtype=np.int64)
        rew = np.asarray(per_agent[a]["rew"], dtype=np.float32)
        mask = np.asarray(per_agent[a]["mask"], dtype=np.int8)
        k = int(obs.shape[0])
        onehot = np.eye(ACTION_DIM, dtype=np.float32)[act_idx]
        rtg = _compute_rtg_1d(rew)
        steps = np.arange(k, dtype=np.int32).reshape(-1, 1)
        trajectories.append(
            AgentTrajectory(
                observations=obs,
                actions=onehot,
                action_indices=act_idx,
                returns_to_go=rtg,
                steps=steps,
                action_masks=mask,
                length=k,
            )
        )
    return trajectories, stats


def _ret_bins_by_rank(values: np.ndarray, bins: int) -> np.ndarray:
    n = int(values.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    bins = max(1, int(bins))
    order = np.argsort(values)
    out = np.zeros((n,), dtype=np.int32)
    for b in range(bins):
        lo = int(np.floor(b * n / bins))
        hi = int(np.floor((b + 1) * n / bins))
        if hi <= lo:
            continue
        out[order[lo:hi]] = int(b)
    return out


def _stratified_mix_indices(records: List[TrajectoryRecord], return_bins: int, rng: np.random.Generator) -> List[int]:
    n = len(records)
    if n <= 1:
        return list(range(n))

    stage_ids = np.asarray([int(r.stage_id) for r in records], dtype=np.int32)
    returns = np.asarray([float(r.return0) for r in records], dtype=np.float32)
    ret_bins = _ret_bins_by_rank(values=returns, bins=max(1, int(return_bins)))

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        key = (int(stage_ids[i]), int(ret_bins[i]))
        buckets.setdefault(key, []).append(i)
    for idxs in buckets.values():
        rng.shuffle(idxs)

    keys = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))
    mixed: List[int] = []
    while True:
        moved = False
        for k in keys:
            arr = buckets[k]
            if not arr:
                continue
            mixed.append(arr.pop())
            moved = True
        if not moved:
            break
    return mixed


def _event_bucket_key(stage_id: int, action_idx: int, balance_by: str) -> Tuple[int, ...]:
    mode = str(balance_by).lower()
    if mode == "none":
        return (0,)
    if mode == "action":
        return (int(action_idx),)
    if mode == "stage":
        return (int(stage_id),)
    return (int(stage_id), int(action_idx))


def _sample_event_refs_balanced(
    records: List[TrajectoryRecord],
    max_samples: int,
    balance_by: str,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if int(max_samples) <= 0:
        raise RuntimeError("Event-level mode requires --max-trajs > 0 to cap sample count.")

    mode = str(balance_by).lower()
    stage_vals = sorted({int(r.stage_id) for r in records}) if records else [0]
    if mode == "none":
        expected_keys: List[Tuple[int, ...]] = [(0,)]
    elif mode == "action":
        expected_keys = [(a,) for a in range(ACTION_DIM)]
    elif mode == "stage":
        expected_keys = [(s,) for s in stage_vals]
    else:
        expected_keys = [(s, a) for s in stage_vals for a in range(ACTION_DIM)]

    bucket_cap = max(1, int(np.ceil(float(max_samples) / max(1, len(expected_keys)))))
    reservoirs: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {k: [] for k in expected_keys}
    seen: Dict[Tuple[int, ...], int] = {k: 0 for k in expected_keys}

    for ridx, rec in enumerate(records):
        acts = rec.traj.action_indices
        for t in range(int(len(acts))):
            a = int(acts[t])
            key = _event_bucket_key(stage_id=int(rec.stage_id), action_idx=a, balance_by=mode)
            if key not in reservoirs:
                reservoirs[key] = []
                seen[key] = 0
            seen[key] += 1
            arr = reservoirs[key]
            if len(arr) < bucket_cap:
                arr.append((int(ridx), int(t)))
            else:
                j = int(rng.integers(0, seen[key]))
                if j < bucket_cap:
                    arr[j] = (int(ridx), int(t))

    for arr in reservoirs.values():
        rng.shuffle(arr)

    keys = sorted(reservoirs.keys())
    out: List[Tuple[int, int]] = []
    while len(out) < int(max_samples):
        moved = False
        for k in keys:
            arr = reservoirs[k]
            if not arr:
                continue
            out.append(arr.pop())
            moved = True
            if len(out) >= int(max_samples):
                break
        if not moved:
            break
    return out


def _slice_event_window(tr: AgentTrajectory, end_t: int, window_len: int) -> AgentTrajectory:
    e = int(end_t) + 1
    s = max(0, e - int(max(1, window_len)))
    obs = np.asarray(tr.observations[s:e], dtype=np.float32)
    acts = np.asarray(tr.actions[s:e], dtype=np.float32)
    act_idx = np.asarray(tr.action_indices[s:e], dtype=np.int64)
    rtg = np.asarray(tr.returns_to_go[s:e], dtype=np.float32).reshape(-1, 1)
    steps = np.asarray(tr.steps[s:e], dtype=np.int32).reshape(-1, 1)
    masks = np.asarray(tr.action_masks[s:e], dtype=np.int8)
    return AgentTrajectory(
        observations=obs,
        actions=acts,
        action_indices=act_idx,
        returns_to_go=rtg,
        steps=steps,
        action_masks=masks,
        length=int(obs.shape[0]),
    )


def _build_event_level_records(
    records: List[TrajectoryRecord],
    event_refs: List[Tuple[int, int]],
    window_len: int,
) -> List[TrajectoryRecord]:
    out: List[TrajectoryRecord] = []
    for ridx, end_t in event_refs:
        base = records[int(ridx)]
        win = _slice_event_window(tr=base.traj, end_t=int(end_t), window_len=int(window_len))
        ret0 = float(win.returns_to_go[0, 0]) if win.returns_to_go.size > 0 else 0.0
        out.append(
            TrajectoryRecord(
                traj=win,
                source_id=int(base.source_id),
                stage_id=int(base.stage_id),
                return0=ret0,
            )
        )
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt_paths = _resolve_ckpt_paths(args)
    env_seeds = _resolve_env_seeds(args)
    plans, total_episodes, epochs = _build_source_plans(
        ckpt_paths=ckpt_paths,
        env_seeds=env_seeds,
        episodes=int(args.episodes),
        per_source_episodes=int(args.per_source_episodes),
        stage_bins=int(args.stage_bins),
    )
    if total_episodes <= 0:
        raise RuntimeError("No rollout episodes allocated. Check --episodes or --per-source-episodes.")

    cfg = dict(CONFIG)
    cfg["use_lstm_summary"] = bool(args.use_lstm_summary)
    if args.lstm_predictor_ckpt:
        cfg["lstm_predictor_ckpt"] = str(args.lstm_predictor_ckpt)

    log_interval = int(args.log_interval) if int(args.log_interval) > 0 else max(1, total_episodes // 20)
    deterministic = bool(args.deterministic)
    step_minutes = float(cfg.get("sim_step_minutes", 5))
    rng = np.random.default_rng(args.seed + 7)

    model_cache: Dict[str, ActorLike] = {}
    source_name_by_id: Dict[int, str] = {}
    stage_name_by_id: Dict[int, str] = {int(p.stage_id): str(p.stage_name) for p in plans}
    records: List[TrajectoryRecord] = []

    total_requests = 0
    total_mcs_requests = 0
    total_success = 0
    total_mcs_served = 0
    total_unresolved = 0
    wait_steps_sum = 0.0
    wait_count = 0
    episodes_done = 0

    print(
        f"[build_ppo] start total_episodes={total_episodes} deterministic={deterministic} "
        f"stratified_mix={bool(args.stratified_mix)} return_bins={int(args.return_bins)} "
        f"ckpts={len(ckpt_paths)} seeds={len(env_seeds)} use_lstm_summary={cfg['use_lstm_summary']}"
    )
    if deterministic:
        print("[build_ppo] warning: deterministic=True may reduce behavior coverage.")

    for plan in plans:
        if plan.episodes <= 0:
            continue
        ckpt_key = str(plan.ckpt_path)
        if ckpt_key not in model_cache:
            model_cache[ckpt_key] = _load_policy_actor(plan.ckpt_path, device=device)
        model = model_cache[ckpt_key]

        source_name_by_id[plan.source_id] = (
            f"{plan.stage_name}|seed={plan.env_seed}|epoch={epochs.get(ckpt_key, -1)}|{plan.ckpt_path}"
        )
        env = Environment(config=cfg, seed=int(plan.env_seed))
        print(
            f"[build_ppo] source#{plan.source_id:03d} stage={plan.stage_name} seed={plan.env_seed} "
            f"episodes={plan.episodes} ckpt={plan.ckpt_path}"
        )

        for _ in range(int(plan.episodes)):
            trajs, st = _collect_one_episode(
                env=env,
                model=model,
                device=device,
                deterministic=deterministic,
                max_steps=args.max_steps,
            )
            for tr in trajs:
                ret0 = float(tr.returns_to_go[0, 0]) if tr.returns_to_go.size > 0 else 0.0
                records.append(
                    TrajectoryRecord(
                        traj=tr,
                        source_id=int(plan.source_id),
                        stage_id=int(plan.stage_id),
                        return0=float(ret0),
                    )
                )

            total_requests += int(st.total_requests)
            total_mcs_requests += int(st.mcs_requests)
            total_success += int(st.success_requests)
            total_mcs_served += int(st.mcs_served)
            total_unresolved += int(st.unresolved_mcs)
            wait_steps_sum += float(st.wait_steps_sum)
            wait_count += int(st.wait_count)
            episodes_done += 1

            if episodes_done == 1 or episodes_done % log_interval == 0 or episodes_done == total_episodes:
                success_rate = float(total_success / max(1, total_requests))
                mcs_success = float(total_mcs_served / max(1, total_mcs_requests))
                avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
                print(
                    f"[build_ppo] progress {episodes_done}/{total_episodes} traj={len(records)} "
                    f"success={success_rate:.3f} mcs_success={mcs_success:.3f} "
                    f"wait={avg_wait_steps * step_minutes:.2f}min unresolved={total_unresolved}"
                )

    if not records:
        raise RuntimeError("No trajectories collected.")

    if bool(args.stratified_mix):
        order = _stratified_mix_indices(records=records, return_bins=int(args.return_bins), rng=rng)
    else:
        order = list(range(len(records)))
        rng.shuffle(order)

    ordered_records = [records[i] for i in order]
    if bool(args.event_level):
        event_refs = _sample_event_refs_balanced(
            records=ordered_records,
            max_samples=int(args.max_trajs),
            balance_by=str(args.balance_by),
            rng=rng,
        )
        picked = _build_event_level_records(
            records=ordered_records,
            event_refs=event_refs,
            window_len=int(args.window_len),
        )
        sample_kind = "event_windows"
    else:
        if int(args.max_trajs) > 0:
            order = order[: int(args.max_trajs)]
        picked = [records[i] for i in order]
        sample_kind = "trajectories"

    if not picked:
        raise RuntimeError("No samples selected after mixing/sampling.")

    source_ids = np.asarray([r.source_id for r in picked], dtype=np.int32)
    stage_ids = np.asarray([r.stage_id for r in picked], dtype=np.int32)
    return0s = np.asarray([r.return0 for r in picked], dtype=np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.array([x.traj.observations for x in picked], dtype=object),
        actions=np.array([x.traj.actions for x in picked], dtype=object),
        action_indices=np.array([x.traj.action_indices for x in picked], dtype=object),
        returns_to_go=np.array([x.traj.returns_to_go for x in picked], dtype=object),
        steps=np.array([x.traj.steps for x in picked], dtype=object),
        action_masks=np.array([x.traj.action_masks for x in picked], dtype=object),
        lengths=np.array([x.traj.length for x in picked], dtype=np.int32),
        strategy_ids=source_ids,
        strategy_names=np.array([source_name_by_id[k] for k in sorted(source_name_by_id.keys())], dtype=object),
        source_ids=source_ids,
        stage_ids=stage_ids,
        trajectory_return0=return0s,
        ppo_ckpts=np.array([str(p) for p in ckpt_paths], dtype=object),
        env_seeds=np.array(env_seeds, dtype=np.int32),
        event_level=np.array([int(bool(args.event_level))], dtype=np.int8),
        window_len=np.array([int(args.window_len)], dtype=np.int32),
        balance_by=np.array([str(args.balance_by)], dtype=object),
    )

    success_rate = float(total_success / max(1, total_requests))
    mcs_success = float(total_mcs_served / max(1, total_mcs_requests))
    avg_wait_steps = float(wait_steps_sum / wait_count) if wait_count > 0 else 0.0
    stage_vals, stage_cnt = np.unique(stage_ids, return_counts=True)
    stage_msg = (
        ", ".join(
            [
                f"{stage_name_by_id.get(int(s), f'stage{int(s)}')}:{int(c)}"
                for s, c in zip(stage_vals.tolist(), stage_cnt.tolist())
            ]
        )
        if stage_vals.size > 0
        else "-"
    )
    print(f"Saved: {output}")
    print(
        f"[metrics] episodes={total_episodes} samples={len(picked)} kind={sample_kind} req={total_requests} "
        f"success={success_rate:.3f} mcs_success={mcs_success:.3f} "
        f"wait={avg_wait_steps:.2f}step/{avg_wait_steps * step_minutes:.2f}min unresolved={total_unresolved} "
        f"stage_mix={stage_msg}"
    )


if __name__ == "__main__":
    main()
