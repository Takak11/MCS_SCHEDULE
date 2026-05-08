from __future__ import annotations

"""
MN-ICRSP-style Transformer actor-critic with policy-gradient training.

This baseline aligns the comparison method with the original paper's
Transformer-based DRL spirit: a Transformer policy encodes the current set of
MCS entities jointly, outputs masked dispatch actions for each MCS, and is
trained with vanilla policy-gradient plus per-agent value baselines.
"""

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
from env import Environment, StepResult
from train_mappo import (
    _business_score,
    _build_pending_req_info,
    _event_mcs_income,
)
from train_ppo import _plot_business_metrics, save_ckpt
from utils import haversine_distance


class TransformerActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, hidden_dim: int, n_layer: int, n_head: int, dropout: float, cand_feat_dim: int = 0) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.hidden_dim = int(hidden_dim)
        self.cand_feat_dim = int(cand_feat_dim)
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.agent_embed = nn.Embedding(n_agents, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.actor_head = nn.Linear(hidden_dim, action_dim) if self.cand_feat_dim <= 0 else None
        self.cand_proj = nn.Linear(self.cand_feat_dim, hidden_dim) if self.cand_feat_dim > 0 else None
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor, cand_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: [B, N, D]
        bsz, n_agents, _ = obs_seq.shape
        ids = torch.arange(n_agents, device=obs_seq.device).unsqueeze(0).expand(bsz, n_agents)
        x = self.input_proj(obs_seq) + self.agent_embed(ids.clamp_max(self.n_agents - 1))
        h = self.encoder(x)
        if self.cand_feat_dim > 0:
            if cand_feats is None:
                raise ValueError("candidate features are required for candidate-target policy")
            cand_h = self.cand_proj(cand_feats)
            logits = torch.einsum("bnh,bnah->bna", h, cand_h) / float(np.sqrt(max(1, self.hidden_dim)))
        else:
            logits = self.actor_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs_mat: np.ndarray, masks: np.ndarray, device: torch.device, deterministic: bool = False, cand_feats: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device).unsqueeze(0)
        cand_t = None if cand_feats is None else torch.as_tensor(cand_feats, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(obs_t, cand_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            actions = logits.argmax(dim=-1) if deterministic else dist.sample()
            logps = dist.log_prob(actions)
        return actions.squeeze(0).cpu().numpy().astype(np.int64), logps.squeeze(0).cpu().numpy().astype(np.float32), float(value.mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MN-ICRSP-style Transformer policy-gradient baseline.")
    p.add_argument("--outdir", type=str, default="result/mn_icrsp_transformer_pg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-lstm-summary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lstm-predictor-ckpt", type=str, default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--episodes-per-epoch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--candidate-requests", type=int, default=8)
    p.add_argument("--candidate-hotspots", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--mini-batch-size", type=int, default=256)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--team-reward-mode", type=str, choices=["mean", "sum"], default="mean")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--eval-max-steps", type=int, default=None)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--save-epoch-interval", type=int, default=0)
    p.add_argument("--log-flush", action="store_true")
    return p.parse_args()


def _obs_matrix(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in agents], axis=0)


def _mask_matrix(mask_dict: Dict[str, np.ndarray], agents: List[str], action_dim: int) -> np.ndarray:
    return np.stack([np.asarray(mask_dict[a], dtype=np.bool_) for a in agents], axis=0).reshape(len(agents), action_dim)


def _candidate_sizes(env: Environment, args: argparse.Namespace) -> Tuple[int, int, int, int]:
    req_k = int(max(1, args.candidate_requests))
    fcs_k = int(len(env.fcs_list))
    hot_k = int(min(max(1, args.candidate_hotspots), max(1, len(env.relocate_hotspots))))
    action_dim = 1 + req_k + fcs_k + hot_k
    return req_k, fcs_k, hot_k, action_dim


def _candidate_features_and_refs(env: Environment, agents: List[str], req_k: int, fcs_k: int, hot_k: int) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[str, int]]]]:
    feat_dim = 10
    action_dim = 1 + req_k + fcs_k + hot_k
    feats = np.zeros((len(agents), action_dim, feat_dim), dtype=np.float32)
    masks = np.zeros((len(agents), action_dim), dtype=np.bool_)
    refs: List[List[Tuple[str, int]]] = []
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    timeout_steps = max(1.0, float(env.config.get("ev_request_timeout_minutes", 30.0)) / max(1e-6, step_minutes))
    radius = max(1e-6, float(env.config.get("mcs_service_radius_km", 3.0)))
    future_horizon = int(env.config.get("ppo_future_horizon_steps", 12))
    risk_by_fid, fut_arr_by_fid = env._fcs_risk_metrics(future_horizon=future_horizon)
    pred_scale = max(1.0, float(env.config.get("ppo_pred_demand_scale", 50.0)))

    for ai, agent in enumerate(agents):
        mcs = env.mcs_by_id[env.agent_to_mcs_id[agent]]
        loc = mcs.location
        row_refs: List[Tuple[str, int]] = [("stay", -1)] * action_dim
        feats[ai, 0, 0] = 1.0
        masks[ai, 0] = True
        if mcs.busy or loc is None:
            refs.append(row_refs)
            continue

        pending: List[Tuple[float, float, dict]] = []
        for req in env.pending_ev_requests:
            req_loc = req.get("location")
            if req_loc is None:
                continue
            dist = float(haversine_distance(loc, req_loc))
            if dist > radius:
                continue
            wait = float(max(0, env.current_step - int(req.get("step", env.current_step))))
            score = -dist + 0.25 * wait
            pending.append((score, dist, req))
        pending.sort(key=lambda x: x[0], reverse=True)
        for j, (_, dist, req) in enumerate(pending[:req_k]):
            idx = 1 + j
            wait = float(max(0, env.current_step - int(req.get("step", env.current_step))))
            feats[ai, idx, 1] = 1.0
            feats[ai, idx, 4] = float(np.clip(dist / radius, 0.0, 3.0))
            feats[ai, idx, 5] = float(np.clip(wait / timeout_steps, 0.0, 3.0))
            feats[ai, idx, 6] = float(req.get("required_kwh", 0.0)) / max(1.0, float(env.config.get("ev_battery_capacity_kwh", 50.0)))
            masks[ai, idx] = True
            row_refs[idx] = ("request", int(req.get("ev_id", -1)))

        base = 1 + req_k
        for j, fcs in enumerate(env.fcs_list[:fcs_k]):
            idx = base + j
            dist = float(haversine_distance(loc, fcs.lat_lon))
            q = float(env.fcs_queue.get(fcs.fcs_id, 0))
            feats[ai, idx, 2] = 1.0
            feats[ai, idx, 4] = float(np.clip(dist / max(radius, float(env.config.get("mcs_speed_km_per_step", 4.0))), 0.0, 5.0))
            feats[ai, idx, 7] = float(np.clip(risk_by_fid.get(fcs.fcs_id, 0.0), 0.0, 5.0))
            feats[ai, idx, 8] = float(np.clip(q / max(1.0, float(fcs.capacity)), 0.0, 5.0))
            masks[ai, idx] = True
            row_refs[idx] = ("fcs", int(fcs.fcs_id))

        base = 1 + req_k + fcs_k
        hotspot_scores: List[Tuple[float, int]] = []
        for hid, hloc in enumerate(env.relocate_hotspots):
            demand = 0.0
            if env._predictive_req_pred.size > hid:
                demand = float(env._predictive_req_pred[hid])
            else:
                for fid, rid in env.fcs_region_idx.items():
                    if int(rid) == int(hid):
                        demand += float(fut_arr_by_fid.get(fid, 0.0))
            dist = float(haversine_distance(loc, hloc))
            hotspot_scores.append((demand / pred_scale - 0.05 * dist, hid))
        hotspot_scores.sort(key=lambda x: x[0], reverse=True)
        for j, (_, hid) in enumerate(hotspot_scores[:hot_k]):
            idx = base + j
            hloc = env.relocate_hotspots[hid]
            dist = float(haversine_distance(loc, hloc))
            demand = float(env._predictive_req_pred[hid]) if env._predictive_req_pred.size > hid else 0.0
            feats[ai, idx, 3] = 1.0
            feats[ai, idx, 4] = float(np.clip(dist / max(radius, float(env.config.get("mcs_speed_km_per_step", 4.0))), 0.0, 5.0))
            feats[ai, idx, 9] = float(np.clip(demand / pred_scale, 0.0, 5.0))
            masks[ai, idx] = True
            row_refs[idx] = ("hotspot", int(hid))
        refs.append(row_refs)
    return feats, masks, refs


def _targeted_step(env: Environment, agents: List[str], actions: np.ndarray, refs: List[List[Tuple[str, int]]]) -> StepResult:
    if env.current_step >= env.total_steps:
        return StepResult(step=env.current_step, active_ev_count=len(env.active_ev_ids), new_request_count=0, requests=[], fcs_arrivals={}, fcs_states={}, mcs_decisions={agent: 4 for agent in agents}, mcs_events=[], timeout_events=[], agent_rewards={agent: 0.0 for agent in agents}, done=True)
    for ev_id in env.ev_ids_by_step.get(env.current_step, []):
        env._create_ev(ev_id)
    requests: List[dict] = []
    for ev_id in list(env.active_ev_ids):
        req = env._ev_step(ev_id)
        if req is not None:
            requests.append(req)
            if req.get("service_mode") == "mcs":
                env.pending_ev_requests.append(req)
    timeout_events = env._drop_timeout_pending_requests()
    env._refresh_mcs_state()
    risk_by_fid, fut_arr_by_fid = env._fcs_risk_metrics(future_horizon=int(env.config.get("ppo_future_horizon_steps", 12)))

    req_by_id = {int(r.get("ev_id", -1)): r for r in env.pending_ev_requests if r.get("location") is not None}
    used_ev_ids = set()
    mcs_events: List[dict] = []
    decisions: Dict[str, int] = {}
    for i, agent in enumerate(agents):
        mcs = env.mcs_by_id[env.agent_to_mcs_id[agent]]
        if mcs.busy or mcs.location is None:
            decisions[agent] = 4
            continue
        ref = refs[i][int(actions[i])]
        kind, target_id = ref
        if kind == "request" and int(target_id) in req_by_id and int(target_id) not in used_ev_ids:
            req = req_by_id[int(target_id)]
            dist = float(haversine_distance(mcs.location, req["location"]))
            if dist <= float(mcs.service_radius_km):
                charge_minutes = float(req.get("charge_minutes", 0.0))
                required_kwh = float(req.get("required_kwh", 0.0))
                req_step = int(req.get("step", env.current_step))
                wait_steps = max(0, int(env.current_step) - req_step)
                travel_steps = env._travel_steps(dist)
                income = required_kwh * float(mcs.price_per_kwh) - dist * float(mcs.cost_per_km)
                mcs.location = req["location"]
                service_steps = env._reserve_mcs(mcs, travel_steps)
                used_ev_ids.add(int(target_id))
                decisions[agent] = 3
                mcs_events.append({"agent": agent, "mcs_id": mcs.mcs_id, "action": "serve_request", "ev_id": int(target_id), "distance_km": round(dist, 3), "required_kwh": round(required_kwh, 4), "income": round(float(income), 4), "travel_steps": int(travel_steps), "charge_steps": int(max(1, int(np.ceil(charge_minutes / float(env.config.get("sim_step_minutes", 5)))))), "service_steps": int(service_steps), "request_step": int(req_step), "wait_steps": int(wait_steps)})
                continue
        if kind == "fcs" and int(target_id) in env.fcs_by_id:
            fcs = env.fcs_by_id[int(target_id)]
            dist = float(haversine_distance(mcs.location, fcs.lat_lon))
            mcs.location = fcs.lat_lon
            travel_steps = env._travel_steps(dist)
            reinforce_steps = max(1, int(env.config.get("mcs_reinforce_busy_steps", 1)))
            busy_steps = env._reserve_mcs(mcs, travel_steps + reinforce_steps)
            decisions[agent] = 1
            mcs_events.append({"agent": agent, "mcs_id": mcs.mcs_id, "action": "reinforce_fcs", "target_fcs": fcs.fcs_id, "distance_km": round(dist, 3), "travel_steps": int(travel_steps), "busy_steps": int(busy_steps)})
            continue
        if kind == "hotspot" and 0 <= int(target_id) < len(env.relocate_hotspots):
            target_loc = env.relocate_hotspots[int(target_id)]
            dist = float(haversine_distance(mcs.location, target_loc))
            source_region = env._infer_region_idx(mcs.location, len(env.relocate_hotspots))
            mcs.location = target_loc
            travel_steps = env._travel_steps(dist)
            busy_steps = env._reserve_mcs(mcs, travel_steps)
            decisions[agent] = 2
            mcs_events.append({"agent": agent, "mcs_id": mcs.mcs_id, "action": "relocate", "source_region": int(source_region), "target_region": int(target_id), "target_hotspot": int(target_id + 1), "target_location": (round(float(target_loc[0]), 6), round(float(target_loc[1]), 6)), "distance_km": round(dist, 3), "travel_steps": int(travel_steps), "busy_steps": int(busy_steps)})
            continue
        decisions[agent] = 4

    if used_ev_ids:
        env.pending_ev_requests = [r for r in env.pending_ev_requests if int(r.get("ev_id", -1)) not in used_ev_ids]
    fcs_states = env._update_fcs_runtime(mcs_events)
    env._update_predictive_summary(requests=requests, fcs_states=fcs_states)
    rewards = env._build_rewards(decisions=decisions, events=mcs_events, timeout_events=timeout_events, risk_by_fid=risk_by_fid, fut_arr_by_fid=fut_arr_by_fid)
    result = StepResult(step=env.current_step, active_ev_count=len(env.active_ev_ids), new_request_count=len(requests), requests=requests, fcs_arrivals=env.fcs_arrival_schedule.get(env.current_step, {}), fcs_states=fcs_states, mcs_decisions=decisions, mcs_events=mcs_events, timeout_events=timeout_events, agent_rewards=rewards, done=(env.current_step + 1 >= env.total_steps))
    env.current_step += 1
    return result


def collect_rollouts(
    env: Environment,
    model: TransformerActorCritic,
    episodes: int,
    gamma: float,
    gae_lambda: float,
    team_reward_mode: str,
    device: torch.device,
    rng: np.random.Generator,
    action_dim: int,
    max_steps: Optional[int],
    req_k: int,
    fcs_k: int,
    hot_k: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    agents = list(env.agents)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    obs_seq: List[np.ndarray] = []
    cand_feat_seq: List[np.ndarray] = []
    mask_seq: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    logp_seq: List[np.ndarray] = []
    adv_seq: List[np.ndarray] = []
    ret_seq: List[np.ndarray] = []
    target_counts = np.zeros((4,), dtype=np.float64)  # stay, request, fcs, hotspot

    stats = {k: 0.0 for k in ["steps", "requests", "mcs_requests", "success_requests", "mcs_served", "wait_steps_sum", "wait_count", "timeout_events_total", "unresolved_mcs_total", "total_agent_reward", "total_mcs_income"]}

    for _ in range(int(episodes)):
        obs_dict = env.reset(seed=int(rng.integers(1_000_000_000)))
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        ep_rewards: List[np.ndarray] = []
        ep_values: List[np.ndarray] = []
        ep_dones: List[float] = []
        last_next_obs: Optional[np.ndarray] = None
        done = False
        executed = 0
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        for _step in range(horizon):
            obs_mat = _obs_matrix(obs_dict, agents)
            cand_feats, mask_mat, refs = _candidate_features_and_refs(env, agents, req_k=req_k, fcs_k=fcs_k, hot_k=hot_k)
            actions, logps, value = model.act(obs_mat, mask_mat, device=device, deterministic=False, cand_feats=cand_feats)
            with torch.no_grad():
                _, value_vec_t = model(
                    torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(cand_feats, dtype=torch.float32, device=device).unsqueeze(0),
                )
            value_vec = value_vec_t.squeeze(0).cpu().numpy().astype(np.float32)
            sr = _targeted_step(env, agents, actions, refs)
            executed += 1
            done = bool(sr.done)
            reward_vec = np.asarray([float(sr.agent_rewards.get(a, 0.0)) for a in agents], dtype=np.float32)

            obs_seq.append(obs_mat)
            cand_feat_seq.append(cand_feats)
            mask_seq.append(mask_mat)
            action_seq.append(actions)
            logp_seq.append(logps)
            for ai, act in enumerate(actions):
                kind = refs[ai][int(act)][0]
                if kind == "request":
                    target_counts[1] += 1.0
                elif kind == "fcs":
                    target_counts[2] += 1.0
                elif kind == "hotspot":
                    target_counts[3] += 1.0
                else:
                    target_counts[0] += 1.0
            ep_values.append(value_vec)
            ep_rewards.append(reward_vec)
            ep_dones.append(1.0 if done else 0.0)

            for req in sr.requests:
                stats["requests"] += 1.0
                if req.get("service_mode") == "fcs":
                    stats["success_requests"] += 1.0
                else:
                    stats["mcs_requests"] += 1.0
                    pending_by_ev[int(req["ev_id"])] = _build_pending_req_info(req, int(sr.step))
            for event in sr.mcs_events:
                req_info = None
                if str(event.get("action", "")) == "serve_request":
                    req_info = pending_by_ev.pop(int(event["ev_id"]), {"step": float(sr.step), "required_kwh": 0.0})
                stats["total_mcs_income"] += _event_mcs_income(env, event, req_info)
                if str(event.get("action", "")) == "serve_request":
                    wait_steps = max(0, int(sr.step) - int((req_info or {}).get("step", sr.step)))
                    stats["mcs_served"] += 1.0
                    stats["success_requests"] += 1.0
                    stats["wait_steps_sum"] += float(wait_steps)
                    stats["wait_count"] += 1.0
            for to in sr.timeout_events:
                stats["timeout_events_total"] += 1.0
                ev_id = int(to.get("ev_id", -1))
                req_info = pending_by_ev.pop(ev_id, None)
                req_step = int((req_info or {}).get("step", to.get("request_step", sr.step)))
                wait_steps = int(to.get("wait_steps", max(0, int(sr.step) - req_step)))
                stats["wait_steps_sum"] += float(wait_steps)
                stats["wait_count"] += 1.0
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))

            if done:
                last_next_obs = None
                break
            obs_dict = env.get_agent_observations()
            last_next_obs = _obs_matrix(obs_dict, agents)

        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed)

        if done or last_next_obs is None:
            bootstrap = np.zeros((len(agents),), dtype=np.float32)
        else:
            with torch.no_grad():
                next_feats, _, _ = _candidate_features_and_refs(env, agents, req_k=req_k, fcs_k=fcs_k, hot_k=hot_k)
                _, v = model(
                    torch.as_tensor(last_next_obs, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(next_feats, dtype=torch.float32, device=device).unsqueeze(0),
                )
            bootstrap = v.squeeze(0).cpu().numpy().astype(np.float32)
        ep_adv = np.zeros((len(ep_rewards), len(agents)), dtype=np.float32)
        ep_ret = np.zeros((len(ep_rewards), len(agents)), dtype=np.float32)
        gae = np.zeros((len(agents),), dtype=np.float32)
        next_value = bootstrap
        for t in range(len(ep_rewards) - 1, -1, -1):
            delta = ep_rewards[t] + gamma * (1.0 - ep_dones[t]) * next_value - ep_values[t]
            gae = delta + gamma * gae_lambda * (1.0 - ep_dones[t]) * gae
            ep_adv[t] = gae
            ep_ret[t] = gae + ep_values[t]
            next_value = ep_values[t]
        adv_seq.extend([x for x in ep_adv])
        ret_seq.extend([x for x in ep_ret])

    avg_wait_steps = float(stats["wait_steps_sum"] / max(1.0, stats["wait_count"]))
    rollout_stats = {
        "episodes": float(episodes),
        "transitions": float(len(obs_seq) * len(agents)),
        "steps": float(stats["steps"]),
        "requests": float(stats["requests"]),
        "success_rate": float(stats["success_requests"] / max(1.0, stats["requests"])),
        "mcs_success_rate": float(stats["mcs_served"] / max(1.0, stats["mcs_requests"])),
        "avg_wait_steps": avg_wait_steps,
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "unresolved_mcs_total": float(stats["unresolved_mcs_total"]),
        "timeout_events_total": float(stats["timeout_events_total"]),
        "avg_total_agent_reward_per_ep": float(stats["total_agent_reward"] / max(1.0, float(episodes))),
        "mcs_total_income": float(stats["total_mcs_income"]),
        "mcs_avg_income": float(stats["total_mcs_income"] / max(1.0, float(episodes * len(env.mcs_list)))),
    }
    action_total = float(np.sum(target_counts))
    rollout_stats["stay_ratio"] = float(target_counts[0] / max(1.0, action_total))
    rollout_stats["serve_ratio"] = float(target_counts[1] / max(1.0, action_total))
    rollout_stats["reinforce_ratio"] = float(target_counts[2] / max(1.0, action_total))
    rollout_stats["relocate_ratio"] = float(target_counts[3] / max(1.0, action_total))
    return {
        "obs": np.asarray(obs_seq, dtype=np.float32),
        "cand_feats": np.asarray(cand_feat_seq, dtype=np.float32),
        "masks": np.asarray(mask_seq, dtype=np.bool_),
        "actions": np.asarray(action_seq, dtype=np.int64),
        "returns": np.asarray(ret_seq, dtype=np.float32),
        "advantages": np.asarray(adv_seq, dtype=np.float32),
    }, rollout_stats


def pg_update(model: TransformerActorCritic, optimizer: torch.optim.Optimizer, batch: Dict[str, np.ndarray], args: argparse.Namespace, device: torch.device, rng: np.random.Generator) -> Dict[str, float]:
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    cand_feats = torch.as_tensor(batch["cand_feats"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    t_steps, n_agents = int(obs.shape[0]), int(obs.shape[1])
    losses: List[float] = []
    p_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    for _ in range(int(args.update_epochs)):
        perm = rng.permutation(t_steps)
        for start in range(0, t_steps, int(args.mini_batch_size)):
            idx = torch.as_tensor(perm[start : start + int(args.mini_batch_size)], dtype=torch.long, device=device)
            logits, values = model(obs.index_select(0, idx), cand_feats.index_select(0, idx))
            logits = logits.masked_fill(~masks.index_select(0, idx), -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions.index_select(0, idx))
            mb_adv = adv.index_select(0, idx)
            p_loss = -(logp * mb_adv).mean()
            v_loss = F.mse_loss(values, returns.index_select(0, idx))
            entropy = dist.entropy().mean()
            loss = p_loss + float(args.vf_coef) * v_loss - float(args.ent_coef) * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.item()))
            p_losses.append(float(p_loss.item()))
            v_losses.append(float(v_loss.item()))
            entropies.append(float(entropy.item()))
    return {"loss": float(np.mean(losses)), "policy_loss": float(np.mean(p_losses)), "value_loss": float(np.mean(v_losses)), "entropy": float(np.mean(entropies))}


def evaluate_policy(model: TransformerActorCritic, env_config: dict, episodes: int, seed: int, device: torch.device, action_dim: int, deterministic: bool, max_steps: Optional[int], req_k: int = 8, fcs_k: Optional[int] = None, hot_k: int = 8) -> Dict[str, float]:
    if episodes <= 0:
        return {}
    env = Environment(config=env_config, seed=seed)
    if fcs_k is None:
        fcs_k = len(env.fcs_list)
    step_minutes = float(env.config.get("sim_step_minutes", 5))
    agents = list(env.agents)
    stats = {k: 0.0 for k in ["steps", "requests", "mcs_requests", "success_requests", "mcs_served", "wait_steps_sum", "wait_count", "timeout_events_total", "unresolved_mcs_total", "total_agent_reward", "total_mcs_income"]}
    for ep in range(int(episodes)):
        obs_dict = env.reset(seed=int(seed + (ep + 1) * 9973))
        pending_by_ev: Dict[int, Dict[str, float]] = {}
        executed = 0
        horizon = env.total_steps if max_steps is None else min(int(max_steps), env.total_steps)
        for _ in range(horizon):
            obs_mat = _obs_matrix(obs_dict, agents)
            cand_feats, mask_mat, refs = _candidate_features_and_refs(env, agents, req_k=req_k, fcs_k=int(fcs_k), hot_k=hot_k)
            actions, _, _ = model.act(obs_mat, mask_mat, device=device, deterministic=deterministic, cand_feats=cand_feats)
            sr = _targeted_step(env, agents, actions, refs)
            executed += 1
            for req in sr.requests:
                stats["requests"] += 1.0
                if req.get("service_mode") == "fcs":
                    stats["success_requests"] += 1.0
                else:
                    stats["mcs_requests"] += 1.0
                    pending_by_ev[int(req["ev_id"])] = _build_pending_req_info(req, int(sr.step))
            for event in sr.mcs_events:
                req_info = None
                if str(event.get("action", "")) == "serve_request":
                    req_info = pending_by_ev.pop(int(event["ev_id"]), {"step": float(sr.step), "required_kwh": 0.0})
                stats["total_mcs_income"] += _event_mcs_income(env, event, req_info)
                if str(event.get("action", "")) == "serve_request":
                    wait_steps = max(0, int(sr.step) - int((req_info or {}).get("step", sr.step)))
                    stats["mcs_served"] += 1.0
                    stats["success_requests"] += 1.0
                    stats["wait_steps_sum"] += float(wait_steps)
                    stats["wait_count"] += 1.0
            for to in sr.timeout_events:
                stats["timeout_events_total"] += 1.0
                ev_id = int(to.get("ev_id", -1))
                req_info = pending_by_ev.pop(ev_id, None)
                req_step = int((req_info or {}).get("step", to.get("request_step", sr.step)))
                stats["wait_steps_sum"] += float(to.get("wait_steps", max(0, int(sr.step) - req_step)))
                stats["wait_count"] += 1.0
            stats["total_agent_reward"] += float(sum(sr.agent_rewards.values()))
            if sr.done:
                break
            obs_dict = env.get_agent_observations()
        for req_info in pending_by_ev.values():
            req_step = int(req_info.get("step", float(executed - 1)))
            stats["wait_steps_sum"] += float(max(0, int(executed - 1) - req_step))
            stats["wait_count"] += 1.0
        stats["unresolved_mcs_total"] += float(len(pending_by_ev))
        stats["steps"] += float(executed)
    avg_wait_steps = float(stats["wait_steps_sum"] / max(1.0, stats["wait_count"]))
    return {
        "episodes": float(episodes),
        "steps": float(stats["steps"]),
        "requests": float(stats["requests"]),
        "success_rate": float(stats["success_requests"] / max(1.0, stats["requests"])),
        "mcs_success_rate": float(stats["mcs_served"] / max(1.0, stats["mcs_requests"])),
        "avg_wait_steps": avg_wait_steps,
        "avg_wait_minutes": float(avg_wait_steps * step_minutes),
        "unresolved_mcs_total": float(stats["unresolved_mcs_total"]),
        "timeout_events_total": float(stats["timeout_events_total"]),
        "avg_total_agent_reward_per_ep": float(stats["total_agent_reward"] / max(1.0, float(episodes))),
        "mcs_total_income": float(stats["total_mcs_income"]),
        "mcs_avg_income": float(stats["total_mcs_income"] / max(1.0, float(episodes * len(env.mcs_list)))),
    }


def _business_score(stats: Dict[str, float]) -> float:
    return float(1000.0 * float(stats.get("success_rate", 0.0)) + 300.0 * float(stats.get("mcs_success_rate", 0.0)) - 25.0 * float(stats.get("avg_wait_minutes", 0.0)) - 0.05 * float(stats.get("timeout_events_total", 0.0)))


def _save_ckpt(path: Path, model: TransformerActorCritic, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, obs_dim: int, action_dim: int, n_agents: int, metrics: Dict[str, float]) -> None:
    torch.save(
        {
            "transformer_actor_critic_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "args": vars(args),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "n_agents": int(n_agents),
            "hidden_dim": int(args.hidden_dim),
            "n_layer": int(args.n_layer),
            "n_head": int(args.n_head),
            "dropout": float(args.dropout),
            "cand_feat_dim": int(model.cand_feat_dim),
            "candidate_requests": int(args.candidate_requests),
            "candidate_hotspots": int(args.candidate_hotspots),
            "policy_type": "candidate_target",
            "metrics": metrics,
        },
        path,
    )


def _save_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CONFIG)
    cfg["use_lstm_summary"] = bool(args.use_lstm_summary)
    if args.lstm_predictor_ckpt:
        cfg["lstm_predictor_ckpt"] = str(args.lstm_predictor_ckpt)
    env = Environment(config=cfg, seed=int(args.seed))
    obs = env.reset(seed=int(args.seed))
    obs_dim = int(next(iter(obs.values())).shape[0])
    n_agents = int(len(env.agents))
    req_k, fcs_k, hot_k, action_dim = _candidate_sizes(env, args)
    cand_feat_dim = 10
    model = TransformerActorCritic(obs_dim, action_dim, n_agents, int(args.hidden_dim), int(args.n_layer), int(args.n_head), float(args.dropout), cand_feat_dim=cand_feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_business = float("-inf")
    best_success = float("-inf")
    rows: List[Dict[str, float]] = []
    log_path = outdir / "mn_icrsp_transformer_pg_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    print(f"mn_icrsp_transformer_pg_init obs_dim={obs_dim} n_agents={n_agents} action_dim={action_dim} req_k={req_k} fcs_k={fcs_k} hot_k={hot_k} device={device}", flush=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batch, rollout_stats = collect_rollouts(env, model, int(args.episodes_per_epoch), float(args.gamma), float(args.gae_lambda), str(args.team_reward_mode), device, rng, action_dim, args.max_steps, req_k=req_k, fcs_k=fcs_k, hot_k=hot_k)
        update_stats = pg_update(model, optimizer, batch, args, device, rng)
        eval_stats: Dict[str, float] = {}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            model.eval()
            eval_stats = evaluate_policy(model, cfg, int(args.eval_episodes), int(args.eval_seed + epoch), device, action_dim, deterministic=not bool(args.eval_stochastic), max_steps=args.eval_max_steps, req_k=req_k, fcs_k=fcs_k, hot_k=hot_k)
        collect_business = _business_score(rollout_stats)
        eval_business = _business_score(eval_stats) if eval_stats else float("nan")
        metric_stats = eval_stats if eval_stats else rollout_stats
        metric_business = float(eval_business) if eval_stats else float(collect_business)
        merged = {**rollout_stats, **update_stats, **eval_stats, "business_score": metric_business}
        _save_ckpt(outdir / "last.pt", model, optimizer, epoch, args, obs_dim, action_dim, n_agents, merged)
        if int(args.save_epoch_interval) > 0 and epoch % int(args.save_epoch_interval) == 0:
            _save_ckpt(outdir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, args, obs_dim, action_dim, n_agents, merged)
        if metric_business > best_business:
            best_business = metric_business
            _save_ckpt(outdir / "best.pt", model, optimizer, epoch, args, obs_dim, action_dim, n_agents, merged)
            _save_ckpt(outdir / "best_by_business.pt", model, optimizer, epoch, args, obs_dim, action_dim, n_agents, merged)
        if float(metric_stats.get("success_rate", -1.0)) > best_success:
            best_success = float(metric_stats.get("success_rate", -1.0))
            _save_ckpt(outdir / "best_by_success.pt", model, optimizer, epoch, args, obs_dim, action_dim, n_agents, merged)
        row = {
            "epoch": float(epoch),
            "reward": float(rollout_stats.get("avg_total_agent_reward_per_ep", 0.0)),
            "success_rate": float(rollout_stats.get("success_rate", 0.0)),
            "mcs_avg_income": float(rollout_stats.get("mcs_avg_income", 0.0)),
            "ev_avg_wait_minutes": float(rollout_stats.get("avg_wait_minutes", 0.0)),
            "business_score": float(collect_business),
            "eval_reward": float(eval_stats.get("avg_total_agent_reward_per_ep", np.nan)) if eval_stats else float("nan"),
            "eval_success_rate": float(eval_stats.get("success_rate", np.nan)) if eval_stats else float("nan"),
            "eval_mcs_avg_income": float(eval_stats.get("mcs_avg_income", np.nan)) if eval_stats else float("nan"),
            "eval_ev_avg_wait_minutes": float(eval_stats.get("avg_wait_minutes", np.nan)) if eval_stats else float("nan"),
            "eval_business_score": float(eval_business),
            "reinforce_ratio": float(rollout_stats.get("reinforce_ratio", 0.0)),
            "relocate_ratio": float(rollout_stats.get("relocate_ratio", 0.0)),
            "serve_ratio": float(rollout_stats.get("serve_ratio", 0.0)),
            "stay_ratio": float(rollout_stats.get("stay_ratio", 0.0)),
        }
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "collect": rollout_stats, "update": update_stats, "eval": eval_stats}, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} collect_success={row['success_rate']:.3f} "
            f"collect_wait={row['ev_avg_wait_minutes']:.2f}min "
            f"serve={row['serve_ratio']:.3f} stay={row['stay_ratio']:.3f} "
            f"eval_success={row['eval_success_rate']:.3f} eval_biz={row['eval_business_score']:.2f}",
            flush=bool(args.log_flush),
        )

    _save_metrics_csv(outdir / "business_metrics.csv", rows)
    _plot_business_metrics(outdir / "business_metrics.png", rows)
    with (outdir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"method": "MN-ICRSP-style Transformer policy-gradient", "best_by_business": float(best_business), "best_by_success": float(best_success)}, f, ensure_ascii=False, indent=2)
    print(f"done: checkpoints/log in {outdir}")


if __name__ == "__main__":
    main()
