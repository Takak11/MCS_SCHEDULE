from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import CONFIG
from entities import EV, FCS, MCS
from utils import haversine_distance, hungarian_assignment


LatLon = Tuple[float, float]
ActionInput = Union[Dict[Union[int, str], int], None]


@dataclass
class TracePoint:
    location: LatLon
    distance_km: float


@dataclass
class StepResult:
    step: int
    active_ev_count: int
    new_request_count: int
    requests: List[dict]
    fcs_arrivals: Dict[int, int]
    fcs_states: Dict[int, dict] = field(default_factory=dict)
    mcs_decisions: Dict[str, int] = field(default_factory=dict)
    mcs_events: List[dict] = field(default_factory=list)
    timeout_events: List[dict] = field(default_factory=list)
    agent_rewards: Dict[str, float] = field(default_factory=dict)
    done: bool = False


class Environment:
    ACTION_SPACE = {
        1: "reinforce_fcs",
        2: "relocate",
        3: "serve_ev_requests_hungarian",
        4: "stay",
    }

    def __init__(self, config: Optional[dict] = None, seed: Optional[int] = None) -> None:
        self.config = config or CONFIG
        self.rng = np.random.default_rng(seed)

        self.total_steps = int(self.config["steps"])
        self.dataset_path = Path(self.config["dataset_path"])
        self.table_path = Path(self.config["table_path"])

        self.current_step = 0

        self.mcs_list: List[MCS] = []
        self.fcs_list: List[FCS] = []
        self.mcs_by_id: Dict[int, MCS] = {}

        self.dataset_df = pd.DataFrame()
        self.evs: Dict[int, EV] = {}
        self.active_ev_ids: set[int] = set()
        self.trace_by_ev: Dict[int, List[TracePoint]] = {}
        self.trace_cursor_by_ev: Dict[int, int] = {}

        self.selected_ev_ids: List[int] = []
        self.appear_step_by_ev: Dict[int, int] = {}
        self.initial_soc_by_ev: Dict[int, float] = {}
        self.ev_ids_by_step: Dict[int, List[int]] = defaultdict(list)
        self.fcs_arrival_schedule: Dict[int, Dict[int, int]] = {}
        self.pending_ev_requests: List[dict] = []
        self.mcs_available_step: Dict[int, int] = {}
        self.fcs_by_id: Dict[int, FCS] = {}
        self.fcs_queue: Dict[int, int] = {}
        self.fcs_charging_remaining: Dict[int, List[int]] = {}
        self.fcs_completed_counts: Dict[int, int] = {}
        self.relocate_hotspots: List[LatLon] = []
        self.relocate_hotspot_weights: np.ndarray = np.array([], dtype=float)
        self.fcs_region_idx: Dict[int, int] = {}
        self._relocate_hotspots_ready = False
        self._predictive_bundle: Optional[object] = None
        self._predictive_history: List[np.ndarray] = []
        self._predictive_summary: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._predictive_req_pred: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._predictive_cong_pred: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._predictive_enabled: bool = bool(self.config.get("use_lstm_summary", False))
        self._predictive_ckpt: Path = Path(self.config.get("lstm_predictor_ckpt", "result/predictor/lstm_predictor.pt"))
        self._predictive_device: str = str(self.config.get("lstm_predictor_device", "cpu"))

        self.agents = [f"mcs_{i + 1}" for i in range(int(self.config["mcs_num"]))]
        self.agent_to_mcs_id = {agent: i + 1 for i, agent in enumerate(self.agents)}
        self.mcs_id_to_agent = {i + 1: agent for i, agent in enumerate(self.agents)}

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.evs.clear()
        self.active_ev_ids.clear()
        self.trace_by_ev.clear()
        self.trace_cursor_by_ev.clear()
        self.pending_ev_requests.clear()
        self.mcs_available_step.clear()
        self.fcs_by_id.clear()
        self.fcs_queue.clear()
        self.fcs_charging_remaining.clear()
        self.fcs_completed_counts.clear()
        self._predictive_history.clear()
        self._predictive_summary = np.zeros((0,), dtype=np.float32)
        self._predictive_req_pred = np.zeros((0,), dtype=np.float32)
        self._predictive_cong_pred = np.zeros((0,), dtype=np.float32)

        self._init_mcs()
        self._init_fcs()
        self._load_dataset_base()
        self._prepare_join_schedule()
        if (not self._relocate_hotspots_ready) or bool(self.config.get("mcs_relocate_hotspot_recompute_each_reset", False)):
            self._build_relocate_hotspots()
            self._relocate_hotspots_ready = True
        self._prepare_initial_soc_pool()
        self._build_fcs_arrival_schedule()
        self._init_predictive_summary()
        return self.get_agent_observations()

    def _init_mcs(self) -> None:
        self.mcs_list = [
            MCS(
                mcs_id=i + 1,
                speed_km_per_step=float(self.config["mcs_speed_km_per_step"]),
                service_radius_km=float(self.config["mcs_service_radius_km"]),
                price_per_kwh=float(self.config["mcs_price_per_kwh"]),
                cost_per_km=float(self.config["mcs_cost_per_km"]),
            )
            for i in range(int(self.config["mcs_num"]))
        ]
        self.mcs_by_id = {m.mcs_id: m for m in self.mcs_list}

    def _init_fcs(self) -> None:
        self.fcs_list = [
            FCS(
                fcs_id=i + 1,
                lon_lat=(float(lon), float(lat)),
                capacity=int(self.config["fcs_capacity"]),
            )
            for i, (lon, lat) in enumerate(self.config["fcs_locations"])
        ]
        self.fcs_by_id = {f.fcs_id: f for f in self.fcs_list}
        self.fcs_queue = {f.fcs_id: 0 for f in self.fcs_list}
        self.fcs_charging_remaining = {f.fcs_id: [] for f in self.fcs_list}
        self.fcs_completed_counts = {f.fcs_id: 0 for f in self.fcs_list}

        if self.fcs_list:
            for i, mcs in enumerate(self.mcs_list):
                mcs.location = self.fcs_list[i % len(self.fcs_list)].lat_lon
                mcs.busy = False
                self.mcs_available_step[mcs.mcs_id] = 0

    def _fcs_service_steps(self) -> int:
        step_minutes = float(self.config.get("sim_step_minutes", 5))
        charge_minutes = float(self.config.get("fcs_charge_minutes", self.config.get("max_charge_minutes", 20)))
        return max(1, int(np.ceil(charge_minutes / step_minutes)))

    def _update_fcs_runtime(self, mcs_events: List[dict]) -> Dict[int, dict]:
        arrivals = self.fcs_arrival_schedule.get(self.current_step, {})
        support_per_mcs = float(self.config.get("mcs_reinforce_ev_per_step", 1.0))
        relief_unit = max(1, int(np.ceil(support_per_mcs))) if support_per_mcs > 0 else 1
        service_steps = self._fcs_service_steps()

        reinforce_relief: Dict[int, int] = defaultdict(int)
        for e in mcs_events:
            if e.get("action") == "reinforce_fcs":
                reinforce_relief[int(e["target_fcs"])] += relief_unit

        states: Dict[int, dict] = {}
        for fcs in self.fcs_list:
            fid = fcs.fcs_id

            # 1) Progress charging sessions and release completed ones.
            updated_remaining: List[int] = []
            finished_step = 0
            for remain in self.fcs_charging_remaining[fid]:
                next_remain = int(remain) - 1
                if next_remain <= 0:
                    finished_step += 1
                else:
                    updated_remaining.append(next_remain)
            self.fcs_charging_remaining[fid] = updated_remaining
            self.fcs_completed_counts[fid] += finished_step
            fcs.occupied = len(self.fcs_charging_remaining[fid])

            # 2) Add incoming arrivals to queue.
            arrivals_step = int(arrivals.get(fid, 0))
            self.fcs_queue[fid] += arrivals_step

            # 3) Admit from queue to FCS chargers by free slots.
            free_slots = max(0, int(fcs.capacity) - int(fcs.occupied))
            admitted = min(self.fcs_queue[fid], free_slots)
            if admitted > 0:
                self.fcs_queue[fid] -= admitted
                self.fcs_charging_remaining[fid].extend([service_steps] * admitted)
                fcs.occupied = len(self.fcs_charging_remaining[fid])

            # 4) Reinforcement from MCS can drain part of waiting queue.
            reinforced_served = min(self.fcs_queue[fid], reinforce_relief.get(fid, 0))
            if reinforced_served > 0:
                self.fcs_queue[fid] -= reinforced_served

            states[fid] = {
                "arrivals": arrivals_step,
                "queue": int(self.fcs_queue[fid]),
                "occupied": int(fcs.occupied),
                "capacity": int(fcs.capacity),
                "finished_step": int(finished_step),
                "admitted_step": int(admitted),
                "reinforced_served_step": int(reinforced_served),
                "completed_total": int(self.fcs_completed_counts[fid]),
            }
        return states

    def _load_dataset_base(self) -> None:
        self.dataset_df = pd.read_csv(
            self.dataset_path,
            header=None,
            names=["id", "lat", "lon", "timestamp"],
            usecols=[0, 1, 2],
        )

    def _prepare_join_schedule(self) -> None:
        table_df = pd.read_csv(self.table_path)
        ev_count = int(self.config.get("ev_count", len(table_df)))
        if ev_count > 0:
            table_df = table_df.iloc[:ev_count]

        self.selected_ev_ids = [int(v) for v in table_df["id"].tolist()]
        self.appear_step_by_ev = {int(row.id): int(row.step) for row in table_df.itertuples(index=False)}

        self.ev_ids_by_step = defaultdict(list)
        for ev_id in self.selected_ev_ids:
            self.ev_ids_by_step[self.appear_step_by_ev[ev_id]].append(ev_id)

    def _prepare_initial_soc_pool(self) -> None:
        soc_values = self._sample_initial_soc_kwh(len(self.selected_ev_ids))
        self.initial_soc_by_ev = {ev_id: float(soc_values[i]) for i, ev_id in enumerate(self.selected_ev_ids)}

    def _kmeans_latlon(self, points: np.ndarray, k: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        n = int(points.shape[0])
        if n <= 0 or k <= 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=np.int64)

        if n >= k:
            centers = points[self.rng.choice(n, size=k, replace=False)].astype(float).copy()
        else:
            centers = points[self.rng.choice(n, size=k, replace=True)].astype(float).copy()

        labels = np.zeros((n,), dtype=np.int64)
        for it in range(max(1, int(max_iter))):
            dist2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(dist2, axis=1).astype(np.int64)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centers[j] = points[mask].mean(axis=0)
                else:
                    centers[j] = points[int(self.rng.integers(0, n))]

        counts = np.bincount(labels, minlength=k).astype(np.int64)
        return centers, counts

    def _build_relocate_hotspots(self) -> None:
        k = int(self.config.get("mcs_relocate_hotspot_k", 15))
        max_iter = int(self.config.get("mcs_relocate_hotspot_max_iter", 30))
        sample_size = int(self.config.get("mcs_relocate_hotspot_sample_size", 30000))

        if k <= 0:
            self.relocate_hotspots = []
            self.relocate_hotspot_weights = np.array([], dtype=float)
            self.fcs_region_idx = {}
            return

        if self.selected_ev_ids:
            rows = self.dataset_df[self.dataset_df["id"].isin(self.selected_ev_ids)]
        else:
            rows = self.dataset_df
        pts = rows[["lat", "lon"]].to_numpy(dtype=float, copy=True)

        if sample_size > 0 and pts.shape[0] > sample_size:
            idx = self.rng.choice(pts.shape[0], size=sample_size, replace=False)
            pts = pts[idx]

        centers, counts = self._kmeans_latlon(pts, k=k, max_iter=max_iter)
        if centers.shape[0] == 0:
            self.relocate_hotspots = [f.lat_lon for f in self.fcs_list]
            if self.relocate_hotspots:
                self.relocate_hotspot_weights = np.ones((len(self.relocate_hotspots),), dtype=float) / len(self.relocate_hotspots)
            else:
                self.relocate_hotspot_weights = np.array([], dtype=float)
            self.fcs_region_idx = {f.fcs_id: i for i, f in enumerate(self.fcs_list)}
            return

        self.relocate_hotspots = [(float(c[0]), float(c[1])) for c in centers]
        w = counts.astype(float)
        if w.sum() <= 0:
            w = np.ones((len(self.relocate_hotspots),), dtype=float)
        self.relocate_hotspot_weights = w / w.sum()

        self.fcs_region_idx = {}
        for fcs in self.fcs_list:
            d = [haversine_distance(fcs.lat_lon, loc) for loc in self.relocate_hotspots]
            self.fcs_region_idx[fcs.fcs_id] = int(np.argmin(d)) if d else 0

    def _init_predictive_summary(self) -> None:
        if not self._predictive_enabled:
            self._predictive_summary = np.zeros((0,), dtype=np.float32)
            self._predictive_req_pred = np.zeros((0,), dtype=np.float32)
            self._predictive_cong_pred = np.zeros((0,), dtype=np.float32)
            self._predictive_history.clear()
            return

        if self._predictive_bundle is None:
            from predictive_summary import load_predictive_summary_bundle

            self._predictive_bundle = load_predictive_summary_bundle(
                path=self._predictive_ckpt,
                device=self._predictive_device,
            )

        summary_dim = int(getattr(self._predictive_bundle, "summary_dim"))
        region_k = int(getattr(self._predictive_bundle, "region_k"))
        fcs_n = int(getattr(self._predictive_bundle, "fcs_n"))
        self._predictive_summary = np.zeros((summary_dim,), dtype=np.float32)
        self._predictive_req_pred = np.zeros((region_k,), dtype=np.float32)
        self._predictive_cong_pred = np.zeros((fcs_n,), dtype=np.float32)
        self._predictive_history.clear()

    def _update_predictive_summary(self, requests: List[dict], fcs_states: Dict[int, dict]) -> None:
        if (not self._predictive_enabled) or self._predictive_bundle is None:
            return

        region_k = int(getattr(self._predictive_bundle, "region_k"))
        fcs_n = int(getattr(self._predictive_bundle, "fcs_n"))
        centers = np.asarray(getattr(self._predictive_bundle, "region_centers"), dtype=np.float32)

        req_vec = np.zeros((region_k,), dtype=np.float32)
        locs = [req.get("location") for req in requests if req.get("location") is not None]
        if locs and centers.shape[0] > 0:
            loc_arr = np.asarray(locs, dtype=np.float32)
            dist2 = ((loc_arr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(dist2, axis=1)
            np.add.at(req_vec, idx, 1.0)

        fcs_vec = np.zeros((fcs_n,), dtype=np.float32)
        for i in range(min(fcs_n, len(self.fcs_list))):
            fcs = self.fcs_list[i]
            st = fcs_states.get(fcs.fcs_id)
            q = float(st["queue"]) if st is not None else float(self.fcs_queue.get(fcs.fcs_id, 0))
            cap = float(st["capacity"]) if st is not None else float(fcs.capacity)
            fcs_vec[i] = q / max(1.0, cap)

        feature = np.concatenate([req_vec, fcs_vec], axis=0).astype(np.float32)
        self._predictive_history.append(feature)

        seq_len = int(getattr(self._predictive_bundle, "seq_len"))
        if len(self._predictive_history) > seq_len:
            self._predictive_history = self._predictive_history[-seq_len:]

        seq = np.asarray(self._predictive_history, dtype=np.float32)
        req_pred, cong_pred, summary = self._predictive_bundle.predict_outputs(seq)
        self._predictive_req_pred = np.asarray(req_pred, dtype=np.float32)
        self._predictive_cong_pred = np.asarray(cong_pred, dtype=np.float32)
        self._predictive_summary = np.asarray(summary, dtype=np.float32)

    def _sample_initial_soc_kwh(self, count: int) -> np.ndarray:
        capacity = float(self.config["ev_battery_capacity_kwh"])
        mean = float(self.config["ev_init_soc_mean"]) * capacity
        std = float(self.config["ev_init_soc_std"]) * capacity
        sigma_clip = float(self.config.get("ev_init_soc_sigma_clip", 2.5))

        low = max(1.0, mean - sigma_clip * std)
        high = min(capacity, mean + sigma_clip * std)

        vals: List[float] = []
        while len(vals) < count:
            batch = self.rng.normal(mean, std, size=max(128, count - len(vals)))
            batch = batch[(batch >= low) & (batch <= high)]
            vals.extend(batch.tolist())
        return np.array(vals[:count], dtype=float)

    def _build_fcs_arrival_schedule(self) -> None:
        s = np.arange(self.total_steps, dtype=float)
        base = float(self.config.get("fcs_arrival_base", 0.2))
        am = float(self.config.get("fcs_arrival_morning_amp", 1.0))
        ae = float(self.config.get("fcs_arrival_evening_amp", 1.2))
        um = float(self.config.get("fcs_arrival_morning_center_step", 24))
        ue = float(self.config.get("fcs_arrival_evening_center_step", 150))
        sm = float(self.config.get("fcs_arrival_morning_sigma", 8))
        se = float(self.config.get("fcs_arrival_evening_sigma", 10))

        curve = (
            base
            + am * np.exp(-((s - um) ** 2) / (2 * sm * sm))
            + ae * np.exp(-((s - ue) ** 2) / (2 * se * se))
        )
        curve = curve / curve.sum()

        total = int(self.config.get("fcs_arrival_total_per_day", self.config.get("ev_count", 1000)))
        step_totals = self.rng.multinomial(total, curve)

        station_weights = np.array(
            self.config.get("fcs_arrival_station_weights", [1.0] * len(self.fcs_list)),
            dtype=float,
        )
        if len(station_weights) != len(self.fcs_list):
            station_weights = np.ones(len(self.fcs_list), dtype=float)
        station_weights = station_weights / station_weights.sum()

        schedule: Dict[int, Dict[int, int]] = {}
        fcs_ids = [f.fcs_id for f in self.fcs_list]
        for step, n in enumerate(step_totals):
            if n <= 0:
                schedule[step] = {}
                continue
            counts = self.rng.multinomial(int(n), station_weights)
            schedule[step] = {fcs_ids[i]: int(v) for i, v in enumerate(counts) if v > 0}
        self.fcs_arrival_schedule = schedule

    def _load_ev_trace(self, ev_id: int) -> List[TracePoint]:
        rows = self.dataset_df[self.dataset_df["id"] == ev_id]
        points: List[TracePoint] = []
        prev: Optional[LatLon] = None
        for row in rows.itertuples(index=False):
            loc = (float(row.lat), float(row.lon))
            dist = 0.0 if prev is None else haversine_distance(prev, loc)
            points.append(TracePoint(location=loc, distance_km=dist))
            prev = loc
        return points

    def _create_ev(self, ev_id: int) -> None:
        self.trace_by_ev[ev_id] = self._load_ev_trace(ev_id)
        self.trace_cursor_by_ev[ev_id] = 0

        ev = EV(
            ev_id=ev_id,
            appear_step=self.appear_step_by_ev[ev_id],
            initial_soc_kwh=self.initial_soc_by_ev[ev_id],
        )
        ev.activate(None)
        self.evs[ev_id] = ev
        self.active_ev_ids.add(ev_id)

    def _ev_step(self, ev_id: int) -> Optional[dict]:
        ev = self.evs[ev_id]
        if ev.request_sent:
            return None

        idx = self.trace_cursor_by_ev[ev_id]
        trace = self.trace_by_ev[ev_id]
        if idx < len(trace):
            point = trace[idx]
            self.trace_cursor_by_ev[ev_id] = idx + 1
            ev.update_location(point.location)
            if point.distance_km > 0:
                ev.consume_by_distance(point.distance_km)

        if ev.should_request_charge():
            reachable = self._nearest_reachable_fcs(ev)
            if reachable is not None:
                target_fcs, distance_km = reachable
                # Navigate to nearby reachable FCS immediately (within attraction radius).
                if distance_km > 0:
                    ev.consume_by_distance(distance_km)
                ev.update_location(target_fcs.lat_lon)
                ev.mark_request(self.current_step)
                self.fcs_queue[target_fcs.fcs_id] += 1
                return {
                    "ev_id": ev_id,
                    "step": self.current_step,
                    "soc_kwh": round(ev.current_soc_kwh, 4),
                    "soc_ratio": round(ev.soc_ratio, 4),
                    "required_kwh": round(ev.required_charge_kwh, 4),
                    "charge_minutes": round(ev.request_charge_minutes or 0.0, 2),
                    "location": ev.current_location,
                    "service_mode": "fcs",
                    "target_fcs": target_fcs.fcs_id,
                    "distance_to_fcs_km": round(distance_km, 4),
                }

            ev.mark_request(self.current_step)
            return {
                "ev_id": ev_id,
                "step": self.current_step,
                "soc_kwh": round(ev.current_soc_kwh, 4),
                "soc_ratio": round(ev.soc_ratio, 4),
                "required_kwh": round(ev.required_charge_kwh, 4),
                "charge_minutes": round(ev.request_charge_minutes or 0.0, 2),
                "location": ev.current_location,
                "service_mode": "mcs",
            }
        return None

    def _nearest_reachable_fcs(self, ev: EV) -> Optional[Tuple[FCS, float]]:
        if ev.current_location is None:
            return None

        absorb_radius = float(self.config.get("fcs_absorb_radius_km", 1.0))
        consumption = float(self.config.get("ev_consumption_rate", 0.3))

        best: Optional[Tuple[FCS, float]] = None
        for fcs in self.fcs_list:
            distance_km = haversine_distance(ev.current_location, fcs.lat_lon)
            if distance_km > absorb_radius:
                continue
            need_kwh = distance_km * consumption
            if need_kwh > ev.current_soc_kwh:
                continue
            if best is None or distance_km < best[1]:
                best = (fcs, distance_km)
        return best

    def _refresh_mcs_state(self) -> None:
        for mcs in self.mcs_list:
            mcs.busy = self.current_step < self.mcs_available_step.get(mcs.mcs_id, 0)

    def _drop_timeout_pending_requests(self) -> List[dict]:
        step_minutes = float(self.config.get("sim_step_minutes", 5))
        timeout_minutes = float(self.config.get("ev_request_timeout_minutes", 20))
        timeout_steps = max(1, int(np.ceil(timeout_minutes / step_minutes)))

        keep: List[dict] = []
        dropped: List[dict] = []
        for req in self.pending_ev_requests:
            req_step = int(req.get("step", self.current_step))
            wait_steps = max(0, int(self.current_step) - req_step)
            if wait_steps < timeout_steps:
                keep.append(req)
                continue

            ev_id = int(req.get("ev_id", -1))
            if ev_id in self.active_ev_ids:
                self.active_ev_ids.remove(ev_id)
            ev = self.evs.get(ev_id)
            if ev is not None:
                ev.active = False
            dropped.append(
                {
                    "ev_id": ev_id,
                    "request_step": req_step,
                    "drop_step": int(self.current_step),
                    "wait_steps": int(wait_steps),
                    "timeout_steps": int(timeout_steps),
                    "reason": "request_timeout",
                }
            )

        self.pending_ev_requests = keep
        return dropped

    def _travel_steps(self, distance_km: float) -> int:
        speed = float(self.config.get("mcs_speed_km_per_step", 0.0))
        if speed <= 0:
            return 0
        return int(np.ceil(max(0.0, float(distance_km)) / speed))

    def _reserve_mcs(self, mcs: MCS, busy_steps: int) -> int:
        hold = max(1, int(busy_steps))
        available_step = self.current_step + hold
        self.mcs_available_step[mcs.mcs_id] = max(self.mcs_available_step.get(mcs.mcs_id, 0), available_step)
        mcs.busy = True
        return hold

    def _decode_action(self, action: int) -> int:
        return action if action in self.ACTION_SPACE else 4

    def _normalize_parallel_actions(self, mcs_actions: ActionInput, default_action: int = 4) -> Dict[str, int]:
        decisions: Dict[str, int] = {}
        for agent in self.agents:
            mcs_id = self.agent_to_mcs_id[agent]
            mcs = self.mcs_by_id[mcs_id]
            if mcs.busy:
                decisions[agent] = 4
                continue

            action = default_action
            if mcs_actions is not None:
                if mcs_id in mcs_actions:
                    action = int(mcs_actions[mcs_id])
                elif agent in mcs_actions:
                    action = int(mcs_actions[agent])
            decisions[agent] = self._decode_action(action)
        return decisions

    def _actions_to_candidates(self, decisions: Dict[str, int]) -> Dict[int, List[MCS]]:
        chosen: Dict[int, List[MCS]] = {1: [], 2: [], 3: [], 4: []}

        for agent, action in decisions.items():
            mcs_id = self.agent_to_mcs_id[agent]
            chosen[action].append(self.mcs_by_id[mcs_id])

        return chosen

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        mask: Dict[str, np.ndarray] = {}
        has_pending = any(req.get("location") is not None for req in self.pending_ev_requests)
        for agent in self.agents:
            mcs = self.mcs_by_id[self.agent_to_mcs_id[agent]]
            valid = np.array([1, 1, 1, 1], dtype=np.int8)  # [act1, act2, act3, act4]
            if mcs.busy:
                valid[:] = 0
                valid[3] = 1
            elif mcs.location is None:
                valid[2] = 0
            elif not has_pending:
                valid[2] = 0
            mask[agent] = valid
        return mask

    def _action_reinforce_fcs(self, candidates: List[MCS]) -> List[dict]:
        idle = [mcs for mcs in candidates if mcs.location is not None]
        if not idle or not self.fcs_list:
            return []

        arrivals = self.fcs_arrival_schedule.get(self.current_step, {})
        support_per_mcs = float(self.config.get("mcs_reinforce_ev_per_step", 1.0))
        if support_per_mcs <= 0:
            support_per_mcs = 1.0

        target_slots: List[FCS] = []
        for fcs in self.fcs_list:
            queue_now = float(self.fcs_queue.get(fcs.fcs_id, 0))
            free_slots = max(0.0, float(fcs.capacity - fcs.occupied))
            need_ev = max(0.0, queue_now + float(arrivals.get(fcs.fcs_id, 0)) - free_slots)
            need_mcs = int(np.ceil(need_ev / support_per_mcs))
            target_slots.extend([fcs] * need_mcs)

        if not target_slots:
            return []

        cost_matrix = [[haversine_distance(mcs.location, slot.lat_lon) for slot in target_slots] for mcs in idle]
        pairs = hungarian_assignment(cost_matrix)

        events = []
        for mcs_idx, slot_idx in pairs:
            mcs = idle[mcs_idx]
            target = target_slots[slot_idx]
            distance_km = float(cost_matrix[mcs_idx][slot_idx])
            mcs.location = target.lat_lon
            travel_steps = self._travel_steps(distance_km)
            reinforce_steps = max(1, int(self.config.get("mcs_reinforce_busy_steps", 1)))
            busy_steps = self._reserve_mcs(mcs, travel_steps + reinforce_steps)
            events.append(
                {
                    "agent": self.mcs_id_to_agent[mcs.mcs_id],
                    "mcs_id": mcs.mcs_id,
                    "action": "reinforce_fcs",
                    "target_fcs": target.fcs_id,
                    "distance_km": round(distance_km, 3),
                    "travel_steps": int(travel_steps),
                    "busy_steps": int(busy_steps),
                }
            )
        return events

    def _action_relocate(self, candidates: List[MCS]) -> List[dict]:
        idle = candidates
        if not idle or not self.relocate_hotspots:
            return []
        weights = self.relocate_hotspot_weights
        if weights.size != len(self.relocate_hotspots):
            weights = np.ones((len(self.relocate_hotspots),), dtype=float) / len(self.relocate_hotspots)

        slots = self.rng.multinomial(len(idle), weights)
        target_slots: List[Tuple[int, LatLon]] = []
        for hid, n in enumerate(slots):
            if n <= 0:
                continue
            loc = self.relocate_hotspots[hid]
            target_slots.extend([(hid, loc)] * int(n))
        if not target_slots:
            target_slots = [(i % len(self.relocate_hotspots), self.relocate_hotspots[i % len(self.relocate_hotspots)]) for i in range(len(idle))]

        cost_matrix = []
        for mcs in idle:
            row = []
            for _, loc in target_slots:
                dist = 0.0 if mcs.location is None else haversine_distance(mcs.location, loc)
                row.append(dist)
            cost_matrix.append(row)
        pairs = hungarian_assignment(cost_matrix)

        events = []
        for mcs_idx, slot_idx in pairs:
            mcs = idle[mcs_idx]
            hotspot_id, target_loc = target_slots[slot_idx]
            distance_km = float(cost_matrix[mcs_idx][slot_idx])
            source_region = self._infer_region_idx(mcs.location, len(self.relocate_hotspots))
            mcs.location = target_loc
            travel_steps = self._travel_steps(distance_km)
            busy_steps = self._reserve_mcs(mcs, travel_steps)
            events.append(
                {
                    "agent": self.mcs_id_to_agent[mcs.mcs_id],
                    "mcs_id": mcs.mcs_id,
                    "action": "relocate",
                    "source_region": int(source_region),
                    "target_region": int(hotspot_id),
                    "target_hotspot": int(hotspot_id + 1),
                    "target_location": (round(float(target_loc[0]), 6), round(float(target_loc[1]), 6)),
                    "distance_km": round(distance_km, 3),
                    "travel_steps": int(travel_steps),
                    "busy_steps": int(busy_steps),
                }
            )
        return events

    def _action_service_ev_requests(self, candidates: List[MCS]) -> List[dict]:
        idle = [m for m in candidates if m.location is not None]
        pending = [r for r in self.pending_ev_requests if r.get("location") is not None]
        if not idle or not pending:
            return []

        cost_matrix = [[haversine_distance(mcs.location, req["location"]) for req in pending] for mcs in idle]
        pairs = hungarian_assignment(cost_matrix)

        assigned_ev_ids = set()
        events = []
        for mcs_idx, req_idx in pairs:
            mcs = idle[mcs_idx]
            req = pending[req_idx]
            distance_km = cost_matrix[mcs_idx][req_idx]
            if distance_km > mcs.service_radius_km:
                continue

            charge_minutes = float(req.get("charge_minutes", 0.0))
            req_step = int(req.get("step", self.current_step))
            wait_steps = max(0, int(self.current_step) - int(req_step))
            travel_steps = self._travel_steps(distance_km)
            mcs.location = req["location"]
            service_steps = self._reserve_mcs(mcs, travel_steps)
            assigned_ev_ids.add(req["ev_id"])

            events.append(
                {
                    "agent": self.mcs_id_to_agent[mcs.mcs_id],
                    "mcs_id": mcs.mcs_id,
                    "action": "serve_request",
                    "ev_id": req["ev_id"],
                    "distance_km": round(distance_km, 3),
                    "travel_steps": int(travel_steps),
                    "charge_steps": int(max(1, int(np.ceil(charge_minutes / float(self.config.get("sim_step_minutes", 5)))))),
                    "service_steps": int(service_steps),
                    "request_step": int(req_step),
                    "wait_steps": int(wait_steps),
                }
            )

        if assigned_ev_ids:
            self.pending_ev_requests = [r for r in self.pending_ev_requests if r["ev_id"] not in assigned_ev_ids]

        return events

    def _norm_location_xy(self, loc: Optional[LatLon]) -> Tuple[float, float]:
        if loc is None:
            return 0.0, 0.0
        lat, lon = float(loc[0]), float(loc[1])
        west = float(self.config.get("WEST", lon - 1.0))
        east = float(self.config.get("EAST", lon + 1.0))
        south = float(self.config.get("SOUTH", lat - 1.0))
        north = float(self.config.get("NORTH", lat + 1.0))
        x = (lon - west) / max(1e-6, east - west)
        y = (lat - south) / max(1e-6, north - south)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def _infer_region_idx(self, loc: Optional[LatLon], region_k: int) -> int:
        if region_k <= 0:
            return 0
        if (loc is None) or (not self.relocate_hotspots):
            return 0
        d = [haversine_distance(loc, h) for h in self.relocate_hotspots]
        if not d:
            return 0
        return int(np.argmin(d))

    def _fcs_risk_metrics(self, future_horizon: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        fut_arr_by_fid: Dict[int, float] = {f.fcs_id: 0.0 for f in self.fcs_list}
        end = min(self.total_steps, self.current_step + 1 + max(1, int(future_horizon)))
        for t in range(self.current_step + 1, end):
            arr = self.fcs_arrival_schedule.get(t, {})
            for fid, n in arr.items():
                fut_arr_by_fid[int(fid)] = fut_arr_by_fid.get(int(fid), 0.0) + float(n)

        risk_by_fid: Dict[int, float] = {}
        for fcs in self.fcs_list:
            fid = int(fcs.fcs_id)
            q = float(self.fcs_queue.get(fid, 0))
            cap = max(1.0, float(fcs.capacity))
            risk = (q + fut_arr_by_fid.get(fid, 0.0)) / cap
            risk_by_fid[fid] = float(risk)
        return risk_by_fid, fut_arr_by_fid

    def _build_observation_ppo16(self, mcs_id: int) -> np.ndarray:
        mcs = self.mcs_by_id[mcs_id]
        loc = mcs.location
        x_norm, y_norm = self._norm_location_xy(loc)

        phase = 2.0 * np.pi * (float(self.current_step) / max(1.0, float(self.total_steps)))
        t_sin = float(np.sin(phase))
        t_cos = float(np.cos(phase))

        idle = 0.0 if mcs.busy else 1.0

        radius_km = float(self.config.get("ppo_obs_radius_km", self.config.get("mcs_service_radius_km", 3.0)))
        radius_km = max(1e-6, radius_km)
        step_minutes = float(self.config.get("sim_step_minutes", 5))
        timeout_minutes = float(self.config.get("ev_request_timeout_minutes", 20))

        local_req: List[dict] = []
        nearest_req_dist = radius_km
        if loc is not None:
            for req in self.pending_ev_requests:
                req_loc = req.get("location")
                if req_loc is None:
                    continue
                dist = haversine_distance(loc, req_loc)
                nearest_req_dist = min(nearest_req_dist, float(dist))
                if dist <= radius_km:
                    local_req.append(req)

        req_cnt = float(len(local_req))
        req_cnt_norm = req_cnt / max(1.0, float(self.config.get("ppo_obs_req_norm", 20.0)))
        nearest_req_norm = float(np.clip(nearest_req_dist / radius_km, 0.0, 2.0))

        if local_req:
            waits = [max(0.0, float(self.current_step - int(r.get("step", self.current_step)))) * step_minutes for r in local_req]
            avg_wait_min = float(np.mean(waits))
        else:
            avg_wait_min = 0.0
        avg_wait_norm = float(np.clip(avg_wait_min / max(step_minutes, timeout_minutes), 0.0, 2.0))

        local_mcs = 0
        if loc is not None:
            for other in self.mcs_list:
                if other.mcs_id == mcs.mcs_id or other.location is None:
                    continue
                if haversine_distance(loc, other.location) <= radius_km:
                    local_mcs += 1

        future_horizon = int(self.config.get("ppo_future_horizon_steps", 12))
        risk_by_fid, fut_arr_by_fid = self._fcs_risk_metrics(future_horizon=future_horizon)
        risk_vals = np.array([risk_by_fid.get(f.fcs_id, 0.0) for f in self.fcs_list], dtype=np.float32)

        high_thr = float(self.config.get("ppo_fcs_high_risk_threshold", 1.0))
        high_risk_cnt = float(np.sum(risk_vals >= high_thr))
        high_risk_cnt_norm = high_risk_cnt / max(1.0, float(len(self.fcs_list)))

        max_fcs_risk = float(np.max(risk_vals)) if risk_vals.size > 0 else 0.0
        max_fcs_risk_norm = float(np.clip(max_fcs_risk / max(1.0, high_thr), 0.0, 3.0))

        nearest_fcs_queue_norm = 0.0
        if loc is not None and self.fcs_list:
            d_fcs = [haversine_distance(loc, f.lat_lon) for f in self.fcs_list]
            j = int(np.argmin(d_fcs))
            nf = self.fcs_list[j]
            nearest_fcs_queue_norm = float(self.fcs_queue.get(nf.fcs_id, 0)) / max(1.0, float(nf.capacity))

        region_k = len(self.relocate_hotspots) if self.relocate_hotspots else int(max(1, self.config.get("mcs_relocate_hotspot_k", 15)))
        region_idx = self._infer_region_idx(loc, region_k)

        pred_scale = max(1.0, float(self.config.get("ppo_pred_demand_scale", 50.0)))
        cur_region_future_demand = 0.0
        if self.fcs_region_idx:
            for fid, rid in self.fcs_region_idx.items():
                if int(rid) == int(region_idx):
                    cur_region_future_demand += float(fut_arr_by_fid.get(fid, 0.0))
        cur_region_future_demand_norm = float(np.clip(cur_region_future_demand / pred_scale, 0.0, 3.0))

        cur_region_future_pred = 0.0
        best_region_future_pred = 0.0
        if self._predictive_req_pred.size > 0:
            ridx = int(np.clip(region_idx, 0, self._predictive_req_pred.shape[0] - 1))
            cur_region_future_pred = float(self._predictive_req_pred[ridx])
            best_region_future_pred = float(np.max(self._predictive_req_pred))
        cur_region_future_pred_norm = float(np.clip(cur_region_future_pred / pred_scale, 0.0, 3.0))
        best_region_future_pred_norm = float(np.clip(best_region_future_pred / pred_scale, 0.0, 3.0))

        cur_region_congestion = 0.0
        if self.fcs_region_idx:
            vals = [risk_by_fid[fid] for fid, rid in self.fcs_region_idx.items() if int(rid) == int(region_idx)]
            if vals:
                cur_region_congestion = float(np.mean(vals))
        cur_region_congestion_norm = float(np.clip(cur_region_congestion / max(1.0, high_thr), 0.0, 3.0))

        local_supply = float(local_mcs + (1 if idle > 0.5 else 0))
        local_gap = (req_cnt - local_supply) / max(1.0, req_cnt + local_supply)

        # 16 dims:
        # [x, y, t_sin, t_cos, idle, local_req, avg_wait, nearest_req, cur_region_future_demand,
        #  high_risk_fcs_cnt, cur_region_future_pred, cur_region_congestion, best_region_future_pred,
        #  max_fcs_risk, nearest_fcs_queue, local_supply_demand_gap]
        return np.asarray(
            [
                x_norm,
                y_norm,
                t_sin,
                t_cos,
                idle,
                req_cnt_norm,
                avg_wait_norm,
                nearest_req_norm,
                cur_region_future_demand_norm,
                high_risk_cnt_norm,
                cur_region_future_pred_norm,
                cur_region_congestion_norm,
                best_region_future_pred_norm,
                max_fcs_risk_norm,
                nearest_fcs_queue_norm,
                float(local_gap),
            ],
            dtype=np.float32,
        )

    def _build_observation(self, mcs_id: int) -> np.ndarray:
        return self._build_observation_ppo16(mcs_id)

    def get_agent_observations(self) -> Dict[str, np.ndarray]:
        return {agent: self._build_observation(self.agent_to_mcs_id[agent]) for agent in self.agents}

    def _build_rewards(
        self,
        decisions: Dict[str, int],
        events: List[dict],
        timeout_events: List[dict],
        risk_by_fid: Dict[int, float],
        fut_arr_by_fid: Dict[int, float],
    ) -> Dict[str, float]:
        cfg = self.config
        service_reward = float(cfg.get("reward_service_reward", 10.0))
        waiting_penalty = float(cfg.get("reward_waiting_penalty", 0.1))
        serve_wait_penalty = float(cfg.get("reward_serve_wait_penalty", 0.0))
        empty_drive_penalty = float(cfg.get("reward_empty_drive_penalty", 0.05))
        fcs_overload_penalty = float(cfg.get("reward_fcs_overload_penalty", 0.2))
        crowd_penalty = float(cfg.get("reward_crowd_penalty", 0.05))
        invalid_action_penalty = float(cfg.get("reward_invalid_action_penalty", 2.0))
        timeout_penalty = float(cfg.get("reward_timeout_penalty", 0.0))
        timeout_wait_penalty = float(cfg.get("reward_timeout_wait_penalty", 0.0))
        pending_count_penalty = float(cfg.get("reward_pending_count_penalty", 0.0))
        fast_service_bonus = float(cfg.get("reward_fast_service_bonus", 0.0))

        shape_relocate_scale = float(cfg.get("reward_shape_relocate_scale", 0.3))
        shape_reinforce_scale = float(cfg.get("reward_shape_reinforce_scale", 0.5))
        shape_stay_scale = float(cfg.get("reward_shape_stay_scale", 0.2))
        shape_clip = float(cfg.get("reward_shape_clip", 0.5))
        shape_value_cong_w = float(cfg.get("reward_shape_value_congestion_weight", 0.5))
        shape_stay_demand_thr = float(cfg.get("reward_shape_stay_demand_thr", 0.6))
        shape_stay_cong_thr = float(cfg.get("reward_shape_stay_cong_thr", 0.8))

        pred_scale = max(1.0, float(cfg.get("ppo_pred_demand_scale", 50.0)))
        high_thr = max(1e-6, float(cfg.get("ppo_fcs_high_risk_threshold", 1.0)))
        timeout_steps = max(
            1.0,
            float(cfg.get("ev_request_timeout_minutes", 30.0)) / max(1e-6, float(cfg.get("sim_step_minutes", 5.0))),
        )
        clip_abs = float(cfg.get("reward_clip_abs", 0.0))

        region_k = len(self.relocate_hotspots)
        if region_k <= 0:
            region_k = int(max(1, cfg.get("mcs_relocate_hotspot_k", 15)))
        region_demand = np.zeros((region_k,), dtype=np.float32)
        region_cong_sum = np.zeros((region_k,), dtype=np.float32)
        region_cong_cnt = np.zeros((region_k,), dtype=np.float32)

        for fid, risk in risk_by_fid.items():
            rid = int(np.clip(self.fcs_region_idx.get(int(fid), 0), 0, region_k - 1))
            region_cong_sum[rid] += float(risk)
            region_cong_cnt[rid] += 1.0
            region_demand[rid] += float(fut_arr_by_fid.get(int(fid), 0.0))

        if self._predictive_req_pred.size > 0:
            n = min(region_k, int(self._predictive_req_pred.shape[0]))
            region_demand[:n] = self._predictive_req_pred[:n]

        region_demand_norm = np.clip(region_demand / pred_scale, 0.0, 3.0)
        region_cong = np.divide(
            region_cong_sum,
            np.maximum(region_cong_cnt, 1.0),
            out=np.zeros_like(region_cong_sum),
            where=region_cong_cnt > 0,
        )
        region_cong_norm = np.clip(region_cong / high_thr, 0.0, 3.0)
        region_value = region_demand_norm - shape_value_cong_w * region_cong_norm

        rewards = {agent: 0.0 for agent in self.agents}
        event_count_by_agent: Dict[str, int] = defaultdict(int)

        for e in events:
            agent = str(e["agent"])
            action = str(e.get("action", ""))
            event_count_by_agent[agent] += 1

            rewards[agent] -= empty_drive_penalty * float(e.get("distance_km", 0.0))

            if action == "serve_request":
                rewards[agent] += service_reward
                wait_steps = float(max(0.0, float(e.get("wait_steps", 0.0))))
                if fast_service_bonus > 0.0:
                    rewards[agent] += fast_service_bonus * float(np.clip(1.0 - wait_steps / timeout_steps, 0.0, 1.0))
                if serve_wait_penalty > 0.0:
                    rewards[agent] -= serve_wait_penalty * wait_steps
            elif action == "reinforce_fcs":
                fid = int(e.get("target_fcs", -1))
                risk = float(risk_by_fid.get(fid, 0.0))
                rewards[agent] += float(np.clip(shape_reinforce_scale * risk, 0.0, shape_clip))
            elif action == "relocate":
                src = int(np.clip(int(e.get("source_region", 0)), 0, region_k - 1))
                dst = int(np.clip(int(e.get("target_region", src)), 0, region_k - 1))
                delta = float(region_value[dst] - region_value[src])
                if delta > 0:
                    rewards[agent] += float(np.clip(shape_relocate_scale * delta, 0.0, shape_clip))

        for agent, action in decisions.items():
            if int(action) != 4 and event_count_by_agent.get(agent, 0) == 0:
                rewards[agent] -= invalid_action_penalty

        pending_count = len(self.pending_ev_requests)
        avg_wait_steps = 0.0
        if pending_count > 0:
            wait_sum = 0.0
            for req in self.pending_ev_requests:
                req_step = int(req.get("step", self.current_step))
                wait_sum += float(max(0, self.current_step - req_step))
            avg_wait_steps = wait_sum / float(pending_count)
        # Shared penalties are normalized by agent count to avoid scale explosion with many MCS agents.
        agent_count = max(1, len(self.agents))
        shared_wait_penalty = (waiting_penalty * avg_wait_steps) / float(agent_count)
        for agent in rewards:
            rewards[agent] -= shared_wait_penalty

        if pending_count_penalty > 0.0 and pending_count > 0:
            shared_pending_penalty = (pending_count_penalty * float(pending_count)) / float(agent_count)
            for agent in rewards:
                rewards[agent] -= shared_pending_penalty

        overload = 0.0
        if risk_by_fid:
            overload = float(np.mean([max(0.0, float(v) - 1.0) for v in risk_by_fid.values()]))
        shared_overload_penalty = (fcs_overload_penalty * overload) / float(agent_count)
        for agent in rewards:
            rewards[agent] -= shared_overload_penalty

        if timeout_events and (timeout_penalty > 0.0 or timeout_wait_penalty > 0.0):
            timeout_total = 0.0
            for te in timeout_events:
                timeout_total += timeout_penalty
                if timeout_wait_penalty > 0.0:
                    timeout_total += timeout_wait_penalty * float(max(0.0, float(te.get("wait_steps", 0.0))))
            shared_timeout_penalty = timeout_total / float(agent_count)
            for agent in rewards:
                rewards[agent] -= shared_timeout_penalty

        radius_km = max(1e-6, float(cfg.get("ppo_obs_radius_km", cfg.get("mcs_service_radius_km", 3.0))))
        total_mcs = len(self.mcs_list)
        for agent in rewards:
            mcs = self.mcs_by_id[self.agent_to_mcs_id[agent]]
            loc = mcs.location
            if loc is None or total_mcs <= 1:
                continue
            local_mcs = 0
            for other in self.mcs_list:
                if other.mcs_id == mcs.mcs_id or other.location is None:
                    continue
                if haversine_distance(loc, other.location) <= radius_km:
                    local_mcs += 1
            crowd_ratio = float(local_mcs) / float(max(1, total_mcs - 1))
            rewards[agent] -= crowd_penalty * crowd_ratio

            if int(decisions.get(agent, 4)) == 4:
                rid = int(np.clip(self._infer_region_idx(loc, region_k), 0, region_k - 1))
                d = float(region_demand_norm[rid])
                c = float(region_cong_norm[rid])
                if d >= shape_stay_demand_thr and c <= shape_stay_cong_thr:
                    cong_factor = max(0.0, 1.0 - c / max(shape_stay_cong_thr, 1e-6))
                    stay_shape = shape_stay_scale * d * cong_factor
                    rewards[agent] += float(np.clip(stay_shape, 0.0, shape_clip))

        if clip_abs > 0:
            for agent in rewards:
                rewards[agent] = float(np.clip(rewards[agent], -clip_abs, clip_abs))
        return rewards

    def step(self, mcs_actions: ActionInput = None, mcs_action: int = 4) -> StepResult:
        if self.current_step >= self.total_steps:
            return StepResult(
                step=self.current_step,
                active_ev_count=len(self.active_ev_ids),
                new_request_count=0,
                requests=[],
                fcs_arrivals={},
                fcs_states={},
                mcs_decisions={agent: 4 for agent in self.agents},
                mcs_events=[],
                timeout_events=[],
                agent_rewards={agent: 0.0 for agent in self.agents},
                done=True,
            )

        for ev_id in self.ev_ids_by_step.get(self.current_step, []):
            self._create_ev(ev_id)

        requests: List[dict] = []
        for ev_id in list(self.active_ev_ids):
            req = self._ev_step(ev_id)
            if req is not None:
                requests.append(req)
                if req.get("service_mode") == "mcs":
                    self.pending_ev_requests.append(req)

        timeout_events = self._drop_timeout_pending_requests()
        self._refresh_mcs_state()
        risk_by_fid, fut_arr_by_fid = self._fcs_risk_metrics(
            future_horizon=int(self.config.get("ppo_future_horizon_steps", 12))
        )
        decisions = self._normalize_parallel_actions(mcs_actions=mcs_actions, default_action=mcs_action)
        chosen = self._actions_to_candidates(decisions)

        mcs_events: List[dict] = []
        mcs_events.extend(self._action_reinforce_fcs(chosen[1]))
        mcs_events.extend(self._action_relocate(chosen[2]))
        mcs_events.extend(self._action_service_ev_requests(chosen[3]))
        fcs_states = self._update_fcs_runtime(mcs_events)
        self._update_predictive_summary(requests=requests, fcs_states=fcs_states)

        rewards = self._build_rewards(
            decisions=decisions,
            events=mcs_events,
            timeout_events=timeout_events,
            risk_by_fid=risk_by_fid,
            fut_arr_by_fid=fut_arr_by_fid,
        )

        result = StepResult(
            step=self.current_step,
            active_ev_count=len(self.active_ev_ids),
            new_request_count=len(requests),
            requests=requests,
            fcs_arrivals=self.fcs_arrival_schedule.get(self.current_step, {}),
            fcs_states=fcs_states,
            mcs_decisions=decisions,
            mcs_events=mcs_events,
            timeout_events=timeout_events,
            agent_rewards=rewards,
            done=(self.current_step + 1 >= self.total_steps),
        )
        self.current_step += 1
        return result

    def step_parallel(self, mcs_actions: Dict[Union[int, str], int]) -> StepResult:
        return self.step(mcs_actions=mcs_actions, mcs_action=4)

    def run(self, max_steps: Optional[int] = None, mcs_action: int = 4) -> List[StepResult]:
        outputs: List[StepResult] = []
        limit = self.total_steps if max_steps is None else min(max_steps, self.total_steps)
        while self.current_step < limit:
            outputs.append(self.step(mcs_action=mcs_action))
        return outputs
