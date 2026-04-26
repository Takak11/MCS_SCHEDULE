from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import config

LatLon = Tuple[float, float]


cfg = config.CONFIG


@dataclass
class EV:
    ev_id: int
    appear_step: int
    initial_soc_kwh: float
    current_soc_kwh: float = field(init=False)
    current_location: Optional[LatLon] = field(default=None)
    active: bool = field(default=False)
    request_sent: bool = field(default=False)
    request_step: Optional[int] = field(default=None)
    request_charge_minutes: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        self.current_soc_kwh = float(self.initial_soc_kwh)

    @property
    def soc_ratio(self) -> float:
        return self.current_soc_kwh / cfg.get("ev_battery_capacity_kwh")

    @property
    def request_threshold_kwh(self) -> float:
        return cfg.get("ev_request_threshold") * cfg.get("ev_battery_capacity_kwh")

    @property
    def target_soc_kwh(self) -> float:
        return cfg.get("ev_target_soc") * cfg.get("ev_battery_capacity_kwh")

    @property
    def required_charge_kwh(self) -> float:
        return max(0.0, self.target_soc_kwh - self.current_soc_kwh)

    def activate(self, location: Optional[LatLon]) -> None:
        self.active = True
        if location is not None:
            self.current_location = location

    def update_location(self, location: LatLon) -> None:
        self.current_location = location

    def consume_by_distance(self, distance_km: float) -> float:
        used_kwh = max(distance_km, 0.0) * cfg.get("ev_consumption_rate")
        self.current_soc_kwh = max(0.0, self.current_soc_kwh - used_kwh)
        return used_kwh

    def should_request_charge(self) -> bool:
        return self.active and (not self.request_sent) and (self.current_soc_kwh <= self.request_threshold_kwh)

    def estimate_charge_minutes(self) -> float:
        kwh_per_step = cfg.get("ev_charge_soc_per_step") * cfg.get("ev_battery_capacity_kwh")
        if kwh_per_step <= 0:
            return 0.0
        step_minutes = float(cfg.get("sim_step_minutes", 5))
        need_steps = self.required_charge_kwh / kwh_per_step
        need_minutes = need_steps * step_minutes
        return min(need_minutes, float(cfg.get("max_charge_minutes", 20)))

    def mark_request(self, step: int) -> None:
        self.request_sent = True
        self.request_step = step
        self.request_charge_minutes = self.estimate_charge_minutes()


@dataclass
class MCS:
    mcs_id: int
    speed_km_per_step: float
    service_radius_km: float
    price_per_kwh: float
    cost_per_km: float
    location: Optional[LatLon] = None
    busy: bool = False


@dataclass
class FCS:
    fcs_id: int
    # Store as (longitude, latitude) because config uses this order.
    lon_lat: Tuple[float, float]
    capacity: int
    occupied: int = 0

    @property
    def lat_lon(self) -> LatLon:
        lon, lat = self.lon_lat
        return (lat, lon)

    def can_accept(self) -> bool:
        return self.occupied < self.capacity
