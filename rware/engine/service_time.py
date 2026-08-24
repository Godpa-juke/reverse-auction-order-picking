"""Stochastic human service-time model and interaction logging.

Why this exists
---------------
In the shipped simulator the time a worker spends picking at a rack is
``order_sku_cnt * sku_per_picking_time`` ticks — a deterministic function of the
task alone. Every worker takes exactly the same time and there is no dispersion,
so the median and the 90th percentile of service time coincide and a risk-aware
allocator degenerates to the static one. Learning a service-time *distribution*
only carries signal once the simulator produces one.

This module adds that dispersion in the shape described by the research plan:
worker heterogeneity (fast/normal/slow) plus per-interaction noise plus rare
heavy-tail disruptions. It stays off by default so existing baseline numbers
remain reproducible.

Determinism
-----------
A draw is a pure function of ``(seed, human_id, robot_id, picking_seq, salt)``.
It does not depend on *when* it is queried or how many times, which matters for
three reasons:

* the auction evaluates many candidate pairs before committing to one, and
  querying a candidate must not consume randomness or change the outcome;
* a re-auction may hand the task to a different worker, and the first worker's
  draw must not perturb the second's;
* the oracle baseline needs to read the realised duration *before* the
  assignment is made.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional

# Worker speed multipliers, cycled over the human roster. Mirrors the
# fast (0.8) / normal (1.0) / slow (1.3) split from the research plan.
_SPEED_PROFILE = (0.80, 0.90, 1.00, 1.00, 1.10, 1.20, 1.30, 1.00)

_NORMAL = NormalDist()


@dataclass(frozen=True)
class VariabilityProfile:
    """Dispersion knobs for one variability level."""

    sigma: float  # lognormal shape of the per-interaction multiplier
    disruption_prob: float  # probability of a heavy-tail delay
    disruption_low: float  # multiplicative delay range when one fires
    disruption_high: float
    heterogeneous: bool  # whether workers differ in speed


PROFILES: Dict[str, VariabilityProfile] = {
    "off": VariabilityProfile(0.0, 0.0, 1.0, 1.0, False),
    "low": VariabilityProfile(0.15, 0.01, 1.5, 2.5, True),
    "medium": VariabilityProfile(0.30, 0.03, 2.0, 3.5, True),
    "high": VariabilityProfile(0.50, 0.08, 2.5, 5.0, True),
}


def _uniform(seed: int, human_id: int, robot_id: int, picking_seq: int, salt: int) -> float:
    """Deterministic uniform draw in [0, 1) for one (worker, task, salt) key."""

    payload = struct.pack("<qqqqq", int(seed), int(human_id), int(robot_id), int(picking_seq), int(salt))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, "big") >> 11) / float(1 << 53)


class ServiceTimeModel:
    """Draws the realised picking duration for a (worker, task) pair."""

    def __init__(self, profile: str = "off", seed: int = 0, base_ticks_per_sku: int = 7) -> None:
        self.profile_name = (profile or "off").lower()
        self.profile = PROFILES.get(self.profile_name, PROFILES["off"])
        self.seed = int(seed)
        self.base_ticks_per_sku = int(base_ticks_per_sku)

    @property
    def enabled(self) -> bool:
        return self.profile_name != "off"

    def speed_multiplier(self, human_index: int) -> float:
        """Per-worker speed factor; 1.0 for every worker when disabled."""

        if not self.profile.heterogeneous:
            return 1.0
        return _SPEED_PROFILE[human_index % len(_SPEED_PROFILE)]

    def base_ticks(self, sku_count: int) -> int:
        return max(1, int(sku_count) * self.base_ticks_per_sku)

    def service_ticks(
        self,
        human_id: int,
        human_index: int,
        robot_id: int,
        picking_seq: int,
        sku_count: int,
    ) -> int:
        """Realised service duration in ticks for this exact task."""

        base = self.base_ticks(sku_count)
        if not self.enabled:
            return base

        u_noise = _uniform(self.seed, human_id, robot_id, picking_seq, 0)
        # Clamp away from the open interval ends so inv_cdf stays finite.
        u_noise = min(max(u_noise, 1e-6), 1.0 - 1e-6)
        # exp(sigma * z) has median 1, so the worker speed stays the median.
        multiplier = self.speed_multiplier(human_index) * math.exp(
            self.profile.sigma * _NORMAL.inv_cdf(u_noise)
        )

        if self.profile.disruption_prob > 0.0:
            u_hit = _uniform(self.seed, human_id, robot_id, picking_seq, 1)
            if u_hit < self.profile.disruption_prob:
                u_size = _uniform(self.seed, human_id, robot_id, picking_seq, 2)
                span = self.profile.disruption_high - self.profile.disruption_low
                multiplier *= self.profile.disruption_low + u_size * span

        return max(1, int(round(base * multiplier)))


@dataclass
class InteractionRecord:
    """One completed human-robot picking interaction."""

    features: Dict[str, Any]
    duration_ticks: int = 0
    human_id: int = 0
    robot_id: int = 0
    start_tick: int = 0
    end_tick: int = 0


@dataclass
class InteractionLog:
    """Completed interactions, in completion order."""

    records: List[InteractionRecord] = field(default_factory=list)

    def append(self, record: InteractionRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def feature_rows(self) -> List[Dict[str, Any]]:
        return [record.features for record in self.records]

    def targets(self) -> List[int]:
        return [record.duration_ticks for record in self.records]

    def to_csv(self, path: str) -> None:
        """Write the log for offline model analysis."""

        import csv

        if not self.records:
            return
        columns = sorted(self.records[0].features.keys())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["human_id", "robot_id", "start_tick", "end_tick", "duration_ticks"] + columns
            )
            for record in self.records:
                writer.writerow(
                    [
                        record.human_id,
                        record.robot_id,
                        record.start_tick,
                        record.end_tick,
                        record.duration_ticks,
                    ]
                    + [record.features.get(name, "") for name in columns]
                )


def build_service_time_model(config: Optional[Any]) -> ServiceTimeModel:
    """Construct the model described by ``config``."""

    return ServiceTimeModel(
        profile=getattr(config, "service_time_variability", "off"),
        seed=int(getattr(config, "service_time_seed", 0) or 0),
        base_ticks_per_sku=int(getattr(config, "sku_per_picking_time", 7) or 7),
    )
