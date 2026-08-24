"""Conversion between simulator ticks and wall-clock seconds.

The engine advances one tick per rendered frame and every duration inside the
simulator (``loading_timer``, ``waiting_time``, ``agent_timer``, movement) is
counted in ticks. Reports, however, are written in seconds: ``WriteLog``
multiplies ``internal_timer`` by ``config.tick_per_time`` before formatting
``Time`` and ``Sec/Pick``.

With the shipped default ``TICKPERTIME = 1.0`` one tick equals one second, so
the two units happen to coincide numerically. That coincidence is easy to rely
on by accident. Anything expressed in seconds (cost weights, risk thresholds,
service-time statistics) must go through the helpers below so that changing the
tick scale stays a one-line change rather than an audit of every call site.

Movement is one grid cell per tick, so ``seconds_per_tick`` also fixes how long
an agent takes to traverse a cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from rware.core.config import SimulationConfig


DEFAULT_SECONDS_PER_TICK = 1.0


def seconds_per_tick(config: Optional["SimulationConfig"] = None) -> float:
    """Wall-clock seconds represented by one simulator tick."""

    value = getattr(config, "tick_per_time", DEFAULT_SECONDS_PER_TICK)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SECONDS_PER_TICK
    return value if value > 0 else DEFAULT_SECONDS_PER_TICK


def ticks_to_seconds(ticks: float, config: Optional["SimulationConfig"] = None) -> float:
    """Convert a tick count to seconds."""

    return float(ticks) * seconds_per_tick(config)


def seconds_to_ticks(seconds: float, config: Optional["SimulationConfig"] = None) -> int:
    """Convert seconds to whole ticks, rounding to the nearest tick."""

    return int(round(float(seconds) / seconds_per_tick(config)))
