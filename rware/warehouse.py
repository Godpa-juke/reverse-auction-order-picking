"""Compatibility layer for the legacy `rware.warehouse` module.

Phase 1 of the refactor extracts the core engine logic into
`rware.engine.warehouse_engine` and exposes a Gym wrapper in `rware.env.gym_env`.
This module simply re-exports the new entry points and keeps a backwards
compatible CLI until it is formally removed."""

from __future__ import annotations

import warnings
from typing import Any

from rware.env.gym_env import WarehouseGymEnv, Warehouse, deprecated_warehouse_entry
from rware.core import Direction, Action, State
from rware.engine.definitions import RewardType, ObservationType, ObserationType, ImageLayer

__all__ = [
    "WarehouseGymEnv",
    "Warehouse",
    "deprecated_warehouse_entry",
    "Direction",
    "Action",
    "State",
    "RewardType",
    "ObservationType",
    "ObserationType",
    "ImageLayer",
]


def main(*args: Any, **kwargs: Any) -> None:
    """Deprecated CLI shim for `python -m rware.warehouse`."""
    warnings.warn(
        "`python -m rware.warehouse` is deprecated. Use `python -m rware.apps.simulator` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from rware.apps import simulator

    if args or kwargs:
        simulator.main(*args, **kwargs)
    else:
        simulator.launch_from_config()


if __name__ == "__main__":  # pragma: no cover
    main()
