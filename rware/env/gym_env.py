"""Gym-compatible wrapper for the robotic warehouse engine."""

import warnings
from typing import Any

import gymnasium as gym

from rware.engine.warehouse_engine import WarehouseEngine


class WarehouseGymEnv(gym.Env):
    """Thin Gym wrapper that delegates all logic to :class:`WarehouseEngine`."""

    metadata = WarehouseEngine.metadata
    _local_attrs = {"engine", "action_space", "observation_space", "reward_range", "spec"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        spec = kwargs.pop("spec", None)
        self.engine = WarehouseEngine(*args, **kwargs)
        self.action_space = self.engine.action_space
        self.observation_space = self.engine.observation_space
        self.reward_range = getattr(self.engine, "reward_range", None)
        # Gym API attributes that aren't handled by the engine
        self.spec = spec

    def reset(self, *args: Any, **kwargs: Any):
        return self.engine.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any):
        return self.engine.step(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any):
        return self.engine.render(*args, **kwargs)

    def close(self) -> None:
        self.engine.close()

    def __getattr__(self, item: str):
        try:
            return getattr(self.engine, item)
        except AttributeError as exc:  # pragma: no cover - passthrough helper
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._local_attrs or key.startswith("_"):
            object.__setattr__(self, key, value)
            return

        engine = self.__dict__.get("engine")
        if engine is None:
            object.__setattr__(self, key, value)
            return

        setattr(engine, key, value)

    # --- extended adapter helpers -----------------------------------------

    def snapshot_state(self):
        """Expose the engine's immutable world state."""

        return self.engine.world_state

    def poll_events(self):
        """Drain the engine's event queue."""

        return self.engine.poll_events()


# Backwards compatibility alias
Warehouse = WarehouseGymEnv


def deprecated_warehouse_entry(*args: Any, **kwargs: Any) -> WarehouseGymEnv:
    warnings.warn(
        "`rware.warehouse.Warehouse` is now provided by WarehouseGymEnv. "
        "Import from rware.env.gym_env or use the alias while it exists.",
        DeprecationWarning,
        stacklevel=2,
    )
    return WarehouseGymEnv(*args, **kwargs)
