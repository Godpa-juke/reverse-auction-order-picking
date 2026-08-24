"""Observation strategy helpers for the warehouse engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rware.engine.definitions import ObservationType


class ObservationStrategy(ABC):
    """Strategy interface for configuring and producing observations."""

    @abstractmethod
    def configure(self, engine) -> None:
        """Prepare the engine's observation space."""

    @abstractmethod
    def observe(self, engine, agent) -> Any:
        """Return the observation for ``agent``."""


class DictObservationStrategy(ObservationStrategy):
    """Dictionary-based observation layout."""

    def configure(self, engine) -> None:
        engine._use_slow_obs()

    def observe(self, engine, agent) -> Any:
        return engine._make_obs(agent)


class FlattenedObservationStrategy(ObservationStrategy):
    """Flattened vector observation layout."""

    def configure(self, engine) -> None:
        engine._use_slow_obs()
        engine._use_fast_obs()

    def observe(self, engine, agent) -> Any:
        return engine._make_obs(agent)


class ImageObservationStrategy(ObservationStrategy):
    """Image-based observation layout."""

    def configure(self, engine) -> None:
        engine._use_image_obs(
            engine.image_observation_layers,
            engine.image_observation_directional,
        )

    def observe(self, engine, agent) -> Any:
        return engine._make_obs(agent)


def build_observation_strategy(observation_type: ObservationType) -> ObservationStrategy:
    """Factory for observation strategies."""

    if observation_type == ObservationType.IMAGE:
        return ImageObservationStrategy()
    if observation_type == ObservationType.FLATTENED:
        return FlattenedObservationStrategy()
    return DictObservationStrategy()

