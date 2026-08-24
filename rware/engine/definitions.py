"""Public enums and protocol definitions for the warehouse engine."""

from __future__ import annotations

from enum import Enum


class RewardType(Enum):
    """Supported reward aggregation modes."""

    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    """Observation layout presented to Gym."""

    DICT = 0
    FLATTENED = 1
    IMAGE = 2


class ImageLayer(Enum):
    """Available layers when building image-style observations."""

    SHELVES = 0
    REQUESTS = 1
    AGENTS = 2
    AGENT_DIRECTION = 3
    AGENT_LOAD = 4
    GOALS = 5
    ACCESSIBLE = 6


# Backwards compatibility -------------------------------------------------------

# Legacy typo kept for downstream modules that still import the old name.
ObserationType = ObservationType
