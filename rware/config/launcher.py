"""Runtime helpers for simulator launch configuration."""

from __future__ import annotations

import os
from typing import Iterable, List, Union

from .defaults import DEFAULT_HUMAN_ASSIGNMENT_STRATEGY

_human_assignment_strategy: Union[str, Iterable[str]] = DEFAULT_HUMAN_ASSIGNMENT_STRATEGY


def _coerce_strategy(value: Union[str, Iterable[str]]) -> Union[str, List[str]]:
    if isinstance(value, (list, tuple, set)):
        cleaned = [str(item).strip().lower() for item in value if str(item).strip()]
        return cleaned if cleaned else ["nearest_idle"]

    strategy = str(value).strip().lower()
    return strategy or "nearest_idle"


def configure_human_assignment_strategy(value: Union[str, Iterable[str]]) -> None:
    """Programmatically override the human assignment strategy."""

    global _human_assignment_strategy
    _human_assignment_strategy = value


def resolve_human_assignment_strategy(as_list: bool = False) -> Union[str, List[str]]:
    """Return the configured human assignment strategy."""

    env_value = os.environ.get("RWARE_HUMAN_ASSIGNMENT_STRATEGY")
    if env_value:
        parts = [part.strip().lower() for part in env_value.split(",") if part.strip()]
        if as_list:
            return parts if parts else ["nearest_idle"]
        return parts[0] if parts else "nearest_idle"

    coerced = _coerce_strategy(_human_assignment_strategy)
    if as_list:
        return coerced if isinstance(coerced, list) else [coerced]
    if isinstance(coerced, list):
        return coerced[0] if coerced else "nearest_idle"
    return coerced
