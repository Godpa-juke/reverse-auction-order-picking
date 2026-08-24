"""유통 창고 시뮬레이션에 사용되는 코스트 맵 로더."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent
LEGACY_FALLBACKS: Iterable[Path] = (
    DATA_ROOT.parent.parent / "source" / "site_a" / "path_data.npy",
)


def _iter_candidates(name: str) -> List[Path]:
    base = DATA_ROOT
    candidates = []
    if name.endswith(".npy"):
        candidates.append(base / name)
    else:
        candidates.append(base / name / "path_data.npy")
        candidates.append(base / f"{name}.npy")

    candidates.extend(LEGACY_FALLBACKS)
    return candidates


def get_cost_map_path(name: str = "default") -> Path:
    """코스트 맵 파일 경로를 반환한다."""
    candidates = _iter_candidates(name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cost map '{name}' not found. Checked: {candidates}".format(
            name=name,
            candidates=[str(path) for path in candidates],
        )
    )


def load_cost_map(name: str = "default") -> np.ndarray:
    """코스트 맵을 numpy 배열로 읽어온다."""
    path = get_cost_map_path(name)
    return np.load(path)


__all__ = ["get_cost_map_path", "load_cost_map"]
