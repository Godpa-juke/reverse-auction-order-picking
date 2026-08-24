"""2-layer map DSL: [base] cell types + [overlay] walls/direction constraints.

Overlay chars:
  .    no constraint
  0-f  allowed-direction bitmask (1=UP, 2=RIGHT, 4=DOWN, 8=LEFT)
  #    transparent wall (robot-only)
  W    solid wall (all agents)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 1, 2, 4, 8
ALL_DIRS = DIR_UP | DIR_RIGHT | DIR_DOWN | DIR_LEFT

WALL_NONE, WALL_TRANSPARENT, WALL_SOLID = 0, 1, 2

# base cells that must stay reachable (load/unload/wait stations)
_STATION_CHARS = "bgw"


@dataclass
class MapLayout:
    base: str
    walls: np.ndarray        # (h, w) uint8: WALL_* codes
    allowed_dirs: np.ndarray  # (h, w) uint8: direction bitmask


def move_bit(dx: int, dy: int) -> int:
    """Direction bit for a unit move; 0 for diagonal or no move."""
    if dx == 0 and dy == -1:
        return DIR_UP
    if dx == 1 and dy == 0:
        return DIR_RIGHT
    if dx == 0 and dy == 1:
        return DIR_DOWN
    if dx == -1 and dy == 0:
        return DIR_LEFT
    return 0


def _clean_lines(block: str) -> list[str]:
    lines = [line.replace(" ", "") for line in block.strip().split("\n")]
    return [line for line in lines if line]


def parse_map_text(text: str) -> MapLayout:
    stripped = text.strip()
    if "[base]" in stripped:
        base_part = stripped.split("[base]", 1)[1]
        if "[overlay]" in base_part:
            base_part, overlay_part = base_part.split("[overlay]", 1)
        else:
            overlay_part = None
    else:
        base_part, overlay_part = stripped, None

    base_lines = _clean_lines(base_part)
    if not base_lines:
        raise ValueError("map has no base layer")
    width = len(base_lines[0])
    for i, line in enumerate(base_lines):
        if len(line) != width:
            raise ValueError(f"base row {i} width {len(line)} != {width}")
    height = len(base_lines)

    walls = np.zeros((height, width), dtype=np.uint8)
    allowed = np.full((height, width), ALL_DIRS, dtype=np.uint8)

    if overlay_part is not None:
        overlay_lines = _clean_lines(overlay_part)
        if len(overlay_lines) != height or any(
            len(line) != width for line in overlay_lines
        ):
            raise ValueError(
                f"overlay size {len(overlay_lines)}x"
                f"{len(overlay_lines[0]) if overlay_lines else 0}"
                f" != base size {height}x{width}"
            )
        for y, line in enumerate(overlay_lines):
            for x, ch in enumerate(line):
                if ch == ".":
                    continue
                if ch == "#":
                    walls[y, x] = WALL_TRANSPARENT
                elif ch == "W":
                    walls[y, x] = WALL_SOLID
                elif ch in "0123456789abcdef":
                    allowed[y, x] = int(ch, 16)
                else:
                    raise ValueError(f"overlay char {ch!r} at ({x},{y})")
                if walls[y, x] and base_lines[y][x].lower() in _STATION_CHARS:
                    raise ValueError(
                        f"wall over station cell {base_lines[y][x]!r} at ({x},{y})"
                    )

    return MapLayout(base="\n".join(base_lines), walls=walls, allowed_dirs=allowed)


def load_map_file(path: str) -> MapLayout:
    with open(path, "r", encoding="utf-8") as f:
        return parse_map_text(f.read())
