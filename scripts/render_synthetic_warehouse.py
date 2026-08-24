#!/usr/bin/env python3
"""Render a deterministic 45-degree warehouse rollout from the public policies.

The assignment shown on screen is computed by the repository's actual cost-matrix
and Bertsekas auction implementation.  Only the geometry and motion schedule are
synthetic; no private warehouse inputs are required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rware.core import State
from rware.engine.human_assignment import (
    HumanSnapshot,
    RobotSnapshot,
    _solve_assignment_by_auction,
    get_human_assignment_strategy,
)

WIDTH, HEIGHT, FPS, FRAMES = 1280, 720, 30, 180
GRID_W, GRID_H = 18, 12
SCALE = 31
ORIGIN_X, ORIGIN_Y = 640, 115
BG = (11, 16, 27)
FLOOR_A, FLOOR_B = (28, 38, 55), (31, 43, 62)
GRID = (58, 73, 94)
SHELF_TOP, SHELF_LEFT, SHELF_RIGHT = (77, 92, 123), (43, 54, 78), (54, 66, 94)
ROBOT, HUMAN, LINK = (255, 158, 67), (46, 205, 191), (117, 181, 255)
WHITE, MUTED, GREEN = (238, 244, 255), (151, 164, 184), (84, 220, 145)


def iso(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
    return ORIGIN_X + (x - y) * SCALE, ORIGIN_Y + (x + y) * SCALE * 0.5 - z * SCALE


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def along(points: list[tuple[float, float]], progress: float) -> tuple[float, float]:
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    target = max(0.0, min(1.0, progress)) * total
    for a, b, length in zip(points, points[1:], lengths):
        if target <= length or length == lengths[-1]:
            u = 0.0 if length == 0 else min(1.0, target / length)
            return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        target -= length
    return points[-1]


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_16, FONT_18, FONT_24, FONT_34 = font(16), font(18, True), font(24, True), font(34, True)


def polygon(draw, points, fill, outline=None, width=1):
    points = [(round(x), round(y)) for x, y in points]
    draw.polygon(points, fill=fill)
    if outline:
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def box(draw, x: float, y: float, w: float, d: float, h: float) -> None:
    a, b, c, d0 = iso(x, y, h), iso(x + w, y, h), iso(x + w, y + d, h), iso(x, y + d, h)
    ab, bb, cb, db = iso(x, y, 0), iso(x + w, y, 0), iso(x + w, y + d, 0), iso(x, y + d, 0)
    polygon(draw, [d0, c, cb, db], SHELF_LEFT)
    polygon(draw, [b, c, cb, bb], SHELF_RIGHT)
    polygon(draw, [a, b, c, d0], SHELF_TOP, GRID)


class StaticTracker:
    def ensure_models(self, _backend):
        return None

    def median_human_trip(self):
        return 30.0

    def build_features(self, **kwargs):
        return kwargs

    def predict(self, rows, is_human=False):
        result = []
        for row in rows:
            ticks = float(row.get("static_ticks", row.get("planned_path_len", 0.0)))
            result.append((ticks, ticks))
        return result


def assignments(method: str):
    humans = [
        HumanSnapshot(1, (2, 2), 7, 0, [], State.NOOP),
        HumanSnapshot(2, (2, 9), 3, 0, [], State.NOOP),
    ]
    robot_state = State.ROBOT_PICKING if method == "auction" else State.ROBOT_MOVESPOT
    robots = [
        RobotSnapshot(3, (3, 4), robot_state, 6, 1, 2, 9.0, (14, 2), 101, 13),
        RobotSnapshot(4, (3, 7), robot_state, 2, 2, 1, 6.0, (14, 9), 102, 13),
    ]
    cfg = SimpleNamespace(
        auction_zone_penalty=0.0,
        rendezvous_robot_wait_weight=1.0,
        rendezvous_human_wait_weight=0.6,
        rendezvous_human_travel_weight=1.0,
        rendezvous_risk_weight=0.5,
        dispatch_lead_limit=20,
    )
    grid = np.zeros((3, GRID_H, GRID_W), dtype=np.int8)
    for x in range(6, 12):
        for y in range(4, 8):
            grid[1, y, x] = 1
    tracker = StaticTracker()
    engine = SimpleNamespace(
        config=cfg,
        grid=grid,
        layer_shelfs=1,
        internal_timer=0,
        get_arrival_tracker=lambda: tracker,
    )
    strategy_name = "auction" if method == "auction" else "rv_static"
    strategy = get_human_assignment_strategy(strategy_name)
    costs = strategy.build_cost_matrix(engine, humans, robots, context=None)
    values = [[-cost for cost in row] for row in costs]
    selected = _solve_assignment_by_auction(values, epsilon=0.01, time_limit_s=0.0, max_bid_updates=20_000)
    mapping = {humans[i].id: robots[j].id for i, j in enumerate(selected) if j >= 0}
    return humans, robots, costs, mapping


def draw_agent(draw, position, label, color, robot=False):
    x, y = iso(position[0] + 0.5, position[1] + 0.5, 0.16)
    draw.ellipse((x - 20, y - 9, x + 20, y + 11), fill=(5, 9, 16, 130))
    if robot:
        polygon(draw, [(x - 17, y), (x, y - 10), (x + 17, y), (x, y + 10)], color, WHITE)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(35, 42, 54))
    else:
        draw.ellipse((x - 10, y - 19, x + 10, y + 1), fill=color, outline=WHITE, width=2)
        draw.line((x, y + 1, x, y + 14), fill=color, width=6)
    draw.text((x + 15, y - 25), label, font=FONT_16, fill=WHITE, stroke_width=2, stroke_fill=BG)


def frame(method: str, index: int, mapping, costs) -> Image.Image:
    t = index / (FRAMES - 1)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image, "RGBA")

    for x in range(GRID_W):
        for y in range(GRID_H):
            a, b, c, d0 = iso(x, y), iso(x + 1, y), iso(x + 1, y + 1), iso(x, y + 1)
            polygon(draw, [a, b, c, d0], FLOOR_A if (x + y) % 2 == 0 else FLOOR_B, GRID)
    for x in range(6, 12, 2):
        box(draw, x, 4, 1.5, 3.5, 0.8)

    targets = {3: (14, 2), 4: (14, 9)}
    for rid, target in targets.items():
        x, y = iso(target[0] + 0.5, target[1] + 0.5, 0.02)
        draw.ellipse((x - 17, y - 9, x + 17, y + 9), outline=GREEN, width=4)
        draw.text((x - 7, y + 12), f"P{rid - 2}", font=FONT_16, fill=GREEN)

    robot_paths = {
        3: [(3, 4), (3, 2), (14, 2)],
        4: [(3, 7), (3, 9), (14, 9)],
    }
    human_paths = {
        1: [(2, 2), (14, 2)],
        2: [(2, 9), (14, 9)],
    }
    if method == "auction":
        rp = ease((t - 0.04) / 0.36)
        hp = ease((t - 0.43) / 0.42)
        assigned = t >= 0.43
        phase = "ROBOTS TRAVEL" if t < 0.40 else ("REVERSE AUCTION" if t < 0.50 else "PICKERS DISPATCHED")
    else:
        rp = ease((t - 0.06) / 0.66)
        hp = ease((t - 0.10) / 0.48)
        assigned = t >= 0.10
        phase = "EARLY RENDEZVOUS AUCTION" if t < 0.20 else ("SIMULTANEOUS TRAVEL" if t < 0.72 else "RENDEZVOUS")

    rpos = {rid: along(path, rp) for rid, path in robot_paths.items()}
    hpos = {hid: along(path, hp) for hid, path in human_paths.items()}
    if assigned:
        for hid, rid in mapping.items():
            draw.line((*iso(hpos[hid][0] + 0.5, hpos[hid][1] + 0.5, 0.2), *iso(rpos[rid][0] + 0.5, rpos[rid][1] + 0.5, 0.2)), fill=(*LINK, 155), width=4)
    for rid in (3, 4):
        draw_agent(draw, rpos[rid], f"R{rid - 2}", ROBOT, robot=True)
    for hid in (1, 2):
        draw_agent(draw, hpos[hid], f"H{hid}", HUMAN)

    title = "REVERSE AUCTION — reactive assignment" if method == "auction" else "AHEAD-A — anticipatory rendezvous"
    draw.rounded_rectangle((34, 28, 650, 137), radius=18, fill=(7, 12, 22, 225), outline=(55, 75, 104), width=2)
    draw.text((58, 48), title, font=FONT_34, fill=WHITE)
    draw.text((59, 95), phase, font=FONT_18, fill=GREEN if assigned else ROBOT)
    draw.text((1030, 35), f"t = {t * 60:04.1f}", font=FONT_24, fill=WHITE)

    draw.rounded_rectangle((878, 558, 1245, 692), radius=14, fill=(7, 12, 22, 225), outline=(55, 75, 104), width=2)
    draw.text((900, 578), "POLICY OUTPUT", font=FONT_18, fill=WHITE)
    pairs = "  ".join(f"H{h}→R{r-2}" for h, r in sorted(mapping.items())) if assigned else "waiting for dispatch gate"
    draw.text((900, 611), pairs, font=FONT_18, fill=LINK if assigned else MUTED)
    draw.text((900, 645), "45° isometric synthetic warehouse", font=FONT_16, fill=MUTED)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("auction", "ahead"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, default=None)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
    _humans, _robots, costs, mapping = assignments(args.method)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for index in range(FRAMES):
        image = frame(args.method, index, mapping, costs)
        if args.poster and index == FRAMES // 2:
            image.save(args.poster)
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    receipt = {
        "method": args.method,
        "assignment": mapping,
        "cost_matrix": costs,
        "camera": "45-degree isometric",
        "fps": FPS,
        "frames": FRAMES,
        "output": str(args.output),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
