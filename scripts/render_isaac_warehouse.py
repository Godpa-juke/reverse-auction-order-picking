#!/usr/bin/env python3
"""Render the public AHEAD/Reverse-Auction policy in an actual Isaac Sim stage."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("auction", "ahead"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--poster", type=Path, required=True)
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

try:
    import cv2
    import imageio_ffmpeg
    import numpy as np
    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    import sys
    import types

    from rware.core import State

    # ``rware.engine.__init__`` eagerly imports the full pandas-backed event loop.
    # The renderer needs only the public assignment module, so expose the package
    # path without importing unrelated optional runtime consumers.
    engine_dir = Path(__file__).resolve().parents[1] / "rware" / "engine"
    engine_pkg = types.ModuleType("rware.engine")
    engine_pkg.__path__ = [str(engine_dir)]
    sys.modules["rware.engine"] = engine_pkg
    from rware.engine.human_assignment import (
        HumanSnapshot,
        RobotSnapshot,
        _solve_assignment_by_auction,
        get_human_assignment_strategy,
    )

    WIDTH, HEIGHT = 1280, 720
    ROBOT_COLOR = (1.0, 0.35, 0.05)
    HUMAN_COLOR = (0.0, 0.75, 0.68)
    SHELF_COLOR = (0.16, 0.24, 0.38)
    TARGET_COLOR = (0.05, 0.9, 0.3)

    class StaticTracker:
        def ensure_models(self, _backend):
            return None

        def median_human_trip(self):
            return 30.0

        def build_features(self, **kwargs):
            return kwargs

        def predict(self, rows, is_human=False):
            output = []
            for row in rows:
                ticks = float(row.get("static_ticks", row.get("planned_path_len", 0.0)))
                output.append((ticks, ticks))
            return output

    def compute_assignment(method: str):
        humans = [
            HumanSnapshot(1, (1, 2), 7, 0, [], State.NOOP),
            HumanSnapshot(2, (1, 10), 3, 0, [], State.NOOP),
        ]
        state = State.ROBOT_PICKING if method == "auction" else State.ROBOT_MOVESPOT
        robots = [
            RobotSnapshot(3, (2, 4), state, 6, 1, 2, 9.0, (14, 2), 101, 14),
            RobotSnapshot(4, (2, 8), state, 2, 2, 1, 6.0, (14, 10), 102, 14),
        ]
        config = SimpleNamespace(
            auction_zone_penalty=0.0,
            rendezvous_robot_wait_weight=1.0,
            rendezvous_human_wait_weight=0.6,
            rendezvous_human_travel_weight=1.0,
            rendezvous_risk_weight=0.5,
            dispatch_lead_limit=20,
        )
        grid = np.zeros((3, 13, 18), dtype=np.int8)
        for x in range(5, 13):
            for y in range(4, 9):
                grid[1, y, x] = 1
        tracker = StaticTracker()
        engine = SimpleNamespace(
            config=config,
            grid=grid,
            layer_shelfs=1,
            internal_timer=0,
            get_arrival_tracker=lambda: tracker,
        )
        strategy = get_human_assignment_strategy("auction" if method == "auction" else "rv_static")
        costs = strategy.build_cost_matrix(engine, humans, robots, context=None)
        selected = _solve_assignment_by_auction(
            [[-cost for cost in row] for row in costs],
            epsilon=0.01,
            time_limit_s=0.0,
            max_bid_updates=20_000,
        )
        mapping = {humans[i].id: robots[j].id for i, j in enumerate(selected) if j >= 0}
        return costs, mapping

    def ease(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def along(points, progress: float):
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        target = max(0.0, min(1.0, progress)) * sum(lengths)
        for a, b, length in zip(points, points[1:], lengths):
            if target <= length:
                u = 0.0 if length == 0 else target / length
                return tuple(a[i] + (b[i] - a[i]) * u for i in range(2))
            target -= length
        return points[-1]

    def make_xform(path: str, position):
        sim_utils.create_prim(path, "Xform", translation=position)
        prim = get_current_stage().GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        translate = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate = op
                break
        if translate is None:
            translate = xform.AddTranslateOp()
        return translate

    def set_position(op, xy, z=0.0):
        op.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), float(z)))

    def spawn_scene():
        sim_utils.GroundPlaneCfg(
            size=(30.0, 24.0),
            color=(0.055, 0.075, 0.105),
        ).func("/World/Ground", sim_utils.GroundPlaneCfg(size=(30.0, 24.0), color=(0.055, 0.075, 0.105)))
        sim_utils.DomeLightCfg(intensity=850.0, color=(0.75, 0.82, 1.0)).func(
            "/World/DomeLight", sim_utils.DomeLightCfg(intensity=850.0, color=(0.75, 0.82, 1.0))
        )
        sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.92, 0.78), angle=0.6).func(
            "/World/KeyLight", sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.92, 0.78), angle=0.6)
        )
        shelf_cfg = sim_utils.CuboidCfg(
            size=(1.5, 3.5, 2.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SHELF_COLOR, metallic=0.45, roughness=0.35),
        )
        for idx, x in enumerate((5.5, 8.0, 10.5, 13.0)):
            shelf_cfg.func(f"/World/Warehouse/Rack_{idx}", shelf_cfg, translation=(x, 6.5, 1.1))
        target_cfg = sim_utils.CylinderCfg(
            radius=0.55,
            height=0.06,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_COLOR, emissive_color=(0.0, 0.35, 0.08)),
        )
        for idx, pos in enumerate(((14.0, 2.0, 0.04), (14.0, 10.0, 0.04)), 1):
            target_cfg.func(f"/World/Warehouse/PickTarget_{idx}", target_cfg, translation=pos)

        robot_ops = {}
        for rid, pos in ((3, (2.0, 4.0, 0.0)), (4, (2.0, 8.0, 0.0))):
            root = f"/World/Agents/Robot_{rid}"
            robot_ops[rid] = make_xform(root, pos)
            base = sim_utils.CuboidCfg(
                size=(1.05, 0.8, 0.42),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROBOT_COLOR, metallic=0.65, roughness=0.22),
            )
            base.func(f"{root}/Base", base, translation=(0.0, 0.0, 0.28))
            top = sim_utils.CuboidCfg(
                size=(0.86, 0.64, 0.15),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.10, 0.14), metallic=0.8),
            )
            top.func(f"{root}/Top", top, translation=(0.0, 0.0, 0.56))
        human_ops = {}
        for hid, pos in ((1, (1.0, 2.0, 0.0)), (2, (1.0, 10.0, 0.0))):
            root = f"/World/Agents/Human_{hid}"
            human_ops[hid] = make_xform(root, pos)
            body = sim_utils.CapsuleCfg(
                radius=0.22,
                height=0.9,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=HUMAN_COLOR, roughness=0.55),
            )
            body.func(f"{root}/Body", body, translation=(0.0, 0.0, 0.7))
            head = sim_utils.SphereCfg(
                radius=0.25,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.58, 0.42), roughness=0.7),
            )
            head.func(f"{root}/Head", head, translation=(0.0, 0.0, 1.45))
        return robot_ops, human_ops

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device or "cuda:0", dt=1.0 / args.fps))
    robot_ops, human_ops = spawn_scene()
    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera = Camera(
        CameraCfg(
            prim_path="/World/CameraRig/CameraSensor",
            update_period=0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=25.0,
                horizontal_aperture=24.0,
                clipping_range=(0.1, 1000.0),
            ),
        )
    )
    sim.reset()
    eye = torch.tensor([[24.0, -9.0, math.hypot(15.0, -15.5)]], device=sim.device)
    target = torch.tensor([[9.0, 6.5, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.util.debug_draw")
    app.update()
    from isaacsim.util.debug_draw import _debug_draw
    debug = _debug_draw.acquire_debug_draw_interface()

    costs, mapping = compute_assignment(args.method)
    robot_paths = {3: [(2, 4), (2, 2), (14, 2)], 4: [(2, 8), (2, 10), (14, 10)]}
    human_paths = {1: [(1, 2), (14, 2)], 2: [(1, 10), (14, 10)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None

    for _ in range(8):
        sim.step(render=True)
        camera.update(dt=sim.get_physics_dt())

    for frame_idx in range(args.frames):
        t = frame_idx / max(1, args.frames - 1)
        if args.method == "auction":
            rp = ease((t - 0.04) / 0.36)
            hp = ease((t - 0.43) / 0.42)
            assigned = t >= 0.43
            phase = "ROBOTS ARRIVE -> REACTIVE AUCTION" if not assigned else "PICKERS DISPATCHED"
        else:
            rp = ease((t - 0.06) / 0.66)
            hp = ease((t - 0.10) / 0.48)
            assigned = t >= 0.10
            phase = "EN-ROUTE ROBOTS ADMITTED -> EARLY RENDEZVOUS" if assigned else "AHEAD-A DISPATCH GATE"
        rpos = {rid: along(path, rp) for rid, path in robot_paths.items()}
        hpos = {hid: along(path, hp) for hid, path in human_paths.items()}
        for rid, pos in rpos.items():
            set_position(robot_ops[rid], pos)
        for hid, pos in hpos.items():
            set_position(human_ops[hid], pos)
        debug.clear_lines()
        if assigned:
            starts, ends = [], []
            for hid, rid in sorted(mapping.items()):
                starts.append((hpos[hid][0], hpos[hid][1], 1.0))
                ends.append((rpos[rid][0], rpos[rid][1], 0.7))
            debug.draw_lines(starts, ends, [(0.1, 0.55, 1.0, 1.0)] * len(starts), [5.0] * len(starts))
        sim.step(render=True)
        camera.update(dt=sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0][..., :3].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
        title = "REVERSE AUCTION / ISAAC SIM" if args.method == "auction" else "AHEAD-A / ISAAC SIM"
        cv2.rectangle(rgb, (32, 28), (760, 116), (10, 16, 28), -1)
        cv2.putText(rgb, title, (55, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(rgb, phase, (56, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (100, 230, 160), 2, cv2.LINE_AA)
        pairs = "  ".join(f"H{hid}->R{rid-2}" for hid, rid in sorted(mapping.items())) if assigned else "assignment pending"
        cv2.putText(rgb, pairs, (930, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 210, 110), 2, cv2.LINE_AA)
        encoder.stdin.write(rgb.tobytes())
        if frame_idx == args.frames // 2:
            cv2.imwrite(str(args.poster), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg encoder failed")
    summary = {
        "renderer": "Isaac Sim / Isaac Lab Camera (Omniverse Replicator)",
        "method": args.method,
        "assignment": mapping,
        "cost_matrix": costs,
        "camera": {"eye": eye[0].tolist(), "lookat": target[0].tolist(), "downward_angle_deg": 45.0},
        "frames": args.frames,
        "fps": args.fps,
        "output": str(args.output),
        "status": "isaac_sim_rollout_ok",
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("ISAAC_WAREHOUSE_ROLLOUT_OK", json.dumps(summary, sort_keys=True), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(1)
else:
    # Kit can hang during teardown after a completed short headless camera run.
    # The encoder is already closed and the PASS receipt is flushed above.
    os._exit(0)
