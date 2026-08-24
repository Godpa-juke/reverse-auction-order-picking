"""ZMQ publisher that streams engine state snapshots to an external viewer.

The engine stays authoritative: this module only serializes the current
world state and pushes it over a PUB socket. Subscribers (e.g. the Isaac
Sim viewer) consume the latest frame and render it; a slow or absent
subscriber never blocks the simulation loop.

Wire format (all JSON, UTF-8, topic-prefixed multipart):
  topic b"layout" : static world description, re-published periodically
                    so late-joining subscribers can build the scene.
  topic b"frame"  : per-tick dynamic state (agents + shelves).
"""

from __future__ import annotations

import json
from typing import Optional

import zmq

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5556"
LAYOUT_REPUBLISH_INTERVAL = 60  # frames


class BridgePublisher:
    """Publishes layout/frame snapshots of a WarehouseEngine over ZMQ."""

    def __init__(self, engine, endpoint: str = DEFAULT_ENDPOINT):
        self.engine = engine
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        # Never block the engine loop: keep at most a handful of pending
        # frames and drop the rest if the subscriber falls behind.
        self._socket.setsockopt(zmq.SNDHWM, 10)
        self._socket.setsockopt(zmq.LINGER, 0)
        self.endpoint = self._bind_with_fallback(endpoint)
        self._frame_count = 0

    def _bind_with_fallback(self, endpoint: str, attempts: int = 16) -> str:
        """Bind to endpoint; on tcp port conflict (parallel strategy workers,
        stale process) walk up to the next free port."""
        if not endpoint.startswith("tcp://") or ":" not in endpoint.rsplit("/", 1)[-1]:
            self._socket.bind(endpoint)
            return endpoint
        host, port = endpoint.rsplit(":", 1)
        for offset in range(attempts):
            candidate = f"{host}:{int(port) + offset}"
            try:
                self._socket.bind(candidate)
                return candidate
            except zmq.ZMQError:
                continue
        raise zmq.ZMQError(msg=f"no free port near {endpoint}")

    def publish(self) -> None:
        """Publish one frame; periodically re-publish the layout."""
        if self._frame_count % LAYOUT_REPUBLISH_INTERVAL == 0:
            self._send("layout", self._layout_payload())
        self._send("frame", self._frame_payload())
        self._frame_count += 1

    def close(self) -> None:
        self._socket.close(0)

    # ------------------------------------------------------------------
    def _send(self, topic: str, payload: dict) -> None:
        try:
            self._socket.send_multipart(
                [topic.encode(), json.dumps(payload).encode()],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass  # subscriber backlog full; drop frame

    def _layout_payload(self) -> dict:
        engine = self.engine
        grid_h, grid_w = engine.grid_size
        walls = getattr(engine, "walls", None)
        return {
            "grid_w": int(grid_w),
            "grid_h": int(grid_h),
            "highways": engine.highways.astype(int).tolist(),
            "goals": [[int(x), int(y)] for (x, y) in engine.goals],
            "n_agents": int(engine.n_agents),
            "human_ids": [int(a.id) for a in engine.agents if a.agent_type],
            "walls": (
                [
                    [int(x), int(y), int(walls[y, x])]
                    for y in range(grid_h)
                    for x in range(grid_w)
                    if walls[y, x]
                ]
                if walls is not None
                else []
            ),
        }

    def _frame_payload(self) -> dict:
        engine = self.engine
        agents = [
            [
                int(a.id),
                int(a.x),
                int(a.y),
                int(a.dir.value) if a.dir is not None else 0,
                int(a.state.value) if a.state is not None else 0,
                1 if a.agent_type else 0,
                int(a.carrying_shelf.id) if getattr(a, "carrying_shelf", None) else -1,
            ]
            for a in engine.agents
        ]
        shelves = [[int(s.id), int(s.x), int(s.y)] for s in engine.shelfs]
        return {
            "tick": int(getattr(engine, "internal_timer", self._frame_count)),
            "agents": agents,
            "shelves": shelves,
        }


def publisher_from_env(engine) -> Optional[BridgePublisher]:
    """Create a publisher when RWARE_BRIDGE=1; endpoint via RWARE_BRIDGE_ENDPOINT."""
    import os

    if os.environ.get("RWARE_BRIDGE", "") not in ("1", "true", "on"):
        return None
    endpoint = os.environ.get("RWARE_BRIDGE_ENDPOINT", DEFAULT_ENDPOINT)
    return BridgePublisher(engine, endpoint=endpoint)
