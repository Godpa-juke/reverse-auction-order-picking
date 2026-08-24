"""Structured state views for the warehouse simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from rware.core import Direction, SimulationConfig, State


@dataclass(frozen=True)
class AgentState:
    """Immutable snapshot of a single agent."""

    agent_id: int
    position: Tuple[int, int]
    direction: Direction
    state: State
    carrying_shelf_id: Optional[int]
    loadbox_station: Optional[Tuple[int, int]]
    coworker_id: Optional[int]
    total_distance: int
    complete_order: int

    @classmethod
    def from_agent(cls, agent) -> "AgentState":
        """Create a snapshot from a runtime agent object."""
        carrying_shelf = getattr(agent.carrying_shelf, "id", None)
        loadbox_station = getattr(agent, "loadbox_station", None)
        if isinstance(loadbox_station, Sequence) and len(loadbox_station) == 2:
            loadbox_station_tuple = (int(loadbox_station[0]), int(loadbox_station[1]))
        else:
            loadbox_station_tuple = None

        return cls(
            agent_id=int(agent.id),
            position=(int(agent.x), int(agent.y)),
            direction=agent.dir,
            state=agent.state,
            carrying_shelf_id=carrying_shelf,
            loadbox_station=loadbox_station_tuple,
            coworker_id=getattr(agent, "coworker", None),
            total_distance=int(getattr(agent, "total_distance", 0)),
            complete_order=int(getattr(agent, "complete_order", 0)),
        )


@dataclass(frozen=True)
class OrderQueueState:
    """Snapshot of the outstanding shelf requests."""

    requested_shelf_ids: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_request_queue(cls, request_queue: Iterable) -> "OrderQueueState":
        shelf_ids: List[int] = []
        for shelf in request_queue:
            shelf_id = getattr(shelf, "id", None)
            if shelf_id is not None:
                shelf_ids.append(int(shelf_id))
        return cls(requested_shelf_ids=tuple(shelf_ids))


@dataclass(frozen=True)
class WorldState:
    """High level snapshot of the simulation world."""

    tick: int
    config: SimulationConfig
    agents: Tuple[AgentState, ...]
    order_queue: OrderQueueState
    completed_batch: int
    completed_orders: int

    @classmethod
    def from_engine(cls, engine) -> "WorldState":
        """Capture the current engine state."""
        agents = tuple(AgentState.from_agent(agent) for agent in engine.agents)
        order_queue = OrderQueueState.from_request_queue(engine.request_queue)
        completed_batch = getattr(engine.task_scheduler, "completed_batch", 0)
        completed_orders = getattr(engine.task_scheduler, "all_of_completed_order", 0)

        return cls(
            tick=int(getattr(engine, "internal_timer", 0)),
            config=engine.config,
            agents=agents,
            order_queue=order_queue,
            completed_batch=int(completed_batch),
            completed_orders=int(completed_orders),
        )

