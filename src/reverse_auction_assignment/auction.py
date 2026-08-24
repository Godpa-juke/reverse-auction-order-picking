"""Data-free reverse-auction assignment primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Sequence

Position = tuple[int, int]
Distance = Callable[[Position, Position], float]


@dataclass(frozen=True)
class Worker:
    worker_id: str
    position: Position
    allowed_nodes: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RobotRequest:
    request_id: str
    position: Position
    node_id: int | None = None
    accumulated_wait: float = 0.0
    expected_service: float = 0.0


@dataclass(frozen=True)
class CostWeights:
    travel: float = 1.0
    robot_wait: float = 0.0
    service: float = 0.0
    zone_mismatch: float = 0.0


def manhattan(a: Position, b: Position) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def pair_cost(
    worker: Worker,
    request: RobotRequest,
    weights: CostWeights,
    distance: Distance = manhattan,
) -> float:
    """Compute one worker-request cost without simulator state."""

    mismatch = (
        request.node_id is not None
        and bool(worker.allowed_nodes)
        and request.node_id not in worker.allowed_nodes
    )
    return (
        weights.travel * distance(worker.position, request.position)
        + weights.robot_wait * request.accumulated_wait
        + weights.service * request.expected_service
        + weights.zone_mismatch * float(mismatch)
    )


def solve_auction(
    costs: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-6,
    max_updates: int | None = None,
) -> list[int | None]:
    """Minimize a rectangular cost matrix with an auction algorithm.

    The return value contains one object index per bidder. ``None`` denotes a
    dummy object when there are fewer real objects than bidders.
    """

    rows = [list(map(float, row)) for row in costs]
    if not rows:
        return []
    width = len(rows[0])
    if width == 0:
        return [None] * len(rows)
    if any(len(row) != width for row in rows):
        raise ValueError("cost matrix must be rectangular")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if any(not isfinite(value) for row in rows for value in row):
        raise ValueError("cost matrix values must be finite")

    bidder_count = len(rows)
    real_objects = width
    object_count = max(real_objects, bidder_count)
    largest = max(value for row in rows for value in row)
    dummy_cost = largest + max(1.0, abs(largest)) * (bidder_count + 1)
    padded = [row + [dummy_cost] * (object_count - real_objects) for row in rows]
    values = [[-cost for cost in row] for row in padded]

    prices = [0.0] * object_count
    owners: list[int | None] = [None] * object_count
    assignment: list[int | None] = [None] * bidder_count
    pending = list(reversed(range(bidder_count)))
    limit = max_updates or max(1000, bidder_count * object_count * 100)
    updates = 0

    while pending:
        bidder = pending.pop()
        utilities = [values[bidder][j] - prices[j] for j in range(object_count)]
        ranked = sorted(range(object_count), key=lambda j: (-utilities[j], j))
        best = ranked[0]
        second_utility = utilities[ranked[1]] if object_count > 1 else utilities[best] - epsilon
        increment = utilities[best] - second_utility + epsilon
        prices[best] += increment

        previous = owners[best]
        owners[best] = bidder
        assignment[bidder] = best
        if previous is not None:
            assignment[previous] = None
            pending.append(previous)

        updates += 1
        if updates > limit:
            raise RuntimeError("auction solver exceeded max_updates")

    return [index if index is not None and index < real_objects else None for index in assignment]


def assign(
    workers: Sequence[Worker],
    requests: Sequence[RobotRequest],
    *,
    weights: CostWeights = CostWeights(),
    distance: Distance = manhattan,
) -> list[tuple[str, str]]:
    """Return deterministic one-to-one worker/request identifiers."""

    ordered_workers = sorted(workers, key=lambda worker: worker.worker_id)
    ordered_requests = sorted(requests, key=lambda request: request.request_id)
    costs = [
        [pair_cost(worker, request, weights, distance) for request in ordered_requests]
        for worker in ordered_workers
    ]
    chosen = solve_auction(costs)
    return [
        (worker.worker_id, ordered_requests[index].request_id)
        for worker, index in zip(ordered_workers, chosen)
        if index is not None
    ]
