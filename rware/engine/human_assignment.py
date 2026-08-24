"""Strategies for assigning human workers to collaborating robots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING

from rware.core import State
from rware.data.cost_maps import DATA_ROOT
import numpy as np
from rware.core.config import AuctionDistanceObstacles

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from rware.engine.warehouse_engine import WarehouseEngine
    from rware.entities import Agent
    from rware.core.config import SimulationConfig


@dataclass(frozen=True)
class HumanSnapshot:
    """Immutable view of a human agent used for matching algorithms."""

    id: int
    position: tuple[int, int]
    agent_timer: int
    waiting_time: int
    zone_nodes: List[int]
    state: State


@dataclass(frozen=True)
class RobotSnapshot:
    """Immutable view of a robot agent used for matching algorithms."""

    id: int
    position: tuple[int, int]
    state: State
    waiting_time: int
    routing_node_id: Optional[int]
    pending_items: int
    estimated_service_time: float
    # Rack the robot is heading to (equal to ``position`` once it has parked).
    # Predictive dispatch sends the worker here rather than to the robot.
    target_position: tuple[int, int] = (0, 0)
    rack_id: Optional[int] = None
    # Steps left on the robot's own planned path; 0 once it has arrived.
    remaining_path: int = 0


@dataclass(frozen=True)
class MatchingContext:
    """Context passed to assignment strategies for global decision making."""

    tick: int
    humans: List[HumanSnapshot]
    robots: List[RobotSnapshot]
    metrics: Dict[str, float]


class HumanAssignmentStrategy(ABC):
    """Base interface for human-to-robot assignment strategies."""

    name: str = "base"

    @abstractmethod
    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        """Return the robot id that should collaborate with ``human``."""

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        """Return a mapping of human id -> robot id for the current tick.

        Default implementation falls back to sequential single-human evaluation.
        """

        assignments: Dict[int, int] = {}
        used_robot_ids: Set[int] = set()

        for snapshot in context.humans:
            human_agent = engine.agents[snapshot.id - 1]
            robot_id = self.select_coworker(engine, human_agent)
            if robot_id is None or robot_id in used_robot_ids:
                continue
            assignments[snapshot.id] = robot_id
            used_robot_ids.add(robot_id)
        return assignments


_STRATEGY_REGISTRY: Dict[str, HumanAssignmentStrategy] = {}


def register_strategy(cls: Type[HumanAssignmentStrategy]) -> Type[HumanAssignmentStrategy]:
    """Class decorator that registers a strategy singleton."""

    instance = cls()
    _STRATEGY_REGISTRY[instance.name] = instance
    return cls


def available_human_assignment_strategies() -> Dict[str, HumanAssignmentStrategy]:
    """Expose the registered assignment strategies."""

    return dict(_STRATEGY_REGISTRY)


def get_human_assignment_strategy(name: Optional[str]) -> HumanAssignmentStrategy:
    key = (name or "").lower()
    if key not in _STRATEGY_REGISTRY:
        key = "nearest_idle"
    return _STRATEGY_REGISTRY[key]


# ---------------------------------------------------------------------------
# Shared distance helpers
#
# All strategies (traditional and auction) must use the same distance metric so
# that benchmark differences come from the matching policy, not from the
# quality of distance information. The metric is the BFS shortest path on the
# static map with shelves as the only obstacles (4-neighbour), matching the
# auction strategy's SHELF_ONLY mode. Manhattan distance is the fallback when
# the target is unreachable on the grid.
# ---------------------------------------------------------------------------

_FAIR_BLOCKED_ATTR = "_fair_distance_blocked_grid"
_FAIR_GRID_CACHE_ATTR = "_fair_distance_grid_cache"

_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_WITH_DIAGONAL = _CARDINAL + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _obstacle_mode_is_maze(engine: "WarehouseEngine") -> bool:
    config = getattr(engine, "config", None)
    mode = getattr(config, "auction_distance_obstacles", AuctionDistanceObstacles.SHELF_ONLY)
    return mode == AuctionDistanceObstacles.HUMAN_MAZE


def _get_blocked_grid(engine: "WarehouseEngine") -> Optional["np.ndarray"]:
    """Static blocked mask shared by all strategies (True = blocked).

    Honours ``config.auction_distance_obstacles`` so every strategy measures
    distance against the same obstacle model.
    """

    cached = getattr(engine, _FAIR_BLOCKED_ATTR, None)
    if cached is not None:
        return cached
    grid = getattr(engine, "grid", None)
    if grid is None:
        return None
    if _obstacle_mode_is_maze(engine):
        from rware.utils.Make_Maze import Make_Maze

        blocked = Make_Maze(engine, mode=1).astype(bool)
    else:
        shelf_layer = getattr(engine, "layer_shelfs", 1)
        blocked = (grid[shelf_layer] > 0)
    setattr(engine, _FAIR_BLOCKED_ATTR, blocked)
    return blocked


def _bfs_grid_from(
    blocked: "np.ndarray",
    start_x: int,
    start_y: int,
    allow_diagonal: bool = False,
) -> "np.ndarray":
    """BFS shortest-path length grid from (start_x, start_y)."""

    height, width = int(blocked.shape[0]), int(blocked.shape[1])
    dist = np.full((height, width), -1, dtype=np.int32)
    if not (0 <= start_x < width and 0 <= start_y < height):
        return dist
    if bool(blocked[start_y, start_x]):
        return dist
    neighbours = _WITH_DIAGONAL if allow_diagonal else _CARDINAL
    q: deque[tuple[int, int]] = deque()
    dist[start_y, start_x] = 0
    q.append((start_y, start_x))
    while q:
        y, x = q.popleft()
        nd = dist[y, x] + 1
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if dist[ny, nx] != -1 or blocked[ny, nx]:
                    continue
                if allow_diagonal and dy != 0 and dx != 0:
                    # Do not cut corners through an obstacle on a diagonal step.
                    if blocked[y, nx] or blocked[ny, x]:
                        continue
                dist[ny, nx] = nd
                q.append((ny, nx))
    return dist


def _human_distance_grid(
    engine: "WarehouseEngine",
    position: tuple[int, int],
) -> Optional["np.ndarray"]:
    """Return (and cache) the BFS distance grid from ``position``.

    The static map does not change during an episode, so grids are cached per
    start position on the engine instance.
    """

    blocked = _get_blocked_grid(engine)
    if blocked is None:
        return None
    cache: Dict[tuple[int, int], "np.ndarray"] = getattr(engine, _FAIR_GRID_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(engine, _FAIR_GRID_CACHE_ATTR, cache)
    key = (int(position[0]), int(position[1]))
    grid = cache.get(key)
    if grid is None:
        grid = _bfs_grid_from(blocked, key[0], key[1], _obstacle_mode_is_maze(engine))
        cache[key] = grid
    return grid


def _pair_distance(
    dist_grid: Optional["np.ndarray"],
    human_pos: tuple[int, int],
    robot_pos: tuple[int, int],
) -> float:
    """BFS real-path distance human -> robot with Manhattan fallback."""

    manhattan = abs(robot_pos[0] - human_pos[0]) + abs(robot_pos[1] - human_pos[1])
    if dist_grid is None:
        return float(manhattan)
    rx, ry = int(robot_pos[0]), int(robot_pos[1])
    if 0 <= ry < dist_grid.shape[0] and 0 <= rx < dist_grid.shape[1]:
        d = int(dist_grid[ry, rx])
        if d >= 0:
            return float(d)
    return float(manhattan)


@register_strategy
class NearestIdleWithinZoneStrategy(HumanAssignmentStrategy):
    """Pick the closest waiting robot (BFS real-path distance, zone-free)."""

    name = "nearest_idle"

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        dist_grid = _human_distance_grid(engine, (human.x, human.y))
        scored: list[tuple[float, int]] = []  # (distance, robot_id)

        for robot_id in engine.robot_id_list:
            robot = engine.agents[robot_id - 1]
            if robot.coworker is not None:
                continue
            if robot.state != State.ROBOT_PICKING:
                continue
            distance = _pair_distance(dist_grid, (human.x, human.y), (robot.x, robot.y))
            scored.append((distance, robot.id))

        if not scored:
            return None
        scored.sort()
        return scored[0][1]

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        if not context.humans or not context.robots:
            return {}

        # Build candidate pair list with cost components to enable fair greedy matching.
        pairs: List[tuple[tuple[float, int, int, int, int], int, int]] = []
        for snapshot in context.humans:
            dist_grid = _human_distance_grid(engine, snapshot.position)

            for robot in context.robots:
                # Skip robots not in picking state to mirror legacy behaviour.
                if robot.state != State.ROBOT_PICKING:
                    continue

                distance = _pair_distance(dist_grid, snapshot.position, robot.position)
                # Prefer humans/robots that have been waiting longer by using negative waiting times.
                cost_key = (
                    distance,
                    -robot.waiting_time,
                    -snapshot.agent_timer,
                    robot.id,
                    snapshot.id,
                )
                pairs.append((cost_key, snapshot.id, robot.id))

        if not pairs:
            return {}

        pairs.sort(key=lambda x: x[0])
        assignments: Dict[int, int] = {}
        used_humans: Set[int] = set()
        used_robots: Set[int] = set()

        for _, human_id, robot_id in pairs:
            if human_id in used_humans or robot_id in used_robots:
                continue
            assignments[human_id] = robot_id
            used_humans.add(human_id)
            used_robots.add(robot_id)

        return assignments


@register_strategy
class NearestRobotFirstStrategy(HumanAssignmentStrategy):
    """Assign each human to the closest available robot, prioritising human wait time."""

    name = "nearest_robot_first"

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        dist_grid = _human_distance_grid(engine, (human.x, human.y))

        best_robot_id: Optional[int] = None
        best_key: Optional[tuple[float, int, int]] = None

        for robot_id in engine.robot_id_list:
            robot = engine.agents[robot_id - 1]
            if robot.state != State.ROBOT_PICKING or robot.coworker is not None:
                continue

            distance = _pair_distance(dist_grid, (human.x, human.y), (robot.x, robot.y))
            key = (
                distance,
                -human.waiting_time,
                -robot.waiting_time,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_robot_id = robot.id

        return best_robot_id

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        if not context.humans or not context.robots:
            return {}

        pairs: List[tuple[tuple[float, int, int, int, int], int, int]] = []
        for human in context.humans:
            dist_grid = _human_distance_grid(engine, human.position)

            for robot in context.robots:
                if robot.state != State.ROBOT_PICKING:
                    continue

                distance = _pair_distance(dist_grid, human.position, robot.position)
                cost_key = (
                    distance,
                    -human.waiting_time,
                    -robot.waiting_time,
                    robot.id,
                    human.id,
                )
                pairs.append((cost_key, human.id, robot.id))

        if not pairs:
            return {}

        pairs.sort(key=lambda x: x[0])
        assignments: Dict[int, int] = {}
        used_humans: Set[int] = set()
        used_robots: Set[int] = set()

        for _, human_id, robot_id in pairs:
            if human_id in used_humans or robot_id in used_robots:
                continue
            assignments[human_id] = robot_id
            used_humans.add(human_id)
            used_robots.add(robot_id)

        return assignments


@register_strategy
class FirstRobotArrivedStrategy(HumanAssignmentStrategy):
    """Prioritise robots that have been waiting the longest at their pick location."""

    name = "first_robot_arrived"

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        dist_grid = _human_distance_grid(engine, (human.x, human.y))

        best_robot_id: Optional[int] = None
        best_key: Optional[tuple[int, float, int]] = None

        for robot_id in engine.robot_id_list:
            robot = engine.agents[robot_id - 1]
            if robot.state != State.ROBOT_PICKING or robot.coworker is not None:
                continue

            distance = _pair_distance(dist_grid, (human.x, human.y), (robot.x, robot.y))
            key = (
                -robot.waiting_time,
                distance,
                robot.id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_robot_id = robot.id

        return best_robot_id

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        if not context.humans or not context.robots:
            return {}

        available_humans: Dict[int, HumanSnapshot] = {h.id: h for h in context.humans}
        distance_grids = {
            h.id: _human_distance_grid(engine, h.position) for h in context.humans
        }
        assignments: Dict[int, int] = {}

        sorted_robots = sorted(
            context.robots,
            key=lambda r: (-r.waiting_time, r.id),
        )

        for robot in sorted_robots:
            best_human_id: Optional[int] = None
            best_key: Optional[tuple[float, int]] = None

            for human_id, human in available_humans.items():
                distance = _pair_distance(
                    distance_grids.get(human_id), human.position, robot.position
                )
                key = (
                    distance,
                    human.id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_human_id = human_id

            if best_human_id is not None:
                assignments[best_human_id] = robot.id
                del available_humans[best_human_id]

            if not available_humans:
                break

        return assignments


@register_strategy
class ShortestServiceRobotStrategy(HumanAssignmentStrategy):
    """Select robots with the lowest predicted remaining service time."""

    name = "shortest_service_robot"

    def _estimate_service_time(self, engine: "WarehouseEngine", robot: "Agent") -> float:
        cfg = engine.config
        if robot.order_sku_cnt:
            return max(1, robot.order_sku_cnt[0]) * cfg.sku_per_picking_time
        return cfg.sku_per_picking_time

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        dist_grid = _human_distance_grid(engine, (human.x, human.y))

        best_robot_id: Optional[int] = None
        best_key: Optional[tuple[float, float, int]] = None

        for robot_id in engine.robot_id_list:
            robot = engine.agents[robot_id - 1]
            if robot.state != State.ROBOT_PICKING or robot.coworker is not None:
                continue

            distance = _pair_distance(dist_grid, (human.x, human.y), (robot.x, robot.y))
            service_time = self._estimate_service_time(engine, robot)
            key = (
                service_time,
                distance,
                robot.id,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_robot_id = robot.id

        return best_robot_id

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        if not context.humans or not context.robots:
            return {}

        available_humans: Dict[int, HumanSnapshot] = {h.id: h for h in context.humans}
        distance_grids = {
            h.id: _human_distance_grid(engine, h.position) for h in context.humans
        }
        assignments: Dict[int, int] = {}

        sorted_robots = sorted(
            context.robots,
            key=lambda r: (r.estimated_service_time, -r.waiting_time, r.id),
        )

        for robot in sorted_robots:
            best_human_id: Optional[int] = None
            best_key: Optional[tuple[float, int]] = None

            for human_id, human in available_humans.items():
                distance = _pair_distance(
                    distance_grids.get(human_id), human.position, robot.position
                )
                key = (
                    distance,
                    human.id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_human_id = human_id

            if best_human_id is not None:
                assignments[best_human_id] = robot.id
                del available_humans[best_human_id]

            if not available_humans:
                break

        return assignments


@register_strategy
class LegacyBatchStrategy(HumanAssignmentStrategy):
    """Legacy heuristic prioritising robots already at picking locations."""

    name = "legacy_batch"

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        scored: list[tuple[float, int]] = []
        if human.agent_type is True:
            dist_grid = _human_distance_grid(engine, (human.x, human.y))
            for robot_id in engine.robot_id_list:
                robot = engine.agents[robot_id - 1]
                if robot.state != State.ROBOT_PICKING or robot.coworker is not None:
                    continue
                distance = _pair_distance(dist_grid, (human.x, human.y), (robot.x, robot.y))
                scored.append((distance, robot.id))

        if not scored:
            return None
        scored.sort()
        return scored[0][1]


def _solve_assignment_by_auction(
    values: List[List[float]],
    epsilon: float = 0.1,
    time_limit_s: float = 0.05,
    max_bid_updates: Optional[int] = None,
) -> List[int]:
    """Solve max assignment using a simple Bertsekas auction algorithm.

    Args:
        values: Bidder x item matrix (maximize).
        epsilon: Small positive constant controlling convergence.

    Returns:
        List[int]: bidder_index -> item_index assignment. Unassigned bidders get -1.
    """

    if not values:
        return []
    bidder_count = len(values)
    item_count = len(values[0]) if values[0] else 0
    if item_count == 0:
        return [-1 for _ in range(bidder_count)]

    prices = [0.0 for _ in range(item_count)]
    owner: List[int] = [-1 for _ in range(item_count)]
    assignment: List[int] = [-1 for _ in range(bidder_count)]

    q: deque[int] = deque(range(bidder_count))
    started = time.perf_counter()
    bid_updates = 0
    if max_bid_updates is None:
        # Safety valve to avoid hang-like behaviour even under pathological ties.
        max_bid_updates = max(1000, bidder_count * item_count * 50)
    # A wall-clock cut-off makes the solution depend on machine load, which
    # would show up as run-to-run noise in an ablation. ``time_limit_s <= 0``
    # disables it and leaves ``max_bid_updates`` as the only (deterministic)
    # termination guard.
    use_time_limit = time_limit_s > 0

    while q:
        if bid_updates >= max_bid_updates:
            break
        if use_time_limit and (time.perf_counter() - started) >= time_limit_s:
            break

        i = q.popleft()
        row = values[i]

        best_j = -1
        best_score = float("-inf")
        second_score = float("-inf")

        for j in range(item_count):
            score = row[j] - prices[j]
            if score > best_score:
                second_score = best_score
                best_score = score
                best_j = j
            elif score > second_score:
                second_score = score

        if best_j < 0:
            continue

        # Bid increment is gap between best and second best.
        increment = best_score - second_score + epsilon
        if increment != increment:  # NaN guard
            increment = epsilon

        prices[best_j] += increment
        bid_updates += 1

        prev_owner = owner[best_j]
        owner[best_j] = i
        assignment[i] = best_j

        if prev_owner != -1:
            assignment[prev_owner] = -1
            q.append(prev_owner)

    return assignment


def _solve_assignment_greedy(values: List[List[float]]) -> List[int]:
    """Greedy 1:1 matching over the same value matrix the auction bids on.

    Pairs are consumed in descending value order and a bidder or item that is
    already matched is skipped. This is the myopic alternative to the auction:
    pairing it with an unchanged cost matrix isolates the solver's contribution
    from the cost function's.

    Args:
        values: bidder_index -> item_index -> value. Higher is better.

    Returns:
        List[int]: bidder_index -> item_index assignment. Unassigned bidders
        get -1, matching :func:`_solve_assignment_by_auction`.
    """

    bidder_count = len(values)
    if bidder_count == 0:
        return []
    item_count = len(values[0])

    assignment: List[int] = [-1 for _ in range(bidder_count)]
    taken_items: Set[int] = set()

    # Ties break on (bidder, item) so a rerun of the same matrix gives the same
    # assignment; the auction path is deterministic for the same reason.
    pairs = sorted(
        (
            (values[bidder][item], bidder, item)
            for bidder in range(bidder_count)
            for item in range(item_count)
        ),
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    )

    for _value, bidder, item in pairs:
        if assignment[bidder] != -1 or item in taken_items:
            continue
        assignment[bidder] = item
        taken_items.add(item)

    return assignment


@dataclass
class _AuctionAssignmentInfo:
    """Re-auction을 위한 할당 상태 추적 정보."""

    robot_id: int
    human_id: int
    assigned_tick: int
    cost: float  # 할당 시점의 비용 (낮을수록 좋음)
    reassign_count: int = 0  # 재할당 횟수


@register_strategy
class AuctionAssignmentStrategy(HumanAssignmentStrategy):
    """Auction-based global human-to-robot assignment.

    Implements a reverse-auction style matching: robots are "tasks", humans bid a cost,
    and we compute a 1:1 assignment using an auction algorithm over the current tick's
    candidate humans/robots.

    Re-auction 지원 (진동 억제 메커니즘):
    - τ_lock: 도착 임박 락 - ETA ≤ τ_lock이면 재경매 대상에서 제외
    - Δ: 최소 이득 임계값 - (old_cost - new_cost) > Δ일 때만 재할당
    - K: 최대 재할당 횟수 - 요청당 최대 재할당 횟수 제한
    """

    name = "auction"

    # Cost weights (MVP). Keep these simple and tune later.
    w_distance: float = 1.0
    w_service_time: float = 0.1
    w_urgency_robot_wait: float = 0.2
    w_fairness_human_wait: float = 0.05

    # Large soft penalty when the robot is outside the worker's zone.
    # 이제 config.auction_zone_penalty에서 읽어옴 (기본값 0 = 존 제약 없음)
    zone_penalty: float = 10000.0  # fallback 기본값

    # Ablation hooks: config always carries these, so a subclass cannot change
    # them through a plain class attribute.
    zone_penalty_override: Optional[float] = None
    reauction_enabled_override: Optional[bool] = None

    _home_to_rack_cache_attr: str = "_auction_home_to_rack_distance"
    _blocked_cache_attr: str = "_auction_distance_blocked_grid"
    _assignment_cache_attr: str = "_auction_assignment_cache"

    # Auction solver knobs. The time limit is disabled so that a run is
    # reproducible; ``auction_max_bid_updates`` bounds the work instead.
    auction_epsilon: float = 0.01
    auction_time_limit_s: float = 0.0
    auction_max_bid_updates: Optional[int] = 20000

    # Which matcher consumes the cost matrix. ``greedy`` keeps the cost
    # function untouched so an ablation can price the solver on its own.
    solver: str = "auction"

    # Predictive dispatch is opt-in per strategy; the static auction keeps the
    # legacy reactive candidate set.
    requires_predictive_dispatch: bool = False

    def is_reauction_enabled(self, cfg: "SimulationConfig") -> bool:
        """Return the effective re-auction setting for this strategy."""

        if self.reauction_enabled_override is not None:
            return self.reauction_enabled_override
        return bool(getattr(cfg, "auction_reauction_enabled", True))

    def _home_to_rack_cache_path(self, name: str = "default") -> Path:
        """Return the default on-disk cache path for home->rack distances."""
        cfg = getattr(getattr(self, "_engine_for_cache_path", None), "config", None)
        mode = getattr(cfg, "auction_distance_obstacles", AuctionDistanceObstacles.SHELF_ONLY)
        suffix = getattr(mode, "value", str(mode))
        return DATA_ROOT / name / f"human_home_to_rack_{suffix}.npy"

    def _get_or_build_home_to_rack_distance(self, engine: "WarehouseEngine") -> Optional["np.ndarray"]:
        """Return cached home->rack distance matrix (human_index x rack_id).

        The matrix is indexed by:
        - row: index in engine.human_id_list
        - col: rack_id (1..len(engine.shelfs)), 0 is unused

        If the on-disk cache is missing, we build it once from the current static map.
        """
        try:
            import numpy as np
        except Exception:
            return None

        cached = getattr(engine, self._home_to_rack_cache_attr, None)
        if cached is not None:
            return cached

        human_count = len(getattr(engine, "human_id_list", []) or [])
        rack_count = len(getattr(engine, "shelfs", []) or [])
        if human_count <= 0 or rack_count <= 0:
            setattr(engine, self._home_to_rack_cache_attr, None)
            return None

        # stash engine to compute a mode-specific cache filename without changing signatures everywhere
        self._engine_for_cache_path = engine
        path = self._home_to_rack_cache_path("default")
        self._engine_for_cache_path = None
        if path.exists():
            try:
                loaded = np.load(path)
                if loaded.ndim == 2 and loaded.shape[0] == human_count and loaded.shape[1] >= (rack_count + 1):
                    setattr(engine, self._home_to_rack_cache_attr, loaded)
                    return loaded
            except Exception:
                # Fall through to rebuild.
                pass

        grid_size = getattr(engine, "grid_size", None)
        if grid_size is None:
            setattr(engine, self._home_to_rack_cache_attr, None)
            return None

        height, width = int(grid_size[0]), int(grid_size[1])
        cfg = getattr(engine, "config", None)
        obstacle_mode = getattr(cfg, "auction_distance_obstacles", AuctionDistanceObstacles.SHELF_ONLY)
        if obstacle_mode == AuctionDistanceObstacles.HUMAN_MAZE:
            # import locally to avoid import cycles at module import time
            from rware.utils.Make_Maze import Make_Maze

            blocked = Make_Maze(engine, mode=1).astype(bool)
            neighbor_dirs = (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            )
            allow_diagonal = True
        else:
            # Default: humans can pass robots; treat only shelves as obstacles.
            shelf_layer = getattr(engine, "layer_shelfs", 1)
            grid = getattr(engine, "grid", None)
            if grid is None:
                setattr(engine, self._home_to_rack_cache_attr, None)
                return None
            blocked = (grid[shelf_layer] > 0)
            neighbor_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            allow_diagonal = False

        def bfs_from(start_x: int, start_y: int) -> "np.ndarray":
            dist = np.full((height, width), -1, dtype=np.int32)
            if not (0 <= start_x < width and 0 <= start_y < height):
                return dist
            if bool(blocked[start_y, start_x]):
                return dist
            q: deque[tuple[int, int]] = deque()
            dist[start_y, start_x] = 0
            q.append((start_y, start_x))
            while q:
                y, x = q.popleft()
                nd = dist[y, x] + 1
                for dy, dx in neighbor_dirs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if dist[ny, nx] != -1:
                            continue
                        if bool(blocked[ny, nx]):
                            continue
                        if allow_diagonal and dy != 0 and dx != 0:
                            # Prevent "cutting corners" through obstacles on diagonal moves.
                            if bool(blocked[y, nx]) or bool(blocked[ny, x]):
                                continue
                        dist[ny, nx] = nd
                        q.append((ny, nx))
            return dist

        # Build matrix.
        matrix = np.full((human_count, rack_count + 1), 10**9, dtype=np.int32)

        # Pre-fetch rack goal positions.
        rack_goals: List[tuple[int, int]] = [(0, 0)] * (rack_count + 1)  # index by rack_id
        for rack_id in range(1, rack_count + 1):
            shelf = engine.shelfs[rack_id - 1]
            rack_goals[rack_id] = (int(getattr(shelf, "goal_x", shelf.x)), int(getattr(shelf, "goal_y", shelf.y)))

        for human_idx, human_id in enumerate(engine.human_id_list):
            human = engine.agents[human_id - 1]
            start_x, start_y = int(getattr(human, "init_x", human.x)), int(getattr(human, "init_y", human.y))
            dist_grid = bfs_from(start_x, start_y)
            matrix[human_idx, 0] = 0
            for rack_id in range(1, rack_count + 1):
                gx, gy = rack_goals[rack_id]
                if 0 <= gx < width and 0 <= gy < height:
                    d = int(dist_grid[gy, gx])
                    if d >= 0:
                        matrix[human_idx, rack_id] = d

        # Persist for future runs (small file; safe to cache).
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        except Exception:
            pass

        setattr(engine, self._home_to_rack_cache_attr, matrix)
        return matrix

    def _get_static_blocked_grid(self, engine: "WarehouseEngine") -> Optional["np.ndarray"]:
        """Return a cached static blocked mask (True means not traversable)."""
        cached = getattr(engine, self._blocked_cache_attr, None)
        if cached is not None:
            return cached
        cfg = getattr(engine, "config", None)
        obstacle_mode = getattr(cfg, "auction_distance_obstacles", AuctionDistanceObstacles.SHELF_ONLY)
        grid = getattr(engine, "grid", None)
        grid_size = getattr(engine, "grid_size", None)
        if grid is None or grid_size is None:
            setattr(engine, self._blocked_cache_attr, None)
            return None
        if obstacle_mode == AuctionDistanceObstacles.HUMAN_MAZE:
            from rware.utils.Make_Maze import Make_Maze

            blocked = Make_Maze(engine, mode=1).astype(bool)
        else:
            shelf_layer = getattr(engine, "layer_shelfs", 1)
            blocked = (grid[shelf_layer] > 0)
        setattr(engine, self._blocked_cache_attr, blocked)
        return blocked

    def _bfs_distance_grid(
        self,
        blocked: "np.ndarray",
        start_x: int,
        start_y: int,
    ) -> "np.ndarray":
        """Shortest-path length grid from (start_x, start_y).

        Delegates to the shared per-position cache so that every strategy uses
        one obstacle model and one set of distances. The static map does not
        change during an episode, so a grid is computed at most once per start
        cell instead of once per candidate per tick.
        """

        engine = getattr(self, "_engine_for_bfs", None)
        if engine is not None:
            grid = _human_distance_grid(engine, (start_x, start_y))
            if grid is not None:
                return grid
        allow_diagonal = bool(engine is not None and _obstacle_mode_is_maze(engine))
        return _bfs_grid_from(blocked, start_x, start_y, allow_diagonal)

    def _debug_log(self, engine: "WarehouseEngine", message: str) -> None:
        """Write a lightweight debug line to a per-process file when enabled.

        Controlled by SimulationConfig:
        - cfg.auction_debug: bool
        - cfg.auction_debug_path: Optional[str]
          - if empty/None -> `/tmp/rware_auction_<pid>.log`
          - if set and doesn't contain `{pid}`, we suffix with `.<pid>` to avoid collisions.
        """
        cfg = getattr(engine, "config", None)
        if not cfg or not getattr(cfg, "auction_debug", False):
            return
        try:
            import os

            pid = os.getpid()
            raw_path = getattr(cfg, "auction_debug_path", None) or ""
            if not raw_path:
                path = f"/tmp/rware_auction_{pid}.log"
            elif "{pid}" in raw_path:
                path = raw_path.format(pid=pid)
            else:
                path = f"{raw_path}.{pid}"

            with open(path, "a", encoding="utf-8") as f:
                f.write(message.rstrip() + "\n")
        except Exception:
            # Never let debug logging break the simulation.
            return

    def _solve(self, values: List[List[float]]) -> List[int]:
        """Run the configured matcher over ``values`` (higher is better)."""

        if self.solver == "greedy":
            return _solve_assignment_greedy(values)
        return _solve_assignment_by_auction(
            values,
            epsilon=float(self.auction_epsilon),
            time_limit_s=float(self.auction_time_limit_s),
            max_bid_updates=self.auction_max_bid_updates,
        )

    def _get_assignment_cache(self, engine: "WarehouseEngine") -> Dict[int, _AuctionAssignmentInfo]:
        """로봇별 할당 상태 캐시를 반환 (robot_id -> AssignmentInfo)."""
        cache = getattr(engine, self._assignment_cache_attr, None)
        if cache is None:
            cache = {}
            setattr(engine, self._assignment_cache_attr, cache)
        return cache

    def _compute_cost(
        self,
        h: HumanSnapshot,
        r: RobotSnapshot,
        engine: "WarehouseEngine",
        blocked: Optional["np.ndarray"],
        home_to_rack: Optional["np.ndarray"],
        human_index_lookup: Dict[int, int],
    ) -> float:
        """단일 (human, robot) 쌍에 대한 비용 계산 (낮을수록 좋음).

        Goes through :meth:`build_cost_matrix` so the re-auction gain test uses
        exactly the cost function the auction bid on, including in subclasses.
        """

        matrix = self.build_cost_matrix(
            engine,
            [h],
            [r],
            context=None,
            blocked=blocked,
            home_to_rack=home_to_rack,
            human_index_lookup=human_index_lookup,
        )
        return matrix[0][0]

    def _estimate_eta(
        self,
        human: HumanSnapshot,
        robot: RobotSnapshot,
        blocked: Optional["np.ndarray"],
    ) -> int:
        """사람이 로봇에게 도착하는 예상 시간 (tick/grid 단위)."""
        distance = abs(robot.position[0] - human.position[0]) + abs(robot.position[1] - human.position[1])
        if blocked is not None:
            dist_grid = self._bfs_distance_grid(blocked, int(human.position[0]), int(human.position[1]))
            rx, ry = int(robot.position[0]), int(robot.position[1])
            if 0 <= ry < dist_grid.shape[0] and 0 <= rx < dist_grid.shape[1]:
                d = int(dist_grid[ry, rx])
                if d >= 0:
                    distance = d
        return int(distance)

    def _is_locked(
        self,
        assignment_info: _AuctionAssignmentInfo,
        human: HumanSnapshot,
        robot: RobotSnapshot,
        cfg: "SimulationConfig",
        blocked: Optional["np.ndarray"],
    ) -> bool:
        """도착 임박 락 조건 확인: ETA ≤ τ_lock이면 True."""
        tau_lock = getattr(cfg, "auction_tau_lock", 3)
        eta = self._estimate_eta(human, robot, blocked)
        return eta <= tau_lock

    def _is_locked_by_eta(
        self,
        human: HumanSnapshot,
        robot: RobotSnapshot,
        cfg: "SimulationConfig",
        blocked: Optional["np.ndarray"],
        tau_lock: int,
    ) -> bool:
        """도착 임박 락 조건 확인 (ETA만 사용): ETA ≤ τ_lock이면 True."""
        eta = self._estimate_eta(human, robot, blocked)
        return eta <= tau_lock

    def _cleanup_stale_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> None:
        """완료되거나 유효하지 않은 할당 정보 정리.

        캐시는 재할당 횟수 추적용으로만 사용됩니다.
        실제 할당 상태는 engine의 coworker 속성에서 확인합니다.
        """
        cache = self._get_assignment_cache(engine)
        robot_ids_in_context = {r.id for r in context.robots}

        # 정리 대상 식별
        stale_robot_ids = []
        for robot_id, info in cache.items():
            # context에 없는 로봇 제거 (작업 완료 또는 다른 상태)
            if robot_id not in robot_ids_in_context:
                stale_robot_ids.append(robot_id)
                continue

            robot_agent = engine.agents[robot_id - 1]

            # 로봇이 더 이상 ROBOT_PICKING 상태가 아니면 제거
            if robot_agent.state != State.ROBOT_PICKING:
                stale_robot_ids.append(robot_id)
                continue

            # coworker가 캐시와 다르면 캐시 갱신이 필요
            if robot_agent.coworker is not None and robot_agent.coworker != info.human_id:
                stale_robot_ids.append(robot_id)
                continue

        for robot_id in stale_robot_ids:
            del cache[robot_id]

    def _travel_distance(
        self,
        engine: "WarehouseEngine",
        h: HumanSnapshot,
        r: RobotSnapshot,
        dist_grid: Optional["np.ndarray"],
        home_to_rack: Optional["np.ndarray"],
        human_index_lookup: Dict[int, int],
        target: Optional[tuple[int, int]] = None,
    ) -> float:
        """Worker travel distance to ``target`` (the robot's cell by default)."""

        goal = target if target is not None else r.position
        distance = float(abs(goal[0] - h.position[0]) + abs(goal[1] - h.position[1]))

        if dist_grid is not None:
            gx, gy = int(goal[0]), int(goal[1])
            if 0 <= gy < dist_grid.shape[0] and 0 <= gx < dist_grid.shape[1]:
                measured = int(dist_grid[gy, gx])
                if measured >= 0:
                    distance = float(measured)

        # Workers eligible for matching are usually parked at home, where a
        # precomputed home->rack table is both exact and free.
        if home_to_rack is not None and r.rack_id is not None:
            human_idx = human_index_lookup.get(h.id, -1)
            if 0 <= human_idx < home_to_rack.shape[0] and 0 <= r.rack_id < home_to_rack.shape[1]:
                human_agent = engine.agents[h.id - 1]
                if human_agent.x == human_agent.init_x and human_agent.y == human_agent.init_y:
                    cached = int(home_to_rack[human_idx, r.rack_id])
                    if 0 <= cached < 10**8:
                        distance = float(cached)

        return distance

    def build_cost_matrix(
        self,
        engine: "WarehouseEngine",
        humans: List[HumanSnapshot],
        robots: List[RobotSnapshot],
        context: MatchingContext,
        blocked: Optional["np.ndarray"] = None,
        home_to_rack: Optional["np.ndarray"] = None,
        human_index_lookup: Optional[Dict[int, int]] = None,
    ) -> List[List[float]]:
        """Cost of every (worker, robot) pair; lower is better.

        Subclasses override this to change what the auction prices in while
        reusing the solver, the re-auction guards and the assignment bookkeeping.
        """

        cfg = getattr(engine, "config", None)
        zone_penalty = getattr(cfg, "auction_zone_penalty", self.zone_penalty) if cfg else self.zone_penalty
        if self.zone_penalty_override is not None:
            zone_penalty = self.zone_penalty_override
        human_index_lookup = human_index_lookup or {}

        matrix: List[List[float]] = []
        for h in humans:
            allowed_nodes = h.zone_nodes
            dist_grid = _human_distance_grid(engine, h.position) if blocked is not None else None

            row: List[float] = []
            for r in robots:
                distance = self._travel_distance(
                    engine, h, r, dist_grid, home_to_rack, human_index_lookup
                )

                cost = 0.0
                if allowed_nodes and r.routing_node_id not in allowed_nodes:
                    cost += zone_penalty
                cost += self.w_distance * distance
                cost += self.w_service_time * float(r.estimated_service_time)
                # Prefer robots that have been waiting longer.
                cost -= self.w_urgency_robot_wait * float(r.waiting_time)
                # Prefer humans that have been waiting longer (proxy: agent_timer).
                cost -= self.w_fairness_human_wait * float(h.agent_timer)
                row.append(cost)
            matrix.append(row)
        return matrix

    def select_coworker(self, engine: "WarehouseEngine", human: "Agent") -> Optional[int]:
        # Fall back to a single-human decision using the existing greedy logic.
        helper = NearestIdleWithinZoneStrategy()
        return helper.select_coworker(engine, human)

    def plan_assignments(
        self,
        engine: "WarehouseEngine",
        context: MatchingContext,
    ) -> Dict[int, int]:
        import time

        t0 = time.perf_counter()
        cfg = getattr(engine, "config", None)
        if cfg is None:
            return {}

        # Re-auction 활성화 여부 확인
        reauction_enabled = self.is_reauction_enabled(cfg)
        tau_lock = getattr(cfg, "auction_tau_lock", 3)
        delta_gain = getattr(cfg, "auction_delta_gain", 5.0)
        max_reassign = getattr(cfg, "auction_max_reassign", 2)

        # 캐시 정리
        if reauction_enabled:
            self._cleanup_stale_assignments(engine, context)

        assignment_cache = self._get_assignment_cache(engine)

        # Respect the engine candidate selection rules (agent_timer threshold + trigger include).
        # This avoids "reserving" robots for humans that are not eligible to be matched yet.
        humans = list(context.humans)
        # The engine offers en-route robots when predictive dispatch is on.
        # Filtering them out here regardless made the whole rendezvous family
        # inert: the candidates arrived and were discarded before any cost was
        # computed for them.
        eligible_states = {State.ROBOT_PICKING}
        if getattr(engine, "predictive_dispatch_enabled", False):
            eligible_states.add(State.ROBOT_MOVESPOT)
        robots = [r for r in context.robots if r.state in eligible_states]

        if not humans or not robots:
            return {}

        self._debug_log(
            engine,
            f"tick={context.tick} strategy={self.name} solver={self.solver} "
            f"start humans={len(humans)} robots={len(robots)} "
            f"avg_robot_wait={(sum(r.waiting_time for r in robots) / max(len(robots), 1)):.3f} "
            f"avg_service={(sum(r.estimated_service_time for r in robots) / max(len(robots), 1)):.3f} "
            f"reauction={reauction_enabled} tau_lock={tau_lock} delta={delta_gain} max_reassign={max_reassign}"
        )

        # Optional: use a precomputed "home -> rack(goal)" shortest-path distance matrix.
        home_to_rack = self._get_or_build_home_to_rack_distance(engine)
        human_index_lookup: Dict[int, int] = {hid: idx for idx, hid in enumerate(engine.human_id_list)}
        # stash engine to let BFS pick neighbor model from config
        self._engine_for_bfs = engine
        blocked = self._get_static_blocked_grid(engine)

        # Re-auction 로직: 락된 할당 식별 및 재경매 대상 필터링
        # 주의: plan_assignments 반환 형식은 human_id -> robot_id
        locked_assignments: Dict[int, int] = {}  # human_id -> robot_id (락된 할당)
        reauction_robots: List[RobotSnapshot] = []  # 재경매 대상 로봇
        in_transit_humans: Set[int] = set()  # 이동 중인 human (대기 후보에서 제외)

        if reauction_enabled:
            human_lookup = {h.id: h for h in humans}
            robot_lookup = {r.id: r for r in robots}

            for robot in robots:
                # 로봇에 이미 coworker가 설정되어 있는지 확인 (실제 engine 상태)
                robot_agent = engine.agents[robot.id - 1]

                if robot_agent.coworker is not None:
                    # 이미 매칭 완료된 로봇: 이동 중인 human을 추적
                    assigned_human_id = robot_agent.coworker
                    human_agent = engine.agents[assigned_human_id - 1]

                    # human이 아직 로봇 위치에 도착하지 않았는지 확인
                    human_at_robot = (human_agent.x == robot_agent.x and human_agent.y == robot_agent.y)

                    if human_at_robot:
                        # 이미 도착해서 작업 중: 재경매 대상 아님
                        continue

                    # 이동 중인 human
                    in_transit_humans.add(assigned_human_id)

                    # 캐시에서 정보 확인 (비용, 재할당 횟수)
                    cached_info = assignment_cache.get(robot.id)

                    # HumanSnapshot 생성 (이동 중인 human용)
                    transit_human_snapshot = HumanSnapshot(
                        id=assigned_human_id,
                        position=(human_agent.x, human_agent.y),
                        agent_timer=human_agent.agent_timer,
                        waiting_time=getattr(human_agent, "waiting_time", 0),
                        zone_nodes=human_lookup.get(assigned_human_id, HumanSnapshot(0, (0, 0), 0, 0, [], State.NOOP)).zone_nodes if assigned_human_id in human_lookup else [],
                        state=human_agent.state,
                    )

                    # 락 조건 확인 (ETA ≤ τ_lock)
                    if self._is_locked_by_eta(transit_human_snapshot, robot, cfg, blocked, tau_lock):
                        # 락됨: 기존 할당 유지
                        locked_assignments[assigned_human_id] = robot.id
                        eta = self._estimate_eta(transit_human_snapshot, robot, blocked)
                        self._debug_log(
                            engine,
                            f"tick={context.tick} LOCKED robot={robot.id} human={assigned_human_id} eta={eta}"
                        )
                        continue

                    # 재할당 횟수 제한 확인
                    if cached_info is not None and cached_info.reassign_count >= max_reassign:
                        locked_assignments[assigned_human_id] = robot.id
                        self._debug_log(
                            engine,
                            f"tick={context.tick} MAX_REASSIGN robot={robot.id} human={assigned_human_id} "
                            f"count={cached_info.reassign_count}"
                        )
                        continue

                    # 재경매 대상
                    reauction_robots.append(robot)
                    self._debug_log(
                        engine,
                        f"tick={context.tick} REAUCTION_CANDIDATE robot={robot.id} current_human={assigned_human_id}"
                    )
                else:
                    # coworker가 없는 로봇: 새 할당 대상
                    reauction_robots.append(robot)
        else:
            # Re-auction 비활성화: 모든 로봇 대상
            reauction_robots = robots

        # 락된 human과 이동 중인 human 제외
        excluded_human_ids = set(locked_assignments.keys()) | in_transit_humans
        available_humans = [h for h in humans if h.id not in excluded_human_ids]

        self._debug_log(
            engine,
            f"tick={context.tick} locked={len(locked_assignments)} reauction_robots={len(reauction_robots)} "
            f"available_humans={len(available_humans)}"
        )

        # 락된 할당은 결과에 바로 포함
        result: Dict[int, int] = dict(locked_assignments)

        if not available_humans or not reauction_robots:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._debug_log(engine, f"tick={context.tick} done (all locked) assigns={len(result)} dt_ms={dt_ms:.2f}")
            self._engine_for_bfs = None
            return result

        t_build0 = time.perf_counter()

        cost_matrix = self.build_cost_matrix(
            engine,
            available_humans,
            reauction_robots,
            context,
            blocked=blocked,
            home_to_rack=home_to_rack,
            human_index_lookup=human_index_lookup,
        )
        # The solver maximises, so bid on value = -cost.
        values: List[List[float]] = [[-cost for cost in row] for row in cost_matrix]

        # Rectangular handling: if more humans than robots, add dummy items so the auction can complete.
        bidder_count = len(available_humans)
        item_count = len(reauction_robots)
        padded = 0
        if item_count < bidder_count:
            dummy_value = -1e12
            pad = bidder_count - item_count
            for i in range(bidder_count):
                values[i].extend([dummy_value] * pad)
            item_count = bidder_count
            padded = pad

        build_ms = (time.perf_counter() - t_build0) * 1000.0
        self._debug_log(
            engine,
            f"tick={context.tick} values_built bidders={bidder_count} items={item_count} padded={padded} dt_ms={build_ms:.2f}",
        )

        t_solve0 = time.perf_counter()
        self._debug_log(engine, f"tick={context.tick} solve_start")
        assignment = self._solve(values)
        solve_ms = (time.perf_counter() - t_solve0) * 1000.0
        self._debug_log(engine, f"tick={context.tick} solve_done dt_ms={solve_ms:.2f}")

        # 결과 처리 및 Re-auction 이득 검증
        for human_idx, item_idx in enumerate(assignment):
            if item_idx is None or item_idx < 0:
                continue
            # Ignore dummy assignments.
            if item_idx >= len(reauction_robots):
                continue

            human_id = available_humans[human_idx].id
            robot_id = reauction_robots[item_idx].id
            new_cost = cost_matrix[human_idx][item_idx]

            # 로봇의 현재 상태 확인
            robot_agent = engine.agents[robot_id - 1]
            current_coworker = robot_agent.coworker

            # Re-auction 이득 검증 (이미 이동 중인 human이 있는 경우)
            if reauction_enabled and current_coworker is not None:
                # 현재 이동 중인 human의 비용 계산
                old_human_agent = engine.agents[current_coworker - 1]
                old_human_snapshot = HumanSnapshot(
                    id=current_coworker,
                    position=(old_human_agent.x, old_human_agent.y),
                    agent_timer=old_human_agent.agent_timer,
                    waiting_time=getattr(old_human_agent, "waiting_time", 0),
                    zone_nodes=[],  # zone 검증은 이미 통과했으므로 빈 리스트
                    state=old_human_agent.state,
                )
                robot_snapshot = [r for r in reauction_robots if r.id == robot_id][0]
                old_cost = self._compute_cost(
                    old_human_snapshot, robot_snapshot, engine, blocked, home_to_rack, human_index_lookup
                )

                gain = old_cost - new_cost

                # 캐시에서 재할당 횟수 확인
                cached_info = assignment_cache.get(robot_id)
                reassign_count = cached_info.reassign_count if cached_info else 0

                if gain <= delta_gain:
                    # 이득이 충분하지 않음: 기존 할당 유지 (result에 추가하지 않음)
                    self._debug_log(
                        engine,
                        f"tick={context.tick} NO_GAIN robot={robot_id} "
                        f"old_human={current_coworker} new_human={human_id} "
                        f"old_cost={old_cost:.2f} new_cost={new_cost:.2f} gain={gain:.2f}"
                    )
                    # 기존 할당 유지 - result에 추가하지 않음 (이동 중인 human이 계속 진행)
                    continue

                # 재할당 성공
                self._debug_log(
                    engine,
                    f"tick={context.tick} REASSIGN robot={robot_id} "
                    f"old_human={current_coworker} new_human={human_id} "
                    f"old_cost={old_cost:.2f} new_cost={new_cost:.2f} gain={gain:.2f} "
                    f"reassign_count={reassign_count + 1}"
                )

                # 캐시 업데이트
                assignment_cache[robot_id] = _AuctionAssignmentInfo(
                    robot_id=robot_id,
                    human_id=human_id,
                    assigned_tick=context.tick,
                    cost=new_cost,
                    reassign_count=reassign_count + 1,
                )
            else:
                # 새로운 할당
                assignment_cache[robot_id] = _AuctionAssignmentInfo(
                    robot_id=robot_id,
                    human_id=human_id,
                    assigned_tick=context.tick,
                    cost=new_cost,
                    reassign_count=0,
                )
                self._debug_log(
                    engine,
                    f"tick={context.tick} NEW_ASSIGN robot={robot_id} human={human_id} cost={new_cost:.2f}"
                )

            result[human_id] = robot_id

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._debug_log(engine, f"tick={context.tick} done assigns={len(result)} dt_ms={dt_ms:.2f}")

        self._engine_for_bfs = None

        return result


# ---------------------------------------------------------------------------
# Ablations of the reverse auction.
#
# Each one changes exactly one thing about ``auction`` so the throughput delta
# can be attributed. They share ``build_cost_matrix`` and the re-auction
# bookkeeping with their parent; only the named attribute differs.
# ---------------------------------------------------------------------------


@register_strategy
class AuctionGreedySolverStrategy(AuctionAssignmentStrategy):
    """``auction``'s cost function settled by a greedy matcher."""

    name = "auction_greedy"
    solver = "greedy"


@register_strategy
class AuctionNoUrgencyStrategy(AuctionAssignmentStrategy):
    """Drops the cumulative robot-wait priority term."""

    name = "auction_no_urgency"
    w_urgency_robot_wait = 0.0


@register_strategy
class AuctionNoFairnessStrategy(AuctionAssignmentStrategy):
    """Drops the worker-fairness term."""

    name = "auction_no_fairness"
    w_fairness_human_wait = 0.0


@register_strategy
class AuctionNoServiceStrategy(AuctionAssignmentStrategy):
    """Drops the expected-service-time term."""

    name = "auction_no_service"
    w_service_time = 0.0


@register_strategy
class AuctionDistanceOnlyStrategy(AuctionAssignmentStrategy):
    """Prices travel distance and nothing else."""

    name = "auction_distance_only"
    w_service_time = 0.0
    w_urgency_robot_wait = 0.0
    w_fairness_human_wait = 0.0


@register_strategy
class AuctionDistanceOnlyGreedyStrategy(AuctionDistanceOnlyStrategy):
    """Distance-only cost settled greedily.

    ``auction_greedy`` prices the solver on top of the full four-term cost,
    which the batch showed to be the worse cost function; this arm prices it
    on top of the better one, so the two together separate the matcher from
    what it is matching on.
    """

    name = "auction_distance_only_greedy"
    solver = "greedy"


@register_strategy
class AuctionGlobalOnlyStrategy(AuctionDistanceOnlyGreedyStrategy):
    """Distance-only greedy matching without assignment re-evaluation.

    The registry name is retained for compatibility with the experiment data,
    but ``nearest_idle`` is also a batch matcher. This arm therefore isolates
    re-evaluation, apart from the two matchers' different deterministic tie
    breaks. It has no priority terms and no auction solver.
    """

    name = "auction_global_only"
    reauction_enabled_override = False


@register_strategy
class AuctionZoneOnStrategy(AuctionAssignmentStrategy):
    """``auction`` with the zone constraint priced back in.

    Zone confinement was dropped from every strategy when the comparison was
    levelled; this measures what that choice alone is worth.
    """

    name = "auction_zone_on"
    zone_penalty_override = 10000.0


# ---------------------------------------------------------------------------
# Predictive rendezvous assignment
#
# The reactive rule only offers a worker to a robot that has already parked, so
# the worker's whole trip is robot idle time (28.9% of makespan on a full run).
# These strategies also consider robots still travelling and choose the worker
# whose predicted arrival best meets the robot's predicted arrival.
#
# The prediction has to be learned rather than read off the map: measured over
# 2,346 worker trips the realised/static travel ratio has median 1.13 and p90
# 1.50, so dispatching on path length alone arrives late more often than not.
# ---------------------------------------------------------------------------


class RendezvousAuctionStrategy(AuctionAssignmentStrategy):
    """Reverse auction over predicted worker/robot rendezvous.

    ``eta_backend`` selects how arrival times are estimated:

    ``static``          path length, i.e. no learning (ablation A1)
    ``rolling_median``  rolling quantiles of the realised/static ratio
    ``catboost``        CatBoost MultiQuantile on the trip log
    ``oracle``          calibrated upper bound

    ``use_risk`` adds the tail term that prices being late (Q90 rather than
    Q50), and ``event_trigger`` re-auctions a pairing whose worker is already
    later than its own Q90 prediction.
    """

    name = "rendezvous"
    requires_predictive_dispatch = True

    eta_backend: str = "static"
    use_risk: bool = False
    event_trigger: bool = False

    # Applied to the robot's *current* starvation in ticks. Kept small on
    # purpose: at weight 1.0 it swamps the travel term (~1.3 x 20 ticks) and the
    # auction sends workers to the longest-starving robot regardless of
    # distance, which measured 65% more worker travel and lower throughput.
    w_urgency_robot_wait: float = 0.2

    def _tracker(self, engine: "WarehouseEngine"):
        tracker = engine.get_arrival_tracker()
        tracker.ensure_models(self.eta_backend)
        return tracker

    def _is_locked_by_eta(
        self,
        human: HumanSnapshot,
        robot: RobotSnapshot,
        cfg: "SimulationConfig",
        blocked: Optional["np.ndarray"],
        tau_lock: int,
    ) -> bool:
        """Arrival lock, with a quantile-exceedance escape hatch.

        A worker that is already later than its own predicted Q90 is no longer
        "about to arrive", so the proximity lock must not keep the robot
        committed to them.
        """

        if self.event_trigger:
            engine = getattr(self, "_engine_for_bfs", None)
            if engine is not None:
                tracker = self._tracker(engine)
                if tracker.is_overdue(human.id, int(engine.internal_timer)):
                    return False
        return super()._is_locked_by_eta(human, robot, cfg, blocked, tau_lock)

    def build_cost_matrix(
        self,
        engine: "WarehouseEngine",
        humans: List[HumanSnapshot],
        robots: List[RobotSnapshot],
        context: Optional[MatchingContext] = None,
        blocked: Optional["np.ndarray"] = None,
        home_to_rack: Optional["np.ndarray"] = None,
        human_index_lookup: Optional[Dict[int, int]] = None,
    ) -> List[List[float]]:
        cfg = getattr(engine, "config", None)
        human_index_lookup = human_index_lookup or {}

        w_robot_wait = float(getattr(cfg, "rendezvous_robot_wait_weight", 1.0))
        w_human_wait = float(getattr(cfg, "rendezvous_human_wait_weight", 0.6))
        w_human_travel = float(getattr(cfg, "rendezvous_human_travel_weight", 1.0))
        w_risk = float(getattr(cfg, "rendezvous_risk_weight", 0.5))
        lead_limit = float(getattr(cfg, "dispatch_lead_limit", 10))
        zone_penalty = getattr(cfg, "auction_zone_penalty", self.zone_penalty) if cfg else 0.0

        tracker = self._tracker(engine)
        tick = int(getattr(engine, "internal_timer", 0))
        # Learned from realised trips, so the budget adapts to the map.
        travel_budget = tracker.median_human_trip()

        # One prediction batch per tick rather than one call per pair.
        robot_rows = [self._robot_row(engine, tracker, r) for r in robots]
        robot_eta = tracker.predict(robot_rows, is_human=False)

        human_rows: List[Dict[str, Any]] = []
        index: List[tuple[int, int]] = []
        for hi, h in enumerate(humans):
            grid = _human_distance_grid(engine, h.position)
            for ri, r in enumerate(robots):
                static = self._travel_distance(
                    engine, h, r, grid, home_to_rack, human_index_lookup,
                    target=r.target_position,
                )
                human_rows.append(
                    tracker.build_features(
                        agent_id=h.id,
                        zone_id=int(h.zone_nodes[0]) if h.zone_nodes else -1,
                        start=h.position,
                        goal=r.target_position,
                        planned_path_len=0,
                        is_human=True,
                        static_ticks=static,
                    )
                )
                index.append((hi, ri))
        human_eta = tracker.predict(human_rows, is_human=True)

        matrix = [[0.0] * len(robots) for _ in humans]
        for (hi, ri), (h_q50, h_q90) in zip(index, human_eta):
            h = humans[hi]
            r = robots[ri]
            r_q50, r_q90 = robot_eta[ri]
            # A parked robot is already there; only en-route robots have an ETA.
            if r.state == State.ROBOT_PICKING:
                r_q50 = r_q90 = 0.0

            # Picking starts when the later of the two arrives. Price the three
            # costs separately: the robot's idle wait, the worker's walk, and
            # the worker's idle wait at the rack.
            #
            # Travel has to stay its own term. Folding it into "time until
            # service starts" makes every worker cost the same whenever the
            # robot arrives last, which collapses the distance signal: an early
            # version did exactly that and worker travel rose 60%.
            service_start = max(h_q50, r_q50)
            robot_idle = service_start - r_q50
            human_idle = service_start - h_q50

            # For a parked robot (r_q50 = 0) this reduces to h_q50, i.e. exactly
            # the static auction's travel cost, so turning predictive dispatch
            # off reproduces the baseline and any measured delta is attributable
            # to the en-route pairings alone.
            cost = (
                w_robot_wait * robot_idle
                + w_human_travel * h_q50
                + w_human_wait * human_idle
            )
            if self.use_risk:
                # Being late is the expensive direction, so price the worker's
                # upper quantile against the robot's median arrival.
                cost += w_risk * max(0.0, h_q90 - max(r_q50, h_q50))
            cost += self.w_service_time * float(r.estimated_service_time)
            # Same urgency signal as the static auction, so the two cost
            # functions coincide in the reactive case.
            cost -= self.w_urgency_robot_wait * float(r.waiting_time)
            cost -= self.w_fairness_human_wait * float(h.agent_timer)

            if h.zone_nodes and r.routing_node_id not in h.zone_nodes:
                cost += zone_penalty

            # Guards on speculative pairings. A robot that has already parked is
            # always allowed; committing a worker to one that has not is only
            # worth it when the two actually meet and the walk is not long.
            #
            # Without the travel budget the auction dispatches every free worker
            # every tick, because en-route robots make the candidate set several
            # times larger. Workers then always have somewhere far to be, worker
            # travel rises ~60% and throughput falls: the reactive rule's
            # implicit "wait for a nearby robot" is doing real work.
            if r.state != State.ROBOT_PICKING:
                if (r_q50 - h_q50) > lead_limit:
                    cost += 1e6
                if travel_budget > 0 and h_q50 > travel_budget:
                    cost += 1e6

            matrix[hi][ri] = cost
        return matrix

    def _robot_row(self, engine: "WarehouseEngine", tracker, r: RobotSnapshot) -> Dict[str, Any]:
        return tracker.build_features(
            agent_id=r.id,
            zone_id=int(r.routing_node_id) if r.routing_node_id is not None else -1,
            start=r.position,
            goal=r.target_position,
            planned_path_len=float(r.remaining_path),
            is_human=False,
        )


def _register_rendezvous(
    strategy_name: str,
    eta_backend: str,
    use_risk: bool,
    event_trigger: bool,
) -> None:
    """Register one rung of the ablation ladder."""

    cls = type(
        f"Rendezvous_{strategy_name}",
        (RendezvousAuctionStrategy,),
        {
            "name": strategy_name,
            "eta_backend": eta_backend,
            "use_risk": use_risk,
            "event_trigger": event_trigger,
        },
    )
    register_strategy(cls)


# A1: predictive dispatch, arrival estimated by path length (no learning).
_register_rendezvous("rv_static", "static", use_risk=False, event_trigger=False)
# A2: + learned median arrival.
_register_rendezvous("rv_q50", "catboost", use_risk=False, event_trigger=False)
# A3: + tail risk from the learned upper quantile.
_register_rendezvous("rv_risk", "catboost", use_risk=True, event_trigger=False)
# A4: + re-auction triggered by a worker running past its own Q90.
_register_rendezvous("rv_risk_rt", "catboost", use_risk=True, event_trigger=True)
# Cheap-learning reference: rolling quantiles instead of a model.
_register_rendezvous("rv_median", "rolling_median", use_risk=True, event_trigger=False)
# Upper bound on what better arrival prediction can buy.
_register_rendezvous("rv_oracle", "oracle", use_risk=True, event_trigger=True)


@register_strategy
class RendezvousRiskRTNoUrgencyStrategy(
    get_human_assignment_strategy("rv_risk_rt").__class__
):
    """``rv_risk_rt`` without the robot cumulative-wait term.

    The rendezvous cost matrix carries that term on top of its own weights,
    and on the plain auction deleting it is worth +2.600 Box/Hour/Human. This
    arm checks whether the rendezvous gain measured against `auction` survives
    once both sides drop the term.
    """

    name = "rv_risk_rt_no_urgency"
    w_urgency_robot_wait = 0.0
