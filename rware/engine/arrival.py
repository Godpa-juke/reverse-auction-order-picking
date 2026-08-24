"""Arrival-time observation and prediction for predictive rendezvous dispatch.

Why arrival time
----------------
In the shipped dispatch rule a worker is only offered to a robot that has
already parked at its rack (``_collect_available_robots`` filters on
``ROBOT_PICKING``). The worker then walks over while the robot sits idle, so the
worker's entire travel time is charged to robot waiting. Measured on a full run
that is 28.9% of the makespan (27,944 of 96,534 ticks; 31.0 ticks per
interaction against a mean worker trip of 26.4 ticks).

Sending the worker while the robot is still en route removes that
serialisation, but only if both arrivals can be timed. Static path length is a
biased estimate of arrival time under multi-agent congestion: measured over
2,346 worker trips the realised/static ratio has median 1.13, p90 1.50 and a
maximum of 3.0. Dispatching on the static estimate therefore arrives late more
often than not, which is why the quantiles are learned rather than assumed.

This module records realised arrivals and serves (Q50, Q90) predictions for
both agent types. It observes state transitions from outside the agent update
loop, so no per-agent bookkeeping leaks into ``entities.py``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from rware.core import State
from rware.engine.human_assignment import _human_distance_grid, _pair_distance
from rware.learning.risk_model import CATBOOST_BACKENDS, build_risk_model, ServiceRiskModel

# Feature names shared by the worker and robot arrival models. Everything here
# is computable at dispatch time, before the trip starts.
ARRIVAL_FEATURES: List[str] = [
    "agent_id",
    "zone_id",
    "static_ticks",
    "manhattan",
    "detour_ratio",
    "planned_path_len",
    "recent_ratio_mean",
    "recent_ratio_p90",
    "recent_samples",
    "active_robots",
    "moving_robots",
    "busy_humans",
    "elapsed_ticks",
]


@dataclass
class _OpenTrip:
    """A trip in progress, with the features captured when it started."""

    agent_id: int
    start_tick: int
    goal: Tuple[int, int]
    features: Dict[str, Any]
    static_ticks: float
    # Upper-quantile arrival predicted at dispatch. A trip that runs past it is
    # the event that triggers a re-auction.
    predicted_q90: float = 0.0
    predictions: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)


@dataclass
class PredictionStats:
    """Accuracy and calibration for predictions captured before a trip."""

    count: int = 0
    ready_count: int = 0
    q50_abs_error_sum: float = 0.0
    ready_q50_abs_error_sum: float = 0.0
    q90_covered: int = 0
    ready_q90_covered: int = 0
    nondegenerate: int = 0
    ready_nondegenerate: int = 0

    def add(self, q50: float, q90: float, actual: float, ready: bool) -> None:
        self.count += 1
        self.q50_abs_error_sum += abs(actual - q50)
        self.q90_covered += int(actual <= q90)
        self.nondegenerate += int(q90 > q50)
        if ready:
            self.ready_count += 1
            self.ready_q50_abs_error_sum += abs(actual - q50)
            self.ready_q90_covered += int(actual <= q90)
            self.ready_nondegenerate += int(q90 > q50)

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "q50_mae": self.q50_abs_error_sum / self.count if self.count else 0.0,
            "q90_coverage": self.q90_covered / self.count if self.count else 0.0,
            "q90_gt_q50_share": self.nondegenerate / self.count if self.count else 0.0,
            "ready_count": self.ready_count,
            "ready_q50_mae": (
                self.ready_q50_abs_error_sum / self.ready_count if self.ready_count else 0.0
            ),
            "ready_q90_coverage": (
                self.ready_q90_covered / self.ready_count if self.ready_count else 0.0
            ),
            "ready_q90_gt_q50_share": (
                self.ready_nondegenerate / self.ready_count if self.ready_count else 0.0
            ),
        }


@dataclass
class ArrivalStats:
    """Realised arrival accuracy, reported at the end of a run."""

    trips: int = 0
    static_sum: float = 0.0
    actual_sum: float = 0.0
    ratios: List[float] = field(default_factory=list)

    def add(self, static_ticks: float, actual_ticks: float) -> None:
        self.trips += 1
        self.static_sum += static_ticks
        self.actual_sum += actual_ticks
        if static_ticks > 0:
            self.ratios.append(actual_ticks / static_ticks)

    def summary(self) -> Dict[str, float]:
        if not self.trips:
            return {"trips": 0, "mean_static": 0.0, "mean_actual": 0.0, "mean_ratio": 1.0}
        ordered = sorted(self.ratios)
        return {
            "trips": self.trips,
            "mean_static": self.static_sum / self.trips,
            "mean_actual": self.actual_sum / self.trips,
            "mean_ratio": (sum(ordered) / len(ordered)) if ordered else 1.0,
            "p90_ratio": ordered[int(0.9 * (len(ordered) - 1))] if ordered else 1.0,
        }


class ArrivalTracker:
    """Watches state transitions, feeds the arrival models, serves predictions."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._prev_state: Dict[int, State] = {}
        self._human_trips: Dict[int, _OpenTrip] = {}
        self._robot_trips: Dict[int, _OpenTrip] = {}
        self._human_at_rack: Dict[int, int] = {}  # human_id -> tick reached rack
        self._robot_parked_tick: Dict[int, int] = {}  # robot_id -> tick it parked
        self._human_trip_lengths: Deque[float] = deque(maxlen=200)

        self.human_model: Optional[ServiceRiskModel] = None
        self.robot_model: Optional[ServiceRiskModel] = None
        self.staging_robot_model: Optional[ServiceRiskModel] = None
        self.human_stats = ArrivalStats()
        self.robot_stats = ArrivalStats()
        self.prediction_stats: Dict[str, PredictionStats] = {}

        # Rolling realised/static ratios, used both as a model feature and as
        # the cold-start estimate before any model is trained.
        self._human_ratio_history: Dict[int, List[float]] = {}
        self._robot_ratio_history: Dict[int, List[float]] = {}
        self._history_window = 40

        # Aggregate diagnostics for the run report.
        self.human_rack_idle_ticks = 0
        self.rendezvous_gap_sum = 0
        self.rendezvous_count = 0
        self.early_human_arrivals = 0
        self.late_human_arrivals = 0
        self._overdue_checks: set[Tuple[int, int]] = set()
        self._overdue_fires: set[Tuple[int, int]] = set()

    # -- model wiring -------------------------------------------------------

    def ensure_models(self, backend: str, consumer: str = "assignment") -> None:
        """Attach models without letting one consumer replace another's backend."""

        config = getattr(self.engine, "config", None)
        if consumer == "staging":
            if self.staging_robot_model is not None and self.staging_robot_model.backend == backend:
                return
            if self.robot_model is not None and self.robot_model.backend == backend:
                self.staging_robot_model = self.robot_model
            else:
                self.staging_robot_model = build_risk_model(config, backend, ARRIVAL_FEATURES)
            return

        if self.human_model is None or self.human_model.backend != backend:
            self.human_model = build_risk_model(config, backend, ARRIVAL_FEATURES)
        if self.robot_model is None or self.robot_model.backend != backend:
            if self.staging_robot_model is not None and self.staging_robot_model.backend == backend:
                self.robot_model = self.staging_robot_model
            else:
                self.robot_model = build_risk_model(config, backend, ARRIVAL_FEATURES)

    # -- feature construction ----------------------------------------------

    def _congestion(self) -> Tuple[int, int, int]:
        engine = self.engine
        active = moving = 0
        for robot_id in getattr(engine, "robot_id_list", []):
            robot = engine.agents[robot_id - 1]
            if robot.state not in (State.NOOP, State.HOME):
                active += 1
            if robot.state in (State.ROBOT_MOVESPOT, State.ROBOT_MOVEGOAL, State.ROBOT_MOVEQUEUE):
                moving += 1
        busy_humans = 0
        for human_id in getattr(engine, "human_id_list", []):
            human = engine.agents[human_id - 1]
            if human.state not in (State.NOOP, State.HOME):
                busy_humans += 1
        return active, moving, busy_humans

    def _ratio_history(self, is_human: bool, agent_id: int) -> List[float]:
        store = self._human_ratio_history if is_human else self._robot_ratio_history
        return store.setdefault(agent_id, [])

    def build_features(
        self,
        agent_id: int,
        zone_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        planned_path_len: int,
        is_human: bool,
        static_ticks: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Feature row for one prospective trip."""

        engine = self.engine
        if static_ticks is None:
            grid = _human_distance_grid(engine, start)
            static_ticks = _pair_distance(grid, start, goal)
        manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        detour = (static_ticks / manhattan) if manhattan > 0 else 1.0

        history = self._ratio_history(is_human, agent_id)
        if history:
            ordered = sorted(history)
            mean_ratio = sum(ordered) / len(ordered)
            p90_ratio = ordered[int(0.9 * (len(ordered) - 1))]
        else:
            mean_ratio, p90_ratio = 1.0, 1.0

        active, moving, busy = self._congestion()
        return {
            "agent_id": int(agent_id),
            "zone_id": int(zone_id),
            "static_ticks": float(static_ticks),
            "manhattan": float(manhattan),
            "detour_ratio": float(detour),
            "planned_path_len": float(planned_path_len),
            "recent_ratio_mean": float(mean_ratio),
            "recent_ratio_p90": float(p90_ratio),
            "recent_samples": float(len(history)),
            "active_robots": float(active),
            "moving_robots": float(moving),
            "busy_humans": float(busy),
            "elapsed_ticks": float(getattr(engine, "internal_timer", 0)),
        }

    # -- prediction ---------------------------------------------------------

    def predict(
        self,
        rows: List[Dict[str, Any]],
        is_human: bool,
        consumer: str = "assignment",
    ) -> List[Tuple[float, float]]:
        """(Q50, Q90) arrival ticks for a batch of prospective trips."""

        if not rows:
            return []
        if consumer == "staging" and not is_human:
            model = self.staging_robot_model
        else:
            model = self.human_model if is_human else self.robot_model
        if model is None or model.backend in ("static", "fixed"):
            return [(row["static_ticks"], row["static_ticks"]) for row in rows]

        if model.backend == "oracle":
            # No realised value is knowable in advance for a trip that has not
            # started, so the oracle falls back to the calibrated ratio.
            return [self._ratio_estimate(row, is_human) for row in rows]

        if model.backend in CATBOOST_BACKENDS and model.model_ready:
            predictions = model.predict_batch(rows)
            return [
                (max(row["static_ticks"], q50), max(max(row["static_ticks"], q50), q90))
                for row, (q50, q90) in zip(rows, predictions)
            ]

        estimates = [self._ratio_estimate(row, is_human) for row in rows]
        if model.backend == "catboost_median":
            return [(q50, q50) for q50, _q90 in estimates]
        return estimates

    def _ratio_estimate(self, row: Dict[str, Any], is_human: bool) -> Tuple[float, float]:
        """Cold-start ladder: agent history -> global history -> static."""

        agent_id = int(row["agent_id"])
        static_ticks = float(row["static_ticks"])
        history = self._ratio_history(is_human, agent_id)
        if len(history) < 5:
            pooled: List[float] = []
            store = self._human_ratio_history if is_human else self._robot_ratio_history
            for values in store.values():
                pooled.extend(values)
            history = pooled
        if len(history) < 5:
            return static_ticks, static_ticks
        ordered = sorted(history)
        q50 = ordered[int(0.5 * (len(ordered) - 1))]
        q90 = ordered[int(0.9 * (len(ordered) - 1))]
        return max(static_ticks, static_ticks * q50), max(static_ticks, static_ticks * q90)

    def median_human_trip(self) -> float:
        """Rolling median worker trip length in ticks, 0 before any data.

        Used as the travel budget for a speculative dispatch: committing a
        worker to a robot that is farther than a typical trip costs more walking
        than the pairing can save, because walking is 36% of worker time and
        workers are the bottleneck resource.
        """

        if not self._human_trip_lengths:
            return 0.0
        ordered = sorted(self._human_trip_lengths)
        return float(ordered[len(ordered) // 2])

    def current_robot_wait(self, robot_id: int, tick: int) -> float:
        """Ticks this robot has been parked at its rack without service.

        ``Agent.waiting_time`` is a run-long cumulative counter (it reaches
        ~28,000 by the end of a full run and varies by only ~300 between
        robots), so it cannot express which robot is starving right now. This
        does.
        """

        parked = self._robot_parked_tick.get(robot_id)
        return 0.0 if parked is None else float(max(0, tick - parked))

    def is_overdue(self, human_id: int, tick: int) -> bool:
        """True when a worker's trip has run past its own predicted Q90.

        This is the event that unlocks a pairing for re-auction: the worker we
        committed to is demonstrably later than the model expected, so holding
        the robot for them is no longer justified by the original estimate.
        """

        trip = self._human_trips.get(human_id)
        if trip is None or trip.predicted_q90 <= 0:
            return False
        key = (human_id, tick)
        self._overdue_checks.add(key)
        overdue = (tick - trip.start_tick) > trip.predicted_q90
        if overdue:
            self._overdue_fires.add(key)
        return overdue

    # -- observation --------------------------------------------------------

    def observe(self) -> None:
        """Detect trip starts and completions. Call once per tick after step."""

        engine = self.engine
        tick = int(getattr(engine, "internal_timer", 0))

        for human_id in getattr(engine, "human_id_list", []):
            self._observe_human(engine, human_id, tick)
        for robot_id in getattr(engine, "robot_id_list", []):
            self._observe_robot(engine, robot_id, tick)

        models = {
            id(model): model
            for model in (self.human_model, self.robot_model, self.staging_robot_model)
            if model is not None
        }
        for model in models.values():
            model.maybe_retrain()

    def _observe_human(self, engine: Any, human_id: int, tick: int) -> None:
        human = engine.agents[human_id - 1]
        previous = self._prev_state.get(human_id)
        current = human.state
        self._prev_state[human_id] = current

        if current == State.HUMAN_MOVESPOT and previous != State.HUMAN_MOVESPOT:
            if not human.node_list:
                return
            shelf = engine.shelfs[human.node_list[0] - 1]
            goal = (int(shelf.goal_x), int(shelf.goal_y))
            features = self.build_features(
                agent_id=human_id,
                zone_id=self._zone_of(engine, human_id),
                start=(int(human.x), int(human.y)),
                goal=goal,
                planned_path_len=len(getattr(human, "path_planning", []) or []),
                is_human=True,
            )
            predicted = self.predict([features], is_human=True)[0]
            predictions: Dict[str, Tuple[float, float, bool]] = {}
            if self.human_model is not None:
                predictions["assignment_human"] = (
                    predicted[0], predicted[1], self.human_model.model_ready
                )
            self._human_trips[human_id] = _OpenTrip(
                agent_id=human_id,
                start_tick=tick,
                goal=goal,
                features=features,
                static_ticks=features["static_ticks"],
                predicted_q90=predicted[1],
                predictions=predictions,
            )
            return

        trip = self._human_trips.get(human_id)
        if trip is not None and human_id not in self._human_at_rack:
            if (int(human.x), int(human.y)) == trip.goal:
                self._human_at_rack[human_id] = tick
                self._close_trip(trip, tick, is_human=True)
                self._human_trips.pop(human_id, None)

        # Worker reached the rack but the robot has not: count the idle wait.
        if human_id in self._human_at_rack:
            if current == State.HUMAN_MOVESPOT:
                self.human_rack_idle_ticks += 1
            elif current in (State.HUMAN_PICKING, State.NOOP, State.HOME):
                arrival = self._human_at_rack.pop(human_id)
                gap = tick - arrival
                if current == State.HUMAN_PICKING:
                    self.rendezvous_count += 1
                    self.rendezvous_gap_sum += gap
                    if gap > 0:
                        self.early_human_arrivals += 1
                    else:
                        self.late_human_arrivals += 1

    def _observe_robot(self, engine: Any, robot_id: int, tick: int) -> None:
        robot = engine.agents[robot_id - 1]
        previous = self._prev_state.get(robot_id)
        current = robot.state
        self._prev_state[robot_id] = current

        if current == State.ROBOT_MOVESPOT and previous != State.ROBOT_MOVESPOT:
            if not robot.node_list:
                return
            shelf = engine.shelfs[robot.node_list[0] - 1]
            goal = (int(shelf.goal_x), int(shelf.goal_y))
            features = self.build_features(
                agent_id=robot_id,
                zone_id=self._zone_of(engine, robot_id),
                start=(int(robot.x), int(robot.y)),
                goal=goal,
                planned_path_len=len(getattr(robot, "path_planning", []) or []),
                is_human=False,
            )
            predictions: Dict[str, Tuple[float, float, bool]] = {}
            if self.robot_model is not None:
                q50, q90 = self.predict([features], is_human=False)[0]
                predictions["assignment_robot"] = (
                    q50, q90, self.robot_model.model_ready
                )
            if self.staging_robot_model is not None:
                q50, q90 = self.predict(
                    [features], is_human=False, consumer="staging"
                )[0]
                predictions["staging_robot"] = (
                    q50, q90, self.staging_robot_model.model_ready
                )
            self._robot_trips[robot_id] = _OpenTrip(
                agent_id=robot_id,
                start_tick=tick,
                goal=goal,
                features=features,
                static_ticks=features["static_ticks"],
                predictions=predictions,
            )
            return

        if current == State.ROBOT_PICKING and previous != State.ROBOT_PICKING:
            self._robot_parked_tick[robot_id] = tick
            trip = self._robot_trips.pop(robot_id, None)
            if trip is not None:
                self._close_trip(trip, tick, is_human=False)
        elif current != State.ROBOT_PICKING:
            self._robot_parked_tick.pop(robot_id, None)

    def _close_trip(self, trip: _OpenTrip, tick: int, is_human: bool) -> None:
        actual = max(1, tick - trip.start_tick)
        stats = self.human_stats if is_human else self.robot_stats
        stats.add(trip.static_ticks, actual)

        for label, (q50, q90, ready) in trip.predictions.items():
            self.prediction_stats.setdefault(label, PredictionStats()).add(
                q50, q90, actual, ready
            )

        if is_human:
            self._human_trip_lengths.append(trip.static_ticks)

        if trip.static_ticks > 0:
            history = self._ratio_history(is_human, trip.agent_id)
            history.append(actual / trip.static_ticks)
            if len(history) > self._history_window:
                del history[0]

        candidate_models = (
            (self.human_model,) if is_human
            else (self.robot_model, self.staging_robot_model)
        )
        models = {id(model): model for model in candidate_models if model is not None}
        for model in models.values():
            model.observe(trip.features, actual)

    @staticmethod
    def _zone_of(engine: Any, agent_id: int) -> int:
        agent = engine.agents[agent_id - 1]
        node_id = engine.routing_node_dict.get((agent.x, agent.y))
        return int(node_id) if node_id is not None else -1

    # -- reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        human = self.human_stats.summary()
        robot = self.robot_stats.summary()
        mean_gap = (
            self.rendezvous_gap_sum / self.rendezvous_count if self.rendezvous_count else 0.0
        )
        report = {
            "human_trips": human["trips"],
            "human_mean_static_ticks": round(human["mean_static"], 2),
            "human_mean_actual_ticks": round(human["mean_actual"], 2),
            "human_mean_ratio": round(human["mean_ratio"], 4),
            "robot_trips": robot["trips"],
            "robot_mean_static_ticks": round(robot["mean_static"], 2),
            "robot_mean_actual_ticks": round(robot["mean_actual"], 2),
            "robot_mean_ratio": round(robot["mean_ratio"], 4),
            "human_rack_idle_ticks": self.human_rack_idle_ticks,
            "rendezvous_count": self.rendezvous_count,
            "mean_human_early_ticks": round(mean_gap, 2),
            "early_human_arrivals": self.early_human_arrivals,
            "late_human_arrivals": self.late_human_arrivals,
            "human_model": self.human_model.stats() if self.human_model else None,
            "robot_model": self.robot_model.stats() if self.robot_model else None,
            "staging_robot_model": (
                self.staging_robot_model.stats() if self.staging_robot_model else None
            ),
            "overdue_unique_checks": len(self._overdue_checks),
            "overdue_unique_fires": len(self._overdue_fires),
        }
        for label, stats in self.prediction_stats.items():
            report[f"prediction_{label}"] = {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in stats.summary().items()
            }
        return report
