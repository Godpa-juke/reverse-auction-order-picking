"""Pre-positioning of idle workers toward anticipated demand.

Why
---
A worker that finishes a job enters ``NOOP`` at the rack where the job ended;
the legacy HOME return is never entered on that transition. This last-job
location is already an adaptive baseline: measured on the auction run, its
next-trip distance averages 18.4 ticks, while the best fixed waiting cell per
worker measures 30.3 ticks/trip. The median idle gap is only 6 ticks, so useful
pre-positioning must start immediately and must target likely near-term demand.

What makes that tractable is that demand is largely *observable* rather than
merely predictable: a robot in ``ROBOT_MOVESPOT`` already knows which rack it is
driving to, and that rack is a future picking request. The open question is
which of those requests this particular worker will get, and how soon -- which
is what the arrival model already estimates.

Nothing here commits a worker. Staging only changes where an idle worker waits;
it stays freely assignable, so unlike predictive dispatch it cannot starve a
robot that has already parked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rware.core import State
from rware.engine.human_assignment import _human_distance_grid, _pair_distance

# Policy names accepted by ``config.staging_policy``.
POLICIES = ("off", "nearest", "learned", "oracle")


class StagingPlanner:
    """Chooses where each idle worker should wait."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.policy = str(getattr(engine.config, "staging_policy", "off") or "off").lower()
        if self.policy not in POLICIES:
            self.policy = "off"
        self.early_weight = float(getattr(engine.config, "staging_early_weight", 0.5))
        self.uncertainty_weight = float(
            getattr(engine.config, "staging_uncertainty_weight", 0.5)
        )
        self.eta_backend = str(
            getattr(engine.config, "staging_eta_backend", "catboost") or "catboost"
        ).lower()
        # Diagnostics for the run report.
        self.decisions = 0
        self.moved_workers = 0
        self.staged_ticks = 0

    @property
    def enabled(self) -> bool:
        return self.policy != "off"

    # -- demand ------------------------------------------------------------

    def _demand(self) -> List[Tuple[int, Tuple[int, int], float, float]]:
        """Open picking requests as (robot_id, rack cell, ready_q50, ready_q90).

        A parked robot is ready now. An en-route robot is ready when it arrives,
        estimated by the arrival model when one is attached and by remaining
        path length otherwise. The spread between the two quantiles is how
        uncertain that arrival is, which the learned policy prices.
        """

        engine = self.engine
        tracker = getattr(engine, "arrival_tracker", None)
        use_model = self.policy == "learned" and tracker is not None

        rows: List[Tuple[int, Tuple[int, int], float, float]] = []
        pending: List[Tuple[int, Tuple[int, int], Any]] = []

        for robot_id in getattr(engine, "robot_id_list", []):
            robot = engine.agents[robot_id - 1]
            if robot.coworker is not None or not robot.node_list:
                continue
            if robot.state not in (State.ROBOT_PICKING, State.ROBOT_MOVESPOT):
                continue
            shelf = engine.shelfs[robot.node_list[0] - 1]
            cell = (int(shelf.goal_x), int(shelf.goal_y))
            if robot.state == State.ROBOT_PICKING:
                rows.append((robot_id, cell, 0.0, 0.0))
            else:
                pending.append((robot_id, cell, robot))

        if not pending:
            return rows

        if use_model:
            features = [
                tracker.build_features(
                    agent_id=robot.id,
                    zone_id=-1,
                    start=(int(robot.x), int(robot.y)),
                    goal=cell,
                    planned_path_len=len(getattr(robot, "path_planning", []) or []),
                    is_human=False,
                )
                for _, cell, robot in pending
            ]
            predictions = tracker.predict(features, is_human=False, consumer="staging")
            for (robot_id, cell, _), (q50, q90) in zip(pending, predictions):
                rows.append((robot_id, cell, float(q50), float(q90)))
        else:
            for robot_id, cell, robot in pending:
                remaining = float(len(getattr(robot, "path_planning", []) or []))
                rows.append((robot_id, cell, remaining, remaining))

        return rows

    # -- target selection ---------------------------------------------------

    def _score(
        self,
        distance: float,
        ready_q50: float,
        ready_q90: float,
    ) -> float:
        """Cost of waiting at one request; lower is better."""

        if self.policy == "nearest":
            return distance

        # Arriving before the robot does is not free: the worker is parked at a
        # rack that is not yet serviceable and is a worse starting point for
        # whatever request comes up meanwhile. Charge the overshoot.
        early = max(0.0, ready_q50 - distance)
        # An arrival the model is unsure about is worth committing to less: the
        # spread between the quantiles is how wrong the wait could be.
        uncertainty = max(0.0, ready_q90 - ready_q50)
        return (
            distance
            + self.early_weight * early
            + self.uncertainty_weight * uncertainty
        )

    def _oracle_target(self, human: Any) -> Optional[Tuple[int, int]]:
        """Upper bound: walk at the rack this worker is actually sent to next.

        Not implementable in practice; it exists to bound what perfect
        anticipation would be worth.
        """

        future = getattr(self.engine, "_staging_oracle_next", None)
        if not future:
            return None
        return future.get(human.id)

    # -- per-tick update ----------------------------------------------------

    def update(self) -> None:
        """Assign a staging target to every idle, unassigned worker."""

        if not self.enabled:
            return

        engine = self.engine
        idle: List[Any] = []
        for human_id in getattr(engine, "human_id_list", []):
            human = engine.agents[human_id - 1]
            if human.coworker is not None:
                continue
            if human.state not in (State.HOME, State.NOOP):
                continue
            idle.append(human)

        if not idle:
            return

        if self.policy == "oracle":
            for human in idle:
                human.staging_target = self._oracle_target(human)
                self.decisions += 1
            return

        demand = self._demand()
        if not demand:
            for human in idle:
                human.staging_target = None
            return

        # Score every (worker, request) pair, then claim greedily from the best
        # pair down. Iterating workers in id order instead would let an
        # arbitrary worker claim a request that a nearer one wanted.
        pairs: List[Tuple[float, int, int]] = []
        grids = {human.id: _human_distance_grid(self.engine, (int(human.x), int(human.y)))
                 for human in idle}
        cells: List[Tuple[int, int]] = []
        for index, (_robot_id, cell, ready_q50, ready_q90) in enumerate(demand):
            cells.append(cell)
            for human in idle:
                distance = _pair_distance(grids[human.id], (int(human.x), int(human.y)), cell)
                pairs.append((self._score(distance, ready_q50, ready_q90), human.id, index))

        pairs.sort()
        taken_humans: set = set()
        taken_cells: set = set()
        chosen: Dict[int, Tuple[int, int]] = {}
        for _score, human_id, index in pairs:
            if human_id in taken_humans or index in taken_cells:
                continue
            taken_humans.add(human_id)
            taken_cells.add(index)
            chosen[human_id] = cells[index]

        for human in idle:
            target = chosen.get(human.id)
            self.decisions += 1
            if target is None:
                human.staging_target = None
                continue
            if human.staging_target != target:
                self.moved_workers += 1
            human.staging_target = target
            if (int(human.x), int(human.y)) != target:
                self.staged_ticks += 1

    def report(self) -> Dict[str, Any]:
        return {
            "staging_policy": self.policy,
            "staging_eta_backend": self.eta_backend,
            "staging_decisions": self.decisions,
            "staging_retargets": self.moved_workers,
            "staging_moving_ticks": self.staged_ticks,
        }
