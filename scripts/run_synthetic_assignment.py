#!/usr/bin/env python3
"""Run a deterministic, data-free reverse-auction assignment example."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rware.core import State
from rware.engine.human_assignment import (
    HumanSnapshot,
    RobotSnapshot,
    _solve_assignment_by_auction,
    _solve_assignment_greedy,
    get_human_assignment_strategy,
)


def main() -> None:
    humans = [
        HumanSnapshot(1, (0, 0), 10, 0, [], State.NOOP),
        HumanSnapshot(2, (8, 0), 4, 0, [], State.NOOP),
    ]
    robots = [
        RobotSnapshot(3, (3, 0), State.ROBOT_PICKING, 20, 1, 2, 14.0),
        RobotSnapshot(4, (7, 0), State.ROBOT_PICKING, 5, 2, 1, 7.0),
    ]
    engine = SimpleNamespace(config=SimpleNamespace(auction_zone_penalty=0.0))
    strategy = get_human_assignment_strategy("auction")
    costs = strategy.build_cost_matrix(engine, humans, robots, context=None)
    values = [[-cost for cost in row] for row in costs]
    auction_assignment = _solve_assignment_by_auction(
        values, epsilon=0.01, time_limit_s=0.0, max_bid_updates=20_000
    )
    greedy_assignment = _solve_assignment_greedy(values)
    result = {
        "human_ids": [human.id for human in humans],
        "robot_ids": [robot.id for robot in robots],
        "cost_matrix": costs,
        "auction_robot_index_by_human": auction_assignment,
        "greedy_robot_index_by_human": greedy_assignment,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
