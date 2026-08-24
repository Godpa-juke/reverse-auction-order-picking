"""Unit tests for the reverse-auction ablation strategies.

These exercise the cost function and the solver directly with hand-built
snapshots, so they run in milliseconds and need no simulator.
"""

from types import SimpleNamespace

import pytest

from rware.core import State
from rware.engine.human_assignment import (
    HumanSnapshot,
    RobotSnapshot,
    _solve_assignment_greedy,
    get_human_assignment_strategy,
)
from rware.engine.warehouse_engine import WarehouseEngine


def test_greedy_solver_is_myopic_where_the_auction_is_optimal():
    # Maximisation. Taking the single best pair (b0, i0) strands b1 on a
    # zero-value item; the optimal assignment gives up 1 to gain 8.
    values = [
        [10.0, 9.0],
        [8.0, 0.0],
    ]
    assert _solve_assignment_greedy(values) == [0, 1]


def test_greedy_solver_handles_an_empty_matrix():
    assert _solve_assignment_greedy([]) == []


def test_auction_strategy_defaults_to_the_auction_solver():
    strategy = get_human_assignment_strategy("auction")
    assert strategy.solver == "auction"
    # The optimal assignment on this matrix trades 1 for 8.
    assert strategy._solve([[10.0, 9.0], [8.0, 0.0]]) == [1, 0]


# The real config always carries auction_zone_penalty=0.0 (defaults.py); with
# config=None the strategy would fall back to its 10000.0 class attribute and
# the zone test below would pass for the wrong reason.
FAKE_ENGINE = SimpleNamespace(config=SimpleNamespace(auction_zone_penalty=0.0))

HUMAN = HumanSnapshot(
    id=1,
    position=(0, 0),
    agent_timer=10,
    waiting_time=0,
    zone_nodes=[],
    state=State.NOOP,
)

ROBOT = RobotSnapshot(
    id=2,
    position=(3, 4),
    state=State.ROBOT_PICKING,
    waiting_time=20,
    routing_node_id=1,
    pending_items=2,
    estimated_service_time=14.0,
)


def _cost(strategy_name: str, human=HUMAN, robot=ROBOT) -> float:
    strategy = get_human_assignment_strategy(strategy_name)
    matrix = strategy.build_cost_matrix(FAKE_ENGINE, [human], [robot], None)
    return matrix[0][0]


@pytest.mark.parametrize(
    "name, expected",
    [
        # distance 7 + 0.1*14 - 0.2*20 - 0.05*10
        ("auction", 3.9),
        # same cost function, different matcher
        ("auction_greedy", 3.9),
        # urgency term dropped: +4.0
        ("auction_no_urgency", 7.9),
        # fairness term dropped: +0.5
        ("auction_no_fairness", 4.4),
        # service term dropped: -1.4
        ("auction_no_service", 2.5),
        # distance only
        ("auction_distance_only", 7.0),
        # same distance-only cost, greedy matcher
        ("auction_distance_only_greedy", 7.0),
    ],
)
def test_ablation_strategies_drop_exactly_one_term(name, expected):
    assert _cost(name) == pytest.approx(expected)


def test_every_ablation_strategy_is_registered_under_its_own_name():
    for name in (
        "auction_greedy",
        "auction_no_urgency",
        "auction_no_fairness",
        "auction_no_service",
        "auction_distance_only",
        "auction_distance_only_greedy",
        "auction_global_only",
        "auction_zone_on",
    ):
        # get_human_assignment_strategy falls back to nearest_idle on a miss,
        # so an unregistered name shows up as the wrong object here.
        assert get_human_assignment_strategy(name).name == name


def test_global_only_disables_reauction_even_when_config_enables_it():
    strategy = get_human_assignment_strategy("auction_global_only")
    config = SimpleNamespace(auction_reauction_enabled=True)

    assert strategy.is_reauction_enabled(config) is False


def test_engine_honors_strategy_reauction_override_when_collecting_robots():
    calls = []
    engine = SimpleNamespace(
        config=SimpleNamespace(auction_reauction_enabled=True),
        _human_assignment_strategy=get_human_assignment_strategy("auction_global_only"),
        _pending_human_assignments={},
        predictive_dispatch_enabled=False,
        _collect_waiting_humans=lambda include=None: [],
        _collect_available_robots=lambda include_with_coworker=False: calls.append(
            include_with_coworker
        )
        or [],
    )

    WarehouseEngine._refresh_assignment_plan(engine)

    assert calls == [False]


def test_zone_on_prices_a_zone_violation_and_the_baseline_does_not():
    zoned_human = HumanSnapshot(
        id=1,
        position=(0, 0),
        agent_timer=10,
        waiting_time=0,
        zone_nodes=[5],
        state=State.NOOP,
    )
    # ROBOT.routing_node_id is 1, which is outside the worker's zone.
    assert _cost("auction", human=zoned_human) == pytest.approx(3.9)
    assert _cost("auction_zone_on", human=zoned_human) == pytest.approx(10003.9)
