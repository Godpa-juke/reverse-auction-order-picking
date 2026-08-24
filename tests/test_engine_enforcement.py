import os
import sys

import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.warehouse import Warehouse, Direction, Action, RewardType

# 7x7: robot r at (5,4) faces walls; human h at (1,4)
MAP_TEXT = """
[base]
ppppppp
pg....p
p.....p
p.....p
ph...rp
p.....p
ppppppp
[overlay]
.......
.......
.......
....#W.
.......
....2..
.......
"""


def make_env():
    env = Warehouse(3, 8, 3, 2, 1, 1, 1, 1, 5, None, None, RewardType.GLOBAL, layout=MAP_TEXT)
    env.reset()
    return env


def _find(env, is_human):
    return next(a for a in env.agents if a.agent_type == is_human)


def _noop_actions(env):
    # msg_bits=1 in this constructor, so each action is [action, msg]
    return [np.array([Action.NOOP.value, 0], dtype="int64") for _ in range(env.n_agents)]


def test_robot_blocked_by_transparent_wall():
    env = make_env()
    robot = _find(env, is_human=False)
    robot.x, robot.y = 4, 4
    robot.dir = Direction.UP
    env._recalc_grid()
    actions = _noop_actions(env)
    actions[robot.id - 1] = np.array([Action.UP.value, 0], dtype="int64")  # target (4,3) is '#'
    env.step(actions)
    assert (robot.x, robot.y) == (4, 4)


def test_any_agent_blocked_by_solid_wall():
    env = make_env()
    human = _find(env, is_human=True)
    human.x, human.y = 5, 2
    human.dir = Direction.DOWN
    env._recalc_grid()
    actions = _noop_actions(env)
    actions[human.id - 1] = np.array([Action.DOWN.value, 0], dtype="int64")  # target (5,3) is 'W'
    env.step(actions)
    assert (human.x, human.y) == (5, 2)


def test_robot_blocked_by_direction_mask():
    env = make_env()
    robot = _find(env, is_human=False)
    robot.x, robot.y = 4, 5   # cell (4,5) allows RIGHT only ('2')
    robot.dir = Direction.LEFT
    env._recalc_grid()
    actions = _noop_actions(env)
    actions[robot.id - 1] = np.array([Action.LEFT.value, 0], dtype="int64")
    env.step(actions)
    assert (robot.x, robot.y) == (4, 5)


def test_robot_allowed_direction_passes():
    env = make_env()
    robot = _find(env, is_human=False)
    robot.x, robot.y = 4, 5
    robot.dir = Direction.RIGHT
    env._recalc_grid()
    actions = _noop_actions(env)
    actions[robot.id - 1] = np.array([Action.RIGHT.value, 0], dtype="int64")
    env.step(actions)
    assert (robot.x, robot.y) == (5, 5)
