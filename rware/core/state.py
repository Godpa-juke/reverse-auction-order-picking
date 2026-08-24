"""
State definitions for Robotic Warehouse Simulation
로봇과 인간 에이전트의 상태, 방향, 행동을 정의하는 모듈
"""

from enum import Enum


class Direction(Enum):
    """에이전트 이동 방향 정의"""
    UP        = 0
    UPRIGHT   = 1
    RIGHT     = 2
    DOWNRIGHT = 3
    DOWN      = 4
    DOWNLEFT  = 5
    LEFT      = 6
    UPLEFT    = 7


class Action(Enum):
    """에이전트 행동 정의"""
    UP        = 0
    UPRIGHT   = 1
    RIGHT     = 2
    DOWNRIGHT = 3
    DOWN      = 4
    DOWNLEFT  = 5
    LEFT      = 6
    UPLEFT    = 7
    NOOP      = 8
    PICKING   = 9


class State(Enum):
    """에이전트 상태 정의"""
    NOOP     = 0
    ROBOT_MOVESPOT          = 1
    ROBOT_MOVEZONE          = 2
    ROBOT_MOVEQUEUE         = 3
    ROBOT_MOVEGOAL          = 4
    ROBOT_DROP              = 5
    ROBOT_PICKING           = 6
    ROBOT_LOAD              = 7
    HUMAN_MOVESPOT          = 8
    HUMAN_PICKING           = 9
    HUMAN_DONE              = 10
    HOME                    = 11
    TIMEOUT                 = 12
