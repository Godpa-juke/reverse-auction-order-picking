import random
import time
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
from typing import List, Tuple, Optional, Dict
import networkx as nx
import os
from rware.source.site_b.wrap import *
from rware.algorithm.path_planning.jps import *
# from rware.algorithm.path_planning.jps_modified import *
from datetime import datetime
import matplotlib.pyplot as plt

# 필요한 매크로 변수 선언
_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2
_COLLISION_LAYERS = 4

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_SPOTS  = 2
_LAYER_HUMAN  = 3

_FIRST_PICKING_STATION = 0
_SECOND_PICKING_STATION = 1

_SHELF_VERTICAL_IDX = True
_MAPPING_HORIZONTAL = False
_PICKING_CAPACITIY = 1
_DROP_CAPACITY = 1
_ROBOT_MAX_CAPACITY = 10000000000000

_DISTANCE_PER_GRID = 1.5
_TICKPERTIME = 1.0

_TIMEOUT_VALUE = 5
_TIMEOUT_VALUE_START = 10
_TIMEOUT_VALUE_END   = 12


_PRODUCTIVITY_FACTOR = 15
_SEQUENCE_PARAM      = 4
_MAX_ZONE_CNT        = 4

global big_asile
global small_asile

# Admin Controller

# 0: Rendering Off, 1: Rendering On
_RENDERING_FLAG = 1

# 0: Batch, 1: Random  2: Small SEQ NODE(Beta)
_ORDER_BATCH_FLAG = 1

# 0: All, 1: Big, 2: Small
_HUMAN_ZONE_FLAG   = 0

# 0: Noseq, 1: Order Seq, 2: Batch Seq(Not Implement)
_ORDER_SEQ_FLAG    = 1,

# 0: NoRoute, 1: Fork Route
_CHK_ROUTING_FLAG  = 0

# 0: Human Move, 1: Human Do not Move
_SELECT_HUMAN_MOVE  = 1

# 0: 2_Way, 1: 1_Way(LG)
_SELECT_STATIC_PATH = 1

# 0: Collision not Allow, 1: Collision Allow
_SELECT_PICKING_COLLISION = 1

_LOADBOX_CNT = 4
_BOX_PER_LOADING_TIME = 1
_BOX_LOADING_TIME = 1

_SKU_PER_PICKING_TIME = 30
_SKU_PER_EXIT_TIME = 1
_TEST_PICKING_TIME = 1

# _SKU_PER_EXIT_TIME = _LOADBOX_CNT * _SKU_PER_PICKING_TIME


dx = [0,1,1,1,0,-1,-1,-1]
dy = [-1,-1,0,1,1,1,0,-1]

random.seed(42)

# VectorWriter 클래스 정의
class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits

# Action 클래스 정의
class Action(Enum):
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


# Direction Enum 정의
class Direction(Enum):
    UP        = 0
    UPRIGHT   = 1
    RIGHT     = 2
    DOWNRIGHT = 3
    DOWN      = 4
    DOWNLEFT  = 5
    LEFT      = 6
    UPLEFT    = 7

# State Enum 정의:
class State(Enum):
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

# RewardType Enum 정의(사용x)
class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

# ObserationType Enum 정의(사용x)
class ObserationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2

# ImageLayer Enum 정의(사용x)
class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """
    SHELVES = 0         # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1        # binary layer indicating requested shelves
    AGENTS = 2          # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3 # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4      # binary layer indicating agents with load
    GOALS = 5           # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6      # binary layer indicating accessible cells (all but occupied cells/ out of map)

#Entity 클래스 정의
class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = x
        self.prev_y = y
        self.x = x
        self.y = y

# Agent(로봇) 클래스 정의
class Agent(Entity):
    counter = 0

    # 생성자
    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = Action.NOOP
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False

        self.state = State.NOOP
        self.pre_state = State.NOOP

        self.shelf = None
        self.done = False
        self.prioirty = False
        self.init_x = self.x
        self.init_y = self.y

        self.target = None
        self.target_zone = None


        # Added Variable
        self.agent_type  = None
        self.agent_timer = 0
        self.coworker    = None
        self.station     = None
        self.loadbox_station = None

        self.routing_node = None
        self.max_capacity   = _ROBOT_MAX_CAPACITY
        self.cur_capacity   = 0
        self.human_pick_cnt = 0
        self.node_list      = list()
        self.path_planning  = list()

        self.total_distance = 0
        self.complete_order = 0
        self.working_time   = 0

        self.order_sku_cnt = list()
        self.cur_sku_cnt   = 0
        self.total_sku_cnt = 0
        self.timeout_cnt = 0
        self.loading_timer = 0
        self.load_box = False
        self.waiting_time = 0

    # collision_layers에 대한 메소드
    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    # Agent loaction 메소드 -> 다음 x,y 좌표 설정하게함
    def req_location(self, grid_size) -> Tuple[int, int]:
        # print("self.req_action : ",self.req_action,", self.dir : ",self.dir,", Timer: ",env.internal_timer)
        # print("grid_size : ",grid_size)
        if self.req_action == Action.UP and self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)

        elif self.req_action == Action.UPRIGHT and self.dir == Direction.UPRIGHT:
            return min(grid_size[1] - 1, self.x + 1), max(0, self.y - 1)

        elif self.req_action == Action.RIGHT and self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        elif self.req_action == Action.DOWNRIGHT and self.dir == Direction.DOWNRIGHT:
            return min(grid_size[1] - 1, self.x + 1), min(grid_size[0] - 1, self.y + 1)

        elif self.req_action == Action.DOWN and self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)

        elif self.req_action == Action.DOWNLEFT and self.dir == Direction.DOWNLEFT:
            return max(0, self.x - 1), min(grid_size[0] - 1, self.y + 1)

        elif self.req_action == Action.LEFT and self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y

        elif self.req_action == Action.UPLEFT and self.dir == Direction.UPLEFT:
            return max(0, self.x - 1), max(0, self.y - 1)
        else:
            return self.x, self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    # Agent direction 메소드 -> 다음 방향 설정 -> Rotate counter-clockwise /clockwise
    # Do not need!
    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.UPRIGHT, Direction.RIGHT, Direction.DOWNRIGHT,
                    Direction.DOWN, Direction.DOWNLEFT, Direction.LEFT, Direction.UPLEFT]

        if self.req_action.value < len(wraplist): return wraplist[self.req_action.value]
        else: return self.dir
        # if self.req_action == Action.RIGHT:
        #     return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        # elif self.req_action == Action.LEFT:
        #     return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        # else:
        #     return self.dir

    # Direction을 Num으로 전환하는 함수
    def Direction2Num(self):
        if self.dir == Direction.UP:          return 0
        elif self.dir == Direction.UPRIGHT:   return 1
        elif self.dir == Direction.RIGHT:     return 2
        elif self.dir == Direction.DOWNRIGHT: return 3
        elif self.dir == Direction.DOWN:      return 4
        elif self.dir == Direction.DOWNLEFT:  return 5
        elif self.dir == Direction.LEFT:      return 6
        elif self.dir == Direction.UPLEFT:    return 7
        else : return None

    # 다음 액션 생성 -> make new method
    def next_action(self, env, human_map, robot_map):
        # Do not go inside Rack
        # For Human
        if self.agent_type == True: cur_map = human_map  # Make Maze for Path Planning
        else : cur_map = robot_map  # Make Maze for Path Planning

        dx = [0,1,1,1,0,-1,-1,-1]
        dy = [-1,-1,0,1,1,1,0,-1]

        # Human type
        if self.agent_type == True:
            if self.state == State.HUMAN_MOVESPOT or self.state == State.HOME:
                target_y, target_x = self.y, self.x

                if self.state == State.HUMAN_MOVESPOT:
                    target_y = env.shelfs[self.node_list[0] - 1].goal_y
                    target_x = env.shelfs[self.node_list[0] - 1].goal_x

                elif self.state == State.HOME:
                    target_y = self.init_y
                    target_x = self.init_x

                cur_dir  = self.dir.value
                # if (self.x, self.y, target_x, target_y, cur_dir) in env.action_map:
                #     value = env.action_map[(self.x, self.y, target_x, target_y, cur_dir)]
                #     if cur_map[self.y + dy[value]][self.x + dx[value]] == 0:
                #         print(self.id, " USE ActionMap")
                #         return np.array([env.action_map[(self.x, self.y, target_x, target_y, cur_dir)], 0], dtype='int64')

                target_value = cur_map[target_y, target_x]
                cur_map[target_y,target_x] = 0
                cur_map[self.y, self.x] = 0

                new_path = jps(cur_map, self.y, self.x, target_y, target_x,cur_dir)
                if new_path is not None and len(new_path) > 2: new_path = new_path[:2]
                self.path_planning = jps_converted_path(cur_map, get_full_path(new_path), cur_dir, True)
                # env.path_list.append(self.path_planning)
                # if self.path_planning is None or len(self.path_planning) <= 1:
                #     self.path_planning = aStar(cur_map, (self.y, self.x), (target_y, target_x), cur_dir, env.internal_timer)

                cur_map[self.y, self.x] = 1
                cur_map[target_y, target_x] = target_value

                if self.path_planning is None or len(self.path_planning) <= 1: return np.array([Action.NOOP.value, 0], dtype='int64')
                else: return np.array([self.path_planning[1][2], 0], dtype='int64')

            else:
                return np.array([Action.NOOP.value, 0], dtype='int64')

        # Robot Type
        else:
            if self.state == State.ROBOT_MOVESPOT or self.state == State.ROBOT_MOVEGOAL or self.state == State.HOME or self.state == State.ROBOT_MOVEZONE:
                target_y, target_x  = self.y, self.x
                routing_path = []
                load_station_block = [cur_map[bl][58] for bl in range(12,19)]
                if (self.x, self.y) in env.loadbox_queue:
                    for bl in range(12, 19): cur_map[bl][58] = 0
                else:
                    for bl in range(12, 19): cur_map[bl][58] = 1

                if self.state == State.ROBOT_MOVESPOT:
                    if self.x == self.prev_x and self.y == self.prev_y:
                        self.agent_timer = self.agent_timer + 1
                    else:
                        self.agent_timer = 0

                    if len(self.node_list) > 0:
                        target_y = env.shelfs[self.node_list[0] - 1].goal_y
                        target_x = env.shelfs[self.node_list[0] - 1].goal_x

                    else:

                        if self.load_box == True:
                            target_y = self.station[1] + 2
                            target_x = self.station[0]
                        else:
                            target_y = self.loadbox_station[1]
                            target_x = self.loadbox_station[0] + 2

                    for wait_x, wait_y in env.wait_queue: cur_map[wait_y][wait_x] = 1

                    if self.agent_timer > _TIMEOUT_VALUE_START:
                        print("Timeout!",self.id)
                        if self.agent_timer > _TIMEOUT_VALUE_END:
                            self.agent_timer = 0
                            self.timeout_cnt += 1
                            cur_idx = env.routing_node_dict[(self.x, self.y)]
                            env.total_timeout_cnt_in_zone[cur_idx] += 1
                        else:
                            if env.grid[_LAYER_SPOTS, self.y, self.x] in [6]: return np.array([Action.RIGHT.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x] in [7]: return np.array([Action.LEFT.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x] in [8]: return np.array([Action.DOWNRIGHT.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x] in [9]: return np.array([Action.UPLEFT.value, 0], dtype='int64')

                            else:
                                if env.internal_timer % 2 == 0: return np.array([Action.UP.value, 0], dtype='int64')
                                else: return np.array([Action.DOWN.value, 0], dtype='int64')

                    if _SELECT_STATIC_PATH == 1:
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [6, 8]:
                            if  self.x == target_x and target_y == self.y + 1: return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y - 1: return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3: return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3: return np.array([Action.DOWN.value, 0], dtype='int64')
                            else:return np.array([Action.RIGHT.value, 0], dtype='int64')
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [7, 9]:
                            if self.x == target_x and target_y == self.y - 1: return np.array([Action.UP.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y + 1: return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3:return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3:return np.array([Action.DOWN.value, 0], dtype='int64')
                            else: return np.array([Action.LEFT.value, 0], dtype='int64')

                    # print(self.id,(target_x,target_y))

                            # tmp = num_list[:start]
                            # num_list = num_list[start:]
                            # num_list = num_list + tmp
                            #
                            # for dir in num_list:
                            #     next_x = self.x + dx[dir]
                            #     next_y = self.y + dy[dir]
                            #
                            #     if next_x < 0 or next_y < 0 or next_x >= env.grid_size[1]  or next_y >= env.grid_size[0]: continue
                            #     if cur_map[next_y][next_x] == 0:
                            #         return np.array([dir, 0], dtype='int64')
                            #     else:
                            #         return np.array([Action.NOOP.value, 0], dtype='int64')



                elif self.state == State.ROBOT_MOVEGOAL:
                    if self.load_box == True:
                        target_y = self.station[1] + 2
                        target_x = self.station[0]
                    else:
                        target_y = self.loadbox_station[1]
                        target_x = self.loadbox_station[0] + 2

                    if _SELECT_STATIC_PATH == 1:
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [6, 8]:
                            if self.x == target_x and target_y == self.y + 1:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y - 1:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            else:
                                return np.array([Action.RIGHT.value, 0], dtype='int64')
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [7, 9]:
                            if self.x == target_x and target_y == self.y - 1:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y + 1:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            else:
                                return np.array([Action.LEFT.value, 0], dtype='int64')

                elif self.state == State.HOME:
                    if _SELECT_STATIC_PATH == 1:
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [6, 8]:
                            if self.x == target_x and target_y == self.y + 1:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y - 1:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            else:
                                return np.array([Action.RIGHT.value, 0], dtype='int64')
                        if env.grid[_LAYER_SPOTS, self.y, self.x] in [7, 9]:
                            if self.x == target_x and target_y == self.y - 1:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif self.x == target_x and target_y == self.y + 1:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x - 1] == 3:
                                return np.array([Action.UP.value, 0], dtype='int64')
                            elif env.grid[_LAYER_SPOTS, self.y, self.x + 1] == 3:
                                return np.array([Action.DOWN.value, 0], dtype='int64')
                            else:
                                return np.array([Action.LEFT.value, 0], dtype='int64')

                    target_y = self.init_y
                    target_x = self.init_x
                    cur_map[21][66:73] = 1
                    # for wait_x, wait_y in env.wait_queue: cur_map[wait_y][wait_x] = 1



                cur_dir = self.dir.value
                # if (self.x, self.y, target_x, target_y, cur_dir) in env.action_map:
                #     value = env.action_map[(self.x, self.y, target_x, target_y, cur_dir)]
                #     if cur_map[self.y + dy[value]][self.x + dx[value]] == 0:
                #         return np.array([env.action_map[(self.x, self.y, target_x, target_y, cur_dir)], 0], dtype='int64')

                target_value = cur_map[target_y, target_x]
                cur_map[target_y, target_x] = 0
                cur_map[self.y, self.x] = 0



                new_path = []
                new_path = jps(cur_map, self.y, self.x, target_y, target_x,cur_dir)

                motion_flag = True

                if env.grid[_LAYER_SHELFS,self.y+1, self.x] > 0 or env.grid[_LAYER_SHELFS,self.y-1, self.x] > 0 or env.grid[_LAYER_SPOTS,self.y+1, self.x] == 5 or env.grid[_LAYER_SPOTS,self.y-1, self.x] == 5: motion_flag = False
                # if env.grid[_LAYER_SHELFS, self.y + 1, self.x] > 0 or env.grid[_LAYER_SHELFS, self.y - 1, self.x] > 0 : motion_flag = False

                if new_path is not None and len(new_path) > 2: new_path = new_path[:2]
                full_path = get_full_path(new_path)
                if motion_flag == True:
                    self.path_planning = jps_converted_path(cur_map, full_path, cur_dir, True)
                else :
                    self.path_planning = jps_converted_path2(cur_map, full_path, cur_dir, False)

                # env.path_list.append(self.path_planning)
                if self.path_planning is None or len(self.path_planning) <= 1:
                    cur_map[self.y, self.x] = 1
                    cur_map[target_y, target_x] = target_value
                    return np.array([Action.NOOP.value, 0], dtype='int64')

                for bl in range(len(load_station_block)): cur_map[bl+12][58] = load_station_block[bl]
                cur_map[self.y, self.x] = 1
                cur_map[target_y, target_x] = target_value

                if self.state == State.HOME: cur_map[21][66:73] = 0
                # if self.state == State.ROBOT_MOVEZONE and len(self.node_list) <= 0:
                #     cur_map[self.station[1] + 2, self.station[0] + 1] = 0
                #     cur_map[self.station[1] + 1, self.station[0] + 1] = 0
                #     cur_map[self.station[1] + 0, self.station[0] + 1] = 0

                if self.path_planning is None or len(self.path_planning) <= 1:
                    return np.array([Action.NOOP.value, 0], dtype='int64')
                else:
                    return np.array([self.path_planning[1][2], 0], dtype='int64')



            elif self.state == State.ROBOT_MOVEQUEUE:
                if self.load_box == True:
                    if self.x > self.station[0]:
                        return np.array([Action.LEFT.value, 0], dtype='int64')
                    elif self.x < self.station[0]:
                        return np.array([Action.RIGHT.value, 0], dtype='int64')
                    else:
                        return np.array([Action.UP.value, 0], dtype='int64')
                else:
                    if  self.y > self.loadbox_station[1]:
                        return np.array([Action.UP.value, 0], dtype='int64')
                    elif self.y < self.loadbox_station[1]:
                        return np.array([Action.DOWN.value, 0], dtype='int64')
                    else:
                        return np.array([Action.LEFT.value, 0], dtype='int64')
            else:
                return np.array([Action.NOOP.value, 0], dtype='int64')

    # 로봇 상황 체크 메소드
    def check_status(self, env, input_order):
        if self.x != self.prev_x or self.y != self.prev_y: self.total_distance += 1

        # Robot
        if self.agent_type == False:
            # AMR에 작업 할당됨 -> 이동 상태로 변함
            if len(input_order) > 0 and (self.state == State.NOOP or self.state == State.HOME):
                self.state = State.ROBOT_MOVESPOT
                # self.state = State.ROBOT_MOVEZONE

                if self.loadbox_station is None:
                    self.select_loadbox_station(env)

                if self.load_box == False and self.loadbox_station is not None:
                    self.state = State.ROBOT_MOVEGOAL


            # AMR이 목표 위치에 도착 -> 피킹 상태로 변함
            elif len(self.node_list) > 0 and self.state == State.ROBOT_MOVESPOT:
                if env.shelfs[self.node_list[0] - 1].goal_x == self.x and env.shelfs[self.node_list[0] - 1].goal_y == self.y:
                    self.state = State.ROBOT_PICKING
                    self.waiting_time += 1

            # 피킹 상태에서 작업 수행이 끝나면 배출지/다음 노드 이동
            elif len(self.node_list) > 0 and self.state == State.ROBOT_PICKING:
                self.waiting_time += 1
                if self.coworker is not None and env.agents[self.coworker - 1].state == State.HUMAN_DONE:  # 사람이 작업을 끝낸 경우,
                    if len(self.node_list) > 1:  # 최소 2개 이상 node 존재
                        self.waiting_time -= (self.order_sku_cnt[0] * _SKU_PER_PICKING_TIME - 1)
                        env.agents[self.coworker - 1].node_list = list()
                        env.agents[self.coworker - 1].state = State.NOOP
                        env.agents[self.coworker - 1].coworker = None
                        env.agents[self.coworker - 1].complete_order += 1

                        self.cur_sku_cnt = self.cur_sku_cnt + self.order_sku_cnt[0]
                        del self.order_sku_cnt[0]
                        del self.node_list[0]  # AMR에서 현재 작업한 노드 제거
                        self.coworker = None  # AMR에서 협업 대상 제거

                        self.state = State.ROBOT_MOVESPOT  # 다음 노드로 이동
                        # self.state = State.ROBOT_MOVEZONE  # 다음 노드로 이동
                        self.complete_order += 1

                    else:
                        self.waiting_time -= (self.order_sku_cnt[0] * _SKU_PER_PICKING_TIME - 1)
                        env.agents[self.coworker - 1].node_list = list()
                        env.agents[self.coworker - 1].state = State.NOOP
                        env.agents[self.coworker - 1].coworker = None

                        self.cur_sku_cnt = self.cur_sku_cnt + self.order_sku_cnt[0]
                        self.order_sku_cnt = list()
                        self.node_list = list()  # 현재 작업한 노드 제거
                        self.coworker = None

                        self.state = State.ROBOT_MOVEGOAL
                        # self.state = State.ROBOT_MOVEZONE
                        if self.station is None:
                            self.select_goals(env)
                            print("select_goal : ",self.id,self.station)

                          # 배출 대기구로 이동
                else: pass  # Just Wait for human

            elif self.state == State.ROBOT_MOVEGOAL :
                if self.load_box == True and [self.x, self.y] == [self.station[0], self.station[1] + 2]:
                    self.state = State.ROBOT_MOVEQUEUE
                    station_idx = env.goals.index(self.station)
                    env.wait_queue_cnt[station_idx] += 1

                elif self.load_box == False and [self.x, self.y] == [self.loadbox_station[0]+2, self.loadbox_station[1]]:
                    self.state = State.ROBOT_MOVEQUEUE
                    loadbox_station_idx = env.loadbox_queue.index(self.loadbox_station)
                    # env.wait_loadbox_cnt[loadbox_station_idx] += 1
                    # print("waitbox_cnt : ",env.wait_loadbox_cnt)

            # AMR이 배출지에 도착 -> 배출 DROP으로 변함
            elif self.state == State.ROBOT_MOVEQUEUE:
                if self.load_box == True and [self.x, self.y] == [self.station[0], self.station[1]]:
                    self.state = State.ROBOT_DROP
                    self.loading_timer = _SKU_PER_EXIT_TIME - 1

                if self.load_box == False and [self.x, self.y] == [self.loadbox_station[0], self.loadbox_station[1]]:
                    self.state = State.ROBOT_LOAD
                    self.loading_timer = _BOX_LOADING_TIME - 1

            # AMR이 배출지에서 배출 작업 수행
            elif self.state == State.ROBOT_DROP and [self.station[0], self.station[1]] == [self.x, self.y]:
                if self.loading_timer > 0:
                    self.loading_timer = self.loading_timer - 1

                else:
                    self.state = State.HOME
                    self.load_box = False
                    self.total_sku_cnt = self.total_sku_cnt + self.cur_sku_cnt
                    env.all_of_completed_order = env.all_of_completed_order + self.cur_sku_cnt
                    self.cur_sku_cnt = 0
                    self.loading_timer = 0
                    env.completed_batch = env.completed_batch + 1
                    env.completed_batch_log.append([env.internal_timer,env.completed_batch])
                    station_idx = env.goals.index(self.station)
                    env.wait_queue_cnt[station_idx] -= 1
                    self.station = None
                    if self.loadbox_station is None: self.select_loadbox_station(env)

            elif self.state == State.ROBOT_LOAD and [self.loadbox_station[0], self.loadbox_station[1]] == [self.x, self.y]:
                if self.loading_timer > 0:
                    self.loading_timer = self.loading_timer - 1

                else:
                    self.state = State.HOME
                    if env.next_order_cnt > 0:
                        self.load_box = True
                        self.node_list = [j[0] for j in input_order[0]]
                        self.order_sku_cnt = [j[1] for j in input_order[0]]
                        del input_order[0]
                        self.state = State.ROBOT_MOVESPOT
                        # self.state = State.ROBOT_MOVEZONE


                    load_station_idx = env.loadbox_queue.index(self.loadbox_station)
                    env.wait_loadbox_cnt[load_station_idx] -= 1
                    self.loadbox_station = None

            elif self.state == State.HOME and [self.init_x, self.init_y] == [self.x, self.y]: self.state = State.NOOP

        # Human
        else:
            # AMR의 작업 요청이 할당 -> 작업자 이동
            if (self.state == State.NOOP or self.state == State.HOME) and self.coworker == None:
                self.agent_timer += 1
                if self.agent_timer > 3:
                    self.coworker = env.select_coworker(env.agents[self.id - 1])  # 작업자 대상자 선정 -> AMR에 할당


                # 협업할 작업자가 존재 시,
                if self.coworker is not None:
                    env.agents[self.coworker - 1].coworker = self.id
                    self.node_list.append(env.agents[self.coworker - 1].node_list[0])
                    self.agent_timer = 0
                    if _SELECT_HUMAN_MOVE == 0 : self.state = State.HUMAN_MOVESPOT
                    else:
                        self.state = State.HUMAN_PICKING
                        self.loading_timer = (env.agents[self.coworker - 1].order_sku_cnt[0] * _SKU_PER_PICKING_TIME) - 1

            # 작업자가 목표한 노드에 도달 -> AMR이 피킹상태면 작업 시작 아니면 대기
            elif self.state == State.HUMAN_MOVESPOT and env.shelfs[self.node_list[0] - 1].goal_x == self.x and \
                env.shelfs[self.node_list[0] - 1].goal_y == self.y:
                if env.agents[self.coworker - 1].state == State.ROBOT_PICKING:
                    self.loading_timer = (env.agents[self.coworker - 1].order_sku_cnt[0] * _SKU_PER_PICKING_TIME) - 1
                    # self.loading_timer = _TEST_PICKING_TIME
                    self.state = State.HUMAN_PICKING


            # 작업자가 피킹 작업 수행 -> 작업 완료 시 변함
            elif self.state == State.HUMAN_PICKING:
                if self.loading_timer - 1 > 0:
                    self.loading_timer = self.loading_timer - 1

                else:
                    self.loading_timer = 0
                    self.total_sku_cnt = self.total_sku_cnt + env.agents[self.coworker - 1].order_sku_cnt[0]
                    self.state = State.HUMAN_DONE
                    # time.sleep(10)
            elif self.state == State.HUMAN_DONE and \
                (env.agents[self.coworker - 1].state == State.ROBOT_MOVEGOAL or \
                 env.agents[self.coworker - 1].state == State.ROBOT_MOVESPOT):
                self.node_list = list()  # 작업자에서 노드 제거
                self.coworker = None  # 작업자에서 협업 대상 제거
                self.state = State.HOME

            elif self.state == State.HOME and self.init_x == self.x and self.init_y == self.y:
                self.state = State.NOOP

    # 노드 변경 메소드
    def node_change(self):
        if len(self.node_list)>0:
            tmp = self.node_list[0]
            self.node_list.append(tmp)
            del self.node_list[0]

    def select_goals(self,env):
        cur_map = Make_Maze(env,2)
        if self.station is None and len(self.node_list)<= 0 and self.state == State.ROBOT_MOVEGOAL and self.load_box == True:
            select_list = list()
            for sample_station in env.goals:
                sample_idx = env.goals.index(sample_station)
                delta_y = abs(sample_station[1] - self.y)
                delta_x = abs(sample_station[0] - self.x)
                select_list.append([env.wait_queue_cnt[sample_idx],delta_y+delta_x,sample_station])
            select_list = sorted(select_list)
            self.station = select_list[0][2]
        return

    def select_loadbox_station(self,env):
        cur_map = Make_Maze(env,2)
        if self.loadbox_station is None and len(self.node_list)<= 0 and self.state == State.ROBOT_MOVESPOT:
            select_list = list()
            for sample_station in env.loadbox_queue:
                sample_idx = env.loadbox_queue.index(sample_station)
                delta_y = abs(sample_station[1] - self.y)
                delta_x = abs(sample_station[0] - self.x)
                select_list.append([env.wait_loadbox_cnt[sample_idx],delta_y+delta_x,sample_station,sample_idx])
            select_list = sorted(select_list)

            self.loadbox_station = select_list[0][2]
            env.wait_loadbox_cnt[select_list[0][3]] += 1

        return



# 랙 클래스 정의
class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.init_x = self.x
        self.init_y = self.y

        # Added Variable
        self.goal_x = self.x
        self.goal_y = self.y

    # collision_layers에 대한 get 메소드
    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


# 웨어하우스 클래스 정의
class Warehouse(gym.Env):

    # human : render to the current display or terminal and return nothing. Usually for human consumption.
    # rgb_array :  Return an numpy. ndarray with shape (x, y, 3),
    #              representing RGB values for an x-by-y pixel image, suitable for turning into a video.
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        n_humans: int,
        n_robots: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: str = None,
        observation_type: ObserationType=ObserationType.FLATTENED,
        image_observation_layers: List[ImageLayer]=[
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE
        ],
        image_observation_directional: bool=True,
        normalised_coordinates: bool=False,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.internal_timer = 0
        self.using_station = [0, 0]
        self.using_agent = 0
        self.completed_batch = 0
        self.all_of_completed_order = 0

        # Modified Jw.son 2022.07.14
        # Add Tuple's thrid element for ID
        self.goals: List[Tuple[int, int]] = []
        self.picking_queue: List[Tuple[int, int]] = []
        self.loadbox_queue: List[Tuple[int, int]] = []

        self.wait_queue: List[Tuple[int, int]] = []
        self.shelf_queue:  List[Tuple[int, int]] = []
        self.wait_queue_cnt = []
        self.human_init_queue: List[Tuple[int, int]] = []
        self.robot_init_queue: List[Tuple[int, int]] = []

        self.routing_node = []
        self.routing_node_all_pos = []
        self.routing_node_dict = dict()
        self.routing_graph = []
        self.node_identifier = []
        self.edge_map = []
        self.rack_list = []
        self.zone_list_in_rack = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.n_agents = n_agents
        self.n_humans = n_humans
        self.n_robots = n_robots
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates
        self.shelfs_table = None
        sa_action_space = [len(Action), *msg_bits * (2,)]

        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.observation_space = None
        if observation_type == ObserationType.IMAGE:
            self._use_image_obs(image_observation_layers, image_observation_directional)
        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if observation_type == ObserationType.FLATTENED:
            self._use_fast_obs()

        self.renderer = None
        self.running_robot_cnt = 0
        self.running_human_cnt = 0
        self.action_map = dict()
        self.total_robot_cnt_in_zone = []
        self.total_timeout_cnt_in_zone = []

        self.not_used_shelf = []
        self.total_map_cnt = None
        self.completed_batch_log = []
        self.path_list = []
        self.next_order_cnt = 0
    # 파라미터를 통한 레이아웃 설계 메소드
    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (self.column_height + 1) == 0)  # horizontal highways
            or (y == self.grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > self.grid_size[0] - (self.column_height + 3))
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
        )
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)


    # 문자열을 통한 레이아웃 설계 메소드
    def _make_layout_from_str(self, layout):

        vertical_idx = _SHELF_VERTICAL_IDX
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1 # row
        lines = layout.split("\n")
        grid_width = len(lines[0])           # col

        vector = ['' for _ in range(grid_width)]

        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        if vertical_idx == False:
            for col in range(grid_width):
                for line in lines:
                    vector[col] = vector[col] + str(line[col])

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        if vertical_idx == True:
            for y, line in enumerate(lines):
                for x, char in enumerate(line):
                    assert char.lower() in "gpwboemzxrnh."
                    if char.lower() == "g":
                        self.grid[_LAYER_SPOTS,y,x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif char.lower() == "w":
                        self.grid[_LAYER_SPOTS,y,x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "p":
                        self.grid[_LAYER_SPOTS,y,x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "b":
                        self.grid[_LAYER_SPOTS,y,x] = 4
                        self.loadbox_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "n":
                        self.grid[_LAYER_SPOTS,y,x] = 5
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1

                    elif char.lower() == "o":
                        self.grid[_LAYER_SPOTS,y,x] = 6
                        self.highways[y, x] = 1
                    elif char.lower() == "e":
                        self.grid[_LAYER_SPOTS,y,x] = 7
                        self.highways[y, x] = 1
                    elif char.lower() == "m":
                        self.grid[_LAYER_SPOTS,y,x] = 8
                        self.highways[y, x] = 1
                    elif char.lower() == "z":
                        self.grid[_LAYER_SPOTS,y,x] = 9
                        self.highways[y, x] = 1

                    elif char.lower() == "x":
                        self.shelf_queue.append((x, y))
                    elif char.lower() == "h":
                        self.human_init_queue.append((x, y))
                    elif char.lower() == "r":
                        self.robot_init_queue.append((x, y))
                    elif char.lower() == ".":
                        self.highways[y, x] = 1
        else:
            for x, line in enumerate(vector):
                for y, char in enumerate(line):
                    assert char.lower() in "gpwxrh."
                    if char.lower() == "g":
                        self.grid[_LAYER_SPOTS,y,x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif char.lower() == "w":
                        self.grid[_LAYER_SPOTS,y,x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "p":
                        self.grid[_LAYER_SPOTS,y,x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1


                    elif char.lower() == "b":

                        self.grid[_LAYER_SPOTS, y, x] = 4

                        self.loadbox_queue.append((x, y))

                        self.highways[y, x] = 1

                    elif char.lower() == "n":

                        self.grid[_LAYER_SPOTS, y, x] = 5

                        self.picking_queue.append((x, y))

                        self.highways[y, x] = 1


                    elif char.lower() == "o":
                        self.grid[_LAYER_SPOTS, y, x] = 6
                        self.highways[y, x] = 1

                    elif char.lower() == "e":
                        self.grid[_LAYER_SPOTS, y, x] = 7
                        self.highways[y, x] = 1

                    elif char.lower() == "m":
                        self.grid[_LAYER_SPOTS, y, x] = 8
                        self.highways[y, x] = 1

                    elif char.lower() == "z":
                        self.grid[_LAYER_SPOTS, y, x] = 9
                        self.highways[y, x] = 1

                    elif char.lower() == "x":
                        self.shelf_queue.append((x, y))
                    elif char.lower() == "h":
                        self.human_init_queue.append((x, y))
                    elif char.lower() == "r":
                        self.robot_init_queue.append((x, y))
                    elif char.lower() == ".":
                        self.highways[y, x] = 1

        self.wait_queue_cnt = [0 for _ in range(len(self.goals))]
        self.wait_loadbox_cnt = [0 for _ in range(len(self.loadbox_queue))]
        assert len(self.goals) >= 1, "At least one goal is required"

    # 이미지 관측 레이어 설정 메소드
    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max([d.value + 1 for d in Direction])
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        self.observation_space = spaces.Tuple(
            tuple([spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    # 이미지 레이어 사용 메소드(slow)
    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        if self.normalised_coordinates:
            location_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
            )
        else:
            location_space = spaces.MultiDiscrete(
                [self.grid_size[1], self.grid_size[0]]
            )

        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Dict(
                        OrderedDict(
                            {
                                "self": spaces.Dict(
                                    OrderedDict(
                                        {
                                            "location": location_space,
                                            "carrying_shelf": spaces.MultiDiscrete([2]),
                                            # "direction": spaces.Discrete(4),
                                            "direction": spaces.Discrete(8),
                                            "on_highway": spaces.MultiBinary(1),
                                        }
                                    )
                                ),
                                "sensors": spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (
                                        spaces.Dict(
                                            OrderedDict(
                                                {
                                                    "has_agent": spaces.MultiBinary(1),
                                                    # "direction": spaces.Discrete(4),
                                                    "direction": spaces.Discrete(8),
                                                    "local_message": spaces.MultiBinary(
                                                        self.msg_bits
                                                    ),
                                                    "has_shelf": spaces.MultiBinary(1),
                                                    "shelf_requested": spaces.MultiBinary(
                                                        1
                                                    ),
                                                }
                                            )
                                        ),
                                    )
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    # 이미지 레이어 사용 메소드(fast)
    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    # 랙 지정 위치 검증 메소드
    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]
    # 관측 레이어 생성 메소드
    def _make_obs(self, agent):
        if self.image_obs:
            # write image observations
            if agent.id == 1:
                layers = []
                # first agent's observation --> update global observation layers
                for layer_type in self.image_observation_layers:
                    if layer_type == ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0

                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0

                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0

                    elif layer_type == ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            agent_direction = ag.dir.value + 1
                            layer[ag.x, ag.y] = float(agent_direction)

                    elif layer_type == ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.carrying_shelf is not None:
                                layer[ag.x, ag.y] = 1.0

                    elif layer_type == ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal_y, goal_x in self.goals:
                            layer[goal_x, goal_y] = 1.0

                    elif layer_type == ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            layer[ag.y, ag.x] = 0.0

                    # pad with 0s for out-of-map cells
                    layer = np.pad(layer, self.sensor_range, mode="constant")
                    layers.append(layer)
                self.global_layers = np.stack(layers)

            # global information was generated --> get information for agent
            start_x = agent.y
            end_x = agent.y + 2 * self.sensor_range + 1
            start_y = agent.x
            end_y = agent.x + 2 * self.sensor_range + 1
            obs = self.global_layers[:, start_x:end_x, start_y:end_y]

            if self.image_observation_directional:
                # rotate image to be in direction of agent
                if agent.dir == Direction.DOWN:
                    # rotate by 180 degrees (clockwise)
                    obs = np.rot90(obs, k=2, axes=(1,2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1,2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1,2))
                # no rotation needed for UP direction
            return obs

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            obs = _VectorWriter(self.observation_space[agent.id - 1].shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, int(agent.carrying_shelf is not None)])
            # direction = np.zeros(4)
            direction = np.zeros(8)

            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    obs.skip(1)
                    obs.write([1.0])
                    obs.skip(3 + self.msg_bits)
                else:
                    obs.write([1.0])
                    # direction = np.zeros(4)
                    direction = np.zeros(8)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)
                if id_shelf == 0:
                    obs.skip(2)
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )

            return obs.vector
 
        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs
    # 그리드 재계산 메소드
    def _recalc_grid(self):
        self.grid[_LAYER_SHELFS] = 0
        self.grid[_LAYER_AGENTS] = 0

        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    # 웨어하우스 리셋 메소드
    def reset(self,initSettingFlag=True):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.internal_timer=0
        self.total_map_cnt = [[0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        self.shelfs = [
            Shelf(x, y)
            for x, y in self.shelf_queue
            if not self._is_highway(x, y)
        ]


        # Made by Jw.son 2022.07.23
        # Make Agent Initial Position
        if initSettingFlag == True:
            agent_locs = []
            for human_sample in self.human_init_queue: agent_locs.append(human_sample[0] + self.grid_size[1] * human_sample[1])
            for robot_sample in self.robot_init_queue: agent_locs.append(robot_sample[0] + self.grid_size[1] * robot_sample[1])

            agent_locs = np.unravel_index(agent_locs, self.grid_size)

            # Direction Information
            # UP = 0, UPRIGHT = 1, RIGHT = 2, DOWNRIGHT = 3
            # DOWN = 4, LEFTDOWN = 5, LEFT = 6, UPLEFT = 7
            agent_dirs = []

            for _ in range(self.n_agents): agent_dirs.append(Direction.LEFT)

            self.agents = [
                Agent(x, y, dir_, self.msg_bits)
                for y, x, dir_ in zip(*agent_locs, agent_dirs)
            ]

        else:
            # spawn agents at random locations
            # location의 index 정보 랜덤 생성
            self._recalc_grid()
            # agent_locs = np.random.choice(
            #     np.arange(self.grid_size[0] * self.grid_size[1]),
            #     size=self.n_agents,
            #     replace=False,
            # )

            already_located = list()
            already_located_cnt = 0
            for r in range(self.grid_size[0]):
                for c in range(self.grid_size[1]):
                    if self.grid[_LAYER_SPOTS,r,c] > 0: already_located.append(already_located_cnt)
                    if self.grid[_LAYER_SHELFS, r, c] > 0: already_located.append(already_located_cnt)
                    already_located_cnt = already_located_cnt + 1

            agent_locs = list()
            for i in range(self.n_agents):
                sample = random.randint(12*self.grid_size[1],self.grid_size[0]*self.grid_size[1]-1)
                while sample in already_located: sample = random.randint(12*self.grid_size[1],self.grid_size[0]*self.grid_size[1]-1)
                already_located.append(sample)
                agent_locs.append(sample)

            # location index정보를 기반으로 grid에 매핑
            #  0  1  2  3  4  5  6  7  8  9  10  11  12  13
            # 14 15 16 17 18 19 20 21 22 23  24  25  26  27
            # ..... like this index mapping

            agent_locs = np.unravel_index(agent_locs, self.grid_size)
            # and direction
            agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents)

            self.agents = [
                Agent(x, y, dir_, self.msg_bits)
                for y, x, dir_ in zip(*agent_locs, agent_dirs)
            ]


        ## Condition Join Point ##
        for idx in range(self.n_humans): self.agents[idx].agent_type = True # Human
        for idx in range(self.n_humans,self.n_agents) : self.agents[idx].agent_type = False

        self._recalc_grid()
        self.request_queue = list()
        return tuple([self._make_obs(agent) for agent in self.agents])

    # 웨어하우스 다음 단계 진행 메소드
    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        self.internal_timer = self.internal_timer + 1
        self.using_agent = 0
        cur_map = Make_Maze(self, mode=3)
        agent_queue = dict()

        failed_agents = list()
        human_commited_agents = list()
        robot_commited_agents = list()
        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        for agent in self.agents:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            # Blocking Next Move at Facility
            if target in self.picking_queue:
                failed_agents.append(agent)
                continue

            # if target in self.loadbox_queue:
            #     failed_agents.append(agent)
            #     continue

            if target in self.shelf_queue:
                failed_agents.append(agent)
                continue

            if target == start:
                failed_agents.append(agent)
                continue

            if agent.agent_type == False:
                if target not in agent_queue:
                    agent_queue[target] = 1
                else:
                    agent_queue[target] = agent_queue[target] + 1

        for agent in failed_agents:
            agent.dir = agent.req_direction()
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)
        motion_list = [Action.UP, Action.UPRIGHT, Action.RIGHT, Action.DOWNRIGHT,
                       Action.DOWN, Action.DOWNLEFT, Action.LEFT, Action.UPLEFT]

        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action in motion_list:
                agent.x, agent.y = agent.req_location(self.grid_size)
                agent.dir = agent.req_direction()
                target_id = self.grid[_LAYER_AGENTS, agent.y, agent.x]
                # Robot
                if agent.agent_type == False:
                    if _SELECT_PICKING_COLLISION == 1:
                        if (agent.x, agent.y) in agent_queue:
                            if agent_queue[(agent.x, agent.y)] >= 2 and self.agents[target_id-1].state != State.ROBOT_PICKING:
                                agent_queue[(agent.x, agent.y)] = agent_queue[(agent.x, agent.y)] - 1
                                agent.x, agent.y = agent.prev_x, agent.prev_y
                                continue

                        if cur_map[agent.y][agent.x] > 0 and self.agents[target_id-1].state != State.ROBOT_PICKING:

                            agent.x = agent.prev_x
                            agent.y = agent.prev_y


                        else:
                            cur_map[agent.y][agent.x] = 1
                            cur_map[agent.prev_y][agent.prev_x] = 0

                    else:
                        if (agent.x, agent.y) in agent_queue:
                            if agent_queue[(agent.x, agent.y)] >= 2:
                                agent_queue[(agent.x, agent.y)] = agent_queue[(agent.x, agent.y)] - 1
                                agent.x, agent.y = agent.prev_x, agent.prev_y
                                continue

                        if cur_map[agent.y][agent.x] > 0:

                            agent.x = agent.prev_x
                            agent.y = agent.prev_y


                        else:
                            cur_map[agent.y][agent.x] = 1
                            cur_map[agent.prev_y][agent.prev_x] = 0

            agent.dir = agent.req_direction()

        # agent 및 shelf 재계산 부분
        self._recalc_grid()
        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]

            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            # remove from queue and replace it
            new_request = np.random.choice(
                list(set(self.shelfs) - set(self.request_queue))
            )
            # Search By Jw.son 2022.07.18
            # request_queue is assigning moving rack
            self.request_queue[self.request_queue.index(shelf)] = new_request

            # Don't Need This Project
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            dones = self.n_agents * [True]
        else:
            dones = self.n_agents * [False]

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}

        if env.grid[_LAYER_AGENTS, self.goals[0][1], self.goals[0][0]] > 0:
            self.using_station[0] += 1
        # if env.grid[_LAYER_AGENTS, self.goals[1][1], self.goals[1][0]] > 0:
        #     self.using_station[1] += 1
        for i in range(self.n_agents):
            if self.agents[i].state != State.NOOP:
                self.using_agent += 1

        return new_obs, list(rewards), dones, info

    # 렌더링 메소드
    def render(self, mode="human"):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    # 렌더링 종료 메소드
    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        ...

    def shelfs_goalpos_remapping(self):
        mapping_horizontal = _MAPPING_HORIZONTAL
        if mapping_horizontal == True:
            # shelfs goal position mapping
            for idx in range(len(self.shelfs)):
                x = self.shelfs[idx].x
                y = self.shelfs[idx].y
                dr = [0,0,1,-1]
                dc = [1,-1,0,0]

                for dir in range(4):
                    next_x = x + dc[dir]
                    next_y = y + dr[dir]

                    if next_x < 0 or next_x >= self.grid_size[1] or next_y < 0 or next_y >= self.grid_size[0]: continue
                    if self.grid[_LAYER_SPOTS, next_y, next_x] in [0,6,7,8,9] and self.grid[_LAYER_SHELFS, next_y, next_x] == 0:
                        self.shelfs[idx].goal_x = next_x
                        self.shelfs[idx].goal_y = next_y
                        break

        else:
            # shelfs goal position mapping
            for idx in range(len(self.shelfs)):
                x = self.shelfs[idx].x
                y = self.shelfs[idx].y
                dr = [1, -1, 0,  0]
                dc = [0, 0,  1, -1]

                for dir in range(4):
                    next_x = x + dc[dir]
                    next_y = y + dr[dir]

                    if next_x < 0 or next_x >= self.grid_size[1] or next_y < 0 or next_y >= self.grid_size[0]: continue
                    if self.grid[_LAYER_SPOTS, next_y, next_x] in [0,6,7,8,9] and self.grid[_LAYER_SHELFS, next_y, next_x] == 0:
                        self.shelfs[idx].goal_x = next_x
                        self.shelfs[idx].goal_y = next_y

                        break

    def select_coworker(self, cur_agent):
        select_list = list()
        cur_map = Make_Maze(env,mode=1)

        working_area = [[x for x in range(len(env.routing_node_all_pos))]]

        if   _HUMAN_ZONE_FLAG == 1: working_area = big_asile
        elif _HUMAN_ZONE_FLAG == 2: working_area = small_asile


        #
        working_area_cnt = len(working_area)

        # Human want Robot
        if cur_agent.agent_type == True:
            for tar_robot in env.agents[env.n_humans:]:
                if tar_robot.state == State.ROBOT_PICKING and tar_robot.coworker is None:
                    if self.routing_node_dict[(tar_robot.x, tar_robot.y)] in working_area[(cur_agent.id - 1) % working_area_cnt]:
                        cur_map[tar_robot.y][tar_robot.x] = 0
                        cur_map[cur_agent.y][cur_agent.x] = 0

                        # new_path = jps(cur_map, cur_agent.y, cur_agent.x, tar_robot.y, tar_robot.x)
                        # planning = jps_converted_path(cur_map, get_full_path(new_path), cur_agent.dir)
                        # if planning is None:
                        #     planning = [0]*100
                        # select_list.append([len(planning), tar_robot.id])
                        delta_x = abs(tar_robot.x - cur_agent.x)
                        delta_y = abs(tar_robot.y - cur_agent.y)
                        if delta_x > 1: delta_x = delta_x * 5
                        select_list.append([delta_x+delta_y, tar_robot.id])
            select_list.sort()
            if len(select_list) > 0:
                return select_list[0][1]
            else:
                return None

            select_list.sort()
            if len(select_list) > 0: return select_list[0][1]
            else: return None

    def making_routing_node(self):
        map = self.grid[_LAYER_SHELFS]
        facility = self.grid[_LAYER_SPOTS]

        # No Pattern Part
        # node_identifier  = [\
        #                              # 1
        #                              [[1, 1], [14, 3]], [[17, 1], [16, 3]], [[19, 1], [26, 3]], [[29, 1], [30, 3]], [[33, 1], [51, 3]], [[54, 1], [55, 3]], [[58, 1], [64, 3]], [[67, 1], [84, 3]],
        #                              # 2
        #                              [[1, 6], [14, 7]], [[17, 6], [16, 7]], [[19, 6], [26, 7]], [[29, 6], [30, 7]], [[33, 6], [51, 7]], [[54, 6], [55, 7]], [[58, 6], [64, 7]], [[67, 6], [84, 7]],
        #                              # 3
        #                              [[1, 10], [14, 11]], [[17, 10], [16, 11]], [[19, 10], [26, 11]], [[29, 10], [30, 11]], [[33, 10], [51, 11]], [[54, 10], [55, 11]], [[58, 10], [64, 11]], [[67, 10], [84, 11]],
        #                              # 4
        #                              [[1, 14], [14, 15]], [[17, 14], [16, 15]], [[19, 14], [26, 15]], [[29, 14], [30, 15]], [[33, 14], [51, 15]],[[54, 14], [55, 15]], [[58, 14], [64, 15]], [[67, 14], [84, 15]],
        #                              # 5
        #                              [[1, 18], [14, 19]], [[17, 18], [16, 19]], [[19, 18], [26, 19]], [[29, 18], [30, 19]], [[33, 18], [51, 19]],[[54, 10], [55, 11]], [[58, 10], [64, 11]], [[67, 10], [84, 11]],
        # ]

        # Pattern Part
        node_identifier = []
        y_list = [(1,3) ,(6,7) ,(10,11),(14,15),(18,19),(22,23),(26,27),(30,31),(34,35),(38,39),(42,43),(46,47),(50,51),(54,55),(58,59),(62,63),(66,67),(70,71),(74,76)]
        x_list = [(1, 14), (17, 16), (19, 26), (29, 30), (33, 51), (54, 57), (60, 64), (67, 72), (75, 84)]

        for srt_y, dst_y in y_list:
            for srt_x, dst_x in x_list:
                node_identifier.append([[srt_x,srt_y],[dst_x,dst_y]])




        self.node_identifier = node_identifier
        self.routing_node = [node_srt_dst[1] for node_srt_dst in node_identifier]
        self.total_robot_cnt_in_zone = [0 for _ in node_identifier]
        self.total_timeout_cnt_in_zone = [0 for _ in node_identifier]
        self.routing_node_all_pos = [[] for _ in range(len(node_identifier))]

        overlap = [[] for _ in range(len(self.shelf_queue)+1)]
        rack_list = [[] for _ in range(len(node_identifier))]

        for idx in range(len(node_identifier)):
            srt_x = node_identifier[idx][0][0]
            srt_y = node_identifier[idx][0][1]
            dst_x = node_identifier[idx][1][0]
            dst_y = node_identifier[idx][1][1]

            for y in range(srt_y-1,dst_y+1+1 ):
                for x in range(srt_x-1,dst_x+1+1):
                    if map[y][x] > 0:
                        rack_list[idx].append(map[y][x])
                        if idx not in self.zone_list_in_rack: self.zone_list_in_rack.append(idx)

                    elif map[y][x] == 0 and (facility[y][x] in [0,1,2,4,5,6,7,8,9]):
                        self.routing_node_all_pos[idx].append((x,y))

        idx = -1
        for routing_node in self.routing_node_all_pos:
            idx = idx + 1
            for x, y in routing_node:
                self.routing_node_dict[(x,y)] = idx

        for idx in range(len(rack_list)):
            if len(rack_list[idx])<= 0: continue
            for sample in rack_list[idx]:
                overlap[sample].append(idx)

        print("########################################################################################## Rack List In Zone #######################################################################################################")
        for idx in range(len(rack_list)): print("zone Id : ",idx,", rack_list : ",rack_list[idx],", length : ", len(rack_list[idx]))
        print()

        print("########################################################################################## Position In Node #######################################################################################################")
        for idx in range(len(self.routing_node_all_pos)):
            print("zone Id : ",idx,", pos cnt : ",len(self.routing_node_all_pos[idx]))
            print("pos list : ",self.routing_node_all_pos[idx])
        print()

        self.rack_list = rack_list
        print(rack_list)
        print(self.zone_list_in_rack)

        return rack_list, overlap

    def making_routing_graph(self):
        # Simply define all of Routing Point with Block Point
        edge_map = list()
        exception_node = [0,1,2,9,10,11,18,19,20,27,28,29,139,140,148,149,157,158,166,167,160,161,169,170]
        for num in range(19):
            tmp = []
            for adder in range(9):
                if (9*num)+adder in exception_node:tmp.append(-1)
                else: tmp.append( (9*num)+adder)
            edge_map.append(tmp)
        print(edge_map)

        self.edge_map = edge_map


        # Calibration Process
        cut_edge_list = []
        cut_edge_list = [\
            # col0
            (36, 45), (45, 54), (54, 63), (63, 72), (72, 81), (81, 90), (90, 99), (99, 108), (108, 117), (117, 126), (126, 135), (135, 144), (144, 153), (153, 162),
            # col1
            # col2
            (38, 47), (47, 56), (56, 65), (65, 74), (74, 83), (83, 92), (92, 101), (101, 110), (110, 119), (119, 128), (128, 137), (137, 146), (146, 155), (155, 164),
            # col3
            (48,57), (57, 66), (66, 75), (75, 84), (84, 93), (93, 102), (102, 111), (111, 120),
            # col4
            (4, 13), (13, 22), (22, 31), (31, 40), (40, 49), (49, 58), (58, 67), (67, 76), (76, 85), (85, 94), (94, 103), (103, 112), (112, 121), (121, 130),
            # col5
            (59, 68), (68, 77), (77, 86), (86, 95), (95, 104),(104, 113), (113, 122), (122, 131),
            # col6
            (6, 15), (15, 24),
            # col7
            (7, 16), (16, 25), (25, 34),
            # col8
            (8, 17), (17, 26),

            # row
            (25,26),(34,35),(42,43),(43,44),(51,52),(52,53),(79,80),(88,89),(97,98),(106,107),

            # Station
            (33,32),(33,24),(33,42)
            ]


        # Check edge list
        dy = [1,-1,0,0]
        dx = [0,0,1,-1]

        edge_list = []
        x_max = len(edge_map[0])
        y_max = len(edge_map)

        for y in range(y_max):
            for x in range(x_max):
                tmp = []
                if edge_map[y][x] < 0: continue
                else:
                    srt_routing_node = edge_map[y][x]
                    for dir in range(4):
                        next_y = y + dy[dir]
                        next_x = x + dx[dir]
                        if next_y < 0 or next_y >= y_max or next_x < 0 or next_x >= x_max: continue
                        if edge_map[next_y][next_x] > 0:
                            dst_routing_node = edge_map[next_y][next_x]
                            if srt_routing_node == dst_routing_node: continue
                            if (srt_routing_node,dst_routing_node) in cut_edge_list or (dst_routing_node,srt_routing_node) in cut_edge_list: continue
                            edge_list.append((srt_routing_node,dst_routing_node,1))
                            tmp.append((srt_routing_node,dst_routing_node,1))
                        else: continue

                print(edge_map[y][x], tmp)

        # time.sleep(100)
        self.routing_graph = nx.Graph()
        self.routing_graph.add_nodes_from([node_num for node_num in range(len(self.routing_node))])
        self.routing_graph.add_weighted_edges_from(edge_list)

        layout = nx.spring_layout(self.routing_graph)  # 각 node의 position을 정해서 그려줘야 edge_label를 맞춰서 넣을 수 있음
        nx.draw_networkx(self.routing_graph, layout)
        labels = nx.get_edge_attributes(self.routing_graph, 'weight')
        nx.draw_networkx_edge_labels(self.routing_graph, layout, edge_labels=labels)


    def check_routing_graph(self):
        robot_in_routing_node = [0 for _ in range(len(self.routing_node))]

        # Edge Clear
        for idx in range(len(robot_in_routing_node)):
            for target in list(self.routing_graph.neighbors(idx)):
                self.routing_graph[idx][target]['weight'] = 1

        # Count Robot Cnt in Routing Node
        for robot in self.agents[self.n_humans:]:
            node_num = self.routing_node_dict[(robot.x,robot.y)]
            robot_in_routing_node[node_num] += 1

        # Update Routing Edge
        for idx in range(len(robot_in_routing_node)):
            if robot_in_routing_node[idx] <= 0: continue
            else:
                edge_neighbor = list(self.routing_graph.neighbors(idx))
                for target in edge_neighbor:
                    # cnt = 2 ** ((robot_in_routing_node[idx] + robot_in_routing_node[target])//2)
                    cnt = ((robot_in_routing_node[idx] + robot_in_routing_node[target]))

                    self.routing_graph[idx][target]['weight'] = cnt
                    self.routing_graph[target][idx]['weight'] = cnt

                    # self.routing_graph[idx][target]['weight'] = 1
                    # self.routing_graph[target][idx]['weight'] = 1

        self.routing_graph[52][43]['weight'] = 1
        self.routing_graph[43][52]['weight'] = 1
        self.routing_graph[34][43]['weight'] = 1
        self.routing_graph[43][34]['weight'] = 1
        self.routing_graph[33][34]['weight'] = 1
        self.routing_graph[34][33]['weight'] = 1

        # if _CHK_ROUTING_FLAG == 1:
        #     self.routing_graph[10][11]['weight'] = 1000
        #     self.routing_graph[11][10]['weight'] = 1000
        #     self.routing_graph[11][12]['weight'] = 1000
        #     self.routing_graph[12][11]['weight'] = 1000

        layout = nx.spring_layout(self.routing_graph)  # 각 node의 position을 정해서 그려줘야 edge_label를 맞춰서 넣을 수 있음
        # nx.draw_networkx(self.routing_graph, layout)
        labels = nx.get_edge_attributes(self.routing_graph, 'weight')
        # nx.draw_networkx_edge_labels(self.routing_graph, layout, edge_labels=labels)

        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.cla()

    def order_sequence(self):
        list_node_counter = [[] for _ in range(len(self.routing_node))]
        for num in range(self.n_humans, self.n_agents):
            if (self.agents[num].state == State.ROBOT_MOVEZONE or self.agents[num].state == State.ROBOT_MOVESPOT or self.agents[num].state == State.ROBOT_PICKING) and len(self.agents[num].node_list)>0 :
                target_y = self.shelfs[self.agents[num].node_list[0] - 1].goal_y
                target_x = self.shelfs[self.agents[num].node_list[0] - 1].goal_x

                target_idx = self.routing_node_dict[(target_x,target_y)]
                list_node_counter[target_idx].append(self.agents[num].id)

                dist = abs(target_y-self.agents[num].y) + abs(target_x-self.agents[num].x)
                if len(list_node_counter[target_idx]) >= _SEQUENCE_PARAM and len(self.agents[num].node_list)>=2 and dist >= 3:
                    # print('-------------------------------------------------------------')
                    # print(list_node_counter)
                    # print(f"order_change! {self.agents[num].node_list} -> ",end=' ')
                    node_list_tmp = self.agents[num].node_list[0]
                    order_sku_cnt_tmp = self.agents[num].order_sku_cnt[0]
                    self.agents[num].node_list.append(node_list_tmp)
                    self.agents[num].order_sku_cnt.append(order_sku_cnt_tmp)
                    del self.agents[num].node_list[0]
                    del self.agents[num].order_sku_cnt[0]
                    del list_node_counter[target_idx][-1]



                    for idx in self.agents[num].node_list[:-1]:
                        target_y  = self.shelfs[idx - 1].goal_y
                        target_x  = self.shelfs[idx - 1].goal_x
                        other_idx = self.routing_node_dict[(target_x, target_y)]
                        list_node_counter[other_idx].append(self.agents[num].id)
                        if len(list_node_counter[other_idx]) >= _SEQUENCE_PARAM and len(self.agents[num].node_list) >= 2:
                            # print('-------------------------------------------------------------')
                            # print(f"order_change! {self.agents[num].node_list} -> ", end=' ')
                            node_list_tmp = self.agents[num].node_list[0]
                            order_sku_cnt_tmp = self.agents[num].order_sku_cnt[0]
                            self.agents[num].node_list.append(node_list_tmp)
                            self.agents[num].order_sku_cnt.append(order_sku_cnt_tmp)
                            del self.agents[num].node_list[0]
                            del self.agents[num].order_sku_cnt[0]
                            del list_node_counter[other_idx][-1]

    def check_running_agent_cnt(self):
        self.running_human_cnt = 0
        self.running_robot_cnt = 0

        for idx in range(self.n_humans):
            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME: continue
            else:
                self.running_human_cnt += 1
                self.agents[idx].working_time += 1

        for idx in range(self.n_humans,self.n_agents):
            # zone cnt
            zone_idx = self.routing_node_dict[(self.agents[idx].x, self.agents[idx].y)]
            self.total_robot_cnt_in_zone[zone_idx] += 1
            self.total_map_cnt[self.agents[idx].y][self.agents[idx].x] += 1

            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME: continue
            else:
                self.running_robot_cnt += 1
                self.agents[idx].working_time += 1


# 패스 생성을 위한 2차원 맵 생성
def Make_Maze(env,mode):
    cfg = getattr(env, "config", None)
    walls = getattr(env, "walls", None)
    enforce = getattr(cfg, "wall_enforce_level", 0) == 2

    arr = np.zeros(env.grid_size)
    if mode == 0:
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[_LAYER_AGENTS, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[_LAYER_SPOTS, i, j] in [3,5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1

    elif mode == 1: # For Human
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                # if env.grid[_LAYER_AGENTS, i, j] <= env.n_humans and env.grid[_LAYER_AGENTS, i, j] > 0:
                #     arr[i][j] = 1

                if env.grid[_LAYER_SHELFS, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[_LAYER_SPOTS, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1

    elif mode == 2:  # For Robot
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[_LAYER_AGENTS, i, j] > env.n_humans:
                    arr[i][j] = 1

                if env.grid[_LAYER_SHELFS, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[_LAYER_SPOTS, i, j] in [3, 5, 8, 9]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] >= 1:
                    arr[i][j] = 1


    elif mode == 3:  # For Robot

        for i in range(env.grid_size[0]):

            for j in range(env.grid_size[1]):

                if env.grid[_LAYER_AGENTS, i, j] > env.n_humans:
                    arr[i][j] = 1

                if env.grid[_LAYER_SHELFS, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[_LAYER_SPOTS, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] >= 1:
                    arr[i][j] = 1

    else :
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[_LAYER_SHELFS, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[_LAYER_SPOTS, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1
    return arr

# Robot and Human Cnt Automated Calc
def AgentCounter(map):
    human_cnt = 0
    robot_cnt = 0

    map_list = map.split()
    for current_row in map_list:
        for idx in range(len(current_row)):
            if current_row[idx] == 'h':
                human_cnt += 1
                continue
            if current_row[idx] == 'r':
                robot_cnt += 1
                continue

    return human_cnt, robot_cnt

def WriteLog(env, robot_cnt, human_cnt, time, stime, etime):

    # Write Pandas
    time_str = str(datetime.now().strftime('%Y%m%d%H%M%S'))
    path = "data/" + time_str
    os.mkdir(path)

    # Agent
    columns = ["Agent ID", "Type", "VisitedRackCnt", "WorkingTime", "WorkingTimeRatio", "TimeoutCnt", "TotalDistance",
               "TotalSkuCnt", "Productivity", "SecPerPick","WaitingTime"]
    data_set = list()
    for i in range(agent_cnt):
        ID_type = None
        WorkingTimeRatio = env.agents[i].working_time / env.internal_timer
        Productivity = (env.agents[i].total_sku_cnt / env.internal_timer) * 60
        SecPerPick = 0
        if env.agents[i].total_sku_cnt > 0: SecPerPick = (env.internal_timer / env.agents[i].total_sku_cnt)

        if env.agents[i].agent_type == True:
            ID_type = 'Human'
        else:
            ID_type = 'Robot'
        data = [f'{env.agents[i].id:>3d}', f'{ID_type}', f'{env.agents[i].complete_order:>5d}',
                f'{env.agents[i].working_time:>5d}', f'{WorkingTimeRatio:.5f}', f'{env.agents[i].timeout_cnt:>2d}',
                f'{env.agents[i].total_distance:>5d}', f'{env.agents[i].total_sku_cnt:>5d}', f'{Productivity:.5f}',
                f'{SecPerPick:.5f}',f'{env.agents[i].waiting_time:>5d}']
        data_set.append(data)

        print("ID :", env.agents[i].id, ", visited_rack_cnt : ", env.agents[i].complete_order, ", working_time : ",
              env.agents[i].working_time, ", total_distance: ", env.agents[i].total_distance, ", total_sku_cnt: ",
              env.agents[i].total_sku_cnt,", waiting_time: ",env.agents[i].waiting_time)

    df_agent = pd.DataFrame(data_set, columns = columns)
    df_agent.to_csv(path + "/" + time_str + "_agent.csv", )

    # Zone
    columns = (["ZONE ID", "ZONE_CNT", "ZONE_AVERAGECNT", "ZONE_TIMEOUTCNT"])
    data_set = list()
    for i in range(len(env.total_robot_cnt_in_zone)):
        value = env.total_robot_cnt_in_zone[i]
        avg = value / env.internal_timer
        timeoutcnt = env.total_timeout_cnt_in_zone[i]
        data = [f"{i:>3d}", f"{value:>5d}", f"{avg:.5f}", f"{timeoutcnt:>3d}\n"]
        data_set.append(data)
    df_zone = pd.DataFrame(data_set, columns = columns)
    df_zone.to_csv(path + "/" + time_str + "_zone.csv")



    # Batch_information
    columns = ["Batch Time", "Batch Value"]
    data_set = list()
    for log in env.completed_batch_log:
        data = ["{0:09d}".format(log[0]), "{0:02d}".format(log[1])]
        data_set.append(data)
    df_batch_information = pd.DataFrame(data_set, columns = columns)
    df_batch_information.to_csv(path + "/" + time_str + "_batch_information.csv")

    # Write Img
    col_length = len(env.edge_map[0])
    row_length = len(env.edge_map)

    data = [[0 for _ in range(col_length)] for _ in range(row_length)]
    for i in range(row_length):
        for j in range(col_length):
            if env.edge_map[i][j] == -1 : continue
            else:
                value = env.total_robot_cnt_in_zone[env.edge_map[i][j]]
                avg = value / env.internal_timer
                data[i][j] = avg

    columns = ['X' + str(x+1).zfill(1) for x in range(col_length)]
    df = pd.DataFrame(data, columns=columns)
    df.index = ['Y' + str(y+1).zfill(1) for y in range(row_length)]

    plt.cla()
    plt.imshow(df,cmap='gray',interpolation='none')
    plt.title("Total Robot Count In Zone")
    plt.colorbar()
    plt.xticks(range(len(df.columns)),df.columns)
    plt.yticks(range(len(df)),df.index)
    plt.savefig(path + "/" + time_str +"_totalRobotCntInZone.png")
    plt.close()


    data = [[0 for _ in range(col_length)] for _ in range(row_length)]
    total_val = 0
    # for val in env.total_timeout_cnt_in_zone: total_val += val
    for i in range(row_length):
        for j in range(col_length):
            if env.edge_map[i][j] == -1:
                continue
            else:
                value = env.total_timeout_cnt_in_zone[env.edge_map[i][j]]
                data[i][j] = value

    columns = ['X' + str(x + 1).zfill(1) for x in range(col_length)]
    df2 = pd.DataFrame(data, columns=columns)
    df2.index = ['Y' + str(y + 1).zfill(1) for y in range(row_length)]

    plt.cla()
    plt.imshow(df2, cmap='gray', interpolation='none')
    plt.title("Total Timeout Count In Zone")
    plt.colorbar()
    plt.xticks(range(len(df2.columns)), df2.columns)
    plt.yticks(range(len(df2)), df2.index)
    plt.savefig(path + "/" + time_str + "_totalTimeoutCntInZone.png")
    plt.close()


    max_val = 0
    for i in range(len(env.total_map_cnt)):
        if max(env.total_map_cnt[i]) > max_val:
            max_val = max(env.total_map_cnt[i])
    data = [[0 for _ in range(len(env.total_map_cnt[0]))] for _ in range(len(env.total_map_cnt))]

    for i in range(len(env.total_map_cnt)):
        for j in range(len(env.total_map_cnt[0])):
            value = ((max_val-env.total_map_cnt[i][j])/max_val)*100
            data[i][j] = int(value)

    columns = ['X' + str(x + 1).zfill(1) for x in range(len(env.total_map_cnt[0]))]
    df3 = pd.DataFrame(data, columns=columns)
    df3.index = ['Y' + str(y + 1).zfill(1) for y in range(len(env.total_map_cnt))]

    plt.cla()
    plt.imshow(df3, cmap='gray', interpolation='none')
    plt.title("Robot Positions")
    plt.colorbar()
    plt.savefig(path + "/" + time_str + "_RobotPositions(Reverse_Ratio).png")
    plt.close()

    abs_val = 5000
    for i in range(len(env.total_map_cnt)):
        if max(env.total_map_cnt[i]) > max_val:
            max_val = max(env.total_map_cnt[i])

    data = [[0 for _ in range(len(env.total_map_cnt[0]))] for _ in range(len(env.total_map_cnt))]

    for i in range(len(env.total_map_cnt)):
        for j in range(len(env.total_map_cnt[0])):
            value = (abs_val - env.total_map_cnt[i][j])
            if (abs_val - env.total_map_cnt[i][j]) < 0: value = 0
            data[i][j] = value

    columns = ['X' + str(x + 1).zfill(1) for x in range(len(env.total_map_cnt[0]))]
    df4 = pd.DataFrame(data, columns=columns)
    df4.index = ['Y' + str(y + 1).zfill(1) for y in range(len(env.total_map_cnt))]



    plt.cla()
    plt.imshow(df4, cmap='gray', interpolation='none')
    plt.title("Robot Positions")
    plt.colorbar()
    plt.savefig(path + "/" + time_str + "_RobotPositions(Reverse_Abs).png")
    plt.close()

    # Result_summary
    f = open(path + "/" + time_str + "_result_summary.csv", 'w')
    human_distance_np = np.array(list(df_agent.iloc[:human_cnt, 6])).astype(int) * _DISTANCE_PER_GRID  # Distance
    robot_distance_np = np.array(list(df_agent.iloc[robot_cnt:, 6])).astype(int) * _DISTANCE_PER_GRID

    human_sku_cnt_np = np.array(list(df_agent.iloc[:human_cnt, 7])).astype(int)  # sku_cnt
    robot_sku_cnt_np = np.array(list(df_agent.iloc[robot_cnt:, 7])).astype(int)

    robot_timeout_cnt_np = np.array(list(df_agent.iloc[robot_cnt:, 5])).astype(int)

    human_avg_distance = np.mean(human_distance_np)
    human_var_distance = np.var(human_distance_np)
    human_std_distance = np.std(human_distance_np)
    robot_avg_distance = np.mean(robot_distance_np)
    robot_var_distance = np.var(robot_distance_np)
    robot_std_distance = np.std(robot_distance_np)

    human_avg_sku_cnt = np.mean(human_sku_cnt_np)
    human_var_sku_cnt = np.var(human_sku_cnt_np)
    human_std_sku_cnt = np.std(human_sku_cnt_np)
    robot_avg_sku_cnt = np.mean(robot_sku_cnt_np)
    robot_var_sku_cnt = np.var(robot_sku_cnt_np)
    robot_std_sku_cnt = np.std(robot_sku_cnt_np)

    human_sec_per_pick = (env.internal_timer * _TICKPERTIME) / human_avg_sku_cnt
    robot_sec_per_pick = (env.internal_timer * _TICKPERTIME) / robot_avg_sku_cnt
    box_per_hour_human = (((env.completed_batch * _LOADBOX_CNT) / (env.internal_timer * _TICKPERTIME)) * 3600) / human_cnt

    cur_hour = int(env.internal_timer * _TICKPERTIME) // 3600
    cur_min = (int(env.internal_timer * _TICKPERTIME) % 3600) // 60
    cur_sec = ((int(env.internal_timer* _TICKPERTIME) % 3600) % 60)
    cur_time_str = str("Time : {0:02d} : {1:02d} : {2:02d}\n".format(cur_hour, cur_min, cur_sec))
    cur_completed_order_str = str("Batch Done : {:04d}\n".format(env.completed_batch))
    cur_completed_box_str = str("Box Done : {:04d}\n".format(_LOADBOX_CNT * env.completed_batch))
    cur_all_off_completed_order_str = str('SKU Done : {:04d}\n'.format(env.all_of_completed_order))

    cur_human_box_per_hour_human_str = str('Box/Hour/Human : {:04f}\n'.format(box_per_hour_human))
    cur_human_sec_per_pick = str('Sec/Pick - Human : {:04f}\n'.format(human_sec_per_pick))
    cur_human_avg_distance_str = str('Human average moving distance : {:04f}\n'.format(human_avg_distance))
    cur_human_var_distance_str = str('Human variance moving distance : {:04f}\n'.format(human_var_distance))
    cur_human_std_distance_str = str('Human std moving distance : {:04f}\n'.format(human_std_distance))
    cur_human_var_sku_str = str('Human variance moving distance : {:04f}\n'.format(human_var_sku_cnt))
    cur_human_std_sku_str = str('Human std moving distance : {:04f}\n'.format(human_std_sku_cnt))

    cur_robot_sec_per_pick = str('Sec/Pick - Robot : {:04f}\n'.format(robot_sec_per_pick))
    cur_robot_time_out_set_str = str('Robot Time Out : {:04f}\n'.format(np.mean(robot_timeout_cnt_np)))
    cur_robot_avg_distance_str = str('Robot average moving distance : {:04f}\n'.format(robot_avg_distance))
    cur_robot_var_distance_str = str('Robot variance moving distance : {:04f}\n'.format(robot_var_distance))
    cur_robot_std_distance_str = str('Robot std moving distance : {:04f}\n'.format(robot_std_distance))
    cur_robot_var_sku_str = str('Robot variance moving distance : {:04f}\n'.format(robot_var_sku_cnt))
    cur_robot_std_sku_str = str('Robot std moving distance : {:04f}\n'.format(robot_std_sku_cnt))
    time = int(time)
    running_hour = time // 3600
    running_min = (time % 3600) // 60
    running_sec = ((time % 3600) % 60)
    running_time_str = str("Actual Running Time : {0:02d} : {1:02d} : {2:02d}\n".format(running_hour, running_min, running_sec))

    strategy_str = str("OrderBatch|HumanZone|OrderSeq|Route|HumanMove : {}|{}|{}|{}|{}\n".format(_ORDER_BATCH_FLAG,_HUMAN_ZONE_FLAG,_ORDER_SEQ_FLAG,_CHK_ROUTING_FLAG,_SELECT_HUMAN_MOVE))
    sim_info = str("BoxLoad|SKUExit|SKUperPicking|Human_cnt|Robot_cnt : {}|{}|{}|{}|{}\n".format(_BOX_LOADING_TIME, _SKU_PER_EXIT_TIME, _SKU_PER_PICKING_TIME, human_cnt, robot_cnt))
    sim_info_2 = str("StaticPath|PickingCollision : {}|{}".format(_SELECT_STATIC_PATH, _SELECT_PICKING_COLLISION))

    f.write("Actual Start Time : {}\n".format(str(stime)))
    f.write("Actual End Time : {}\n".format(str(etime)))
    f.write(running_time_str)
    f.write(cur_time_str)
    f.write(cur_completed_order_str)
    f.write(cur_completed_box_str)
    f.write(cur_all_off_completed_order_str)
    f.write(cur_human_box_per_hour_human_str)
    f.write(cur_human_sec_per_pick)
    f.write(cur_human_avg_distance_str)
    f.write(cur_human_var_distance_str)
    f.write(cur_human_std_distance_str)
    f.write(cur_human_var_sku_str)
    f.write(cur_human_std_sku_str)

    f.write(cur_robot_sec_per_pick)
    f.write(cur_robot_time_out_set_str)
    f.write(cur_robot_avg_distance_str)
    f.write(cur_robot_var_distance_str)
    f.write(cur_robot_std_distance_str)
    f.write(cur_robot_var_sku_str)
    f.write(cur_robot_std_sku_str)
    f.write(strategy_str)
    f.write(sim_info)
    f.write(sim_info_2)

    f.close()

#
# def WorkerBatch(worker_ration_in_zone, actual_worker_cnt):
#

if __name__ == "__main__":
    start_time_print = time.strftime('%Y.%m.%d - %H:%M:%S')
    start_time = time.time()
    # 맵 설정
    # map = """
    # ppppppppppppppppppp
    # p.................p
    # p.................p
    # p..xx..xx..xx.....p
    # p..xx..xx..xx.....p
    # p..xx..xx..xx.....p
    # p..xx..xx..xx...wpp
    # p...............gpp
    # p...............wpp
    # p..xx..xx..xx...wpp
    # p..xx..xx..xx.....p
    # p..xx..xx..xx.....p
    # p..xx..xx..xx.....p
    # p.................p
    # p.................p
    # ppppppppppppppppppp
    # """

    # Layout for Yongin Cross
    map = """
    pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
    ppppppppppppppppppppppppppppppppnnnnnnnnnnnnnnnnnnnnnppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxpp
    pppppppppppppppppppppppppppppp..oooooooooooooooooooom....ooooooooooooooooooooooooooopp
    pppppppppppppppppppppppppppppp..zeeeeeeeeeeeeeeeeeeee....zeeeeeeeeeeeeeeeeeeeeeeeeeenp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx....nnnnnnnnnnnnnnnnnnnnnnnnnnnpp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx...............................pp
    pppppppppppppppppppppppppppppp..oooooooooooooooooooom...............................pp
    pppppppppppppppppppppppppppppp..zeeeeeeeeeeeeeeeeeeee....ppppppppppppppppppppppppppppp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx....ppppppppppppppppppppppppppppp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx........ppp..........pppppppppppp
    pppppppppppppppppppppppppppppp..oooooooooooooooooooom........ppp..........pppppp....pp
    pppppppppppppppppppppppppppppp..zeeeeeeeeeeeeeeeeeeee.............pppppppp.ppppp....pp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx......ppppppppppppppp.ppppp....pp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx......pppppppppppppppppppp...pppp
    pppppppppppppppppppppppppppppp..oooooooooooooooooooom......bww............pppp....pppp
    pppppppppppppppppppppppppppppp..zeeeeeeeeeeeeeeeeeeee......p..............pppp......pp
    pppppppppppppppppppppppppppppp..xxxxxxxxxxxxxxxxxxxxx......bww............pppp......pp
    ppnnnnnnnnnnnnnnnnnnnnnnnnnnnn..xxxxxxxxxxxxxxxxxxxxx......ppppppp........pppp......pp
    pnooooooooooooom..ooooooooom....oooooooooooooooooooom........ppppp........pppp......pp
    ppeeeeeeeeeeeeee..zeeeeeeeee....zeeeeeeeeeeeeeeeeeeee........ppppppp......pppp......pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pxxxxxxxxxxxxxxxxxxxxx........ppppppp......pppp......pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pxxxxxxxxxxxxxxxxxxxxx..........pppppgpgpgpppppp.....pp
    pnooooooooooooom..ooooooooom....oooooooooooooooooooom..........ppp.pw.w.wpppppp.....pp
    ppeeeeeeeeeeeeee..zeeeeeeeee....zeeeeeeeeeeeeeeeeeeee..........ppp.pw.w.wpppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxx........ppp........pppp......pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxx........ppp........pppp......pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooom.............................pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.............rrrrrrrrrr......pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.....rrrrrrrrrr......pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom...hpppppppppppp.....pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom...hpppppppppppp..ppppp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee...hpppppppppppp..ppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom...hpppppppppppp.....pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom...hpppppppppppp.....pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom...hpppppppppppp.....pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...hpppppppppppp.....pp
    pnooooooooooooom..oooooooooooooooooooooooooooooooooooooooooooom.....................pp
    ppeeeeeeeeeeeeee..zeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx..................ppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxppppxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx..................ppppp
    pnooooooooooooom..ooooooooom....oooooooooooooooooooooooooooooom.....................pp
    ppeeeeeeeeeeeeee..zeeeeeeeee....zeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx..........p..........pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.....................pp
    pnooooooooooooom..ooooooooom....oooooooooooooooooooooooooooooom.....................pp
    ppeeeeeeeeeeeeee..zeeeeeeeee....zeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pxxxxxxxxxxxxxxxxxxxxx...............................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..ppppppppppppppppppppppnnnnnnnnnn.....................pp
    pnooooooooooooom..ooooooooom...pppppppppppppppppppppppppppppppp.....................pp
    ppeeeeeeeeeeeeee..zeeeeeeeee...pppppppppppppppppppppppppppppppp.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.....................pp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.....................pp
    pnooooooooooooom..ooooooooom...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppeeeeeeeeeeeeee..zeeeeeeeee...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    pnooooooooooooom..ooooooooom...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppeeeeeeeeeeeeee..zeeeeeeeee...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    pnooooooooooooom..ooooooooom...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppeeeeeeeeeeeeee..zeeeeeeeee...pppppppppppppppppppppppppppppppp.pppppppppppppppppppppp
    ppxxxxxxxxxxxxxx..xxxxxxxxxxp..ppppppppppppppppppppppppppppppppppppppppppppppppppppppp
    pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
    """

    # We need calculation
    human_cnt, robot_cnt = AgentCounter(map)
    agent_cnt = robot_cnt + human_cnt
    envInitSettingFlag = True

    env = Warehouse(5, 5, 3, agent_cnt, human_cnt, robot_cnt, 1, 1, 5, None, None, RewardType.GLOBAL,layout=map) # Warehouse 환경 객체 생성
    env.reset(initSettingFlag=envInitSettingFlag) # Warehouse 환경 초기화
    # env.not_used_shelf = [45,75,151]

    env.request_queue.clear()                     # shelf request queue 초기화
    env.shelfs_goalpos_remapping()                # shelf goalposition remapping for blocking
    routing_node_list, overlap = env.making_routing_node()             # Just Making for Zone
    env.making_routing_graph()
    # env.check_routing_graph()

    # 초기 상태 렌더링 및 웨이팅
    if _RENDERING_FLAG == 1: env.render()

    pre_actions = [np.array([Action.NOOP.value, 0], dtype='int64') for i in range(env.n_agents)]
    actions = [np.array([Action.NOOP.value, 0], dtype='int64') for i in range(env.n_agents)]

    # Order Batch Setting
    # input_order, random_order = order_generate(routing_node_list)
    # if _ORDER_BATCH_FLAG == 1: input_order = randomize_order(random_order)[:_SELECT_ORDER_CNT]
    # elif _ORDER_BATCH_FLAG == 0: input_order = input_order[:_SELECT_ORDER_CNT]
    # elif _ORDER_BATCH_FLAG == 2 : input_order = smallest_sku_node(input_order)[:_SELECT_ORDER_CNT]


    input_order = list()
    order_list = list()
    for _ in range(10):
        for node_list_idx in range(len(routing_node_list)):
            if len(routing_node_list[node_list_idx]) > 0:
                for node in routing_node_list[node_list_idx]:
                    order_list.append([node, 1, node_list_idx])
    random.shuffle(order_list)
    batch_list = [sorted(order_list[i:i + 12]) for i in range(0, len(order_list), 12)]
    # batch_list = [sorted(order_list[i:i + 1]) for i in range(0, len(order_list), 2)]
    input_order = batch_list

    big_asile   = human_batch(1,input_order)
    small_asile = human_batch(2,input_order)

    # for _ in range(9000):
    #     ran_pcs = random.randint(1,1497 + 1)
    #     order_list.append([ran_pcs, 1, 9999])
    # batch_list = [sorted(order_list[i:i+8]) for i in range(0,len(order_list), 8)]
    # input_order = batch_list

    # input_order_length = len(input_order)
    input_order_length = 750

    # Agent Init State Setting and Order Mapping
    for i in range(env.n_agents):
        env.agents[i].state = State.NOOP
    env.next_order_cnt = len(input_order)

    check_cnt = 0
    # 메인 루프
    while True:
        start = time.time()
        if _RENDERING_FLAG == 1: env.render()

        env.request_queue.clear()
        start_map = time.time()
        human_map = Make_Maze(env,mode=1)
        robot_map = Make_Maze(env,mode=2)


        start_act = time.time()
        # 로봇 액션 갱신
        for i in range(env.n_agents): actions[i] = env.agents[i].next_action(env, human_map, robot_map)




        start_step = time.time()
        # 로봇 액션 수행
        env.step(actions)


        start_check = time.time()
        # Human and Robot State Check
        for i in range(agent_cnt):env.agents[i].check_status(env,input_order)

        env.next_order_cnt = len(input_order)

        for i in range(agent_cnt): print(env.agents[i].id, env.agents[i].state, env.agents[i].node_list, actions[i], env.agents[i].load_box, env.agents[i].loadbox_station,env.agents[i].station)

        # Order Sequence
        if   _ORDER_SEQ_FLAG == 1: env.order_sequence()
        elif _ORDER_SEQ_FLAG == 2: pass

        # Routing Graph Check
        # if check_cnt > 300 and env.internal_timer > 600:
        #     env.check_routing_graph()
        #     check_cnt = 0
        # check_cnt = check_cnt + 1

        # Agent Result Check
        env.check_running_agent_cnt()

        # Exit Condition
        if env.completed_batch >= 750:
            end_time_print = time.strftime('%Y.%m.%d - %H:%M:%S')
            print(end_time_print)
            length_time = (time.time() - start_time)
            print('================end===================')
            WriteLog(env, robot_cnt, human_cnt, length_time, start_time_print, end_time_print)
            while True:
                tmp = input()
                if tmp == 'q':
                    break


