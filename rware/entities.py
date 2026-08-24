from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np
from rware.algorithm.path_planning.jps import jps, get_full_path, jps_converted_path
from rware.utils.Make_Maze import Make_Maze
from rware.core import Direction, Action, State, Entity
from rware.core.config import SimulationConfig


class Agent(Entity):
    counter = 0

    # 생성자
    def __init__(
            self,
            x: int,
            y: int,
            dir_: Direction,
            msg_bits: int,
            config: Optional[SimulationConfig] = None,
    ):
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
        self.agent_type = None
        self.agent_timer = 0
        self.coworker = None
        self.station = None
        self.loadbox_station = None

        self.routing_node = None
        self.cur_capacity = 0
        self.human_pick_cnt = 0
        self.node_list = list()
        self.path_planning = list()

        self.total_distance = 0
        self.complete_order = 0
        self.working_time = 0

        self.order_sku_cnt = list()
        self.cur_sku_cnt = 0
        self.total_sku_cnt = 0
        self.timeout_cnt = 0
        self.loading_timer = 0
        self.load_box = False
        self.waiting_time = 0
        # Per-robot picking task counter and the realised duration of the
        # current interaction, used to correct waiting_time once it completes.
        self.picking_seq = 0
        self.last_service_ticks = 0
        # Cell an idle worker should move toward. Set by the staging planner;
        # None leaves the worker where the last job ended.
        self.staging_target = None

        self.stop_flag = False

        #알고리즘
        self.get_full_path = None
        self.jps_converted_path = None
        self.jps = None
        self.make_maze = Make_Maze
        self.get_full_path = get_full_path
        self.jps_converted_path = jps_converted_path
        self.jps = jps

        self.config = config or SimulationConfig.from_legacy_config()
        self.max_capacity = self.config.robot_max_capacity

    def _at_waiting_position(self) -> bool:
        """True when an idle worker has reached where it should wait."""

        if self.staging_target is not None:
            return (self.x, self.y) == tuple(self.staging_target)
        return self.init_x == self.x and self.init_y == self.y

    # collision_layers에 대한 메소드
    @property
    def collision_layers(self):
        agent_layer = self.config.layer_agents
        if self.loaded:
            return (agent_layer, self.config.layer_shelfs)
        return (agent_layer,)

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

        if self.req_action.value < len(wraplist):
            return wraplist[self.req_action.value]
        else:
            return self.dir
        # if self.req_action == Action.RIGHT:
        #     return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        # elif self.req_action == Action.LEFT:
        #     return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        # else:
        #     return self.dir

    # Direction을 Num으로 전환하는 함수
    def Direction2Num(self):
        if self.dir == Direction.UP:
            return 0
        elif self.dir == Direction.UPRIGHT:
            return 1
        elif self.dir == Direction.RIGHT:
            return 2
        elif self.dir == Direction.DOWNRIGHT:
            return 3
        elif self.dir == Direction.DOWN:
            return 4
        elif self.dir == Direction.DOWNLEFT:
            return 5
        elif self.dir == Direction.LEFT:
            return 6
        elif self.dir == Direction.UPLEFT:
            return 7
        else:
            return None

    # 다음 액션 생성 -> make new method
    def next_action(self, env, human_map, robot_map):
        # Do not go inside Rack
        # For Human
        cfg = getattr(env, "config", self.config)
        layer_spots = self.config.layer_spots

        if self.agent_type == True:
            cur_map = human_map  # Make Maze for Path Planning
        else:
            cur_map = robot_map  # Make Maze for Path Planning

        dx = [0, 1, 1, 1, 0, -1, -1, -1]
        dy = [-1, -1, 0, 1, 1, 1, 0, -1]

        # Human type
        if self.agent_type == True:
            # An idle worker sits in NOOP wherever its last job left it: the
            # robot clears the pairing by setting the worker straight to NOOP,
            # so the HOME return never runs. Staging gives that idle time a
            # destination, which is the only way the worker moves before its
            # next dispatch.
            staging_move = self.state == State.NOOP and self.staging_target is not None
            if (
                self.state == State.HUMAN_MOVESPOT
                or self.state == State.HOME
                or staging_move
            ):
                target_y, target_x = self.y, self.x

                if self.state == State.HUMAN_MOVESPOT:
                    target_y = env.shelfs[self.node_list[0] - 1].goal_y
                    target_x = env.shelfs[self.node_list[0] - 1].goal_x

                elif self.staging_target is not None:
                    # Wait near anticipated demand rather than at home.
                    target_x, target_y = self.staging_target

                else:
                    target_y = self.init_y
                    target_x = self.init_x

                cur_dir = self.dir.value
                # if (self.x, self.y, target_x, target_y, cur_dir) in env.action_map:
                #     value = env.action_map[(self.x, self.y, target_x, target_y, cur_dir)]
                #     if cur_map[self.y + dy[value]][self.x + dx[value]] == 0:
                #         print(self.id, " USE ActionMap")
                #         return np.array([env.action_map[(self.x, self.y, target_x, target_y, cur_dir)], 0], dtype='int64')

                target_value = cur_map[target_y, target_x]
                cur_map[target_y, target_x] = 0
                cur_map[self.y, self.x] = 0

                new_path = self.jps(cur_map, self.y, self.x, target_y, target_x)
                if new_path is not None and len(new_path) > 2:
                    new_path = new_path[:2]
                full_path = self.get_full_path(new_path)
                self.path_planning = self.jps_converted_path(cur_map, full_path, cur_dir)
                # self.path_planning = jps_converted_path(cur_map, get_full_path(new_path), cur_dir, True)
                # env.path_list.append(self.path_planning)
                # if self.path_planning is None or len(self.path_planning) <= 1:
                #     self.path_planning = aStar(cur_map, (self.y, self.x), (target_y, target_x), cur_dir, env.internal_timer)

                cur_map[self.y, self.x] = 1
                cur_map[target_y, target_x] = target_value

                if self.path_planning is None or len(self.path_planning) <= 1:
                    return np.array([Action.NOOP.value, 0], dtype='int64')
                else:
                    return np.array([self.path_planning[1][2], 0], dtype='int64')

            else:
                return np.array([Action.NOOP.value, 0], dtype='int64')

        # Robot Type
        else:
            if self.state == State.ROBOT_MOVESPOT or self.state == State.ROBOT_MOVEGOAL or self.state == State.HOME or self.state == State.ROBOT_MOVEZONE or self.state == State.ROBOT_MOVEQUEUE:
                target_y, target_x = self.y, self.x
                routing_path = []
                # load_station_block = [cur_map[bl][58] for bl in range(12,19)]
                # if (self.x, self.y) in env.loadbox_queue:
                #     for bl in range(12, 19): cur_map[bl][58] = 0
                # else:
                #     for bl in range(12, 19): cur_map[bl][58] = 1
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

                    if self.agent_timer > cfg.timeout_value_start:
                        if getattr(cfg, "debug_prints", False):
                            print("Timeout!", self.id)
                        if self.agent_timer > cfg.timeout_value_end:
                            self.agent_timer = 0
                            self.timeout_cnt += 1
                            # cur_idx = env.routing_node_dict[(self.x, self.y)]
                            # env.total_timeout_cnt_in_zone[cur_idx] += 1
                        else:
                            action_list = [Action.LEFT.value, Action.LEFT.value, Action.LEFT.value, Action.RIGHT.value,
                                           Action.RIGHT.value, Action.RIGHT.value]
                            return np.array([action_list[env.internal_timer % len(action_list)], 0], dtype='int64')

                            # if env.grid[layer_spots, self.y, self.x] in [6]: return np.array([Action.RIGHT.value, 0], dtype='int64')
                            # elif env.grid[layer_spots, self.y, self.x] in [7]: return np.array([Action.LEFT.value, 0], dtype='int64')
                            # elif env.grid[layer_spots, self.y, self.x] in [8]: return np.array([Action.DOWNRIGHT.value, 0], dtype='int64')
                            # elif env.grid[layer_spots, self.y, self.x] in [9]: return np.array([Action.UPLEFT.value, 0], dtype='int64')
                            #
                            # else:
                            #     if env.internal_timer % 2 == 0: return np.array([Action.UP.value, 0], dtype='int64')
                            #     else: return np.array([Action.DOWN.value, 0], dtype='int64')

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



                elif self.state == State.ROBOT_MOVEGOAL or self.state == State.ROBOT_MOVEQUEUE:
                    if self.load_box == True:
                        target_y = self.station[1]
                        target_x = self.station[0]
                    else:
                        target_y = self.loadbox_station[1]
                        target_x = self.loadbox_station[0]

                elif self.state == State.HOME:
                    target_y = self.init_y
                    target_x = self.init_x

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
                allowed = None
                if cfg.wall_enforce_level == 2:
                    allowed = getattr(env, "allowed_dirs", None)
                new_path = self.jps(cur_map, self.y, self.x, target_y, target_x, allowed=allowed)

                motion_flag = True

                # if env.grid[layer_shelfs,self.y+1, self.x] > 0 or env.grid[layer_shelfs,self.y-1, self.x] > 0 or env.grid[layer_spots,self.y+1, self.x] == 5 or env.grid[layer_spots,self.y-1, self.x] == 5: motion_flag = False
                # if env.grid[layer_shelfs, self.y + 1, self.x] > 0 or env.grid[layer_shelfs, self.y - 1, self.x] > 0 : motion_flag = False

                if new_path is not None and len(new_path) > 2: new_path = new_path[:2]
                full_path = self.get_full_path(new_path)
                self.path_planning = self.jps_converted_path(cur_map, full_path, cur_dir)

                # if motion_flag == True:
                #     self.path_planning = jps_converted_path(cur_map, full_path, cur_dir, True)
                # else :
                #     self.path_planning = jps_converted_path2(cur_map, full_path, cur_dir, False)

                # env.path_list.append(self.path_planning)

                if self.state == State.ROBOT_MOVEGOAL or self.state == State.ROBOT_MOVEQUEUE:
                    if not self.load_box:
                        target_y = self.loadbox_station[1]
                        target_x = self.loadbox_station[0]

                if self.path_planning is None or len(self.path_planning) <= 1:
                    cur_map[self.y, self.x] = 1
                    cur_map[target_y, target_x] = target_value
                    return np.array([Action.NOOP.value, 0], dtype='int64')
                # for bl in range(len(load_station_block)): cur_map[bl+12][58] = load_station_block[bl]
                # cur_map[self.y, self.x] = 1
                # cur_map[target_y, target_x] = target_value
                #
                # if self.state == State.HOME: cur_map[21][66:73] = 0
                # if self.state == State.ROBOT_MOVEZONE and len(self.node_list) <= 0:
                #     cur_map[self.station[1] + 2, self.station[0] + 1] = 0
                #     cur_map[self.station[1] + 1, self.station[0] + 1] = 0
                #     cur_map[self.station[1] + 0, self.station[0] + 1] = 0

                if self.path_planning is None or len(self.path_planning) <= 1:
                    return np.array([Action.NOOP.value, 0], dtype='int64')
                else:
                    return np.array([self.path_planning[1][2], 0], dtype='int64')

            # elif self.state == State.ROBOT_MOVEQUEUE:
            #     if self.load_box == True:
            #         if self.x > self.station[0]:
            #             return np.array([Action.LEFT.value, 0], dtype='int64')
            #         elif self.x < self.station[0]:
            #             return np.array([Action.RIGHT.value, 0], dtype='int64')
            #         else:
            #             return np.array([Action.UP.value, 0], dtype='int64')
            #     else:
            #         if  self.y > self.loadbox_station[1]:
            #             return np.array([Action.UP.value, 0], dtype='int64')
            #         elif self.y < self.loadbox_station[1]:
            #             return np.array([Action.DOWN.value, 0], dtype='int64')
            #         else:
            #             return np.array([Action.LEFT.value, 0], dtype='int64')

            else:
                return np.array([Action.NOOP.value, 0], dtype='int64')

    # 로봇 상황 체크 메소드
    def check_status(self, env, input_order):
        cfg = getattr(env, "config", self.config)

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
                if env.shelfs[self.node_list[0] - 1].goal_x == self.x and env.shelfs[
                    self.node_list[0] - 1].goal_y == self.y:
                    self.state = State.ROBOT_PICKING
                    self.waiting_time += 1
                    # Identifies this picking task; keeps the service-time draw
                    # stable across re-auctions of the same task.
                    self.picking_seq += 1

            # 피킹 상태에서 작업 수행이 끝나면 배출지/다음 노드 이동
            elif len(self.node_list) > 0 and self.state == State.ROBOT_PICKING:
                self.waiting_time += 1
                if self.coworker is not None and env.agents[
                    self.coworker - 1].state == State.HUMAN_DONE:  # 사람이 작업을 끝낸 경우,
                    if len(self.node_list) > 1:  # 최소 2개 이상 node 존재
                        # waiting_time accrued every tick of the interaction; subtract the
                        # duration that actually elapsed, which the service-time
                        # model may have drawn rather than computed.
                        service_ticks = self.last_service_ticks or (
                            self.order_sku_cnt[0] * cfg.sku_per_picking_time
                        )
                        self.waiting_time -= (service_ticks - 1)
                        self.last_service_ticks = 0
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
                        # waiting_time accrued every tick of the interaction; subtract the
                        # duration that actually elapsed, which the service-time
                        # model may have drawn rather than computed.
                        service_ticks = self.last_service_ticks or (
                            self.order_sku_cnt[0] * cfg.sku_per_picking_time
                        )
                        self.waiting_time -= (service_ticks - 1)
                        self.last_service_ticks = 0
                        env.agents[self.coworker - 1].node_list = list()
                        env.agents[self.coworker - 1].state = State.NOOP
                        env.agents[self.coworker - 1].coworker = None

                        self.cur_sku_cnt = self.cur_sku_cnt + self.order_sku_cnt[0]
                        self.order_sku_cnt = list()
                        self.node_list = list()  # 현재 작업한 노드 제거
                        self.coworker = None
                        env.current_loaded_node_list[self.id - 1] = list()
                        self.state = State.ROBOT_MOVEGOAL
                        # self.state = State.ROBOT_MOVEZONE
                        if self.station is None:
                            self.select_goals(env)
                            print("select_goal : ", self.id, self.station)

                        # 배출 대기구로 이동
                else:
                    pass  # Just Wait for human

            elif self.state == State.ROBOT_MOVEGOAL:
                if self.load_box == True and [self.x, self.y] == [self.station[0], self.station[1]]:
                    self.state = State.ROBOT_MOVEQUEUE
                    station_idx = env.goals.index(self.station)
                    env.wait_queue_cnt[station_idx] += 1

                elif self.load_box == False and [self.x, self.y] == [self.loadbox_station[0], self.loadbox_station[1]]:
                    self.state = State.ROBOT_MOVEQUEUE
                    loadbox_station_idx = env.loadbox_queue.index(self.loadbox_station)
                    env.waitLOADBOX_CNT[loadbox_station_idx] += 1
                    # print("waitbox_cnt : ",env.waitLOADBOX_CNT)

            # AMR이 배출지에 도착 -> 배출 DROP으로 변함
            elif self.state == State.ROBOT_MOVEQUEUE:
                if self.load_box == True and [self.x, self.y] == [self.station[0], self.station[1]]:
                    self.state = State.ROBOT_DROP
                    self.loading_timer = cfg.sku_per_exit_time - 1

                if self.load_box == False and [self.x, self.y] == [self.loadbox_station[0], self.loadbox_station[1]]:
                    self.state = State.ROBOT_LOAD
                    self.loading_timer = cfg.box_loading_time - 1

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
                    env.completed_batch_log.append([env.internal_timer, env.completed_batch])
                    station_idx = env.goals.index(self.station)
                    env.wait_queue_cnt[station_idx] -= 1
                    self.station = None
                    if self.loadbox_station is None: self.select_loadbox_station(env)

            elif self.state == State.ROBOT_LOAD and [self.loadbox_station[0], self.loadbox_station[1]] == [self.x,
                                                                                                           self.y]:

                if self.loading_timer > 0:
                    self.loading_timer = self.loading_timer - 1

                else:
                    self.state = State.HOME
                    env.next_order_cnt = len(input_order)
                    # Load Node List
                    if len(input_order) > 0:
                        self.load_box = True

                        self.node_list = [j[0] for j in input_order[0]]
                        env.current_loaded_node_list[self.id - 1] = input_order[0].copy()
                        self.order_sku_cnt = [j[1] for j in input_order[0]]
                        # ################# TEST #################################
                        # first_task = input_order[0][0]
                        # self.node_list = [first_task[0]]
                        # env.current_loaded_node_list[self.id - 1] = [first_task]
                        # self.order_sku_cnt = [first_task[1]]
                        # ################# TEST #################################
                        del input_order[0]
                        env.next_order_cnt = len(input_order)

                        self.state = State.ROBOT_MOVESPOT
                        # self.state = State.ROBOT_MOVEZONE



                    load_station_idx = env.loadbox_queue.index(self.loadbox_station)
                    env.waitLOADBOX_CNT[load_station_idx] -= 1
                    self.loadbox_station = None

            elif self.state == State.HOME and [self.init_x, self.init_y] == [self.x, self.y]:
                self.state = State.NOOP

        # Human
        else:
            # AMR의 작업 요청이 할당 -> 작업자 이동
            if (self.state == State.NOOP or self.state == State.HOME) and self.coworker == None:
                self.agent_timer += 1
                if self.agent_timer > 3:
                    self.coworker = env.select_coworker(env.agents[self.id - 1])  # 작업자 대상자 선정 -> AMR에 할당

                # 협업할 작업자가 존재 시,
                if self.coworker is not None:
                    self.staging_target = None
                    env.agents[self.coworker - 1].coworker = self.id
                    self.node_list.append(env.agents[self.coworker - 1].node_list[0])
                    self.agent_timer = 0
                    if not cfg.human_move_strategy:
                        self.state = State.HUMAN_MOVESPOT
                    else:
                        self.state = State.HUMAN_PICKING
                        robot = env.agents[self.coworker - 1]
                        duration = env.service_ticks_for(self, robot, robot.order_sku_cnt[0])
                        robot.last_service_ticks = duration
                        self.loading_timer = duration - 1

            # 작업자가 목표한 노드에 도달 -> AMR이 피킹상태면 작업 시작 아니면 대기
            elif self.state == State.HUMAN_MOVESPOT and env.shelfs[self.node_list[0] - 1].goal_x == self.x and \
                    env.shelfs[self.node_list[0] - 1].goal_y == self.y:
                if env.agents[self.coworker - 1].state == State.ROBOT_PICKING:
                    robot = env.agents[self.coworker - 1]
                    duration = env.service_ticks_for(self, robot, robot.order_sku_cnt[0])
                    robot.last_service_ticks = duration
                    self.loading_timer = duration - 1
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

            elif self.state == State.HOME and self._at_waiting_position():
                self.state = State.NOOP

    # 노드 변경 메소드
    def node_change(self):
        if len(self.node_list) > 0:
            tmp = self.node_list[0]
            self.node_list.append(tmp)
            del self.node_list[0]

    def select_goals(self, env):
        cur_map = self.make_maze(env, 2)
        random.shuffle(env.goals)
        if self.station is None and len(
                self.node_list) <= 0 and self.state == State.ROBOT_MOVEGOAL and self.load_box == True:
            select_list = list()
            for sample_station in env.goals:
                sample_idx = env.goals.index(sample_station)
                # delta_y = abs(sample_station[1] - self.y)
                # delta_x = abs(sample_station[0] - self.x)
                select_list.append([env.wait_queue_cnt[sample_idx], 1, sample_station])
            select_list = sorted(select_list)
            self.station = select_list[0][2]
        return

    def select_loadbox_station(self, env):
        cur_map = self.make_maze(env, 2)
        if self.loadbox_station is None and len(self.node_list) <= 0 and self.state == State.ROBOT_MOVESPOT:
            select_list = list()
            cfg = getattr(env, "config", self.config)
            if getattr(cfg, "debug_prints", False):
                print("loadbox_queue: ", env.loadbox_queue)
            for sample_station in env.loadbox_queue:
                sample_idx = env.loadbox_queue.index(sample_station)
                delta_y = abs(sample_station[1] - self.y)
                delta_x = abs(sample_station[0] - self.x)
                select_list.append([env.waitLOADBOX_CNT[sample_idx], delta_y + delta_x, sample_station, sample_idx])
            select_list = sorted(select_list)
            if getattr(cfg, "debug_prints", False):
                print(self.id, select_list)
            self.loadbox_station = select_list[0][2]
            env.waitLOADBOX_CNT[select_list[0][3]] += 1

        return


# 랙 클래스 정의
class Shelf(Entity):
    counter = 0

    def __init__(self, x, y, config: Optional[SimulationConfig] = None):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.init_x = self.x
        self.init_y = self.y

        # Added Variable
        self.goal_x = self.x
        self.goal_y = self.y
        self.config = config or SimulationConfig.from_legacy_config()

    # collision_layers에 대한 get 메소드
    @property
    def collision_layers(self):
        return (self.config.layer_shelfs,)


# 웨어하우스 클래스 정의
