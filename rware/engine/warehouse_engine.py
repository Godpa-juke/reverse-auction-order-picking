import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium import spaces

from rware.algorithm.batch_sequence.batch_sequencing import *
from rware.algorithm.human_batch.human_batch import *
from rware.core import Action, Direction, SimulationConfig, State
from rware.core.config import HumanZoneStrategy
from rware.core.map_dsl import move_bit
from rware.engine.definitions import ImageLayer, ObservationType, RewardType
from rware.engine.event_loop import Command, CommandQueue, Event, EventQueue
from rware.engine.human_assignment import (
    AuctionAssignmentStrategy,
    get_human_assignment_strategy,
    HumanSnapshot,
    MatchingContext,
    RobotSnapshot,
)
from rware.engine.observations import build_observation_strategy
from rware.engine.services import EngineServices, create_engine_services
from rware.engine.state import WorldState
from rware.entities import Agent, Shelf
from rware.source.site_a.wrap import *
from rware.utils.Make_Maze import Make_Maze

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

# Agent(로봇) 클래스 정의

class WarehouseEngine:
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
            observation_type: ObservationType = ObservationType.FLATTENED,
            image_observation_layers: List[ImageLayer] = [
                ImageLayer.SHELVES,
                ImageLayer.REQUESTS,
                ImageLayer.AGENTS,
                ImageLayer.GOALS,
                ImageLayer.ACCESSIBLE
            ],
            image_observation_directional: bool = True,
            normalised_coordinates: bool = False,
    ):
        # SimulationConfig 생성
        self.config = SimulationConfig.from_legacy_config()
        cfg = self.config
        cfg.request_queue_size = request_queue_size

        # 레이어/설정 단축 참조
        self.layer_agents = cfg.layer_agents
        self.layer_shelfs = cfg.layer_shelfs
        self.layer_spots = cfg.layer_spots
        self.layer_human = cfg.layer_human

        # 서브 시스템 초기화
        services: EngineServices = create_engine_services(cfg)
        self.services = services
        self.environment = services.environment
        self.agent_manager = services.agent_manager
        self.task_scheduler = services.task_scheduler
        self.data_collector = services.data_collector

        # 서브 시스템 간 연결 설정
        self.agent_manager.environment = self.environment

        # 명령/이벤트 루프 구성
        self.command_queue = CommandQueue()
        self.event_queue = EventQueue()
        self._command_handlers = {
            "advance_tick": self._handle_command_advance_tick,
            "apply_actions": self._handle_command_apply_actions,
            "reset": self._handle_command_reset,
        }
        self._pending_actions: Optional[List[Action]] = None

        # 기존 파라미터 저장 (하위 호환성 유지)
        self.shelf_columns = shelf_columns
        self.column_height = column_height
        self.shelf_rows = shelf_rows
        self.n_agents = n_agents
        self.n_humans = n_humans
        self.n_robots = n_robots
        schema_message_bits = cfg.action_schema.get("message_bits") if cfg.action_schema else None
        if schema_message_bits is not None:
            msg_bits = int(schema_message_bits)
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps = max_inactivity_steps
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.layout = layout
        self.observation_type = observation_type
        self.image_observation_layers = image_observation_layers
        self.image_observation_directional = image_observation_directional
        self.normalised_coordinates = normalised_coordinates

        # 환경 초기화 (먼저 해야 grid_size 등이 설정됨)
        self.environment.initialize_grid(layout, shelf_columns, shelf_rows, column_height)

        # EnvironmentCore에서 속성들을 가져와서 Warehouse에도 설정 (하위 호환성)
        self.grid_size = self.environment.grid_size
        self.grid = self.environment.grid
        self.highways = self.environment.highways
        self.walls = self.environment.walls
        self.allowed_dirs = self.environment.allowed_dirs
        self.goals = self.environment.goals
        self.shelfs = self.environment.shelfs
        self.shelf_queue = self.environment.shelf_queue
        self.human_init_queue = self.environment.human_init_queue
        self.robot_init_queue = self.environment.robot_init_queue
        self.picking_queue = self.environment.picking_queue
        self.loadbox_queue = self.environment.loadbox_queue
        self.wait_queue = self.environment.wait_queue
        self.wait_queue_cnt = self.environment.wait_queue_cnt
        self.waitLOADBOX_CNT = self.environment.waitLOADBOX_CNT

        # 작업 존 데이터 기본값
        self.big_asile: List[List[int]] = []
        self.small_asile: List[List[int]] = []

        # Routing 관련 속성들 초기화 (EnvironmentCore에 없으므로 직접 초기화)
        self.routing_node = []
        self.routing_node_all_pos = []
        self.routing_node_dict = dict()

        # 전문 협업 전략 설정
        self.set_human_assignment_strategy(self.config.human_assignment_strategy)
        self._assignment_plan_tick: int = -1
        self._pending_human_assignments: Dict[int, int] = {}
        self.assignment_total_count = 0
        self.assignment_en_route_count = 0
        # Predictive rendezvous dispatch. Strategies may turn this on for
        # themselves; the config value is the run-wide default.
        self._predictive_dispatch = bool(getattr(self.config, "predictive_dispatch", False))
        self.arrival_tracker = None
        self.staging_planner = None
        self.zone_list_in_rack = []
        self.rack_list = []
        self.node_identifier = []
        self.edge_map = []
        self.routing_graph = None

        # 관측 공간 설정 (grid_size가 필요하므로 여기서 호출)
        self._setup_observation_space()

        # 액션 공간 설정
        self._setup_action_space()

        # 에이전트 매니저 초기화
        self.agent_manager.initialize_agents(
            self.environment.human_init_queue,
            self.environment.robot_init_queue,
            msg_bits,
            self.environment.grid_size
        )
        self.agents = self.agent_manager.agents

        # 태스크 스케줄러 초기화
        self.task_scheduler.initialize_task_system(self.n_agents)
        self.request_queue = self.task_scheduler.request_queue

        # 렌더러 초기화
        self.renderer = None

        # Gym 환경 표준 속성
        self.reward_range = (0, 1)
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.next_order_cnt = 0
        self._episode_started = False
        self._last_home_blockers: List[Tuple[int, str, Any]] = []

        # 렌더링 관련 속성들 초기화
        self.not_used_shelf = []
        self.path_list = []

        # AgentManager의 모든 속성들 연결
        self.running_human_cnt = self.agent_manager.running_human_cnt
        self.running_robot_cnt = self.agent_manager.running_robot_cnt
        self.total_robot_cnt_in_zone = self.agent_manager.total_robot_cnt_in_zone
        self.total_timeout_cnt_in_zone = self.agent_manager.total_timeout_cnt_in_zone
        self.completed_batch_log = self.agent_manager.completed_batch_log
        self.agent_id_list = self.agent_manager.agent_id_list
        self.n_max_humans = self.agent_manager.n_max_humans
        self.n_max_robots = self.agent_manager.n_max_robots
        self.n_max_agents = self.agent_manager.n_max_agents
        self.human_id_list = self.agent_manager.human_id_list
        self.robot_id_list = self.agent_manager.robot_id_list

        # EnvironmentCore의 타이머 및 통계 속성들 연결
        self.internal_timer = self.environment.internal_timer
        self.completed_batch = self.environment.completed_batch
        self.all_of_completed_order = self.environment.all_of_completed_order
        self.using_station = self.environment.using_station
        self.using_agent = self.environment.using_agent

        self._refresh_world_state()

    def _setup_observation_space(self) -> None:
        """관측 공간 설정"""
        self.image_obs = False
        self.fast_obs = False
        self.observation_strategy = build_observation_strategy(self.observation_type)
        self.observation_strategy.configure(self)

    def _setup_action_space(self) -> None:
        """액션 공간 설정"""
        schema = self.config.action_schema or {}
        schema_type = schema.get("type", "multi_discrete").lower()

        if schema_type == "discrete":
            action_space = spaces.Discrete(int(schema.get("n", len(Action))))
        else:
            message_bits = int(schema.get("message_bits", self.msg_bits))
            dimensions = [len(Action)]
            if message_bits > 0:
                dimensions.extend([2] * message_bits)
            if len(dimensions) == 1:
                action_space = spaces.Discrete(dimensions[0])
            else:
                action_space = spaces.MultiDiscrete(dimensions)

        self.action_space = spaces.Tuple(tuple(self.n_agents * [action_space]))

    # --- command/event helpers -------------------------------------------------

    def _build_sensor_space_schema(self) -> OrderedDict:
        schema = OrderedDict(
            {
                "has_agent": spaces.MultiBinary(1),
                "direction": spaces.Discrete(8),
            }
        )
        if self.msg_bits > 0:
            schema["local_message"] = spaces.MultiBinary(self.msg_bits)
        schema["has_shelf"] = spaces.MultiBinary(1)
        schema["shelf_requested"] = spaces.MultiBinary(1)
        return schema

    def queue_command(self, command: Command) -> None:
        self.command_queue.push(command)

    def submit_command(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.queue_command(Command(name=name, payload=payload, tick=self.internal_timer))

    def emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None, tick: Optional[int] = None) -> None:
        event_tick = tick if tick is not None else self.internal_timer
        self.event_queue.emit(Event(name=name, payload=payload, tick=event_tick))

    def poll_events(self) -> List[Event]:
        return self.event_queue.drain()

    def _process_commands(self) -> None:
        for command in self.command_queue.drain():
            handler = self._command_handlers.get(command.name)
            if handler is None:
                continue
            handler(command)

    def _handle_command_advance_tick(self, command: Command) -> None:
        self.internal_timer += 1
        self.emit_event("tick", {"tick": self.internal_timer}, tick=self.internal_timer)

    def _handle_command_apply_actions(self, command: Command) -> None:
        self._pending_actions = command.payload.get("actions") if command.payload else None

    def _handle_command_reset(self, command: Command) -> None:
        self._pending_actions = None
        self.internal_timer = 0
        self._cur_steps = 0
        self._cur_inactive_steps = 0

    def _refresh_world_state(self) -> None:
        self.world_state = WorldState.from_engine(self)

    def _make_observation(self, agent):
        return self.observation_strategy.observe(self, agent)

    def _orders_remaining(self) -> bool:
        """Return True when orders or robot tasks are still in flight."""
        external_orders = getattr(self, "next_order_cnt", 0) or 0
        scheduler_orders = getattr(self.task_scheduler, "next_order_cnt", 0) or 0
        if (external_orders > 0) or (scheduler_orders > 0):
            return True

        # Fall back to robot state to guard against counters getting out of sync.
        robot_ids = getattr(self.agent_manager, "robot_id_list", [])
        for robot_id in robot_ids:
            if robot_id <= 0 or robot_id > len(self.agents):
                continue

            robot = self.agents[robot_id - 1]
            if getattr(robot, "agent_type", False) is True:
                continue

            if robot.state not in (State.NOOP, State.HOME):
                return True
            if getattr(robot, "load_box", False):
                return True
            if getattr(robot, "carrying_shelf", None) is not None:
                return True
            if getattr(robot, "node_list", None):
                if len(robot.node_list) > 0:
                    return True
            if getattr(robot, "order_sku_cnt", None):
                if len(robot.order_sku_cnt) > 0:
                    return True

        return False

    def _robots_at_home(self) -> bool:
        """Return True when every robot is back at its spawn point and idle."""
        robot_ids = getattr(self.agent_manager, "robot_id_list", [])
        home_blockers = []
        for robot_id in robot_ids:
            if robot_id <= 0 or robot_id > len(self.agents):
                continue

            robot = self.agents[robot_id - 1]
            if getattr(robot, "agent_type", False) is True:
                continue

            if robot.state not in (State.NOOP, State.HOME):
                home_blockers.append((robot.id, "state", getattr(robot.state, "name", robot.state)))
                continue
            if (robot.x, robot.y) != (robot.init_x, robot.init_y):
                home_blockers.append((robot.id, "position", (robot.x, robot.y)))
                continue
            if getattr(robot, "carrying_shelf", None) is not None:
                home_blockers.append((robot.id, "carrying_shelf", True))
                continue
            if getattr(robot, "load_box", False):
                home_blockers.append((robot.id, "load_box", True))
                continue

        self._last_home_blockers = home_blockers
        return len(home_blockers) == 0

    def _should_end_episode(self) -> Tuple[bool, Optional[str]]:
        """Determine whether the current episode should terminate."""
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ):
            return True, "max_inactivity_steps"

        if self.max_steps and self._cur_steps >= self.max_steps:
            return True, "max_steps"

        robots_home = self._robots_at_home()
        orders_pending = self._orders_remaining()

        if robots_home and not orders_pending:
            return True, "robots_at_home"

        return False, None

    # 기존 메소드들 (아직 정리되지 않음)
    # 파라미터를 통한 레이아웃 설계 메소드
    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((self.config.collision_layers, *self.grid_size), dtype=np.int32)
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

        vertical_idx = self.config.shelf_vertical_idx
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1  # row
        lines = layout.split("\n")
        grid_width = len(lines[0])  # col

        vector = ['' for _ in range(grid_width)]

        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        if vertical_idx == False:
            for col in range(grid_width):
                for line in lines:
                    vector[col] = vector[col] + str(line[col])

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((self.config.collision_layers, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        if vertical_idx == True:
            for y, line in enumerate(lines):
                for x, char in enumerate(line):
                    assert char.lower() in "gpwboemzxrnh."
                    if char.lower() == "g":
                        self.grid[self.layer_spots, y, x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif char.lower() == "w":
                        self.grid[self.layer_spots, y, x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "p":
                        self.grid[self.layer_spots, y, x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "b":
                        self.grid[self.layer_spots, y, x] = 4
                        self.loadbox_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "n":
                        self.grid[self.layer_spots, y, x] = 5
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1

                    elif char.lower() == "o":
                        self.grid[self.layer_spots, y, x] = 6
                        self.highways[y, x] = 1
                    elif char.lower() == "e":
                        self.grid[self.layer_spots, y, x] = 7
                        self.highways[y, x] = 1
                    elif char.lower() == "m":
                        self.grid[self.layer_spots, y, x] = 8
                        self.highways[y, x] = 1
                    elif char.lower() == "z":
                        self.grid[self.layer_spots, y, x] = 9
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
                        self.grid[self.layer_spots, y, x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif char.lower() == "w":
                        self.grid[self.layer_spots, y, x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif char.lower() == "p":
                        self.grid[self.layer_spots, y, x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1


                    elif char.lower() == "b":

                        self.grid[self.layer_spots, y, x] = 4

                        self.loadbox_queue.append((x, y))

                        self.highways[y, x] = 1

                    elif char.lower() == "n":

                        self.grid[self.layer_spots, y, x] = 5

                        self.picking_queue.append((x, y))

                        self.highways[y, x] = 1


                    elif char.lower() == "o":
                        self.grid[self.layer_spots, y, x] = 6
                        self.highways[y, x] = 1

                    elif char.lower() == "e":
                        self.grid[self.layer_spots, y, x] = 7
                        self.highways[y, x] = 1

                    elif char.lower() == "m":
                        self.grid[self.layer_spots, y, x] = 8
                        self.highways[y, x] = 1

                    elif char.lower() == "z":
                        self.grid[self.layer_spots, y, x] = 9
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
        self.waitLOADBOX_CNT = [0 for _ in range(len(self.loadbox_queue))]
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
                                            self._build_sensor_space_schema()
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
                        layer = self.grid[self.layer_shelfs].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0

                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0

                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[self.layer_agents].copy().astype(np.float32)
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
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1, 2))
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
                self.grid[self.layer_agents], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[self.layer_shelfs], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[self.layer_agents]
            padded_shelfs = self.grid[self.layer_shelfs]

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
                if self.msg_bits > 0:
                    obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                if self.msg_bits > 0:
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
        self.grid[self.layer_shelfs] = 0
        self.grid[self.layer_agents] = 0

        for s in self.shelfs:
            self.grid[self.layer_shelfs, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[self.layer_agents, a.y, a.x] = a.id

    # 웨어하우스 리셋 메소드
    def reset(self, initSettingFlag=True):
        self.command_queue.clear()
        self.event_queue.drain()

        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.internal_timer = 0
        self.next_order_cnt = 0
        self._episode_started = False
        self._last_home_blockers = []
        self.total_map_cnt = [[0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        self.shelfs = [
            Shelf(x, y, self.config)
            for x, y in self.shelf_queue
            if not self._is_highway(x, y)
        ]

        # Made by Jw.son 2022.07.23
        # Make Agent Initial Position
        if initSettingFlag == True:
            agent_locs = []
            for human_sample in self.human_init_queue: agent_locs.append(
                human_sample[0] + self.grid_size[1] * human_sample[1])
            for robot_sample in self.robot_init_queue: agent_locs.append(
                robot_sample[0] + self.grid_size[1] * robot_sample[1])

            agent_locs = np.unravel_index(agent_locs, self.grid_size)

            # Direction Information
            # UP = 0, UPRIGHT = 1, RIGHT = 2, DOWNRIGHT = 3
            # DOWN = 4, LEFTDOWN = 5, LEFT = 6, UPLEFT = 7
            agent_dirs = []

            for _ in range(self.n_agents): agent_dirs.append(Direction.LEFT)

            self.agents = [
                Agent(x, y, dir_, self.msg_bits, self.config)
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
                    if self.grid[self.layer_spots, r, c] > 0: already_located.append(already_located_cnt)
                    if self.grid[self.layer_shelfs, r, c] > 0: already_located.append(already_located_cnt)
                    already_located_cnt = already_located_cnt + 1

            agent_locs = list()
            for i in range(self.n_agents):
                sample = random.randint(12 * self.grid_size[1], self.grid_size[0] * self.grid_size[1] - 1)
                while sample in already_located: sample = random.randint(12 * self.grid_size[1],
                                                                         self.grid_size[0] * self.grid_size[1] - 1)
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
                Agent(x, y, Direction(random.randint(0, 7)), self.msg_bits, self.config)
                for x, y in agent_locs
            ]

        ## Condition Join Point ##
        for idx in range(self.n_humans): self.agents[idx].agent_type = True  # Human
        for idx in range(self.n_humans, self.n_agents): self.agents[idx].agent_type = False

        self._recalc_grid()

        self.task_scheduler.reset_task_system()
        self.task_scheduler.initialize_task_system(self.n_agents)
        self.request_queue = self.task_scheduler.request_queue

        self.submit_command("reset")
        self._process_commands()
        self._refresh_world_state()
        self.emit_event("reset_completed", {"agents": len(self.agents)})

        return tuple(self._make_observation(agent) for agent in self.agents)

    # 웨어하우스 다음 단계 진행 메소드
    def step(
            self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        self.submit_command("apply_actions", {"actions": actions})
        self.submit_command("advance_tick")
        self._process_commands()

        applied_actions = self._pending_actions or actions
        self._pending_actions = None

        self.using_agent = 0
        cur_map = Make_Maze(self, mode=3)
        agent_queue = dict()

        failed_agents = list()
        human_commited_agents = list()
        robot_commited_agents = list()

        for agent, action in zip(self.agents, applied_actions):
            if agent.id not in self.agent_id_list: continue
            if self.msg_bits > 0:
                action_array = np.asarray(action)
                agent.req_action = Action(int(action_array[0]))
                if action_array.shape[0] > 1:
                    agent.message[:] = action_array[1:1 + self.msg_bits]
            else:
                if isinstance(action, np.ndarray):
                    scalar_action = np.asarray(action).ravel()[0]
                    agent.req_action = Action(int(scalar_action))
                else:
                    agent.req_action = Action(action)

        for agent in self.agents:
            if agent.stop_flag == True:
                failed_agents.append(agent)
                agent.stop_flag = False
                continue

            if agent.id not in self.agent_id_list: continue

            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            # Map-overlay wall/direction enforcement (WALL_ENFORCE_LEVEL == 2)
            if self.config.wall_enforce_level == 2 and target != start:
                tx, ty = target
                if self.walls[ty, tx] == 2:
                    failed_agents.append(agent)
                    continue
                if agent.agent_type == False:
                    if self.walls[ty, tx] == 1:
                        failed_agents.append(agent)
                        continue
                    bit = move_bit(tx - agent.x, ty - agent.y)
                    if bit and not (self.allowed_dirs[agent.y, agent.x] & bit):
                        failed_agents.append(agent)
                        continue

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
            if agent.id not in self.agent_id_list: continue

            agent.dir = agent.req_direction()
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)
        motion_list = [Action.UP, Action.UPRIGHT, Action.RIGHT, Action.DOWNRIGHT,
                       Action.DOWN, Action.DOWNLEFT, Action.LEFT, Action.UPLEFT]

        for agent in self.agents:
            if agent.id not in self.agent_id_list: continue
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action in motion_list:
                agent.x, agent.y = agent.req_location(self.grid_size)
                agent.dir = agent.req_direction()
                target_id = self.grid[self.layer_agents, agent.y, agent.x]
                # Robot
                if agent.agent_type == False:
                    if self.config.picking_collision_allowed:
                        if (agent.x, agent.y) in agent_queue:
                            if agent_queue[(agent.x, agent.y)] >= 2 and self.agents[
                                target_id - 1].state != State.ROBOT_PICKING:
                                agent_queue[(agent.x, agent.y)] = agent_queue[(agent.x, agent.y)] - 1
                                agent.x, agent.y = agent.prev_x, agent.prev_y
                                continue

                        if cur_map[agent.y][agent.x] > 0 and self.agents[target_id - 1].state != State.ROBOT_PICKING:
                            agent.x = agent.prev_x
                            agent.y = agent.prev_y

                        elif cur_map[agent.y][agent.x] > 0 and self.agents[target_id - 1].state == State.ROBOT_PICKING:
                            cur_map[agent.y][agent.x] = 1
                            cur_map[agent.prev_y][agent.prev_x] = 0
                            agent.stop_flag = True

                        else:
                            # This Part
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
            shelf_id = self.grid[self.layer_shelfs, x, y]

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
            self.emit_event(
                "shelf_delivered",
                {
                    "shelf_id": shelf.id,
                    "tick": self.internal_timer,
                },
            )

            # Don't Need This Project
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[self.layer_agents, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[self.layer_agents, x, y]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        # Keep task scheduler in sync with any externally tracked order count
        if hasattr(self.task_scheduler, "next_order_cnt"):
            self.task_scheduler.next_order_cnt = getattr(self, "next_order_cnt", 0)

        # Record realised arrivals for the predictive dispatch models. Only the
        # strategies that learn attach a tracker, so this is free otherwise.
        tracker = getattr(self, "arrival_tracker", None)
        if tracker is not None:
            tracker.observe()

        # Re-aim idle workers at anticipated demand.
        if str(getattr(self.config, "staging_policy", "off") or "off") != "off":
            self.get_staging_planner().update()

        terminated, termination_reason = self._should_end_episode()
        dones = [True] * self.n_agents if terminated else [False] * self.n_agents

        new_obs = tuple(self._make_observation(agent) for agent in self.agents)
        info: Dict[str, Any] = {}
        if termination_reason:
            info["termination_reason"] = termination_reason
        elif getattr(self, "_last_home_blockers", None):
            info["robots_home_blockers"] = self._last_home_blockers

        if self.grid[self.layer_agents, self.goals[0][1], self.goals[0][0]] > 0:
            self.using_station[0] += 1
        # if env.grid[self.layer_agents, self.goals[1][1], self.goals[1][0]] > 0:
        #     self.using_station[1] += 1
        for id in self.agent_id_list:
            if self.agents[id - 1].state != State.NOOP:
                self.using_agent += 1

        self._refresh_world_state()
        event_payload = {
            "tick": self.internal_timer,
            "rewards": list(rewards),
            "dones": dones,
        }
        if termination_reason:
            event_payload["termination_reason"] = termination_reason

        self.emit_event("step_completed", event_payload)

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
        mapping_horizontal = self.config.mapping_horizontal
        if mapping_horizontal == True:
            # shelfs goal position mapping
            for idx in range(len(self.shelfs)):
                x = self.shelfs[idx].x
                y = self.shelfs[idx].y
                dr = [0, 0, 1, -1]
                dc = [1, -1, 0, 0]

                for dir in range(4):
                    next_x = x + dc[dir]
                    next_y = y + dr[dir]

                    if next_x < 0 or next_x >= self.grid_size[1] or next_y < 0 or next_y >= self.grid_size[0]: continue
                    if self.grid[self.layer_spots, next_y, next_x] in [0, 6, 7, 8, 9] and self.grid[
                        self.layer_shelfs, next_y, next_x] == 0:
                        self.shelfs[idx].goal_x = next_x
                        self.shelfs[idx].goal_y = next_y
                        break

        else:
            # shelfs goal position mapping
            for idx in range(len(self.shelfs)):
                x = self.shelfs[idx].x
                y = self.shelfs[idx].y
                dr = [1, -1, 0, 0]
                dc = [0, 0, 1, -1]

                for dir in range(4):
                    next_x = x + dc[dir]
                    next_y = y + dr[dir]

                    if next_x < 0 or next_x >= self.grid_size[1] or next_y < 0 or next_y >= self.grid_size[0]: continue
                    if self.grid[self.layer_spots, next_y, next_x] in [0, 6, 7, 8, 9] and self.grid[
                        self.layer_shelfs, next_y, next_x] == 0:
                        self.shelfs[idx].goal_x = next_x
                        self.shelfs[idx].goal_y = next_y

                        break

    @property
    def human_assignment_strategy(self) -> str:
        return self._human_assignment_strategy.name

    def set_human_assignment_strategy(self, name: str) -> None:
        """Update the cooperative assignment strategy at runtime."""

        self._human_assignment_strategy = get_human_assignment_strategy(name)
        self.config.human_assignment_strategy = self._human_assignment_strategy.name
        # A strategy that dispatches on predicted arrivals needs en-route robots
        # in the candidate set; one that does not must keep the legacy set.
        if getattr(self._human_assignment_strategy, "requires_predictive_dispatch", False):
            self._predictive_dispatch = True

    @property
    def predictive_dispatch_enabled(self) -> bool:
        return bool(getattr(self, "_predictive_dispatch", False))

    def service_ticks_for(self, human: Agent, robot: Agent, sku_count: int) -> int:
        """Picking duration for one interaction, in ticks.

        Deterministic (``sku_count * sku_per_picking_time``) unless a
        variability profile is configured, in which case worker speed and
        per-interaction dispersion apply. The draw is a pure function of the
        (worker, task) pair, so evaluating a candidate never changes it.
        """

        model = getattr(self, "_service_time_model", None)
        if model is None:
            from rware.engine.service_time import build_service_time_model

            model = build_service_time_model(self.config)
            self._service_time_model = model

        if not model.enabled:
            return model.base_ticks(sku_count)

        try:
            human_index = self.human_id_list.index(human.id)
        except ValueError:
            human_index = 0
        return model.service_ticks(
            human_id=human.id,
            human_index=human_index,
            robot_id=robot.id,
            picking_seq=int(getattr(robot, "picking_seq", 0)),
            sku_count=sku_count,
        )

    def get_staging_planner(self):
        """Lazily attach the idle-worker pre-positioning planner."""

        planner = getattr(self, "staging_planner", None)
        if planner is None:
            from rware.engine.staging import StagingPlanner

            planner = StagingPlanner(self)
            self.staging_planner = planner
            # The learned policy reads robot arrival quantiles.
            if planner.policy == "learned":
                self.get_arrival_tracker().ensure_models(
                    planner.eta_backend, consumer="staging"
                )
        return planner

    def get_arrival_tracker(self):
        """Lazily attach the arrival observer used by predictive strategies."""

        tracker = getattr(self, "arrival_tracker", None)
        if tracker is None:
            from rware.engine.arrival import ArrivalTracker

            tracker = ArrivalTracker(self)
            self.arrival_tracker = tracker
        return tracker

    def _collect_waiting_humans(self, include: Optional[Agent] = None) -> List[Agent]:
        """Gather human agents that are eligible for matching on the current tick."""

        candidates: List[Agent] = []
        threshold = 3  # legacy value used in Agent.check_status
        include_id = include.id if include is not None else None

        for human_id in self.human_id_list:
            human = self.agents[human_id - 1]
            if human.agent_type is not True:
                continue
            if human.coworker is not None:
                continue
            if human.state not in (State.NOOP, State.HOME):
                continue
            if human.agent_timer <= threshold and human_id != include_id:
                continue
            candidates.append(human)

        if include and include not in candidates:
            candidates.append(include)
        return candidates

    def _collect_available_robots(self, include_with_coworker: bool = False) -> List[Agent]:
        """Return robots that are waiting for human cooperation.

        Args:
            include_with_coworker: Re-auction을 위해 이미 coworker가 있는 로봇도 포함할지 여부
        """

        # Predictive dispatch also offers robots that are still travelling to
        # their rack, so a worker can set off before the robot parks instead of
        # after. Without it the worker's whole trip is charged to robot waiting.
        eligible = {State.ROBOT_PICKING}
        if self.predictive_dispatch_enabled:
            eligible.add(State.ROBOT_MOVESPOT)

        robots: List[Agent] = []
        for robot_id in self.robot_id_list:
            robot = self.agents[robot_id - 1]
            if robot.agent_type is True:
                continue
            if robot.state not in eligible:
                continue
            # An en-route robot is only a candidate once its next rack is known.
            if robot.state == State.ROBOT_MOVESPOT and not robot.node_list:
                continue
            # Re-auction을 위해 coworker가 있는 로봇도 포함
            if robot.coworker is not None and not include_with_coworker:
                continue
            robots.append(robot)
        return robots

    def _build_matching_context(
        self,
        humans: List[Agent],
        robots: List[Agent],
    ) -> MatchingContext:
        working_areas = self.get_human_working_areas()
        human_index_lookup: Dict[int, int] = {
            hid: idx for idx, hid in enumerate(self.human_id_list)
        }

        human_snapshots: List[HumanSnapshot] = []
        for human in humans:
            human_idx = human_index_lookup.get(human.id, -1)
            zone_nodes: List[int] = []
            if 0 <= human_idx < len(working_areas):
                zone_nodes = list(working_areas[human_idx])
            human_snapshots.append(
                HumanSnapshot(
                    id=human.id,
                    position=(human.x, human.y),
                    agent_timer=human.agent_timer,
                    waiting_time=human.waiting_time,
                    zone_nodes=zone_nodes,
                    state=human.state,
                )
            )

        robot_snapshots: List[RobotSnapshot] = []
        for robot in robots:
            node_id = self.routing_node_dict.get((robot.x, robot.y))
            pending_items = 0
            if getattr(robot, "order_sku_cnt", None):
                pending_items = int(robot.order_sku_cnt[0]) if robot.order_sku_cnt else 0
            estimated_service = (
                pending_items * self.config.sku_per_picking_time
                if pending_items > 0
                else float(self.config.sku_per_picking_time)
            )
            rack_id = int(robot.node_list[0]) if getattr(robot, "node_list", None) else None
            if rack_id is not None and 1 <= rack_id <= len(self.shelfs):
                shelf = self.shelfs[rack_id - 1]
                target_position = (int(shelf.goal_x), int(shelf.goal_y))
            else:
                target_position = (robot.x, robot.y)
            robot_snapshots.append(
                RobotSnapshot(
                    id=robot.id,
                    position=(robot.x, robot.y),
                    state=robot.state,
                    waiting_time=robot.waiting_time,
                    routing_node_id=node_id,
                    pending_items=pending_items,
                    estimated_service_time=estimated_service,
                    target_position=target_position,
                    rack_id=rack_id,
                    remaining_path=len(getattr(robot, "path_planning", []) or []),
                )
            )

        metrics = {
            "avg_robot_waiting_time": float(
                np.mean([robot.waiting_time for robot in robots]) if robots else 0.0
            ),
            "avg_human_waiting_time": float(
                np.mean([human.waiting_time for human in humans]) if humans else 0.0
            ),
            "avg_estimated_service_time": float(
                np.mean([snapshot.estimated_service_time for snapshot in robot_snapshots])
                if robot_snapshots
                else 0.0
            ),
        }

        return MatchingContext(
            tick=self.internal_timer,
            humans=human_snapshots,
            robots=robot_snapshots,
            metrics=metrics,
        )

    def _refresh_assignment_plan(self, trigger: Optional[Agent] = None) -> None:
        # Re-auction is a property of the auction machinery, not of one
        # registered name. Matching on the name silently disabled it for every
        # subclass (the rv_* rendezvous family), which no change inside the
        # strategy class could undo.
        is_auction_strategy = isinstance(
            self._human_assignment_strategy, AuctionAssignmentStrategy
        )
        reauction_enabled = (
            self._human_assignment_strategy.is_reauction_enabled(self.config)
            if is_auction_strategy
            else False
        )

        humans = self._collect_waiting_humans(include=trigger)
        # Re-auction이 활성화된 auction 전략에서는 이미 coworker가 있는 로봇도 포함
        include_with_coworker = reauction_enabled and is_auction_strategy
        robots = self._collect_available_robots(include_with_coworker=include_with_coworker)

        if not humans or not robots:
            self._pending_human_assignments.clear()
            return

        context = self._build_matching_context(humans, robots)
        assignments = self._human_assignment_strategy.plan_assignments(self, context)

        filtered: Dict[int, int] = {}
        reassignments: List[tuple[int, int, int]] = []  # (new_human, robot, old_human)

        eligible_states = {State.ROBOT_PICKING}
        if self.predictive_dispatch_enabled:
            eligible_states.add(State.ROBOT_MOVESPOT)

        for human_id, robot_id in assignments.items():
            if not human_id or not robot_id:
                continue
            human_agent = self.agents[human_id - 1]
            robot_agent = self.agents[robot_id - 1]

            # 새 human이 이미 다른 로봇과 작업 중이면 무시
            if human_agent.coworker is not None:
                continue

            # 로봇이 협업 가능한 상태가 아니면 무시
            if robot_agent.state not in eligible_states:
                continue
            # 이동 중인 로봇은 목표 랙이 정해져 있어야 작업자를 보낼 수 있다
            if robot_agent.state == State.ROBOT_MOVESPOT and not robot_agent.node_list:
                continue

            # 로봇에 이미 coworker가 있는 경우
            if robot_agent.coworker is not None:
                # Re-auction이 활성화된 auction 전략에서만 재할당 허용
                if reauction_enabled and is_auction_strategy:
                    old_human_id = robot_agent.coworker
                    reassignments.append((human_id, robot_id, old_human_id))
                continue

            filtered[human_id] = robot_id

        # Re-auction 재할당 처리
        for new_human_id, robot_id, old_human_id in reassignments:
            # 기존 human의 상태 및 coworker 해제
            old_human_agent = self.agents[old_human_id - 1]
            if old_human_agent.coworker == robot_id:
                old_human_agent.coworker = None
                old_human_agent.node_list = []  # 목표 노드 초기화
                old_human_agent.agent_timer = 0  # 대기 타이머 리셋
                # 상태를 NOOP으로 리셋하여 다시 대기 상태로 전환
                old_human_agent.state = State.NOOP

            # 로봇의 coworker 해제 (새로 설정될 예정)
            robot_agent = self.agents[robot_id - 1]
            robot_agent.coworker = None

            # 새 할당 추가
            filtered[new_human_id] = robot_id

        self._pending_human_assignments = filtered

    def get_human_working_areas(self) -> List[List[int]]:
        """Return the active working zones for each human agent."""

        total_nodes = list(range(len(self.routing_node_all_pos)))
        if not total_nodes:
            total_nodes = [0]

        human_count = max(1, len(self.human_id_list) or self.n_max_humans or 1)

        def normalize(area: List[List[int]]) -> List[List[int]]:
            cleaned = [list(nodes) for nodes in area if nodes] if area else []
            if not cleaned:
                cleaned = [total_nodes.copy()]
            result: List[List[int]] = []
            idx = 0
            while len(result) < human_count:
                template = cleaned[idx % len(cleaned)]
                result.append(list(template))
                idx += 1
            return result[:human_count]

        if self.config.human_zone_strategy == HumanZoneStrategy.ALL:
            return [total_nodes.copy() for _ in range(human_count)]
        if self.config.human_zone_strategy == HumanZoneStrategy.BIG_ASILE:
            return normalize(self.big_asile)
        if self.config.human_zone_strategy == HumanZoneStrategy.SMALL_ASILE:
            return normalize(self.small_asile)
        return normalize(self.big_asile or self.small_asile)

    def select_coworker(self, cur_agent):
        """Return the robot id that should collaborate with ``cur_agent``."""

        if self._assignment_plan_tick != self.internal_timer:
            self._assignment_plan_tick = self.internal_timer
            self._pending_human_assignments.clear()

        if cur_agent.id not in self._pending_human_assignments:
            self._refresh_assignment_plan(trigger=cur_agent)

        robot_id = self._pending_human_assignments.get(cur_agent.id)
        if robot_id is None:
            return None

        # Consume this assignment to prevent duplicate pairing.
        del self._pending_human_assignments[cur_agent.id]
        self.assignment_total_count += 1
        if self.agents[robot_id - 1].state == State.ROBOT_MOVESPOT:
            self.assignment_en_route_count += 1
        return robot_id

    def making_routing_node(self):
        map = self.grid[self.layer_shelfs]
        facility = self.grid[self.layer_spots]

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
        # y_list = [(1,3) ,(6,7) ,(10,11),(14,15),(18,19),(22,23),(26,27),(30,31),(34,35),(38,39),(42,43),(46,47),(50,51),(54,55),(58,59),(62,63),(66,67),(70,71),(74,76)]
        # x_list = [(1, 14), (17, 16), (19, 26), (29, 30), (33, 51), (54, 57), (60, 64), (67, 72), (75, 84)]

        x_list = [(1, 1), (4, 18), (21, 20), (23, 36), (39, 38), (41, 54), (57, 67), (70, 75)]
        y_list = []
        for mul in range(35 + 1): y_list.append((((4 * mul) + 1), ((4 * mul) + 2)))

        #
        for srt_y, dst_y in y_list:
            for srt_x, dst_x in x_list:
                node_identifier.append([[srt_x, srt_y], [dst_x, dst_y]])

        self.node_identifier = node_identifier
        self.routing_node = [node_srt_dst[1] for node_srt_dst in node_identifier]
        self.total_robot_cnt_in_zone = [0 for _ in node_identifier]
        self.total_timeout_cnt_in_zone = [0 for _ in node_identifier]
        self.routing_node_all_pos = [[] for _ in range(len(node_identifier))]

        overlap = [[] for _ in range(len(self.shelf_queue) + 1)]
        rack_list = [[] for _ in range(len(node_identifier))]

        for idx in range(len(node_identifier)):
            srt_x = node_identifier[idx][0][0]
            srt_y = node_identifier[idx][0][1]
            dst_x = node_identifier[idx][1][0]
            dst_y = node_identifier[idx][1][1]

            for y in range(srt_y - 1, dst_y + 1 + 1):
                for x in range(srt_x - 1, dst_x + 1 + 1):
                    if map[y][x] > 0:
                        rack_list[idx].append(map[y][x])
                        if idx not in self.zone_list_in_rack: self.zone_list_in_rack.append(idx)

                    elif map[y][x] == 0 and (facility[y][x] in [0, 1, 2, 4, 5, 6, 7, 8, 9]):
                        self.routing_node_all_pos[idx].append((x, y))

        idx = -1
        for routing_node in self.routing_node_all_pos:
            idx = idx + 1
            for x, y in routing_node:
                self.routing_node_dict[(x, y)] = idx

        for idx in range(len(rack_list)):
            if len(rack_list[idx]) <= 0: continue
            for sample in rack_list[idx]:
                overlap[sample].append(idx)

        print(
            "########################################################################################## Rack List In Zone #######################################################################################################")
        for idx in range(len(rack_list)): print("zone Id : ", idx, ", rack_list : ", rack_list[idx], ", length : ",
                                                len(rack_list[idx]))
        print()

        print(
            "########################################################################################## Position In Node #######################################################################################################")
        for idx in range(len(self.routing_node_all_pos)):
            print("zone Id : ", idx, ", pos cnt : ", len(self.routing_node_all_pos[idx]))
            print("pos list : ", self.routing_node_all_pos[idx])
        print()

        self.rack_list = rack_list
        print(rack_list)
        print(self.zone_list_in_rack)

        return rack_list, overlap

    def making_routing_graph(self):
        # Simply define all of Routing Point with Block Point
        edge_map = list()
        exception_node = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 139, 140, 148, 149, 157, 158, 166, 167, 160, 161,
                          169, 170]
        for num in range(19):
            tmp = []
            for adder in range(9):
                if (9 * num) + adder in exception_node:
                    tmp.append(-1)
                else:
                    tmp.append((9 * num) + adder)
            edge_map.append(tmp)
        print(edge_map)

        self.edge_map = edge_map

        # Calibration Process
        cut_edge_list = []
        cut_edge_list = [ \
            # col0
            (36, 45), (45, 54), (54, 63), (63, 72), (72, 81), (81, 90), (90, 99), (99, 108), (108, 117), (117, 126),
            (126, 135), (135, 144), (144, 153), (153, 162),
            # col1
            # col2
            (38, 47), (47, 56), (56, 65), (65, 74), (74, 83), (83, 92), (92, 101), (101, 110), (110, 119), (119, 128),
            (128, 137), (137, 146), (146, 155), (155, 164),
            # col3
            (48, 57), (57, 66), (66, 75), (75, 84), (84, 93), (93, 102), (102, 111), (111, 120),
            # col4
            (4, 13), (13, 22), (22, 31), (31, 40), (40, 49), (49, 58), (58, 67), (67, 76), (76, 85), (85, 94),
            (94, 103), (103, 112), (112, 121), (121, 130),
            # col5
            (59, 68), (68, 77), (77, 86), (86, 95), (95, 104), (104, 113), (113, 122), (122, 131),
            # col6
            (6, 15), (15, 24),
            # col7
            (7, 16), (16, 25), (25, 34),
            # col8
            (8, 17), (17, 26),

            # row
            (25, 26), (34, 35), (42, 43), (43, 44), (51, 52), (52, 53), (79, 80), (88, 89), (97, 98), (106, 107),

            # Station
            (33, 32), (33, 24), (33, 42)
        ]

        # Check edge list
        dy = [1, -1, 0, 0]
        dx = [0, 0, 1, -1]

        edge_list = []
        x_max = len(edge_map[0])
        y_max = len(edge_map)

        for y in range(y_max):
            for x in range(x_max):
                tmp = []
                if edge_map[y][x] < 0:
                    continue
                else:
                    srt_routing_node = edge_map[y][x]
                    for dir in range(4):
                        next_y = y + dy[dir]
                        next_x = x + dx[dir]
                        if next_y < 0 or next_y >= y_max or next_x < 0 or next_x >= x_max: continue
                        if edge_map[next_y][next_x] > 0:
                            dst_routing_node = edge_map[next_y][next_x]
                            if srt_routing_node == dst_routing_node: continue
                            if (srt_routing_node, dst_routing_node) in cut_edge_list or (
                            dst_routing_node, srt_routing_node) in cut_edge_list: continue
                            edge_list.append((srt_routing_node, dst_routing_node, 1))
                            tmp.append((srt_routing_node, dst_routing_node, 1))
                        else:
                            continue

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
            node_num = self.routing_node_dict[(robot.x, robot.y)]
            robot_in_routing_node[node_num] += 1

        # Update Routing Edge
        for idx in range(len(robot_in_routing_node)):
            if robot_in_routing_node[idx] <= 0:
                continue
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

    def check_routing_graph(self):
        robot_in_routing_node = [0 for _ in range(len(self.routing_node))]

        # Edge Clear
        for idx in range(len(robot_in_routing_node)):
            for target in list(self.routing_graph.neighbors(idx)):
                self.routing_graph[idx][target]['weight'] = 1

        # Count Robot Cnt in Routing Node
        for robot in self.agents[self.n_humans:]:
            node_num = self.routing_node_dict[(robot.x, robot.y)]
            robot_in_routing_node[node_num] += 1

        # Update Routing Edge
        for idx in range(len(robot_in_routing_node)):
            if robot_in_routing_node[idx] <= 0:
                continue
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

        # if CHK_ROUTING_FLAG == 1:
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
        self.routing_graph[33][34]['weight'] = 1
        self.routing_graph[34][33]['weight'] = 1

        # if CHK_ROUTING_FLAG == 1:
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

    def check_running_agent_cnt(self):
        self.running_human_cnt = 0
        self.running_robot_cnt = 0

        for idx in range(self.n_humans):
            if idx + 1 not in self.agent_id_list: continue
            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME:
                continue
            else:
                self.running_human_cnt += 1
                self.agents[idx].working_time += 1

        for idx in range(self.n_humans, self.n_agents):
            if idx + 1 not in self.agent_id_list: continue
            # zone cnt
            zone_idx = self.routing_node_dict[(self.agents[idx].x, self.agents[idx].y)]
            self.total_robot_cnt_in_zone[zone_idx] += 1
            self.total_map_cnt[self.agents[idx].y][self.agents[idx].x] += 1

            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME:
                continue
            else:
                self.running_robot_cnt += 1
                self.agents[idx].working_time += 1





# Robot and Human Cnt Automated Calc
def AgentCounter(map):
    human_cnt = 0
    robot_cnt = 0

    map_list = map.split()
    for current_row in map_list:
        # Skip map-DSL section markers (e.g. "[base]", "[overlay]"); their
        # literal text (e.g. the "r" in "overlay") is not grid content.
        if current_row.startswith('[') and current_row.endswith(']'):
            continue
        for idx in range(len(current_row)):
            if current_row[idx] == 'h':
                human_cnt += 1
                continue
            if current_row[idx] == 'r':
                robot_cnt += 1
                continue

    return human_cnt, robot_cnt

def AddHuman(env, cnt):
    for _ in range(cnt):
        if len(env.deleted_human_id_list) > 0:
            env.human_id_list.append(env.deleted_human_id_list[0])
            env.agent_id_list.append(env.deleted_human_id_list[0])
            del env.deleted_human_id_list[0]

        else:
            last_human_id = 0
            if len(env.human_id_list) > 0: last_human_id = env.human_id_list[-1]
            new_human_id = last_human_id + 1
            if new_human_id > len(env.human_init_queue): break
            else:
                env.human_id_list.append(new_human_id)
                env.agent_id_list.append(new_human_id)

    env.human_id_list = sorted(env.human_id_list)
    env.agent_id_list = sorted(env.agent_id_list)

    return env
def AddRobot(env, cnt):
    for _ in range(cnt):
        if len(env.deleted_robot_id_list) > 0:
            env.robot_id_list.append(env.deleted_robot_id_list[0])
            env.agent_id_list.append(env.deleted_robot_id_list[0])
            del env.deleted_robot_id_list[0]

        else:
            last_robot_id = len(env.human_init_queue)
            if len(env.robot_id_list) > 0: last_robot_id = env.robot_id_list[-1]
            new_robot_id = last_robot_id + 1
            print("new_robot_id : ",new_robot_id)
            if new_robot_id > (len(env.human_init_queue)+len(env.robot_init_queue)): break
            else:
                env.robot_id_list.append(new_robot_id)
                env.agent_id_list.append(new_robot_id)

    env.robot_id_list = sorted(env.robot_id_list)
    env.agent_id_list = sorted(env.agent_id_list)

    return env
def DelHuman(env, cnt):
    for _ in range(cnt):
        flag = False
        if len(env.human_id_list) <= 0: break
        else:
            # Human State Not Picking
            for idx in reversed(range(len(env.human_id_list))):
                id    = env.human_id_list[idx]
                state = env.human_id_list[idx]

                if state != State.HUMAN_PICKING and state != State.HUMAN_DONE:
                    if env.agents[id-1].coworker is not None:
                        env.agents[env.agents[id-1].coworker - 1].coworker = None

                    env.agents[id - 1].node_list = list()
                    env.agents[id - 1].state = State.NOOP
                    env.agents[id - 1].coworker = None
                    env.agents[id - 1].x = env.agents[id - 1].init_x
                    env.agents[id - 1].y = env.agents[id - 1].init_y

                    env.deleted_human_id_list.append(id)
                    del env.human_id_list[idx]
                    env.agent_id_list = env.human_id_list + env.robot_id_list
                    flag = True
                    break

            if flag == True: continue
            else:
                # Human State Not Picking
                for idx in reversed(range(len(env.human_id_list))):
                    id = env.human_id_list[idx]
                    state = env.human_id_list[idx]

                    if state == State.HUMAN_PICKING or state == State.HUMAN_DONE:
                        if env.agents[id - 1].coworker is not None:
                            env.agents[env.agents[id - 1].coworker - 1].coworker = None

                        env.agents[id - 1].node_list = list()
                        env.agents[id - 1].state = State.NOOP
                        env.agents[id - 1].coworker = None
                        env.agents[id - 1].x = env.agents[id - 1].init_x
                        env.agents[id - 1].y = env.agents[id - 1].init_y

                        env.deleted_human_id_list.append(id)
                        del env.human_id_list[idx]
                        env.agent_id_list = env.human_id_list + env.robot_id_list
                        flag = True
                        break

    return env

def DelRobot(env, cnt):
    left_order = []
    for _ in range(cnt):
        if len(env.robot_id_list) <= 0:
            break
        else:
            # Human State Not Picking
            for idx in reversed(range(len(env.robot_id_list))):
                id = env.robot_id_list[idx]
                state = env.robot_id_list[idx]

                if state == State.ROBOT_LOAD or state == State.NOOP or state == State.HOME:
                    if env.agents[id - 1].coworker is not None:
                        env.agents[env.agents[id - 1].coworker - 1].coworker = None
                        env.agents[env.agents[id - 1].coworker - 1].node_list = list()
                        env.agents[env.agents[id - 1].coworker - 1].state = State.NOOP

                    env.agents[id - 1].node_list = list()
                    env.agents[id - 1].state = State.NOOP
                    env.agents[id - 1].coworekr = None
                    env.agents[id - 1].x = env.agents[id - 1].init_x
                    env.agents[id - 1].y = env.agents[id - 1].init_y

                    env.deleted_robot_id_list.append(id)
                    del env.robot_id_list[idx]
                    env.agent_id_list = env.human_id_list + env.robot_id_list
                    break
    return env

def _split_agent_rows(df_agent, human_cnt: int, robot_cnt: int):
    """Return the human and robot blocks from the human-first agent table."""

    expected = human_cnt + robot_cnt
    if len(df_agent) < expected:
        raise ValueError(
            f"agent table has {len(df_agent)} rows, expected at least {expected}"
        )
    return df_agent.iloc[:human_cnt], df_agent.iloc[human_cnt:expected]


def WriteLog(env, robot_cnt, human_cnt, time, stime, etime, simulation_name: Optional[str] = None):
    cfg = getattr(env, "config", SimulationConfig.from_legacy_config())

    # Write Pandas
    time_str = str(datetime.now().strftime('%Y%m%d%H%M%S'))
    # 결과는 rware/data/<simulation_name>/<timestamp>/ 아래로 저장
    def _sanitize(name: str) -> str:
        cleaned = []
        for ch in (name or ""):
            if ch.isalnum() or ch in ("-", "_"):
                cleaned.append(ch)
            elif ch.isspace():
                cleaned.append("_")
        result = "".join(cleaned).strip("_").lower()
        return result or "default"

    sim_dir_name = _sanitize(simulation_name or getattr(env, "human_assignment_strategy", "") or "default")
    data_root = Path(__file__).resolve().parents[1] / "data"
    base_dir = data_root / sim_dir_name / time_str

    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base_dir}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    path = str(run_dir)
    # Keep artifact names aligned with the directory when parallel runs finish
    # in the same second and a numeric suffix is required.
    time_str = run_dir.name

    # Agent
    columns = ["Agent ID", "Type", "VisitedRackCnt", "WorkingTime", "WorkingTimeRatio", "TimeoutCnt", "TotalDistance",
               "TotalSkuCnt", "Productivity", "SecPerPick","WaitingTime"]
    data_set = list()
    for i in range(env.n_max_agents):
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
            if  max_val <= 0: max_val = 1
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
    human_rows, robot_rows = _split_agent_rows(df_agent, human_cnt, robot_cnt)
    human_grid_distance_np = np.array(list(human_rows.iloc[:, 6])).astype(int)
    robot_grid_distance_np = np.array(list(robot_rows.iloc[:, 6])).astype(int)
    human_distance_np = human_grid_distance_np * cfg.distance_per_grid
    robot_distance_np = robot_grid_distance_np * cfg.distance_per_grid

    human_sku_cnt_np = np.array(list(human_rows.iloc[:, 7])).astype(int)  # sku_cnt
    robot_sku_cnt_np = np.array(list(robot_rows.iloc[:, 7])).astype(int)

    robot_timeout_cnt_np = np.array(list(robot_rows.iloc[:, 5])).astype(int)

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

    tick = cfg.tick_per_time
    human_sec_per_pick = (env.internal_timer * tick) / human_avg_sku_cnt
    robot_sec_per_pick = (env.internal_timer * tick) / robot_avg_sku_cnt
    box_per_hour_human = (((env.completed_batch * cfg.loadbox_count) / (env.internal_timer * tick)) * 3600) / human_cnt

    cur_hour = int(env.internal_timer * tick) // 3600
    cur_min = (int(env.internal_timer * tick) % 3600) // 60
    cur_sec = ((int(env.internal_timer * tick) % 3600) % 60)
    cur_time_str = str("Time : {0:02d} : {1:02d} : {2:02d}\n".format(cur_hour, cur_min, cur_sec))
    cur_completed_order_str = str("Batch Done : {:04d}\n".format(env.completed_batch))
    cur_completed_box_str = str("Box Done : {:04d}\n".format(cfg.loadbox_count * env.completed_batch))
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

    strategy_str = (
        "OrderBatch|HumanZone|OrderSeq|Route|HumanMove : "
        f"{cfg.order_batch_strategy.value}|{cfg.human_zone_strategy.value}|"
        f"{cfg.order_sequence_flag}|{cfg.routing_strategy}|{int(cfg.human_move_strategy)}\n"
    )
    sim_info = (
        "BoxLoad|SKUExit|SKUperPicking|Human_cnt|Robot_cnt : "
        f"{cfg.box_loading_time}|{cfg.sku_per_exit_time}|{cfg.sku_per_picking_time}|{human_cnt}|{robot_cnt}\n"
    )
    sim_info_2 = (
        "StaticPath|PickingCollision : "
        f"0|{int(cfg.picking_collision_allowed)}"
    )
    # Robustness-sweep runs of the same strategy land in sibling directories
    # that differ only by timestamp, so record which scenario produced this one.
    from rware.config.defaults import ORDER_DATE

    scenario_info = (
        "\nServiceVariability|ServiceSeed|Staging : "
        f"{cfg.service_time_variability}|{cfg.service_time_seed}|"
        f"{cfg.staging_policy}\n"
        "ExperimentID : "
        f"{os.environ.get('RWARE_EXPERIMENT_ID', '')}\n"
        "StagingEarlyW|StagingUncertaintyW : "
        f"{cfg.staging_early_weight}|{cfg.staging_uncertainty_weight}\n"
        "StagingBackend : "
        f"{cfg.staging_eta_backend}\n"
        "OrderDate|WallEnforce|Strategy : "
        f"{ORDER_DATE}|{cfg.wall_enforce_level}|"
        f"{getattr(env, 'human_assignment_strategy', 'unknown')}\n"
    )

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
    f.write(scenario_info)

    # Robot idle time is the metric predictive dispatch targets, so report it
    # directly rather than leaving it to be derived from the per-agent CSV.
    robot_wait_total = sum(
        env.agents[robot_id - 1].waiting_time for robot_id in env.robot_id_list
    )
    robot_wait_share = (
        robot_wait_total / (len(env.robot_id_list) * max(1, env.internal_timer))
        if env.robot_id_list
        else 0.0
    )
    f.write("\nRobot wait total ticks : {}\n".format(robot_wait_total))
    f.write("Robot wait share : {:.6f}\n".format(robot_wait_share))
    f.write("Human total grid steps : {}\n".format(int(np.sum(human_grid_distance_np))))
    f.write("Human mean grid steps : {:.6f}\n".format(np.mean(human_grid_distance_np)))
    f.write("Distance per grid : {:.6f}\n".format(cfg.distance_per_grid))
    assignment_total = int(getattr(env, "assignment_total_count", 0))
    assignment_en_route = int(getattr(env, "assignment_en_route_count", 0))
    assignment_share = assignment_en_route / assignment_total if assignment_total else 0.0
    f.write("Assignment total : {}\n".format(assignment_total))
    f.write("Assignment en route : {}\n".format(assignment_en_route))
    f.write("Assignment en route share : {:.6f}\n".format(assignment_share))

    tracker = getattr(env, "arrival_tracker", None)
    if tracker is not None:
        for key, value in tracker.report().items():
            f.write("{} : {}\n".format(key, value))

    planner = getattr(env, "staging_planner", None)
    if planner is not None:
        for key, value in planner.report().items():
            f.write("{} : {}\n".format(key, value))

    f.close()
