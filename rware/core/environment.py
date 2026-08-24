"""
Environment Core for Robotic Warehouse Simulation
환경 상태 관리 및 그리드 시스템을 담당하는 모듈
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict

from rware.core.state import Direction
from rware.core.config import SimulationConfig
from rware.core.map_dsl import parse_map_text, ALL_DIRS
from rware.entities import Agent, Shelf


class EnvironmentCore:
    """
    Warehouse 환경의 핵심 상태를 관리하는 클래스

    이 클래스는 그리드, 에이전트, 선반 등의 환경 상태를 중앙에서 관리하며,
    Warehouse 클래스에서 환경 관련 로직을 분리합니다.
    """

    def __init__(self, config: SimulationConfig):
        """
        EnvironmentCore 초기화

        Args:
            config (SimulationConfig): 시뮬레이션 설정
        """
        self.config = config

        # 그리드 시스템
        self.grid_size: Tuple[int, int] = (0, 0)
        self.grid: Optional[np.ndarray] = None
        self.highways: Optional[np.ndarray] = None
        self.walls: Optional[np.ndarray] = None
        self.allowed_dirs: Optional[np.ndarray] = None

        # 엔티티들
        self.agents: List[Agent] = []
        self.shelfs: List[Shelf] = []

        # 위치 관련 큐들
        self.goals: List[Tuple[int, int]] = []
        self.picking_queue: List[Tuple[int, int]] = []
        self.loadbox_queue: List[Tuple[int, int]] = []
        self.wait_queue: List[Tuple[int, int]] = []
        self.shelf_queue: List[Tuple[int, int]] = []
        self.human_init_queue: List[Tuple[int, int]] = []
        self.robot_init_queue: List[Tuple[int, int]] = []

        # 카운터 및 상태 변수들
        self.wait_queue_cnt: List[int] = []
        self.waitLOADBOX_CNT: List[int] = []

        # 타이머 및 통계
        self.internal_timer: int = 0
        self.completed_batch: int = 0
        self.all_of_completed_order: int = 0

        # 기타 상태
        self.using_station: List[int] = [0, 0]
        self.using_agent: int = 0

    def initialize_grid(self, layout: Optional[str] = None,
                       shelf_columns: int = 3, shelf_rows: int = 2,
                       column_height: int = 8) -> None:
        """
        그리드 시스템 초기화

        Args:
            layout: 사용자 정의 레이아웃 문자열
            shelf_columns: 선반 열 개수
            shelf_rows: 선반 행 개수
            column_height: 선반 높이
        """
        if layout:
            self._make_layout_from_str(layout)
        else:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)

    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int) -> None:
        """파라미터 기반 레이아웃 생성"""
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.grid = np.zeros((self.config.collision_layers, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
                (x % 3 == 0)  # vertical highways
                or (y % (column_height + 1) == 0)  # horizontal highways
                or (y == self.grid_size[0] - 1)  # delivery row
                or (  # remove a box for queuing
                        (y > self.grid_size[0] - (column_height + 3))
                        and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
                )
        )
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)

        self.walls = np.zeros(self.grid_size, dtype=np.uint8)
        self.allowed_dirs = np.full(self.grid_size, ALL_DIRS, dtype=np.uint8)

    def _make_layout_from_str(self, layout: str) -> None:
        """문자열 기반 레이아웃 생성"""
        cfg = self.config
        vertical_idx = cfg.shelf_vertical_idx
        layer_spots = cfg.layer_spots
        parsed = parse_map_text(layout)
        self.walls = parsed.walls
        self.allowed_dirs = parsed.allowed_dirs
        layout = parsed.base
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])

        vector = ['' for _ in range(grid_width)]

        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        if vertical_idx == False:
            for col in range(grid_width):
                for line in lines:
                    vector[col] = vector[col] + str(line[col])

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((cfg.collision_layers, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        if vertical_idx == True:
            for y, line in enumerate(lines):
                for x, char in enumerate(line):
                    assert char.lower() in "gpwboemzxrnh."
                    c = char.lower()
                    if c == "g":
                        self.grid[layer_spots, y, x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif c == "w":
                        self.grid[layer_spots, y, x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif c == "p":
                        self.grid[layer_spots, y, x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif c == "b":
                        self.grid[layer_spots, y, x] = 4
                        self.loadbox_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif c == "n":
                        self.grid[layer_spots, y, x] = 5
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif c == "o":
                        self.grid[layer_spots, y, x] = 6
                        self.highways[y, x] = 1
                    elif c == "e":
                        self.grid[layer_spots, y, x] = 7
                        self.highways[y, x] = 1
                    elif c == "m":
                        self.grid[layer_spots, y, x] = 8
                        self.highways[y, x] = 1
                    elif c == "z":
                        self.grid[layer_spots, y, x] = 9
                        self.highways[y, x] = 1
                    elif c == "x":
                        self.shelf_queue.append((x, y))
                    elif c == "h":
                        self.human_init_queue.append((x, y))
                    elif c == "r":
                        self.robot_init_queue.append((x, y))
                    elif c == ".":
                        self.highways[y, x] = 1
        else:
            for x, line in enumerate(vector):
                for y, char in enumerate(line):
                    assert char.lower() in "gpwxrh."
                    _apply_char = char.lower()
                    if _apply_char == "g":
                        self.grid[layer_spots, y, x] = 1
                        self.goals.append((x, y))
                        self.highways[y, x] = 0
                    elif _apply_char == "w":
                        self.grid[layer_spots, y, x] = 2
                        self.wait_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif _apply_char == "p":
                        self.grid[layer_spots, y, x] = 3
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif _apply_char == "b":
                        self.grid[layer_spots, y, x] = 4
                        self.loadbox_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif _apply_char == "n":
                        self.grid[layer_spots, y, x] = 5
                        self.picking_queue.append((x, y))
                        self.highways[y, x] = 1
                    elif _apply_char == "o":
                        self.grid[layer_spots, y, x] = 6
                        self.highways[y, x] = 1
                    elif _apply_char == "e":
                        self.grid[layer_spots, y, x] = 7
                        self.highways[y, x] = 1
                    elif _apply_char == "m":
                        self.grid[layer_spots, y, x] = 8
                        self.highways[y, x] = 1
                    elif _apply_char == "z":
                        self.grid[layer_spots, y, x] = 9
                        self.highways[y, x] = 1
                    elif _apply_char == "x":
                        self.shelf_queue.append((x, y))
                    elif _apply_char == "h":
                        self.human_init_queue.append((x, y))
                    elif _apply_char == "r":
                        self.robot_init_queue.append((x, y))
                    elif _apply_char == ".":
                        self.highways[y, x] = 1

        self.wait_queue_cnt = [0 for _ in range(len(self.goals))]
        self.waitLOADBOX_CNT = [0 for _ in range(len(self.loadbox_queue))]
        assert len(self.goals) >= 1, "At least one goal is required"

    def update_grid(self) -> None:
        """그리드 상태 업데이트"""
        layer_shelfs = self.config.layer_shelfs
        layer_agents = self.config.layer_agents

        self.grid[layer_shelfs] = 0
        self.grid[layer_agents] = 0

        for s in self.shelfs:
            self.grid[layer_shelfs, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[layer_agents, a.y, a.x] = a.id

    def is_highway(self, x: int, y: int) -> bool:
        """해당 위치가 highway인지 확인"""
        return bool(self.highways[y, x])

    def reset_environment(self) -> None:
        """환경 상태 초기화"""
        # 선반 초기화
        from rware.entities import Shelf
        Shelf.counter = 0

        self.shelfs = [
            Shelf(x, y, self.config)
            for x, y in self.shelf_queue
            if not self.is_highway(x, y)
        ]

        # 그리드 업데이트
        self.update_grid()

    def get_grid_info(self) -> Dict:
        """그리드 관련 정보 반환"""
        return {
            'grid_size': self.grid_size,
            'grid': self.grid,
            'highways': self.highways,
            'goals': self.goals,
            'shelfs': self.shelfs,
            'agents': self.agents
        }

    def get_statistics(self) -> Dict:
        """현재 환경 통계 반환"""
        return {
            'internal_timer': self.internal_timer,
            'completed_batch': self.completed_batch,
            'all_of_completed_order': self.all_of_completed_order,
            'using_station': self.using_station,
            'using_agent': self.using_agent,
            'n_shelfs': len(self.shelfs),
            'n_agents': len(self.agents)
        }
