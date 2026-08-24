"""
Task Scheduler for Robotic Warehouse Simulation
작업 할당 및 스케줄링을 담당하는 모듈
"""

from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from rware.core.state import State
from rware.core.config import SimulationConfig


class TaskScheduler:
    """
    작업 할당 및 스케줄링을 관리하는 클래스

    이 클래스는 주문 처리, 작업 할당, 우선순위 결정 등의
    스케줄링 로직을 담당합니다.
    """

    def __init__(self, config: SimulationConfig):
        """
        TaskScheduler 초기화

        Args:
            config (SimulationConfig): 시뮬레이션 설정
        """
        self.config = config

        # 작업 관련 데이터
        self.request_queue: List = []
        self.request_queue_size: int = config.request_queue_size or 5
        self.next_order_cnt: int = 0
        self.current_loaded_node_list: List[List] = []

        # 작업 통계
        self.completed_batch: int = 0
        self.all_of_completed_order: int = 0

        # 작업 우선순위 관련
        self.working_area: List[List] = []
        self.big_asile: List[List] = []
        self.small_asile: List[List] = []

    def initialize_task_system(self, n_agents: int) -> None:
        """작업 시스템 초기화"""
        self.current_loaded_node_list = [[] for _ in range(n_agents)]

    def update_request_queue(self, shelfs: List, input_order: List) -> None:
        """요청 큐 업데이트"""
        self.request_queue.clear()

        # 새로운 주문 할당 로직
        if len(input_order) > 0 and self.next_order_cnt > 0:
            # 여기서는 간단히 구현 - 실제 로직은 기존 코드에서 가져와야 함
            pass

    def select_coworker(self, cur_agent, env) -> Optional[int]:
        """
        협업할 파트너 선택

        Args:
            cur_agent: 현재 에이전트
            env: 환경 객체

        Returns:
            Optional[int]: 선택된 파트너 ID
        """
        from rware.core.config import HumanZoneStrategy
        from rware.utils.Make_Maze import Make_Maze

        select_list = []
        cur_map = Make_Maze(env, mode=1)

        working_area = [[] for _ in range(env.n_max_humans)]

        if self.config.human_zone_strategy == HumanZoneStrategy.BIG_ASILE:
            working_area = self.big_asile
        elif self.config.human_zone_strategy == HumanZoneStrategy.SMALL_ASILE:
            working_area = self.small_asile

        working_area_cnt = len(working_area)
        cur_agent_idx = env.human_id_list.index(cur_agent.id)

        # Human want Robot
        if cur_agent.agent_type == True:
            for id in env.robot_id_list:
                tar_robot = env.agents[id - 1]
                if tar_robot.state == State.ROBOT_PICKING and tar_robot.coworker is None:
                    if env.routing_node_dict[(tar_robot.x, tar_robot.y)] in working_area[cur_agent_idx]:
                        cur_map[tar_robot.y][tar_robot.x] = 0
                        cur_map[cur_agent.y][cur_agent.x] = 0

                        delta_x = abs(tar_robot.x - cur_agent.x)
                        delta_y = abs(tar_robot.y - cur_agent.y)
                        if delta_y > 1:
                            delta_y = delta_y * 5
                        select_list.append([delta_x + delta_y, tar_robot.id])
            select_list.sort()
            if len(select_list) > 0:
                return select_list[0][1]
            else:
                return None

    def assign_tasks_to_agents(self, agents: List, input_order: List, env) -> None:
        """에이전트들에게 작업 할당"""
        for agent in agents:
            if len(input_order) > 0 and (agent.state == State.NOOP or agent.state == State.HOME):
                agent.state = State.ROBOT_MOVESPOT

                if agent.loadbox_station is None and hasattr(agent, "select_loadbox_station"):
                    agent.select_loadbox_station(env)

                if agent.load_box == False and agent.loadbox_station is not None:
                    agent.state = State.ROBOT_MOVEGOAL

    def check_task_completion(self, agents: List) -> bool:
        """작업 완료 상태 확인"""
        # 배치 완료 조건 확인
        # 기존 코드의 로직을 참고하여 구현
        return False

    def update_task_statistics(self, agents: List) -> None:
        """작업 통계 업데이트"""
        for agent in agents:
            if agent.state != State.NOOP and agent.state != State.HOME:
                agent.working_time += 1

    def get_task_statistics(self) -> Dict:
        """작업 관련 통계 반환"""
        return {
            'completed_batch': self.completed_batch,
            'all_of_completed_order': self.all_of_completed_order,
            'request_queue_size': len(self.request_queue),
            'next_order_cnt': self.next_order_cnt
        }

    def reset_task_system(self) -> None:
        """작업 시스템 리셋"""
        self.request_queue.clear()
        self.completed_batch = 0
        self.all_of_completed_order = 0
        self.next_order_cnt = 0

    def assign_initial_orders(self, agents: List, input_order: List) -> None:
        """초기 주문 할당"""
        if len(input_order) > 0:
            for agent in agents:
                if agent.agent_type == False and agent.load_box == False:
                    if len(input_order) > 0:
                        agent.load_box = True
                        agent.node_list = [j[0] for j in input_order[0]]
                        self.current_loaded_node_list[agent.id - 1] = input_order[0].copy()
                        agent.order_sku_cnt = [j[1] for j in input_order[0]]
                        del input_order[0]

                        agent.state = State.ROBOT_MOVESPOT
                        break
