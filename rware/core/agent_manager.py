"""
Agent Manager for Robotic Warehouse Simulation
에이전트 생명주기 및 상태 관리를 담당하는 모듈
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from rware.core.state import Direction, State
from rware.core.config import SimulationConfig
from rware.entities import Agent


class AgentManager:
    """
    에이전트들을 관리하고 상태를 추적하는 클래스

    이 클래스는 모든 에이전트의 생성, 초기화, 상태 관리,
    통계 수집 등의 기능을 담당합니다.
    """

    def __init__(self, config: SimulationConfig):
        """
        AgentManager 초기화

        Args:
            config (SimulationConfig): 시뮬레이션 설정
        """
        self.config = config
        self.environment = None  # EnvironmentCore 참조

        # 에이전트 관리
        self.agents: List[Agent] = []
        self.human_id_list: List[int] = []
        self.robot_id_list: List[int] = []
        self.agent_id_list: List[int] = []

        # 에이전트 카운터
        self.deleted_human_id_list: List[int] = []
        self.deleted_robot_id_list: List[int] = []

        # 에이전트 설정
        self.n_agents: int = 0
        self.n_humans: int = 0
        self.n_robots: int = 0
        self.n_max_agents: int = 0
        self.n_max_humans: int = 0
        self.n_max_robots: int = 0

        # 통계 데이터
        self.running_robot_cnt: int = 0
        self.running_human_cnt: int = 0
        self.total_robot_cnt_in_zone: List[int] = []
        self.total_timeout_cnt_in_zone: List[int] = []
        self.completed_batch_log: List[List[int]] = []

    def initialize_agents(self, human_init_queue: List[Tuple[int, int]],
                         robot_init_queue: List[Tuple[int, int]],
                         msg_bits: int, grid_size: Tuple[int, int]) -> None:
        """
        에이전트 초기화

        Args:
            human_init_queue: 인간 초기 위치 큐
            robot_init_queue: 로봇 초기 위치 큐
            msg_bits: 메시지 비트 수
        """
        # ID 리스트 설정
        self.human_id_list = [x for x in range(1, len(human_init_queue) + 1)]
        self.robot_id_list = [x + len(human_init_queue) for x in range(1, len(robot_init_queue) + 1)]
        self.agent_id_list = self.human_id_list + self.robot_id_list

        # 최대 에이전트 수 설정
        self.n_max_humans = len(self.human_id_list)
        self.n_max_robots = len(self.robot_id_list)
        self.n_max_agents = self.n_max_humans + self.n_max_robots

        # 실제 에이전트 수 (처음에는 최대값과 동일)
        self.n_humans = self.n_max_humans
        self.n_robots = self.n_max_robots
        self.n_agents = self.n_max_agents

        # 에이전트 객체 생성
        agent_locs = []
        for human_sample in human_init_queue:
            agent_locs.append(human_sample[0] + grid_size[1] * human_sample[1])
        for robot_sample in robot_init_queue:
            agent_locs.append(robot_sample[0] + grid_size[1] * robot_sample[1])

        agent_locs = np.unravel_index(agent_locs, grid_size)

        # 방향 설정 (모두 LEFT로 초기화)
        agent_dirs = [Direction.LEFT] * self.n_agents

        # 에이전트 생성
        self.agents = [
            Agent(x, y, dir_, msg_bits, self.config)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]

        # 에이전트 타입 설정
        for idx in range(self.n_humans):
            self.agents[idx].agent_type = True  # Human
        for idx in range(self.n_humans, self.n_agents):
            self.agents[idx].agent_type = False  # Robot

    def reset_agents(self) -> None:
        """에이전트 상태 초기화"""
        from rware.entities import Agent
        Agent.counter = 0

        for agent in self.agents:
            agent.state = State.NOOP

    def update_agent_statistics(self, routing_node_dict: Dict) -> None:
        """에이전트 통계 업데이트"""
        self.running_human_cnt = 0
        self.running_robot_cnt = 0

        # 인간 에이전트 통계
        for idx in range(self.n_humans):
            if idx + 1 not in self.agent_id_list:
                continue
            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME:
                continue
            else:
                self.running_human_cnt += 1
                self.agents[idx].working_time += 1

        # 로봇 에이전트 통계
        for idx in range(self.n_humans, self.n_agents):
            if idx + 1 not in self.agent_id_list:
                continue

            # 존 카운트
            zone_idx = routing_node_dict[(self.agents[idx].x, self.agents[idx].y)]
            self.total_robot_cnt_in_zone[zone_idx] += 1

            if self.agents[idx].state == State.NOOP or self.agents[idx].state == State.HOME:
                continue
            else:
                self.running_robot_cnt += 1
                self.agents[idx].working_time += 1

    def get_agent_actions(self, env, human_map: np.ndarray, robot_map: np.ndarray) -> List[np.ndarray]:
        """
        모든 에이전트의 액션 결정

        Args:
            human_map: 인간용 맵
            robot_map: 로봇용 맵

        Returns:
            List[np.ndarray]: 각 에이전트의 액션 리스트
        """
        actions = []
        for id in self.agent_id_list:
            if id * 2 > self.internal_timer + 120:
                actions.append(np.array([Action.NOOP.value, 0], dtype='int64'))
                continue

            agent = self.agents[id - 1]
            if hasattr(agent, "next_action"):
                actions.append(agent.next_action(env, human_map, robot_map))
            else:
                actions.append(np.array([Action.NOOP.value, 0], dtype="int64"))
        return actions

    def update_agent_states(self, env, input_order: List) -> None:
        """모든 에이전트 상태 업데이트"""
        for id in self.agent_id_list:
            agent = self.agents[id - 1]
            if hasattr(agent, "check_status"):
                agent.check_status(env, input_order)

    def get_active_agent_count(self) -> int:
        """활성 에이전트 수 반환"""
        count = 0
        for id in self.agent_id_list:
            if self.agents[id-1].state != State.NOOP:
                count += 1
        return count

    def get_agent_statistics(self) -> Dict:
        """에이전트 관련 통계 반환"""
        total_distance = sum(agent.total_distance for agent in self.agents)
        total_timeout = sum(agent.timeout_cnt for agent in self.agents)
        total_sku_processed = sum(agent.total_sku_cnt for agent in self.agents)
        total_working_time = sum(agent.working_time for agent in self.agents)

        return {
            'n_active_agents': self.get_active_agent_count(),
            'running_human_cnt': self.running_human_cnt,
            'running_robot_cnt': self.running_robot_cnt,
            'total_distance': total_distance,
            'total_timeout': total_timeout,
            'total_sku_processed': total_sku_processed,
            'total_working_time': total_working_time,
            'total_robot_cnt_in_zone': self.total_robot_cnt_in_zone,
            'total_timeout_cnt_in_zone': self.total_timeout_cnt_in_zone
        }

    def add_human(self, count: int, env) -> None:
        """인간 에이전트 추가"""
        for _ in range(count):
            if len(self.deleted_human_id_list) > 0:
                self.human_id_list.append(self.deleted_human_id_list[0])
                self.agent_id_list.append(self.deleted_human_id_list[0])
                del self.deleted_human_id_list[0]
            else:
                last_human_id = 0
                if len(self.human_id_list) > 0:
                    last_human_id = self.human_id_list[-1]
                new_human_id = last_human_id + 1
                if new_human_id > len(env.human_init_queue):
                    break
                else:
                    self.human_id_list.append(new_human_id)
                    self.agent_id_list.append(new_human_id)

        self.human_id_list = sorted(self.human_id_list)
        self.agent_id_list = sorted(self.agent_id_list)

    def add_robot(self, count: int, env) -> None:
        """로봇 에이전트 추가"""
        for _ in range(count):
            if len(self.deleted_robot_id_list) > 0:
                self.robot_id_list.append(self.deleted_robot_id_list[0])
                self.agent_id_list.append(self.deleted_robot_id_list[0])
                del self.deleted_robot_id_list[0]
            else:
                last_robot_id = len(env.human_init_queue)
                if len(self.robot_id_list) > 0:
                    last_robot_id = self.robot_id_list[-1]
                new_robot_id = last_robot_id + 1
                if new_robot_id > (len(env.human_init_queue) + len(env.robot_init_queue)):
                    break
                else:
                    self.robot_id_list.append(new_robot_id)
                    self.agent_id_list.append(new_robot_id)

        self.robot_id_list = sorted(self.robot_id_list)
        self.agent_id_list = sorted(self.agent_id_list)

    def remove_human(self, count: int, env) -> None:
        """인간 에이전트 제거"""
        for _ in range(count):
            if len(self.human_id_list) <= 0:
                break
            else:
                # Picking 상태가 아닌 인간부터 제거
                for idx in reversed(range(len(self.human_id_list))):
                    id = self.human_id_list[idx]
                    state = self.agents[id-1].state

                    if state != State.HUMAN_PICKING and state != State.HUMAN_DONE:
                        if self.agents[id-1].coworker is not None:
                            env.agents[self.agents[id-1].coworker - 1].coworker = None

                        self.agents[id - 1].node_list = list()
                        self.agents[id - 1].state = State.NOOP
                        self.agents[id - 1].coworker = None
                        self.agents[id - 1].x = self.agents[id - 1].init_x
                        self.agents[id - 1].y = self.agents[id - 1].init_y

                        self.deleted_human_id_list.append(id)
                        del self.human_id_list[idx]
                        self.agent_id_list = self.human_id_list + self.robot_id_list
                        break

    def remove_robot(self, count: int, env) -> None:
        """로봇 에이전트 제거"""
        for _ in range(count):
            if len(self.robot_id_list) <= 0:
                break
            else:
                # LOAD나 NOOP 상태인 로봇부터 제거
                for idx in reversed(range(len(self.robot_id_list))):
                    id = self.robot_id_list[idx]
                    state = self.agents[id-1].state

                    if state == State.ROBOT_LOAD or state == State.NOOP or state == State.HOME:
                        if self.agents[id - 1].coworker is not None:
                            env.agents[self.agents[id - 1].coworker - 1].coworker = None
                            env.agents[self.agents[id - 1].coworker - 1].node_list = list()
                            env.agents[self.agents[id - 1].coworker - 1].state = State.NOOP

                        self.agents[id - 1].node_list = list()
                        self.agents[id - 1].state = State.NOOP
                        self.agents[id - 1].coworker = None
                        self.agents[id - 1].x = self.agents[id - 1].init_x
                        self.agents[id - 1].y = self.agents[id - 1].init_y

                        self.deleted_robot_id_list.append(id)
                        del self.robot_id_list[idx]
                        self.agent_id_list = self.human_id_list + self.robot_id_list
                        break
