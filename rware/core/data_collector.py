"""
Data Collector for Robotic Warehouse Simulation
데이터 수집 및 로깅을 담당하는 모듈
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from rware.core.config import SimulationConfig
from rware.entities import Agent


class DataCollector:
    """
    시뮬레이션 데이터를 수집하고 로깅하는 클래스

    이 클래스는 실시간 메트릭 수집, 로그 파일 생성,
    통계 분석 등의 기능을 담당합니다.
    """

    def __init__(self, config: SimulationConfig):
        """
        DataCollector 초기화

        Args:
            config (SimulationConfig): 시뮬레이션 설정
        """
        self.config = config

        # 데이터 저장소
        self.completed_batch_log: List[List[int]] = []
        self.total_map_cnt: Optional[List[List[int]]] = None

        # 메트릭
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # 파일 경로
        self.data_dir: str = "data"

    def initialize_data_collection(self, grid_size: Tuple[int, int]) -> None:
        """
        데이터 수집 초기화

        Args:
            grid_size: 그리드 크기
        """
        self.total_map_cnt = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        self.completed_batch_log = []
        self.start_time = time.time()

    def update_realtime_metrics(self, agents: List[Agent],
                              routing_node_dict: Dict,
                              internal_timer: int) -> None:
        """
        실시간 메트릭 업데이트

        Args:
            agents: 에이전트 리스트
            routing_node_dict: 라우팅 노드 딕셔너리
            internal_timer: 내부 타이머
        """
        # 맵 카운트 업데이트
        for agent in agents:
            if agent.agent_type == False:  # Robot
                if agent.x < len(self.total_map_cnt[0]) and agent.y < len(self.total_map_cnt):
                    self.total_map_cnt[agent.y][agent.x] += 1

    def collect_batch_completion(self, internal_timer: int, completed_batch: int) -> None:
        """
        배치 완료 데이터 수집

        Args:
            internal_timer: 내부 타이머
            completed_batch: 완료된 배치 수
        """
        self.completed_batch_log.append([internal_timer, completed_batch])

    def get_current_metrics(self, agents: List[Agent]) -> Dict:
        """
        현재 메트릭 계산

        Args:
            agents: 에이전트 리스트

        Returns:
            Dict: 현재 메트릭 데이터
        """
        metrics = {
            'total_sku_processed': sum(agent.total_sku_cnt for agent in agents),
            'total_distance_traveled': sum(agent.total_distance for agent in agents),
            'total_working_time': sum(agent.working_time for agent in agents),
            'total_timeout_events': sum(agent.timeout_cnt for agent in agents),
            'average_waiting_time': np.mean([agent.waiting_time for agent in agents]) if agents else 0,
            'active_agents': sum(1 for agent in agents if agent.state.value != 0),  # NOOP이 아닌 에이전트
        }

        return metrics

    def generate_simulation_report(self, agents: List[Agent],
                                internal_timer: int,
                                robot_cnt: int,
                                human_cnt: int) -> Dict:
        """
        시뮬레이션 완료 후 최종 리포트 생성

        Args:
            agents: 에이전트 리스트
            internal_timer: 총 실행 시간
            robot_cnt: 로봇 수
            human_cnt: 인간 수

        Returns:
            Dict: 최종 리포트 데이터
        """
        self.end_time = time.time()

        # 기본 메트릭 계산
        total_distance = sum(agent.total_distance for agent in agents)
        total_sku = sum(agent.total_sku_cnt for agent in agents)
        total_timeout = sum(agent.timeout_cnt for agent in agents)

        # 시간 기반 메트릭
        simulation_time_hours = internal_timer * self.config.tick_per_time / 3600

        # 생산성 메트릭
        productivity_per_hour = (total_sku / simulation_time_hours) if simulation_time_hours > 0 else 0
        box_per_hour_human = (self.completed_batch * self.config.loadbox_count / simulation_time_hours / human_cnt) if human_cnt > 0 and simulation_time_hours > 0 else 0

        report = {
            'simulation_duration': simulation_time_hours,
            'total_sku_processed': total_sku,
            'productivity_per_hour': productivity_per_hour,
            'box_per_hour_per_human': box_per_hour_human,
            'total_distance_traveled': total_distance,
            'total_timeout_events': total_timeout,
            'completed_batches': len(self.completed_batch_log),
            'average_distance_per_sku': (total_distance / total_sku) if total_sku > 0 else 0,
            'timeout_rate': (total_timeout / len(agents)) if agents else 0,
        }

        return report

    def save_agent_data(self, agents: List[Agent], time_str: str, path: str) -> None:
        """
        에이전트 데이터 저장

        Args:
            agents: 에이전트 리스트
            time_str: 타임스탬프 문자열
            path: 저장 경로
        """
        columns = ["Agent ID", "Type", "VisitedRackCnt", "WorkingTime", "WorkingTimeRatio",
                  "TimeoutCnt", "TotalDistance", "TotalSkuCnt", "Productivity", "SecPerPick", "WaitingTime"]
        data_set = []

        for agent in agents:
            ID_type = 'Human' if agent.agent_type else 'Robot'
            WorkingTimeRatio = agent.working_time / max(self.internal_timer, 1)
            Productivity = (agent.total_sku_cnt / max(self.internal_timer, 1)) * 60
            SecPerPick = self.internal_timer / max(agent.total_sku_cnt, 1)

            data = [
                f'{agent.id:>3d}', ID_type, f'{agent.complete_order:>5d}',
                f'{agent.working_time:>5d}', f'{WorkingTimeRatio:.5f}',
                f'{agent.timeout_cnt:>2d}', f'{agent.total_distance:>5d}',
                f'{agent.total_sku_cnt:>5d}', f'{Productivity:.5f}',
                f'{SecPerPick:.5f}', f'{agent.waiting_time:>5d}'
            ]
            data_set.append(data)

        df_agent = pd.DataFrame(data_set, columns=columns)
        df_agent.to_csv(f"{path}/{time_str}_agent.csv")

    def save_zone_data(self, total_robot_cnt_in_zone: List[int],
                      total_timeout_cnt_in_zone: List[int],
                      time_str: str, path: str) -> None:
        """
        존 데이터 저장

        Args:
            total_robot_cnt_in_zone: 존 별 로봇 수
            total_timeout_cnt_in_zone: 존 별 타임아웃 수
            time_str: 타임스탬프 문자열
            path: 저장 경로
        """
        columns = ["ZONE ID", "ZONE_CNT", "ZONE_AVERAGECNT", "ZONE_TIMEOUTCNT"]
        data_set = []

        for i in range(len(total_robot_cnt_in_zone)):
            value = total_robot_cnt_in_zone[i]
            avg = value / max(self.internal_timer, 1)
            timeoutcnt = total_timeout_cnt_in_zone[i]
            data = [f"{i:>3d}", f"{value:>5d}", f"{avg:.5f}", f"{timeoutcnt:>3d}"]
            data_set.append(data)

        df_zone = pd.DataFrame(data_set, columns=columns)
        df_zone.to_csv(f"{path}/{time_str}_zone.csv")

    def save_batch_data(self, time_str: str, path: str) -> None:
        """
        배치 데이터 저장

        Args:
            time_str: 타임스탬프 문자열
            path: 저장 경로
        """
        columns = ["Batch Time", "Batch Value"]
        data_set = []

        for log in self.completed_batch_log:
            data = ["{0:09d}".format(log[0]), "{0:02d}".format(log[1])]
            data_set.append(data)

        df_batch = pd.DataFrame(data_set, columns=columns)
        df_batch.to_csv(f"{path}/{time_str}_batch_information.csv")

    def save_heatmap_data(self, edge_map: List[List],
                         total_robot_cnt_in_zone: List[int],
                         total_timeout_cnt_in_zone: List[int],
                         time_str: str, path: str) -> None:
        """
        히트맵 데이터 저장

        Args:
            edge_map: 엣지 맵
            total_robot_cnt_in_zone: 존 별 로봇 수
            total_timeout_cnt_in_zone: 존 별 타임아웃 수
            time_str: 타임스탬프 문자열
            path: 저장 경로
        """
        col_length = len(edge_map[0]) if edge_map else 0
        row_length = len(edge_map) if edge_map else 0

        # 로봇 수 히트맵
        data = [[0 for _ in range(col_length)] for _ in range(row_length)]
        for i in range(row_length):
            for j in range(col_length):
                if edge_map[i][j] >= 0:
                    value = total_robot_cnt_in_zone[edge_map[i][j]]
                    avg = value / max(self.internal_timer, 1)
                    data[i][j] = avg

        columns = ['X' + str(x+1).zfill(1) for x in range(col_length)]
        df = pd.DataFrame(data, columns=columns)
        df.index = ['Y' + str(y+1).zfill(1) for y in range(row_length)]

        plt.figure(figsize=(10, 8))
        plt.imshow(df, cmap='gray', interpolation='none')
        plt.title("Total Robot Count In Zone")
        plt.colorbar()
        plt.savefig(f"{path}/{time_str}_totalRobotCntInZone.png")
        plt.close()

        # 타임아웃 수 히트맵
        data = [[0 for _ in range(col_length)] for _ in range(row_length)]
        for i in range(row_length):
            for j in range(col_length):
                if edge_map[i][j] >= 0:
                    value = total_timeout_cnt_in_zone[edge_map[i][j]]
                    data[i][j] = value

        df2 = pd.DataFrame(data, columns=columns)
        df2.index = ['Y' + str(y+1).zfill(1) for y in range(row_length)]

        plt.figure(figsize=(10, 8))
        plt.imshow(df2, cmap='gray', interpolation='none')
        plt.title("Total Timeout Count In Zone")
        plt.colorbar()
        plt.savefig(f"{path}/{time_str}_totalTimeoutCntInZone.png")
        plt.close()

    def save_map_data(self, time_str: str, path: str) -> None:
        """
        맵 데이터 저장

        Args:
            time_str: 타임스탬프 문자열
            path: 저장 경로
        """
        if self.total_map_cnt is None:
            return

        # 로봇 위치 히트맵 (역비율)
        max_val = max(max(row) for row in self.total_map_cnt) if self.total_map_cnt else 0
        data = [[0 for _ in range(len(self.total_map_cnt[0]))] for _ in range(len(self.total_map_cnt))]

        for i in range(len(self.total_map_cnt)):
            for j in range(len(self.total_map_cnt[0])):
                if max_val <= 0:
                    max_val = 1
                value = ((max_val - self.total_map_cnt[i][j]) / max_val) * 100
                data[i][j] = int(value)

        columns = ['X' + str(x + 1).zfill(1) for x in range(len(self.total_map_cnt[0]))]
        df3 = pd.DataFrame(data, columns=columns)
        df3.index = ['Y' + str(y + 1).zfill(1) for y in range(len(self.total_map_cnt))]

        plt.figure(figsize=(10, 8))
        plt.imshow(df3, cmap='gray', interpolation='none')
        plt.title("Robot Positions (Reverse Ratio)")
        plt.colorbar()
        plt.savefig(f"{path}/{time_str}_RobotPositions(Reverse_Ratio).png")
        plt.close()

        # 로봇 위치 히트맵 (역절대값)
        abs_val = 5000
        data = [[0 for _ in range(len(self.total_map_cnt[0]))] for _ in range(len(self.total_map_cnt))]

        for i in range(len(self.total_map_cnt)):
            for j in range(len(self.total_map_cnt[0])):
                value = abs_val - self.total_map_cnt[i][j]
                if abs_val - self.total_map_cnt[i][j] < 0:
                    value = 0
                data[i][j] = value

        df4 = pd.DataFrame(data, columns=columns)
        df4.index = ['Y' + str(y + 1).zfill(1) for y in range(len(self.total_map_cnt))]

        plt.figure(figsize=(10, 8))
        plt.imshow(df4, cmap='gray', interpolation='none')
        plt.title("Robot Positions (Reverse Absolute)")
        plt.colorbar()
        plt.savefig(f"{path}/{time_str}_RobotPositions(Reverse_Abs).png")
        plt.close()

    def save_final_summary(self, report: Dict, time_str: str, path: str,
                          start_time: str, end_time: str, human_cnt: int, robot_cnt: int) -> None:
        """
        최종 요약 리포트 저장

        Args:
            report: 리포트 데이터
            time_str: 타임스탬프 문자열
            path: 저장 경로
            start_time: 시작 시간
            end_time: 종료 시간
            human_cnt: 인간 수
            robot_cnt: 로봇 수
        """
        f = open(f"{path}/{time_str}_result_summary.csv", 'w')

        # 시간 정보
        cur_hour = int(report['simulation_duration'] * 3600) // 3600
        cur_min = (int(report['simulation_duration'] * 3600) % 3600) // 60
        cur_sec = ((int(report['simulation_duration'] * 3600) % 3600) % 60)
        cur_time_str = f"Time : {cur_hour:02d} : {cur_min:02d} : {cur_sec:02d}"

        # 기본 정보
        f.write(f"Actual Start Time : {start_time}\n")
        f.write(f"Actual End Time : {end_time}\n")
        f.write(f"Simulation Time : {cur_time_str}\n")
        f.write(f"Total SKU Processed : {report['total_sku_processed']:,}\n")
        f.write(f"Productivity (SKU/hour) : {report['productivity_per_hour']:.2f}\n")
        f.write(f"Box/Hour/Human : {report['box_per_hour_per_human']:.2f}\n")
        f.write(f"Total Distance : {report['total_distance_traveled']:,}\n")
        f.write(f"Timeout Events : {report['total_timeout_events']}\n")
        f.write(f"Completed Batches : {report['completed_batches']}\n")

        f.close()
