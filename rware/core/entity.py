"""
Base Entity class for Robotic Warehouse Simulation
모든 엔티티(에이전트, 선반)의 기본 클래스를 정의하는 모듈
"""

from typing import Tuple


class Entity:
    """
    모든 엔티티의 기본 클래스

    Attributes:
        id (int): 엔티티 고유 ID
        x (int): 현재 X 좌표
        y (int): 현재 Y 좌표
        prev_x (int): 이전 X 좌표
        prev_y (int): 이전 Y 좌표
    """

    def __init__(self, id_: int, x: int, y: int):
        """
        Entity 초기화

        Args:
            id_ (int): 엔티티 고유 ID
            x (int): 초기 X 좌표
            y (int): 초기 Y 좌표
        """
        self.id = id_
        self.prev_x = x
        self.prev_y = y
        self.x = x
        self.y = y
