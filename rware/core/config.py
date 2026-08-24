"""
Configuration management for Robotic Warehouse Simulation
시뮬레이션 설정을 중앙화하여 관리하는 모듈
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from rware.config import (
    RENDERING_FLAG,
    HUMAN_ZONE_FLAG,
    ROBOT_CAPA,
    ORDER_BATCH_FLAG,
    ORDER_BATCH_SEQ_FLAG,
    ORDER_SEQ_FLAG,
    SELECT_PICKING_COLLISION,
    SELECT_HUMAN_MOVE,
    WALL_ENFORCE_LEVEL,
    DEBUG_PRINTS,
    VERBOSE_ZONE,
    AUCTION_DEBUG,
    AUCTION_DEBUG_PATH,
    AUCTION_REAUCTION_ENABLED,
    AUCTION_TAU_LOCK,
    AUCTION_DELTA_GAIN,
    AUCTION_MAX_REASSIGN,
    AUCTION_ZONE_PENALTY,
    PREDICTIVE_DISPATCH,
    DISPATCH_LEAD_LIMIT,
    RENDEZVOUS_ROBOT_WAIT_WEIGHT,
    RENDEZVOUS_HUMAN_WAIT_WEIGHT,
    RENDEZVOUS_HUMAN_TRAVEL_WEIGHT,
    RENDEZVOUS_RISK_WEIGHT,
    RISK_WARMUP_TASKS,
    RISK_RETRAIN_INTERVAL,
    RISK_CATBOOST_ITERATIONS,
    STAGING_EARLY_WEIGHT,
    STAGING_ETA_BACKEND,
    STAGING_POLICY,
    STAGING_UNCERTAINTY_WEIGHT,
    SERVICE_TIME_VARIABILITY,
    SERVICE_TIME_SEED,
    SKU_PER_PICKING_TIME,
    BOX_LOADING_TIME,
    resolve_human_assignment_strategy,
)


def _default_assignment_strategy() -> str:
    try:
        strategy = resolve_human_assignment_strategy()
    except Exception:
        return "nearest_idle"
    return strategy or "nearest_idle"


class HumanZoneStrategy(Enum):
    """인간 존 할당 전략"""
    ALL = 0
    BIG_ASILE = 1
    SMALL_ASILE = 2


class OrderBatchStrategy(Enum):
    """주문 배치 전략"""
    RANDOM = 0
    SITE_A_BATCH = 1
    GA_OPTIMIZED = 2


class BatchSequenceStrategy(Enum):
    """배치 시퀀싱 전략"""
    OFF = 0
    SHORT_FIRST = 1
    LONG_FIRST = 2
    RANDOM = 3


class AuctionDistanceObstacles(Enum):
    """Auction 거리(ETA proxy) 계산 시 사용할 장애물 모델."""

    # 사람은 로봇을 통과 가능하다는 전제: 기본은 선반만 장애물로 취급.
    SHELF_ONLY = "shelf_only"
    # Human planner의 장애물 정의(spot/시설 포함)를 그대로 사용.
    HUMAN_MAZE = "human_maze"


@dataclass
class SimulationConfig:
    """
    Robotic Warehouse 시뮬레이션 설정 클래스

    기존의 전역 변수들을 구조화하여 관리하며,
    시뮬레이션의 모든 설정을 중앙에서 제어합니다.
    """

    # 렌더링 설정
    rendering: bool = True

    # 디버그/진단 설정 (stdout/파일 로그 등)
    debug_prints: bool = False
    verbose_zone: bool = False
    auction_debug: bool = False
    auction_debug_path: Optional[str] = None
    auction_distance_obstacles: AuctionDistanceObstacles = AuctionDistanceObstacles.SHELF_ONLY

    # Auction Re-auction 설정 (진동 억제 메커니즘)
    # 참고: docs/auction-algorithm/auction_human_agv_mapping_methodology.md 섹션 6
    auction_reauction_enabled: bool = True  # Re-auction 활성화 여부
    auction_tau_lock: int = 3  # 도착 임박 락: ETA ≤ τ_lock이면 재경매 제외
    auction_delta_gain: float = 5.0  # 최소 이득 임계값: (old_cost - new_cost) > Δ
    auction_max_reassign: int = 2  # 요청당 최대 재할당 횟수
    auction_zone_penalty: float = 0.0  # 존 외 로봇 패널티 (0이면 존 제약 없음)

    # Predictive rendezvous dispatch
    # 기본 배정은 로봇이 랙에 주차(ROBOT_PICKING)한 뒤에야 시작되므로 작업자의
    # 이동시간이 전부 로봇 대기로 계상된다. 이 값을 켜면 이동 중(ROBOT_MOVESPOT)인
    # 로봇도 후보에 포함되어 작업자를 미리 파견할 수 있다.
    predictive_dispatch: bool = False
    # 예측 도착시각 기준 최대 선행 파견 틱. 로봇이 이보다 더 늦게 도착할 것으로
    # 예측되면 파견을 미뤄 작업자가 랙에서 노는 시간을 제한한다.
    dispatch_lead_limit: int = 60
    rendezvous_robot_wait_weight: float = 1.0  # 로봇이 기다리는 틱의 비용
    rendezvous_human_wait_weight: float = 1.0  # 작업자가 랙에서 노는 틱의 비용
    rendezvous_human_travel_weight: float = 0.0  # 작업자 이동 틱의 비용
    rendezvous_risk_weight: float = 0.5  # 지각 위험(Q90 기반) 가중치

    # 서비스타임/도착시각 위험 학습
    risk_warmup_tasks: int = 500
    risk_retrain_interval: int = 500
    risk_catboost_iterations: int = 300

    # 유휴 작업자 사전 배치: off | nearest | learned | oracle
    staging_policy: str = "off"
    staging_eta_backend: str = "catboost"
    # learned 점수식의 항별 가중치. 0으로 두면 그 항을 끈다.
    staging_early_weight: float = 0.5
    staging_uncertainty_weight: float = 0.5

    # 서비스타임 변동성 주입 (robustness 시나리오). off면 기존 결정론적 동작 유지.
    service_time_variability: str = "off"  # off | low | medium | high
    service_time_seed: int = 0

    # 인간 에이전트 설정
    human_zone_strategy: HumanZoneStrategy = HumanZoneStrategy.SMALL_ASILE
    human_move_strategy: bool = False  # 0: 이동 후 작업, 1: 즉시 작업
    human_assignment_strategy: str = field(default_factory=_default_assignment_strategy)

    # 로봇 설정
    robot_capacity: int = 8
    robot_max_capacity: int = 10000000000000

    # 주문 및 배치 설정
    order_batch_strategy: OrderBatchStrategy = OrderBatchStrategy.RANDOM
    batch_sequence_strategy: BatchSequenceStrategy = BatchSequenceStrategy.LONG_FIRST
    order_sequence_flag: int = 0

    # 시간 설정
    tick_per_time: float = 1.0
    timeout_value: int = 5
    timeout_value_start: int = 10
    timeout_value_end: int = 12

    # 생산성 설정
    box_loading_time: int = 149
    sku_per_picking_time: int = 7
    sku_per_exit_time: int = 1

    # 충돌 및 라우팅 설정
    picking_collision_allowed: bool = False
    routing_strategy: int = 0  # 0: 일반, 1: 포크
    wall_enforce_level: int = 2  # 2=map overlay, 1=no overlay constraints, 0=off

    # 그리드 및 레이어 설정
    collision_layers: int = 4
    layer_agents: int = 0
    layer_shelfs: int = 1
    layer_spots: int = 2
    layer_human: int = 3

    # 거리 및 용량 설정
    distance_per_grid: float = 1.5
    loadbox_count: int = 8  # robot_capacity와 동일

    # 맵 생성 설정
    shelf_vertical_idx: bool = True
    mapping_horizontal: bool = False

    # 최적화 파라미터
    productivity_factor: int = 15
    sequence_param: int = 4
    max_zone_count: int = 4

    # 액션/관측 스키마 설정
    action_schema: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "multi_discrete",
            "message_bits": 0,
        }
    )
    observation_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """설정 값 검증 및 자동 계산"""
        if self.loadbox_count != self.robot_capacity:
            self.loadbox_count = self.robot_capacity

    @classmethod
    def from_legacy_config(cls) -> 'SimulationConfig':
        """
        기존 config.py의 설정값들을 불러와 SimulationConfig로 변환
        레거시 코드와의 호환성을 위해 유지
        """
        try:
            order_batch_strategy = OrderBatchStrategy(ORDER_BATCH_FLAG)
        except Exception:
            order_batch_strategy = OrderBatchStrategy.RANDOM

        return cls(
            rendering=bool(RENDERING_FLAG),
            debug_prints=bool(DEBUG_PRINTS),
            verbose_zone=bool(VERBOSE_ZONE),
            auction_debug=bool(AUCTION_DEBUG),
            auction_debug_path=(str(AUCTION_DEBUG_PATH).strip() or None),
            auction_distance_obstacles=AuctionDistanceObstacles(
                str(
                    __import__("os").environ.get(
                        "RWARE_AUCTION_DISTANCE_OBSTACLES",
                        AuctionDistanceObstacles.SHELF_ONLY.value,
                    )
                ).strip()
                or AuctionDistanceObstacles.SHELF_ONLY.value
            ),
            human_zone_strategy=HumanZoneStrategy(HUMAN_ZONE_FLAG),
            robot_capacity=ROBOT_CAPA,
            order_batch_strategy=order_batch_strategy,
            batch_sequence_strategy=BatchSequenceStrategy(ORDER_BATCH_SEQ_FLAG),
            order_sequence_flag=int(ORDER_SEQ_FLAG),
            picking_collision_allowed=bool(SELECT_PICKING_COLLISION),
            wall_enforce_level=int(WALL_ENFORCE_LEVEL),
            human_move_strategy=bool(SELECT_HUMAN_MOVE),
            human_assignment_strategy=_default_assignment_strategy(),
            auction_reauction_enabled=bool(AUCTION_REAUCTION_ENABLED),
            auction_tau_lock=int(AUCTION_TAU_LOCK),
            auction_delta_gain=float(AUCTION_DELTA_GAIN),
            auction_max_reassign=int(AUCTION_MAX_REASSIGN),
            auction_zone_penalty=float(AUCTION_ZONE_PENALTY),
            predictive_dispatch=bool(PREDICTIVE_DISPATCH),
            dispatch_lead_limit=int(DISPATCH_LEAD_LIMIT),
            rendezvous_robot_wait_weight=float(RENDEZVOUS_ROBOT_WAIT_WEIGHT),
            rendezvous_human_wait_weight=float(RENDEZVOUS_HUMAN_WAIT_WEIGHT),
            rendezvous_human_travel_weight=float(RENDEZVOUS_HUMAN_TRAVEL_WEIGHT),
            rendezvous_risk_weight=float(RENDEZVOUS_RISK_WEIGHT),
            risk_warmup_tasks=int(RISK_WARMUP_TASKS),
            risk_retrain_interval=int(RISK_RETRAIN_INTERVAL),
            risk_catboost_iterations=int(RISK_CATBOOST_ITERATIONS),
            staging_policy=str(STAGING_POLICY),
            staging_eta_backend=str(STAGING_ETA_BACKEND),
            staging_early_weight=float(STAGING_EARLY_WEIGHT),
            staging_uncertainty_weight=float(STAGING_UNCERTAINTY_WEIGHT),
            service_time_variability=str(SERVICE_TIME_VARIABILITY),
            service_time_seed=int(SERVICE_TIME_SEED),
            sku_per_picking_time=int(SKU_PER_PICKING_TIME),
            box_loading_time=int(BOX_LOADING_TIME),
            action_schema={
                "type": "multi_discrete",
                "message_bits": 0,
            },
        )

    def to_dict(self) -> dict:
        """설정값들을 딕셔너리로 변환"""
        return {
            'rendering': self.rendering,
            'human_zone_strategy': self.human_zone_strategy.value,
            'robot_capacity': self.robot_capacity,
            'batch_sequence_strategy': self.batch_sequence_strategy.value,
            'picking_collision_allowed': self.picking_collision_allowed,
            'timeout_value': self.timeout_value,
            'box_loading_time': self.box_loading_time,
            'sku_per_picking_time': self.sku_per_picking_time,
            'human_assignment_strategy': self.human_assignment_strategy,
            'debug_prints': self.debug_prints,
            'verbose_zone': self.verbose_zone,
            'auction_debug': self.auction_debug,
            'auction_debug_path': self.auction_debug_path,
            'auction_distance_obstacles': self.auction_distance_obstacles.value,
            'auction_reauction_enabled': self.auction_reauction_enabled,
            'auction_tau_lock': self.auction_tau_lock,
            'auction_delta_gain': self.auction_delta_gain,
            'auction_max_reassign': self.auction_max_reassign,
            'auction_zone_penalty': self.auction_zone_penalty,
        }
