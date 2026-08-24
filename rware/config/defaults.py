# 필요한 매크로 변수 선언
import os

AXIS_Z = 0
AXIS_Y = 1
AXIS_X = 2
COLLISION_LAYERS = 4

LAYER_AGENTS = 0
LAYER_SHELFS = 1
LAYER_SPOTS  = 2
LAYER_HUMAN  = 3

FIRST_PICKING_STATION = 0
SECOND_PICKING_STATION = 1

SHELF_VERTICAL_IDX = True
MAPPING_HORIZONTAL = False
PICKING_CAPACITIY = 1
DROP_CAPACITY = 1
ROBOT_MAX_CAPACITY = 10000000000000

DISTANCE_PER_GRID = 1.5
TICKPERTIME = 1.0

TIMEOUT_VALUE = 5
TIMEOUT_VALUE_START = 10
TIMEOUT_VALUE_END   = 12


PRODUCTIVITY_FACTOR = 15
SEQUENCE_PARAM      = 4
MAX_ZONE_CNT        = 4

# Admin Controller

# 0: Rendering Off, 1: Rendering On (env override: RWARE_RENDERING)
RENDERING_FLAG = int(os.environ.get("RWARE_RENDERING", "1"))

# 2=new map-overlay walls/directions (planner+engine)
# 1=no overlay constraints (legacy entities.py hardcoded virtual walls removed)
# 0=no traffic constraints
WALL_ENFORCE_LEVEL = int(os.environ.get("RWARE_WALL_ENFORCE", "2"))

# Path to the active map file (used by rware.apps.simulator)
MAP_FILE = os.environ.get(
    "RWARE_MAP",
    os.path.join(os.path.dirname(__file__), "..", "data", "maps", "warehouse_main.map"),
)

# 0: Random, 1: Batch  2: GA (if use this strategy, change ROBOT_CAPA to 6)
ORDER_BATCH_FLAG = 1

# --- simulation termination / logging --------------------------------------
# 처리 완료 SKU(=물건) 개수가 이 값에 도달하면 시뮬레이션을 종료하고 통계를 저장합니다.
# - 전체(예: 28100) 처리: 28100
# - 빠른 테스트: 1000
TARGET_TOTAL_SKU = int(os.environ.get("RWARE_TARGET_TOTAL_SKU", "28099"))

# 주문 데이터에서 사용할 날짜(일). 데이터셋은 1~31일을 담고 있으며 하루치가 한 번의
# 실험 워크로드가 된다. 일반화 실험(다른 날짜)을 위해 env로 노출한다.
ORDER_DATE = int(os.environ.get("RWARE_ORDER_DATE", "27"))

# 0: Batch Sequence off, 1: First shortest Last longest, 2:First longest Last shortest, 3: Random
ORDER_BATCH_SEQ_FLAG = 3

# 0: (IN) Batch Sequence Original(static), 1: Shortest_seq(need table)
ORDER_INBATCH_SEQ_FLAG = 1

# 0: All, 1: Big, 2: Small
HUMAN_ZONE_FLAG   = 2

# 0: Noseq, 1: Order Seq, 2: Batch Seq(Not Implement) 현재 업데이트 되었음 사용 x 0 고정
ORDER_SEQ_FLAG    = 0

# 0: NoRoute, 1: Fork Route
CHK_ROUTING_FLAG  = 0

# 0: Human Move, 1: Human Do not Move
SELECT_HUMAN_MOVE  = 0

# 0: Collision not Allow, 1: Collision Allow
SELECT_PICKING_COLLISION = 0

ROBOT_CAPA = 6  #for GA robot capa / 4
LOADBOX_CNT = ROBOT_CAPA
BOX_PER_LOADING_TIME = 1
# 로봇이 적재 스테이션에 머무는 tick
BOX_LOADING_TIME = 149

# SKU 하나를 피킹하는 데 걸리는 tick
SKU_PER_PICKING_TIME = 7
SKU_PER_EXIT_TIME = 1
TEST_PICKING_TIME = 1

dx = [0,1,1,1,0,-1,-1,-1]
dy = [-1,-1,0,1,1,1,0,-1]

# Human assignment strategy defaults
DEFAULT_HUMAN_ASSIGNMENT_STRATEGY = [
    "nearest_idle",
    "nearest_robot_first",
    "first_robot_arrived",
    "shortest_service_robot",
    "legacy_batch",
    "auction",
]

# -----------------------------------------------------------------------------
# Debug / diagnostics (managed via SimulationConfig.from_legacy_config)
# -----------------------------------------------------------------------------
# 0: off, 1: on
DEBUG_PRINTS = 0

# Print verbose human-zone / "needed people" diagnostics during batching.
VERBOSE_ZONE = 0

# Auction strategy profiling log.
AUCTION_DEBUG = 1
# If empty, defaults to `/tmp/rware_auction_<pid>.log`.
# If set and you run multi-process, each process will get its own file suffix.
AUCTION_DEBUG_PATH = ""

# -----------------------------------------------------------------------------
# Auction Re-auction settings (진동 억제 메커니즘)
# 참고: docs/auction-algorithm/auction_human_agv_mapping_methodology.md 섹션 6
# -----------------------------------------------------------------------------
# 0: Re-auction 비활성화, 1: Re-auction 활성화
AUCTION_REAUCTION_ENABLED = 1

# τ_lock (도착 임박 락): 사람의 ETA가 이 값 이하이면 재경매 대상에서 제외
# 단위: tick (그리드 이동 수)
AUCTION_TAU_LOCK = 3

# Δ (최소 이득 임계값): (old_cost - new_cost) > Δ 일 때만 재할당 허용
# 스위칭으로 인한 비용 감소가 이 값보다 커야 재할당
AUCTION_DELTA_GAIN = 5.0

# K (최대 재할당 횟수): 요청(task) 1건당 최대 재할당 횟수
AUCTION_MAX_REASSIGN = 2

# Zone penalty: 사람의 존 외의 로봇에 대한 패널티
# 0으로 설정하면 존 제약 없이 가장 가까운 로봇에 할당
AUCTION_ZONE_PENALTY = 0.0

# --- Predictive rendezvous dispatch ----------------------------------------
# 기본(0)에서는 로봇이 랙에 주차한 뒤에야 작업자 배정이 시작되므로, 작업자의
# 이동시간 전체가 로봇 대기시간으로 계상된다(전체 런타임의 약 29%).
# 1로 켜면 이동 중인 로봇도 배정 후보가 되어 작업자를 미리 보낼 수 있다.
# 전략 이름이 predictive 계열이면 전략 쪽에서 자동으로 켠다.
PREDICTIVE_DISPATCH = int(os.environ.get("RWARE_PREDICTIVE_DISPATCH", "0"))

# 예측된 로봇 도착이 이 틱보다 더 남았으면 파견하지 않는다.
# 작업자가 랙 앞에서 노는 시간을 제한하는 안전장치.
DISPATCH_LEAD_LIMIT = int(os.environ.get("RWARE_DISPATCH_LEAD_LIMIT", "10"))

# Rendezvous 비용 가중치.
# 병목은 작업자다: 실측 가동률 61.8%(이동 36.3% + 피킹 25.5%)에 작업자 8명이
# 로봇 20대를 담당한다. 로봇 대기 28.9%는 대부분 희소한 서버를 기다리는 큐잉이라
# 조기 파견으로 없앨 수 없다. 따라서 작업자 시간(이동+랙 대기)을 온전히 계상하고
# 로봇 유휴는 부차적으로 둔다. 로봇 유휴를 크게 잡으면 먼 이동 중 로봇을 선호해
# 작업자 이동거리가 59% 늘고 처리량이 떨어진다(실측).
RENDEZVOUS_ROBOT_WAIT_WEIGHT = float(os.environ.get("RWARE_RV_ROBOT_WAIT_W", "1.0"))
RENDEZVOUS_HUMAN_WAIT_WEIGHT = float(os.environ.get("RWARE_RV_HUMAN_WAIT_W", "1.0"))
# Worker walking time. Kept separate from idle-at-rack so the distance signal
# survives; folding the two together collapses it.
RENDEZVOUS_HUMAN_TRAVEL_WEIGHT = float(os.environ.get("RWARE_RV_HUMAN_TRAVEL_W", "0.0"))
RENDEZVOUS_RISK_WEIGHT = float(os.environ.get("RWARE_RV_RISK_W", "0.5"))

# --- Arrival-time risk learning --------------------------------------------
RISK_WARMUP_TASKS = int(os.environ.get("RWARE_RISK_WARMUP", "500"))
RISK_RETRAIN_INTERVAL = int(os.environ.get("RWARE_RISK_RETRAIN", "500"))
RISK_CATBOOST_ITERATIONS = int(os.environ.get("RWARE_RISK_ITERATIONS", "300"))

# --- Idle-worker pre-positioning -------------------------------------------
# off      작업이 끝난 마지막 랙에서 그대로 대기 (기존 동작)
# nearest  가장 가까운 미할당 요청 쪽에서 대기 (학습 없음)
# learned  학습된 로봇 도착 분위수로 임박한 요청을 골라 대기
# oracle   실제로 다음에 배정될 랙으로 이동 (상한, 구현 불가능한 기준선)
# 근거 측정은 docs/IARL_findings.md 참조.
STAGING_POLICY = os.environ.get("RWARE_STAGING", "off")
# Assignment and staging may require different estimators in the same run.
# Keeping this explicit prevents rv_static from replacing staging's learner.
STAGING_ETA_BACKEND = os.environ.get("RWARE_STAGING_BACKEND", "catboost")

# learned 정책의 점수식 distance + early_w * early + uncertainty_w * uncertainty.
# early 는 로봇보다 일찍 도착해 노는 시간(q50 기준), uncertainty 는 q90 - q50.
# 한쪽 가중치를 0으로 두면 그 항의 기여를 분리 측정할 수 있다.
STAGING_EARLY_WEIGHT = float(os.environ.get("RWARE_STAGING_EARLY_W", "0.5"))
STAGING_UNCERTAINTY_WEIGHT = float(os.environ.get("RWARE_STAGING_UNCERTAINTY_W", "0.5"))

# --- Service-time variability (robustness scenario) ------------------------
# off | low | medium | high. off이면 서비스타임은 sku_count * sku_per_picking_time
# 으로 결정론적이며 기존 baseline 수치가 그대로 재현된다.
SERVICE_TIME_VARIABILITY = os.environ.get("RWARE_SERVICE_VARIABILITY", "off")
SERVICE_TIME_SEED = int(os.environ.get("RWARE_SERVICE_SEED", "0"))
