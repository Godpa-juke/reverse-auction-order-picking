"""Quantile prediction (Q50 / Q90) for arrival and service durations.

The allocator needs two numbers per candidate trip: how long it usually takes
(Q50) and how long it takes when it goes badly (Q90). The gap between them is
the tail risk the auction prices in when deciding how early to dispatch a
worker.

Backends
--------
``static``          Return the caller's static estimate unchanged. No learning;
                    Q90 == Q50, so tail risk is always zero. This is what a
                    path-length-based dispatcher assumes.
``rolling_median``  Rolling quantiles of the realised/static ratio per agent.
                    Cheap, dependency-free, and the honest "you could have done
                    this with a moving average" baseline.
``catboost``        CatBoost with MultiQuantile loss, retrained periodically on
                    the accumulated trip log.
``catboost_median`` CatBoost with a single Q50 quantile loss. It returns the
                    median for both outputs so consumers cannot use a learned
                    uncertainty spread.
``oracle``          Upper bound wired by the caller when the realised value is
                    knowable in advance.

Callers supply their own feature column list, so the same class serves the
worker-arrival, robot-arrival and service-time models.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Default columns for a service-time model. Arrival models pass their own.
FEATURE_COLUMNS: List[str] = [
    "agent_id",
    "zone_id",
    "static_ticks",
    "manhattan",
    "detour_ratio",
    "planned_path_len",
    "recent_ratio_mean",
    "recent_ratio_p90",
    "recent_samples",
    "active_robots",
    "moving_robots",
    "busy_humans",
    "elapsed_ticks",
]

# Columns treated as categorical when present in the feature list.
CATEGORICAL_CANDIDATES = ("agent_id", "zone_id", "human_id", "robot_id")

CATBOOST_BACKENDS = ("catboost", "catboost_median")
LEARNING_BACKENDS = CATBOOST_BACKENDS + ("rolling_median",)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile over an already-sorted sequence."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


class ServiceRiskModel:
    """Predicts (Q50, Q90) durations in ticks."""

    def __init__(
        self,
        backend: str = "static",
        feature_columns: Optional[List[str]] = None,
        warmup_tasks: int = 500,
        retrain_interval: int = 500,
        history_window: int = 40,
        min_group_samples: int = 8,
        catboost_iterations: int = 300,
        max_training_rows: int = 20000,
    ) -> None:
        self.backend = (backend or "static").lower()
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.categorical_columns = [
            name for name in CATEGORICAL_CANDIDATES if name in self.feature_columns
        ]
        self.warmup_tasks = int(warmup_tasks)
        self.retrain_interval = int(retrain_interval)
        self.min_group_samples = int(min_group_samples)
        self.catboost_iterations = int(catboost_iterations)
        self.max_training_rows = int(max_training_rows)

        self._group_history: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_window)
        )
        self._rows: Deque[List[Any]] = deque(maxlen=self.max_training_rows)
        self._targets: Deque[float] = deque(maxlen=self.max_training_rows)
        self._model = None
        self._seen_since_train = 0
        self.train_count = 0
        self.predict_count = 0
        self.observe_count = 0
        self.model_ready = False

    @property
    def learns(self) -> bool:
        return self.backend in LEARNING_BACKENDS

    # -- observation --------------------------------------------------------

    def observe(self, features: Dict[str, Any], duration_ticks: float) -> None:
        """Record one completed trip."""

        self.observe_count += 1
        group = int(features.get("agent_id", features.get("human_id", 0)))
        static = float(features.get("static_ticks", 0.0) or 0.0)
        if static > 0:
            self._group_history[group].append(float(duration_ticks) / static)

        if self.backend in CATBOOST_BACKENDS:
            self._rows.append([features.get(name, 0) for name in self.feature_columns])
            self._targets.append(float(duration_ticks))
            self._seen_since_train += 1

    def maybe_retrain(self) -> bool:
        """Retrain once enough new trips have accumulated."""

        if self.backend not in CATBOOST_BACKENDS:
            return False
        if len(self._rows) < self.warmup_tasks:
            return False
        if self.model_ready and self._seen_since_train < self.retrain_interval:
            return False
        return self._train()

    def _train(self) -> bool:
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            # Degrade to the rolling ladder rather than failing the run.
            self.backend = "rolling_median"
            return False

        cat_indices = [self.feature_columns.index(name) for name in self.categorical_columns]
        cat_index_set = set(cat_indices)
        rows = [
            [int(value) if idx in cat_index_set else float(value) for idx, value in enumerate(row)]
            for row in self._rows
        ]
        pool = Pool(rows, list(self._targets), cat_features=cat_indices)
        loss_function = (
            "Quantile:alpha=0.5"
            if self.backend == "catboost_median"
            else "MultiQuantile:alpha=0.5,0.9"
        )
        model = CatBoostRegressor(
            loss_function=loss_function,
            iterations=self.catboost_iterations,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False,
            # One thread per process: ablation variants run as parallel
            # processes and must not oversubscribe the machine.
            thread_count=1,
        )
        model.fit(pool)
        self._model = model
        self.model_ready = True
        self._seen_since_train = 0
        self.train_count += 1
        return True

    # -- prediction ---------------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        return self.predict_batch([features])[0]

    def predict_batch(self, rows: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Predict (Q50, Q90) for a batch of candidate trips."""

        if not rows:
            return []
        self.predict_count += len(rows)

        if self.backend in CATBOOST_BACKENDS and self.model_ready and self._model is not None:
            return self._catboost_predict(rows)
        estimates = [self._history_estimate(row) for row in rows]
        if self.backend == "catboost_median":
            return [(q50, q50) for q50, _q90 in estimates]
        return estimates

    def _catboost_predict(self, rows: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        cat_index_set = {self.feature_columns.index(name) for name in self.categorical_columns}
        matrix = [
            [
                int(row.get(name, 0)) if idx in cat_index_set else float(row.get(name, 0) or 0.0)
                for idx, name in enumerate(self.feature_columns)
            ]
            for row in rows
        ]
        raw = self._model.predict(matrix)
        results: List[Tuple[float, float]] = []
        for prediction in raw:
            if self.backend == "catboost_median":
                q50 = q90 = float(prediction)
            else:
                q50, q90 = float(prediction[0]), float(prediction[1])
            if q90 < q50:  # quantile crossing
                q50, q90 = q90, q50
            q50 = max(1.0, q50)
            results.append((q50, max(q50, q90)))
        return results

    def _history_estimate(self, row: Dict[str, Any]) -> Tuple[float, float]:
        """Agent history -> pooled history -> the caller's static estimate."""

        static = float(row.get("static_ticks", 0.0) or 0.0)
        group = int(row.get("agent_id", row.get("human_id", 0)))

        history = self._group_history.get(group)
        values: Sequence[float] = list(history) if history else []
        if len(values) < self.min_group_samples:
            pooled: List[float] = []
            for other in self._group_history.values():
                pooled.extend(other)
            values = pooled
        if len(values) < self.min_group_samples or static <= 0:
            return max(1.0, static), max(1.0, static)

        ordered = sorted(values)
        q50 = max(1.0, static * quantile(ordered, 0.5))
        q90 = max(q50, static * quantile(ordered, 0.9))
        return q50, q90

    # -- diagnostics --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_ready": self.model_ready,
            "train_count": self.train_count,
            "observe_count": self.observe_count,
            "predict_count": self.predict_count,
        }


def build_risk_model(
    config: Optional[Any],
    backend: str,
    feature_columns: Optional[List[str]] = None,
) -> ServiceRiskModel:
    """Construct the model requested by a strategy."""

    return ServiceRiskModel(
        backend=backend,
        feature_columns=feature_columns,
        warmup_tasks=int(getattr(config, "risk_warmup_tasks", 500) or 500),
        retrain_interval=int(getattr(config, "risk_retrain_interval", 500) or 500),
        catboost_iterations=int(getattr(config, "risk_catboost_iterations", 300) or 300),
    )


__all__ = [
    "FEATURE_COLUMNS",
    "CATBOOST_BACKENDS",
    "CATEGORICAL_CANDIDATES",
    "LEARNING_BACKENDS",
    "ServiceRiskModel",
    "build_risk_model",
    "quantile",
]
