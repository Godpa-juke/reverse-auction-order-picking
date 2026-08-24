"""Helper utilities for wiring together engine subsystems."""

from __future__ import annotations

from dataclasses import dataclass
from rware.core.agent_manager import AgentManager
from rware.core.config import SimulationConfig
from rware.core.data_collector import DataCollector
from rware.core.environment import EnvironmentCore
from rware.core.task_scheduler import TaskScheduler


@dataclass
class EngineServices:
    """Bundle of core services that power :class:`WarehouseEngine`."""

    environment: EnvironmentCore
    agent_manager: AgentManager
    task_scheduler: TaskScheduler
    data_collector: DataCollector


def create_engine_services(config: SimulationConfig) -> EngineServices:
    """Instantiate the canonical set of engine services for the given config."""

    environment = EnvironmentCore(config)
    agent_manager = AgentManager(config)
    task_scheduler = TaskScheduler(config)
    data_collector = DataCollector(config)

    return EngineServices(
        environment=environment,
        agent_manager=agent_manager,
        task_scheduler=task_scheduler,
        data_collector=data_collector,
    )
