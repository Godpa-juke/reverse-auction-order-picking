"""Lightweight command/event primitives used by the warehouse engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional


@dataclass
class Command:
    """Unit of work queued for the engine."""

    name: str
    payload: Optional[Dict[str, Any]] = None
    tick: Optional[int] = None


@dataclass
class Event:
    """Notification emitted by the engine during simulation."""

    name: str
    payload: Optional[Dict[str, Any]] = None
    tick: Optional[int] = None


class CommandQueue:
    """FIFO queue for engine commands."""

    def __init__(self) -> None:
        self._queue: Deque[Command] = deque()

    def push(self, command: Command) -> None:
        self._queue.append(command)

    def extend(self, commands: Iterable[Command]) -> None:
        for command in commands:
            self.push(command)

    def drain(self) -> Iterator[Command]:
        while self._queue:
            yield self._queue.popleft()

    def clear(self) -> None:
        self._queue.clear()


class EventQueue:
    """FIFO queue for engine events."""

    def __init__(self) -> None:
        self._queue: Deque[Event] = deque()

    def emit(self, event: Event) -> None:
        self._queue.append(event)

    def drain(self) -> List[Event]:
        events: List[Event] = []
        while self._queue:
            events.append(self._queue.popleft())
        return events

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)


class EngineEventBus:
    """Simple pub-sub helper used by the Gym adapter or CLI."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_name, []).append(callback)

    def publish(self, event: Event) -> None:
        for callback in self._subscribers.get(event.name, []):
            callback(event)

