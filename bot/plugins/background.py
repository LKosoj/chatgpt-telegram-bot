"""Descriptor for plugin-supplied background tasks.

Lifecycle (start/stop/retry) is handled by ``PluginManager``; plugins only
declare what they want to run by returning a list of ``BackgroundTask`` from
``Plugin.get_background_tasks()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class BackgroundTask:
    """A single recurring background task description.

    Attributes:
        name: Stable name (used for log identification, e.g.
            ``"<plugin_id>.<task.name>"``).
        interval_seconds: Delay between iterations. Also acts as the retry
            backoff if the coroutine raises.
        coroutine_factory: Callable returning a fresh awaitable on each tick.
            Will be invoked as ``coroutine_factory(application=app)``.
    """

    name: str
    interval_seconds: float
    coroutine_factory: Callable[..., Awaitable[None]]
