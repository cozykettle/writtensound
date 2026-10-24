"""A tiny thread-safe SSE pub/sub hub."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

EventData = Any

log = logging.getLogger(__name__)

MAX_QUEUE = 256


class EventHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[tuple[str, EventData]]] = set()

    def bind(self) -> None:
        self._loop = asyncio.get_running_loop()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[tuple[str, EventData]]:
        queue: asyncio.Queue[tuple[str, EventData]] = asyncio.Queue(maxsize=MAX_QUEUE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[tuple[str, EventData]]) -> None:
        self._subscribers.discard(queue)

    def _put(self, event: str, data: EventData) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait((event, data))
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait((event, data))
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    log.warning("dropping SSE event for a slow client")

    def publish(self, event: str, data: EventData) -> None:
        """Safe from the worker thread, route handlers, or the event loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._put, event, data)

    async def stream(
        self, queue: asyncio.Queue[tuple[str, EventData]] | None = None
    ) -> AsyncIterator[str]:
        if queue is None:
            queue = self.subscribe()
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": hb\n\n"
                    continue
                payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                yield f"event: {event}\ndata: {payload}\n\n"
        finally:
            self.unsubscribe(queue)
