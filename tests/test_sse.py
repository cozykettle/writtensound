from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from backend.events import EventHub
from backend.routes.api import events as events_endpoint


def make_request(state) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(app_state=state))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/events",
            "headers": [],
            "app": app,
        }
    )


@pytest.mark.asyncio
async def test_event_hub_publishes_cross_thread() -> None:
    hub = EventHub()
    hub.bind()
    queue = hub.subscribe()
    hub.publish("job", {"id": "abc", "status": "running"})
    await asyncio.sleep(0)
    event, data = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event == "job"
    assert data["id"] == "abc"
    hub.unsubscribe(queue)
    assert hub.subscriber_count == 0


@pytest.mark.asyncio
async def test_sse_snapshot_then_live_event() -> None:
    hub = EventHub()
    hub.bind()
    state = SimpleNamespace(
        hub=hub,
        snapshot=lambda: {"jobs": [], "hardware": {"devices": {"cpu": {}}}},
    )
    response = await events_endpoint(make_request(state))
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.body_iterator
    try:
        first = await anext(body)
        assert first.startswith("event: snapshot\ndata: ")
        snapshot = json.loads(first.split("data: ", 1)[1])
        assert snapshot["jobs"] == []

        hub.publish("queue", [{"id": "one"}])
        await asyncio.sleep(0)
        second = await asyncio.wait_for(anext(body), timeout=1.0)
        assert second.startswith("event: queue\ndata: ")
        assert json.loads(second.split("data: ", 1)[1]) == [{"id": "one"}]
    finally:
        await body.aclose()
    assert hub.subscriber_count == 0
