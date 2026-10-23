"""Regression tests for GPU queueing, presence, History pagination and Dock drag.

These tests are intentionally fast: they exercise the new backend helpers
directly and use a temporary output directory for the History API.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Generation queue
# ---------------------------------------------------------------------------

def test_generation_queue_is_fifo_and_single_active():
    from backend.generation_queue import GenerationQueue

    queue = GenerationQueue()
    assert queue.enqueue("a") == 1
    assert queue.enqueue("b") == 2
    assert queue.enqueue("c") == 3

    assert queue.acquire("a") is True
    assert queue.is_active("a") is True
    assert queue.snapshot()["pending"] == ["b", "c"]

    result = []
    waiter = threading.Thread(target=lambda: result.append(queue.acquire("b")), daemon=True)
    waiter.start()
    time.sleep(0.05)
    # b must remain blocked while a owns the only slot.
    assert queue.snapshot()["active"] == "a"
    assert waiter.is_alive()

    queue.release("a")
    waiter.join(2)
    assert result == [True]
    assert queue.is_active("b") is True
    assert queue.snapshot()["pending"] == ["c"]
    queue.release("b")


def test_generation_queue_cancels_queued_and_active_jobs():
    from backend.generation_queue import GenerationQueue

    queue = GenerationQueue()
    queue.enqueue("queued")
    assert queue.cancel("queued") is False
    assert queue.acquire("queued") is False
    assert queue.position("queued") == 0

    queue.enqueue("active")
    assert queue.acquire("active") is True
    assert queue.cancel("active") is True
    assert queue.is_active("active") is True
    queue.release("active")
    assert queue.is_active("active") is False


# ---------------------------------------------------------------------------
# AppState integration
# ---------------------------------------------------------------------------

def test_app_state_exposes_queue_metadata_and_preserves_cancel():
    from app import AppState
    from backend.worker import JobStatus, ProgressEvent

    state = AppState()
    state.register_job("a", owner_id="host", owner_name="Host", kind="generation", serialized=True)
    state.register_job("b", owner_id="guest-1", owner_name="Guest", kind="generation", serialized=True)

    job_a = state.get_job("a")
    job_b = state.get_job("b")
    assert job_a["owner_id"] == "host"
    assert job_b["owner_name"] == "Guest"
    assert job_a["queue_position"] == 1
    assert job_b["queue_position"] == 2
    assert job_b["queue_size"] == 2
    assert job_b["status"] == JobStatus.QUEUED

    assert state.acquire_generation("a") is True
    assert state.get_job("a")["queue_active"] is True
    assert state.get_job("a")["queue_position"] == 0
    assert state.get_job("b")["queue_position"] == 1

    state.release_generation("a")
    assert state.acquire_generation("b") is True
    state.release_generation("b")

    state.register_job("queued-cancel", owner_id="host", kind="generation", serialized=True)
    assert state.cancel_job("queued-cancel") is False
    assert state.get_job("queued-cancel")["status"] == JobStatus.CANCELLED
    assert state.acquire_generation("queued-cancel") is False

    # A late worker error must not turn a user cancellation into an error.
    state.update_job("queued-cancel", ProgressEvent(status=JobStatus.ERROR, error="late failure"))
    assert state.get_job("queued-cancel")["status"] == JobStatus.CANCELLED

    state.register_job("active-cancel", owner_id="host", kind="generation", serialized=True)
    assert state.acquire_generation("active-cancel") is True
    assert state.cancel_job("active-cancel") is True
    state.release_generation("active-cancel")
    assert state.get_job("active-cancel")["status"] == JobStatus.CANCELLED

    # Race: the queue slot is active, then a cancel lands before AppState writes
    # INITIALIZING. acquire_generation must not overwrite CANCELLED or leak the
    # slot to a job whose worker was already stopped.
    state.register_job("activation-cancel", owner_id="host", kind="generation", serialized=True)
    assert state.generation_queue.acquire("activation-cancel") is True
    assert state.cancel_job("activation-cancel") is True
    assert state.acquire_generation("activation-cancel") is False
    assert state.get_job("activation-cancel")["status"] == JobStatus.CANCELLED
    assert state.generation_queue.is_active("activation-cancel") is False


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------

def test_presence_counts_unique_accounts_and_expires(monkeypatch):
    from backend import presence

    now = [1000.0]
    monkeypatch.setattr(presence, "get_settings", lambda: SimpleNamespace(monitor_connections=True))
    registry = presence.PresenceRegistry(clock=lambda: now[0])

    registry.touch({"role": "host", "id": "host", "username": "Host", "is_guest": False})
    registry.touch({"role": "host", "id": "host", "username": "Host renamed", "is_guest": False})
    registry.touch({"role": "guest", "id": "g1", "username": "Guest 1", "is_guest": True})
    # Same id but different role is a different account key.
    registry.touch({"role": "host", "id": "g1", "username": "Host g1", "is_guest": False})

    snapshot = registry.snapshot(include_users=True)
    assert snapshot["enabled"] is True
    assert snapshot["count"] == 3
    assert len(snapshot["users"]) == 3
    assert registry.snapshot(include_users=False)["users"] == []

    now[0] += presence.STALE_AFTER_SECONDS + 1
    assert registry.snapshot(include_users=True)["count"] == 0

    registry.touch({"role": "host", "id": "host", "username": "Host", "is_guest": False})
    monkeypatch.setattr(presence, "get_settings", lambda: SimpleNamespace(monitor_connections=False))
    disabled = registry.snapshot(include_users=True)
    assert disabled["enabled"] is False
    assert disabled["count"] == 0
    assert disabled["users"] == []


# ---------------------------------------------------------------------------
# History helper + API
# ---------------------------------------------------------------------------

def test_history_paginate_clamps_pages_and_sizes():
    from backend.history_paging import paginate

    items = [{"i": i} for i in range(35)]
    page = paginate(items, page=2, page_size=10)
    assert page["total"] == 35
    assert page["pages"] == 4
    assert page["page"] == 2
    assert [x["i"] for x in page["items"]] == list(range(10, 20))

    clamped = paginate(items, page=99, page_size=10)
    assert clamped["page"] == 4
    assert len(clamped["items"]) == 5

    assert paginate(items, page=0, page_size=0)["page"] == 1
    assert paginate(items, page=1, page_size=0)["page_size"] == 1
    assert paginate(items, page=1, page_size=999)["page_size"] == 100
    empty = paginate([], page=5, page_size=24)
    assert empty == {"items": [], "total": 0, "page": 1, "page_size": 24, "pages": 0}


def test_history_api_pagination_filters_and_presence_route(tmp_path, monkeypatch):
    import backend.cache as cache_module
    from fastapi.testclient import TestClient
    from app import create_app

    monkeypatch.setattr(
        cache_module,
        "get_settings",
        lambda: SimpleNamespace(output_dir=tmp_path, models_dir=tmp_path),
    )

    for i in range(30):
        image = tmp_path / f"history_{i:02d}.png"
        image.write_bytes(b"not-an-image-but-enough-for-history-metadata")
        meta = {
            "prompt": f"prompt number {i}",
            "negative_prompt": "",
            "model_id": "test-model",
            "owner_id": "host" if i % 2 == 0 else "guest-x",
            "owner_name": "Host" if i % 2 == 0 else "Guest X",
            "shared": i % 3 == 0,
        }
        image.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    client = TestClient(create_app())
    response = client.get("/api/history?page=2&page_size=10")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 30
    assert body["page"] == 2
    assert body["page_size"] == 10
    assert body["pages"] == 3
    assert len(body["items"]) == 10

    legacy = client.get("/api/history")
    assert legacy.status_code == 200
    assert isinstance(legacy.json(), list)
    assert len(legacy.json()) == 30

    mine = client.get("/api/history?page=1&page_size=24&filter=mine").json()
    assert mine["total"] == 15
    assert all(item["owner_id"] == "host" for item in mine["items"])

    search = client.get("/api/history?page=1&page_size=24&q=number%207").json()
    assert search["total"] == 1
    assert "number 7" in search["items"][0]["prompt"]

    clamped = client.get("/api/history?page=99&page_size=10").json()
    assert clamped["page"] == 3

    assert client.get("/api/history?page=0").status_code == 422
    assert client.get("/api/history?page_size=101").status_code == 422

    presence = client.get("/api/presence")
    assert presence.status_code == 200, presence.text
    assert presence.json()["enabled"] is True
    heartbeat = client.post("/api/presence/heartbeat")
    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["count"] >= 1


# ---------------------------------------------------------------------------
# Static integration guards
# ---------------------------------------------------------------------------

def test_backend_and_frontend_wiring():
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "TOOL_CPU_MAX_CONCURRENCY = 4" in app_py
    assert "STATE.acquire_generation(job_id)" in app_py
    assert "STATE.release_generation(job_id)" in app_py
    assert 'cache.list_outputs(limit=None)' in app_py
    assert 'owner_filter: str = Query("all", alias="filter")' in app_py
    assert 'STATE.register_job(job_id, owner_id=owner_id, owner_name=owner_name, kind="generation", serialized=True, shared=shared)' in app_py
    assert 'caller.get("is_guest", False) and job.get("owner_id") != caller.get("id")' in app_py

    settings_py = (ROOT / "backend" / "settings.py").read_text(encoding="utf-8")
    assert "monitor_connections: bool = True" in settings_py

    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "queue-overlay", "queue-position", "queue-total", "queue-message",
        "presence-indicator", "presence-count", "setting-monitor-connections",
        "presence-settings-count", "presence-users-list",
        "dock-drag-toggle-btn", "dock-reset-btn", "setting-dock-drag-toggle",
        "history-pagination", "history-page-size",
    ):
        assert f'id="{element_id}"' in html, f"missing #{element_id}"

    app_js = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    assert "showQueueStatus(status, jobId" in app_js
    assert "hideQueueStatus()" in app_js
    assert "renderHistoryPagination()" in app_js
    assert "setHistoryPageSize(" in app_js
    assert "handle.addEventListener(\"pointerdown\"" in app_js
    assert "wrap.addEventListener(\"pointerdown\"" not in app_js
    assert "dockDragEnabled" in app_js
    assert "dock.drag_enable" in app_js

    css = (ROOT / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")
    free_drag = re.search(r"\.gemini-dock-wrap\.dock-free-drag\s*\{(.*?)\}", css, re.S)
    assert free_drag is not None
    assert "position: fixed !important" in free_drag.group(1)
    assert "left: auto !important" not in free_drag.group(1)
    assert ".queue-overlay" in css
    assert ".presence-indicator" in css
    assert ".history-page-btn" in css

    ru = json.loads((ROOT / "frontend" / "locale" / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((ROOT / "frontend" / "locale" / "en.json").read_text(encoding="utf-8"))
    for key in ("queue.title", "queue.waiting", "presence.title", "presence.connected", "dock.drag_enable", "manga.queue_waiting"):
        assert key in ru, f"missing ru locale key {key}"
        assert key in en, f"missing en locale key {key}"
    assert "{{position}}" in ru["queue.waiting"]
    assert "{{count}}" in en["presence.connected"]
