"""Connected-user presence tracking.

HTTP is stateless, so the UI sends a lightweight heartbeat and the access
middleware also refreshes presence on normal API traffic.  A user is counted
as online while an authenticated account has been seen during the last
``STALE_AFTER_SECONDS`` seconds.

The registry stores no tokens or passwords.  Guest callers only receive the
aggregate count; the host may request the user list for the settings panel.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .settings import get_settings

STALE_AFTER_SECONDS = 90.0


class PresenceRegistry:
    """Thread-safe map of account id -> last-seen metadata."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}

    @staticmethod
    def _key(caller: dict) -> str:
        role = str(caller.get("role") or "guest")
        account_id = str(caller.get("id") or "unknown")
        return f"{role}:{account_id}"

    def _enabled(self) -> bool:
        try:
            return bool(getattr(get_settings(), "monitor_connections", True))
        except Exception:
            return True

    def touch(self, caller: dict) -> None:
        """Record activity for an authenticated caller."""
        if not caller or not self._enabled():
            return
        key = self._key(caller)
        now = self._clock()
        with self._lock:
            entry = self._users.get(key)
            if entry is None:
                entry = {
                    "id": str(caller.get("id") or "unknown"),
                    "username": str(caller.get("username") or "Пользователь"),
                    "role": str(caller.get("role") or "guest"),
                    "is_guest": bool(caller.get("is_guest", False)),
                    "connected_at": now,
                    "last_seen": now,
                }
                self._users[key] = entry
            else:
                entry["username"] = str(caller.get("username") or entry["username"])
                entry["role"] = str(caller.get("role") or entry["role"])
                entry["is_guest"] = bool(caller.get("is_guest", entry["is_guest"]))
                entry["last_seen"] = now
            self._prune_locked(now)

    def snapshot(self, include_users: bool = True) -> dict:
        """Return ``{enabled, count, users, stale_after}``.

        Disabled monitoring always reports an empty list and clears stale
        records so re-enabling starts from a clean state.
        """
        enabled = self._enabled()
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            if not enabled:
                self._users.clear()
                users = []
            else:
                users = [
                    {
                        "id": entry["id"],
                        "username": entry["username"],
                        "role": entry["role"],
                        "is_guest": entry["is_guest"],
                        "connected_at": round(entry["connected_at"], 3),
                        "last_seen": round(entry["last_seen"], 3),
                        "last_seen_ago": round(max(0.0, now - entry["last_seen"]), 1),
                    }
                    for entry in self._users.values()
                ]
        users.sort(key=lambda u: (u["is_guest"], u["username"].lower()))
        return {
            "enabled": enabled,
            "count": len(users),
            "users": users if include_users else [],
            "stale_after": STALE_AFTER_SECONDS,
        }

    def clear(self) -> None:
        with self._lock:
            self._users.clear()

    def _prune_locked(self, now: float) -> None:
        stale = [key for key, entry in self._users.items()
                 if now - entry["last_seen"] > STALE_AFTER_SECONDS]
        for key in stale:
            self._users.pop(key, None)


_registry: Optional[PresenceRegistry] = None
_registry_lock = threading.Lock()


def get_presence() -> PresenceRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = PresenceRegistry()
        return _registry


def record_activity(caller: dict) -> None:
    """Best-effort middleware hook; presence errors must never break a request."""
    try:
        get_presence().touch(caller)
    except Exception:
        pass
