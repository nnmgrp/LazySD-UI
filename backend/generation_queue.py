"""FIFO queue for serialized GPU generation jobs.

The WebUI uses a single global worker for diffusion generation.  Without a
queue two browser tabs (or a host + a guest) can start jobs at the same time,
which leads to VRAM OOMs and corrupt engine state.  :class:`GenerationQueue`
is a small, dependency-free mutex + FIFO waiter used by :class:`app.AppState`.

Only GPU jobs (text-to-image and Manga AI) are enqueued.  Downloads and CPU
Halftone/Manga processing stay parallel; they have their own (small) limits.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class GenerationQueue:
    """A single-active, FIFO job queue with cancellation support."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._pending: deque[str] = deque()
        self._active: Optional[str] = None
        self._cancelled: set[str] = set()

    # ---------- Queue mutation ----------
    def enqueue(self, job_id: str) -> int:
        """Add a job to the tail.  Returns its 1-based queue position."""
        with self._cv:
            if job_id == self._active:
                return 0
            if job_id not in self._pending:
                self._pending.append(job_id)
            self._cv.notify_all()
            try:
                return self._pending.index(job_id) + 1
            except ValueError:
                return 0

    def acquire(self, job_id: str, timeout: Optional[float] = None) -> bool:
        """Block until ``job_id`` reaches the head, then make it active.

        Returns ``False`` when the job was cancelled or removed while waiting.
        A timeout is primarily useful in tests and for future API limits.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._cv:
            while True:
                if job_id in self._cancelled:
                    self._cancelled.discard(job_id)
                    return False
                if self._active == job_id:
                    return True
                if self._active is None and self._pending and self._pending[0] == job_id:
                    self._pending.popleft()
                    self._active = job_id
                    self._cv.notify_all()
                    return True
                if job_id not in self._pending:
                    # A queued job may have been removed by cancel() before
                    # its worker thread reached acquire().
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(min(remaining, 0.5))
                else:
                    self._cv.wait(0.5)

    def release(self, job_id: str) -> None:
        """Release the active slot if ``job_id`` currently owns it."""
        with self._cv:
            if self._active == job_id:
                self._active = None
            self._cancelled.discard(job_id)
            self._cv.notify_all()

    def cancel(self, job_id: str) -> bool:
        """Cancel a waiting job, or report whether an active job was cancelled.

        Returns ``True`` only when ``job_id`` is the currently active job; the
        caller still has to stop the worker process for active jobs.
        """
        with self._cv:
            was_active = self._active == job_id
            if was_active or job_id in self._pending:
                self._cancelled.add(job_id)
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            self._cv.notify_all()
            return was_active

    # ---------- Inspection ----------
    def snapshot(self) -> dict:
        """Return a copy of the queue state.  Never hold the lock externally."""
        with self._cv:
            return {
                "active": self._active,
                "pending": list(self._pending),
                "pending_count": len(self._pending),
            }

    def position(self, job_id: str) -> int:
        """Return the 1-based pending position, or 0 when not waiting."""
        with self._cv:
            try:
                return self._pending.index(job_id) + 1
            except ValueError:
                return 0

    def is_active(self, job_id: str) -> bool:
        with self._cv:
            return self._active == job_id

    def clear(self) -> None:
        """Reset state (used by tests / a full settings reset)."""
        with self._cv:
            self._pending.clear()
            self._active = None
            self._cancelled.clear()
            self._cv.notify_all()
