# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Scoped, synchronous lock-acquisition deadlines for optional lifecycle work."""

from __future__ import annotations

import contextvars
import math
import time
from contextlib import contextmanager
from typing import Any, Iterator


DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS = 0.25


class LockTimeoutError(TimeoutError):
    """Raised when a scoped lock-acquisition deadline expires."""


_LOCK_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "epistemic_harness_lock_deadline",
    default=None,
)


@contextmanager
def lock_deadline(seconds: float = DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS) -> Iterator[float]:
    """Set one monotonic acquisition deadline for the current execution context.

    Nested scopes never extend an already-shorter deadline. The context variable
    is always restored, including when the body raises, so ordinary model/probe
    operations retain their blocking behavior after lifecycle work returns.
    """
    if isinstance(seconds, bool):
        raise TypeError("lock deadline must be a number of seconds")
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("lock deadline must be a finite non-negative number")
    now = time.monotonic()
    requested = now + seconds
    current = _LOCK_DEADLINE.get()
    deadline = min(current, requested) if current is not None else requested
    token = _LOCK_DEADLINE.set(deadline)
    try:
        yield deadline
    finally:
        _LOCK_DEADLINE.reset(token)


def remaining_lock_seconds() -> float | None:
    """Return the current scope's remaining seconds, or ``None`` when unscoped."""
    deadline = _LOCK_DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _timeout(label: str) -> LockTimeoutError:
    return LockTimeoutError(
        f"lock acquisition deadline exceeded while acquiring {label}"
    )


@contextmanager
def acquire_lock(lock: Any, *, label: str) -> Iterator[None]:
    """Acquire a Python lock synchronously, bounded only in a deadline scope.

    The lock is released only after this call successfully acquires it. This
    preserves RLock reentrancy and prevents a timed-out acquisition from
    releasing a lock owned by another caller.
    """
    remaining = remaining_lock_seconds()
    if remaining is None:
        acquired = lock.acquire()
    elif remaining <= 0:
        acquired = lock.acquire(blocking=False)
    else:
        acquired = lock.acquire(timeout=remaining)
    if not acquired:
        raise _timeout(label)
    try:
        yield
    finally:
        lock.release()


__all__ = [
    "DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS",
    "LockTimeoutError",
    "acquire_lock",
    "lock_deadline",
    "remaining_lock_seconds",
]
