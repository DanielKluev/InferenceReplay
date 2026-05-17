"""
`pytest_context` stores per-test InferenceGate header policy for pytest consumers.

The InferenceGate pytest plugin writes ``X-InferenceGate-*`` header overrides
here for the duration of each test.  Downstream clients can read these values
and attach them to outbound HTTP requests without Gate importing those clients.

The context is task-local by default, with a process-wide fallback stack for
worker threads that do not inherit :class:`contextvars.ContextVar` state.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Iterator, Mapping

_HEADERS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("inference_gate_pytest_headers", default={})

# Cross-thread fallback stack for :func:`update_headers`.  Each entry is a
# ``(global_token, frame_dict)`` tuple.  ``global_token`` lets
# :func:`reset_headers` remove the correct frame even when nested contexts are
# unwound out of strict stack order.
_GLOBAL_HEADERS_LOCK = threading.RLock()
_GLOBAL_HEADERS_STACK: list[tuple[object, dict[str, str]]] = []


class _HeaderToken:
    """
    Opaque reset token combining ContextVar state with fallback-stack state.
    """

    __slots__ = ("cv_token", "global_token")

    def __init__(self, cv_token: contextvars.Token[dict[str, str]], global_token: object) -> None:
        self.cv_token = cv_token
        self.global_token = global_token


def current_headers() -> dict[str, str]:
    """
    Return a shallow copy of the headers set in the current task context.
    """
    return dict(_HEADERS.get())


def effective_headers() -> dict[str, str]:
    """
    Return headers visible to request injection in this thread.

    The current task context wins when present.  Threads that did not inherit
    that context fall back to the process-wide stack populated by
    :func:`update_headers`.
    """
    ctx = _HEADERS.get()
    if ctx:
        return dict(ctx)

    with _GLOBAL_HEADERS_LOCK:
        merged: dict[str, str] = {}
        for _global_token, frame in _GLOBAL_HEADERS_STACK:
            merged.update(frame)
        return merged


def set_headers(new_headers: Mapping[str, str]) -> contextvars.Token[dict[str, str]]:
    """
    Replace the current task-local header mapping with ``new_headers``.

    This primitive does not update the cross-thread fallback stack.  Callers
    that need request-injection visibility across worker threads should use
    :func:`update_headers` instead.
    """
    return _HEADERS.set(dict(new_headers))


def reset_headers(token: "contextvars.Token[dict[str, str]] | _HeaderToken") -> None:
    """
    Restore headers captured by :func:`set_headers` or :func:`update_headers`.
    """
    if isinstance(token, _HeaderToken):
        _HEADERS.reset(token.cv_token)
        with _GLOBAL_HEADERS_LOCK:
            for idx, (global_token, _frame) in enumerate(_GLOBAL_HEADERS_STACK):
                if global_token is token.global_token:
                    _GLOBAL_HEADERS_STACK.pop(idx)
                    return
        return
    _HEADERS.reset(token)


def update_headers(extra: Mapping[str, str]) -> _HeaderToken:
    """
    Merge ``extra`` into the current header context and fallback stack.

    Returns a reset token that must be passed to :func:`reset_headers` when
    the caller's scope ends.
    """
    merged = dict(_HEADERS.get())
    merged.update(extra)
    cv_token = _HEADERS.set(merged)
    global_token = object()
    with _GLOBAL_HEADERS_LOCK:
        _GLOBAL_HEADERS_STACK.append((global_token, dict(extra)))
    return _HeaderToken(cv_token, global_token)


@contextmanager
def headers(**extra: str) -> Iterator[None]:
    """
    Merge header values for the duration of the ``with`` block.
    """
    token = update_headers(extra)
    try:
        yield
    finally:
        reset_headers(token)