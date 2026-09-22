"""Helpers for the parity tests ported from the Go and JavaScript Rainbird SDKs.

The original suites drive each SDK call against a per-test fake HTTP server (Go uses
``httptest.NewServer``, JS mocks ``global.fetch``) and assert on the exact request that was sent
and on the value returned. ``make_client`` reproduces that setup for the Python SDK: a
``Rainbird`` whose ``httpx.Client`` talks to an ``httpx.MockTransport`` running the given handler,
plus a ``Recorder`` that captures every request. Retries are disabled so a 500 fails at once,
as it does in the other SDKs.
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from rainbird_ai import NO_RETRY, Rainbird
from rainbird_ai.aio import AsyncRainbird

API_KEY = "test-api-key"
BASE_URL = "https://api.test"

Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class Call:
    """One request the SDK sent, as seen by the fake server."""

    method: str
    url: httpx.URL
    headers: httpx.Headers
    content: bytes

    @property
    def path(self) -> str:
        return self.url.path

    @property
    def query(self) -> dict[str, str]:
        return dict(self.url.params)

    @property
    def json(self) -> Any:
        return _json.loads(self.content) if self.content else None


@dataclass
class Recorder:
    """Every request the fake server saw, in order."""

    calls: list[Call] = field(default_factory=list)

    @property
    def last(self) -> Call:
        assert self.calls, "the SDK sent no request"
        return self.calls[-1]

    def __len__(self) -> int:
        return len(self.calls)


def respond(
    status: int,
    body: Any = None,
    *,
    text: str | None = None,
    headers: dict[str, str] | None = None,
) -> Handler:
    """A handler that always returns the same response (Go's one-line ``httptest`` handler)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text, headers=headers)
        if body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    return handler


def raise_transport_error(exc: Exception | None = None) -> Handler:
    """A handler that fails at the transport level (JS ``fetch`` rejecting)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc or httpx.ConnectError("fetch failed", request=request)

    return handler


def _recording(handler: Handler, recorder: Recorder) -> Handler:
    def wrapped(request: httpx.Request) -> httpx.Response:
        recorder.calls.append(Call(request.method, request.url, request.headers, request.content))
        return handler(request)

    return wrapped


def make_client(
    handler: Handler,
    *,
    api_key: str | None = API_KEY,
    base_url: str | None = BASE_URL,
    **kwargs: Any,
) -> tuple[Rainbird, Recorder]:
    """A sync client wired to ``handler`` and the recorder of what it sends."""
    recorder = Recorder()
    http = httpx.Client(transport=httpx.MockTransport(_recording(handler, recorder)))
    kwargs.setdefault("retry", NO_RETRY)
    return Rainbird(api_key=api_key, base_url=base_url, http_client=http, **kwargs), recorder


def make_async_client(
    handler: Handler,
    *,
    api_key: str | None = API_KEY,
    base_url: str | None = BASE_URL,
    **kwargs: Any,
) -> tuple[AsyncRainbird, Recorder]:
    """Async twin of :func:`make_client`."""
    recorder = Recorder()
    http = httpx.AsyncClient(transport=httpx.MockTransport(_recording(handler, recorder)))
    kwargs.setdefault("retry", NO_RETRY)
    return AsyncRainbird(api_key=api_key, base_url=base_url, http_client=http, **kwargs), recorder
