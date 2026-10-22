"""HTTP plumbing shared by the sync and async clients.

Everything here is independent of whether the caller is sync or async: URL and
header construction, retry decisions, and mapping responses to exceptions. The
two transports (:class:`SyncTransport`, :class:`AsyncTransport`) are thin loops
around ``httpx`` that call into these helpers.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Generic, TypeVar
from urllib.parse import quote

import anyio
import httpx

from ._version import __version__
from .errors import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTimeoutError,
    RateLimitError,
    ServerError,
    UnexpectedResponseError,
)

T = TypeVar("T")

USER_AGENT = f"rainbird-ai-python/{__version__} httpx/{httpx.__version__}"
DEFAULT_BASE_URL = "https://api.rainbird.ai"
# Engine names accepted by ``x-rainbird-engine`` (the same constants the JavaScript SDK exports).
ENGINE_CORE = "Core"
ENGINE_BETA = "Experimental (Beta)"
DEFAULT_TIMEOUT = 30.0
NL_VERSION = "v1"
MAX_ERROR_MESSAGE = 300
# A scheme-like prefix ("https://", "http:/", "ftp://") but not "localhost:8080" (colon + digit).
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):(?!\d)")

ENV_API_KEY = "RAINBIRD_API_KEY"
ENV_BASE_URL = "RAINBIRD_API_URL"
ENV_KM_ID = "RAINBIRD_KM_ID"
ENV_EVIDENCE_KEY = "RAINBIRD_EVIDENCE_KEY"
ENV_INTERACTION_KEY = "RAINBIRD_INTERACTION_KEY"
ENV_ENGINE = "RAINBIRD_ENGINE"


class Unset:
    """Type of :data:`UNSET`, which distinguishes "argument not given" from ``None``."""

    _instance: Unset | None = None

    def __new__(cls) -> Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = Unset()


@dataclass(frozen=True)
class RetryPolicy:
    """When and how to retry.

    Rainbird's decision endpoints are stateful: a ``POST /response`` that reached the
    engine but whose reply was lost could answer a question twice if replayed. The
    defaults therefore retry:

    * ``GET`` (and ``HEAD``/``OPTIONS``) requests on connection errors, timeouts,
      429, 500, 502, 503 and 504 (``retry_statuses``);
    * ``POST`` requests only on 429 and 503, where the service did not accept the
      request. 502 and 504 are **not** retried for ``POST`` because a gateway can
      report them after the engine has already applied the request. Transport
      errors on ``POST`` are not retried either unless
      ``retry_post_on_transport_error=True``.

    Delays grow exponentially (``backoff_factor * 2**attempt`` plus jitter, capped
    at ``max_backoff``) and honour ``Retry-After`` when the server sends one.
    """

    max_retries: int = 2
    backoff_factor: float = 0.5
    max_backoff: float = 20.0
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    post_retry_statuses: frozenset[int] = frozenset({429, 503})
    retry_post_on_transport_error: bool = False

    def should_retry_status(self, method: str, status: int, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if method.upper() in {"GET", "HEAD", "OPTIONS"}:
            return status in self.retry_statuses
        return status in self.post_retry_statuses

    def should_retry_transport(self, method: str, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        return method.upper() in {"GET", "HEAD", "OPTIONS"} or self.retry_post_on_transport_error

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_backoff)
        base = self.backoff_factor * (2.0**attempt)
        return min(self.max_backoff, base + random.uniform(0, base / 2))


NO_RETRY = RetryPolicy(max_retries=0)


def is_retryable_transport_error(exc: httpx.HTTPError) -> bool:
    """Network-level failures that may succeed on a retry (not URL/protocol misuse)."""
    return isinstance(
        exc,
        (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError),
    )


@dataclass(frozen=True)
class Request(Generic[T]):
    """A fully described API call plus how to turn its response into ``T``."""

    method: str
    path: str
    parse: Callable[[httpx.Response], T]
    params: Mapping[str, Any] | None = None
    json: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    accept: str = "application/json"

    def with_headers(self, extra: Mapping[str, str]) -> Request[T]:
        """Copy of this request with ``extra`` headers merged in (case-insensitively)."""
        return Request(
            method=self.method,
            path=self.path,
            parse=self.parse,
            params=self.params,
            json=self.json,
            headers=merge_headers(self.headers, extra),
            accept=self.accept,
        )


def normalize_base_url(value: Any, env: Mapping[str, str] | None = None) -> str:
    """Resolve and validate the API host.

    Explicit value, else ``RAINBIRD_API_URL``, else the Community host. A bare host
    (``acme-api.rainbird.ai``) gets ``https://``; anything that is not an ``http(s)``
    URL with a host, or that carries a query string or fragment, raises
    :class:`RainbirdConfigurationError`.
    """
    env = os.environ if env is None else env
    raw = str(value).strip() if value is not None and str(value).strip() else ""
    if not raw:
        raw = (env.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
    match = _SCHEME_RE.match(raw)
    if match:
        scheme = match.group(1).lower()
        rest = raw[match.end() :]
        if scheme not in ("http", "https") or not rest.startswith("//"):
            raise RainbirdConfigurationError(
                f"base_url {raw!r} must start with http:// or https:// (or be a bare host)"
            )
        url = f"{scheme}:{rest}"
    else:
        url = f"https://{raw}"
    if "?" in url or "#" in url:  # httpx reports an *empty* query/fragment as "", so check the text
        raise RainbirdConfigurationError(
            f"base_url {raw!r} must not include a query string or fragment"
        )
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise RainbirdConfigurationError(f"base_url {raw!r} is not a valid URL: {exc}") from exc
    if not parsed.host:
        raise RainbirdConfigurationError(f"base_url {raw!r} has no host")
    return str(parsed).rstrip("/")


@dataclass(frozen=True)
class ClientConfig:
    """Resolved client configuration (explicit arguments win over environment variables).

    ``timeout_explicit`` records whether the caller passed ``timeout``; when they did
    and also supplied their own ``httpx`` client, the value is applied per request so
    it is never silently ignored.

    The keys and ``default_headers`` (which may carry ``x-api-key``) are left out of
    ``repr()``, so printing or logging a config never shows a secret.
    """

    api_key: str = field(repr=False)
    base_url: str
    evidence_key: str | None = field(default=None, repr=False)
    interaction_key: str | None = field(default=None, repr=False)
    engine: str | None = None
    timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT
    timeout_explicit: bool = False
    retry: RetryPolicy = RetryPolicy()
    default_headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def resolve(
        cls,
        *,
        api_key: str | None,
        base_url: str | None,
        evidence_key: str | None,
        interaction_key: str | None,
        engine: str | None,
        timeout: float | httpx.Timeout | Unset | None = UNSET,
        retry: RetryPolicy | None,
        default_headers: Mapping[str, str] | None,
        env: Mapping[str, str] | None = None,
        require_api_key: bool = True,
    ) -> ClientConfig:
        env = os.environ if env is None else env
        raw_key = api_key if api_key is not None else env.get(ENV_API_KEY)
        key = str(raw_key).strip() if raw_key is not None else ""
        if require_api_key and not key:
            raise RainbirdConfigurationError(
                "An API key is required. Pass api_key=... or set the RAINBIRD_API_KEY "
                "environment variable. Keys are shown on the Publish page of a knowledge "
                "map in Rainbird Studio."
            )
        if isinstance(timeout, Unset):
            timeout_value: float | httpx.Timeout | None = DEFAULT_TIMEOUT
            explicit = False
        else:
            timeout_value, explicit = timeout, True
        return cls(
            api_key=key,
            base_url=normalize_base_url(base_url, env),
            evidence_key=clean_key(
                evidence_key if evidence_key is not None else env.get(ENV_EVIDENCE_KEY)
            ),
            interaction_key=clean_key(
                interaction_key if interaction_key is not None else env.get(ENV_INTERACTION_KEY)
            ),
            engine=engine if engine is not None else env.get(ENV_ENGINE),
            timeout=timeout_value,
            timeout_explicit=explicit,
            retry=retry if retry is not None else RetryPolicy(),
            default_headers=dict(default_headers or {}),
        )

    def base_headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.engine:
            headers["x-rainbird-engine"] = self.engine
        return merge_headers(headers, self.default_headers)


def clean_key(value: Any) -> str | None:
    """Strip a key (a pasted value, a CRLF ``.env`` file); ``None`` when nothing is left."""
    text = "" if value is None else str(value).strip()
    return text or None


# What an HTTP header value may hold: visible ASCII, spaces and tabs (no line breaks or controls).
_HEADER_VALUE_RE = re.compile(r"[\t\x20-\x7e]*")


def check_header_values(headers: Mapping[str, str]) -> None:
    """Reject header values httpx cannot send, naming the header but never echoing the value.

    The value is usually a key. Left to httpx, a stray line break fails in the transport
    with the raw value in the error message, and a smart quote raises ``UnicodeEncodeError``.
    """
    for name, value in headers.items():
        if isinstance(value, str) and not _HEADER_VALUE_RE.fullmatch(value):
            raise RainbirdConfigurationError(
                f"the {name} header contains a line break, control character or non-ASCII "
                "character (check how the value was copied); the request was not sent"
            )


def merge_headers(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge header dicts case-insensitively; later layers win and keep their spelling."""
    merged: dict[str, tuple[str, str]] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            merged[key.lower()] = (key, value)
    return dict(merged.values())


def path_segment(value: Any, name: str) -> str:
    """URL-encode a path segment and reject empty values with a clear message."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise RainbirdConfigurationError(f"{name} must not be empty")
    return quote(text, safe="")


def build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def clean_params(params: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Drop ``None`` values, render booleans as ``true``/``false`` and join lists with commas."""
    if not params:
        return None
    cleaned: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = ",".join(str(v) for v in value)
        else:
            cleaned[key] = str(value)
    return cleaned or None


# --------------------------------------------------------------------------- error handling


def extract_error_messages(body: Any) -> list[str]:
    """Collect human-readable error strings from any error body shape Rainbird uses.

    Known shapes: ``{"err": ["..."]}`` (400s), ``{"error": "..."}``,
    ``{"error": {"message": "..."}}``, ``{"errors": [...]}``, ``{"message": "..."}``.
    """
    messages: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                messages.append(text)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            found = False
            for key in ("message", "msg", "detail", "description", "error", "reason", "text"):
                if key in value:
                    found = True
                    add(value[key])
            if not found:
                messages.append(json.dumps(value, sort_keys=True, default=str))
        else:
            messages.append(str(value))

    if isinstance(body, dict):
        for key in ("err", "errors", "error", "message", "detail", "reason"):
            if key in body:
                add(body[key])
    elif isinstance(body, (list, str)):
        add(body)
    seen: set[str] = set()
    unique: list[str] = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def parse_body(response: httpx.Response) -> Any:
    """Parse JSON if possible, else return text (or ``None`` for empty bodies)."""
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "json" in content_type or text[:1] in "{[":
        try:
            return response.json()
        except ValueError:
            return text
    return text


def parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            return max(0.0, (when - _dt.datetime.now(tz=_dt.UTC)).total_seconds())
        except (TypeError, ValueError):
            return None


def _request_of(response: httpx.Response) -> httpx.Request | None:
    try:
        return response.request
    except RuntimeError:  # httpx raises when no request was attached
        return None


def _where(response: httpx.Response) -> tuple[str | None, str | None]:
    request = _request_of(response)
    if request is None:
        return None, None
    return request.method, str(request.url)


def _clean_text(text: str, content_type: str = "") -> str:
    """Collapse whitespace and truncate; strip tags only when the body looks like HTML."""
    if "html" in content_type.lower() or text.lstrip().startswith("<"):
        text = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > MAX_ERROR_MESSAGE:
        cleaned = cleaned[: MAX_ERROR_MESSAGE - 3].rstrip() + "..."
    return cleaned


_STATUS_DEFAULTS = {
    400: "Your request is missing a required parameter or a value is invalid",
    401: "Your API key is invalid",
    403: "Your API key does not permit access to this resource",
    404: "The resource (session, knowledge map or fact) could not be found",
    429: "Too many requests",
}


def error_for_response(response: httpx.Response) -> RainbirdAPIError:
    """Build the right :class:`RainbirdAPIError` subclass for a non-2xx response."""
    status = response.status_code
    body = parse_body(response)
    if isinstance(body, str):
        text = _clean_text(body, response.headers.get("content-type", ""))
        errors = [text] if text else []
    else:
        errors = extract_error_messages(body)
    detail = ""
    if errors:
        detail = errors[0] if len(errors) == 1 else "; ".join(errors)
        if len(detail) > MAX_ERROR_MESSAGE:
            detail = detail[: MAX_ERROR_MESSAGE - 3].rstrip() + "..."
    if 300 <= status < 400:
        location = response.headers.get("location", "<unknown>")
        message = f"unexpected redirect to {location}" + (f": {detail}" if detail else "")
    elif detail:
        message = detail
    else:
        message = _STATUS_DEFAULTS.get(status) or (
            "Rainbird server error" if status >= 500 else response.reason_phrase or "Request failed"
        )
    method, url = _where(response)
    kwargs: dict[str, Any] = {
        "status_code": status,
        "errors": errors,
        "body": body,
        "method": method,
        "url": url,
        "retry_after": parse_retry_after(response),
    }
    if status == 400:
        return BadRequestError(message, **kwargs)
    if status == 401:
        return AuthenticationError(message, **kwargs)
    if status == 403:
        return PermissionDeniedError(message, **kwargs)
    if status == 404:
        return NotFoundError(message, **kwargs)
    if status == 429:
        return RateLimitError(message, **kwargs)
    if status >= 500:
        return ServerError(message, **kwargs)
    return RainbirdAPIError(message, **kwargs)


def transport_error(exc: httpx.HTTPError, method: str, url: str) -> RainbirdError:
    """Map an ``httpx`` exception raised before any response arrived."""
    if isinstance(exc, httpx.TimeoutException):
        return RainbirdTimeoutError(
            f"request to {method} {url} timed out: {exc}", method=method, url=url
        )
    if isinstance(exc, httpx.UnsupportedProtocol):
        return RainbirdConfigurationError(f"invalid base_url for {method} {url}: {exc}")
    return RainbirdConnectionError(f"could not reach {method} {url}: {exc}", method=method, url=url)


def json_body(response: httpx.Response) -> Any:
    """Parse a 2xx response body as JSON, raising :class:`UnexpectedResponseError` otherwise."""
    try:
        return response.json()
    except ValueError as exc:
        method, url = _where(response)
        where = f" from {method} {url}" if method and url else ""
        snippet = response.text[:200]
        raise UnexpectedResponseError(
            f"expected a JSON body{where} but got: {snippet!r}",
            body=response.text,
            status_code=response.status_code,
        ) from exc


def text_body(response: httpx.Response) -> str:
    return response.text.strip()


# --------------------------------------------------------------------------- transports


class _BaseTransport:
    def __init__(self, config: ClientConfig, owns_client: bool) -> None:
        self.config = config
        self._owns_client = owns_client

    def _prepare(
        self, request: Request[Any], extra_headers: Mapping[str, str] | None
    ) -> dict[str, Any]:
        headers = merge_headers(
            self.config.base_headers(), {"Accept": request.accept}, request.headers, extra_headers
        )
        check_header_values(headers)
        kwargs: dict[str, Any] = {
            "method": request.method.upper(),
            "url": build_url(self.config.base_url, request.path),
            "params": clean_params(request.params),
            "json": request.json,
            "headers": headers,
        }
        if self.config.timeout_explicit and not self._owns_client:
            kwargs["timeout"] = self.config.timeout
        return kwargs


class SyncTransport(_BaseTransport):
    def __init__(self, config: ClientConfig, client: httpx.Client | None = None) -> None:
        super().__init__(config, owns_client=client is None)
        self._client = client or httpx.Client(timeout=config.timeout, follow_redirects=False)

    @property
    def http(self) -> httpx.Client:
        return self._client

    def send(self, request: Request[T], *, extra_headers: Mapping[str, str] | None = None) -> T:
        kwargs = self._prepare(request, extra_headers)
        method, url = kwargs["method"], kwargs["url"]
        policy = self.config.retry
        attempt = 0
        while True:
            try:
                response = self._client.request(**kwargs)
            except httpx.InvalidURL as exc:
                raise RainbirdConfigurationError(f"invalid URL {url!r}: {exc}") from exc
            except httpx.HTTPError as exc:
                if is_retryable_transport_error(exc) and policy.should_retry_transport(
                    method, attempt
                ):
                    time.sleep(policy.delay(attempt))
                    attempt += 1
                    continue
                raise transport_error(exc, method, url) from exc
            if response.is_success:
                return request.parse(response)
            if policy.should_retry_status(method, response.status_code, attempt):
                time.sleep(policy.delay(attempt, parse_retry_after(response)))
                attempt += 1
                continue
            raise error_for_response(response)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class AsyncTransport(_BaseTransport):
    def __init__(self, config: ClientConfig, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(config, owns_client=client is None)
        self._client = client or httpx.AsyncClient(timeout=config.timeout, follow_redirects=False)

    @property
    def http(self) -> httpx.AsyncClient:
        return self._client

    async def send(
        self, request: Request[T], *, extra_headers: Mapping[str, str] | None = None
    ) -> T:
        kwargs = self._prepare(request, extra_headers)
        method, url = kwargs["method"], kwargs["url"]
        policy = self.config.retry
        attempt = 0
        while True:
            try:
                response = await self._client.request(**kwargs)
            except httpx.InvalidURL as exc:
                raise RainbirdConfigurationError(f"invalid URL {url!r}: {exc}") from exc
            except httpx.HTTPError as exc:
                if is_retryable_transport_error(exc) and policy.should_retry_transport(
                    method, attempt
                ):
                    await anyio.sleep(policy.delay(attempt))
                    attempt += 1
                    continue
                raise transport_error(exc, method, url) from exc
            if response.is_success:
                return request.parse(response)
            if policy.should_retry_status(method, response.status_code, attempt):
                await anyio.sleep(policy.delay(attempt, parse_retry_after(response)))
                attempt += 1
                continue
            raise error_for_response(response)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
