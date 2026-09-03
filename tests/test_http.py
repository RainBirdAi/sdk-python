"""Unit tests for :mod:`rainbird_ai._http`.

Covers retry decisions and delays, error-body extraction and exception mapping,
configuration resolution, URL/param helpers, and the sync/async transports'
retry loops (driven by ``httpx.MockTransport`` handlers, with sleeping patched out).
"""

from __future__ import annotations

import datetime as dt
import random
import time
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from email.utils import format_datetime
from typing import Any

import anyio
import httpx
import pytest

from rainbird_ai import Rainbird, __version__
from rainbird_ai._http import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    NO_RETRY,
    UNSET,
    USER_AGENT,
    AsyncTransport,
    ClientConfig,
    Request,
    RetryPolicy,
    SyncTransport,
    build_url,
    clean_params,
    error_for_response,
    extract_error_messages,
    is_retryable_transport_error,
    json_body,
    merge_headers,
    normalize_base_url,
    parse_body,
    parse_retry_after,
    path_segment,
    text_body,
    transport_error,
)
from rainbird_ai.aio import AsyncRainbird
from rainbird_ai.errors import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTimeoutError,
    RainbirdTransportError,
    RateLimitError,
    ServerError,
    UnexpectedResponseError,
)
from tests.conftest import API_KEY, BASE_URL, KM_ID, FakeRainbird

# --------------------------------------------------------------------------- helpers


def _config(**overrides: Any) -> ClientConfig:
    fields: dict[str, Any] = {"api_key": API_KEY, "base_url": BASE_URL}
    fields.update(overrides)
    return ClientConfig(**fields)


def _resolve(**overrides: Any) -> ClientConfig:
    """Call ``ClientConfig.resolve`` with everything unset and an empty environment."""
    kwargs: dict[str, Any] = {
        "api_key": None,
        "base_url": None,
        "evidence_key": None,
        "interaction_key": None,
        "engine": None,
        "timeout": UNSET,
        "retry": None,
        "default_headers": None,
        "env": {},
    }
    kwargs.update(overrides)
    return ClientConfig.resolve(**kwargs)


def _response(
    status: int,
    *,
    json: Any = None,
    text: str | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    url: str = f"{BASE_URL}/things",
) -> httpx.Response:
    """An ``httpx.Response`` with a request attached, as the client would produce."""
    return httpx.Response(
        status,
        json=json,
        text=text,
        content=content,
        headers=headers,
        request=httpx.Request(method, url),
    )


def _fail(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"err": [f"fail {status}"]}, headers=headers)


class ScriptedHandler:
    """``MockTransport`` handler that replays ``script`` (responses or exceptions) then succeeds."""

    def __init__(self, *script: int | BaseException | httpx.Response) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, httpx.Response):
                return step
            return _fail(step)
        return httpx.Response(200, json={"ok": True})


def _sync(handler: ScriptedHandler, retry: RetryPolicy | None = None) -> SyncTransport:
    config = _config(retry=retry) if retry is not None else _config()
    return SyncTransport(config, httpx.Client(transport=httpx.MockTransport(handler)))


def _async(handler: ScriptedHandler, retry: RetryPolicy | None = None) -> AsyncTransport:
    config = _config(retry=retry) if retry is not None else _config()
    return AsyncTransport(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


GET_THINGS: Request[Any] = Request("GET", "things", parse=json_body)
POST_THINGS: Request[Any] = Request("POST", "things", parse=json_body, json={"a": 1})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping (patches ``time.sleep`` and ``anyio.sleep``)."""
    recorded: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    async def fake_async_sleep(seconds: float, result: Any = None) -> Any:
        recorded.append(seconds)
        return result

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(anyio, "sleep", fake_async_sleep)
    return recorded


@pytest.fixture
def no_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "uniform", lambda _a, _b: 0.0)


@pytest.fixture
def fake_transport(fake: FakeRainbird) -> Iterator[SyncTransport]:
    http = httpx.Client(transport=httpx.MockTransport(fake.handle))
    yield SyncTransport(_config(), http)
    http.close()


# --------------------------------------------------------------------------- RetryPolicy


class TestRetryPolicyStatus:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_get_retries_429_and_5xx(self, status: int) -> None:
        assert RetryPolicy().should_retry_status("GET", status, 0) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 418, 501])
    def test_get_does_not_retry_other_statuses(self, status: int) -> None:
        assert RetryPolicy().should_retry_status("GET", status, 0) is False

    @pytest.mark.parametrize("status", [429, 503])
    def test_post_retries_only_statuses_refused_before_processing(self, status: int) -> None:
        assert RetryPolicy().should_retry_status("POST", status, 0) is True

    @pytest.mark.parametrize("status", [500, 502, 504])
    def test_post_does_not_retry_5xx_the_engine_may_have_applied(self, status: int) -> None:
        # A gateway can report 502/504 after the engine has already processed the request.
        assert RetryPolicy().should_retry_status("POST", status, 0) is False

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 418])
    def test_post_does_not_retry_4xx(self, status: int) -> None:
        assert RetryPolicy().should_retry_status("POST", status, 0) is False

    def test_method_is_case_insensitive(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry_status("get", 500, 0) is True
        assert policy.should_retry_status("post", 500, 0) is False

    @pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
    def test_head_and_options_behave_like_get(self, method: str) -> None:
        assert RetryPolicy().should_retry_status(method, 500, 0) is True

    @pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
    def test_other_write_methods_behave_like_post(self, method: str) -> None:
        policy = RetryPolicy()
        assert policy.should_retry_status(method, 500, 0) is False
        assert policy.should_retry_status(method, 503, 0) is True

    def test_stops_when_attempt_reaches_max_retries(self) -> None:
        policy = RetryPolicy(max_retries=2)
        assert policy.should_retry_status("GET", 503, 0) is True
        assert policy.should_retry_status("GET", 503, 1) is True
        assert policy.should_retry_status("GET", 503, 2) is False
        assert policy.should_retry_status("GET", 503, 3) is False

    def test_custom_status_sets_are_honoured(self) -> None:
        policy = RetryPolicy(
            retry_statuses=frozenset({503}), post_retry_statuses=frozenset({429, 500})
        )
        assert policy.should_retry_status("GET", 503, 0) is True
        assert policy.should_retry_status("GET", 500, 0) is False
        assert policy.should_retry_status("POST", 500, 0) is True
        assert policy.should_retry_status("POST", 503, 0) is False


class TestRetryPolicyTransport:
    def test_get_retries_transport_errors(self) -> None:
        assert RetryPolicy().should_retry_transport("GET", 0) is True

    @pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "get"])
    def test_safe_methods_retry_transport_errors(self, method: str) -> None:
        assert RetryPolicy().should_retry_transport(method, 0) is True

    @pytest.mark.parametrize("method", ["POST", "post", "PUT", "DELETE"])
    def test_post_never_retries_transport_errors_by_default(self, method: str) -> None:
        assert RetryPolicy().should_retry_transport(method, 0) is False

    def test_post_retries_transport_errors_when_opted_in(self) -> None:
        policy = RetryPolicy(retry_post_on_transport_error=True)
        assert policy.should_retry_transport("POST", 0) is True

    def test_stops_when_attempt_reaches_max_retries(self) -> None:
        policy = RetryPolicy(max_retries=2, retry_post_on_transport_error=True)
        assert policy.should_retry_transport("GET", 1) is True
        assert policy.should_retry_transport("GET", 2) is False
        assert policy.should_retry_transport("POST", 2) is False


class TestIsRetryableTransportError:
    @pytest.mark.parametrize(
        "exc_type",
        [
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
        ],
    )
    def test_network_failures_are_retryable(self, exc_type: type[httpx.HTTPError]) -> None:
        assert is_retryable_transport_error(exc_type("x")) is True

    @pytest.mark.parametrize(
        "exc_type",
        [
            httpx.UnsupportedProtocol,
            httpx.LocalProtocolError,
            httpx.DecodingError,
            httpx.TooManyRedirects,
        ],
    )
    def test_misuse_is_not_retryable(self, exc_type: type[httpx.HTTPError]) -> None:
        assert is_retryable_transport_error(exc_type("x")) is False


class TestRetryPolicyDelay:
    def test_honours_retry_after(self) -> None:
        assert RetryPolicy().delay(0, retry_after=3.0) == 3.0

    def test_retry_after_is_used_regardless_of_attempt(self) -> None:
        assert RetryPolicy().delay(5, retry_after=1.5) == 1.5

    def test_retry_after_capped_at_max_backoff(self) -> None:
        assert RetryPolicy(max_backoff=20.0).delay(0, retry_after=1000.0) == 20.0

    def test_zero_retry_after_means_no_wait(self) -> None:
        assert RetryPolicy().delay(0, retry_after=0.0) == 0.0

    def test_negative_retry_after_falls_back_to_backoff(self, no_jitter: None) -> None:
        assert RetryPolicy(backoff_factor=0.5).delay(0, retry_after=-1.0) == 0.5

    def test_exponential_growth_without_jitter(self, no_jitter: None) -> None:
        policy = RetryPolicy(backoff_factor=0.5, max_backoff=100.0)
        assert [policy.delay(n) for n in range(4)] == [0.5, 1.0, 2.0, 4.0]

    def test_backoff_factor_scales_delay(self, no_jitter: None) -> None:
        assert RetryPolicy(backoff_factor=2.0, max_backoff=100.0).delay(2) == 8.0

    def test_jitter_adds_up_to_half_the_base(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=1000.0)
        for attempt in range(4):
            base = 2.0**attempt
            for _ in range(25):
                delay = policy.delay(attempt)
                assert base <= delay <= base * 1.5

    def test_backoff_capped_at_max_backoff(self) -> None:
        assert RetryPolicy(backoff_factor=0.5, max_backoff=20.0).delay(10) == 20.0

    def test_default_max_backoff_is_20_seconds(self) -> None:
        policy = RetryPolicy()
        assert policy.max_backoff == 20.0
        assert policy.delay(30) == 20.0


class TestRetryPolicyDefaults:
    def test_default_values(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.backoff_factor == 0.5
        assert policy.retry_statuses == frozenset({429, 500, 502, 503, 504})
        assert policy.post_retry_statuses == frozenset({429, 503})
        assert policy.retry_post_on_transport_error is False

    def test_no_retry_has_zero_retries(self) -> None:
        assert NO_RETRY.max_retries == 0
        assert NO_RETRY == RetryPolicy(max_retries=0)

    def test_no_retry_never_retries(self) -> None:
        assert NO_RETRY.should_retry_status("GET", 503, 0) is False
        assert NO_RETRY.should_retry_status("POST", 429, 0) is False
        assert NO_RETRY.should_retry_transport("GET", 0) is False

    def test_policy_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            RetryPolicy().max_retries = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- error extraction


class TestExtractErrorMessages:
    def test_err_list(self) -> None:
        assert extract_error_messages({"err": ["first", "second"]}) == ["first", "second"]

    def test_error_string(self) -> None:
        assert extract_error_messages({"error": "boom"}) == ["boom"]

    def test_error_object_with_message(self) -> None:
        assert extract_error_messages({"error": {"message": "inner"}}) == ["inner"]

    def test_errors_list_of_objects(self) -> None:
        body = {"errors": [{"message": "m1"}, {"message": "m2"}]}
        assert extract_error_messages(body) == ["m1", "m2"]

    def test_message_key(self) -> None:
        assert extract_error_messages({"message": "msg"}) == ["msg"]

    @pytest.mark.parametrize("key", ["detail", "reason"])
    def test_detail_and_reason_top_level_keys(self, key: str) -> None:
        assert extract_error_messages({key: "why"}) == ["why"]

    def test_nested_lists_are_flattened(self) -> None:
        body = {"err": [["a", ["b", ["c"]]], "d"]}
        assert extract_error_messages(body) == ["a", "b", "c", "d"]

    def test_string_body(self) -> None:
        assert extract_error_messages("plain text") == ["plain text"]

    def test_list_body(self) -> None:
        assert extract_error_messages(["x", {"message": "y"}]) == ["x", "y"]

    def test_deduplicates_preserving_order(self) -> None:
        body = {"err": ["same", "other", "same"], "message": "same", "error": "other"}
        assert extract_error_messages(body) == ["same", "other"]

    def test_multiple_top_level_keys_in_priority_order(self) -> None:
        body = {"message": "m", "error": "e", "err": ["list"]}
        assert extract_error_messages(body) == ["list", "e", "m"]

    def test_whitespace_only_strings_dropped(self) -> None:
        assert extract_error_messages({"err": ["   ", "", "ok"]}) == ["ok"]

    def test_strips_whitespace(self) -> None:
        assert extract_error_messages({"error": "  padded  "}) == ["padded"]

    def test_none_values_ignored(self) -> None:
        assert extract_error_messages({"err": None, "error": [None, "x"]}) == ["x"]

    def test_non_string_leaves_are_stringified(self) -> None:
        assert extract_error_messages({"err": [404, 1.5]}) == ["404", "1.5"]

    def test_error_object_without_known_keys_is_serialised(self) -> None:
        assert extract_error_messages({"errors": [{"code": 5, "a": 1}]}) == ['{"a": 1, "code": 5}']

    def test_error_object_with_several_known_keys(self) -> None:
        body = {"error": {"message": "m", "detail": "d", "code": 1}}
        assert extract_error_messages(body) == ["m", "d"]

    @pytest.mark.parametrize("body", [None, 42, 1.5, True, object()])
    def test_non_dict_non_list_bodies_yield_nothing(self, body: Any) -> None:
        assert extract_error_messages(body) == []

    @pytest.mark.parametrize("body", [{}, {"err": []}, {"result": "OK"}, [], ""])
    def test_bodies_without_errors_yield_empty_list(self, body: Any) -> None:
        assert extract_error_messages(body) == []


# --------------------------------------------------------------------------- body parsing


class TestParseBody:
    def test_json_object(self) -> None:
        assert parse_body(_response(400, json={"err": ["x"]})) == {"err": ["x"]}

    def test_json_array(self) -> None:
        assert parse_body(_response(200, json=[1, 2])) == [1, 2]

    def test_text(self) -> None:
        assert parse_body(_response(502, text="Bad Gateway")) == "Bad Gateway"

    def test_empty_body_is_none(self) -> None:
        assert parse_body(_response(204)) is None
        assert parse_body(_response(500, content=b"")) is None

    def test_json_looking_text_is_parsed_despite_content_type(self) -> None:
        response = _response(400, content=b'{"a": 1}', headers={"content-type": "text/plain"})
        assert parse_body(response) == {"a": 1}

    def test_invalid_json_with_json_content_type_returns_text(self) -> None:
        response = _response(
            500, content=b"{not json", headers={"content-type": "application/json"}
        )
        assert parse_body(response) == "{not json"

    def test_json_content_type_variants(self) -> None:
        response = _response(
            400, content=b'{"x": 1}', headers={"content-type": "application/problem+json"}
        )
        assert parse_body(response) == {"x": 1}

    def test_html_body_returned_as_text(self) -> None:
        response = _response(
            503, content=b"<html>down</html>", headers={"content-type": "text/html"}
        )
        assert parse_body(response) == "<html>down</html>"


class TestParseRetryAfter:
    def test_missing_header(self) -> None:
        assert parse_retry_after(_response(429)) is None

    def test_empty_header(self) -> None:
        assert parse_retry_after(_response(429, headers={"Retry-After": ""})) is None

    def test_integer_seconds(self) -> None:
        assert parse_retry_after(_response(429, headers={"Retry-After": "5"})) == 5.0

    def test_fractional_seconds(self) -> None:
        assert parse_retry_after(_response(429, headers={"Retry-After": "2.5"})) == 2.5

    def test_negative_clamped_to_zero(self) -> None:
        assert parse_retry_after(_response(429, headers={"Retry-After": "-3"})) == 0.0

    def test_header_lookup_is_case_insensitive(self) -> None:
        assert parse_retry_after(_response(429, headers={"retry-after": "7"})) == 7.0

    def test_http_date_in_future(self) -> None:
        when = dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(seconds=60)
        header = format_datetime(when, usegmt=True)
        value = parse_retry_after(_response(503, headers={"Retry-After": header}))
        assert value is not None
        assert 55.0 <= value <= 61.0

    def test_http_date_in_past_is_zero(self) -> None:
        header = "Wed, 21 Oct 2015 07:28:00 GMT"
        assert parse_retry_after(_response(503, headers={"Retry-After": header})) == 0.0

    @pytest.mark.parametrize("garbage", ["soon", "five seconds", "1s", "2025-01-01"])
    def test_garbage_is_none(self, garbage: str) -> None:
        assert parse_retry_after(_response(429, headers={"Retry-After": garbage})) is None


# --------------------------------------------------------------------------- error mapping


class TestErrorForResponse:
    @pytest.mark.parametrize(
        ("status", "cls"),
        [
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (429, RateLimitError),
            (500, ServerError),
            (502, ServerError),
            (503, ServerError),
            (599, ServerError),
            (418, RainbirdAPIError),
            (409, RainbirdAPIError),
        ],
    )
    def test_status_maps_to_exact_class(self, status: int, cls: type[RainbirdAPIError]) -> None:
        err = error_for_response(_response(status, json={"err": ["x"]}))
        assert type(err) is cls
        assert err.status_code == status
        assert isinstance(err, RainbirdError)

    def test_single_err_message(self) -> None:
        err = error_for_response(_response(400, json={"err": ["Please provide a relationship"]}))
        assert err.message == "Please provide a relationship"
        assert err.errors == ["Please provide a relationship"]
        assert err.body == {"err": ["Please provide a relationship"]}

    def test_multiple_messages_joined_with_semicolon(self) -> None:
        err = error_for_response(_response(400, json={"err": ["one", "two"]}))
        assert err.message == "one; two"
        assert err.errors == ["one", "two"]

    def test_error_object_shape(self) -> None:
        err = error_for_response(_response(403, json={"error": {"message": "nope"}}))
        assert err.message == "nope"

    def test_message_shape(self) -> None:
        err = error_for_response(_response(404, json={"message": "gone"}))
        assert err.message == "gone"

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, "Your request is missing a required parameter or a value is invalid"),
            (401, "Your API key is invalid"),
            (403, "Your API key does not permit access to this resource"),
            (404, "The resource (session, knowledge map or fact) could not be found"),
            (429, "Too many requests"),
            (500, "Rainbird server error"),
            (503, "Rainbird server error"),
        ],
    )
    def test_default_message_when_body_empty(self, status: int, expected: str) -> None:
        err = error_for_response(_response(status))
        assert err.message == expected
        assert err.errors == []
        assert err.body is None

    def test_default_message_uses_reason_phrase_for_unmapped_status(self) -> None:
        assert error_for_response(_response(418)).message == "I'm a teapot"

    def test_default_message_when_no_reason_phrase(self) -> None:
        assert error_for_response(_response(499)).message == "Request failed"

    def test_default_message_when_body_has_no_errors(self) -> None:
        err = error_for_response(_response(401, json={"result": "nope"}))
        assert err.message == "Your API key is invalid"
        assert err.body == {"result": "nope"}

    def test_plain_text_body_becomes_message(self) -> None:
        response = _response(
            502, content=b"<html>Bad Gateway</html>", headers={"content-type": "text/html"}
        )
        err = error_for_response(response)
        assert err.message == "Bad Gateway"
        assert err.errors == ["Bad Gateway"]
        assert err.body == "<html>Bad Gateway</html>"

    def test_text_body_whitespace_collapsed(self) -> None:
        err = error_for_response(_response(502, text="Bad\n\n   Gateway  \t here "))
        assert err.message == "Bad Gateway here"

    def test_long_text_body_truncated_to_300_chars(self) -> None:
        err = error_for_response(_response(502, text="x" * 1000))
        assert len(err.message) == 300
        assert err.message.endswith("...")
        assert err.errors == [err.message]
        assert err.body == "x" * 1000

    def test_long_json_message_truncated_to_300_chars(self) -> None:
        err = error_for_response(_response(400, json={"err": ["y" * 1000]}))
        assert len(err.message) == 300
        assert err.message.endswith("...")
        assert err.errors == ["y" * 1000]

    def test_redirect_message_mentions_location(self) -> None:
        response = _response(302, headers={"Location": "https://login.example/"})
        err = error_for_response(response)
        assert type(err) is RainbirdAPIError
        assert err.status_code == 302
        assert err.message == "unexpected redirect to https://login.example/"
        assert err.errors == []

    def test_redirect_without_location_header(self) -> None:
        assert error_for_response(_response(301)).message == "unexpected redirect to <unknown>"

    def test_whitespace_only_text_body_falls_back_to_default(self) -> None:
        err = error_for_response(_response(404, text="   "))
        assert err.message == "The resource (session, knowledge map or fact) could not be found"

    def test_retry_after_set_on_429(self) -> None:
        err = error_for_response(_response(429, headers={"Retry-After": "12"}))
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 12.0

    def test_retry_after_none_when_absent(self) -> None:
        assert error_for_response(_response(429)).retry_after is None

    def test_retry_after_set_on_other_statuses_too(self) -> None:
        err = error_for_response(_response(503, headers={"Retry-After": "3"}))
        assert err.retry_after == 3.0

    def test_method_and_url_recorded(self) -> None:
        response = _response(404, method="POST", url=f"{BASE_URL}/sid/query")
        err = error_for_response(response)
        assert err.method == "POST"
        assert err.url == f"{BASE_URL}/sid/query"

    def test_str_includes_status_method_url_and_message(self) -> None:
        response = _response(
            404, json={"err": ["Session not found"]}, method="POST", url=f"{BASE_URL}/S/query"
        )
        assert str(error_for_response(response)) == (
            f"HTTP 404 calling POST {BASE_URL}/S/query: Session not found"
        )

    def test_str_without_method_and_url(self) -> None:
        assert str(RainbirdAPIError("boom", status_code=500)) == "HTTP 500: boom"

    def test_str_with_only_method_omits_location(self) -> None:
        err = RainbirdAPIError("boom", status_code=500, method="GET")
        assert str(err) == "HTTP 500: boom"

    def test_repr_format(self) -> None:
        response = _response(404, json={"err": ["Session not found"]})
        assert repr(error_for_response(response)) == (
            "NotFoundError(status_code=404, message='Session not found', "
            "errors=['Session not found'])"
        )

    def test_repr_for_base_class(self) -> None:
        err = error_for_response(_response(418, json={"error": "teapot"}))
        assert repr(err) == "RainbirdAPIError(status_code=418, message='teapot', errors=['teapot'])"

    def test_errors_attribute_is_independent_list(self) -> None:
        errors = ["a"]
        err = RainbirdAPIError("a", status_code=400, errors=errors)
        errors.append("b")
        assert err.errors == ["a"]

    def test_tolerates_response_without_attached_request(self) -> None:
        err = error_for_response(httpx.Response(404, json={"err": ["x"]}))
        assert err.method is None
        assert err.url is None


class TestTransportError:
    @pytest.mark.parametrize(
        "exc_type",
        [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout],
    )
    def test_timeouts_map_to_timeout_error(self, exc_type: type[httpx.TimeoutException]) -> None:
        err = transport_error(exc_type("slow"), "GET", f"{BASE_URL}/version")
        assert isinstance(err, RainbirdTimeoutError)
        assert isinstance(err, RainbirdTransportError)
        assert isinstance(err, RainbirdError)
        assert err.method == "GET"
        assert err.url == f"{BASE_URL}/version"
        assert "timed out" in str(err)
        assert f"GET {BASE_URL}/version" in str(err)
        assert "slow" in str(err)

    @pytest.mark.parametrize(
        "exc_type",
        [httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError],
    )
    def test_network_failures_map_to_connection_error(
        self, exc_type: type[httpx.TransportError]
    ) -> None:
        err = transport_error(exc_type("refused"), "POST", f"{BASE_URL}/S/query")
        assert isinstance(err, RainbirdConnectionError)
        assert not isinstance(err, RainbirdTimeoutError)
        assert err.method == "POST"
        assert err.url == f"{BASE_URL}/S/query"
        assert "could not reach" in str(err)
        assert f"POST {BASE_URL}/S/query" in str(err)
        assert "refused" in str(err)

    def test_unsupported_protocol_is_a_configuration_error(self) -> None:
        err = transport_error(httpx.UnsupportedProtocol("bad scheme"), "GET", "ftp://x/version")
        assert isinstance(err, RainbirdConfigurationError)
        assert not isinstance(err, RainbirdTransportError)
        assert "invalid base_url" in str(err)
        assert "GET ftp://x/version" in str(err)
        assert "bad scheme" in str(err)


class TestJsonBody:
    def test_returns_parsed_json(self) -> None:
        assert json_body(_response(200, json={"id": "S"})) == {"id": "S"}

    def test_non_json_2xx_raises_unexpected_response_error(self) -> None:
        response = _response(200, text="<html>login</html>", method="GET", url=f"{BASE_URL}/x")
        with pytest.raises(UnexpectedResponseError) as info:
            json_body(response)
        err = info.value
        assert type(err) is UnexpectedResponseError
        assert isinstance(err, RainbirdError)
        assert err.status_code == 200
        assert err.body == "<html>login</html>"
        assert err.message == (
            f"expected a JSON body from GET {BASE_URL}/x but got: '<html>login</html>'"
        )
        assert isinstance(err.__cause__, ValueError)

    def test_snippet_truncated_to_200_chars(self) -> None:
        with pytest.raises(UnexpectedResponseError) as info:
            json_body(_response(200, text="x" * 500))
        assert "x" * 200 in info.value.message
        assert "x" * 201 not in info.value.message
        assert info.value.body == "x" * 500

    def test_empty_2xx_raises(self) -> None:
        with pytest.raises(UnexpectedResponseError) as info:
            json_body(_response(200))
        assert info.value.message.endswith("but got: ''")
        assert info.value.status_code == 200

    def test_text_body_strips_whitespace(self) -> None:
        assert text_body(_response(200, text="  4.89.0\n")) == "4.89.0"

    def test_non_json_without_attached_request_omits_location(self) -> None:
        with pytest.raises(UnexpectedResponseError) as info:
            json_body(httpx.Response(200, text="nope"))
        assert info.value.message == "expected a JSON body but got: 'nope'"


# --------------------------------------------------------------------------- ClientConfig


ALL_ENV = {
    "RAINBIRD_API_KEY": "env-key",
    "RAINBIRD_API_URL": "https://env-api.rainbird.ai",
    "RAINBIRD_EVIDENCE_KEY": "env-evidence",
    "RAINBIRD_INTERACTION_KEY": "env-interaction",
    "RAINBIRD_ENGINE": "env-engine",
}


class TestClientConfigResolve:
    def test_explicit_arguments_win_over_env(self) -> None:
        cfg = _resolve(
            api_key="explicit-key",
            base_url="https://explicit-api.rainbird.ai",
            evidence_key="explicit-evidence",
            interaction_key="explicit-interaction",
            engine="explicit-engine",
            env=ALL_ENV,
        )
        assert cfg.api_key == "explicit-key"
        assert cfg.base_url == "https://explicit-api.rainbird.ai"
        assert cfg.evidence_key == "explicit-evidence"
        assert cfg.interaction_key == "explicit-interaction"
        assert cfg.engine == "explicit-engine"

    def test_env_fallbacks_for_every_variable(self) -> None:
        cfg = _resolve(env=ALL_ENV)
        assert cfg.api_key == "env-key"
        assert cfg.base_url == "https://env-api.rainbird.ai"
        assert cfg.evidence_key == "env-evidence"
        assert cfg.interaction_key == "env-interaction"
        assert cfg.engine == "env-engine"

    def test_optional_values_default_to_none(self) -> None:
        cfg = _resolve(api_key="k")
        assert cfg.evidence_key is None
        assert cfg.interaction_key is None
        assert cfg.engine is None

    def test_reads_os_environ_when_env_not_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in ALL_ENV.items():
            monkeypatch.setenv(name, value)
        cfg = _resolve(env=None)
        assert cfg.api_key == "env-key"
        assert cfg.base_url == "https://env-api.rainbird.ai"
        assert cfg.engine == "env-engine"

    def test_missing_key_raises_mentioning_env_var(self) -> None:
        with pytest.raises(RainbirdConfigurationError) as info:
            _resolve()
        assert "RAINBIRD_API_KEY" in str(info.value)
        assert "api_key" in str(info.value)
        assert isinstance(info.value, ValueError)
        assert isinstance(info.value, RainbirdError)

    @pytest.mark.parametrize("key", ["", "   ", "\t\n"])
    def test_blank_explicit_key_rejected(self, key: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            _resolve(api_key=key)

    @pytest.mark.parametrize("key", ["", "   "])
    def test_blank_env_key_rejected(self, key: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            _resolve(env={"RAINBIRD_API_KEY": key})

    def test_key_is_stripped(self) -> None:
        assert _resolve(api_key="  abc  ").api_key == "abc"

    def test_default_base_url_when_unset(self) -> None:
        cfg = _resolve(api_key="k")
        assert cfg.base_url == DEFAULT_BASE_URL == "https://api.rainbird.ai"

    @pytest.mark.parametrize("url", ["https://x-api.rainbird.ai/", "https://x-api.rainbird.ai///"])
    def test_trailing_slashes_stripped(self, url: str) -> None:
        assert _resolve(api_key="k", base_url=url).base_url == "https://x-api.rainbird.ai"

    def test_scheme_added_when_missing(self) -> None:
        cfg = _resolve(api_key="k", base_url="acme-api.rainbird.ai")
        assert cfg.base_url == "https://acme-api.rainbird.ai"

    def test_scheme_added_and_slash_stripped_together(self) -> None:
        cfg = _resolve(api_key="k", base_url="acme-api.rainbird.ai/")
        assert cfg.base_url == "https://acme-api.rainbird.ai"

    def test_http_scheme_preserved(self) -> None:
        cfg = _resolve(api_key="k", base_url="http://localhost:8080/")
        assert cfg.base_url == "http://localhost:8080"

    def test_base_url_whitespace_stripped(self) -> None:
        cfg = _resolve(api_key="k", base_url="  https://x-api.rainbird.ai  ")
        assert cfg.base_url == "https://x-api.rainbird.ai"

    def test_empty_env_base_url_falls_back_to_default(self) -> None:
        cfg = _resolve(env={"RAINBIRD_API_KEY": "k", "RAINBIRD_API_URL": ""})
        assert cfg.base_url == DEFAULT_BASE_URL

    def test_env_base_url_is_normalised(self) -> None:
        env = {"RAINBIRD_API_KEY": "k", "RAINBIRD_API_URL": "enterprise-api.rainbird.ai/"}
        assert _resolve(env=env).base_url == "https://enterprise-api.rainbird.ai"

    @pytest.mark.parametrize("url", ["", "   "])
    def test_empty_explicit_base_url_falls_back_to_default(self, url: str) -> None:
        assert _resolve(api_key="k", base_url=url).base_url == DEFAULT_BASE_URL

    @pytest.mark.parametrize("url", ["", "   "])
    def test_empty_explicit_base_url_falls_back_to_env(self, url: str) -> None:
        env = {"RAINBIRD_API_KEY": "k", "RAINBIRD_API_URL": "https://env-api.rainbird.ai"}
        assert _resolve(base_url=url, env=env).base_url == "https://env-api.rainbird.ai"

    @pytest.mark.parametrize("url", ["https:///path", "/"])
    def test_base_url_without_host_rejected(self, url: str) -> None:
        with pytest.raises(RainbirdConfigurationError, match="has no host"):
            _resolve(api_key="k", base_url=url)

    def test_invalid_base_url_rejected(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="not a valid URL"):
            _resolve(api_key="k", base_url="https://[::1")

    def test_default_headers_none_becomes_empty_dict(self) -> None:
        assert _resolve(api_key="k").default_headers == {}

    def test_default_headers_copied(self) -> None:
        headers = {"X-Tenant": "acme"}
        cfg = _resolve(api_key="k", default_headers=headers)
        headers["X-Tenant"] = "changed"
        assert cfg.default_headers == {"X-Tenant": "acme"}

    def test_retry_defaults_to_retry_policy(self) -> None:
        assert _resolve(api_key="k").retry == RetryPolicy()

    def test_retry_passthrough(self) -> None:
        policy = RetryPolicy(max_retries=7)
        assert _resolve(api_key="k", retry=policy).retry is policy

    def test_timeout_passthrough(self) -> None:
        timeout = httpx.Timeout(5.0, connect=1.0)
        assert _resolve(api_key="k", timeout=timeout).timeout is timeout
        assert _resolve(api_key="k", timeout=None).timeout is None
        assert _resolve(api_key="k", timeout=12.5).timeout == 12.5

    def test_default_timeout_is_30_seconds(self) -> None:
        assert DEFAULT_TIMEOUT == 30.0
        assert _config().timeout == 30.0

    def test_timeout_defaults_to_unset_sentinel(self) -> None:
        cfg = _resolve(api_key="k")
        assert cfg.timeout == DEFAULT_TIMEOUT
        assert cfg.timeout_explicit is False
        assert _config().timeout_explicit is False

    @pytest.mark.parametrize("timeout", [5.0, None, httpx.Timeout(1.0)])
    def test_explicit_timeout_is_recorded_as_explicit(self, timeout: Any) -> None:
        assert _resolve(api_key="k", timeout=timeout).timeout_explicit is True

    def test_unset_sentinel_is_falsy_singleton(self) -> None:
        assert not UNSET
        assert repr(UNSET) == "UNSET"
        assert type(UNSET)() is UNSET

    def test_require_api_key_false_permits_missing_key(self) -> None:
        cfg = _resolve(require_api_key=False)
        assert cfg.api_key == ""
        assert "X-API-Key" not in cfg.base_headers()

    def test_require_api_key_false_still_uses_key_when_given(self) -> None:
        assert _resolve(api_key="k", require_api_key=False).api_key == "k"

    def test_config_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            _config().api_key = "other"  # type: ignore[misc]


class TestBaseHeaders:
    def test_only_user_agent_and_api_key_by_default(self) -> None:
        assert _config().base_headers() == {"User-Agent": USER_AGENT, "X-API-Key": API_KEY}

    def test_user_agent_format(self) -> None:
        assert USER_AGENT == f"rainbird-ai-python/{__version__} httpx/{httpx.__version__}"

    def test_engine_header_when_set(self) -> None:
        assert _config(engine="alt").base_headers()["x-rainbird-engine"] == "alt"

    @pytest.mark.parametrize("engine", [None, ""])
    def test_engine_header_absent_when_unset_or_blank(self, engine: str | None) -> None:
        assert "x-rainbird-engine" not in _config(engine=engine).base_headers()

    def test_default_headers_merged(self) -> None:
        headers = _config(default_headers={"X-Tenant": "acme"}).base_headers()
        assert headers["X-Tenant"] == "acme"
        assert headers["X-API-Key"] == API_KEY
        assert headers["User-Agent"] == USER_AGENT

    def test_default_headers_override_builtins(self) -> None:
        headers = _config(default_headers={"User-Agent": "custom/1"}).base_headers()
        assert headers["User-Agent"] == "custom/1"

    def test_default_headers_override_builtins_case_insensitively(self) -> None:
        headers = _config(default_headers={"x-api-key": "other"}).base_headers()
        assert headers == {"User-Agent": USER_AGENT, "x-api-key": "other"}

    def test_returns_fresh_dict_each_call(self) -> None:
        cfg = _config()
        first = cfg.base_headers()
        first["X-Mutated"] = "yes"
        assert "X-Mutated" not in cfg.base_headers()


class TestMergeHeaders:
    def test_later_layers_win_case_insensitively_and_keep_their_spelling(self) -> None:
        merged = merge_headers({"X-API-Key": "a", "Accept": "*/*"}, {"x-api-key": "b"})
        assert merged == {"x-api-key": "b", "Accept": "*/*"}

    def test_none_and_empty_layers_ignored(self) -> None:
        assert merge_headers(None, {"A": "1"}, {}, None) == {"A": "1"}

    def test_no_layers_gives_empty_dict(self) -> None:
        assert merge_headers() == {}

    def test_inputs_not_mutated(self) -> None:
        base = {"A": "1"}
        merge_headers(base, {"a": "2"})
        assert base == {"A": "1"}


class TestNormalizeBaseUrl:
    def test_explicit_value_wins_over_env(self) -> None:
        env = {"RAINBIRD_API_URL": "https://env-api.rainbird.ai"}
        assert normalize_base_url("x-api.rainbird.ai/", env) == "https://x-api.rainbird.ai"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_blank_value_uses_env_then_default(self, value: Any) -> None:
        env = {"RAINBIRD_API_URL": "env-api.rainbird.ai/"}
        assert normalize_base_url(value, env) == "https://env-api.rainbird.ai"
        assert normalize_base_url(value, {}) == DEFAULT_BASE_URL

    def test_invalid_url_raises_configuration_error(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="not a valid URL"):
            normalize_base_url("https://[::1", {})

    @pytest.mark.parametrize("value", ["https:///path", "/"])
    def test_url_without_host_raises_configuration_error(self, value: str) -> None:
        with pytest.raises(RainbirdConfigurationError, match="has no host"):
            normalize_base_url(value, {})

    @pytest.mark.parametrize("value", ["https://", "http://", "HTTPS://", "https:", "http:/host"])
    def test_scheme_without_host_is_rejected(self, value: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            normalize_base_url(value, {})

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("HTTPS://API.rainbird.ai", "https://api.rainbird.ai"),
            ("Http://localhost:8000/", "http://localhost:8000"),
            ("localhost:8080", "https://localhost:8080"),
        ],
    )
    def test_scheme_is_matched_case_insensitively(self, value: str, expected: str) -> None:
        assert normalize_base_url(value, {}) == expected

    @pytest.mark.parametrize("value", ["ftp://host", "https://host/api?x=1", "https://host#frag"])
    def test_non_http_scheme_query_or_fragment_is_rejected(self, value: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            normalize_base_url(value, {})


# --------------------------------------------------------------------------- URL helpers


class TestCleanParams:
    @pytest.mark.parametrize("params", [None, {}, {"a": None}, {"a": None, "b": None}])
    def test_nothing_to_send_returns_none(self, params: Any) -> None:
        assert clean_params(params) is None

    def test_drops_none_values(self) -> None:
        assert clean_params({"a": None, "b": "x"}) == {"b": "x"}

    def test_booleans_rendered_lowercase(self) -> None:
        assert clean_params({"useDraft": True, "other": False}) == {
            "useDraft": "true",
            "other": "false",
        }

    def test_lists_joined_with_commas(self) -> None:
        assert clean_params({"relationships": ["speaks", "lives in"]}) == {
            "relationships": "speaks,lives in"
        }

    def test_tuples_and_sets_joined(self) -> None:
        assert clean_params({"a": ("x", "y"), "b": {"z"}}) == {"a": "x,y", "b": "z"}

    def test_list_items_stringified(self) -> None:
        assert clean_params({"v": [1, 2.5, True]}) == {"v": "1,2.5,True"}

    def test_numbers_stringified(self) -> None:
        assert clean_params({"version": 3, "ratio": 0.5}) == {"version": "3", "ratio": "0.5"}

    def test_strings_untouched(self) -> None:
        assert clean_params({"contextid": "ctx 1"}) == {"contextid": "ctx 1"}

    def test_empty_string_kept(self) -> None:
        assert clean_params({"a": ""}) == {"a": ""}


class TestPathSegment:
    def test_encodes_spaces(self) -> None:
        assert path_segment("a b", "x") == "a%20b"

    def test_encodes_slashes(self) -> None:
        assert path_segment("a/b", "x") == "a%2Fb"

    def test_encodes_reserved_characters(self) -> None:
        assert path_segment("a?b&c#d", "x") == "a%3Fb%26c%23d"

    def test_encodes_colons_in_fact_ids(self) -> None:
        assert path_segment("WA:RF:abc", "fact_id") == "WA%3ARF%3Aabc"

    def test_encodes_unicode(self) -> None:
        assert path_segment("café", "x") == "caf%C3%A9"

    def test_safe_characters_untouched(self) -> None:
        assert path_segment("abc-123_.~XYZ", "x") == "abc-123_.~XYZ"

    def test_strips_surrounding_whitespace(self) -> None:
        assert path_segment("  sid  ", "x") == "sid"

    def test_non_string_values_stringified(self) -> None:
        assert path_segment(42, "version") == "42"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_empty_rejected_with_name_in_message(self, value: Any) -> None:
        with pytest.raises(RainbirdConfigurationError, match="session_id must not be empty"):
            path_segment(value, "session_id")


class TestBuildUrl:
    def test_joins_with_single_slash(self) -> None:
        assert (
            build_url("https://api.rainbird.ai", "start/km") == "https://api.rainbird.ai/start/km"
        )

    def test_strips_duplicate_slashes_at_join(self) -> None:
        assert build_url("https://api.rainbird.ai/", "/start/km") == (
            "https://api.rainbird.ai/start/km"
        )

    def test_multiple_slashes(self) -> None:
        assert build_url("https://x///", "///version") == "https://x/version"

    def test_preserves_path_prefix_in_base(self) -> None:
        assert build_url("https://x/api/v1", "version") == "https://x/api/v1/version"


class TestRequest:
    def test_defaults(self) -> None:
        req = Request("GET", "version", parse=text_body)
        assert req.params is None
        assert req.json is None
        assert req.headers == {}
        assert req.accept == "application/json"

    def test_with_headers_merges_and_returns_new_request(self) -> None:
        req = Request("POST", "nl/interact", parse=json_body, headers={"Version": "v1"}, json={})
        merged = req.with_headers({"x-extra": "1", "Version": "v2"})
        assert merged is not req
        assert merged.headers == {"Version": "v2", "x-extra": "1"}
        assert req.headers == {"Version": "v1"}
        assert merged.method == "POST"
        assert merged.path == "nl/interact"
        assert merged.parse is json_body
        assert merged.json == {}
        assert merged.accept == req.accept

    def test_with_headers_merges_case_insensitively(self) -> None:
        req = Request("GET", "version", parse=text_body, headers={"X-Trace": "a", "Accept": "*/*"})
        assert req.with_headers({"x-trace": "b"}).headers == {"x-trace": "b", "Accept": "*/*"}

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            Request("GET", "version", parse=text_body).path = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- request preparation


class TestSyncTransportPrepare:
    def test_sends_api_key_user_agent_and_accept(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        assert fake_transport.send(Request("GET", "version", parse=text_body)) == "4.89.0"
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["user-agent"] == USER_AGENT
        assert fake.last.headers["accept"] == "application/json"
        assert "x-rainbird-engine" not in fake.last.headers

    def test_custom_accept_header(self, fake_transport: SyncTransport, fake: FakeRainbird) -> None:
        fake_transport.send(Request("GET", "version", parse=text_body, accept="text/plain"))
        assert fake.last.headers["accept"] == "text/plain"

    def test_request_headers_then_extra_headers_override(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        req = Request("GET", "version", parse=text_body, headers={"X-Req": "1", "Accept": "*/*"})
        fake_transport.send(req, extra_headers={"X-Req": "2", "X-Extra": "3"})
        assert fake.last.headers["x-req"] == "2"
        assert fake.last.headers["x-extra"] == "3"
        assert fake.last.headers["accept"] == "*/*"

    def test_engine_and_default_headers_sent(self, fake: FakeRainbird) -> None:
        config = _config(engine="alt-engine", default_headers={"X-Tenant": "acme"})
        with httpx.Client(transport=httpx.MockTransport(fake.handle)) as http:
            SyncTransport(config, http).send(Request("GET", "version", parse=text_body))
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"
        assert fake.last.headers["x-tenant"] == "acme"

    def test_params_cleaned_before_sending(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        req = Request(
            "GET",
            f"start/{KM_ID}",
            parse=json_body,
            params={"useDraft": True, "version": None, "contextid": "ctx 1"},
        )
        body = fake_transport.send(req)
        assert body["id"] in fake.sessions
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {"useDraft": "true", "contextid": "ctx 1"}

    def test_json_body_sent_verbatim(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        sid = fake_transport.send(Request("GET", f"start/{KM_ID}", parse=json_body))["id"]
        facts = [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
        ]
        result = fake_transport.send(Request("POST", f"{sid}/inject", parse=json_body, json=facts))
        assert result == {"result": "OK"}
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{sid}/inject"
        assert fake.last.json == facts
        assert len(fake.calls("/inject")) == 1

    def test_method_is_uppercased(self, fake_transport: SyncTransport, fake: FakeRainbird) -> None:
        fake_transport.send(Request("get", "version", parse=text_body))
        assert fake.last.method == "GET"

    def test_leading_slash_in_path_tolerated(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        fake_transport.send(Request("GET", "/version", parse=text_body))
        assert fake.last.path == "/version"

    def test_encoded_segments_reach_the_server(self) -> None:
        handler = ScriptedHandler()
        transport = _sync(handler)
        fact, sid = path_segment("WA:RF:1", "fact_id"), path_segment("s 1", "session_id")
        transport.send(Request("GET", f"analysis/evidence/{fact}/{sid}", parse=json_body))
        sent = handler.requests[0].url
        assert sent.raw_path == b"/analysis/evidence/WA%3ARF%3A1/s%201"
        assert sent.path == "/analysis/evidence/WA:RF:1/s 1"

    def test_url_built_from_base_url(self) -> None:
        handler = ScriptedHandler()
        _sync(handler).send(GET_THINGS)
        assert str(handler.requests[0].url) == f"{BASE_URL}/things"

    def test_api_errors_from_fake_are_mapped(
        self, fake_transport: SyncTransport, fake: FakeRainbird
    ) -> None:
        fake.fail_next(404, {"err": ["Session not found"]})
        with pytest.raises(NotFoundError) as info:
            fake_transport.send(Request("POST", "nope/query", parse=json_body, json={}))
        assert info.value.errors == ["Session not found"]
        assert info.value.url == f"{BASE_URL}/nope/query"
        assert info.value.method == "POST"

    def test_explicit_timeout_applied_per_request_on_user_client(self) -> None:
        handler = ScriptedHandler()
        http = httpx.Client(transport=httpx.MockTransport(handler), timeout=99.0)
        SyncTransport(_config(timeout=5.0, timeout_explicit=True), http).send(GET_THINGS)
        assert handler.requests[0].extensions["timeout"] == httpx.Timeout(5.0).as_dict()

    def test_default_timeout_leaves_user_client_timeout_alone(self) -> None:
        handler = ScriptedHandler()
        http = httpx.Client(transport=httpx.MockTransport(handler), timeout=99.0)
        SyncTransport(_config(), http).send(GET_THINGS)
        assert handler.requests[0].extensions["timeout"] == httpx.Timeout(99.0).as_dict()

    def test_default_headers_replace_api_key_case_insensitively(self) -> None:
        handler = ScriptedHandler()
        config = _config(default_headers={"x-api-key": "override"})
        http = httpx.Client(transport=httpx.MockTransport(handler))
        SyncTransport(config, http).send(GET_THINGS)
        assert handler.requests[0].headers.get_list("x-api-key") == ["override"]


# --------------------------------------------------------------------------- sync retry loop


class TestSyncTransportRetries:
    def test_get_retries_5xx_then_succeeds(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503, 502)
        assert _sync(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 3
        assert len(sleeps) == 2
        assert all(delay > 0 for delay in sleeps)

    def test_get_retries_429(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(429)
        assert _sync(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 2

    def test_get_gives_up_after_max_retries_and_raises_last_error(
        self, sleeps: list[float]
    ) -> None:
        handler = ScriptedHandler(503, 503, 503, 503)
        with pytest.raises(ServerError) as info:
            _sync(handler).send(GET_THINGS)
        assert handler.calls == 3  # 1 attempt + 2 retries
        assert len(sleeps) == 2
        assert info.value.status_code == 503
        assert info.value.errors == ["fail 503"]
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/things"

    def test_max_retries_controls_attempt_count(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(500, 500, 500, 500, 500)
        with pytest.raises(ServerError):
            _sync(handler, RetryPolicy(max_retries=4)).send(GET_THINGS)
        assert handler.calls == 5
        assert len(sleeps) == 4

    def test_get_retries_transport_errors_then_succeeds(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"), httpx.ReadTimeout("slow"))
        assert _sync(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 3
        assert len(sleeps) == 2

    def test_get_connect_errors_exhausted_raise_connection_error(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(*[httpx.ConnectError("refused")] * 5)
        with pytest.raises(RainbirdConnectionError) as info:
            _sync(handler).send(GET_THINGS)
        assert handler.calls == 3
        assert isinstance(info.value.__cause__, httpx.ConnectError)
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/things"

    def test_get_timeouts_exhausted_raise_timeout_error(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(*[httpx.ReadTimeout("slow")] * 5)
        with pytest.raises(RainbirdTimeoutError) as info:
            _sync(handler).send(GET_THINGS)
        assert handler.calls == 3
        assert isinstance(info.value.__cause__, httpx.ReadTimeout)
        assert "timed out" in str(info.value)

    def test_post_does_not_retry_transport_errors(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"))
        with pytest.raises(RainbirdConnectionError):
            _sync(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    def test_post_does_not_retry_timeouts(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ReadTimeout("slow"))
        with pytest.raises(RainbirdTimeoutError):
            _sync(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    def test_post_retries_transport_errors_when_opted_in(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"))
        policy = RetryPolicy(retry_post_on_transport_error=True)
        assert _sync(handler, policy).send(POST_THINGS) == {"ok": True}
        assert handler.calls == 2
        assert len(sleeps) == 1

    def test_post_does_not_retry_500(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(500)
        with pytest.raises(ServerError):
            _sync(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    @pytest.mark.parametrize("status", [429, 503])
    def test_post_retries_refused_statuses(self, status: int, sleeps: list[float]) -> None:
        handler = ScriptedHandler(status)
        assert _sync(handler).send(POST_THINGS) == {"ok": True}
        assert handler.calls == 2
        assert len(sleeps) == 1

    @pytest.mark.parametrize("status", [502, 504])
    def test_post_does_not_retry_gateway_errors(self, status: int, sleeps: list[float]) -> None:
        handler = ScriptedHandler(status)
        with pytest.raises(ServerError) as info:
            _sync(handler).send(POST_THINGS)
        assert info.value.status_code == status
        assert handler.calls == 1
        assert sleeps == []

    @pytest.mark.parametrize("status", [502, 504])
    def test_get_still_retries_gateway_errors(self, status: int, sleeps: list[float]) -> None:
        handler = ScriptedHandler(status)
        assert _sync(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 2

    def test_redirect_is_surfaced_not_followed_or_retried(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(
            httpx.Response(302, headers={"Location": "https://login.example/"})
        )
        with pytest.raises(RainbirdAPIError) as info:
            _sync(handler).send(GET_THINGS)
        assert type(info.value) is RainbirdAPIError
        assert info.value.status_code == 302
        assert info.value.message == "unexpected redirect to https://login.example/"
        assert handler.calls == 1
        assert sleeps == []

    def test_unsupported_protocol_is_configuration_error_and_not_retried(
        self, sleeps: list[float]
    ) -> None:
        handler = ScriptedHandler(httpx.UnsupportedProtocol("unsupported protocol 'ftp://'"))
        with pytest.raises(RainbirdConfigurationError, match="invalid base_url"):
            _sync(handler).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    def test_invalid_url_is_configuration_error_and_not_retried(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.InvalidURL("bad url"))
        with pytest.raises(RainbirdConfigurationError, match="invalid URL"):
            _sync(handler).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_not_retried(self, status: int, sleeps: list[float]) -> None:
        handler = ScriptedHandler(status)
        with pytest.raises(RainbirdAPIError) as info:
            _sync(handler).send(GET_THINGS)
        assert info.value.status_code == status
        assert handler.calls == 1
        assert sleeps == []

    def test_retry_after_header_sets_delay(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(_fail(429, {"Retry-After": "7"}))
        assert _sync(handler).send(POST_THINGS) == {"ok": True}
        assert sleeps == [7.0]

    def test_retry_after_capped_by_max_backoff(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(_fail(503, {"Retry-After": "1000"}))
        _sync(handler, RetryPolicy(max_backoff=4.0)).send(GET_THINGS)
        assert sleeps == [4.0]

    def test_exhausted_429_carries_retry_after(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(*[_fail(429, {"Retry-After": "3"}) for _ in range(3)])
        with pytest.raises(RateLimitError) as info:
            _sync(handler).send(GET_THINGS)
        assert info.value.retry_after == 3.0
        assert sleeps == [3.0, 3.0]

    def test_delays_grow_exponentially(self, sleeps: list[float], no_jitter: None) -> None:
        handler = ScriptedHandler(503, 503, 503)
        _sync(handler, RetryPolicy(max_retries=3, backoff_factor=0.5)).send(GET_THINGS)
        assert sleeps == [0.5, 1.0, 2.0]

    def test_no_retry_policy_raises_immediately(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503)
        with pytest.raises(ServerError):
            _sync(handler, NO_RETRY).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    def test_no_retry_policy_does_not_retry_transport_errors(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"))
        with pytest.raises(RainbirdConnectionError):
            _sync(handler, NO_RETRY).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    def test_parse_callback_applied_to_successful_response(self) -> None:
        handler = ScriptedHandler(httpx.Response(200, text="4.89.0\n"))
        assert _sync(handler).send(Request("GET", "version", parse=text_body)) == "4.89.0"

    def test_non_json_2xx_is_not_retried(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.Response(200, text="<html>login</html>"))
        with pytest.raises(UnexpectedResponseError) as info:
            _sync(handler).send(GET_THINGS)
        assert info.value.status_code == 200
        assert info.value.body == "<html>login</html>"
        assert handler.calls == 1
        assert sleeps == []

    def test_attempt_counter_resets_between_sends(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503, 503, httpx.Response(200, json={"n": 1}), 503, 503)
        transport = _sync(handler)
        assert transport.send(GET_THINGS) == {"n": 1}
        assert transport.send(GET_THINGS) == {"ok": True}
        assert handler.calls == 6
        assert len(sleeps) == 4

    def test_retried_requests_are_identical(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503)
        _sync(handler).send(POST_THINGS.with_headers({"X-Trace": "t"}))
        first, second = handler.requests
        assert first.method == second.method == "POST"
        assert first.url == second.url
        assert first.content == second.content == b'{"a":1}'
        assert first.headers["x-trace"] == second.headers["x-trace"] == "t"
        assert first.headers["x-api-key"] == second.headers["x-api-key"] == API_KEY


# --------------------------------------------------------------------------- async retry loop


@pytest.mark.anyio
class TestAsyncTransportRetries:
    async def test_get_retries_5xx_then_succeeds(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503, 502)
        assert await _async(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 3
        assert len(sleeps) == 2
        assert all(delay > 0 for delay in sleeps)

    async def test_get_gives_up_after_max_retries(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503, 503, 503, 503)
        with pytest.raises(ServerError) as info:
            await _async(handler).send(GET_THINGS)
        assert handler.calls == 3
        assert len(sleeps) == 2
        assert info.value.errors == ["fail 503"]
        assert info.value.url == f"{BASE_URL}/things"

    async def test_get_retries_transport_errors_then_succeeds(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"), httpx.ReadTimeout("slow"))
        assert await _async(handler).send(GET_THINGS) == {"ok": True}
        assert handler.calls == 3
        assert len(sleeps) == 2

    async def test_get_connect_errors_exhausted_raise_connection_error(
        self, sleeps: list[float]
    ) -> None:
        handler = ScriptedHandler(*[httpx.ConnectError("refused")] * 5)
        with pytest.raises(RainbirdConnectionError) as info:
            await _async(handler).send(GET_THINGS)
        assert handler.calls == 3
        assert isinstance(info.value.__cause__, httpx.ConnectError)
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/things"

    async def test_get_timeouts_exhausted_raise_timeout_error(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(*[httpx.ConnectTimeout("slow")] * 5)
        with pytest.raises(RainbirdTimeoutError) as info:
            await _async(handler).send(GET_THINGS)
        assert handler.calls == 3
        assert isinstance(info.value.__cause__, httpx.ConnectTimeout)

    async def test_post_does_not_retry_transport_errors(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ConnectError("refused"))
        with pytest.raises(RainbirdConnectionError):
            await _async(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_post_retries_transport_errors_when_opted_in(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.ReadTimeout("slow"))
        policy = RetryPolicy(retry_post_on_transport_error=True)
        assert await _async(handler, policy).send(POST_THINGS) == {"ok": True}
        assert handler.calls == 2

    async def test_post_does_not_retry_500(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(500)
        with pytest.raises(ServerError):
            await _async(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_post_retries_503_then_succeeds(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503)
        assert await _async(handler).send(POST_THINGS) == {"ok": True}
        assert handler.calls == 2
        assert len(sleeps) == 1

    @pytest.mark.parametrize("status", [502, 504])
    async def test_post_does_not_retry_gateway_errors(
        self, status: int, sleeps: list[float]
    ) -> None:
        handler = ScriptedHandler(status)
        with pytest.raises(ServerError):
            await _async(handler).send(POST_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_redirect_is_surfaced_not_followed(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.Response(307, headers={"Location": "https://elsewhere/"}))
        with pytest.raises(RainbirdAPIError) as info:
            await _async(handler).send(GET_THINGS)
        assert info.value.status_code == 307
        assert info.value.message == "unexpected redirect to https://elsewhere/"
        assert handler.calls == 1
        assert sleeps == []

    async def test_non_json_2xx_raises_unexpected_response_error(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(httpx.Response(200, text="nope"))
        with pytest.raises(UnexpectedResponseError):
            await _async(handler).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_retry_after_header_sets_delay(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(_fail(429, {"Retry-After": "7"}))
        assert await _async(handler).send(POST_THINGS) == {"ok": True}
        assert sleeps == [7.0]

    async def test_delays_grow_exponentially(self, sleeps: list[float], no_jitter: None) -> None:
        handler = ScriptedHandler(503, 503, 503)
        await _async(handler, RetryPolicy(max_retries=3, backoff_factor=0.5)).send(GET_THINGS)
        assert sleeps == [0.5, 1.0, 2.0]

    async def test_no_retry_policy_raises_immediately(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(503)
        with pytest.raises(ServerError):
            await _async(handler, NO_RETRY).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_client_errors_are_not_retried(self, sleeps: list[float]) -> None:
        handler = ScriptedHandler(404)
        with pytest.raises(NotFoundError):
            await _async(handler).send(GET_THINGS)
        assert handler.calls == 1
        assert sleeps == []

    async def test_headers_params_and_body_sent(self, fake: FakeRainbird) -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
        transport = AsyncTransport(_config(engine="alt"), http)
        req = Request(
            "GET",
            f"start/{KM_ID}",
            parse=json_body,
            params={"useDraft": False, "version": None},
            headers={"X-Req": "1"},
        )
        body = await transport.send(req, extra_headers={"X-Extra": "2"})
        assert body["id"] in fake.sessions
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {"useDraft": "false"}
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["user-agent"] == USER_AGENT
        assert fake.last.headers["accept"] == "application/json"
        assert fake.last.headers["x-rainbird-engine"] == "alt"
        assert fake.last.headers["x-req"] == "1"
        assert fake.last.headers["x-extra"] == "2"
        await http.aclose()


# --------------------------------------------------------------------------- client ownership


class TestOwnsClient:
    def test_sync_transport_creates_and_closes_its_own_client(self) -> None:
        transport = SyncTransport(_config(timeout=12.0))
        assert transport._owns_client is True
        assert isinstance(transport.http, httpx.Client)
        assert transport.http.timeout == httpx.Timeout(12.0)
        assert transport.http.follow_redirects is False
        transport.close()
        assert transport.http.is_closed

    def test_sync_transport_does_not_close_user_client(self) -> None:
        http = httpx.Client(transport=httpx.MockTransport(ScriptedHandler()))
        transport = SyncTransport(_config(), http)
        assert transport._owns_client is False
        assert transport.http is http
        transport.close()
        assert http.is_closed is False
        http.close()

    def test_rainbird_close_leaves_user_client_open(self, fake: FakeRainbird) -> None:
        http = httpx.Client(transport=httpx.MockTransport(fake.handle))
        rb = Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http)
        assert rb.http is http
        rb.close()
        assert http.is_closed is False
        assert rb.version() == "4.89.0"  # still usable
        http.close()

    def test_rainbird_context_manager_closes_owned_client(self) -> None:
        with Rainbird(api_key=API_KEY, base_url=BASE_URL) as rb:
            http = rb.http
            assert http.is_closed is False
        assert http.is_closed is True

    def test_rainbird_none_timeout_propagates(self) -> None:
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, timeout=None) as rb:
            assert rb.http.timeout == httpx.Timeout(None)

    def test_rainbird_owned_client_uses_default_timeout_and_no_redirects(self) -> None:
        with Rainbird(api_key=API_KEY, base_url=BASE_URL) as rb:
            assert rb.http.timeout == httpx.Timeout(DEFAULT_TIMEOUT)
            assert rb.http.follow_redirects is False

    @pytest.mark.anyio
    async def test_async_transport_creates_and_closes_its_own_client(self) -> None:
        transport = AsyncTransport(_config(timeout=7.0))
        assert transport._owns_client is True
        assert isinstance(transport.http, httpx.AsyncClient)
        assert transport.http.timeout == httpx.Timeout(7.0)
        assert transport.http.follow_redirects is False
        await transport.aclose()
        assert transport.http.is_closed

    @pytest.mark.anyio
    async def test_async_transport_does_not_close_user_client(self) -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(ScriptedHandler()))
        transport = AsyncTransport(_config(), http)
        assert transport._owns_client is False
        await transport.aclose()
        assert http.is_closed is False
        await http.aclose()

    @pytest.mark.anyio
    async def test_async_rainbird_aclose_leaves_user_client_open(self, fake: FakeRainbird) -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
        rb = AsyncRainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http)
        await rb.aclose()
        assert http.is_closed is False
        assert await rb.version() == "4.89.0"
        await http.aclose()

    @pytest.mark.anyio
    async def test_async_rainbird_context_manager_closes_owned_client(self) -> None:
        async with AsyncRainbird(api_key=API_KEY, base_url=BASE_URL) as rb:
            http = rb.http
            assert http.is_closed is False
        assert http.is_closed is True


# --------------------------------------------------------------------------- env through the clients


class TestClientEnvironment:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", "env-key")
        with Rainbird() as rb:
            assert rb.config.api_key == "env-key"
            assert rb.base_url == DEFAULT_BASE_URL

    def test_all_client_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in ALL_ENV.items():
            monkeypatch.setenv(name, value)
        with Rainbird() as rb:
            assert rb.config.api_key == "env-key"
            assert rb.base_url == "https://env-api.rainbird.ai"
            assert rb.config.evidence_key == "env-evidence"
            assert rb.config.interaction_key == "env-interaction"
            assert rb.config.engine == "env-engine"

    def test_explicit_arguments_beat_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in ALL_ENV.items():
            monkeypatch.setenv(name, value)
        with Rainbird(api_key="explicit", base_url="explicit-api.rainbird.ai/", engine="e") as rb:
            assert rb.config.api_key == "explicit"
            assert rb.base_url == "https://explicit-api.rainbird.ai"
            assert rb.config.engine == "e"
            assert rb.config.evidence_key == "env-evidence"

    def test_empty_base_url_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_API_URL", "https://env-api.rainbird.ai")
        with Rainbird(api_key="k", base_url="") as rb:
            assert rb.base_url == "https://env-api.rainbird.ai"

    def test_missing_key_raises(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="RAINBIRD_API_KEY"):
            Rainbird()

    def test_async_client_missing_key_raises(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="RAINBIRD_API_KEY"):
            AsyncRainbird()

    def test_km_id_from_env_used_by_start(
        self, client: Rainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", KM_ID)
        session = client.start()
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {}
        assert session.km_id == KM_ID
        assert session.id in fake.sessions

    def test_start_without_km_id_raises_mentioning_env_var(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="RAINBIRD_KM_ID"):
            client.start()
        assert fake.requests == []

    def test_engine_from_env_is_sent(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_ENGINE", "env-engine")
        http = httpx.Client(transport=httpx.MockTransport(fake.handle))
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http) as rb:
            rb.version()
        assert fake.last.headers["x-rainbird-engine"] == "env-engine"
        http.close()
