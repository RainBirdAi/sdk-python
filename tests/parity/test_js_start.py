"""Port of the JavaScript base SDK tests for ``start`` and the exported constants.

Sources (``/tmp/rb-sdk-js/packages/base/lib``):

* ``__tests__/start.test.js`` - ``describe("start")``: 9 cases (endpoint + parsing, engine
  option, context option, useDraft option, context-and-useDraft option, fetch-generated errors,
  404, 500, 400). Implementation under test: ``start.ts``.
* ``__tests__/index.test.js`` - ``describe("constants")``: 4 cases (``URL_COMMUNITY``,
  ``URL_ENTERPRISE``, ``ENGINE_CORE``, ``ENGINE_BETA``). Implementation: ``index.ts``.

Python counterparts: ``Rainbird(api_key=, base_url=, engine=).start(km_id, use_draft=,
context_id=)`` returning a :class:`~rainbird_ai.Session`, and ``rainbird_ai.Environment`` /
``rainbird_ai.DEFAULT_BASE_URL``.

Differences found (each is flagged inline with a ``# DIFFERENCE(js):`` comment):

* Auth: JS sends ``Authorization: Basic base64(apiKey + ":")``; Python sends ``X-API-Key`` and
  never an ``Authorization`` header.
* Headers: JS sends ``Content-Type: application/json`` on the (body-less) GET; Python sends no
  ``Content-Type`` on GET but adds ``Accept: application/json`` and a ``User-Agent``.
* Base URL: JS concatenates the string verbatim (``thisistheurl/start/...``); Python normalises
  a bare host to ``https://thisistheurl``.
* Context option: JS sends ``?contextId=...`` (and ``contextId`` before ``useDraft``); Python
  sends ``contextid`` (lower case, as the Go SDK and the API docs do) after ``useDraft``.
* Engine option: JS passes ``{engine}`` per call; Python sets it once on the client
  (``Rainbird(engine=...)``) and sends ``x-rainbird-engine`` on every request.
* Return value: JS returns ``{sessionID, kmVersionID}``; Python returns a ``Session`` with
  ``.id``, ``.km_version_id`` and the full ``StartResponse`` on ``.start_response``.
* Errors: JS throws ``Error("API error code: N")``; Python raises typed ``RainbirdAPIError``
  subclasses (400 ``BadRequestError``, 404 ``NotFoundError``, 500 ``ServerError``) carrying
  ``status_code``, ``message``, ``body``, ``method`` and ``url``. A transport failure is
  wrapped in ``RainbirdConnectionError`` (``"could not reach GET <url>: <original>"``) instead
  of being re-thrown untouched.
* Constants: ``URL_COMMUNITY``/``URL_ENTERPRISE`` map to ``Environment.COMMUNITY``/
  ``Environment.ENTERPRISE`` (and ``DEFAULT_BASE_URL``); ``ENGINE_CORE``/``ENGINE_BETA`` exist
  with the same values since 1.1.0 (strict xfails before); Python exports no ``RESPONSE_TYPE_*``
  (it uses ``is_question``/``is_result`` instead).
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

import rainbird_ai
from rainbird_ai import (
    DEFAULT_BASE_URL,
    BadRequestError,
    Environment,
    NotFoundError,
    Rainbird,
    RainbirdAPIError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTransportError,
    ServerError,
    Session,
    StartResponse,
)
from tests.parity.support import (
    Call,
    Handler,
    Recorder,
    make_client,
    raise_transport_error,
    respond,
)

# Arguments used verbatim by every JS case: start("thisistheurl", "thisisthekey", "thisisthekmid")
JS_BASE_URL = "thisistheurl"
JS_API_KEY = "thisisthekey"
JS_KM_ID = "thisisthekmid"
JS_CONTEXT = "thisisthecontext"
JS_ENGINE = "thisistheengine"

# Python normalises the bare host JS uses verbatim (see normalize_base_url in _http.py).
PY_BASE_URL = "https://thisistheurl"
START_PATH = f"/start/{JS_KM_ID}"

# What JS asserts in every start header set: Basic auth of "thisisthekey:".
JS_BASIC_AUTH = "Basic dGhpc2lzdGhla2V5Og=="

# Canned 200 body from start.test.js, copied verbatim.
START_BODY = {
    "id": "12345678-1234-1234-1234-1234567890ab",
    "kmVersion": {"id": "abcdefgh-abcd-abcd-abcd-abcdefghijkl"},
}
EXPECTED_SESSION_ID = "12345678-1234-1234-1234-1234567890ab"
EXPECTED_KM_VERSION_ID = "abcdefgh-abcd-abcd-abcd-abcdefghijkl"


def _start_client(handler: Handler, **kwargs: Any) -> tuple[Rainbird, Recorder]:
    """``start("thisistheurl", "thisisthekey", ...)`` as a Python client."""
    return make_client(handler, api_key=JS_API_KEY, base_url=JS_BASE_URL, **kwargs)


def _assert_start_headers(call: Call, *, engine: str | None = None) -> None:
    """The header assertions every JS start case makes, translated to Python's behaviour.

    JS asserts the header set is exactly ``{"Content-Type": "application/json",
    "Authorization": "Basic dGhpc2lzdGhla2V5Og=="}`` (plus ``x-rainbird-engine`` when given).
    """
    # Sanity check that the JS literal really is base64("thisisthekey:"), i.e. the same key.
    assert base64.b64decode(JS_BASIC_AUTH.split(" ", 1)[1]) == f"{JS_API_KEY}:".encode()

    # DIFFERENCE(js): JS authenticates with "Authorization: Basic <base64(apiKey:)>"; Python
    # sends the key as "X-API-Key" and never emits an Authorization header.
    assert call.headers["x-api-key"] == JS_API_KEY
    assert "authorization" not in call.headers
    assert all(JS_BASIC_AUTH not in value for value in call.headers.values())

    # DIFFERENCE(js): JS sends "Content-Type: application/json" on the body-less GET; Python
    # sends no Content-Type on a GET (there is no body) but does send "Accept: application/json"
    # and a User-Agent identifying the SDK.
    assert "content-type" not in call.headers
    assert call.headers["accept"] == "application/json"
    assert call.headers["user-agent"].startswith("rainbird-ai-python/")

    if engine is None:
        assert "x-rainbird-engine" not in call.headers
    else:
        assert call.headers["x-rainbird-engine"] == engine


def _assert_start_result(session: Session) -> None:
    """JS: ``expect(result).toEqual({kmVersionID: ..., sessionID: ...})``."""
    # DIFFERENCE(js): JS returns a plain {sessionID, kmVersionID} object; Python returns a
    # Session whose .id / .km_version_id carry the same values and whose .start_response keeps
    # the parsed StartResponse (with the untouched JSON on .raw).
    assert isinstance(session, Session)
    assert session.id == EXPECTED_SESSION_ID
    assert session.km_version_id == EXPECTED_KM_VERSION_ID
    assert session.km_id == JS_KM_ID
    assert isinstance(session.start_response, StartResponse)
    assert session.start_response.session_id == EXPECTED_SESSION_ID
    assert session.start_response.km_version_id == EXPECTED_KM_VERSION_ID
    assert session.start_response.raw == START_BODY
    assert session.start_response.to_dict() == START_BODY
    assert session.last_response is None


def _assert_get_start(recorder: Recorder, query: dict[str, str]) -> Call:
    """JS: ``expect(fetch).toHaveBeenCalledTimes(1)`` + ``toHaveBeenCalledWith(url, ...)``."""
    assert len(recorder) == 1
    call = recorder.last
    assert call.method == "GET"
    assert call.path == START_PATH
    assert call.query == query
    assert call.content == b""  # GET without a body, as in JS
    return call


# --------------------------------------------------------------------------- describe("start")


# source: start.test.js:11 should call appropriate endpoint and parse response
def test_should_call_appropriate_endpoint_and_parse_response() -> None:
    rb, recorder = _start_client(respond(200, START_BODY))

    session = rb.start(JS_KM_ID)

    call = _assert_get_start(recorder, {})
    # DIFFERENCE(js): JS asserts the URL "thisistheurl/start/thisisthekmid" (base URL used
    # verbatim); Python normalises the bare host to "https://thisistheurl".
    assert str(call.url) == f"{PY_BASE_URL}{START_PATH}"
    assert rb.base_url == PY_BASE_URL
    _assert_start_headers(call)
    _assert_start_result(session)
    assert session.context_id is None


# source: start.test.js:38 should handle engine option
def test_should_handle_engine_option() -> None:
    # DIFFERENCE(js): JS passes {engine: "thisistheengine"} to this one start() call; Python
    # configures the engine once on the client (Rainbird(engine=...)) and sends the
    # x-rainbird-engine header on every request, the start request included.
    rb, recorder = _start_client(respond(200, START_BODY), engine=JS_ENGINE)

    session = rb.start(JS_KM_ID)

    call = _assert_get_start(recorder, {})
    assert str(call.url) == f"{PY_BASE_URL}{START_PATH}"
    _assert_start_headers(call, engine=JS_ENGINE)
    _assert_start_result(session)


# source: start.test.js:71 should handle context option
def test_should_handle_context_option() -> None:
    rb, recorder = _start_client(respond(200, START_BODY))

    session = rb.start(JS_KM_ID, context_id=JS_CONTEXT, use_draft=True)

    # DIFFERENCE(js): JS builds "?contextId=thisisthecontext&useDraft=true" (camel-case key,
    # context first); Python sends the lower-case "contextid" key used by the Go SDK and the
    # API docs, and orders useDraft before contextid.
    call = _assert_get_start(recorder, {"useDraft": "true", "contextid": JS_CONTEXT})
    assert str(call.url) == f"{PY_BASE_URL}{START_PATH}?useDraft=true&contextid={JS_CONTEXT}"
    assert "contextId" not in call.query
    _assert_start_headers(call)
    _assert_start_result(session)
    assert session.context_id == JS_CONTEXT


# source: start.test.js:103 should handle useDraft option
def test_should_handle_useDraft_option() -> None:
    rb, recorder = _start_client(respond(200, START_BODY))

    session = rb.start(JS_KM_ID, use_draft=True)

    call = _assert_get_start(recorder, {"useDraft": "true"})
    assert str(call.url) == f"{PY_BASE_URL}{START_PATH}?useDraft=true"
    _assert_start_headers(call)
    _assert_start_result(session)
    assert session.context_id is None


# source: start.test.js:135 should handle context and useDraft option
def test_should_handle_context_and_useDraft_option() -> None:
    # Despite its name the JS case passes only {context: "thisisthecontext"} and asserts
    # "?contextId=thisisthecontext" with no useDraft; the port does the same.
    rb, recorder = _start_client(respond(200, START_BODY))

    session = rb.start(JS_KM_ID, context_id=JS_CONTEXT)

    # DIFFERENCE(js): "contextId" in JS, "contextid" in Python (see test_should_handle_context_option).
    call = _assert_get_start(recorder, {"contextid": JS_CONTEXT})
    assert str(call.url) == f"{PY_BASE_URL}{START_PATH}?contextid={JS_CONTEXT}"
    assert "useDraft" not in call.query
    _assert_start_headers(call)
    _assert_start_result(session)
    assert session.context_id == JS_CONTEXT


# source: start.test.js:167 should throw fetch-generated errors
def test_should_throw_fetch_generated_errors() -> None:
    handler = raise_transport_error(httpx.ConnectError("This was generated by fetch"))
    rb, recorder = _start_client(handler)

    with pytest.raises(RainbirdConnectionError) as info:
        rb.start(JS_KM_ID, context_id=JS_CONTEXT)

    # DIFFERENCE(js): JS re-throws the fetch error untouched (message === "This was generated
    # by fetch"); Python wraps it in RainbirdConnectionError whose message is
    # "could not reach GET <url>: <original message>" and exposes .method / .url.
    exc = info.value
    assert isinstance(exc, RainbirdTransportError)
    assert isinstance(exc, RainbirdError)
    assert "This was generated by fetch" in str(exc)
    assert str(exc) == f"could not reach GET {PY_BASE_URL}{START_PATH}: This was generated by fetch"
    assert exc.method == "GET"
    # The URL reported by a transport error carries no query string (the request did).
    assert exc.url == f"{PY_BASE_URL}{START_PATH}"
    assert isinstance(exc.__cause__, httpx.ConnectError)
    # The request was attempted exactly once (retries are disabled in make_client).
    _assert_get_start(recorder, {"contextid": JS_CONTEXT})


def _assert_api_error(
    exc: RainbirdAPIError, *, status: int, error_type: type[RainbirdAPIError], message: str
) -> None:
    """JS: ``rejects.toHaveProperty("message", "API error code: <status>")``.

    The JS mock replies with an empty ``{}`` JSON body, so Python falls back to its default
    message for the status.
    """
    # DIFFERENCE(js): JS throws Error("API error code: N") for any !response.ok; Python raises
    # the RainbirdAPIError subclass for the status with structured attributes and a
    # "HTTP N calling GET <url>: <message>" string form.
    url = f"{PY_BASE_URL}{START_PATH}?contextid={JS_CONTEXT}"
    assert type(exc) is error_type
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == status
    assert exc.message == message
    assert exc.errors == []
    assert exc.body == {}
    assert exc.method == "GET"
    assert exc.url == url
    assert exc.retry_after is None
    assert str(exc) == f"HTTP {status} calling GET {url}: {message}"
    assert f"API error code: {status}" not in str(exc)


# source: start.test.js:179 should handle 404
def test_should_handle_404() -> None:
    rb, recorder = _start_client(respond(404, {}))

    with pytest.raises(NotFoundError) as info:
        rb.start(JS_KM_ID, context_id=JS_CONTEXT)

    _assert_api_error(
        info.value,
        status=404,
        error_type=NotFoundError,
        message="The resource (session, knowledge map or fact) could not be found",
    )
    _assert_get_start(recorder, {"contextid": JS_CONTEXT})


# source: start.test.js:193 should handle 500
def test_should_handle_500() -> None:
    rb, recorder = _start_client(respond(500, {}))

    with pytest.raises(ServerError) as info:
        rb.start(JS_KM_ID, context_id=JS_CONTEXT)

    _assert_api_error(
        info.value, status=500, error_type=ServerError, message="Rainbird server error"
    )
    # Retries are disabled (NO_RETRY), so the 500 surfaced after a single request as in JS.
    _assert_get_start(recorder, {"contextid": JS_CONTEXT})


# source: start.test.js:207 should handle 400
def test_should_handle_400() -> None:
    rb, recorder = _start_client(respond(400, {}))

    with pytest.raises(BadRequestError) as info:
        rb.start(JS_KM_ID, context_id=JS_CONTEXT)

    _assert_api_error(
        info.value,
        status=400,
        error_type=BadRequestError,
        message="Your request is missing a required parameter or a value is invalid",
    )
    _assert_get_start(recorder, {"contextid": JS_CONTEXT})


# --------------------------------------------------------------------------- describe("constants")


# source: index.test.js:12 should export constant URL_COMMUNITY
def test_should_export_constant_URL_COMMUNITY() -> None:
    # DIFFERENCE(js): JS exports the bare string URL_COMMUNITY; Python exports the
    # Environment enum (a str subclass, so it compares equal to the URL) and DEFAULT_BASE_URL.
    assert "Environment" in rainbird_ai.__all__
    assert "DEFAULT_BASE_URL" in rainbird_ai.__all__
    assert Environment.COMMUNITY is not None
    assert Environment.COMMUNITY == "https://api.rainbird.ai"  # JS: URL_COMMUNITY
    assert Environment.COMMUNITY.value == "https://api.rainbird.ai"
    assert str(Environment.COMMUNITY) == "https://api.rainbird.ai"
    assert DEFAULT_BASE_URL == Environment.COMMUNITY
    # Usable as base_url, like the JS constant is usable as baseURL.
    rb, _ = make_client(respond(200, START_BODY), base_url=Environment.COMMUNITY)
    assert rb.base_url == "https://api.rainbird.ai"


# source: index.test.js:12 should export constant URL_ENTERPRISE
def test_should_export_constant_URL_ENTERPRISE() -> None:
    assert Environment.ENTERPRISE is not None
    assert Environment.ENTERPRISE == "https://enterprise-api.rainbird.ai"  # JS: URL_ENTERPRISE
    assert Environment.ENTERPRISE.value == "https://enterprise-api.rainbird.ai"
    assert str(Environment.ENTERPRISE) == "https://enterprise-api.rainbird.ai"
    rb, _ = make_client(respond(200, START_BODY), base_url=Environment.ENTERPRISE)
    assert rb.base_url == "https://enterprise-api.rainbird.ai"


# source: index.test.js:12 should export constant ENGINE_CORE
def test_should_export_constant_ENGINE_CORE() -> None:
    # JS: ENGINE_CORE = "Core" for the x-rainbird-engine header; Python exports the same since
    # 1.1.0 (a strict xfail before) and takes it via Rainbird(engine=ENGINE_CORE).
    assert rainbird_ai.ENGINE_CORE is not None
    assert rainbird_ai.ENGINE_CORE == "Core"
    assert "ENGINE_CORE" in rainbird_ai.__all__


# source: index.test.js:12 should export constant ENGINE_BETA
def test_should_export_constant_ENGINE_BETA() -> None:
    assert rainbird_ai.ENGINE_BETA is not None
    assert rainbird_ai.ENGINE_BETA == "Experimental (Beta)"
    assert "ENGINE_BETA" in rainbird_ai.__all__
    rb, rec = make_client(respond(200, START_BODY), engine=rainbird_ai.ENGINE_BETA)
    rb.start(JS_KM_ID)
    assert rec.last.headers["x-rainbird-engine"] == "Experimental (Beta)"
