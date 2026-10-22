"""Port of the Go SDK client tests: ``/tmp/rb-sdk-go/client_test.go`` lines 1-326.

Source cases (implementation under test: ``client.go`` ``NewSession``, ``ResumeSession``,
``Version`` and ``HTTP``):

* ``TestClientNewSessionValidation``: "Missing API Key", "Missing EnvironmentURL", "Missing KMID".
* ``TestClientNewSessionStartCalls``: "Missing API Key", "Missing KMID", "Simple success",
  "Correctly request context", "Handle 400", "Handle good code but no ID",
  "Success with useDraft", "Success with version".
* ``TestClientHTTP``: "Default", "Custom".
* ``TestClientVersion``: "Simple success", "Error code" (each also ported against the keyless
  module-level ``engine_version()``, the true counterpart of Go's key-less ``Version()``).

Extras derived from reading ``client.go``: ``NewSession`` concatenates every query parameter
with ``"?"`` (``?contextid=foo?useDraft=true``), so two combined-parameter tests document what
Python sends instead; ``ResumeSession`` never validates its argument, ``resume("")`` does.

Differences found (all asserted below as Python's real behaviour):

* Go authenticates ``/start`` with ``Authorization: Basic base64(apiKey + ":")``; Python sends
  ``X-API-Key`` on every request (``/version`` included) and never sends ``Authorization``.
* Go validates the API key, the environment URL and the KM id inside ``NewSession``; Python
  rejects a missing API key when ``Rainbird(...)`` is constructed, falls back to the Community
  host (``https://api.rainbird.ai``) instead of failing on a missing base URL, and rejects an
  empty KM id in ``start()``.
* Go returns ``error`` values with messages such as ``"API returned an error 400: Bad request"``;
  Python raises typed exceptions (``BadRequestError``, ``ServerError``,
  ``UnexpectedResponseError``, ``RainbirdConfigurationError``) with ``str()`` of the form
  ``"HTTP 400 calling GET <url>: Bad request"``.
* Go's ``HTTP()`` hands back the process-wide ``http.DefaultClient``; Python creates a private
  ``httpx.Client`` (timeout 30 s, redirects off) per ``Rainbird`` and closes it in ``close()``.
* Go's ``Version()`` fails at once on a 500; Python's ``Rainbird.version()`` honours the client's
  ``RetryPolicy`` (disabled here) and ``engine_version()`` defaults to ``RetryPolicy()``, so it
  retries a 500 twice with back-off before raising ``ServerError`` unless ``retry=NO_RETRY`` is
  passed (the parameter exists since 1.1.0).
* Go's ``Version()`` returns any 2xx body verbatim; Python validates that it looks like a
  version (``"2.3.4"`` passes).
"""

from __future__ import annotations

import base64
import time

import httpx
import pytest

from rainbird_ai import (
    DEFAULT_BASE_URL,
    NO_RETRY,
    Environment,
    Rainbird,
    RetryPolicy,
    engine_version,
)
from rainbird_ai.errors import (
    BadRequestError,
    RainbirdConfigurationError,
    ServerError,
    UnexpectedResponseError,
)
from tests.parity.support import BASE_URL, Call, Handler, Recorder, make_client

# Values from the Go table (client_test.go).
API_KEY = "abcdefgh-abcd-abcd-abcdefghijkl"
KEY_ENCODED = "Basic YWJjZGVmZ2gtYWJjZC1hYmNkLWFiY2RlZmdoaWprbDo="
KM_ID = "12345678-1234-1234-1234567890ab"
GO_ENV_COMMUNITY = "https://api.rainbird.ai"  # constants.go:5 EnvCommunity


def _go_start_handler(status: int, body: str | None) -> Handler:
    """Mirror of the Go ``httptest`` handler in ``TestClientNewSessionStartCalls``.

    It always adds ``Content-Type: application/json`` and writes the raw body (if any).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, content=body.encode(), headers=headers)

    return handler


def _go_version_handler(status: int, body: str) -> Handler:
    """Mirror of the Go ``httptest`` handler in ``TestClientVersion`` (plain-text body)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


def _recording_http_client(handler: Handler) -> tuple[httpx.Client, Recorder]:
    """A bare ``httpx.Client`` plus recorder, for the keyless ``engine_version()`` calls."""
    recorder = Recorder()

    def wrapped(request: httpx.Request) -> httpx.Response:
        recorder.calls.append(Call(request.method, request.url, request.headers, request.content))
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped)), recorder


def _forbidden_handler(seen: list[httpx.Request]) -> Handler:
    """Go's ``expectCalls: 0``: any request reaching the server fails the test outright."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        pytest.fail(f"server must not be called, got {request.method} {request.url}")

    return handler


def _assert_python_auth(call: Call) -> None:
    """What Go asserts: ``Authorization == KEY_ENCODED``. What Python does: ``X-API-Key``."""
    # Sanity: the Go constant really is the Basic encoding of the key Python sends.
    assert KEY_ENCODED == "Basic " + base64.b64encode(f"{API_KEY}:".encode()).decode()
    # DIFFERENCE(go): Go sends "Authorization: Basic base64(apiKey + ':')" on /start; Python
    # sends "X-API-Key: <apiKey>" and never sends an Authorization header.
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers


# --------------------------------------------------------------------------------------------
# TestClientNewSessionValidation (client_test.go:20-62)
# Go: client.NewSession(kmid, "", nil, nil) returns (nil, <sentinel error>) without any request.
# --------------------------------------------------------------------------------------------


# source: client_test.go:28 TestClientNewSessionValidation/"Missing API Key"
def test_new_session_validation_missing_api_key() -> None:
    # Go: Client{EnvironmentURL: EnvCommunity}.NewSession(kmid, ...) == ErrClientMissingAPIKey.
    # DIFFERENCE(go): Go fails lazily inside NewSession(); Python fails eagerly when the
    # Rainbird client is constructed (RAINBIRD_API_KEY is cleared by the autouse fixture),
    # so there is no client on which start() could even be called.
    with pytest.raises(RainbirdConfigurationError) as excinfo:
        Rainbird(api_key=None, base_url=Environment.COMMUNITY)
    assert "API key is required" in str(excinfo.value)
    assert "RAINBIRD_API_KEY" in str(excinfo.value)

    # An explicit empty string (Go's zero value for APIKey) is rejected the same way.
    with pytest.raises(RainbirdConfigurationError):
        Rainbird(api_key="", base_url=Environment.COMMUNITY)


# source: client_test.go:36 TestClientNewSessionValidation/"Missing EnvironmentURL"
def test_new_session_validation_missing_environment_url() -> None:
    # Go: Client{APIKey: ...}.NewSession(kmid, ...) == ErrClientMissingEnvironmentURL, no request.
    # DIFFERENCE(go): Python has no "missing base URL" error at all: None (and "") fall back to
    # RAINBIRD_API_URL and then to the Community host, which is exactly Go's EnvCommunity, and
    # start() goes ahead and sends the request there.
    assert str(Environment.COMMUNITY) == DEFAULT_BASE_URL == GO_ENV_COMMUNITY

    rb, rec = make_client(
        _go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY, base_url=None
    )
    assert rb.base_url == GO_ENV_COMMUNITY

    session = rb.start(KM_ID)
    assert session.id == "success-id"
    assert len(rec) == 1
    assert rec.last.url.host == "api.rainbird.ai"
    assert str(rec.last.url) == f"{GO_ENV_COMMUNITY}/start/{KM_ID}"

    # An explicit empty base_url counts as "not given" too.
    rb_empty, _ = make_client(_go_start_handler(200, '{"id":"x"}'), api_key=API_KEY, base_url="")
    assert rb_empty.base_url == GO_ENV_COMMUNITY


# source: client_test.go:44 TestClientNewSessionValidation/"Missing KMID"
def test_new_session_validation_missing_kmid() -> None:
    # Go: NewSession("", "", nil, nil) == ErrNewSessionInvalidKMID, result nil, no request.
    # DIFFERENCE(go): Go returns the sentinel error value ErrNewSessionInvalidKMID; Python raises
    # RainbirdConfigurationError with a message that points at RAINBIRD_KM_ID.
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    with pytest.raises(RainbirdConfigurationError) as excinfo:
        rb.start("")
    assert "km_id is required" in str(excinfo.value)
    assert "RAINBIRD_KM_ID" in str(excinfo.value)
    assert len(rec) == 0

    # Python also accepts km_id=None (falls back to RAINBIRD_KM_ID, cleared here) and rejects
    # whitespace-only ids; neither sends a request.
    with pytest.raises(RainbirdConfigurationError):
        rb.start()
    with pytest.raises(RainbirdConfigurationError):
        rb.start("   ")
    assert len(rec) == 0


# --------------------------------------------------------------------------------------------
# TestClientNewSessionStartCalls (client_test.go:64-221)
# Go's handler asserts per request: RequestURI, method GET, Authorization == keyEncoded and
# Accept == application/json, then answers with Content-Type application/json + returnBody.
# --------------------------------------------------------------------------------------------


# source: client_test.go:83 TestClientNewSessionStartCalls/"Missing API Key"
def test_start_calls_missing_api_key() -> None:
    # Go: APIKey "" -> ErrClientMissingAPIKey, expectCalls 0.
    # DIFFERENCE(go): Python raises RainbirdConfigurationError from Rainbird(...) itself, so
    # make_client never returns a client (nor its Recorder). The handler therefore fails the
    # test if the server is contacted, and its own call list backs Go's expectCalls == 0.
    seen: list[httpx.Request] = []
    handler = _forbidden_handler(seen)
    with pytest.raises(RainbirdConfigurationError) as excinfo:
        make_client(handler, api_key="")
    assert "API key is required" in str(excinfo.value)
    with pytest.raises(RainbirdConfigurationError):
        make_client(handler, api_key=None)
    assert seen == []


# source: client_test.go:88 TestClientNewSessionStartCalls/"Missing KMID"
def test_start_calls_missing_kmid() -> None:
    # Go: kmid "" -> ErrNewSessionInvalidKMID, expectCalls 0.
    # DIFFERENCE(go): sentinel error value in Go, RainbirdConfigurationError in Python; neither
    # sends a request.
    rb, rec = make_client(_forbidden_handler([]), api_key=API_KEY)
    with pytest.raises(RainbirdConfigurationError) as excinfo:
        rb.start("")
    assert "km_id is required" in str(excinfo.value)
    assert len(rec) == 0


# source: client_test.go:93 TestClientNewSessionStartCalls/"Simple success"
def test_start_calls_simple_success() -> None:
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    session = rb.start(KM_ID)

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}"  # Go: RequestURI == "/start/<kmid>"
    assert call.headers["accept"] == "application/json"
    assert call.content == b""
    _assert_python_auth(call)

    # Go: result.ID == "success-id". Python: Session.id plus the parsed StartResponse.
    assert session.id == "success-id"
    assert session.start_response is not None
    assert session.start_response.session_id == "success-id"
    assert session.start_response.km_version_id is None  # body carried no kmVersion
    assert session.start_response.raw == {"id": "success-id"}
    assert session.km_id == KM_ID
    assert session.context_id is None


# source: client_test.go:107 TestClientNewSessionStartCalls/"Correctly request context"
def test_start_calls_correctly_request_context() -> None:
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    session = rb.start(KM_ID, context_id="foo")

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    # Go: RequestURI == "/start/<kmid>?contextid=foo"; Python uses the same parameter name.
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {"contextid": "foo"}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}?contextid=foo"
    assert call.headers["accept"] == "application/json"
    _assert_python_auth(call)

    assert session.id == "success-id"
    assert session.context_id == "foo"


# source: client_test.go:122 TestClientNewSessionStartCalls/"Handle 400"
def test_start_calls_handle_400() -> None:
    # Go returns errors.New("API returned an error 400: Bad request"), result nil.
    rb, rec = make_client(_go_start_handler(400, "Bad request"), api_key=API_KEY)

    with pytest.raises(BadRequestError) as excinfo:
        rb.start(KM_ID)

    # Go's handler asserts the same four things on this request as on the success cases.
    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}"  # Go: RequestURI == "/start/<kmid>"
    assert call.headers["accept"] == "application/json"
    _assert_python_auth(call)

    exc = excinfo.value
    # DIFFERENCE(go): Go's error text is "API returned an error 400: Bad request"; Python raises
    # a typed BadRequestError whose str() also names the method and URL.
    assert exc.status_code == 400
    assert exc.message == "Bad request"
    assert exc.errors == ["Bad request"]
    assert exc.body == "Bad request"  # not JSON despite the Content-Type, kept as text
    assert exc.method == "GET"
    assert exc.url == f"{BASE_URL}/start/{KM_ID}"
    assert str(exc) == f"HTTP 400 calling GET {BASE_URL}/start/{KM_ID}: Bad request"


# source: client_test.go:136 TestClientNewSessionStartCalls/"Handle good code but no ID"
def test_start_calls_handle_good_code_but_no_id() -> None:
    # Go returns errors.New("API returned no error but no ID either"), result nil.
    rb, rec = make_client(_go_start_handler(200, "{}"), api_key=API_KEY)

    with pytest.raises(UnexpectedResponseError) as excinfo:
        rb.start(KM_ID)

    # Go's handler asserts the same four things on this request as on the success cases.
    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}"  # Go: RequestURI == "/start/<kmid>"
    assert call.headers["accept"] == "application/json"
    _assert_python_auth(call)

    exc = excinfo.value
    # DIFFERENCE(go): message wording differs; Python additionally exposes the offending body
    # and the HTTP status on the exception.
    assert exc.message == "start response did not include a session id"
    assert exc.body == {}
    assert exc.status_code == 200


# source: client_test.go:150 TestClientNewSessionStartCalls/"Success with useDraft"
def test_start_calls_success_with_use_draft() -> None:
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    session = rb.start(KM_ID, use_draft=True)

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    # Go: RequestURI == "/start/<kmid>?useDraft=true".
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {"useDraft": "true"}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}?useDraft=true"
    assert call.headers["accept"] == "application/json"
    _assert_python_auth(call)
    assert session.id == "success-id"


# source: client_test.go:165 TestClientNewSessionStartCalls/"Success with version"
def test_start_calls_success_with_version() -> None:
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    session = rb.start(KM_ID, version=1)

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    # Go: RequestURI == "/start/<kmid>?version=1".
    assert call.path == f"/start/{KM_ID}"
    assert call.query == {"version": "1"}
    assert str(call.url) == f"{BASE_URL}/start/{KM_ID}?version=1"
    assert call.headers["accept"] == "application/json"
    _assert_python_auth(call)
    assert session.id == "success-id"


# source: client.go:126-136 NewSession URL building (not covered by a Go test case)
def test_start_combined_params_build_one_query_string() -> None:
    # Go appends every parameter with its own "?", so contextID + useDraft would request
    # "/start/<kmid>?contextid=foo?useDraft=true" (a broken query string; the Go table only
    # ever sets one parameter per case, which is why it never surfaces there).
    # DIFFERENCE(go): Python encodes a single, properly "&"-joined query string.
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    rb.start(KM_ID, context_id="foo", use_draft=True)
    assert rec.last.path == f"/start/{KM_ID}"
    assert rec.last.query == {"useDraft": "true", "contextid": "foo"}
    assert str(rec.last.url).count("?") == 1
    assert rec.last.url.query == b"useDraft=true&contextid=foo"

    rb.start(KM_ID, context_id="foo", version=2)
    assert rec.last.query == {"version": "2", "contextid": "foo"}
    assert str(rec.last.url).count("?") == 1


# source: client.go:130-136 NewSession useDraft + version (not covered by a Go test case)
def test_start_rejects_use_draft_together_with_version() -> None:
    # Go would happily send "/start/<kmid>?useDraft=true?version=1".
    # DIFFERENCE(go): Python refuses the combination locally, before any request is sent, and
    # also rejects a non-positive version number.
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    with pytest.raises(RainbirdConfigurationError) as excinfo:
        rb.start(KM_ID, use_draft=True, version=1)
    assert "use_draft and version cannot be used together" in str(excinfo.value)

    with pytest.raises(RainbirdConfigurationError) as excinfo:
        rb.start(KM_ID, version=0)
    assert "version must be a positive integer" in str(excinfo.value)
    assert len(rec) == 0


# --------------------------------------------------------------------------------------------
# TestClientHTTP (client_test.go:223-258)
# Go: Client.HTTP() returns http.DefaultClient unless HTTPClient was set.
# --------------------------------------------------------------------------------------------


# source: client_test.go:238 TestClientHTTP/"Default"
def test_http_default() -> None:
    # Go: (&Client{}).HTTP() == http.DefaultClient (a process-wide shared client).
    # DIFFERENCE(go): Rainbird.http is an httpx.Client the SDK creates for this instance
    # (timeout 30 s, redirects never followed) and owns; two clients do not share it and
    # close() shuts it down. Go's Client{} also has no API key, which Python requires.
    rb = Rainbird(api_key=API_KEY, base_url=BASE_URL)
    other = Rainbird(api_key=API_KEY, base_url=BASE_URL)
    try:
        assert isinstance(rb.http, httpx.Client)
        assert rb.http is rb.http  # stable accessor, like Go's HTTP()
        assert rb.http is not other.http
        assert rb.http.timeout == httpx.Timeout(30.0)
        assert rb.http.follow_redirects is False
        assert rb.http.is_closed is False
    finally:
        rb.close()
        other.close()
    assert rb.http.is_closed is True


# source: client_test.go:243 TestClientHTTP/"Custom"
def test_http_custom() -> None:
    # Go: (&Client{HTTPClient: custom}).HTTP() == custom.
    custom = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    rb = Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=custom)
    assert rb.http is custom

    # A user-supplied client is used as-is and never closed by the SDK.
    rb.close()
    assert custom.is_closed is False
    with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=custom) as rb2:
        assert rb2.http is custom
    assert custom.is_closed is False
    custom.close()


# --------------------------------------------------------------------------------------------
# TestClientVersion (client_test.go:260-326)
# Go: Client{EnvironmentURL, HTTPClient} (no API key) .Version(); handler asserts
# RequestURI == "/version" and method GET, then writes the plain-text body.
# --------------------------------------------------------------------------------------------


# source: client_test.go:273 TestClientVersion/"Simple success"
def test_version_simple_success() -> None:
    # Adapted: Go's client has no API key; Rainbird(...) needs one, so the keyed variant is
    # exercised here and the true key-less twin in test_engine_version_simple_success below.
    rb, rec = make_client(_go_version_handler(200, "2.3.4"), api_key=API_KEY)

    assert rb.version() == "2.3.4"

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    assert call.path == "/version"
    assert call.query == {}
    assert str(call.url) == f"{BASE_URL}/version"  # Go: RequestURI == "/version"
    # DIFFERENCE(go): Go's Version() sets no headers at all; Rainbird.version() sends the
    # client's X-API-Key like every other call, plus a text-first Accept header.
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers
    assert call.headers["accept"] == "text/plain, application/json, */*"


# source: client_test.go:284 TestClientVersion/"Error code"
def test_version_error_code() -> None:
    # Go returns ("", errors.New("API returned error code 500: Foobarbaz")). Its table lists
    # expectCalls 1, but TestClientVersion only counts the calls and never asserts the count.
    rb, rec = make_client(_go_version_handler(500, "Foobarbaz"), api_key=API_KEY)

    with pytest.raises(ServerError) as excinfo:
        rb.version()

    # Stronger than Go: NO_RETRY (set by make_client) keeps this at one call; the default
    # RetryPolicy would retry a GET 500 twice more, whereas Go never retries.
    assert len(rec) == 1
    assert rec.last.method == "GET"
    assert rec.last.path == "/version"

    exc = excinfo.value
    # DIFFERENCE(go): typed ServerError instead of an error string; the message is the body.
    assert exc.status_code == 500
    assert exc.message == "Foobarbaz"
    assert exc.errors == ["Foobarbaz"]
    assert exc.body == "Foobarbaz"
    assert str(exc) == f"HTTP 500 calling GET {BASE_URL}/version: Foobarbaz"


# source: client_test.go:273 TestClientVersion/"Simple success" (key-less engine_version twin)
def test_engine_version_simple_success() -> None:
    # Go's Version() needs no API key; Python's equivalent is the module-level engine_version().
    http, rec = _recording_http_client(_go_version_handler(200, "2.3.4"))

    assert engine_version(BASE_URL, http_client=http) == "2.3.4"

    assert len(rec) == 1
    call = rec.last
    assert call.method == "GET"
    assert call.path == "/version"
    assert str(call.url) == f"{BASE_URL}/version"
    # No key was given, so no X-API-Key and (as always) no Authorization header.
    assert "x-api-key" not in call.headers
    assert "authorization" not in call.headers
    assert call.headers["accept"] == "text/plain, application/json, */*"
    # The caller's client is never closed by the SDK.
    assert http.is_closed is False
    http.close()


# source: client_test.go:284 TestClientVersion/"Error code" (key-less engine_version twin)
def test_engine_version_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Go makes one call (its table lists expectCalls 1 without asserting it), then returns
    # errors.New("API returned error code 500: Foobarbaz").
    # DIFFERENCE(go): Go never retries; engine_version() defaults to RetryPolicy(), so a GET 500
    # is retried max_retries (2) more times with exponential back-off before ServerError is
    # raised. time.sleep is patched so the test does not wait ~1.5 s. Pass retry=NO_RETRY for
    # Go's single attempt (test_engine_version_error_code_no_retry below; retry= exists since
    # 1.1.0).
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    http, rec = _recording_http_client(_go_version_handler(500, "Foobarbaz"))

    with pytest.raises(ServerError) as excinfo:
        engine_version(BASE_URL, http_client=http)

    expected_calls = RetryPolicy().max_retries + 1
    assert len(rec) == expected_calls == 3
    assert len(slept) == expected_calls - 1
    assert all(delay > 0 for delay in slept)
    assert all(call.method == "GET" and call.path == "/version" for call in rec.calls)

    exc = excinfo.value
    assert exc.status_code == 500
    assert exc.message == "Foobarbaz"
    assert str(exc) == f"HTTP 500 calling GET {BASE_URL}/version: Foobarbaz"
    http.close()


# source: client_test.go:284 TestClientVersion/"Error code" (key-less twin, single attempt)
def test_engine_version_error_code_no_retry() -> None:
    # With retry=NO_RETRY the key-less helper behaves like Go: one call, then the error.
    http, rec = _recording_http_client(_go_version_handler(500, "Foobarbaz"))

    with pytest.raises(ServerError) as excinfo:
        engine_version(BASE_URL, http_client=http, retry=NO_RETRY)

    assert len(rec) == 1
    assert rec.last.method == "GET" and rec.last.path == "/version"
    assert excinfo.value.status_code == 500
    assert excinfo.value.message == "Foobarbaz"
    http.close()


# --------------------------------------------------------------------------------------------
# ResumeSession (client.go:188-190) - no Go test case, documented from the implementation.
# --------------------------------------------------------------------------------------------


# source: client.go:188 ResumeSession (implementation only; no Go test case)
def test_resume_session() -> None:
    # Go: ResumeSession(id) returns &Session{ID: id} for any string, "" included, never errors.
    rb, rec = make_client(_go_start_handler(200, '{"id":"success-id"}'), api_key=API_KEY)

    session = rb.resume("resumed-id")
    assert session.id == "resumed-id"
    assert session.start_response is None
    assert len(rec) == 0  # no request, as in Go

    # DIFFERENCE(go): Python rejects an empty session id instead of building a useless session.
    with pytest.raises(RainbirdConfigurationError):
        rb.resume("")
    assert len(rec) == 0
