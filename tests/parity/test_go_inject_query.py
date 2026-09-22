"""Port of the Go SDK's ``session_test.go`` (lines 1-345): ``TestSessionInject`` and
``TestSessionQuery``.

Source: ``/tmp/rb-sdk-go/session_test.go``; implementation under test ``session.go``
(``Session.Inject``, ``Session.Query``) and ``options.go`` (``AddHeaders``). One pytest test per
Go table entry, named after the entry's ``description``. Each Go case runs against a two-call
``httptest`` server: call 1 answers ``/start`` with ``{"id":"success-id"}`` and call 2 is the
endpoint under test, asserting the request inline; ``go_server()`` below reproduces that.

Differences found (each asserted where it applies, with a ``# DIFFERENCE(go)`` comment):

* Auth: Go sends ``Authorization: Basic base64(key + ':')`` on ``/start``; Python sends
  ``X-API-Key`` on every request and never ``Authorization``.
* Inject payload: Go sends ``"cf": "<string>"`` plus an always-present ``"metadata": {}``;
  Python sends a numeric ``"certainty"`` (it accepts ``cf`` as input) and ``metadata`` only
  when the fact carries one.
* Inject validation: Go sends whatever it is given (``[]``, certainty ``"121"``); Python rejects
  an empty list and a certainty outside 0-100 locally (``InvalidFactError``) before any request.
* Inject metadata (``metadata.source`` = ``"system"`` / ``"user"``): supported since 1.1.0
  (``Fact(metadata=...)``, a ``metadata`` key, ``inject_fact(source=...)``); it used to be
  rejected with ``InvalidFactError`` and was pinned here as a strict xfail.
* Inject response: Go ignores the body of a 2xx; Python parses the documented
  ``{"result": "OK"}`` object and raises ``UnexpectedResponseError`` on Go's empty 200.
* Query payload: Go sends ``"object": ""``; Python omits ``subject``/``object`` when ``None``
  or ``""``.
* Errors: Go returns ``error`` values ``"API returned error N"`` (inject) or
  ``"API returned error N: <raw body>"`` (query); Python raises typed ``RainbirdAPIError``
  subclasses with ``.status_code``/``.message``/``.body``/``.errors``, extracts the message from
  JSON error bodies and supplies a default message for empty bodies.
* Query result: Go returns ``([]Question{}, []Answer{...}, nil)``; Python returns one
  ``ResultResponse`` (``.question is None``, ``.results``).
* ``Answer.String()`` is ``"John - lives in - England [100%]"``; ``str(ResultItem)`` is
  ``"John lives in England (100%)"``.
* Per-call headers: Go's ``AddHeaders`` applies to one ``Query``; Python only has
  ``Rainbird(default_headers=...)``, which applies to every request, ``/start`` included.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    InjectResponse,
    InvalidFactError,
    RainbirdConfigurationError,
    ResultItem,
    ResultResponse,
    ServerError,
    Session,
    UnexpectedResponseError,
)
from tests.parity.support import BASE_URL, Handler, Recorder, make_client

# The Go tests build every Client with this key and start the session with NewSession("kmid", ...).
GO_API_KEY = "1234567890-1234-1234-1234-1234567890ab"
KM_ID = "kmid"
SESSION_ID = "success-id"
START_BODY = {"id": "success-id"}

# --------------------------------------------------------------------------- Go test harness


def go_server(on_second: Handler) -> Handler:
    """Go's two-call ``httptest`` server: call 1 is ``/start``, call 2 is the endpoint under test."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Start (as part of making the session)
            assert request.method == "GET"
            assert request.url.path == f"/start/{KM_ID}"
            assert not request.url.query
            # DIFFERENCE(go): NewSession authenticates with req.SetBasicAuth(apiKey, "");
            # Python sends X-API-Key and never an Authorization header.
            assert request.headers["x-api-key"] == GO_API_KEY
            assert "authorization" not in request.headers
            return httpx.Response(
                200, json=START_BODY, headers={"Content-Type": "application/json"}
            )
        if calls == 2:
            return on_second(request)
        raise AssertionError("Unexpected call")

    return handler


def unexpected_call(request: httpx.Request) -> httpx.Response:
    """Second-call handler for cases where Go expects exactly one request (``expectCalls: 1``)."""
    raise AssertionError(f"Unexpected call: {request.method} {request.url}")


def check_options_headers(headers: httpx.Headers, expected: Mapping[str, str]) -> None:
    """Go's ``checkOptionsHeaders``: every optional header must be present (case-insensitive)."""
    for key, value in expected.items():
        assert headers.get(key) == value


def inject_endpoint(expect_body: Any, status: int, *, body: Any = None) -> Handler:
    """Go's ``case 2`` for TestSessionInject.

    Go's server only writes the status (no body); ``body`` lets the adapted cases reply with the
    documented ``{"result": "OK"}`` that Python requires.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-type"] == "application/json"
        assert request.url.path == f"/{SESSION_ID}/inject"
        assert not request.url.query
        assert request.method == "POST"
        assert json.loads(request.content) == expect_body
        if body is None:
            return httpx.Response(status, headers={"Content-Type": "application/json"})
        return httpx.Response(status, json=body, headers={"Content-Type": "application/json"})

    return handler


def query_endpoint(
    expect_body: Any,
    status: int,
    response_body: str,
    additional_headers: Mapping[str, str] | None = None,
) -> Handler:
    """Go's ``case 2`` for TestSessionQuery: the raw ``responseBody`` string is written verbatim."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/json"
        assert request.url.path == f"/{SESSION_ID}/query"
        assert not request.url.query
        assert request.method == "POST"
        check_options_headers(request.headers, additional_headers or {})
        assert json.loads(request.content) == expect_body
        return httpx.Response(
            status,
            content=response_body.encode(),
            headers={"Content-Type": "application/json"},
        )

    return handler


def new_session(handler: Handler, **kwargs: Any) -> tuple[Session, Recorder]:
    """``client.NewSession("kmid", "", nil, nil)`` against ``handler`` (Go's first server call)."""
    client, recorder = make_client(handler, api_key=GO_API_KEY, **kwargs)
    session = client.start(KM_ID)
    assert session.id == SESSION_ID
    assert session.start_response is not None
    assert session.start_response.raw == START_BODY
    assert len(recorder) == 1
    return session, recorder


# --------------------------------------------------------------------------- TestSessionInject

# Go's "Some facts" entries as Python input. Go's InjectFact.Certainty is the JSON "cf" field, and
# Python accepts "cf" in a fact dict (converting it to "certainty" on the wire).
SOME_FACTS = [
    {"subject": "foo1", "relationship": "bar1", "object": "baz1", "cf": "121"},
    {"subject": "foo2", "relationship": "bar2", "object": "baz2", "cf": "122"},
]
# The same facts with certainties Python can send (0-100); see test_inject_some_facts.
SOME_FACTS_IN_RANGE = [
    {"subject": "foo1", "relationship": "bar1", "object": "baz1", "cf": "21"},
    {"subject": "foo2", "relationship": "bar2", "object": "baz2", "cf": "22"},
]
SOME_FACTS_IN_RANGE_BODY = [
    {"subject": "foo1", "relationship": "bar1", "object": "baz1", "certainty": 21},
    {"subject": "foo2", "relationship": "bar2", "object": "baz2", "certainty": 22},
]
METADATA_FACTS = [
    {
        "subject": "foo1",
        "relationship": "bar1",
        "object": "baz1",
        "cf": "21",
        "metadata": {"source": "system"},
    },
    {
        "subject": "foo2",
        "relationship": "bar2",
        "object": "baz2",
        "cf": "22",
        "metadata": {"source": "user"},
    },
]
# One fact so the "Bad request" cases (Go posts "[]") actually reach the server in Python.
ONE_FACT = ("foo1", "bar1", "baz1")
ONE_FACT_BODY = [{"subject": "foo1", "relationship": "bar1", "object": "baz1", "certainty": 100}]


# source: /tmp/rb-sdk-go/session_test.go:36 TestSessionInject/Empty
def test_inject_empty() -> None:
    session, recorder = new_session(go_server(inject_endpoint([], 200)))
    # DIFFERENCE(go): Go marshals the empty slice and POSTs "[]" (expectCalls 2, err nil);
    # Python refuses an empty fact list locally and never sends the request.
    with pytest.raises(InvalidFactError, match="facts must not be empty"):
        session.inject([])
    assert len(recorder) == 1  # Go: 2


# source: /tmp/rb-sdk-go/session_test.go:44 TestSessionInject/Some facts
def test_inject_some_facts_original_certainties_are_rejected() -> None:
    """The Go table's facts verbatim: certainty ``"121"``/``"122"``."""
    session, recorder = new_session(go_server(inject_endpoint(SOME_FACTS_IN_RANGE_BODY, 200)))
    # DIFFERENCE(go): InjectFact.Certainty is an opaque string that Go sends as "cf":"121";
    # Python validates certainty as a number in 0-100 before sending, so 121/122 never reach
    # the wire (Go's expectBody carries them unchanged).
    with pytest.raises(InvalidFactError, match="certainty must be between 0 and 100"):
        session.inject(SOME_FACTS)
    assert len(recorder) == 1  # Go: 2


# source: /tmp/rb-sdk-go/session_test.go:44 TestSessionInject/Some facts
def test_inject_some_facts() -> None:
    """Go's "Some facts" with certainties Python can send and the documented 200 body."""
    session, recorder = new_session(
        go_server(inject_endpoint(SOME_FACTS_IN_RANGE_BODY, 200, body={"result": "OK"}))
    )
    response = session.inject(SOME_FACTS_IN_RANGE)
    assert len(recorder) == 2
    assert isinstance(response, InjectResponse)
    assert response.ok and response.result == "OK"
    # DIFFERENCE(go): Go's expectBody is
    # [{"subject":"foo1","relationship":"bar1","object":"baz1","cf":"121","metadata":{}},
    #  {"subject":"foo2","relationship":"bar2","object":"baz2","cf":"122","metadata":{}}]
    # i.e. "cf" as a string plus an always-present empty "metadata" object. Python converts "cf"
    # to a numeric "certainty" and sends neither "cf" nor "metadata".
    sent = recorder.last.json
    assert sent == SOME_FACTS_IN_RANGE_BODY
    assert [list(fact) for fact in sent] == [["subject", "relationship", "object", "certainty"]] * 2
    assert all("cf" not in fact and "metadata" not in fact for fact in sent)
    assert recorder.last.headers["content-type"] == "application/json"
    assert recorder.last.headers["x-api-key"] == GO_API_KEY


# source: /tmp/rb-sdk-go/session_test.go:44 TestSessionInject/Some facts
def test_inject_some_facts_empty_200_body() -> None:
    """Go's server answers inject with a bare 200 (``w.WriteHeader`` only) and Inject returns nil."""
    session, recorder = new_session(go_server(inject_endpoint(SOME_FACTS_IN_RANGE_BODY, 200)))
    # DIFFERENCE(go): Go never reads a 2xx inject body; Python parses the documented
    # {"result": "OK"} object and raises UnexpectedResponseError when the body is empty. The
    # request itself was sent and asserted by the handler (calls == 2, as in Go).
    with pytest.raises(UnexpectedResponseError, match="expected a JSON body") as info:
        session.inject(SOME_FACTS_IN_RANGE)
    assert info.value.status_code == 200
    assert info.value.body == ""
    assert len(recorder) == 2
    assert recorder.last.json == SOME_FACTS_IN_RANGE_BODY


# source: /tmp/rb-sdk-go/session_test.go:65 TestSessionInject/Some facts with metadata
def test_inject_some_facts_with_metadata() -> None:
    """``metadata.source`` travels with each injected fact (a strict xfail before 1.1.0).

    Go's expectBody is
    ``[{"subject":"foo1",...,"cf":"121","metadata":{"source":"system"}},
    {"subject":"foo2",...,"cf":"122","metadata":{"source":"user"}}]``. Certainties are kept in
    range and spelled ``certainty`` so this test isolates the metadata behaviour (the
    cf/certainty spelling and the 0-100 check are documented differences asserted above).
    """
    expect_body = [
        {**fact, "metadata": meta}
        for fact, meta in zip(
            SOME_FACTS_IN_RANGE_BODY, [{"source": "system"}, {"source": "user"}], strict=True
        )
    ]
    session, recorder = new_session(
        go_server(inject_endpoint(expect_body, 200, body={"result": "OK"}))
    )
    assert session.inject(METADATA_FACTS).ok
    assert len(recorder) == 2
    assert [fact["metadata"] for fact in recorder.last.json] == [
        {"source": "system"},
        {"source": "user"},
    ]


# source: /tmp/rb-sdk-go/session_test.go:92 TestSessionInject/Bad request
def test_inject_bad_request() -> None:
    # DIFFERENCE(go): Go POSTs "[]" and gets errors.New("API returned error 400"); Python cannot
    # send an empty list (test_inject_empty), so one fact is injected to reach the server, and
    # the 400 surfaces as a typed BadRequestError instead of an error value.
    session, recorder = new_session(go_server(inject_endpoint(ONE_FACT_BODY, 400)))
    with pytest.raises(BadRequestError) as info:
        session.inject(ONE_FACT)
    err = info.value
    assert err.status_code == 400
    assert err.body is None
    assert err.errors == []
    assert err.message == "Your request is missing a required parameter or a value is invalid"
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/{SESSION_ID}/inject"
    assert str(err) == f"HTTP 400 calling POST {BASE_URL}/{SESSION_ID}/inject: {err.message}"
    assert len(recorder) == 2


# source: /tmp/rb-sdk-go/session_test.go:100 TestSessionInject/Bad request
# (the table's second "Bad request" entry, which answers 500)
def test_inject_bad_request_internal_server_error() -> None:
    # DIFFERENCE(go): errors.New("API returned error 500") in Go; ServerError in Python. As in
    # Go, a single request is made: the SDK never retries a 500 on POST (and the parity helper
    # disables retries altogether).
    session, recorder = new_session(go_server(inject_endpoint(ONE_FACT_BODY, 500)))
    with pytest.raises(ServerError) as info:
        session.inject(ONE_FACT)
    err = info.value
    assert err.status_code == 500
    assert err.body is None
    assert err.errors == []
    assert err.message == "Rainbird server error"
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/{SESSION_ID}/inject"
    assert (
        str(err) == f"HTTP 500 calling POST {BASE_URL}/{SESSION_ID}/inject: Rainbird server error"
    )
    assert len(recorder) == 2


# --------------------------------------------------------------------------- TestSessionQuery

# Go's responseBody for the two successful entries (whitespace as in the Go source, tabs aside).
RESULT_BODY = """{
    "result": [{
        "certainty": 100,
        "factID": "thisisthefactid",
        "object": "England",
        "relationship": "lives in",
        "subject": "John"
    }]
}"""
RESULT_ITEM_RAW = {
    "certainty": 100,
    "factID": "thisisthefactid",
    "object": "England",
    "relationship": "lives in",
    "subject": "John",
}
# Go's expectBody for every entry that reaches the server:
#   {"subject":"John","relationship":"lives in","object":""}
# DIFFERENCE(go): Go serialises the empty object (interface{} holding "" is not omitted by
# omitempty); Python drops subject/object when they are None or "".
PY_QUERY_BODY = {"relationship": "lives in", "subject": "John"}


def assert_john_lives_in_england(response: Any) -> ResultItem:
    """Go's expectQuestion ``[]Question{}`` / expectAnswers ``[Answer{John, lives in, England, 100,
    thisisthefactid}]`` in Python terms."""
    # DIFFERENCE(go): Query returns (questions, answers, err); Python returns one ResultResponse.
    assert isinstance(response, ResultResponse)
    assert response.is_result and not response.is_question
    assert response.question is None
    assert len(response) == 1
    [item] = response.results
    assert isinstance(item, ResultItem)
    assert item.subject == "John"
    assert item.relationship == "lives in"
    assert item.object == "England"
    assert item.certainty == 100
    assert item.fact_id == "thisisthefactid"
    assert item.raw == RESULT_ITEM_RAW
    assert response.raw == {"result": [RESULT_ITEM_RAW]}
    assert response.best is item
    assert response.fact_ids == ["thisisthefactid"]
    # DIFFERENCE(go): Answer.String() is "John - lives in - England [100%]".
    assert str(item) == "John lives in England (100%)"
    return item


# source: /tmp/rb-sdk-go/session_test.go:184 TestSessionQuery/Empty
def test_query_empty() -> None:
    session, recorder = new_session(go_server(unexpected_call))
    # Go returns ErrQueryBlankRelationship ("blank relationship") without a request; Python raises
    # RainbirdConfigurationError, also before any request (expectCalls 1 in both).
    with pytest.raises(RainbirdConfigurationError, match="relationship must not be empty"):
        session.query("", subject="", object="")
    assert len(recorder) == 1
    assert session.last_response is None


# source: /tmp/rb-sdk-go/session_test.go:193 TestSessionQuery/John - lives in - ? leading to answer
def test_query_john_lives_in_leading_to_answer() -> None:
    session, recorder = new_session(go_server(query_endpoint(PY_QUERY_BODY, 200, RESULT_BODY)))
    response = session.query("lives in", subject="John", object="")
    assert len(recorder) == 2
    assert recorder.last.method == "POST"
    assert recorder.last.path == f"/{SESSION_ID}/query"
    assert recorder.last.headers["accept"] == "application/json"
    assert recorder.last.headers["content-type"] == "application/json"
    assert recorder.last.json == PY_QUERY_BODY
    assert "object" not in recorder.last.json  # DIFFERENCE(go): Go sends "object":""
    assert_john_lives_in_england(response)
    assert session.last_response is response
    assert session.result is response
    assert session.pending is None


# source: /tmp/rb-sdk-go/session_test.go:224 TestSessionQuery/Handle bad request
def test_query_handle_bad_request() -> None:
    session, recorder = new_session(go_server(query_endpoint(PY_QUERY_BODY, 400, "Foo bar baz")))
    with pytest.raises(BadRequestError) as info:
        session.query("lives in", subject="John", object="")
    err = info.value
    # DIFFERENCE(go): Go returns nil slices and fmt.Errorf("API returned error 400: Foo bar baz");
    # Python raises BadRequestError with the status and the raw text body attached.
    assert err.status_code == 400
    assert err.message == "Foo bar baz"
    assert err.body == "Foo bar baz"
    assert err.errors == ["Foo bar baz"]
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/{SESSION_ID}/query"
    assert str(err) == f"HTTP 400 calling POST {BASE_URL}/{SESSION_ID}/query: Foo bar baz"
    assert len(recorder) == 2
    assert recorder.last.json == PY_QUERY_BODY
    assert session.last_response is None  # a failed query leaves no pending state


# source: /tmp/rb-sdk-go/session_test.go:239 TestSessionQuery/Handle internal server error
def test_query_handle_internal_server_error() -> None:
    session, recorder = new_session(
        go_server(query_endpoint(PY_QUERY_BODY, 500, '{ "error": "some JSON returned" }'))
    )
    with pytest.raises(ServerError) as info:
        session.query("lives in", subject="John", object="")
    err = info.value
    # DIFFERENCE(go): Go's error is 'API returned error 500: { "error": "some JSON returned" }'
    # (the raw body text); Python parses the JSON error body, so .message is the extracted
    # "some JSON returned" and .body is the parsed object rather than the text.
    assert err.status_code == 500
    assert err.message == "some JSON returned"
    assert err.body == {"error": "some JSON returned"}
    assert err.errors == ["some JSON returned"]
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/{SESSION_ID}/query"
    assert str(err) == f"HTTP 500 calling POST {BASE_URL}/{SESSION_ID}/query: some JSON returned"
    assert len(recorder) == 2
    assert recorder.last.json == PY_QUERY_BODY
    assert session.last_response is None


# source: /tmp/rb-sdk-go/session_test.go:254 TestSessionQuery/check the optional headers propagate
def test_query_check_the_optional_headers_propagate() -> None:
    additional_headers = {"x-api-source": "natlang"}
    # DIFFERENCE(go): Go passes AddHeaders(http.Header{"x-api-source": {"natlang"}}) to this one
    # Query call (options.go); Python has no per-call header option on query(), only
    # Rainbird(default_headers=...), which applies to every request the client makes, so the
    # header is also present on the /start call (asserted below).
    session, recorder = new_session(
        go_server(query_endpoint(PY_QUERY_BODY, 200, RESULT_BODY, additional_headers)),
        default_headers=additional_headers,
    )
    response = session.query("lives in", subject="John", object="")
    assert len(recorder) == 2
    check_options_headers(recorder.last.headers, additional_headers)
    assert recorder.last.json == PY_QUERY_BODY
    # default_headers are merged with, not substituted for, the SDK's own headers.
    assert recorder.last.headers["x-api-key"] == GO_API_KEY
    assert recorder.last.headers["accept"] == "application/json"
    assert recorder.last.headers["content-type"] == "application/json"
    # Client-level, so /start carried it too (Go's AddHeaders would not have touched /start).
    start_call = recorder.calls[0]
    assert start_call.path == f"/start/{KM_ID}"
    assert start_call.headers["x-api-source"] == "natlang"
    assert_john_lives_in_england(response)
