"""Port of the JavaScript base SDK tests for ``query`` and ``parse``.

Source (``/tmp/rb-sdk-js/packages/base/lib``):

- ``__tests__/query.test.js`` -> ``describe("query")``: "should make a simple query for object",
  "... for subject", "... with both subject and object", "should handle a question response",
  "should make a query for specific engine", "should handle a question response with extra
  questions for grouped relationships", "should throw fetch-generated errors", "should handle
  404", "should handle 500", "should handle 400". Implementation: ``query.ts``.
- ``__tests__/response_types.test.js`` -> ``describe("parse")``: "should handle a question",
  "should handle extra questions", "should handle a result" and a second, duplicated "should
  handle a question" that actually exercises ``RESPONSE_TYPE_UNKNOWN``. Implementation:
  ``response_types.js``.

Mapping: ``query(baseURL, sid, subject, relationship, object, {engine})`` becomes
``Rainbird(engine=...).resume(sid).query(relationship, subject=..., object=...)`` (``resume``
sends nothing, so exactly one request is made, as ``fetch`` is called once in JS);
``parse(data)`` becomes ``rainbird_ai.parse_query_response(data)``.

Differences found (each is also flagged inline with ``# DIFFERENCE(js)``):

- JS sends only ``Content-Type`` (plus ``x-rainbird-engine``) on ``/query``, no credentials;
  Python sends ``X-API-Key`` on every request.
- JS takes the engine per call; Python sets it once on the client (``Rainbird(engine=...)``).
- JS spreads ``subject: null`` / ``object: null`` into the body when ``null`` is passed
  (``null !== undefined``); Python omits the key for ``None`` and ``""``.
- JS returns ``{type: RESPONSE_TYPE_RESULT, data: <raw result value>}``; Python returns a
  ``ResultResponse`` whose ``.raw`` is the whole body. The unrealistic ``result: {"foo": "bar"}``
  (the API always returns a list) is not rejected: ``_as_list`` wraps it into one empty
  ``ResultItem``.
- JS returns the raw question dicts; Python wraps them in ``Question`` models (``.raw`` keeps
  the original) and tracks the pending question on the session.
- JS throws ``Error("API error code: N")``; Python raises the typed ``RainbirdAPIError``
  subclass with ``status_code``, a default message, ``body``, ``method`` and ``url``.
- JS rethrows the ``fetch`` error unchanged; Python wraps it in ``RainbirdConnectionError``
  (original exception as ``__cause__``, its message embedded).
- A body with neither ``question`` nor ``result`` is ``RESPONSE_TYPE_UNKNOWN`` in JS; Python's
  ``parse_query_response`` raises ``UnexpectedResponseError`` carrying the body (a bare
  ``ValueError`` before 1.1.0) and ``Session.query`` re-raises it with the HTTP status.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from rainbird_ai import (
    Question,
    QuestionResponse,
    Rainbird,
    ResultItem,
    ResultResponse,
    Session,
    parse_query_response,
)
from rainbird_ai.errors import (
    BadRequestError,
    NotFoundError,
    RainbirdConnectionError,
    RainbirdError,
    ServerError,
    UnexpectedResponseError,
)
from tests.parity.support import (
    API_KEY,
    BASE_URL,
    Call,
    Handler,
    Recorder,
    make_client,
    raise_transport_error,
    respond,
)


def _session(handler: Handler, session_id: str, **client_kwargs: Any) -> tuple[Session, Recorder]:
    """JS ``query(baseURL, sid, ...)`` addresses a session by id; ``resume`` does the same
    without sending anything, so the recorder only ever sees the ``/query`` call."""
    rb: Rainbird
    rb, recorder = make_client(handler, **client_kwargs)
    return rb.resume(session_id), recorder


def _assert_query_request(call: Call, session_id: str, body: dict[str, Any]) -> None:
    """The request-side assertions every JS ``query`` case makes.

    JS checks ``Object.keys(call).sort() == ["body", "headers", "method"]``, ``method ==
    "POST"``, ``Content-Type: application/json`` and ``toMatchObject`` on the parsed body
    (partial match). Python is asserted on the exact body.
    """
    assert call.method == "POST"
    assert call.path == f"/{session_id}/query"
    assert str(call.url) == f"{BASE_URL}/{session_id}/query"
    assert call.query == {}
    assert call.content, "the request must carry a body"
    assert call.headers["content-type"] == "application/json"
    assert call.json == body
    # DIFFERENCE(js): query.ts sends no credentials at all on /query (only Content-Type and,
    # when given, x-rainbird-engine). Python sends X-API-Key on every request and never
    # sends Authorization.
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers


# --------------------------------------------------------------------------- describe("query")


# source: query.test.js:15 should make a simple query for object
def test_query_should_make_a_simple_query_for_object() -> None:
    session, rec = _session(respond(200, {"result": {"foo": "bar"}}), "thisisthesessionid")

    result = session.query("thisistherelationship", subject="thisisthesubject", object=None)

    assert len(rec) == 1  # fetch toHaveBeenCalledTimes(1)
    # DIFFERENCE(js): query.ts spreads ``object: null`` into the body when null is passed
    # (``null !== undefined``); the JS test only partial-matches so never notices. Python
    # omits the key for None (and for "").
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {"subject": "thisisthesubject", "relationship": "thisistherelationship"},
    )
    assert "x-rainbird-engine" not in rec.last.headers  # no engine option given
    # DIFFERENCE(js): JS returns {type: RESPONSE_TYPE_RESULT, data: {foo: "bar"}}; Python
    # returns a ResultResponse whose .raw is the whole body, so the JS ``data`` is
    # ``.raw["result"]``.
    assert isinstance(result, ResultResponse)
    assert result.is_result and not result.is_question
    assert result.raw == {"result": {"foo": "bar"}}
    assert result.raw["result"] == {"foo": "bar"}
    assert result.to_dict() == {"result": {"foo": "bar"}}
    # DIFFERENCE(js): the real API returns a list under ``result``; this JS body is not
    # realistic. Python does not reject it: ``_as_list`` wraps the bare object into a single
    # ResultItem with every field empty (its .raw keeps the object).
    assert len(result.results) == 1
    assert isinstance(result.results[0], ResultItem)
    assert result.results[0].raw == {"foo": "bar"}
    assert result.results[0].fact_id is None
    assert result.fact_ids == []
    assert session.last_response is result
    assert session.result is result
    assert session.pending is None


# source: query.test.js:46 should make a simple query for subject
def test_query_should_make_a_simple_query_for_subject() -> None:
    session, rec = _session(respond(200, {"result": {"foo": "bar"}}), "thisisthesessionid")

    result = session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    # DIFFERENCE(js): JS sends ``subject: null`` here (null !== undefined); Python omits it.
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {"relationship": "thisistherelationship", "object": "thisistheobject"},
    )
    assert "x-rainbird-engine" not in rec.last.headers
    # DIFFERENCE(js): ResultResponse instead of {type: RESPONSE_TYPE_RESULT, data: {foo: bar}}.
    assert isinstance(result, ResultResponse)
    assert result.is_result
    assert result.raw == {"result": {"foo": "bar"}}
    assert result.raw["result"] == {"foo": "bar"}
    assert [item.raw for item in result.results] == [{"foo": "bar"}]
    assert session.result is result


# source: query.test.js:77 should make a simple query with both subject and object
def test_query_should_make_a_simple_query_with_both_subject_and_object() -> None:
    session, rec = _session(respond(200, {"result": {"foo": "bar"}}), "thisisthesessionid")

    result = session.query(
        "thisistherelationship", subject="thisisthesubject", object="thisistheobject"
    )

    assert len(rec) == 1
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {
            "subject": "thisisthesubject",
            "relationship": "thisistherelationship",
            "object": "thisistheobject",
        },
    )
    assert "x-rainbird-engine" not in rec.last.headers
    # DIFFERENCE(js): ResultResponse instead of {type: RESPONSE_TYPE_RESULT, data: {foo: bar}}.
    assert isinstance(result, ResultResponse)
    assert result.is_result
    assert result.raw == {"result": {"foo": "bar"}}
    assert result.raw["result"] == {"foo": "bar"}
    assert [item.raw for item in result.results] == [{"foo": "bar"}]
    assert session.result is result


# source: query.test.js:109 should handle a question response
def test_query_should_handle_a_question_response() -> None:
    body = {"question": {"foo": "bar"}, "extraQuestions": []}
    session, rec = _session(respond(200, body), "thisisthesessionid")

    result = session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {"relationship": "thisistherelationship", "object": "thisistheobject"},
    )
    # DIFFERENCE(js): JS returns {type: RESPONSE_TYPE_QUESTION, data: {question, extraQuestions}}
    # with the raw dicts; Python returns a QuestionResponse wrapping Question models whose
    # .raw is the original dict.
    assert isinstance(result, QuestionResponse)
    assert result.is_question and not result.is_result
    assert result.raw == body
    assert result.to_dict() == {"question": {"foo": "bar"}, "extraQuestions": []}
    assert isinstance(result.question, Question)
    assert result.question.raw == {"foo": "bar"}
    assert result.question.to_dict() == {"foo": "bar"}
    assert result.extra_questions == []
    assert result.questions == [result.question]
    assert result.results == []
    # Python-only: the session remembers the pending question for answer()/skip().
    assert session.last_response is result
    assert session.pending is result
    assert session.pending_question is result.question
    assert session.result is None


# source: query.test.js:144 should make a query for specific engine
def test_query_should_make_a_query_for_specific_engine() -> None:
    # DIFFERENCE(js): JS passes {engine: "thisistheengine"} per call; Python sets it once with
    # Rainbird(engine=...) and sends x-rainbird-engine on every request.
    session, rec = _session(
        respond(200, {"result": {"foo": "bar"}}), "thisisthesessionid", engine="thisistheengine"
    )

    result = session.query("thisistherelationship", subject="thisisthesubject", object=None)

    assert len(rec) == 1
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {"subject": "thisisthesubject", "relationship": "thisistherelationship"},
    )
    assert rec.last.headers["x-rainbird-engine"] == "thisistheengine"
    # DIFFERENCE(js): ResultResponse instead of {type: RESPONSE_TYPE_RESULT, data: {foo: bar}}.
    assert isinstance(result, ResultResponse)
    assert result.is_result
    assert result.raw == {"result": {"foo": "bar"}}
    assert result.raw["result"] == {"foo": "bar"}
    assert [item.raw for item in result.results] == [{"foo": "bar"}]


# source: query.test.js:179 should handle a question response with extra questions for grouped relationships
def test_query_should_handle_a_question_response_with_extra_questions_for_grouped_relationships() -> (
    None
):
    body = {
        "question": {"foo": "bar"},
        "extraQuestions": [
            {"qux": "quux"},
            {"quuz": "corge"},
            {"grault": "garply"},
        ],
    }
    session, rec = _session(respond(200, body), "thisisthesessionid")

    result = session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    _assert_query_request(
        rec.last,
        "thisisthesessionid",
        {"relationship": "thisistherelationship", "object": "thisistheobject"},
    )
    # DIFFERENCE(js): QuestionResponse with Question models instead of the raw
    # {type: RESPONSE_TYPE_QUESTION, data: {question, extraQuestions}} dict.
    assert isinstance(result, QuestionResponse)
    assert result.is_question
    assert result.raw == body
    assert result.to_dict() == body
    assert result.question.raw == {"foo": "bar"}
    assert [q.raw for q in result.extra_questions] == [
        {"qux": "quux"},
        {"quuz": "corge"},
        {"grault": "garply"},
    ]
    assert all(isinstance(q, Question) for q in result.extra_questions)
    # ``questions`` is the main question followed by the extra ones (grouped relationships).
    assert [q.raw for q in result.questions] == [
        {"foo": "bar"},
        {"qux": "quux"},
        {"quuz": "corge"},
        {"grault": "garply"},
    ]
    assert session.pending_question is result.question


# source: query.test.js:222 should throw fetch-generated errors
def test_query_should_throw_fetch_generated_errors() -> None:
    session, rec = _session(
        raise_transport_error(httpx.ConnectError("This was generated by fetch")), "thisistheid"
    )

    with pytest.raises(RainbirdConnectionError) as excinfo:
        session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    # DIFFERENCE(js): JS rethrows the fetch error unchanged (message "This was generated by
    # fetch"); Python wraps transport failures in RainbirdConnectionError, keeps the original
    # as __cause__ and embeds its message.
    assert isinstance(excinfo.value, RainbirdError)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)
    assert str(excinfo.value.__cause__) == "This was generated by fetch"
    assert "This was generated by fetch" in str(excinfo.value)
    assert str(excinfo.value) == (
        f"could not reach POST {BASE_URL}/thisistheid/query: This was generated by fetch"
    )
    assert excinfo.value.method == "POST"
    assert excinfo.value.url == f"{BASE_URL}/thisistheid/query"


# source: query.test.js:238 should handle 404
def test_query_should_handle_404() -> None:
    session, rec = _session(respond(404, {}), "thisistheid")

    with pytest.raises(NotFoundError) as excinfo:
        session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    assert rec.last.method == "POST"
    assert rec.last.path == "/thisistheid/query"
    # DIFFERENCE(js): JS throws Error("API error code: 404"); Python raises NotFoundError
    # (a RainbirdAPIError) with status_code 404 and, for a body with no error text, the
    # SDK's default 404 message.
    err = excinfo.value
    assert err.status_code == 404
    assert err.message == "The resource (session, knowledge map or fact) could not be found"
    assert err.errors == []
    assert err.body == {}
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/thisistheid/query"
    assert str(err) == f"HTTP 404 calling POST {BASE_URL}/thisistheid/query: {err.message}"
    assert session.last_response is None  # a failed query is not tracked


# source: query.test.js:256 should handle 500
def test_query_should_handle_500() -> None:
    session, rec = _session(respond(500, {}), "thisistheid")

    with pytest.raises(ServerError) as excinfo:
        session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1  # retries are disabled in make_client, so exactly one attempt
    assert rec.last.method == "POST"
    assert rec.last.path == "/thisistheid/query"
    # DIFFERENCE(js): JS throws Error("API error code: 500"); Python raises ServerError with
    # status_code 500 and the default server-error message.
    err = excinfo.value
    assert err.status_code == 500
    assert err.message == "Rainbird server error"
    assert err.errors == []
    assert err.body == {}
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/thisistheid/query"
    assert str(err) == f"HTTP 500 calling POST {BASE_URL}/thisistheid/query: Rainbird server error"
    assert session.last_response is None


# source: query.test.js:274 should handle 400
def test_query_should_handle_400() -> None:
    session, rec = _session(respond(400, {}), "thisistheid")

    with pytest.raises(BadRequestError) as excinfo:
        session.query("thisistherelationship", subject=None, object="thisistheobject")

    assert len(rec) == 1
    assert rec.last.method == "POST"
    assert rec.last.path == "/thisistheid/query"
    # DIFFERENCE(js): JS throws Error("API error code: 400"); Python raises BadRequestError
    # with status_code 400 and the default 400 message.
    err = excinfo.value
    assert err.status_code == 400
    assert err.message == "Your request is missing a required parameter or a value is invalid"
    assert err.errors == []
    assert err.body == {}
    assert err.method == "POST"
    assert err.url == f"{BASE_URL}/thisistheid/query"
    assert str(err) == f"HTTP 400 calling POST {BASE_URL}/thisistheid/query: {err.message}"
    assert session.last_response is None


# --------------------------------------------------------------------------- describe("parse")


# source: response_types.test.js:12 should handle a question
def test_parse_should_handle_a_question() -> None:
    parsed = parse_query_response({"question": {"foo": "bar"}})

    # DIFFERENCE(js): JS returns {type: RESPONSE_TYPE_QUESTION, data: {question, extraQuestions:
    # []}}; Python returns a QuestionResponse (is_question) whose Question keeps the raw dict.
    assert isinstance(parsed, QuestionResponse)
    assert parsed.is_question and not parsed.is_result
    assert parsed.question.raw == {"foo": "bar"}
    assert parsed.extra_questions == []  # JS: data.extraQuestions || []
    assert parsed.questions == [parsed.question]
    assert parsed.raw == {"question": {"foo": "bar"}}


# source: response_types.test.js:19 should handle extra questions
def test_parse_should_handle_extra_questions() -> None:
    parsed = parse_query_response({"question": {"foo": "bar"}, "extraQuestions": [{"qux": "quux"}]})

    # DIFFERENCE(js): QuestionResponse with Question models instead of the raw dicts under
    # {type: RESPONSE_TYPE_QUESTION, data: {...}}.
    assert isinstance(parsed, QuestionResponse)
    assert parsed.is_question
    assert parsed.question.raw == {"foo": "bar"}
    assert [q.raw for q in parsed.extra_questions] == [{"qux": "quux"}]
    assert [q.raw for q in parsed.questions] == [{"foo": "bar"}, {"qux": "quux"}]
    assert parsed.raw == {"question": {"foo": "bar"}, "extraQuestions": [{"qux": "quux"}]}


# source: response_types.test.js:28 should handle a result
def test_parse_should_handle_a_result() -> None:
    parsed = parse_query_response({"result": {"foo": "bar"}})

    # DIFFERENCE(js): JS returns {type: RESPONSE_TYPE_RESULT, data: {foo: "bar"}} (data is the
    # raw ``result`` value); Python returns a ResultResponse whose .raw is the whole body.
    assert isinstance(parsed, ResultResponse)
    assert parsed.is_result and not parsed.is_question
    assert parsed.raw == {"result": {"foo": "bar"}}
    assert parsed.raw["result"] == {"foo": "bar"}
    assert parsed.question is None
    # DIFFERENCE(js): a non-list ``result`` (never produced by the API) is not rejected;
    # ``_as_list`` wraps it into one ResultItem with every field empty.
    assert len(parsed.results) == 1
    assert parsed.results[0].raw == {"foo": "bar"}
    assert (
        parsed.results[0].subject,
        parsed.results[0].relationship,
        parsed.results[0].object,
    ) == (
        None,
        "",
        None,
    )


# source: response_types.test.js:35 should handle a question
# (sic: the JS file reuses the name "should handle a question"; this second case actually
# feeds a body with neither ``question`` nor ``result`` and expects RESPONSE_TYPE_UNKNOWN.)
def test_parse_should_handle_a_question_unknown_body() -> None:
    data = {"baz": {"foo": "bar"}}

    # DIFFERENCE(js): JS returns {type: RESPONSE_TYPE_UNKNOWN, data} and leaves the caller to
    # deal with it; Python's parse_query_response refuses to guess and raises
    # UnexpectedResponseError (a RainbirdError, not a ValueError; before 1.1.0 it was a bare
    # ValueError) naming the keys it saw and carrying the body.
    with pytest.raises(
        UnexpectedResponseError, match=r"neither a questionResponse nor a resultResponse"
    ) as ei:
        parse_query_response(data)
    assert "keys: ['baz']" in str(ei.value)
    assert ei.value.body is data
    assert ei.value.status_code is None
    assert not isinstance(ei.value, ValueError)
    assert data == {"baz": {"foo": "bar"}}  # the input is not mutated

    # The SDK-level counterpart of JS ``query`` returning RESPONSE_TYPE_UNKNOWN: Session.query
    # re-raises the error with the HTTP status attached (the parse error as __cause__).
    session, rec = _session(respond(200, data), "thisistheid")
    with pytest.raises(UnexpectedResponseError) as excinfo:
        session.query("thisistherelationship", subject=None, object="thisistheobject")
    assert len(rec) == 1
    assert excinfo.value.body == data
    assert excinfo.value.status_code == 200
    assert isinstance(excinfo.value.__cause__, UnexpectedResponseError)
    assert session.last_response is None
