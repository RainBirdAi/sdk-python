"""Port of the JavaScript base SDK ``undo`` tests to the Python SDK.

Source: ``/tmp/rb-sdk-js/packages/base/lib/__tests__/undo.test.js`` (``describe("undo")``),
exercising ``/tmp/rb-sdk-js/packages/base/lib/undo.ts`` and the ``parse`` helper in
``response_types.js``. Python counterpart: ``Session.undo()`` (``POST /{sid}/undo``) on a session
obtained with ``Rainbird.resume``.

Cases ported, one test per ``it``:

- should make a undo call and parse result
- should make a undo call for a specific engine
- should handle a question response
- should handle a question response with extra questions for grouped relationships
- should throw fetch-generated errors
- should handle 404 / should handle 500 / should handle 400

Differences found (all intentional; the tests assert Python's real behaviour):

- Body: JS posts ``{"data": {}}``; Python posts ``{}``.
- Auth: JS ``undo`` sends no credentials at all (only ``Content-Type`` and, when given, the engine
  header); Python sends ``X-API-Key`` and ``User-Agent`` on every request and never
  ``Authorization``.
- Engine: JS passes ``{engine}`` per call; Python sets it once with ``Rainbird(engine=...)`` and
  then sends ``x-rainbird-engine`` on every request.
- Result parsing: JS ``parse`` returns ``data.result`` untouched (here the object
  ``{"foo": "bar"}``); Python wraps a non-list ``result`` into a one-item ``results`` list of
  ``ResultItem`` and keeps the original body on ``.raw``.
- Errors: JS throws ``Error("API error code: N")``; Python raises the typed ``RainbirdAPIError``
  subclass for the status with ``.status_code``, a default ``.message`` (the ``{}`` body carries
  none) and ``.body == {}``. A transport failure is ``RainbirdConnectionError`` whose message wraps
  the underlying httpx message with the method and URL instead of passing it through verbatim.
"""

from __future__ import annotations

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    NotFoundError,
    QuestionResponse,
    RainbirdAPIError,
    RainbirdConnectionError,
    ResultResponse,
    ServerError,
)
from tests.parity.support import API_KEY, BASE_URL, make_client, raise_transport_error, respond

SESSION_ID = "thisIsTheSessionID"
ERROR_SESSION_ID = "thisistheid"  # the error cases in the JS file use a different id


# source: undo.test.js:13 should make a undo call and parse result
def test_should_make_a_undo_call_and_parse_result() -> None:
    rb, rec = make_client(respond(200, {"result": {"foo": "bar"}}))

    response = rb.resume(SESSION_ID).undo()

    assert len(rec) == 1
    call = rec.last
    # JS checks the fetch init has exactly {body, headers, method}; the Python counterpart is
    # that a POST with headers and a JSON body went to /{sid}/undo.
    assert call.method == "POST"
    assert call.path == f"/{SESSION_ID}/undo"
    assert call.headers["content-type"] == "application/json"
    # DIFFERENCE(js): JS posts {"data": {}}; Python posts {}.
    assert call.json == {}
    assert call.content == b"{}"
    # DIFFERENCE(js): JS undo sends no credentials (its only headers are Content-Type and the
    # optional engine header); Python sends X-API-Key on every request and never Authorization.
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers
    assert "x-rainbird-engine" not in call.headers

    assert isinstance(response, ResultResponse)
    assert response.is_result is True
    assert response.is_question is False
    assert response.question is None
    assert response.raw == {"result": {"foo": "bar"}}
    # DIFFERENCE(js): JS returns {type: RESULT, data: {foo: "bar"}}, i.e. data.result untouched;
    # Python wraps a non-list result into a one-item results list of ResultItem and keeps the
    # original object on the item's .raw.
    assert len(response.results) == 1
    item = response.results[0]
    assert item.raw == {"foo": "bar"}
    assert item.subject is None
    assert item.relationship == ""
    assert item.object is None
    assert item.certainty is None
    assert item.fact_id is None


# source: undo.test.js:33 should make a undo call for a specific engine
def test_should_make_a_undo_call_for_a_specific_engine() -> None:
    # DIFFERENCE(js): JS passes {engine: "thisIsTheEngine"} to the undo call; Python configures
    # the engine once on the client and sends the header on every request.
    rb, rec = make_client(respond(200, {"result": {"foo": "bar"}}), engine="thisIsTheEngine")

    response = rb.resume(SESSION_ID).undo()

    assert len(rec) == 1
    call = rec.last
    assert call.method == "POST"
    assert call.path == f"/{SESSION_ID}/undo"
    assert call.headers["content-type"] == "application/json"
    assert call.headers["x-rainbird-engine"] == "thisIsTheEngine"
    # DIFFERENCE(js): JS posts {"data": {}}; Python posts {}.
    assert call.json == {}
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers

    assert isinstance(response, ResultResponse)
    assert response.is_result is True
    assert response.raw == {"result": {"foo": "bar"}}
    # DIFFERENCE(js): see test_should_make_a_undo_call_and_parse_result.
    assert [r.raw for r in response.results] == [{"foo": "bar"}]


# source: undo.test.js:62 should handle a question response
def test_should_handle_a_question_response() -> None:
    body = {"question": {"foo": "bar"}, "extraQuestions": []}
    rb, rec = make_client(respond(200, body), engine="thisIsTheEngine")
    session = rb.resume(SESSION_ID)

    response = session.undo()

    assert len(rec) == 1
    call = rec.last
    assert call.method == "POST"
    assert call.path == f"/{SESSION_ID}/undo"
    assert call.headers["content-type"] == "application/json"
    assert call.headers["x-rainbird-engine"] == "thisIsTheEngine"
    # JS only asserts toMatchObject({}) here, which any object satisfies; Python's body is
    # exactly {} (DIFFERENCE(js): JS sends {"data": {}}).
    assert call.json == {}

    assert isinstance(response, QuestionResponse)
    assert response.is_question is True
    assert response.is_result is False
    assert response.results == []
    # JS returns {type: QUESTION, data: {question: {foo: "bar"}, extraQuestions: []}}.
    assert response.raw == {"question": {"foo": "bar"}, "extraQuestions": []}
    assert response.question.raw == {"foo": "bar"}
    assert response.question.relationship == ""
    assert response.question.prompt == ""
    assert response.extra_questions == []
    assert response.questions == [response.question]
    # Python additionally remembers the question on the session.
    assert session.pending is response
    assert session.pending_question is response.question
    assert session.result is None


# source: undo.test.js:88 should handle a question response with extra questions for grouped relationships
def test_should_handle_a_question_response_with_extra_questions_for_grouped_relationships() -> None:
    body = {
        "extraQuestions": [
            {"qux": "quux"},
            {"quuz": "corge"},
            {"grault": "garply"},
        ],
        "question": {"foo": "bar"},
    }
    rb, rec = make_client(respond(200, body), engine="thisIsTheEngine")

    response = rb.resume(SESSION_ID).undo()

    assert len(rec) == 1
    call = rec.last
    assert call.method == "POST"
    assert call.path == f"/{SESSION_ID}/undo"
    assert call.headers["content-type"] == "application/json"
    assert call.headers["x-rainbird-engine"] == "thisIsTheEngine"
    # DIFFERENCE(js): JS sends {"data": {}}; Python sends {}.
    assert call.json == {}

    assert isinstance(response, QuestionResponse)
    assert response.is_question is True
    assert response.raw == body
    assert response.question.raw == {"foo": "bar"}
    assert [q.raw for q in response.extra_questions] == [
        {"qux": "quux"},
        {"quuz": "corge"},
        {"grault": "garply"},
    ]
    # questions is the main question followed by the extra ones, in order.
    assert [q.raw for q in response.questions] == [
        {"foo": "bar"},
        {"qux": "quux"},
        {"quuz": "corge"},
        {"grault": "garply"},
    ]
    assert response.results == []


# source: undo.test.js:128 should throw fetch-generated errors
def test_should_throw_fetch_generated_errors() -> None:
    rb, rec = make_client(raise_transport_error(httpx.ConnectError("This was generated by fetch")))

    with pytest.raises(RainbirdConnectionError) as info:
        rb.resume(ERROR_SESSION_ID).undo()

    assert len(rec) == 1
    exc = info.value
    # DIFFERENCE(js): JS rethrows the fetch error so .message is exactly the fetch text; Python
    # wraps it in RainbirdConnectionError with the method and URL, keeping the original text.
    assert str(exc) == (
        f"could not reach POST {BASE_URL}/{ERROR_SESSION_ID}/undo: This was generated by fetch"
    )
    assert exc.method == "POST"
    assert exc.url == f"{BASE_URL}/{ERROR_SESSION_ID}/undo"
    assert isinstance(exc.__cause__, httpx.ConnectError)


# source: undo.test.js:138 should handle 404
def test_should_handle_404() -> None:
    rb, rec = make_client(respond(404, {}))

    with pytest.raises(NotFoundError) as info:
        rb.resume(ERROR_SESSION_ID).undo()

    assert len(rec) == 1
    assert rec.last.method == "POST"
    assert rec.last.path == f"/{ERROR_SESSION_ID}/undo"
    exc = info.value
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == 404
    # DIFFERENCE(js): JS throws Error("API error code: 404"); Python raises NotFoundError whose
    # message falls back to the SDK default because the {} body carries no error text.
    assert exc.message == "The resource (session, knowledge map or fact) could not be found"
    assert str(exc) == f"HTTP 404 calling POST {BASE_URL}/{ERROR_SESSION_ID}/undo: {exc.message}"
    assert exc.body == {}
    assert exc.errors == []
    assert exc.method == "POST"
    assert exc.url == f"{BASE_URL}/{ERROR_SESSION_ID}/undo"


# source: undo.test.js:150 should handle 500
def test_should_handle_500() -> None:
    rb, rec = make_client(respond(500, {}))

    with pytest.raises(ServerError) as info:
        rb.resume(ERROR_SESSION_ID).undo()

    # Not retried: POSTs are never retried on 500 and the helper disables retries anyway.
    assert len(rec) == 1
    assert rec.last.method == "POST"
    assert rec.last.path == f"/{ERROR_SESSION_ID}/undo"
    exc = info.value
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == 500
    # DIFFERENCE(js): JS throws Error("API error code: 500"); Python raises ServerError with the
    # generic 5xx default message because the {} body carries no error text.
    assert exc.message == "Rainbird server error"
    assert str(exc) == f"HTTP 500 calling POST {BASE_URL}/{ERROR_SESSION_ID}/undo: {exc.message}"
    assert exc.body == {}
    assert exc.errors == []


# source: undo.test.js:162 should handle 400
def test_should_handle_400() -> None:
    rb, rec = make_client(respond(400, {}))

    with pytest.raises(BadRequestError) as info:
        rb.resume(ERROR_SESSION_ID).undo()

    assert len(rec) == 1
    assert rec.last.method == "POST"
    assert rec.last.path == f"/{ERROR_SESSION_ID}/undo"
    exc = info.value
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == 400
    # DIFFERENCE(js): JS throws Error("API error code: 400"); Python raises BadRequestError with
    # the 400 default message because the {} body carries no error text.
    assert exc.message == "Your request is missing a required parameter or a value is invalid"
    assert str(exc) == f"HTTP 400 calling POST {BASE_URL}/{ERROR_SESSION_ID}/undo: {exc.message}"
    assert exc.body == {}
    assert exc.errors == []
