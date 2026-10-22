"""Port of the JavaScript base SDK ``response`` unit tests to the Python SDK.

Source: ``/tmp/rb-sdk-js/packages/base/lib/__tests__/response.test.js`` (``describe("response")``
including the nested ``describe("should pass answer when specified")``). Implementation under
test there: ``/tmp/rb-sdk-js/packages/base/lib/response.ts`` -- ``response(baseURL, sessionID,
answers, {engine})`` does ``POST {baseURL}/{sessionID}/response`` with ``{"answers": [...]}`` and
``Content-Type: application/json`` (plus ``x-rainbird-engine`` when given), throws
``Error("API error code: N")`` on a non-2xx status and returns ``response_types.parse(json)``:
``{type: QUESTION, data: {question, extraQuestions: []}}`` or ``{type: RESULT, data: result}``.

Python counterpart: ``Rainbird(...).resume(session_id).respond(answers)`` (``Session.respond`` ->
``_answers.normalize_answers`` -> ``_ops.respond`` -> ``POST /{sid}/response``), returning a
``QuestionResponse`` or ``ResultResponse`` and remembering it on ``session.last_response``.

Cases ported, one test each: single answer, question response, certainty, answer ``yes``,
answer ``no``, multiple answers, engine header, extra questions, fetch-generated (transport)
error, 404, 500 and 400.

Differences found (each asserted below and marked ``DIFFERENCE(js)``):

- JS sends the answer objects verbatim. Python normalises every answer and always sends
  ``certainty`` (default ``100``); it also accepts ``cf`` and converts it to ``certainty``.
- JS sends only ``Content-Type`` (no auth) on ``/response``. Python also sends ``X-API-Key``,
  ``Accept`` and ``User-Agent`` on every request and never sends ``Authorization``.
- JS passes ``engine`` per call. Python sets it once with ``Rainbird(engine=...)`` and then sends
  ``x-rainbird-engine`` on every request.
- JS returns ``{type, data}`` with the raw ``result``/``question`` payload. Python returns typed
  ``ResultResponse``/``QuestionResponse`` objects (raw JSON kept on ``.raw``); a non-list
  ``result`` such as ``{"foo": "bar"}`` is wrapped into a single, empty ``ResultItem``.
- JS ``response(url, sid, [])`` sends ``{"answers": []}`` to the server. Python rejects an empty
  answer list locally with ``InvalidFactError`` before any request is made, so the error-path
  tests send one answer to reach the server.
- JS throws ``Error("API error code: N")``. Python raises the ``RainbirdAPIError`` subclass for
  the status (``BadRequestError``/``NotFoundError``/``ServerError``) with ``.status_code``, a
  status-default ``.message`` when the body is ``{}``, ``.errors == []`` and ``.body == {}``.
- JS uses the base URL string verbatim (``"thisistheurl"``, ``"url"``). Python normalises a bare
  host to ``https://<host>``.
- JS ``response()`` is a stateless function. Python's ``Session.respond`` also records the outcome
  on ``session.last_response`` (exposed as ``pending`` / ``pending_question`` / ``result``) and a
  failed call leaves it untouched.
- Python-only extra: ``test_should_pass_certainty_when_specified`` additionally sends an answer
  spelled with ``cf`` (no JS counterpart) to show it is converted to ``certainty`` on the wire.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    InvalidFactError,
    NotFoundError,
    QuestionResponse,
    RainbirdAPIError,
    RainbirdConnectionError,
    ResultResponse,
    ServerError,
    Session,
)
from tests.parity.support import (
    API_KEY,
    Call,
    Handler,
    Recorder,
    make_client,
    raise_transport_error,
    respond,
)

# The JS suite passes these strings verbatim as baseURL / sessionID.
BASE_URL = "thisistheurl"
SESSION_ID = "thisisthesessionid"
ERROR_SESSION_ID = "thisistheid"

RESULT_BODY: dict[str, Any] = {"result": {"foo": "bar"}}
QUESTION_BODY: dict[str, Any] = {"question": {"foo": "bar"}}

BOB_ENGLAND: dict[str, Any] = {"subject": "Bob", "relationship": "lives in", "object": "England"}
BOB_FRANCE: dict[str, Any] = {"subject": "Bob", "relationship": "lives in", "object": "France"}


def _session(
    handler: Handler,
    *,
    base_url: str = BASE_URL,
    session_id: str = SESSION_ID,
    **kwargs: Any,
) -> tuple[Session, Recorder]:
    """``response(baseURL, sessionID, ...)`` in JS is a bare function; in Python it is a method on
    a ``Session`` attached with ``Rainbird.resume`` (which sends no request)."""
    rb, recorder = make_client(handler, base_url=base_url, **kwargs)
    return rb.resume(session_id), recorder


def _with_default_certainty(answer: dict[str, Any]) -> dict[str, Any]:
    # DIFFERENCE(js): JS sends the answer object verbatim ({subject, relationship, object});
    # Python always adds the protocol default "certainty": 100.
    return {**answer, "certainty": 100}


def _assert_response_call(
    call: Call, *, base_url: str = BASE_URL, session_id: str = SESSION_ID
) -> None:
    """JS: ``fetch`` called once with exactly ``{body, headers, method}``, ``method == "POST"`` and
    ``headers`` matching ``{"Content-Type": "application/json"}``."""
    assert call.method == "POST"
    # DIFFERENCE(js): JS fetches "<baseURL>/<sessionID>/response" with the baseURL string as
    # given; Python normalises the bare host to https://<host>.
    assert str(call.url) == f"https://{base_url}/{session_id}/response"
    assert call.path == f"/{session_id}/response"
    assert call.query == {}  # JS passes no options besides body/headers/method: no query string
    assert call.headers["content-type"] == "application/json"
    # DIFFERENCE(js): JS sends only Content-Type (and x-rainbird-engine when given) on /response;
    # Python also sends X-API-Key, Accept and User-Agent on every request, never Authorization.
    assert call.headers["x-api-key"] == API_KEY
    assert "authorization" not in call.headers
    assert call.headers["accept"] == "application/json"
    assert call.headers["user-agent"].startswith("rainbird-ai-python/")


def _assert_result_foo_bar(result: Any, session: Session) -> None:
    """JS: ``expect(result).toEqual({type: RESPONSE_TYPE_RESULT, data: {foo: "bar"}})``."""
    # DIFFERENCE(js): JS returns {type, data} with the raw "result" value; Python returns a
    # ResultResponse whose .raw holds the whole body.
    assert isinstance(result, ResultResponse)
    assert result.is_result is True
    assert result.is_question is False
    assert result.question is None
    assert result.raw == RESULT_BODY
    assert result.to_dict()["result"] == {"foo": "bar"}
    # DIFFERENCE(js): JS hands back data = {foo: "bar"} untouched; Python wraps a non-list
    # "result" into a single ResultItem with every known field empty (raw is preserved).
    assert len(result.results) == 1
    item = result.results[0]
    assert item.raw == {"foo": "bar"}
    assert (item.subject, item.relationship, item.object) == (None, "", None)
    assert (item.certainty, item.fact_id) == (None, None)
    # DIFFERENCE(js): response() in JS is stateless; Session.respond also records the outcome.
    assert session.last_response is result
    assert session.result is result
    assert session.pending is None


def _assert_question_foo_bar(result: Any, session: Session, extra: list[dict[str, Any]]) -> None:
    """JS: ``{type: RESPONSE_TYPE_QUESTION, data: {question: {foo: "bar"}, extraQuestions}}``."""
    # DIFFERENCE(js): JS returns the raw question dict; Python returns a QuestionResponse with a
    # typed Question (raw preserved on .raw / to_dict()).
    assert isinstance(result, QuestionResponse)
    assert result.is_question is True
    assert result.is_result is False
    assert result.question.to_dict() == {"foo": "bar"}
    assert result.question.raw == {"foo": "bar"}
    assert [q.to_dict() for q in result.extra_questions] == extra
    assert result.questions == [result.question, *result.extra_questions]
    assert result.results == []
    # DIFFERENCE(js): response() in JS is stateless; Session.respond also records the outcome.
    assert session.last_response is result
    assert session.pending is result
    assert session.pending_question is result.question
    assert session.result is None


def _assert_failure_left_no_state(session: Session) -> None:
    # DIFFERENCE(js): response() in JS is stateless; Python's Session.respond records each outcome
    # on session.last_response, and a failed call leaves it untouched (still None here).
    assert session.last_response is None
    assert session.pending is None
    assert session.pending_question is None
    assert session.result is None


def _assert_empty_answers_rejected(session: Session, recorder: Recorder) -> None:
    """JS calls ``response(url, sid, [])`` and the server decides; Python refuses locally."""
    # DIFFERENCE(js): JS sends {"answers": []}; Python raises InvalidFactError before sending.
    with pytest.raises(InvalidFactError, match="answers must not be empty"):
        session.respond([])
    assert len(recorder) == 0


def _assert_api_error(
    exc: RainbirdAPIError,
    *,
    error_type: type[RainbirdAPIError],
    status: int,
    message: str,
) -> None:
    """JS: ``rejects.toHaveProperty("message", "API error code: N")``."""
    # DIFFERENCE(js): JS throws Error("API error code: N"); Python raises the typed subclass with
    # status_code, a status-default message (the body {} carries none), errors, body, method, url.
    assert type(exc) is error_type
    assert exc.status_code == status
    assert exc.message == message
    assert exc.errors == []
    assert exc.body == {}
    assert exc.method == "POST"
    assert exc.url == f"https://{BASE_URL}/{ERROR_SESSION_ID}/response"
    assert str(exc) == f"HTTP {status} calling POST {exc.url}: {message}"
    assert exc.retry_after is None


# source: response.test.js:15 should make a response call containing a single answer
def test_should_make_a_response_call_containing_a_single_answer() -> None:
    session, recorder = _session(respond(200, RESULT_BODY))

    result = session.respond([BOB_ENGLAND])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call)
    assert "x-rainbird-engine" not in call.headers
    assert call.json == {"answers": [_with_default_certainty(BOB_ENGLAND)]}
    # toMatchObject in JS is a subset match; the original answer is still contained verbatim.
    assert BOB_ENGLAND.items() <= call.json["answers"][0].items()
    _assert_result_foo_bar(result, session)


# source: response.test.js:51 should handle a question response
def test_should_handle_a_question_response() -> None:
    session, recorder = _session(respond(200, QUESTION_BODY))

    result = session.respond([BOB_ENGLAND])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call)
    assert call.json == {"answers": [_with_default_certainty(BOB_ENGLAND)]}
    assert result.raw == QUESTION_BODY
    # JS: data.extraQuestions defaults to [] when the body has none.
    assert result.extra_questions == []
    _assert_question_foo_bar(result, session, extra=[])


# source: response.test.js:87 should pass certainty when specified
def test_should_pass_certainty_when_specified() -> None:
    session, recorder = _session(respond(200, RESULT_BODY))
    answer = {"subject": "Bob", "relationship": "lives in", "object": "England", "certainty": 80}

    result = session.respond([answer])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call)
    assert call.json == {"answers": [answer]}
    assert call.json["answers"][0]["certainty"] == 80
    _assert_result_foo_bar(result, session)

    # Python-only extra sub-case (no counterpart in response.test.js:87).
    # DIFFERENCE(js): the JS Fact type only knows "certainty" and forwards it as-is; Python also
    # accepts the API's "cf" spelling and sends it as "certainty" (same wire payload as above).
    session.respond([{"subject": "Bob", "relationship": "lives in", "object": "England", "cf": 80}])
    assert len(recorder) == 2
    assert recorder.last.json == {"answers": [answer]}
    assert "cf" not in recorder.last.json["answers"][0]


# source: response.test.js:126 should pass answer when specified > should pass yes as an answer
def test_should_pass_yes_as_an_answer() -> None:
    session, recorder = _session(respond(200, RESULT_BODY), base_url="url", session_id="sessionID")
    payload = {
        "answer": "yes",
        "certainty": 80,
        "object": "England",
        "relationship": "lives in",
        "subject": "Bob",
    }

    result = session.respond([payload])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call, base_url="url", session_id="sessionID")
    assert call.json == {"answers": [payload]}
    assert call.json["answers"][0]["answer"] == "yes"
    _assert_result_foo_bar(result, session)


# source: response.test.js:155 should pass answer when specified > should pass no as an answer
def test_should_pass_no_as_an_answer() -> None:
    session, recorder = _session(respond(200, RESULT_BODY), base_url="url", session_id="sessionID")
    payload = {
        "answer": "no",
        "certainty": 80,
        "object": "England",
        "relationship": "lives in",
        "subject": "Bob",
    }

    result = session.respond([payload])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call, base_url="url", session_id="sessionID")
    assert call.json == {"answers": [payload]}
    assert call.json["answers"][0]["answer"] == "no"
    _assert_result_foo_bar(result, session)


# source: response.test.js:185 should make a response call containing multiple answers
def test_should_make_a_response_call_containing_multiple_answers() -> None:
    session, recorder = _session(respond(200, RESULT_BODY))

    result = session.respond([BOB_ENGLAND, BOB_FRANCE])

    assert len(recorder) == 1  # one request carrying both answers
    call = recorder.last
    _assert_response_call(call)
    assert call.json == {
        "answers": [_with_default_certainty(BOB_ENGLAND), _with_default_certainty(BOB_FRANCE)]
    }
    assert [a["object"] for a in call.json["answers"]] == ["England", "France"]
    _assert_result_foo_bar(result, session)


# source: response.test.js:231 should make a response call for a specific engine
def test_should_make_a_response_call_for_a_specific_engine() -> None:
    # DIFFERENCE(js): JS passes {engine} per call as the 4th argument of response(); Python
    # configures it once with Rainbird(engine=...) and sends the header on every request.
    session, recorder = _session(respond(200, RESULT_BODY), engine="thisistheengine")

    result = session.respond([BOB_ENGLAND])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call)
    assert call.headers["content-type"] == "application/json"
    assert call.headers["x-rainbird-engine"] == "thisistheengine"
    assert call.json == {"answers": [_with_default_certainty(BOB_ENGLAND)]}
    _assert_result_foo_bar(result, session)


# source: response.test.js:275 should handle a question response with extra questions for
# grouped relationships
def test_should_handle_a_question_response_with_extra_questions_for_grouped_relationships() -> None:
    extra = [{"qux": "quux"}, {"quuz": "corge"}, {"grault": "garply"}]
    body = {"extraQuestions": extra, "question": {"foo": "bar"}}
    session, recorder = _session(respond(200, body))

    result = session.respond([BOB_ENGLAND])

    assert len(recorder) == 1
    call = recorder.last
    _assert_response_call(call)
    assert call.json == {"answers": [_with_default_certainty(BOB_ENGLAND)]}
    assert result.raw == body
    _assert_question_foo_bar(result, session, extra=extra)
    assert len(result.questions) == 4
    assert [q.raw for q in result.extra_questions] == extra


# source: response.test.js:323 should throw fetch-generated errors
def test_should_throw_fetch_generated_errors() -> None:
    handler = raise_transport_error(httpx.ConnectError("This was generated by fetch"))
    session, recorder = _session(handler, session_id=ERROR_SESSION_ID)

    # JS calls response("thisistheurl", "thisistheid", []) and fetch rejects.
    _assert_empty_answers_rejected(session, recorder)

    with pytest.raises(RainbirdConnectionError) as info:
        session.respond([BOB_ENGLAND])

    # DIFFERENCE(js): JS re-throws the fetch error unchanged (message "This was generated by
    # fetch"); Python wraps it in RainbirdConnectionError, keeping the original text.
    exc = info.value
    assert "This was generated by fetch" in str(exc)
    assert str(exc) == (
        f"could not reach POST https://{BASE_URL}/{ERROR_SESSION_ID}/response: "
        "This was generated by fetch"
    )
    assert exc.method == "POST"
    assert exc.url == f"https://{BASE_URL}/{ERROR_SESSION_ID}/response"
    assert isinstance(exc.__cause__, httpx.ConnectError)
    assert len(recorder) == 1  # the request was attempted exactly once (no retry)
    _assert_failure_left_no_state(session)


# source: response.test.js:333 should handle 404
def test_should_handle_404() -> None:
    session, recorder = _session(respond(404, {}), session_id=ERROR_SESSION_ID)
    _assert_empty_answers_rejected(session, recorder)

    with pytest.raises(NotFoundError) as info:
        session.respond([BOB_ENGLAND])

    _assert_api_error(
        info.value,
        error_type=NotFoundError,
        status=404,
        message="The resource (session, knowledge map or fact) could not be found",
    )
    assert len(recorder) == 1
    _assert_response_call(recorder.last, session_id=ERROR_SESSION_ID)
    _assert_failure_left_no_state(session)


# source: response.test.js:345 should handle 500
def test_should_handle_500() -> None:
    session, recorder = _session(respond(500, {}), session_id=ERROR_SESSION_ID)
    _assert_empty_answers_rejected(session, recorder)

    with pytest.raises(ServerError) as info:
        session.respond([BOB_ENGLAND])

    _assert_api_error(
        info.value, error_type=ServerError, status=500, message="Rainbird server error"
    )
    assert len(recorder) == 1  # POST /response is never retried on 500 (and NO_RETRY is set)
    _assert_response_call(recorder.last, session_id=ERROR_SESSION_ID)
    _assert_failure_left_no_state(session)


# source: response.test.js:357 should handle 400
def test_should_handle_400() -> None:
    session, recorder = _session(respond(400, {}), session_id=ERROR_SESSION_ID)
    _assert_empty_answers_rejected(session, recorder)

    with pytest.raises(BadRequestError) as info:
        session.respond([BOB_ENGLAND])

    _assert_api_error(
        info.value,
        error_type=BadRequestError,
        status=400,
        message="Your request is missing a required parameter or a value is invalid",
    )
    assert len(recorder) == 1
    _assert_response_call(recorder.last, session_id=ERROR_SESSION_ID)
    _assert_failure_left_no_state(session)
