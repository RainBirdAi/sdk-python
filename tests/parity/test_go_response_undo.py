"""Port of ``TestSessionResponse`` and ``TestSessionUndo`` from the Go SDK's ``session_test.go``.

Source: ``/tmp/rb-sdk-go/session_test.go`` lines 346-743; implementation under test is
``Session.Response`` and ``Session.Undo`` in ``/tmp/rb-sdk-go/session.go``. Every table entry
becomes one test named after its Go ``description``. Each Go case first creates a session
(``NewSession("kmid", "", nil, nil)`` answered with ``{"id":"success-id"}``) and then makes exactly
one more call, so ``_go_server`` serves requests by call count exactly like the Go ``httptest``
handler (a third call is ``t.Fatal("Unexpected call")``).

Differences found (Python's real behaviour is asserted; each is flagged inline as
``DIFFERENCE(go)``):

* Certainty key: Go's ``QAnswer`` serialises ``"cf": "100"`` (a string); Python accepts ``cf``
  as input (dict answers) but sends ``"certainty": 100`` (a number) and never a ``cf`` key; an
  ``Answer(..., certainty=100)`` object produces the same payload.
* Float object: Go marshals ``float64(12)`` as ``12``; Python writes ``12.0`` on the wire (both
  parse as the number 12).
* Auth: Go sends ``Authorization: Basic base64(apiKey + ':')`` on ``/start`` only; Python sends
  ``X-API-Key`` on every request and never an ``Authorization`` header.
* Errors: Go returns ``errors.New("API returned error 400: Foo bar baz")``; Python raises
  ``BadRequestError`` / ``ServerError`` with ``.status_code``, ``.message == "Foo bar baz"``,
  ``.errors``, ``.body``, ``.method``, ``.url`` and ``str()`` of the form
  ``"HTTP 400 calling POST <url>: Foo bar baz"``.
* Return shape: Go returns ``([]Question, []Answer, error)``; Python returns a single
  ``QuestionResponse`` (``.questions`` is the main question followed by ``extraQuestions``,
  ``.results`` is ``[]`` on a question response).
* ``str(Question)`` is the prompt alone; Go's ``Question.String()`` is
  ``"<prompt> (<subject> - <relationship> - ?)"``.
* Accept header: Go's ``Undo`` sends no ``Accept`` header (``Response`` does); Python sends
  ``Accept: application/json`` on both.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import pytest

from rainbird_ai import (
    Answer,
    BadRequestError,
    Question,
    QuestionForm,
    QuestionResponse,
    RainbirdAPIError,
    ServerError,
)
from tests.parity.support import BASE_URL, Call, Handler, make_client

# Go: client := &Client{APIKey: "1234567890-1234-1234-1234-1234567890ab", ...}
GO_API_KEY = "1234567890-1234-1234-1234-1234567890ab"

# Go call 1: `{"id":"success-id"}` (the /start reply behind NewSession("kmid", "", nil, nil)).
START_BODY = {"id": "success-id"}

# Go: QAnswer{Subject: "John", Relationship: "Speaks", Object: "English", CF: "100"}
GO_ANSWER = {"subject": "John", "relationship": "Speaks", "object": "English", "cf": "100"}

# Go expectBody: {"answers":[{"subject":"John","relationship":"Speaks","object":"English","cf":"100"}]}
# DIFFERENCE(go): Python sends "certainty": 100 (number) instead of "cf": "100" (string).
PY_ANSWER_SENT = {
    "subject": "John",
    "relationship": "Speaks",
    "object": "English",
    "certainty": 100,
}

# Go responseBody payloads, verbatim.
QUESTION_LIVES_IN = {
    "subject": "John",
    "dataType": "string",
    "relationship": "lives in",
    "type": "Second Form Object",
    "plural": False,
    "allowCF": True,
    "allowUnknown": False,
    "canAdd": True,
    "prompt": "Where does John live?",
    "knownAnswers": [],
}

QUESTION_VISITS_TO = {
    "subject": "John",
    "dataType": "string",
    "relationship": "visits to",
    "type": "Second Form Object",
    "plural": False,
    "allowCF": True,
    "allowUnknown": False,
    "canAdd": True,
    "prompt": "Does John travel here a lot?",
    "knownAnswers": [],
}

QUESTION_LIVES_IN_OBJECT_12 = {
    "subject": "John",
    "object": "12",
    "dataType": "string",
    "relationship": "lives in",
    "type": "Second Form Object",
    "plural": False,
    "allowCF": True,
    "allowUnknown": False,
    "canAdd": True,
    "prompt": "Where does John live?",
    "knownAnswers": [],
}


# --------------------------------------------------------------------------- helpers


def _go_server(status: int, body: Any = None, *, text: str | None = None) -> Handler:
    """Mirror the Go ``httptest`` handler: call 1 answers ``/start`` with ``{"id":"success-id"}``,
    call 2 returns the scripted ``responseCode``/``responseBody``, anything else is
    ``t.Fatal("Unexpected call")``.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=START_BODY)
        if calls == 2:
            if text is not None:
                # The Go handler writes the plain text "Foo bar baz" under a JSON content type;
                # keep that quirk so the SDK's error-body parsing sees exactly what Go's did.
                return httpx.Response(
                    status, content=text.encode(), headers={"Content-Type": "application/json"}
                )
            return httpx.Response(status, json=body)
        raise AssertionError("Unexpected call")

    return handler


def _assert_start_call(call: Call) -> None:
    """The Go handler does not inspect call 1; these are the Python-side facts about it."""
    assert call.method == "GET"
    assert call.path == "/start/kmid"
    assert call.query == {}
    # DIFFERENCE(go): Go authenticates /start with "Authorization: Basic base64(apiKey + ':')";
    # Python sends "X-API-Key" on every request and never an Authorization header.
    assert call.headers["x-api-key"] == GO_API_KEY
    assert "authorization" not in call.headers


def _assert_response_call(call: Call, expected_answers: list[dict[str, Any]]) -> None:
    """Go call 2 assertions for ``Response``: content type, request URI, method and exact body."""
    assert call.headers["content-type"] == "application/json"
    assert call.path == "/success-id/response"  # Go: r.RequestURI == "/success-id/response"
    assert call.query == {}
    assert call.method == "POST"
    # DIFFERENCE(go): Go's expectBody carries "cf":"100" (string) per answer; Python sends
    # "certainty": 100 (number) and no "cf" key. The rest of the payload is identical.
    assert call.json == {"answers": expected_answers}
    for sent in call.json["answers"]:
        assert "cf" not in sent
        assert isinstance(sent["certainty"], int)
        # Go compares the raw body string, so field order is part of its assertion: Python's
        # Answer.to_payload() keeps Go's order with "certainty" in the slot Go uses for "cf".
        assert list(sent) == ["subject", "relationship", "object", "certainty"]
    # Python-side extras (Go's Response sets Accept too but does not assert on it).
    assert call.headers["accept"] == "application/json"
    assert call.headers["x-api-key"] == GO_API_KEY
    assert "authorization" not in call.headers


def _assert_undo_call(call: Call) -> None:
    """Go call 2 assertions for ``Undo``: content type, request URI, method and body ``{}``."""
    assert call.headers["content-type"] == "application/json"
    assert call.path == "/success-id/undo"  # Go: r.RequestURI == "/success-id/undo"
    assert call.query == {}
    assert call.method == "POST"
    assert call.content == b"{}"  # Go: assert.Equal(t, "{}", string(body))
    assert call.json == {}
    # DIFFERENCE(go): Go's Undo sends no Accept header (only Content-Type); Python sends
    # Accept: application/json on every decision call.
    assert call.headers["accept"] == "application/json"
    assert call.headers["x-api-key"] == GO_API_KEY
    assert "authorization" not in call.headers


def _assert_second_form_question(
    question: Question, *, relationship: str, prompt: str, expected_object: Any = None
) -> None:
    """Field-by-field check of a Go ``expectQuestions`` entry (all Go zero values included)."""
    assert question.allow_cf is True  # AllowCF: true
    assert question.allow_unknown is False  # AllowUnknown: false
    assert question.can_add is True  # CanAdd: true
    assert question.concepts == []  # Concepts: nil
    assert question.data_type == "string"  # DataType: "string"
    assert question.known_answers == []  # KnownAnswers: []KnownAnswer{}
    assert question.plural is False  # Plural: false
    assert question.prompt == prompt
    assert question.relationship == relationship
    assert question.subject == "John"  # Subject: "John"
    if expected_object is None:
        assert question.object is None  # Object: interface{}(nil) (no "object" key in the body)
    else:
        assert question.object == expected_object  # Object: "12"
    assert question.type == "Second Form Object"
    assert question.object_type is None  # ObjectType zero value ""
    assert question.object_metadata == {}  # ObjectMetadata zero value nil
    # Python extras derived from the same fields.
    assert question.form is QuestionForm.SECOND_OBJECT
    assert question.asks_for_object is True
    assert question.is_first_form is False
    assert question.options == []


def _assert_go_error(exc: RainbirdAPIError, status: int, path: str) -> None:
    """Python's counterpart of Go's ``errors.New("API returned error N: Foo bar baz")``."""
    # DIFFERENCE(go): Go returns an error value whose message is
    # "API returned error {status}: Foo bar baz"; Python raises a typed RainbirdAPIError subclass
    # that carries the same pieces as attributes and renders them as
    # "HTTP {status} calling POST <url>: Foo bar baz".
    assert exc.status_code == status
    assert exc.message == "Foo bar baz"
    assert exc.errors == ["Foo bar baz"]
    assert exc.body == "Foo bar baz"
    assert exc.method == "POST"
    assert exc.url == f"{BASE_URL}{path}"
    assert str(exc) == f"HTTP {status} calling POST {BASE_URL}{path}: Foo bar baz"


# --------------------------------------------------------------------------- TestSessionResponse
# source: /tmp/rb-sdk-go/session_test.go:346 TestSessionResponse


# source: /tmp/rb-sdk-go/session_test.go:361 "Simple response that returns a question"
def test_response_simple_response_that_returns_a_question() -> None:
    rb, recorder = make_client(_go_server(200, {"question": QUESTION_LIVES_IN}), api_key=GO_API_KEY)
    session = rb.start("kmid")
    assert session.id == "success-id"

    # respond() takes prebuilt answers and, like Go's Response, needs no pending question.
    response = session.respond([GO_ANSWER])
    # The same Go QAnswer written as a Python Answer object yields the identical payload.
    assert Answer("John", "Speaks", "English", certainty=100).to_payload() == PY_ANSWER_SENT

    assert len(recorder) == 2  # expectCalls: 2
    _assert_start_call(recorder.calls[0])
    _assert_response_call(recorder.calls[1], [PY_ANSWER_SENT])

    # DIFFERENCE(go): Go returns ([]Question, []Answer, error); Python returns one
    # QuestionResponse whose .questions is the main question followed by extraQuestions.
    assert isinstance(response, QuestionResponse)
    assert response.is_question is True
    assert response.is_result is False
    assert len(response.questions) == 1
    _assert_second_form_question(
        response.questions[0], relationship="lives in", prompt="Where does John live?"
    )
    assert response.question is response.questions[0]
    assert response.extra_questions == []
    assert response.results == []  # expectAnswers: nil
    assert response.raw == {"question": QUESTION_LIVES_IN}
    assert response.question.raw == QUESTION_LIVES_IN
    assert session.last_response is response
    assert session.pending_question is response.question
    # DIFFERENCE(go): Go's Question.String() renders "Where does John live? (John - lives in - ?)";
    # Python's str(Question) is the prompt alone.
    assert str(response.question) == "Where does John live?"


# source: /tmp/rb-sdk-go/session_test.go:408 "Simple response that returns extra questions"
def test_response_simple_response_that_returns_extra_questions() -> None:
    body = {"question": QUESTION_LIVES_IN, "extraQuestions": [QUESTION_VISITS_TO]}
    rb, recorder = make_client(_go_server(200, body), api_key=GO_API_KEY)
    session = rb.start("kmid")

    response = session.respond([GO_ANSWER])

    assert len(recorder) == 2  # expectCalls: 2
    _assert_start_call(recorder.calls[0])
    _assert_response_call(recorder.calls[1], [PY_ANSWER_SENT])

    assert isinstance(response, QuestionResponse)
    assert response.is_question is True
    # Go: questions = [*body.Question, body.ExtraQuestions...] -> main question first.
    assert len(response.questions) == 2
    _assert_second_form_question(
        response.questions[0], relationship="lives in", prompt="Where does John live?"
    )
    _assert_second_form_question(
        response.questions[1], relationship="visits to", prompt="Does John travel here a lot?"
    )
    assert response.question is response.questions[0]
    assert response.extra_questions == [response.questions[1]]
    assert response.extra_questions[0].raw == QUESTION_VISITS_TO
    assert response.results == []  # expectAnswers: nil
    assert response.raw == body
    # Only the main question is "pending"; extras are available on the response.
    assert session.pending_question is response.question


# source: /tmp/rb-sdk-go/session_test.go:481 "Simple response that returns a question with float object"
def test_response_simple_response_that_returns_a_question_with_float_object() -> None:
    rb, recorder = make_client(
        _go_server(200, {"question": QUESTION_LIVES_IN_OBJECT_12}), api_key=GO_API_KEY
    )
    session = rb.start("kmid")

    # Go: QAnswer{..., Object: float64(12), CF: "100"}
    response = session.respond(
        [{"subject": "John", "relationship": "Speaks", "object": 12.0, "cf": "100"}]
    )

    assert len(recorder) == 2  # expectCalls: 2
    _assert_start_call(recorder.calls[0])
    # Go expectBody: {"answers":[{"subject":"John","relationship":"Speaks","object":12,"cf":"100"}]}
    _assert_response_call(
        recorder.calls[1],
        [{"subject": "John", "relationship": "Speaks", "object": 12.0, "certainty": 100}],
    )
    sent = recorder.calls[1].json["answers"][0]
    assert sent["object"] == 12
    # DIFFERENCE(go): Go's encoding/json writes float64(12) as `12`; Python keeps the float and
    # writes `12.0` on the wire. Both decode to the number 12.
    assert isinstance(sent["object"], float)
    assert re.search(rb'"object":\s*12\.0', recorder.calls[1].content)

    assert isinstance(response, QuestionResponse)
    assert len(response.questions) == 1
    # The engine echoes the object back as the string "12" (Go: Object: "12").
    _assert_second_form_question(
        response.questions[0],
        relationship="lives in",
        prompt="Where does John live?",
        expected_object="12",
    )
    assert isinstance(response.question.object, str)
    assert response.results == []  # expectAnswers: nil
    assert response.raw == {"question": QUESTION_LIVES_IN_OBJECT_12}


# source: /tmp/rb-sdk-go/session_test.go:530 "Bad request"
def test_response_bad_request() -> None:
    rb, recorder = make_client(_go_server(400, text="Foo bar baz"), api_key=GO_API_KEY)
    session = rb.start("kmid")

    # DIFFERENCE(go): Go returns (nil, nil, errors.New("API returned error 400: Foo bar baz"));
    # Python raises BadRequestError (the 400 subclass of RainbirdAPIError).
    with pytest.raises(BadRequestError, match="Foo bar baz") as info:
        session.respond([GO_ANSWER])

    _assert_go_error(info.value, 400, "/success-id/response")
    assert len(recorder) == 2  # expectCalls: 2 (the answers were sent before the 400 came back)
    _assert_start_call(recorder.calls[0])
    _assert_response_call(recorder.calls[1], [PY_ANSWER_SENT])
    # Go: expectQuestions nil, expectAnswers nil -> Python leaves the session state untouched.
    assert session.last_response is None
    assert session.pending_question is None


# source: /tmp/rb-sdk-go/session_test.go:549 "Internal server error"
def test_response_internal_server_error() -> None:
    rb, recorder = make_client(_go_server(500, text="Foo bar baz"), api_key=GO_API_KEY)
    session = rb.start("kmid")

    # DIFFERENCE(go): Go returns errors.New("API returned error 500: Foo bar baz"); Python raises
    # ServerError. A POST is never retried on 500 (even with the default RetryPolicy, which only
    # replays POSTs on 429/503), so the server still sees exactly two calls.
    with pytest.raises(ServerError, match="Foo bar baz") as info:
        session.respond([GO_ANSWER])

    _assert_go_error(info.value, 500, "/success-id/response")
    assert len(recorder) == 2  # expectCalls: 2
    _assert_start_call(recorder.calls[0])
    _assert_response_call(recorder.calls[1], [PY_ANSWER_SENT])
    assert session.last_response is None
    assert session.pending_question is None


# --------------------------------------------------------------------------- TestSessionUndo
# source: /tmp/rb-sdk-go/session_test.go:623 TestSessionUndo


# source: /tmp/rb-sdk-go/session_test.go:635 "Success back to question"
def test_undo_success_back_to_question() -> None:
    rb, recorder = make_client(_go_server(200, {"question": QUESTION_LIVES_IN}), api_key=GO_API_KEY)
    session = rb.start("kmid")
    assert session.id == "success-id"

    response = session.undo()

    assert len(recorder) == 2  # Go: assert.Equal(t, int64(2), calls)
    _assert_start_call(recorder.calls[0])
    _assert_undo_call(recorder.calls[1])

    # DIFFERENCE(go): Go returns ([]Question, []Answer, error); Python returns one QuestionResponse.
    assert isinstance(response, QuestionResponse)
    assert response.is_question is True
    assert response.is_result is False
    assert len(response.questions) == 1
    _assert_second_form_question(
        response.questions[0], relationship="lives in", prompt="Where does John live?"
    )
    assert response.question is response.questions[0]
    assert response.extra_questions == []
    assert response.results == []  # expectAnswers: nil
    assert response.raw == {"question": QUESTION_LIVES_IN}
    # undo() tracks the returned question as the new pending one.
    assert session.last_response is response
    assert session.pending_question is response.question


# source: /tmp/rb-sdk-go/session_test.go:672 "Bad request"
def test_undo_bad_request() -> None:
    rb, recorder = make_client(_go_server(400, text="Foo bar baz"), api_key=GO_API_KEY)
    session = rb.start("kmid")

    # DIFFERENCE(go): Go returns errors.New("API returned error 400: Foo bar baz"); Python raises
    # BadRequestError.
    with pytest.raises(BadRequestError, match="Foo bar baz") as info:
        session.undo()

    _assert_go_error(info.value, 400, "/success-id/undo")
    assert len(recorder) == 2  # Go: assert.Equal(t, int64(2), calls)
    _assert_start_call(recorder.calls[0])
    _assert_undo_call(recorder.calls[1])
    assert session.last_response is None
    assert session.pending_question is None


# source: /tmp/rb-sdk-go/session_test.go:681 "Internal server error"
def test_undo_internal_server_error() -> None:
    rb, recorder = make_client(_go_server(500, text="Foo bar baz"), api_key=GO_API_KEY)
    session = rb.start("kmid")

    # DIFFERENCE(go): Go returns errors.New("API returned error 500: Foo bar baz"); Python raises
    # ServerError; POST /undo is not retried on 500 so the server sees exactly two calls.
    with pytest.raises(ServerError, match="Foo bar baz") as info:
        session.undo()

    _assert_go_error(info.value, 500, "/success-id/undo")
    assert len(recorder) == 2  # Go: assert.Equal(t, int64(2), calls)
    _assert_start_call(recorder.calls[0])
    _assert_undo_call(recorder.calls[1])
    assert session.last_response is None
    assert session.pending_question is None
