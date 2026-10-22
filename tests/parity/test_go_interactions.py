"""Port of ``TestInteractionLog`` from the Go SDK (``/tmp/rb-sdk-go/client_test.go`` lines
548-938), which exercises ``Client.Interactions`` (``client.go:278-318``) and the typed per-event
decoding in ``interaction.go`` (``InteractionEvents``, ``parseEventTime``).

Go table entries, ported one test each (the Go ``httptest`` handler's method/RequestURI
assertions are repeated for every case through ``assert_go_request``):

* "Correct interaction log for Start returned"
* "Correct interaction log for Query, Questions returned"
* "Correct interaction log for Injects returned"
* "Correct interaction log for Answers returned"
* "Correct interaction log for Datasources, Result returned"
* "Bad request"
* "Internal server error"

Adapted checks for behaviour of ``client.go``/``interaction.go`` that the Go table does not
reach: the ``x-interaction-key`` header (given / ``nil`` / empty string), the request headers,
an unknown event name and an unparseable ``created`` value.

Python counterpart: ``Rainbird.resume(sid).interactions()`` -> ``list[InteractionEvent]``.

Differences found (each asserted below with a ``# DIFFERENCE(go)`` comment):

* Go decodes ``values`` into typed data per event (``Start``, ``[]Question``, ``[]QAnswer``,
  ``Query``, ``[]InjectFact``, ``[]Datasource``, ``[]Answer``) and exposes ``Event`` as an int
  enum; Python keeps ``InteractionEvent.event`` as the API string, ``.values`` as the raw dict
  and ``.raw`` as the whole entry. ``Question.from_dict``/``ResultItem.from_dict`` can be
  applied to those dicts by hand.
* Go requests ``/analysis/interactions/{sid}`` with no query string; Python always appends
  ``?format=json``.
* Go sends ``Content-Type: application/json`` and no credentials on this GET; Python sends
  ``X-API-Key``, ``Accept: application/json`` and ``User-Agent`` and no ``Content-Type``.
* Go returns ``errors.New("API returned error 400: Foo bar baz")`` plus a nil log; Python raises
  ``BadRequestError``/``ServerError`` carrying ``.status_code``, ``.message``, ``.body``.
* Go rejects the whole log on an unknown event name (``unsupported event: ...``) or an
  ``created`` value that is not RFC3339; Python keeps the entry and yields ``created=None``.
* ``str(ResultItem)`` is ``"Dan lives in England (100%)"`` where Go's ``Answer.String()`` gives
  ``"Dan - lives in - England [100%]"``.
* Go never retries; Python's default ``RetryPolicy`` (``max_retries=2``) retries a GET 500 twice,
  three requests in total, so the parity helper disables retries (``NO_RETRY``) to get Go's
  single-request behaviour.
* Go sends an empty ``x-interaction-key`` header when the caller passes a pointer to ``""``;
  Python treats an explicit ``interaction_key=""`` as "send no header", even when the client
  was built with a default ``Rainbird(interaction_key=...)``.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    InteractionEvent,
    Question,
    RainbirdAPIError,
    ResultItem,
    ServerError,
)
from tests.parity.support import BASE_URL, Handler, Recorder, make_client

# client_test.go:549-550
SESSION_ID = "1234-5678"
# client_test.go:928
GO_API_KEY = "1234567890-1234-1234-1234-1234567890ab"
# Go: time.Parse("2006-01-02T03:04:05", "2022-02-15T00:00:00") is 2022-02-15 00:00:00 UTC.
CREATED = dt.datetime(2022, 2, 15, 0, 0, 0, tzinfo=dt.UTC)
INTERACTIONS_PATH = f"/analysis/interactions/{SESSION_ID}"
# DIFFERENCE(go): Go's handler asserts r.RequestURI == "/analysis/interactions/1234-5678" (bare
# path, client_test.go:918); Python's request-URI always carries ?format=json.
PYTHON_REQUEST_URI = f"{INTERACTIONS_PATH}?format=json"

# --------------------------------------------------------------------------- canned bodies
# Copied verbatim from the Go table (client_test.go:563-851).

# client_test.go:563-575
START_LOG: list[dict[str, Any]] = [
    {
        "event": "start",
        "values": {
            "start": {
                "sessionID": "1",
                "useDraft": True,
                "kmVersionID": "1234-5678",
            }
        },
        "created": "2022-02-15T00:00:00Z",
    }
]

# client_test.go:592-635
QUERY_QUESTIONS_LOG: list[dict[str, Any]] = [
    {
        "event": "query",
        "values": {
            "query": {
                "subject": "Dan",
                "relationship": "speaks",
            }
        },
        "created": "2022-02-15T00:00:00Z",
    },
    {
        "event": "question",
        "values": {
            "questions": [
                {
                    "subject": "Dan",
                    "dataType": "string",
                    "relationship": "speaks",
                    "type": "Second Form Object",
                    "plural": False,
                    "allowCF": True,
                    "allowUnknown": False,
                    "canAdd": True,
                    "prompt": "Where does Dan live?",
                    "knownAnswers": [],
                },
                {
                    "subject": "Tom",
                    "dataType": "string",
                    "relationship": "speaks",
                    "type": "Second Form Object",
                    "plural": False,
                    "allowCF": True,
                    "allowUnknown": False,
                    "canAdd": True,
                    "prompt": "Where does Tom live?",
                    "knownAnswers": [],
                },
            ]
        },
        "created": "2022-02-15T00:00:00Z",
    },
]

# client_test.go:686-713
INJECT_LOG: list[dict[str, Any]] = [
    {
        "event": "inject",
        "values": {
            "facts": [
                {
                    "subject": "Dan",
                    "object": "English",
                    "relationship": "speaks",
                    "cf": "100",
                },
                {
                    "subject": "Dan",
                    "object": "German",
                    "relationship": "speaks",
                    "cf": "90",
                },
                {
                    "subject": "Dan",
                    "object": "French",
                    "relationship": "speaks",
                    "cf": "80",
                },
            ]
        },
        "created": "2022-02-15T00:00:00Z",
    }
]

# client_test.go:745-773
ANSWER_LOG: list[dict[str, Any]] = [
    {
        "event": "answer",
        "values": {
            "answers": [
                {
                    "subject": "Dan",
                    "object": "English",
                    "relationship": "speaks",
                    "cf": "100",
                    "answer": "England",
                },
                {
                    "subject": "Dan",
                    "object": "German",
                    "relationship": "speaks",
                    "cf": "90",
                },
                {
                    "subject": "Dan",
                    "object": "French",
                    "relationship": "speaks",
                    "cf": "80",
                },
            ]
        },
        "created": "2022-02-15T00:00:00Z",
    }
]

# client_test.go:808-851
DATASOURCE_RESULT_LOG: list[dict[str, Any]] = [
    {
        "event": "datasource",
        "values": {
            "datasources": [
                {
                    "relationship": "speaks",
                    "certainty": 100,
                },
                {
                    "relationship": "speaks",
                    "certainty": 90,
                },
                {
                    "relationship": "speaks",
                    "certainty": 80,
                },
            ]
        },
        "created": "2022-02-15T00:00:00Z",
    },
    {
        "event": "result",
        "values": {
            "results": [
                {
                    "certainty": 100,
                    "factID": "thisisthefactid",
                    "object": "England",
                    "relationship": "lives in",
                    "subject": "Dan",
                },
                {
                    "certainty": 90,
                    "factID": "thisisthefactid",
                    "object": "England",
                    "relationship": "lives in",
                    "subject": "Tom",
                },
            ]
        },
        "created": "2022-02-15T00:00:00Z",
    },
]


# --------------------------------------------------------------------------- helpers


def go_server(status: int, body: str) -> Handler:
    """Go's ``httptest`` handler (client_test.go:915-923): writes ``body`` under
    ``Content-Type: application/json`` whatever it contains (the error cases send plain text)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body.encode(), headers={"Content-Type": "application/json"}
        )

    return handler


def call_interactions(
    status: int,
    body: str,
    *,
    interaction_key: str | None = None,
    **client_kwargs: Any,
) -> tuple[list[InteractionEvent], Recorder]:
    """``client.Interactions(sessionID, nil)`` -> ``rb.resume(sid).interactions()``."""
    rb, recorder = make_client(go_server(status, body), api_key=GO_API_KEY, **client_kwargs)
    events = rb.resume(SESSION_ID).interactions(interaction_key=interaction_key)
    return events, recorder


def assert_go_request(recorder: Recorder) -> None:
    """The assertions Go's handler makes on the request (client_test.go:917-918)."""
    assert len(recorder) == 1
    call = recorder.last
    assert call.method == "GET"
    assert call.path == INTERACTIONS_PATH
    # DIFFERENCE(go): Go asserts the bare RequestURI "/analysis/interactions/1234-5678"; Python
    # sends "/analysis/interactions/1234-5678?format=json".
    assert call.url.raw_path.decode() == PYTHON_REQUEST_URI
    assert call.query == {"format": "json"}
    assert call.content == b""


# --------------------------------------------------------------------------- table entries


# source: client_test.go:560-588 "Correct interaction log for Start returned"
def test_correct_interaction_log_for_start_returned() -> None:
    events, recorder = call_interactions(200, json.dumps(START_LOG))
    assert_go_request(recorder)

    assert len(events) == 1
    (event,) = events
    assert isinstance(event, InteractionEvent)
    # DIFFERENCE(go): Go maps "start" to the StartEvent enum member and decodes values.start into
    # Start{SessionID, UseDraft, KmVersionID}; Python keeps the event name string and the values
    # dict untyped (this typed-vs-untyped difference applies to every event below).
    assert event.event == "start"
    assert event.created == CREATED
    start = event.values["start"]
    assert start["sessionID"] == "1"
    assert start["kmVersionID"] == "1234-5678"
    assert start["useDraft"] is True
    assert event.values == START_LOG[0]["values"]
    assert event.raw == START_LOG[0]


# source: client_test.go:589-682 "Correct interaction log for Query, Questions returned"
def test_correct_interaction_log_for_query_questions_returned() -> None:
    events, recorder = call_interactions(200, json.dumps(QUERY_QUESTIONS_LOG))
    assert_go_request(recorder)

    assert [event.event for event in events] == ["query", "question"]
    query_event, question_event = events

    # Go: Query{Subject: stringPtr("Dan"), Object: nil, Relationship: "speaks"}
    assert query_event.created == CREATED
    query = query_event.values["query"]
    assert query["subject"] == "Dan"
    assert query["relationship"] == "speaks"
    assert "object" not in query  # Go: Object nil
    assert query_event.values == QUERY_QUESTIONS_LOG[0]["values"]
    assert query_event.raw == QUERY_QUESTIONS_LOG[0]

    # Go: []Question{...} with the fields below; Python: the raw question dicts.
    assert question_event.created == CREATED
    questions = question_event.values["questions"]
    assert len(questions) == 2
    expected = [("Dan", "Where does Dan live?"), ("Tom", "Where does Tom live?")]
    for question, (subject, prompt) in zip(questions, expected, strict=True):
        assert question["allowCF"] is True
        assert question["allowUnknown"] is False
        assert question["canAdd"] is True
        assert "concepts" not in question  # Go: Concepts nil
        assert question["dataType"] == "string"
        assert question["knownAnswers"] == []  # Go: []KnownAnswer{}
        assert question["plural"] is False
        assert question["prompt"] == prompt
        assert question["relationship"] == "speaks"
        assert question["subject"] == subject
        assert "object" not in question  # Go: Object interface{}(nil)
        assert question["type"] == "Second Form Object"

        # Python-specific: the same typed view Go builds is one from_dict call away.
        typed = Question.from_dict(question)
        assert typed.subject == subject
        assert typed.relationship == "speaks"
        assert typed.object is None
        assert typed.type == "Second Form Object"
        assert typed.asks_for_object
        assert typed.prompt == prompt
        assert typed.data_type == "string"
        assert typed.plural is False
        assert typed.allow_cf is True
        assert typed.allow_unknown is False
        assert typed.can_add is True
        assert typed.known_answers == []
        assert typed.concepts == []  # Go: nil slice; Python normalises to an empty list
        assert typed.raw == question
    assert question_event.values == QUERY_QUESTIONS_LOG[1]["values"]
    assert question_event.raw == QUERY_QUESTIONS_LOG[1]


# source: client_test.go:683-741 "Correct interaction log for Injects returned"
def test_correct_interaction_log_for_injects_returned() -> None:
    events, recorder = call_interactions(200, json.dumps(INJECT_LOG))
    assert_go_request(recorder)

    assert len(events) == 1
    (event,) = events
    assert event.event == "inject"
    assert event.created == CREATED
    facts = event.values["facts"]
    assert len(facts) == 3
    # Go: []InjectFact{Subject, Object, Relationship, Certainty (json "cf", a string)}; the
    # "certainty" (*int) and "metadata" fields stay at their zero values.
    expected = [("English", "100"), ("German", "90"), ("French", "80")]
    for fact, (obj, cf) in zip(facts, expected, strict=True):
        assert fact["subject"] == "Dan"
        assert fact["object"] == obj
        assert fact["relationship"] == "speaks"
        assert fact["cf"] == cf
        assert isinstance(fact["cf"], str)  # kept as the API sent it, like Go's string field
        assert "certainty" not in fact
        assert "metadata" not in fact
    assert event.values == INJECT_LOG[0]["values"]
    assert event.raw == INJECT_LOG[0]


# source: client_test.go:742-804 "Correct interaction log for Answers returned"
def test_correct_interaction_log_for_answers_returned() -> None:
    events, recorder = call_interactions(200, json.dumps(ANSWER_LOG))
    assert_go_request(recorder)

    assert len(events) == 1
    (event,) = events
    assert event.event == "answer"
    assert event.created == CREATED
    answers = event.values["answers"]
    assert len(answers) == 3
    # Go: []QAnswer{Subject, Relationship, Object, CF (json "cf", a string), Certainty (*int),
    # Answer, Unanswered} (session.go:73-81); Go's expected literal leaves Certainty nil,
    # Unanswered false and, for the last two answers, Answer "" (all zero values).
    expected = [("English", "100", "England"), ("German", "90", ""), ("French", "80", "")]
    for answer, (obj, cf, text) in zip(answers, expected, strict=True):
        assert answer["subject"] == "Dan"
        assert answer["object"] == obj
        assert answer["relationship"] == "speaks"
        assert answer["cf"] == cf
        assert isinstance(answer["cf"], str)  # kept as the API sent it, like Go's string field
        assert answer.get("answer", "") == text
        assert "certainty" not in answer  # Go: Certainty nil
        assert "unanswered" not in answer  # Go: Unanswered false
    assert answers[0]["answer"] == "England"
    assert "answer" not in answers[1]  # Go: Answer ""
    assert "answer" not in answers[2]  # Go: Answer ""
    assert event.values == ANSWER_LOG[0]["values"]
    assert event.raw == ANSWER_LOG[0]


# source: client_test.go:805-893 "Correct interaction log for Datasources, Result returned"
def test_correct_interaction_log_for_datasources_result_returned() -> None:
    events, recorder = call_interactions(200, json.dumps(DATASOURCE_RESULT_LOG))
    assert_go_request(recorder)

    assert [event.event for event in events] == ["datasource", "result"]
    datasource_event, result_event = events

    # Go: []Datasource{Relationship, Certainty (int)}
    assert datasource_event.created == CREATED
    datasources = datasource_event.values["datasources"]
    assert datasources == [
        {"relationship": "speaks", "certainty": 100},
        {"relationship": "speaks", "certainty": 90},
        {"relationship": "speaks", "certainty": 80},
    ]
    for datasource in datasources:
        assert isinstance(datasource["certainty"], int)
    assert datasource_event.values == DATASOURCE_RESULT_LOG[0]["values"]
    assert datasource_event.raw == DATASOURCE_RESULT_LOG[0]

    # Go: []Answer{Subject, Relationship, Object, Certainty, FactID}
    assert result_event.created == CREATED
    results = result_event.values["results"]
    assert len(results) == 2
    expected = [("Dan", 100), ("Tom", 90)]
    for result, (subject, certainty) in zip(results, expected, strict=True):
        assert result["subject"] == subject
        assert result["relationship"] == "lives in"
        assert result["object"] == "England"
        assert result["certainty"] == certainty
        assert result["factID"] == "thisisthefactid"

        # Python-specific: ResultItem.from_dict gives the typed view Go decodes into Answer.
        item = ResultItem.from_dict(result)
        assert item.subject == subject
        assert item.relationship == "lives in"
        assert item.object == "England"
        assert item.certainty == certainty
        assert item.fact_id == "thisisthefactid"
        # DIFFERENCE(go): Go's Answer.String() renders "Dan - lives in - England [100%]";
        # Python's str(ResultItem) renders "Dan lives in England (100%)".
        assert str(item) == f"{subject} lives in England ({certainty}%)"
    assert result_event.values == DATASOURCE_RESULT_LOG[1]["values"]
    assert result_event.raw == DATASOURCE_RESULT_LOG[1]


# source: client_test.go:894-900 "Bad request"
def test_bad_request() -> None:
    rb, recorder = make_client(go_server(400, "Foo bar baz"), api_key=GO_API_KEY)
    # DIFFERENCE(go): Go returns (nil, errors.New("API returned error 400: Foo bar baz"))
    # (client.go:298-309); Python raises a typed BadRequestError instead of returning an error
    # value.
    with pytest.raises(BadRequestError) as info:
        rb.resume(SESSION_ID).interactions()
    assert_go_request(recorder)

    exc = info.value
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == 400
    assert exc.message == "Foo bar baz"
    assert exc.errors == ["Foo bar baz"]
    assert exc.body == "Foo bar baz"  # text body kept although the server labelled it JSON
    assert exc.method == "GET"
    assert exc.url == f"{BASE_URL}{PYTHON_REQUEST_URI}"
    assert str(exc) == f"HTTP 400 calling GET {BASE_URL}{PYTHON_REQUEST_URI}: Foo bar baz"


# source: client_test.go:901-907 "Internal server error"
def test_internal_server_error() -> None:
    rb, recorder = make_client(go_server(500, "Foo bar baz"), api_key=GO_API_KEY)
    # DIFFERENCE(go): Go returns (nil, errors.New("API returned error 500: Foo bar baz"));
    # Python raises ServerError. Go never retries; Python's default RetryPolicy (max_retries=2)
    # retries a GET 500 twice, i.e. three requests in total, so the parity helper builds the
    # client with NO_RETRY to reproduce Go's single request.
    with pytest.raises(ServerError) as info:
        rb.resume(SESSION_ID).interactions()
    assert_go_request(recorder)  # exactly one request, as in Go

    exc = info.value
    assert isinstance(exc, RainbirdAPIError)
    assert exc.status_code == 500
    assert exc.message == "Foo bar baz"
    assert exc.errors == ["Foo bar baz"]
    assert exc.body == "Foo bar baz"
    assert exc.method == "GET"
    assert exc.url == f"{BASE_URL}{PYTHON_REQUEST_URI}"
    assert str(exc) == f"HTTP 500 calling GET {BASE_URL}{PYTHON_REQUEST_URI}: Foo bar baz"


# --------------------------------------------------------------------------- adapted checks


# source: client.go:288-290 (x-interaction-key is set only when interactionKey != nil)
def test_x_interaction_key_header_only_when_given() -> None:
    events, recorder = call_interactions(200, "[]", interaction_key="secret-key")
    assert events == []
    assert recorder.last.headers["x-interaction-key"] == "secret-key"

    events, recorder = call_interactions(200, "[]")  # Go: nil pointer -> no header
    assert events == []
    assert "x-interaction-key" not in recorder.last.headers

    # Python extension (no Go counterpart): Rainbird(interaction_key=...) supplies the header
    # by default and the per-call argument still wins.
    rb, recorder = make_client(
        go_server(200, "[]"), api_key=GO_API_KEY, interaction_key="client-key"
    )
    session = rb.resume(SESSION_ID)
    assert session.interactions() == []
    assert recorder.last.headers["x-interaction-key"] == "client-key"
    assert session.interactions(interaction_key="explicit-key") == []
    assert recorder.last.headers["x-interaction-key"] == "explicit-key"
    assert len(recorder) == 2


# source: client.go:288-290 (a *string pointing at "" sets an empty x-interaction-key header)
def test_empty_interaction_key_sends_no_header() -> None:
    # DIFFERENCE(go): Go's caller can pass stringPtr(""), which is != nil, so Go sends
    # "x-interaction-key: " (empty value); Python's builder keeps the header only for a truthy
    # key (`if interaction_key`), so an explicit "" sends no header at all.
    events, recorder = call_interactions(200, "[]", interaction_key="")
    assert events == []
    assert "x-interaction-key" not in recorder.last.headers

    # Python-only: "" counts as "given" for the per-call override (Session.interactions uses
    # `is not None`), so it also suppresses a client-level Rainbird(interaction_key=...) default
    # rather than falling back to it.
    rb, recorder = make_client(
        go_server(200, "[]"), api_key=GO_API_KEY, interaction_key="client-key"
    )
    assert rb.resume(SESSION_ID).interactions(interaction_key="") == []
    assert "x-interaction-key" not in recorder.last.headers
    assert len(recorder) == 1


# source: client.go:287 (Content-Type: application/json, no credentials on Interactions)
def test_request_headers_differ_from_go() -> None:
    _, recorder = call_interactions(200, "[]")
    headers = recorder.last.headers
    # DIFFERENCE(go): Go sets Content-Type: application/json on the bodiless GET and sends no
    # credentials at all for the interaction log; Python sends X-API-Key on every request,
    # Accept: application/json and a User-Agent, and no Content-Type.
    assert headers["x-api-key"] == GO_API_KEY
    assert headers["accept"] == "application/json"
    assert headers["user-agent"].startswith("rainbird-ai-python/")
    assert "content-type" not in headers
    assert "authorization" not in headers


# source: interaction.go:75-77 InteractionEvents ("unsupported event: %s returned in the
# interactions log response")
def test_unknown_event_name_is_kept_not_rejected() -> None:
    body = [{"event": "bogus", "values": {"foo": "bar"}, "created": "2022-02-15T00:00:00Z"}]
    events, recorder = call_interactions(200, json.dumps(body))
    assert_go_request(recorder)
    # DIFFERENCE(go): Go fails the whole call with "unsupported event: bogus returned in the
    # interactions log response"; Python keeps the entry verbatim (tolerant parsing).
    assert [event.event for event in events] == ["bogus"]
    assert events[0].values == {"foo": "bar"}
    assert events[0].created == CREATED
    assert events[0].raw == body[0]


# source: interaction.go:203-208 parseEventTime (time.Parse(time.RFC3339) error fails the call)
@pytest.mark.parametrize("created", ["yesterday", ""], ids=["garbage", "empty"])
def test_unparseable_created_yields_none(created: str) -> None:
    body = [{"event": "start", "values": {"start": {"sessionID": "1"}}, "created": created}]
    events, recorder = call_interactions(200, json.dumps(body))
    assert_go_request(recorder)
    # DIFFERENCE(go): Go returns the time.Parse error and no log; Python keeps the event with
    # created=None and the original text on .raw.
    assert len(events) == 1
    assert events[0].event == "start"
    assert events[0].created is None
    assert events[0].values == {"start": {"sessionID": "1"}}
    assert events[0].raw["created"] == created
