"""Port of the Go SDK session-analysis tests to the Python SDK.

Source: /tmp/rb-sdk-go/client_test.go

- ``TestSessionKmVersion`` (lines 940-1015): ``Client.KnowledgeMapVersion(sessionID)`` ->
  ``rb.resume(sid).knowledge_map()`` (``GET /analysis/session/{sid}?filter=version``).
  Table entries: "Bad request", "Internal server error", "Returns version info".
- ``TestSessionFacts`` (lines 1017-1251): ``Client.SessionFacts(sessionID)`` ->
  ``rb.resume(sid).facts()`` (``GET /analysis/session/{sid}?filter=facts``).
  Table entries: "Bad request", "Internal server error", "Returns facts info".

Both Go methods share the ``session(sessionID, filter)`` helper in client.go, which builds the
URL by string concatenation (``"?filter=" + filterStr``), authenticates with
``req.SetBasicAuth(APIKey, "")`` and wraps any status >= 400 as
``fmt.Errorf("API returned error %d: %s", status, body)``.

Differences found (Python behaviour is asserted, Go behaviour noted inline):

- Auth: Go sends ``Authorization: Basic base64(apiKey + ":")``; Python sends ``X-API-Key`` and
  never an ``Authorization`` header.
- Errors: Go returns ``error`` values whose text is ``"API returned error 400: Foo bar baz"``;
  Python raises ``BadRequestError``/``ServerError`` (``RainbirdAPIError`` subclasses) with
  ``.status_code``, ``.message == "Foo bar baz"``, ``.body``, ``.errors``, ``.method`` and
  ``.url``; ``str(exc)`` is ``"HTTP 400 calling GET <url>: Foo bar baz"``.
- API shape: Go exposes the calls on the client (``client.KnowledgeMapVersion(sid)``); Python on
  a ``Session`` obtained with ``rb.resume(sid)`` (no request is made by ``resume``).
- ``KnowledgeMap.VersionNumber`` in Go is decoded from a ``versionNumber`` JSON field; Python's
  ``KnowledgeMapVersion.version_number`` is a property derived from ``versionStatus``
  (``"Version3"`` -> 3, drafts -> ``None``) and ignores an explicit ``versionNumber`` field.
- ``Fact.Source`` is a plain Go string; Python parses it into the ``FactSource`` str-enum
  (``FactSource.ANSWER == "answer"`` still holds).
- Go's ``encoding/json`` decodes every JSON number into ``float64`` (the test expects
  ``float64(8)``); Python keeps JSON integers as ``int`` (``8``, ``1655856000000``).
- Python's ``Session.facts`` accepts an extra ``relationships=`` filter that Go does not have;
  when omitted no ``relationships`` query parameter is sent, matching Go's URL.
- Go's ``ConceptInstance`` has a separate ``Type *string`` field (``expectFacts`` pins it to
  ``nil`` on every subject/object); Python's ``FactValue`` has no ``type`` attribute (``concept``
  falls back to a ``type`` key), so the port checks that no ``type`` key reached ``.raw``.
- Go's ``assert.Equal`` is type-strict (``true`` != ``1``; ``Fact.Certainty`` is an ``int``);
  the port asserts ``fact.object.value is True`` for the boolean fact and ``isinstance(..., int)``
  for the numeric object values and for ``certainty`` (Python keeps a JSON ``100.0`` as float).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    FactSource,
    KnowledgeMapVersion,
    RainbirdAPIError,
    ServerError,
    SessionFact,
    SessionFacts,
)
from tests.parity.support import BASE_URL, Handler, Recorder, make_client

# Values used by the Go tests.
GO_API_KEY = "1234567890-1234-1234-1234-1234567890ab"
SESSION_ID = "1234"

# client_test.go:966 "Returns version info" response body.
KM_VERSION_BODY = {
    "km": {
        "id": "5042ae3a-723a-45fa-bd6c-2d09e96e75f6",
        "name": "concept types",
        "versionID": "5042ae3a-723a-45fa-bd6c-2d09e96e75f6",
        "versionCreated": "2022-04-20T13:13:31.000Z",
        "versionStatus": "Draft",
    }
}

_RESULT_VALUE = "string singular 1 string plural 2 8 5.15164 1656028800000 1655856000000 true"

# client_test.go:1042 "Returns facts info" response body.
FACTS_BODY = {
    "facts": {
        "global": [],
        "context": [],
        "local": [
            {
                "id": "WA:AF:029f78551644e583e9214c6fa1cb85ee83949f97f3068ec0bca26b750e3106e9",
                "subject": {"concept": "Subject", "value": "Dan", "dataType": "string"},
                "relationship": "has date plural",
                "object": {"concept": "Date plural", "value": 1655856000000, "dataType": "date"},
                "certainty": 100,
                "source": "answer",
            },
            {
                "id": "WA:AF:49a8af17bddfb9a5cafe73ee3332cb056d4dc9e5e9895e4c2e2180c586523fe1",
                "subject": {"concept": "Subject", "value": "Dan", "dataType": "string"},
                "relationship": "has number singular",
                "object": {"concept": "Number singular", "value": 8, "dataType": "number"},
                "certainty": 100,
                "source": "answer",
            },
            {
                "id": "WA:AF:2cef310b47a6dd3072a59c8752e507753deedbda073eba2fa5189b9058ef795a",
                "subject": {"concept": "Subject", "value": "Dan", "dataType": "string"},
                "relationship": "has string plural",
                "object": {
                    "concept": "String plural",
                    "value": "string plural 2",
                    "dataType": "string",
                },
                "certainty": 100,
                "source": "answer",
            },
            {
                "id": "WA:AF:fcd65f85fc5f01e76575ae5b245e71f8fe258cfa1c6e2da3795620b08010a8ba",
                "subject": {"concept": "Subject", "value": "Dan", "dataType": "string"},
                "relationship": "has truth",
                "object": {"concept": "Truth", "value": True, "dataType": "boolean"},
                "certainty": 100,
                "source": "answer",
            },
            {
                "id": "WA:RF:e70b10add03a2745860a81b9625fe0d9fe62373ebcb599281a9fc3f6813318d3",
                "subject": {"concept": "Subject", "value": "Dan", "dataType": "string"},
                "relationship": "relationship",
                "object": {
                    "concept": "Result object",
                    "value": _RESULT_VALUE,
                    "dataType": "string",
                },
                "certainty": 100,
                "source": "rule",
            },
        ],
    }
}

# Go's expectFacts for "Returns facts info", one entry per Fact:
# (ID, Subject(concept, value, dataType), Relationship, Object(concept, value, dataType),
#  Certainty, Source). Go decodes the numeric values as float64; Python keeps them as int.
EXPECTED_LOCAL_FACTS = [
    (
        "WA:AF:029f78551644e583e9214c6fa1cb85ee83949f97f3068ec0bca26b750e3106e9",
        ("Subject", "Dan", "string"),
        "has date plural",
        ("Date plural", 1655856000000, "date"),
        100,
        "answer",
    ),
    (
        "WA:AF:49a8af17bddfb9a5cafe73ee3332cb056d4dc9e5e9895e4c2e2180c586523fe1",
        ("Subject", "Dan", "string"),
        "has number singular",
        ("Number singular", 8, "number"),
        100,
        "answer",
    ),
    (
        "WA:AF:2cef310b47a6dd3072a59c8752e507753deedbda073eba2fa5189b9058ef795a",
        ("Subject", "Dan", "string"),
        "has string plural",
        ("String plural", "string plural 2", "string"),
        100,
        "answer",
    ),
    (
        "WA:AF:fcd65f85fc5f01e76575ae5b245e71f8fe258cfa1c6e2da3795620b08010a8ba",
        ("Subject", "Dan", "string"),
        "has truth",
        ("Truth", True, "boolean"),
        100,
        "answer",
    ),
    (
        "WA:RF:e70b10add03a2745860a81b9625fe0d9fe62373ebcb599281a9fc3f6813318d3",
        ("Subject", "Dan", "string"),
        "relationship",
        ("Result object", _RESULT_VALUE, "string"),
        100,
        "rule",
    ),
]


def go_server(status: int, body: str | dict) -> Handler:
    """Mirror the Go httptest handler: assert GET, set application/json, write the raw body.

    The Go table stores the body as a string; the 400/500 entries write the plain text
    ``"Foo bar baz"`` under an ``application/json`` content type.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        content = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(
            status, content=content.encode(), headers={"Content-Type": "application/json"}
        )

    return handler


def assert_session_request(recorder: Recorder, filter_value: str) -> None:
    """Assertions on the request Go's ``session(sessionID, filter)`` helper would build."""
    assert len(recorder) == 1  # resume() makes no request; the analysis call makes exactly one
    call = recorder.last
    assert call.method == "GET"
    assert call.path == f"/analysis/session/{SESSION_ID}"
    assert call.query == {"filter": filter_value}
    assert call.json is None  # GET, no body
    # DIFFERENCE(go): Go's session helper calls req.SetBasicAuth(APIKey, ""), i.e. it sends
    # "Authorization: Basic base64(apiKey + ':')" (for GO_API_KEY that is
    # "Basic MTIzNDU2Nzg5MC0xMjM0LTEyMzQtMTIzNC0xMjM0NTY3ODkwYWI6"). Python sends X-API-Key on
    # every request and never sets an Authorization header.
    assert "authorization" not in call.headers
    assert call.headers["x-api-key"] == GO_API_KEY


def assert_go_api_error(
    exc: RainbirdAPIError, *, status: int, filter_value: str, expected_type: type
) -> None:
    """The Python counterpart of Go's ``errors.New("API returned error N: Foo bar baz")``."""
    assert type(exc) is expected_type
    assert exc.status_code == status
    # Python keeps the response text as the message (same text Go interpolates after ": ").
    assert exc.message == "Foo bar baz"
    assert exc.errors == ["Foo bar baz"]
    assert exc.body == "Foo bar baz"
    assert exc.method == "GET"
    assert exc.url == f"{BASE_URL}/analysis/session/{SESSION_ID}?filter={filter_value}"
    # DIFFERENCE(go): Go returns an error whose text is "API returned error 400: Foo bar baz";
    # Python raises a typed exception whose str() is "HTTP 400 calling GET <url>: Foo bar baz".
    assert str(exc) == (
        f"HTTP {status} calling GET {BASE_URL}/analysis/session/{SESSION_ID}"
        f"?filter={filter_value}: Foo bar baz"
    )


def assert_fact(fact: SessionFact, expected: tuple[Any, ...]) -> None:
    fact_id, subject, relationship, object_, certainty, source = expected
    assert fact.id == fact_id
    assert (fact.subject.concept, fact.subject.value, fact.subject.data_type) == subject
    assert fact.relationship == relationship
    assert (fact.object.concept, fact.object.value, fact.object.data_type) == object_
    # Go's assert.Equal is type-strict: the "has truth" fact must hold the JSON bool `true`, not
    # 1 or 1.0 (which Python tuple equality would accept).
    if isinstance(object_[1], bool):
        assert fact.object.value is object_[1]
    # Go's ConceptInstance has a separate Type *string field that expectFacts pins to nil on every
    # subject/object. Python's FactValue has no `type` attribute (`concept` falls back to a `type`
    # key), so the counterpart is: no `type` key reached either side and `concept` is the concept.
    assert "type" not in fact.subject.raw
    assert "type" not in fact.object.raw
    assert fact.subject.raw["concept"] == subject[0]
    assert fact.object.raw["concept"] == object_[0]
    # Go's Fact.Certainty is an `int` (evidence.go), so assert.Equal(100, ...) is type-strict and
    # decoding a JSON 100.0 or "100" into it fails. Python's _as_number keeps 100.0 as float but
    # coerces the string "100" to int, so the wire value on .raw is pinned to a JSON integer too.
    assert fact.certainty == certainty
    assert isinstance(fact.certainty, int) and not isinstance(fact.certainty, bool)
    assert fact.raw["certainty"] == certainty
    assert type(fact.raw["certainty"]) is int
    # DIFFERENCE(go): Go's Fact.Source is a plain string ("answer"/"rule"); Python parses it into
    # the FactSource str-enum, which still compares equal to the string.
    assert fact.source == source
    assert isinstance(fact.source, FactSource)
    assert fact.source is FactSource(source)
    # Go decodes JSON numbers as float64 (the Go test expects float64(8) and
    # float64(1655856000000)); Python's json keeps integers as int. The tuple comparison above
    # already checks the value (8 == 8.0 in Python); this pins the Python type.
    if isinstance(object_[1], int) and not isinstance(object_[1], bool):
        assert isinstance(fact.object.value, int)


# ----------------------------------------------------------------------- TestSessionKmVersion


# source: client_test.go:952 TestSessionKmVersion "Bad request"
def test_session_km_version_bad_request() -> None:
    rb, rec = make_client(go_server(400, "Foo bar baz"), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    # Go: expectKm == nil, expectErr == errors.New("API returned error 400: Foo bar baz").
    with pytest.raises(BadRequestError) as info:
        session.knowledge_map()

    assert_go_api_error(
        info.value, status=400, filter_value="version", expected_type=BadRequestError
    )
    assert_session_request(rec, "version")


# source: client_test.go:959 TestSessionKmVersion "Internal server error"
def test_session_km_version_internal_server_error() -> None:
    rb, rec = make_client(go_server(500, "Foo bar baz"), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    # Go: expectKm == nil, expectErr == errors.New("API returned error 500: Foo bar baz").
    # Retries are disabled by make_client (NO_RETRY), so the 500 surfaces at once, as in Go.
    with pytest.raises(ServerError) as info:
        session.knowledge_map()

    assert_go_api_error(info.value, status=500, filter_value="version", expected_type=ServerError)
    assert_session_request(rec, "version")


# source: client_test.go:966 TestSessionKmVersion "Returns version info"
def test_session_km_version_returns_version_info() -> None:
    rb, rec = make_client(go_server(200, KM_VERSION_BODY), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    km = session.knowledge_map()

    assert_session_request(rec, "version")
    assert isinstance(km, KnowledgeMapVersion)
    # Go's expectKm: &KnowledgeMap{ID, Name, VersionID, VersionCreated: &created,
    # VersionStatus: "Draft"} with VersionNumber nil.
    assert km.id == "5042ae3a-723a-45fa-bd6c-2d09e96e75f6"
    assert km.name == "concept types"
    assert km.version_id == "5042ae3a-723a-45fa-bd6c-2d09e96e75f6"
    created = datetime(2022, 4, 20, 13, 13, 31, tzinfo=timezone.utc)  # time.Parse(RFC3339, ...)
    assert km.version_created == created
    assert km.version_created is not None and km.version_created.tzinfo is not None
    assert km.version_status == "Draft"
    # Go's VersionNumber is a *int decoded from a "versionNumber" JSON field (absent here, so
    # nil). Python honours that field when present and otherwise derives version_number from
    # versionStatus, which is None for a draft; it also exposes is_draft.
    assert km.version_number is None
    assert km.is_draft is True
    # Python keeps the whole original body on .raw (Go only keeps the decoded struct).
    assert km.raw == KM_VERSION_BODY
    assert km.to_dict() == KM_VERSION_BODY


# ---------------------------------------------------------------------------- TestSessionFacts


# source: client_test.go:1028 TestSessionFacts "Bad request"
def test_session_facts_bad_request() -> None:
    rb, rec = make_client(go_server(400, "Foo bar baz"), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    # Go: expectFacts == nil, expectErr == errors.New("API returned error 400: Foo bar baz").
    with pytest.raises(BadRequestError) as info:
        session.facts()

    assert_go_api_error(info.value, status=400, filter_value="facts", expected_type=BadRequestError)
    assert_session_request(rec, "facts")


# source: client_test.go:1035 TestSessionFacts "Internal server error"
def test_session_facts_internal_server_error() -> None:
    rb, rec = make_client(go_server(500, "Foo bar baz"), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    # Go: expectFacts == nil, expectErr == errors.New("API returned error 500: Foo bar baz").
    with pytest.raises(ServerError) as info:
        session.facts()

    assert_go_api_error(info.value, status=500, filter_value="facts", expected_type=ServerError)
    assert_session_request(rec, "facts")


# source: client_test.go:1042 TestSessionFacts "Returns facts info"
def test_session_facts_returns_facts_info() -> None:
    rb, rec = make_client(go_server(200, FACTS_BODY), api_key=GO_API_KEY)
    session = rb.resume(SESSION_ID)

    facts = session.facts()

    # Go builds "?filter=facts" only; Python drops the unused relationships= parameter, so the
    # query string is the same.
    assert_session_request(rec, "facts")
    assert isinstance(facts, SessionFacts)
    # Go's expectFacts: Global: []Fact{}, Context: []Fact{}, Local: five facts.
    assert facts.global_ == []
    assert facts.context == []
    assert len(facts.local) == len(EXPECTED_LOCAL_FACTS) == 5
    for fact, expected in zip(facts.local, EXPECTED_LOCAL_FACTS, strict=True):
        assert isinstance(fact, SessionFact)
        assert_fact(fact, expected)
    # Go Flatten() (global + context + local) -> Python .all, __len__ and __iter__.
    assert facts.all == [*facts.global_, *facts.context, *facts.local] == facts.local
    assert len(facts) == 5
    assert list(facts) == facts.all
    # Every fact keeps its original JSON on .raw; the container keeps the whole body.
    assert [f.raw for f in facts.local] == FACTS_BODY["facts"]["local"]
    assert facts.raw == FACTS_BODY
    assert facts.to_dict() == FACTS_BODY
