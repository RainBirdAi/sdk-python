"""Port of the JavaScript base SDK analysis-endpoint tests to the Python SDK.

Sources (``/tmp/rb-sdk-js/packages/base/lib/__tests__/``):

- ``interaction_log.test.js`` ``describe("interactionLog")`` exercising ``interaction_log.ts``
  (``GET {baseURL}/analysis/interactions/{sessionID}``, optional ``x-interaction-key``).
  Python counterpart: ``Session.interactions(interaction_key=...)``.
  Cases: should perform simple GET for interactionLog / should throw fetch-generated errors /
  should handle 404 / should handle 500 / should handle 400 / should call the evidence endpoint
  with the evidence key header (misnamed in JS: it asserts the ``x-interaction-key`` header).
- ``session.test.js`` ``describe("knowledgeMap")`` and ``describe("facts")`` exercising
  ``session.ts`` (``GET {baseURL}/analysis/session/{sessionID}?filter=version|facts`` with
  ``Authorization: Basic base64(apiKey + ":")``). Python counterparts:
  ``Session.knowledge_map()`` and ``Session.facts()``.
  Cases (each describe): should perform simple GET / should throw fetch-generated errors /
  should handle 404 / should handle 500 / should handle 400.

Every JS case is mapped one-to-one with the original canned bodies. Because Python returns
typed models and raises typed exceptions where JS hands back ``response.json()`` untouched and
throws ``Error("API error code: N")`` or the fetch error itself, every mapped case is "adapted":
the request-side assertions (one GET, path, query, headers, no body) are the same, while the
result/error assertions are Python's equivalents. A few extra companion tests show the
Python-specific counterpart of what JS asserts (the parsed
``InteractionEvent``/``KnowledgeMapVersion``/``SessionFacts`` models for bodies in the shapes
declared by ``domain.ts``, the client-level ``interaction_key`` default and the ``relationships``
filter), because the verbatim JS body ``{"result": {"foo": "bar"}}`` is not a valid payload for
any of these endpoints.

Differences found (Python behaviour is asserted; JS behaviour is noted inline):

- Auth: JS ``knowledgeMap``/``facts`` send ``Authorization: Basic base64(apiKey + ":")`` and
  ``interactionLog`` sends no credentials at all; Python sends ``X-API-Key`` (plus ``User-Agent``
  and ``Accept: application/json``) on every request and never ``Authorization``.
- ``Content-Type``: JS sets ``Content-Type: application/json`` on these body-less GETs; Python
  sends no ``Content-Type`` on a GET.
- Query string: JS ``interactionLog`` has none; Python asks for ``?format=json`` (or
  ``?format=csv`` with ``format="csv"``). ``filter=version`` / ``filter=facts`` match JS.
- Return values: JS hands back ``response.json()`` untouched (the tests use
  ``{"result": {"foo": "bar"}}``). Python parses into models: ``knowledge_map()`` yields a
  ``KnowledgeMapVersion`` with every field ``None`` and the body on ``.raw``; ``facts()`` yields an
  empty ``SessionFacts`` with the body on ``.raw``; ``interactions()`` validates the shape and
  raises ``UnexpectedResponseError("expected a list of interaction events")`` for that body
  (``.body`` keeps it, ``.status_code == 200``).
- Errors: JS throws ``Error("API error code: N")``; Python raises the typed ``RainbirdAPIError``
  subclass (``NotFoundError``/``ServerError``/``BadRequestError``) with ``.status_code``, a default
  ``.message`` (the ``{}`` body carries none), ``.errors == []``, ``.body == {}``, ``.method`` and
  ``.url``; ``str(exc)`` is ``"HTTP N calling GET <url>: <message>"``. A rejected fetch becomes
  ``RainbirdConnectionError`` whose message wraps the underlying text with the method and URL
  instead of passing it through verbatim; the original exception is ``__cause__``.
- SDK inconsistency pinned here: ``RainbirdConnectionError.url`` omits the query string
  (``.../analysis/interactions/{sid}``) whereas ``RainbirdAPIError.url`` includes it
  (``...?format=json``, ``...?filter=version``), because the transport reports ``base + path``
  for transport failures but ``str(request.url)`` for HTTP errors.
- ``Session.interactions(interaction_key="")`` silently drops the client-level
  ``Rainbird(interaction_key=...)`` default: an empty string counts as "given" but is falsy, so
  no ``x-interaction-key`` header is sent at all. Pinned in the companion test.
- API shape: JS functions take ``(baseURL, apiKey, sessionID)``; Python configures the client
  once and calls the methods on ``rb.resume(session_id)`` (``resume`` sends nothing). The JS error
  tests pass ``{}`` as the session id (URL ``.../[object Object]``); Python is given a string.
- ``Km.versionNumber`` from ``domain.ts`` is honoured by ``KnowledgeMapVersion.version_number``
  since 1.1.0 (before, it was derived from ``versionStatus`` only); without the field Python
  still derives it (``"Version3"`` -> 3, ``"Draft"`` -> ``None``). ``Facts.global`` is
  ``SessionFacts.global_`` (Python keyword).
- ``SessionFact.source`` is parsed into the ``FactSource`` str-enum (equal to the plain string).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from rainbird_ai import (
    BadRequestError,
    FactSource,
    InteractionEvent,
    KnowledgeMapVersion,
    NotFoundError,
    RainbirdAPIError,
    RainbirdConnectionError,
    ServerError,
    SessionFacts,
    UnexpectedResponseError,
)
from tests.parity.support import BASE_URL, Call, make_client, raise_transport_error, respond

# Values used by the JS tests. interaction_log.test.js uses "thisisthesession" for the happy
# paths and the object {} for the error paths (which JS renders as "[object Object]" in the URL);
# session.test.js always uses "thisisthesessionid" and the api key "thisistheapikey".
SESSION_ID = "thisisthesession"
ANALYSIS_SESSION_ID = "thisisthesessionid"
JS_API_KEY = "thisistheapikey"
INTERACTION_KEY = "thisistheinteractionkey"

# The canned 200 body every JS "simple GET" case returns and expects back verbatim.
JS_BODY = {"result": {"foo": "bar"}}

INTERACTIONS_URL = f"{BASE_URL}/analysis/interactions/{SESSION_ID}"
SESSION_URL = f"{BASE_URL}/analysis/session/{ANALYSIS_SESSION_ID}"

# Python's default messages for an error body without any error text (here JS's `{}`).
DEFAULT_MESSAGES = {
    400: "Your request is missing a required parameter or a value is invalid",
    404: "The resource (session, knowledge map or fact) could not be found",
    500: "Rainbird server error",
}

# A body in the shape the Python interaction log expects (JS types interactionLog as `any`).
INTERACTION_LOG_BODY = [
    {
        "event": "start",
        "created": "2022-04-20T13:13:31.000Z",
        "values": {
            "start": {
                "useDraft": False,
                "kmVersionID": "5042ae3a-723a-45fa-bd6c-2d09e96e75f6",
                "sessionID": SESSION_ID,
            }
        },
    },
    {
        "event": "query",
        "created": "2022-04-20T13:13:44.532Z",
        "values": {"query": {"subject": "Tom", "relationship": "speaks"}},
    },
]

# A body in the shape of the `Km` interface from domain.ts (top-level, no `km` wrapper).
JS_KM_BODY = {
    "id": "5042ae3a-723a-45fa-bd6c-2d09e96e75f6",
    "name": "concept types",
    "versionID": "8b1f3c2e-0d4a-4f6b-9c7d-1e2f3a4b5c6d",
    "versionNumber": 3,
    "versionCreated": "2022-04-20T13:13:31.000Z",
    "versionStatus": "Version3",
}

# A body in the shape of the `Facts` interface from domain.ts (top-level, no `facts` wrapper),
# one `SessionFact` per scope.
JS_FACTS_BODY = {
    "global": [
        {
            "id": "WA:KF:global1",
            "source": "knowledgemap",
            "subject": {"value": "France", "dataType": "string", "concept": "country"},
            "relationship": "national language",
            "object": {"value": "French", "dataType": "string", "concept": "language"},
            "certainty": 100,
        }
    ],
    "context": [
        {
            "id": "WA:CF:context1",
            "source": "injection",
            "subject": {"value": "Tom", "dataType": "string", "concept": "person"},
            "relationship": "lives in",
            "object": {"value": "Germany", "dataType": "string", "concept": "country"},
            "certainty": 90,
        }
    ],
    "local": [
        {
            "id": "WA:RF:local1",
            "source": "rule",
            "subject": {"value": "Tom", "dataType": "string", "concept": "person"},
            "relationship": "speaks",
            "object": {"value": "German", "dataType": "string", "concept": "language"},
            "certainty": 75,
        }
    ],
}


def _check_get(call: Call, path: str, query: dict[str, str], api_key: str) -> None:
    """The Python counterpart of JS's `Object.keys(call).sort() == ["headers", "method"]`,
    `call.method == "GET"` and `call.headers toMatchObject({...})` checks."""
    assert call.method == "GET"
    assert call.path == path
    assert call.query == query
    # JS's fetch init has no `body` key; Python sends an empty body on these GETs.
    assert call.content == b""
    assert call.json is None
    # DIFFERENCE(js): JS sets "Content-Type: application/json" on these body-less GETs; Python
    # sends no Content-Type header at all on a GET.
    assert "content-type" not in call.headers
    # DIFFERENCE(js): JS never sends X-API-Key (knowledgeMap/facts send
    # `Authorization: Basic ${btoa(apiKey + ":")}`, i.e. "Basic dGhpc2lzdGhlYXBpa2V5Og==" for
    # "thisistheapikey"; interactionLog sends no credentials); Python sends X-API-Key on every
    # request and never an Authorization header.
    assert call.headers["x-api-key"] == api_key
    assert "authorization" not in call.headers
    assert call.headers["accept"] == "application/json"
    assert call.headers["user-agent"].startswith("rainbird-ai-python/")


def _check_api_error(exc: RainbirdAPIError, status: int, url: str) -> None:
    """Python's typed counterpart of JS's `Error("API error code: N")` for a `{}` body."""
    # DIFFERENCE(js): JS throws a plain Error whose message is "API error code: <status>"; Python
    # raises a RainbirdAPIError subclass carrying the status, a default message (the `{}` body
    # has no error text), the parsed body and where the request went.
    assert exc.status_code == status
    assert exc.message == DEFAULT_MESSAGES[status]
    assert exc.errors == []
    assert exc.body == {}
    assert exc.method == "GET"
    assert exc.url == url
    assert str(exc) == f"HTTP {status} calling GET {url}: {DEFAULT_MESSAGES[status]}"


# ---------------------------------------------------------------- describe("interactionLog")


# source: interaction_log.test.js:11 should perform simple GET for interactionLog
def test_should_perform_simple_get_for_interaction_log() -> None:
    rb, rec = make_client(respond(200, JS_BODY))

    # DIFFERENCE(js): JS resolves with the body untouched ({"result": {"foo": "bar"}}); Python
    # validates the interaction-log shape (a list of events, or an envelope holding one) and
    # raises UnexpectedResponseError for this body, keeping it on `.body`.
    with pytest.raises(UnexpectedResponseError) as info:
        rb.resume(SESSION_ID).interactions()
    assert info.value.message == "expected a list of interaction events"
    assert info.value.body == JS_BODY
    assert info.value.status_code == 200

    # The request itself matches what JS sends (fetch called once, GET, no body).
    assert len(rec) == 1
    # DIFFERENCE(js): JS requests `{baseURL}/analysis/interactions/{sid}` with no query string;
    # Python adds `?format=json`.
    _check_get(rec.last, f"/analysis/interactions/{SESSION_ID}", {"format": "json"}, "test-api-key")
    assert str(rec.last.url) == f"{INTERACTIONS_URL}?format=json"
    # JS: `call.headers toMatchObject({})` -- no interaction key was given, so none is sent.
    assert "x-interaction-key" not in rec.last.headers


# source: interaction_log.test.js:11 should perform simple GET for interactionLog (adapted:
# the same call with a body in the shape Python parses, so the "returns the payload" half of the
# JS assertion has a counterpart)
def test_should_perform_simple_get_for_interaction_log_parses_events() -> None:
    rb, rec = make_client(respond(200, INTERACTION_LOG_BODY))

    events = rb.resume(SESSION_ID).interactions()

    assert len(rec) == 1
    _check_get(rec.last, f"/analysis/interactions/{SESSION_ID}", {"format": "json"}, "test-api-key")
    assert "x-interaction-key" not in rec.last.headers
    # DIFFERENCE(js): JS returns the raw JSON; Python returns list[InteractionEvent] and keeps
    # each original object on `.raw`.
    assert isinstance(events, list) and len(events) == 2
    assert all(isinstance(event, InteractionEvent) for event in events)
    assert [event.event for event in events] == ["start", "query"]
    assert events[0].created == datetime(2022, 4, 20, 13, 13, 31, tzinfo=UTC)
    assert events[1].created == datetime(2022, 4, 20, 13, 13, 44, 532000, tzinfo=UTC)
    assert events[0].values["start"]["sessionID"] == SESSION_ID
    assert events[1].values == {"query": {"subject": "Tom", "relationship": "speaks"}}
    assert [event.to_dict() for event in events] == INTERACTION_LOG_BODY


# source: interaction_log.test.js:31 should throw fetch-generated errors
def test_interaction_log_should_throw_fetch_generated_errors() -> None:
    rb, rec = make_client(raise_transport_error(httpx.ConnectError("This was generated by fetch")))

    # DIFFERENCE(js): JS rejects with the fetch error itself (message "This was generated by
    # fetch"); Python wraps the transport failure in RainbirdConnectionError whose message adds
    # the method and URL, and chains the original exception as __cause__.
    with pytest.raises(RainbirdConnectionError) as info:
        rb.resume(SESSION_ID).interactions()

    assert str(info.value) == f"could not reach GET {INTERACTIONS_URL}: This was generated by fetch"
    assert info.value.method == "GET"
    # SDK inconsistency: the connection error's .url has no "?format=json", unlike the .url of the
    # RainbirdAPIError raised for a 404/500/400 on the very same call (see _check_api_error).
    assert info.value.url == INTERACTIONS_URL
    assert "?" not in info.value.url
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert str(info.value.__cause__) == "This was generated by fetch"
    assert len(rec) == 1


# source: interaction_log.test.js:41 should handle 404
def test_interaction_log_should_handle_404() -> None:
    rb, _ = make_client(respond(404, {}))

    with pytest.raises(NotFoundError) as info:
        rb.resume(SESSION_ID).interactions()

    _check_api_error(info.value, 404, f"{INTERACTIONS_URL}?format=json")


# source: interaction_log.test.js:53 should handle 500
def test_interaction_log_should_handle_500() -> None:
    rb, _ = make_client(respond(500, {}))

    with pytest.raises(ServerError) as info:
        rb.resume(SESSION_ID).interactions()

    _check_api_error(info.value, 500, f"{INTERACTIONS_URL}?format=json")


# source: interaction_log.test.js:65 should handle 400
def test_interaction_log_should_handle_400() -> None:
    rb, _ = make_client(respond(400, {}))

    with pytest.raises(BadRequestError) as info:
        rb.resume(SESSION_ID).interactions()

    _check_api_error(info.value, 400, f"{INTERACTIONS_URL}?format=json")


# source: interaction_log.test.js:76 should call the evidence endpoint with the evidence key header
def test_should_call_the_evidence_endpoint_with_the_evidence_key_header() -> None:
    # Despite its name the JS case calls interactionLog with an interaction key and asserts the
    # x-interaction-key header.
    rb, rec = make_client(respond(200, JS_BODY))

    # DIFFERENCE(js): JS returns {"result": {"foo": "bar"}} untouched; Python rejects that body
    # for the interaction log (see test_should_perform_simple_get_for_interaction_log).
    with pytest.raises(UnexpectedResponseError) as info:
        rb.resume(SESSION_ID).interactions(interaction_key=INTERACTION_KEY)
    assert info.value.body == JS_BODY

    assert len(rec) == 1
    _check_get(rec.last, f"/analysis/interactions/{SESSION_ID}", {"format": "json"}, "test-api-key")
    assert rec.last.headers["x-interaction-key"] == INTERACTION_KEY


# source: interaction_log.test.js:76 should call the evidence endpoint with the evidence key header
# (adapted: JS passes the key per call; Python can also set it once on the client, and a per-call
# key overrides that default. Uses a body Python can parse so the result is checked too.)
def test_interaction_key_default_from_client_and_per_call_override() -> None:
    rb, rec = make_client(respond(200, INTERACTION_LOG_BODY), interaction_key="client-default-key")
    session = rb.resume(SESSION_ID)

    events = session.interactions()
    assert rec.last.headers["x-interaction-key"] == "client-default-key"
    assert [event.event for event in events] == ["start", "query"]

    events = session.interactions(interaction_key=INTERACTION_KEY)
    assert rec.last.headers["x-interaction-key"] == INTERACTION_KEY
    assert [event.event for event in events] == ["start", "query"]

    # Pinned SDK behaviour: an explicit empty key is "given" (not None) so it replaces the client
    # default, but it is falsy so no header is sent at all -- an undocumented opt-out.
    events = session.interactions(interaction_key="")
    assert "x-interaction-key" not in rec.last.headers
    assert [event.event for event in events] == ["start", "query"]

    assert len(rec) == 3
    for call in rec.calls:
        _check_get(call, f"/analysis/interactions/{SESSION_ID}", {"format": "json"}, "test-api-key")


# ----------------------------------------------------------------- describe("knowledgeMap")


# source: session.test.js:11 should perform simple GET
def test_knowledge_map_should_perform_simple_get() -> None:
    rb, rec = make_client(respond(200, JS_BODY), api_key=JS_API_KEY)

    result = rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    assert len(rec) == 1
    _check_get(
        rec.last, f"/analysis/session/{ANALYSIS_SESSION_ID}", {"filter": "version"}, JS_API_KEY
    )
    assert str(rec.last.url) == f"{SESSION_URL}?filter=version"
    # DIFFERENCE(js): JS resolves with the body untouched ({"result": {"foo": "bar"}}); Python
    # returns a KnowledgeMapVersion whose fields are all None for this body and whose `.raw` /
    # `.to_dict()` keep the JSON verbatim.
    assert isinstance(result, KnowledgeMapVersion)
    assert result.to_dict() == JS_BODY
    assert result.raw == JS_BODY
    assert result.id is None
    assert result.name is None
    assert result.version_id is None
    assert result.version_status is None
    assert result.version_created is None
    assert result.is_draft is False
    assert result.version_number is None


# source: session.test.js:11 should perform simple GET (adapted: a body in the shape of the `Km`
# interface from domain.ts, so the "returns the payload" half of the assertion is meaningful)
def test_knowledge_map_parses_js_km_shape() -> None:
    rb, rec = make_client(respond(200, JS_KM_BODY), api_key=JS_API_KEY)

    result = rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    assert len(rec) == 1
    _check_get(
        rec.last, f"/analysis/session/{ANALYSIS_SESSION_ID}", {"filter": "version"}, JS_API_KEY
    )
    assert result.to_dict() == JS_KM_BODY
    assert result.id == "5042ae3a-723a-45fa-bd6c-2d09e96e75f6"
    assert result.name == "concept types"
    assert result.version_id == "8b1f3c2e-0d4a-4f6b-9c7d-1e2f3a4b5c6d"
    assert result.version_status == "Version3"
    assert result.version_created == datetime(2022, 4, 20, 13, 13, 31, tzinfo=UTC)
    assert result.is_draft is False
    # `Km.versionNumber` is a field in domain.ts; Python's `version_number` returns the explicit
    # field when present (since 1.1.0; before, it only derived it from `versionStatus`).
    assert result.version_number == 3
    assert result.raw["versionNumber"] == 3

    # Same call against a draft body: the explicit versionNumber: 3 wins over the Draft status.
    draft_body = {**JS_KM_BODY, "versionStatus": "Draft"}
    rb, rec = make_client(respond(200, draft_body), api_key=JS_API_KEY)
    draft = rb.resume(ANALYSIS_SESSION_ID).knowledge_map()
    assert len(rec) == 1
    assert isinstance(draft, KnowledgeMapVersion)
    assert draft.to_dict() == draft_body
    assert draft.version_status == "Draft"
    assert draft.is_draft is True
    assert draft.version_number == 3
    assert draft.raw["versionNumber"] == 3

    # Without the field the number is still derived from the status, as before.
    derived_body = {k: v for k, v in JS_KM_BODY.items() if k != "versionNumber"}
    rb, _ = make_client(respond(200, derived_body), api_key=JS_API_KEY)
    assert rb.resume(ANALYSIS_SESSION_ID).knowledge_map().version_number == 3


# source: session.test.js:35 should throw fetch-generated errors
def test_knowledge_map_should_throw_fetch_generated_errors() -> None:
    rb, rec = make_client(
        raise_transport_error(httpx.ConnectError("This was generated by fetch")), api_key=JS_API_KEY
    )

    # DIFFERENCE(js): JS rejects with the fetch error's own message; Python wraps it in
    # RainbirdConnectionError (method + URL added, original chained as __cause__).
    with pytest.raises(RainbirdConnectionError) as info:
        rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    assert str(info.value) == f"could not reach GET {SESSION_URL}: This was generated by fetch"
    assert info.value.method == "GET"
    # SDK inconsistency: no "?filter=version" here, unlike RainbirdAPIError.url for the same call.
    assert info.value.url == SESSION_URL
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert str(info.value.__cause__) == "This was generated by fetch"
    assert len(rec) == 1


# source: session.test.js:45 should handle 404
def test_knowledge_map_should_handle_404() -> None:
    rb, _ = make_client(respond(404, {}), api_key=JS_API_KEY)

    with pytest.raises(NotFoundError) as info:
        rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    _check_api_error(info.value, 404, f"{SESSION_URL}?filter=version")


# source: session.test.js:57 should handle 500
def test_knowledge_map_should_handle_500() -> None:
    rb, _ = make_client(respond(500, {}), api_key=JS_API_KEY)

    with pytest.raises(ServerError) as info:
        rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    _check_api_error(info.value, 500, f"{SESSION_URL}?filter=version")


# source: session.test.js:69 should handle 400
def test_knowledge_map_should_handle_400() -> None:
    rb, _ = make_client(respond(400, {}), api_key=JS_API_KEY)

    with pytest.raises(BadRequestError) as info:
        rb.resume(ANALYSIS_SESSION_ID).knowledge_map()

    _check_api_error(info.value, 400, f"{SESSION_URL}?filter=version")


# ------------------------------------------------------------------------ describe("facts")


# source: session.test.js:87 should perform simple GET
def test_facts_should_perform_simple_get() -> None:
    rb, rec = make_client(respond(200, JS_BODY), api_key=JS_API_KEY)

    result = rb.resume(ANALYSIS_SESSION_ID).facts()

    assert len(rec) == 1
    # No `relationships` parameter is sent when the Python-only filter is not used.
    _check_get(
        rec.last, f"/analysis/session/{ANALYSIS_SESSION_ID}", {"filter": "facts"}, JS_API_KEY
    )
    assert str(rec.last.url) == f"{SESSION_URL}?filter=facts"
    # DIFFERENCE(js): JS resolves with the body untouched ({"result": {"foo": "bar"}}); Python
    # returns an empty SessionFacts whose `.raw` / `.to_dict()` keep the JSON verbatim.
    assert isinstance(result, SessionFacts)
    assert result.to_dict() == JS_BODY
    assert result.raw == JS_BODY
    assert result.global_ == []
    assert result.context == []
    assert result.local == []
    assert result.all == []
    assert len(result) == 0
    assert list(result) == []


# source: session.test.js:87 should perform simple GET (adapted: a body in the shape of the
# `Facts` interface from domain.ts, so the "returns the payload" half of the assertion is
# meaningful)
def test_facts_parses_js_facts_shape() -> None:
    rb, rec = make_client(respond(200, JS_FACTS_BODY), api_key=JS_API_KEY)

    result = rb.resume(ANALYSIS_SESSION_ID).facts()

    assert len(rec) == 1
    _check_get(
        rec.last, f"/analysis/session/{ANALYSIS_SESSION_ID}", {"filter": "facts"}, JS_API_KEY
    )
    assert result.to_dict() == JS_FACTS_BODY
    # DIFFERENCE(js): `Facts.global` is `SessionFacts.global_` in Python (`global` is a keyword).
    assert len(result.global_) == len(result.context) == len(result.local) == 1
    assert len(result) == 3
    # `.all` is global, then context, then local.
    assert [fact.id for fact in result.all] == ["WA:KF:global1", "WA:CF:context1", "WA:RF:local1"]
    assert [fact.id for fact in result] == [fact.id for fact in result.all]

    fact = result.global_[0]
    # DIFFERENCE(js): `source` is a plain string in domain.ts; Python parses it into the
    # FactSource str-enum, which still compares equal to the string.
    assert fact.source is FactSource.KNOWLEDGE_MAP
    assert fact.source == "knowledgemap"
    assert fact.subject.value == "France"
    assert fact.subject.concept == "country"
    assert fact.subject.data_type == "string"
    assert fact.relationship == "national language"
    assert fact.object.value == "French"
    assert fact.object.concept == "language"
    assert fact.certainty == 100
    assert fact.to_dict() == JS_FACTS_BODY["global"][0]

    assert result.context[0].source is FactSource.INJECTION
    assert result.context[0].certainty == 90
    assert result.local[0].source is FactSource.RULE
    assert str(result.local[0]) == "Tom speaks German (75%)"


# source: session.test.js:87 should perform simple GET (adapted: Python-only `relationships=`
# filter; JS `facts` has no such parameter and always sends just `?filter=facts`)
def test_facts_relationships_filter_param() -> None:
    rb, rec = make_client(respond(200, JS_FACTS_BODY), api_key=JS_API_KEY)
    session = rb.resume(ANALYSIS_SESSION_ID)

    session.facts(relationships="speaks")
    assert rec.last.query == {"filter": "facts", "relationships": "speaks"}

    session.facts(relationships=["speaks", "lives in"])
    assert rec.last.query == {"filter": "facts", "relationships": "speaks,lives in"}

    session.facts(relationships=None)
    assert rec.last.query == {"filter": "facts"}

    assert len(rec) == 3
    for call in rec.calls:
        assert call.method == "GET"
        assert call.path == f"/analysis/session/{ANALYSIS_SESSION_ID}"


# source: session.test.js:111 should throw fetch-generated errors
def test_facts_should_throw_fetch_generated_errors() -> None:
    rb, rec = make_client(
        raise_transport_error(httpx.ConnectError("This was generated by fetch")), api_key=JS_API_KEY
    )

    # DIFFERENCE(js): JS rejects with the fetch error's own message; Python wraps it in
    # RainbirdConnectionError (method + URL added, original chained as __cause__).
    with pytest.raises(RainbirdConnectionError) as info:
        rb.resume(ANALYSIS_SESSION_ID).facts()

    assert str(info.value) == f"could not reach GET {SESSION_URL}: This was generated by fetch"
    assert info.value.method == "GET"
    # SDK inconsistency: no "?filter=facts" here, unlike RainbirdAPIError.url for the same call.
    assert info.value.url == SESSION_URL
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert str(info.value.__cause__) == "This was generated by fetch"
    assert len(rec) == 1


# source: session.test.js:121 should handle 404
def test_facts_should_handle_404() -> None:
    rb, _ = make_client(respond(404, {}), api_key=JS_API_KEY)

    with pytest.raises(NotFoundError) as info:
        rb.resume(ANALYSIS_SESSION_ID).facts()

    _check_api_error(info.value, 404, f"{SESSION_URL}?filter=facts")


# source: session.test.js:133 should handle 500
def test_facts_should_handle_500() -> None:
    rb, _ = make_client(respond(500, {}), api_key=JS_API_KEY)

    with pytest.raises(ServerError) as info:
        rb.resume(ANALYSIS_SESSION_ID).facts()

    _check_api_error(info.value, 500, f"{SESSION_URL}?filter=facts")


# source: session.test.js:145 should handle 400
def test_facts_should_handle_400() -> None:
    rb, _ = make_client(respond(400, {}), api_key=JS_API_KEY)

    with pytest.raises(BadRequestError) as info:
        rb.resume(ANALYSIS_SESSION_ID).facts()

    _check_api_error(info.value, 400, f"{SESSION_URL}?filter=facts")
