"""Port of ``TestEvidence`` from the Go SDK (``/tmp/rb-sdk-go/client_test.go`` lines 327-547).

Source cases (one test per table entry, carrying the Go ``description``):

* ``Bad request``                              -> :func:`test_bad_request`
* ``Internal server error``                    -> :func:`test_internal_server_error`
* ``Returns correct evidence for rule``        -> :func:`test_returns_correct_evidence_for_rule`
* ``Returns correct evidence for datasource``  -> :func:`test_returns_correct_evidence_for_datasource`

The ``Fact.Source`` expectation of the two 200 table entries (the envelope source copied onto
the fact) is additionally isolated in :func:`test_fact_source_is_propagated_to_fact`,
parametrised over both bodies; it was a strict xfail until ``Evidence.from_dict`` started
copying the source (1.1.0), so nothing Go asserts is dropped or silently weakened.

The remaining tests are NOT table entries and have no 1:1 counterpart in ``TestEvidence``; do
not expect to find them in the Go table. They are adapted from the implementation under test
because the unit guidance asks for them and the Go table does not cover them:

* the 404 -> ``ErrFactNotFound`` mapping (``client.go:252-254``):
  :func:`test_not_found_with_body_maps_to_not_found_error` and
  :func:`test_not_found_with_empty_body_maps_to_not_found_error`;
* the ``x-evidence-key`` header (``client.go:236-238``; the table only passes ``nil``):
  :func:`test_evidence_key_header_is_sent`, :func:`test_evidence_key_header_is_sent_by_session`,
  :func:`test_evidence_key_header_is_absent_when_not_given` and the Python-only
  :func:`test_client_wide_evidence_key_is_default_and_per_call_wins`;
* ``asConditions`` classification (``evidence.go:154-177``):
  :func:`test_unsupported_condition_type_is_kept_not_rejected`.

The Python SDK is ``Rainbird.evidence(session_id, fact_id, evidence_key=...)`` /
``Session.evidence(fact_id, evidence_key=...)`` returning :class:`rainbird_ai.models.Evidence`.

Differences found (all asserted below with ``# DIFFERENCE(go)`` comments):

* Go's ``Evidence`` sends no auth header at all and a ``Content-Type: application/json`` on the
  GET; Python sends ``X-API-Key`` on every request, ``Accept: application/json`` and no
  ``Content-Type`` (the request has no body). Neither sends ``Authorization``.
* Go asserts the raw request URI ``/analysis/evidence/WA:RF:1234/1234``; Python percent-encodes
  the path segments so the wire path is ``/analysis/evidence/WA%3ARF%3A1234/1234`` (the decoded
  path is identical).
* Go returns ``error`` values (``"API returned error 400: Foo bar baz"``, body-independent
  ``ErrFactNotFound`` for 404); Python raises ``BadRequestError``/``ServerError``/``NotFoundError``
  carrying ``status_code``, ``message``, ``body``, ``errors``, ``method`` and ``url``.
* Go's ``asEvidence`` copies the envelope ``source`` onto ``Fact.Source`` exactly as it copies
  ``factID`` onto ``Fact.ID``; Python's ``Evidence.from_dict`` does the same since 1.1.0 (it
  used to leave ``evidence.fact.source`` ``None``), and a ``source`` inside ``fact`` wins.
* Go's ``asConditions`` fails the whole call with ``"unsupported condition type"`` for a condition
  with neither ``relationship`` nor ``expression.text``; Python keeps it as a
  :class:`~rainbird_ai.models.Condition` with an empty relationship and the raw JSON intact.
  Go classifies ``relationship`` first; Python classifies on the presence of ``expression``.
* Go exposes ``Conditioner.Type()``/``Salience()`` and a ``*Rule`` that is ``nil`` for
  non-inferred facts; Python uses ``isinstance(cond, Condition | ExpressionCondition)``,
  ``cond.salience`` and ``evidence.rule is None`` / ``evidence.is_inferred``.
* Go decodes JSON numbers in ``bindings`` as ``float64`` (``100.0``); Python keeps ``int`` 100.
* Go passes the evidence key per call only (``*string``, ``nil`` = no header); Python also has a
  client-wide ``Rainbird(evidence_key=...)``/``RAINBIRD_EVIDENCE_KEY`` default. A per-call
  ``None`` falls back to that default, a non-empty value replaces it and an explicit ``""``
  suppresses the header even when a default is configured; ``""`` is therefore equivalent to
  ``None`` only when no client-wide key is set.
* ``str(evidence.fact)`` is ``"Dan might speak English (100%)"`` where Go's ``Fact.String()``
  would give ``"WA:RF:1234 (rule): Dan, might speak, English (100)"``.
"""

from __future__ import annotations

from typing import Any

import pytest

from rainbird_ai.errors import (
    BadRequestError,
    NotFoundError,
    RainbirdAPIError,
    ServerError,
)
from rainbird_ai.models import Condition, Evidence, ExpressionCondition, FactSource, Rule
from tests.parity.support import Call, Recorder, make_client, respond

# Constants from TestEvidence (client_test.go:328-329) and the Go client literal (:534-538).
SESSION_ID = "1234"
FACT_ID = "WA:RF:1234"
GO_API_KEY = "1234567890-1234-1234-1234-1234567890ab"
EXPECTED_PATH = f"/analysis/evidence/{FACT_ID}/{SESSION_ID}"
# Go's server sets Content-Type: application/json on every response, error bodies included.
JSON_CONTENT_TYPE = {"Content-Type": "application/json"}

# client_test.go:356-411, verbatim.
RULE_BODY: dict[str, Any] = {
    "factID": "WA:RF:1234",
    "source": "rule",
    "fact": {
        "subject": {"type": "Person", "value": "Dan", "dataType": "string"},
        "relationship": {"type": "might speak"},
        "object": {"type": "Language", "value": "English", "dataType": "string"},
        "certainty": 100,
    },
    "time": 123456789,
    "rule": {
        "bindings": {"S": "Dan", "O": 100, "COUNTRY": "England"},
        "conditions": [
            {
                "subject": "Dan",
                "relationship": "lives in",
                "object": "England",
                "salience": 100,
                "certainty": 100,
                "factID": "WA:RF:1234",
                "objectType": "string",
                "factKey": "30d74cc5-7eb6-4c88-8e12-4897b47e3ee7",
            },
            {
                "subject": "England",
                "relationship": "has national language",
                "object": "English",
                "salience": 100,
                "certainty": 100,
                "factID": "WA:RF:1234",
                "objectType": "string",
            },
            {"wasMet": True, "salience": 100, "expression": {"text": "1 gte 0"}},
        ],
    },
}

# client_test.go:472-492, verbatim.
DATASOURCE_BODY: dict[str, Any] = {
    "factID": "WA:DF:1234",
    "source": "datasource",
    "fact": {
        "subject": {"type": "location", "value": "London", "dataType": "string"},
        "relationship": {"type": "has temperature"},
        "object": {"type": "temperature", "value": "290.00", "dataType": "string"},
        "certainty": 100,
    },
    "time": 123456789,
}


def _assert_go_call(call: Call, *, evidence_key: str | None = None) -> Call:
    """The assertions Go's httptest handler makes on a request (client_test.go:524-525).

    ``evidence_key`` is the ``x-evidence-key`` value the request must carry; ``None`` (Go's
    ``nil``, the only value the table passes) means the header must be absent.
    """
    assert call.method == "GET"
    assert call.path == EXPECTED_PATH
    # DIFFERENCE(go): Go asserts r.RequestURI == "/analysis/evidence/WA:RF:1234/1234" (colons
    # sent verbatim). Python percent-encodes every path segment, so the wire path carries %3A;
    # the decoded path asserted above is identical.
    assert call.url.raw_path == b"/analysis/evidence/WA%3ARF%3A1234/1234"
    assert call.query == {}
    assert call.content == b""
    if evidence_key is None:
        assert "x-evidence-key" not in call.headers
    else:
        assert call.headers["x-evidence-key"] == evidence_key
    # DIFFERENCE(go): Go's Evidence sets Content-Type: application/json and NO auth header at
    # all (client.go:235-238; SetBasicAuth is only used by NewSession and session()). Python
    # sends X-API-Key on every request plus Accept, and no Content-Type on a body-less GET.
    assert call.headers["x-api-key"] == GO_API_KEY
    assert "authorization" not in call.headers
    assert call.headers["accept"] == "application/json"
    assert "content-type" not in call.headers
    return call


def _assert_go_request(recorder: Recorder, *, evidence_key: str | None = None) -> Call:
    """:func:`_assert_go_call` on the single request Go's ``Evidence`` makes."""
    assert len(recorder) == 1, "Go's Evidence makes exactly one request"
    return _assert_go_call(recorder.last, evidence_key=evidence_key)


def _evidence_client(status: int, body: object = None, *, text: str | None = None):
    handler = respond(status, body, text=text, headers=JSON_CONTENT_TYPE)
    return make_client(handler, api_key=GO_API_KEY)


# source: client_test.go:340 Bad request
def test_bad_request() -> None:
    rb, recorder = _evidence_client(400, text="Foo bar baz")

    # DIFFERENCE(go): Go returns (nil, errors.New("API returned error 400: Foo bar baz")); Python
    # raises BadRequestError, so there is no evidence value to compare with nil.
    with pytest.raises(BadRequestError) as excinfo:
        rb.evidence(SESSION_ID, FACT_ID)

    call = _assert_go_request(recorder)
    err = excinfo.value
    assert isinstance(err, RainbirdAPIError)
    assert err.status_code == 400
    assert err.message == "Foo bar baz"
    assert err.errors == ["Foo bar baz"]
    assert err.body == "Foo bar baz"  # text kept even though the server claimed JSON
    assert err.method == "GET"
    assert err.url == str(call.url)
    assert str(err) == f"HTTP 400 calling GET {call.url}: Foo bar baz"


# source: client_test.go:347 Internal server error
def test_internal_server_error() -> None:
    rb, recorder = _evidence_client(500, text="Foo bar baz")

    # DIFFERENCE(go): Go returns errors.New("API returned error 500: Foo bar baz"); Python raises
    # ServerError (5xx). Retries are disabled in make_client, as Go has none.
    with pytest.raises(ServerError) as excinfo:
        rb.evidence(SESSION_ID, FACT_ID)

    call = _assert_go_request(recorder)
    err = excinfo.value
    assert isinstance(err, RainbirdAPIError)
    assert err.status_code == 500
    assert err.message == "Foo bar baz"
    assert err.errors == ["Foo bar baz"]
    assert err.body == "Foo bar baz"
    assert err.method == "GET"
    assert err.url == str(call.url)
    assert str(err) == f"HTTP 500 calling GET {call.url}: Foo bar baz"


# source: client_test.go:354 Returns correct evidence for rule
def test_returns_correct_evidence_for_rule() -> None:
    rb, recorder = _evidence_client(200, RULE_BODY)

    evidence = rb.evidence(SESSION_ID, FACT_ID)

    _assert_go_request(recorder)
    assert isinstance(evidence, Evidence)

    # Go: Fact{ID, Source, Subject, Relationship, Object, Certainty}
    assert evidence.fact_id == FACT_ID
    assert evidence.fact.id == FACT_ID
    assert evidence.source == "rule"
    assert evidence.source is FactSource.RULE
    # Go's asEvidence copies the envelope's source onto Fact.Source ("rule") exactly as it
    # copies factID onto Fact.ID; Python's Evidence.from_dict does the same (and also keeps it
    # on Evidence.source). See test_fact_source_is_propagated_to_fact below.
    assert evidence.fact.source is FactSource.RULE
    # Go: ConceptInstance{Type: "Person", Value: "Dan", DataType: "string"}; Python names the
    # concept `concept` (read from "type") and the data type `data_type`.
    assert evidence.fact.subject.concept == "Person"
    assert evidence.fact.subject.value == "Dan"
    assert evidence.fact.subject.data_type == "string"
    assert evidence.fact.relationship == "might speak"
    assert evidence.fact.object.concept == "Language"
    assert evidence.fact.object.value == "English"
    assert evidence.fact.object.data_type == "string"
    assert evidence.fact.certainty == 100
    # Go: Time: 123456789
    assert evidence.time == 123456789

    # Go: Rule{Bindings: {"S": "Dan", "O": 100.0, "COUNTRY": "England"}}
    rule = evidence.rule
    assert isinstance(rule, Rule)
    assert evidence.is_inferred is True
    # DIFFERENCE(go): Go decodes JSON numbers into float64, so it expects "O": 100.0; Python
    # keeps the int 100 (100 == 100.0 holds either way).
    assert rule.bindings == {"S": "Dan", "O": 100, "COUNTRY": "England"}
    assert isinstance(rule.bindings["O"], int)
    assert rule.max_certainty is None  # no ruleMaxCertainty in the body

    # Go: Conditions: []Conditioner{ConditionRelationship, ConditionRelationship,
    # ConditionExpression}. Python: Condition / Condition / ExpressionCondition.
    assert len(rule.conditions) == 3
    first, second, third = rule.conditions

    # DIFFERENCE(go): Go exposes Type() == RelationshipType and Salience(); Python uses
    # isinstance(Condition) and the `salience` field.
    assert isinstance(first, Condition)
    assert first.subject == "Dan"
    assert first.relationship == "lives in"
    assert first.object == "England"
    assert first.certainty == 100
    assert first.salience == 100
    assert first.fact_id == FACT_ID
    assert first.object_type == "string"
    assert first.fact_key == "30d74cc5-7eb6-4c88-8e12-4897b47e3ee7"  # Go: FactKey: stringPtr(...)
    assert first.impact is None and first.alt is None  # Python-only fields, absent here

    assert isinstance(second, Condition)
    assert second.subject == "England"
    assert second.relationship == "has national language"
    assert second.object == "English"
    assert second.certainty == 100
    assert second.salience == 100
    assert second.fact_id == FACT_ID
    assert second.object_type == "string"
    assert second.fact_key is None  # Go: FactKey: nil

    # DIFFERENCE(go): Go's ConditionExpression{WasMet, Expression{Text}, salience}; Python
    # flattens expression.text to `text` and adds value/functions (empty here).
    assert isinstance(third, ExpressionCondition)
    assert third.was_met is True
    assert third.text == "1 gte 0"
    assert third.salience == 100
    assert third.value is None
    assert third.functions == {}
    assert third.fact_ids == []

    # Python-specific views over the same data.
    assert rule.relationship_conditions == [first, second]
    assert rule.expression_conditions == [third]
    # Every condition's factID equals the evidence's own factID, and child_fact_ids drops the
    # self-reference, so this root has no children to walk.
    assert evidence.child_fact_ids == []
    # Never lose API data: the original JSON is kept on .raw / to_dict().
    assert evidence.raw == RULE_BODY
    assert evidence.to_dict() == RULE_BODY
    assert evidence.to_dict() is not RULE_BODY
    assert rule.raw == RULE_BODY["rule"]
    assert first.raw == RULE_BODY["rule"]["conditions"][0]
    assert third.raw == RULE_BODY["rule"]["conditions"][2]
    # DIFFERENCE(go): Go's Evidence.String()/Fact.String() would render
    # "WA:RF:1234 (rule): Dan, might speak, English (100)"; Python's SessionFact.__str__ is
    # "<subject> <relationship> <object> (<certainty>%)".
    assert str(evidence.fact) == "Dan might speak English (100%)"


# source: client_test.go:354 Returns correct evidence for rule / client_test.go:470 Returns
# correct evidence for datasource (the Fact.Source expectation of both 200 entries only).
# Was a strict xfail until Evidence.from_dict copied the envelope source onto the fact (1.1.0).
@pytest.mark.parametrize(
    ("body", "expected_source"),
    [(RULE_BODY, FactSource.RULE), (DATASOURCE_BODY, FactSource.DATASOURCE)],
    ids=["rule", "datasource"],
)
def test_fact_source_is_propagated_to_fact(
    body: dict[str, Any], expected_source: FactSource
) -> None:
    rb, recorder = _evidence_client(200, body)

    evidence = rb.evidence(SESSION_ID, FACT_ID)

    _assert_go_request(recorder)
    assert evidence.source is expected_source  # the envelope source is parsed ...
    assert evidence.fact.id == body["factID"]  # ... and the id is propagated onto the fact ...
    assert evidence.fact.source is expected_source  # ... and so is the source


# source: client_test.go:470 Returns correct evidence for datasource
def test_returns_correct_evidence_for_datasource() -> None:
    rb, recorder = _evidence_client(200, DATASOURCE_BODY)

    # Go still requests factID "WA:RF:1234" (the table shares one factID) while the body
    # describes "WA:DF:1234"; like Go, Python takes the fact id from the body.
    evidence = rb.evidence(SESSION_ID, FACT_ID)

    _assert_go_request(recorder)
    assert isinstance(evidence, Evidence)

    # Go: Fact{ID: "WA:DF:1234", Source: "datasource", ...}
    assert evidence.fact_id == "WA:DF:1234"
    assert evidence.fact.id == "WA:DF:1234"
    assert evidence.source == "datasource"
    assert evidence.source is FactSource.DATASOURCE
    # Go's asEvidence copies the envelope's source onto Fact.Source ("datasource"); so does
    # Python (see test_fact_source_is_propagated_to_fact, whose "datasource" parameter is this
    # body).
    assert evidence.fact.source is FactSource.DATASOURCE
    assert evidence.fact.subject.concept == "location"
    assert evidence.fact.subject.value == "London"
    assert evidence.fact.subject.data_type == "string"
    assert evidence.fact.relationship == "has temperature"
    assert evidence.fact.object.concept == "temperature"
    assert evidence.fact.object.value == "290.00"  # stays a string, as in Go's interface{}
    assert isinstance(evidence.fact.object.value, str)
    assert evidence.fact.object.data_type == "string"
    assert evidence.fact.certainty == 100
    # Go: Time: 123456789
    assert evidence.time == 123456789

    # Go: Rule: nil
    # DIFFERENCE(go): Go's *Rule is nil; Python's rule is None and is_inferred is False.
    assert evidence.rule is None
    assert evidence.is_inferred is False
    assert evidence.child_fact_ids == []
    assert evidence.raw == DATASOURCE_BODY
    assert evidence.to_dict() == DATASOURCE_BODY
    assert str(evidence.fact) == "London has temperature 290.00 (100%)"


# source: client.go:252-254 (Evidence: 404 -> ErrFactNotFound; not in the TestEvidence table)
def test_not_found_with_body_maps_to_not_found_error() -> None:
    rb, recorder = _evidence_client(404, text="Foo bar baz")

    # DIFFERENCE(go): Go discards the body and returns the sentinel ErrFactNotFound ("fact not
    # found") for any 404; Python raises NotFoundError (a RainbirdAPIError) and keeps the body.
    with pytest.raises(NotFoundError) as excinfo:
        rb.evidence(SESSION_ID, FACT_ID)

    call = _assert_go_request(recorder)
    err = excinfo.value
    assert isinstance(err, RainbirdAPIError)
    assert err.status_code == 404
    assert err.message == "Foo bar baz"
    assert err.errors == ["Foo bar baz"]
    assert err.body == "Foo bar baz"
    assert err.method == "GET"
    assert err.url == str(call.url)
    assert str(err) == f"HTTP 404 calling GET {call.url}: Foo bar baz"


# source: client.go:252-254 (Evidence: 404 with an empty body -> ErrFactNotFound; not in the
# TestEvidence table)
def test_not_found_with_empty_body_maps_to_not_found_error() -> None:
    rb, recorder = _evidence_client(404)

    # DIFFERENCE(go): Go's ErrFactNotFound ("fact not found") is body-independent; with an empty
    # body Python's NotFoundError falls back to a fixed message, the closest thing to Go's
    # sentinel, and reports no body and no error list.
    with pytest.raises(NotFoundError) as excinfo:
        rb.evidence(SESSION_ID, FACT_ID)

    call = _assert_go_request(recorder)
    err = excinfo.value
    assert isinstance(err, RainbirdAPIError)
    assert err.status_code == 404
    assert err.message == "The resource (session, knowledge map or fact) could not be found"
    assert err.errors == []
    assert err.body is None
    assert err.method == "GET"
    assert err.url == str(call.url)
    assert str(err) == (
        f"HTTP 404 calling GET {call.url}: "
        "The resource (session, knowledge map or fact) could not be found"
    )


# source: client.go:236-238 (Evidence: evidenceKey != nil -> x-evidence-key; the table only
# exercises nil, so this is adapted from the implementation)
def test_evidence_key_header_is_sent() -> None:
    rb, recorder = _evidence_client(200, RULE_BODY)

    # Go: client.Evidence(sessionID, factID, stringPtr("secret")) sets x-evidence-key: secret.
    evidence = rb.evidence(SESSION_ID, FACT_ID, evidence_key="secret")

    assert isinstance(evidence, Evidence)
    call = _assert_go_request(recorder, evidence_key="secret")
    assert call.headers["x-evidence-key"] == "secret"


# source: client.go:236-238 (same header through Session.evidence, Python's second entry point)
def test_evidence_key_header_is_sent_by_session() -> None:
    rb, recorder = _evidence_client(200, RULE_BODY)

    # Session.evidence(fact_id, evidence_key=...) hits the same URL with the session id taken
    # from the Session object; resume() itself sends nothing.
    session = rb.resume(SESSION_ID)
    assert session.id == SESSION_ID
    assert len(recorder) == 0
    evidence = session.evidence(FACT_ID, evidence_key="secret")

    assert isinstance(evidence, Evidence)
    # Same request as the client-level twin above: one GET, the Go path, x-evidence-key,
    # X-API-Key, Accept, and neither Authorization nor Content-Type.
    call = _assert_go_request(recorder, evidence_key="secret")
    assert call.headers["x-evidence-key"] == "secret"


# source: client.go:236-238 (Evidence: evidenceKey == nil -> no x-evidence-key header)
@pytest.mark.parametrize("evidence_key", [None, ""], ids=["none", "empty"])
def test_evidence_key_header_is_absent_when_not_given(evidence_key: str | None) -> None:
    rb, recorder = _evidence_client(200, RULE_BODY)

    # Go: nil -> header absent. Python: with no client-wide key configured (as here) both None
    # and "" leave the header out. They are NOT interchangeable in general: None falls back to
    # Rainbird(evidence_key=...) while "" suppresses the header even when that default is set
    # (client.py: `evidence_key if evidence_key is not None else self.config.evidence_key`, then
    # _ops.evidence only adds the header for a truthy key); see the client-wide test below.
    evidence = rb.evidence(SESSION_ID, FACT_ID, evidence_key=evidence_key)

    assert isinstance(evidence, Evidence)
    call = _assert_go_request(recorder)  # asserts "x-evidence-key" not in call.headers
    assert "x-evidence-key" not in call.headers


# source: client.go:236-238 (Python-only: client-wide evidence key default; Go has none)
def test_client_wide_evidence_key_is_default_and_per_call_wins() -> None:
    # DIFFERENCE(go): Go has no client-wide evidence key (the *string is per call only);
    # Python's Rainbird(evidence_key=...) (or RAINBIRD_EVIDENCE_KEY) is the default for every
    # evidence call: a per-call None (the default argument) falls back to it, a non-empty value
    # wins, and an explicit "" is the way to send NO key despite the configured default.
    rb, recorder = make_client(
        respond(200, RULE_BODY, headers=JSON_CONTENT_TYPE),
        api_key=GO_API_KEY,
        evidence_key="default-key",
    )

    rb.evidence(SESSION_ID, FACT_ID)
    _assert_go_call(recorder.last, evidence_key="default-key")

    rb.evidence(SESSION_ID, FACT_ID, evidence_key=None)
    _assert_go_call(recorder.last, evidence_key="default-key")

    rb.evidence(SESSION_ID, FACT_ID, evidence_key="per-call")
    _assert_go_call(recorder.last, evidence_key="per-call")

    rb.evidence(SESSION_ID, FACT_ID, evidence_key="")
    _assert_go_call(recorder.last, evidence_key=None)  # header suppressed, unlike None above
    assert "x-evidence-key" not in recorder.last.headers

    assert len(recorder) == 4


# source: evidence.go:154-177 (asConditions: "unsupported condition type" and the
# relationship-before-expression classification; not in the TestEvidence table)
def test_unsupported_condition_type_is_kept_not_rejected() -> None:
    body = {
        **RULE_BODY,
        "rule": {
            "bindings": {"S": "Dan"},
            "conditions": [
                # neither "relationship" nor "expression.text": Go -> error
                {"salience": 50, "certainty": 100, "factID": "WA:RF:9999"},
                # both: Go -> ConditionRelationship (relationship checked first)
                {
                    "subject": "Dan",
                    "relationship": "lives in",
                    "object": "England",
                    "salience": 100,
                    "expression": {"text": "1 gte 0"},
                },
            ],
        },
    }
    rb, recorder = _evidence_client(200, body)

    # DIFFERENCE(go): Go's asConditions returns errors.New("unsupported condition type") and
    # Evidence() fails as a whole. Python never drops API data: the unknown condition becomes
    # a Condition with an empty relationship and its raw JSON intact, and the call succeeds.
    evidence = rb.evidence(SESSION_ID, FACT_ID)

    _assert_go_request(recorder)
    assert evidence.rule is not None
    unknown, both = evidence.rule.conditions
    assert isinstance(unknown, Condition)
    assert unknown.relationship == ""
    assert unknown.subject is None and unknown.object is None
    assert unknown.salience == 50
    assert unknown.certainty == 100
    assert unknown.fact_id == "WA:RF:9999"
    assert unknown.raw == body["rule"]["conditions"][0]
    assert evidence.child_fact_ids == ["WA:RF:9999"]

    # DIFFERENCE(go): Go classifies on `relationship != ""` first, so this would be a
    # ConditionRelationship; Python classifies on the presence of the "expression" key, so it
    # is an ExpressionCondition (the relationship fields remain available on .raw).
    assert isinstance(both, ExpressionCondition)
    assert both.text == "1 gte 0"
    assert both.salience == 100
    assert both.raw["relationship"] == "lives in"
    assert evidence.rule.relationship_conditions == [unknown]
    assert evidence.rule.expression_conditions == [both]
