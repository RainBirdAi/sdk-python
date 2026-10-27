"""Operation builders: one function per API endpoint.

Each builder returns a :class:`~rainbird_ai._http.Request` describing the HTTP
call and how to parse its response. The sync and async clients only differ in
how they *send* these requests, so every endpoint's URL, headers, body and
parsing live here exactly once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import httpx

from ._answers import coerce_value
from ._http import (
    NL_VERSION,
    Request,
    clean_key,
    extract_error_messages,
    json_body,
    path_segment,
    text_body,
)
from .errors import (
    InvalidFactError,
    KnowledgeMapError,
    NaturalLanguageError,
    RainbirdConfigurationError,
    UnexpectedResponseError,
)
from .models import (
    CreateMapResult,
    Evidence,
    ExplainResponse,
    InjectResponse,
    InteractionEvent,
    InteractResponse,
    KnowledgeMapSource,
    KnowledgeMapVersion,
    QueryResponse,
    ReadmeAdded,
    SessionFacts,
    StartResponse,
    TestRun,
    TestRunStarted,
    TestSuite,
    TestSuiteCreated,
    ValidationResult,
    parse_query_response,
)

# /nl/*, /maps and the map-management endpoints (/maps/{kmID}/..., /runs, /validate).
_V1_HEADERS = {"Version": NL_VERSION}


def _decision(response: httpx.Response) -> QueryResponse:
    body = json_body(response)
    try:
        return parse_query_response(body)
    except UnexpectedResponseError as exc:
        # Re-raise with the HTTP status and any error text the body carries (e.g. {"err": [...]}).
        errors = extract_error_messages(body)
        message = errors[0] if errors else exc.message
        raise UnexpectedResponseError(message, body=body, status_code=response.status_code) from exc


def _object(response: httpx.Response) -> dict[str, Any]:
    data = json_body(response)
    if not isinstance(data, dict):
        raise UnexpectedResponseError(
            f"expected a JSON object, got {type(data).__name__}",
            body=data,
            status_code=response.status_code,
        )
    return data


def _require(value: Any, name: str) -> str:
    if value is None or not str(value).strip():
        raise RainbirdConfigurationError(f"{name} must not be empty")
    return str(value)


# --------------------------------------------------------------------------- decisions


def start(
    km_id: str,
    *,
    use_draft: bool | None = None,
    version: int | None = None,
    context_id: str | None = None,
) -> Request[StartResponse]:
    if use_draft is not None and version is not None:
        raise RainbirdConfigurationError("use_draft and version cannot be used together")
    if version is not None and (
        not isinstance(version, int) or isinstance(version, bool) or version < 1
    ):
        raise RainbirdConfigurationError(f"version must be a positive integer, got {version!r}")

    def parse(response: httpx.Response) -> StartResponse:
        data = _object(response)
        try:
            return StartResponse.from_dict(data)
        except ValueError as exc:
            raise UnexpectedResponseError(
                str(exc), body=data, status_code=response.status_code
            ) from exc

    return Request(
        method="GET",
        path=f"start/{path_segment(km_id, 'km_id')}",
        params={"useDraft": use_draft, "version": version, "contextid": context_id},
        parse=parse,
    )


def inject(session_id: str, facts: list[dict[str, Any]]) -> Request[InjectResponse]:
    return Request(
        method="POST",
        path=f"{path_segment(session_id, 'session_id')}/inject",
        json=facts,
        parse=lambda r: InjectResponse.from_dict(_object(r)),
    )


def query_body(relationship: str, *, subject: Any = None, object: Any = None) -> dict[str, Any]:
    """The ``POST /query`` body, with subject and object coerced like fact values.

    A ``date`` is sent as ``"YYYY-MM-DD"``; an unsupported value (``NaN``, a dict, ...)
    raises :class:`~rainbird_ai.errors.InvalidAnswerError` before any request is made.
    """
    body: dict[str, Any] = {"relationship": _require(relationship, "relationship")}
    if subject is not None and subject != "":
        body["subject"] = coerce_value(subject)
    if object is not None and object != "":
        body["object"] = coerce_value(object)
    return body


def query(
    session_id: str,
    relationship: str,
    *,
    subject: Any = None,
    object: Any = None,
) -> Request[QueryResponse]:
    return Request(
        method="POST",
        path=f"{path_segment(session_id, 'session_id')}/query",
        json=query_body(relationship, subject=subject, object=object),
        parse=_decision,
    )


def respond(session_id: str, answers: list[dict[str, Any]]) -> Request[QueryResponse]:
    return Request(
        method="POST",
        path=f"{path_segment(session_id, 'session_id')}/response",
        json={"answers": answers},
        parse=_decision,
    )


def undo(session_id: str) -> Request[QueryResponse]:
    return Request(
        method="POST",
        path=f"{path_segment(session_id, 'session_id')}/undo",
        json={},
        parse=_decision,
    )


# --------------------------------------------------------------------------- evidence & analysis


def evidence(fact_id: str, session_id: str, evidence_key: str | None = None) -> Request[Evidence]:
    key = clean_key(evidence_key)
    headers = {"x-evidence-key": key} if key else {}
    return Request(
        method="GET",
        path=(
            f"analysis/evidence/{path_segment(fact_id, 'fact_id')}/"
            f"{path_segment(session_id, 'session_id')}"
        ),
        headers=headers,
        parse=lambda r: Evidence.from_dict(_object(r)),
    )


def interactions_json(
    session_id: str, interaction_key: str | None = None
) -> Request[list[InteractionEvent]]:
    key = clean_key(interaction_key)
    headers = {"x-interaction-key": key} if key else {}

    def parse(response: httpx.Response) -> list[InteractionEvent]:
        data = json_body(response)
        if isinstance(data, dict):
            if "event" in data:  # a single event object
                data = [data]
            else:
                for key in ("values", "events", "interactions"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    hints = extract_error_messages(data)
                    raise UnexpectedResponseError(
                        "; ".join(hints) if hints else "expected a list of interaction events",
                        body=data,
                        status_code=response.status_code,
                    )
        if not isinstance(data, list):
            raise UnexpectedResponseError(
                "expected a list of interaction events", body=data, status_code=response.status_code
            )
        return [InteractionEvent.from_dict(item) for item in data if isinstance(item, dict)]

    return Request(
        method="GET",
        path=f"analysis/interactions/{path_segment(session_id, 'session_id')}",
        params={"format": "json"},
        headers=headers,
        parse=parse,
    )


def interactions_csv(session_id: str, interaction_key: str | None = None) -> Request[str]:
    key = clean_key(interaction_key)
    headers = {"x-interaction-key": key} if key else {}
    return Request(
        method="GET",
        path=f"analysis/interactions/{path_segment(session_id, 'session_id')}",
        params={"format": "csv"},
        headers=headers,
        accept="text/csv, text/plain, */*",
        parse=lambda r: r.text,
    )


def session_version(session_id: str) -> Request[KnowledgeMapVersion]:
    return Request(
        method="GET",
        path=f"analysis/session/{path_segment(session_id, 'session_id')}",
        params={"filter": "version"},
        parse=lambda r: KnowledgeMapVersion.from_dict(_object(r)),
    )


def session_facts(
    session_id: str, relationships: str | Iterable[str] | None = None
) -> Request[SessionFacts]:
    if isinstance(relationships, str):
        rels: list[str] | None = [relationships]
    elif relationships is None:
        rels = None
    else:
        rels = [str(r) for r in relationships]
    return Request(
        method="GET",
        path=f"analysis/session/{path_segment(session_id, 'session_id')}",
        params={"filter": "facts", "relationships": rels or None},
        parse=lambda r: SessionFacts.from_dict(_object(r)),
    )


# ------------------------------------------------------------------ natural language (beta)


def interact(session_id: str, user_prompt: str) -> Request[InteractResponse]:
    body = {
        "sessionID": _require(session_id, "session_id"),
        "userPrompt": _require(user_prompt, "user_prompt"),
    }
    return Request(
        method="POST",
        path="nl/interact",
        json=body,
        headers=_V1_HEADERS,
        parse=lambda r: InteractResponse.from_dict(_object(r)),
    )


def explain(
    session_id: str,
    fact_id: str,
    *,
    language: str = "en",
    instructions: str | None = None,
) -> Request[ExplainResponse]:
    body: dict[str, Any] = {
        "language": language or "en",
        "sessionID": _require(session_id, "session_id"),
        "factID": _require(fact_id, "fact_id"),
    }
    if instructions:
        body["instructions"] = instructions

    def parse(response: httpx.Response) -> ExplainResponse:
        result = ExplainResponse.from_dict(_object(response))
        if result.explanation is None and result.error is not None:
            raise NaturalLanguageError(
                result.error.message or "explain returned an error",
                code=result.error.code,
                suggested_chat_response=result.error.suggested_chat_response,
                status_code=result.error.status_code,
                body=result.raw,
            )
        return result

    return Request(method="POST", path="nl/explain", json=body, headers=_V1_HEADERS, parse=parse)


# --------------------------------------------------------------------------- platform


def version() -> Request[str]:
    def parse(response: httpx.Response) -> str:
        text = text_body(response)
        candidate: str | None = text
        if text and text[0] in "{[":
            # A JSON body must carry the version under a known key; never echo raw JSON back.
            candidate = None
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                for key in ("version", "engine", "semver"):
                    if key in data:
                        found = data[key]
                        if isinstance(found, str) and found.strip():
                            candidate = found
                        elif isinstance(found, (int, float)) and not isinstance(found, bool):
                            candidate = str(found)
                        break
        if candidate is not None:
            candidate = candidate.strip().strip('"')
        looks_like_version = (
            bool(candidate)
            and len(candidate or "") <= 64
            and "<" not in (candidate or "")
            and not any(ch.isspace() for ch in candidate or "")
        )
        if not looks_like_version or candidate is None:
            raise UnexpectedResponseError(
                f"expected an engine version from GET /version but got: {text[:200]!r}",
                body=text,
                status_code=response.status_code,
            )
        return candidate

    return Request(
        method="GET", path="version", accept="text/plain, application/json, */*", parse=parse
    )


# ------------------------------------------------------------------ knowledge maps (undocumented)


def map_source(km_id: str) -> Request[KnowledgeMapSource]:
    """``GET /analysis/file/{kmID}``: RBLang source of a map. Not in the public OpenAPI spec."""
    segment = path_segment(km_id, "km_id")

    def parse(response: httpx.Response) -> KnowledgeMapSource:
        data = _object(response)
        if "rblang" not in data:
            errors = extract_error_messages(data)
            raise UnexpectedResponseError(
                "; ".join(errors) if errors else "response did not include 'rblang'",
                body=data,
                status_code=response.status_code,
            )
        return KnowledgeMapSource.from_dict(data, km_id=str(km_id))

    return Request(method="GET", path=f"analysis/file/{segment}", parse=parse)


def create_map(name: str, rblang: str, description: str | None = None) -> Request[CreateMapResult]:
    """``POST /maps``: create a knowledge map from RBLang. Not in the public OpenAPI spec.

    The body is ``{"rblang": ..., "name": ..., "description": ...}``; ``description`` is sent
    only when given.

    Rainbird answers HTTP 200 even when the RBLang fails to parse, signalling the
    failure with an ``error`` key; the parser turns that into
    :class:`~rainbird_ai.errors.KnowledgeMapError`. Informational keys such as
    ``message`` on a successful body are not treated as errors.
    """
    clean_name = _require(name, "name")
    clean_rblang = _require(rblang, "rblang")

    def parse(response: httpx.Response) -> CreateMapResult:
        data = _object(response)
        # Only truthy error payloads count: {"error": false}, 0, "", [], {} and null are success.
        error_fields = {key: data[key] for key in ("err", "errors", "error") if data.get(key)}
        errors = extract_error_messages(error_fields) if error_fields else []
        if errors:
            raise KnowledgeMapError(
                errors[0] if len(errors) == 1 else "; ".join(errors), errors=errors, body=data
            )
        try:
            return CreateMapResult.from_dict(data, name=clean_name, description=description)
        except ValueError as exc:
            hints = extract_error_messages(data)
            raise UnexpectedResponseError(
                hints[0] if hints else str(exc), body=data, status_code=response.status_code
            ) from exc

    body: dict[str, Any] = {"rblang": clean_rblang, "name": clean_name}
    if description is not None:
        body["description"] = description
    return Request(method="POST", path="maps", json=body, headers=_V1_HEADERS, parse=parse)


# ------------------------------------- map tests, readme and validation (not in the OpenAPI spec)

# Lists of facts inside a test step, keyed by (step part, field): Inject, Respond, Expect Result.
_SUITE_FACT_LISTS = (("then", "inject"), ("then", "with"), ("expect", "results"))


def _missing(data: dict[str, Any], key: str, response: httpx.Response) -> UnexpectedResponseError:
    errors = extract_error_messages(data)
    return UnexpectedResponseError(
        "; ".join(errors) if errors else f"response did not include {key!r}",
        body=data,
        status_code=response.status_code,
    )


def suite_payload(tests: Any) -> list[Any]:
    """Check a test suite for ``POST /maps/{kmID}/tests`` and return it as a JSON-ready list.

    Accepts a list of test objects, a :class:`~rainbird_ai.TestSuite` or a ``{"tests": [...]}``
    body (what ``get_tests()`` returns), so a suite can be copied between maps. Only the outer
    shape is checked here, leaving the rest to the server, plus one mistake the server cannot
    see: a fact giving ``certainty`` instead of ``cf``, which the test runner ignores (it then
    uses certainty 100) and so is rejected with :class:`~rainbird_ai.InvalidFactError`.
    """
    if isinstance(tests, TestSuite):
        tests = tests.tests
    elif isinstance(tests, Mapping) and "tests" in tests:
        tests = tests["tests"]
    if not isinstance(tests, (list, tuple)) or not tests:
        raise RainbirdConfigurationError("tests must be a non-empty list of test objects")
    suite: list[Any] = []
    for i, test in enumerate(tests):
        if not isinstance(test, Mapping):
            raise RainbirdConfigurationError(
                f"tests[{i}] must be an object with 'name' and 'steps', got {type(test).__name__}"
            )
        steps = test.get("steps")
        for j, step in enumerate(steps if isinstance(steps, list) else []):
            for part, key in _SUITE_FACT_LISTS:
                section = step.get(part) if isinstance(step, Mapping) else None
                facts = section.get(key) if isinstance(section, Mapping) else None
                for k, fact in enumerate(facts if isinstance(facts, list) else []):
                    if isinstance(fact, Mapping) and "certainty" in fact and "cf" not in fact:
                        raise InvalidFactError(
                            f"tests[{i}].steps[{j}].{part}.{key}[{k}] sets 'certainty'; test "
                            "suites write it as 'cf' (the test runner ignores 'certainty' and "
                            "uses 100)"
                        )
        suite.append(dict(test))
    try:
        json.dumps(suite, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RainbirdConfigurationError(f"tests must be JSON-serialisable: {exc}") from exc
    return suite


def create_tests(km_id: str, tests: Any) -> Request[TestSuiteCreated]:
    """``POST /maps/{kmID}/tests``: save a test suite on a map that has none (409 otherwise)."""
    segment = path_segment(km_id, "km_id")
    body = {"tests": suite_payload(tests)}
    return Request(
        method="POST",
        path=f"maps/{segment}/tests",
        json=body,
        headers=_V1_HEADERS,
        parse=lambda r: TestSuiteCreated.from_dict(_object(r), km_id=str(km_id).strip()),
    )


def get_tests(km_id: str) -> Request[TestSuite]:
    """``GET /maps/{kmID}/tests``: the stored suite (404 when the map has none)."""
    segment = path_segment(km_id, "km_id")

    def parse(response: httpx.Response) -> TestSuite:
        data = _object(response)
        if not isinstance(data.get("tests"), list):
            raise _missing(data, "tests", response)
        return TestSuite.from_dict(data, km_id=str(km_id).strip())

    return Request(method="GET", path=f"maps/{segment}/tests", headers=_V1_HEADERS, parse=parse)


def start_test_run(km_id: str) -> Request[TestRunStarted]:
    """``POST /maps/{kmID}/runs`` (no body): start a run of the whole suite on the draft (202)."""
    segment = path_segment(km_id, "km_id")

    def parse(response: httpx.Response) -> TestRunStarted:
        data = _object(response)
        try:
            return TestRunStarted.from_dict(data, km_id=str(km_id).strip())
        except ValueError as exc:
            raise _missing(data, "runID", response) from exc

    return Request(method="POST", path=f"maps/{segment}/runs", headers=_V1_HEADERS, parse=parse)


def get_test_run(run_id: str) -> Request[TestRun]:
    """``GET /runs/{runID}``: status and results (404 for an unknown run or another user's)."""
    segment = path_segment(run_id, "run_id")

    def parse(response: httpx.Response) -> TestRun:
        data = _object(response)
        if data.get("status") is None:
            raise _missing(data, "status", response)
        return TestRun.from_dict(data, run_id=str(run_id).strip())

    return Request(method="GET", path=f"runs/{segment}", headers=_V1_HEADERS, parse=parse)


def add_readme(km_id: str, readme: str) -> Request[ReadmeAdded]:
    """``POST /maps/{kmID}/readme``: set the draft's Markdown readme if it has none (409 otherwise).

    Only blank text is rejected here; the server enforces the rest (at most 100,000
    characters, must not start with ``<``).
    """
    segment = path_segment(km_id, "km_id")
    body = {"readme": _require(readme, "readme")}
    return Request(
        method="POST",
        path=f"maps/{segment}/readme",
        json=body,
        headers=_V1_HEADERS,
        parse=lambda r: ReadmeAdded.from_dict(_object(r), km_id=str(km_id).strip()),
    )


def validate_rblang(rblang: str) -> Request[ValidationResult]:
    """``POST /validate``: compile RBLang without storing it.

    Invalid RBLang is still HTTP 200 with ``"valid": false``, so the parser only requires
    the ``valid`` key and leaves the verdict to the caller.
    """
    body = {"rblang": _require(rblang, "rblang")}

    def parse(response: httpx.Response) -> ValidationResult:
        data = _object(response)
        if data.get("valid") is None:
            raise _missing(data, "valid", response)
        return ValidationResult.from_dict(data)

    return Request(method="POST", path="validate", json=body, headers=_V1_HEADERS, parse=parse)
