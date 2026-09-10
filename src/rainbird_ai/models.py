"""Typed models for Rainbird API payloads.

All response models are frozen dataclasses built from the API's JSON with
``Model.from_dict(data)``. The original JSON is always kept on ``.raw`` so
nothing the API returns is ever lost, even fields this SDK does not model.
``to_dict()`` returns a deep copy of that raw JSON, which makes every model
directly JSON-serialisable.

Input models (:class:`Fact`, :class:`Answer`) build the exact payload the API
expects via ``to_payload()``.

Field names follow Python conventions (``fact_id``) while ``raw`` keeps the
API's names (``factID``).
"""

from __future__ import annotations

import copy
import datetime as _dt
import enum
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import NaturalLanguageError, UnexpectedResponseError

__all__ = [
    "Answer",
    "Concept",
    "Condition",
    "CreateMapResult",
    "Environment",
    "Evidence",
    "ExplainResponse",
    "ExpressionCondition",
    "Fact",
    "FactSource",
    "FactValue",
    "InjectResponse",
    "InteractResponse",
    "InteractionEvent",
    "KnowledgeMapSource",
    "KnowledgeMapVersion",
    "NLError",
    "NLMetadata",
    "QueryResponse",
    "Question",
    "QuestionForm",
    "QuestionResponse",
    "ResultItem",
    "ResultResponse",
    "Rule",
    "SessionFact",
    "SessionFacts",
    "StartResponse",
    "parse_query_response",
    "studio_url_for",
]

JSON = dict[str, Any]


# --------------------------------------------------------------------------- helpers


def _get(data: JSON, *names: str, default: Any = None) -> Any:
    """Return the first present key among ``names`` (handles API casing variants)."""
    for name in names:
        if name in data:
            return data[name]
    lowered = {k.lower(): v for k, v in data.items() if isinstance(k, str)}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def _parse_datetime(value: Any) -> _dt.datetime | None:
    """Parse ISO-8601 strings (with ``Z``) or epoch milliseconds into aware datetimes.

    Anything unparseable (garbage text, booleans, NaN, out-of-range numbers) yields
    ``None`` rather than raising, so one odd field never breaks a whole model.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        seconds = value / 1000 if abs(value) > 1e11 else value
        try:
            return _dt.datetime.fromtimestamp(seconds, tz=_dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            # Python 3.10 only accepts 3 or 6 fractional digits; Rainbird sends up to 9.
            head, dot, tail = text.partition(".")
            digits = len(tail) - len(tail.lstrip("0123456789"))
            if not dot or digits == 0:
                return None
            frac, offset = tail[:digits][:6].ljust(6, "0"), tail[digits:]
            try:
                parsed = _dt.datetime.fromisoformat(f"{head}.{frac}{offset}")
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> JSON:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None:
        return default
    return bool(value)


def _as_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _part(value: Any) -> str:
    """Render one side of a triple for ``__str__``; an unknown side (``None``/``""``) is ``?``."""
    return "?" if value is None or value == "" else str(value)


class _RawMixin:
    raw: JSON

    def to_dict(self) -> JSON:
        """Deep copy of the original API JSON for this object."""
        return copy.deepcopy(self.raw)


# --------------------------------------------------------------------------- enums


class Environment(str, enum.Enum):
    """Well-known Rainbird API hosts.

    Private deployments use ``https://<client>-api.rainbird.ai``; pass that string
    directly as ``base_url``.
    """

    COMMUNITY = "https://api.rainbird.ai"
    ENTERPRISE = "https://enterprise-api.rainbird.ai"
    ENTERPRISE_SOUTH_AMERICA = "https://sa-enterprise-api.rainbird.ai"

    def __str__(self) -> str:  # so f"{Environment.COMMUNITY}" gives the URL
        return self.value

    @property
    def studio_url(self) -> str:
        """The Rainbird Studio host paired with this API host."""
        return studio_url_for(self.value)


def studio_url_for(api_url: str) -> str:
    """Derive the Studio host from an API host (``api.`` -> ``app.``, ``x-api.`` -> ``x.``)."""
    url = api_url.rstrip("/")
    if url == Environment.COMMUNITY.value:
        return "https://app.rainbird.ai"
    scheme, sep, host = url.partition("://")
    if not sep:
        scheme, host = "https", url
    if host.endswith("-api.rainbird.ai"):
        return f"{scheme}://{host[: -len('-api.rainbird.ai')]}.rainbird.ai"
    if host.startswith("api."):
        return f"{scheme}://app.{host[len('api.') :]}"
    return url


class QuestionForm(str, enum.Enum):
    """How a question must be answered.

    - ``FIRST``: yes/no confirmation of a full triple ("Does Tom speak French?").
    - ``SECOND_OBJECT``: the engine knows the subject and asks for the object
      ("What language does Tom speak?"). The answer goes in ``object``.
    - ``SECOND_SUBJECT``: the engine knows the object and asks for the subject
      ("Who speaks French?"). The answer goes in ``subject``.
    """

    FIRST = "first"
    SECOND_OBJECT = "second_object"
    SECOND_SUBJECT = "second_subject"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class FactSource(str, enum.Enum):
    """Where a fact came from (``source`` on evidence and session facts)."""

    KNOWLEDGE_MAP = "knowledgemap"
    RULE = "rule"
    ANSWER = "answer"
    INJECTION = "injection"
    DATASOURCE = "datasource"
    SYNTHESIS = "synthesis"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: Any) -> FactSource | str | None:
        if value is None:
            return None
        try:
            return cls(str(value))
        except ValueError:
            return str(value)


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Fact:
    """A subject/relationship/object triple to inject into a session.

    ``certainty`` is 0-100 (100 means certain; the API documents 1-100).
    """

    subject: str
    relationship: str
    object: Any
    certainty: int | float = 100
    metadata: JSON | None = None
    """Optional ``metadata`` sent with the fact; the API uses ``{"source": "system" | "user"}``
    to record whether the fact came from another system (the default) or from a person."""

    def to_payload(self) -> JSON:
        payload: JSON = {
            "subject": self.subject,
            "relationship": self.relationship,
            "object": self.object,
            "certainty": self.certainty,
        }
        if self.metadata is not None:
            payload["metadata"] = copy.deepcopy(self.metadata)
        return payload


@dataclass(frozen=True)
class Answer:
    """One answer to send to ``POST /{sessionID}/response``.

    - Second Form questions: fill ``subject``/``object`` (whichever was asked for).
    - First Form questions: keep the question's triple and set ``answer`` to
      ``"yes"`` or ``"no"``.
    - Skipping (only when the question has ``allowUnknown``): set
      ``unanswered=True`` and leave the unknown side as ``""``.
    """

    subject: str
    relationship: str
    object: Any
    certainty: int | float = 100
    answer: str | None = None
    unanswered: bool = False

    def to_payload(self) -> JSON:
        payload: JSON = {
            "subject": self.subject,
            "relationship": self.relationship,
            "object": self.object,
            "certainty": self.certainty,
        }
        if self.answer is not None:
            payload["answer"] = self.answer
        if self.unanswered:
            payload["unanswered"] = True
        return payload


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class StartResponse(_RawMixin):
    """Response to ``GET /start/{kmID}``."""

    session_id: str
    km_version_id: str | None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> StartResponse:
        version = _as_dict(_get(data, "kmVersion"))
        session_id = _get(data, "id", "sessionID", "sid")
        if not session_id:
            raise ValueError("start response did not include a session id")
        return cls(
            session_id=str(session_id),
            km_version_id=(
                str(version["id"]) if version.get("id") is not None else _get(data, "kmVersionID")
            ),
            raw=data,
        )


@dataclass(frozen=True)
class InjectResponse(_RawMixin):
    """Response to ``POST /{sessionID}/inject`` (``{"result": "OK"}``)."""

    result: str | None
    raw: JSON = field(repr=False, default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result is None or str(self.result).strip().upper() in {"OK", "TRUE", "SUCCESS"}

    @classmethod
    def from_dict(cls, data: JSON) -> InjectResponse:
        result = _get(data, "result", "ok", "status")
        return cls(result=None if result is None else str(result), raw=data)


@dataclass(frozen=True)
class Concept(_RawMixin):
    """A candidate answer offered with a question (``question.concepts[]``)."""

    name: str
    value: Any
    concept_type: str | None = None
    data_type: str | None = None
    invalid_response: bool = False
    fsid: int | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> Concept:
        value = _get(data, "value")
        name = _get(data, "name")
        return cls(
            name=str(name if name is not None else value),
            value=value if value is not None else name,
            concept_type=_get(data, "conceptType"),
            data_type=_get(data, "type", "dataType"),
            invalid_response=_as_bool(_get(data, "invalidResponse")),
            fsid=_get(data, "fsid"),
            raw=data,
        )


@dataclass(frozen=True)
class Question(_RawMixin):
    """A question the reasoning engine needs answered before it can decide.

    Use :attr:`form` (or :attr:`is_first_form` / :attr:`asks_for_subject` /
    :attr:`asks_for_object`) to work out what to send back, and
    :attr:`options` for the candidate values in ``concepts``.
    """

    subject: str | None
    relationship: str
    object: Any
    type: str
    prompt: str
    data_type: str | None = None
    object_type: str | None = None
    plural: bool = False
    allow_cf: bool = False
    allow_unknown: bool = False
    can_add: bool = True
    known_answers: list[Any] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    object_metadata: JSON = field(default_factory=dict)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> Question:
        return cls(
            subject=_get(data, "subject"),
            relationship=str(_get(data, "relationship", default="")),
            object=_get(data, "object"),
            type=str(_get(data, "type", default="")),
            prompt=str(_get(data, "prompt", default="")),
            data_type=_get(data, "dataType"),
            object_type=_get(data, "objectType"),
            plural=_as_bool(_get(data, "plural")),
            allow_cf=_as_bool(_get(data, "allowCF", "allowCf")),
            allow_unknown=_as_bool(_get(data, "allowUnknown")),
            can_add=_as_bool(_get(data, "canAdd"), default=True),
            known_answers=list(_as_list(_get(data, "knownAnswers"))),
            concepts=[Concept.from_dict(_as_dict(c)) for c in _as_list(_get(data, "concepts"))],
            object_metadata=_as_dict(_get(data, "objectMetadata")),
            raw=data,
        )

    @property
    def form(self) -> QuestionForm:
        kind = self.type.lower()
        if "first" in kind:
            return QuestionForm.FIRST
        if "second" in kind:
            if "subject" in kind:
                return QuestionForm.SECOND_SUBJECT
            if "object" in kind:
                return QuestionForm.SECOND_OBJECT
            # Legacy engines report plain "Second Form": infer from the missing side.
            if self.subject in (None, "") and self.object not in (None, ""):
                return QuestionForm.SECOND_SUBJECT
            return QuestionForm.SECOND_OBJECT
        return QuestionForm.UNKNOWN

    @property
    def is_first_form(self) -> bool:
        """True for yes/no confirmation questions."""
        return self.form is QuestionForm.FIRST

    @property
    def is_second_form(self) -> bool:
        return self.form in (QuestionForm.SECOND_OBJECT, QuestionForm.SECOND_SUBJECT)

    @property
    def asks_for_subject(self) -> bool:
        return self.form is QuestionForm.SECOND_SUBJECT

    @property
    def asks_for_object(self) -> bool:
        return self.form is QuestionForm.SECOND_OBJECT

    @property
    def options(self) -> list[Any]:
        """Candidate values from ``concepts`` (empty for free-text/number/date questions)."""
        return [c.value for c in self.concepts if not c.invalid_response]

    @property
    def is_truth(self) -> bool:
        return (self.data_type or "").lower() == "truth"

    def __str__(self) -> str:
        if self.prompt:
            return self.prompt
        parts = [_part(self.subject), self.relationship, _part(self.object)]
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class QuestionResponse(_RawMixin):
    """The engine needs more information: one or more questions to answer."""

    question: Question
    extra_questions: list[Question] = field(default_factory=list)
    sid: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    is_question: Literal[True] = field(default=True, init=False, repr=False)
    is_result: Literal[False] = field(default=False, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: JSON) -> QuestionResponse:
        return cls(
            question=Question.from_dict(_as_dict(_get(data, "question"))),
            extra_questions=[
                Question.from_dict(_as_dict(q)) for q in _as_list(_get(data, "extraQuestions"))
            ],
            sid=_get(data, "sid", "sessionID"),
            raw=data,
        )

    @property
    def questions(self) -> list[Question]:
        """The main question followed by any extra questions."""
        return [self.question, *self.extra_questions]

    @property
    def results(self) -> list[ResultItem]:
        """Always empty on a question response (mirrors :class:`ResultResponse`)."""
        return []


@dataclass(frozen=True)
class ResultItem(_RawMixin):
    """One decision produced by a query."""

    subject: Any
    relationship: str
    object: Any
    certainty: int | float | None
    fact_id: str | None
    relationship_type: str | None = None
    subject_metadata: JSON = field(default_factory=dict)
    object_metadata: JSON = field(default_factory=dict)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> ResultItem:
        subject = _get(data, "subject", "Subject")
        if subject is None:
            subject = _get(data, "subjectValue", "SubjectValue")
        obj = _get(data, "object", "Object")
        if obj is None:
            obj = _get(data, "objectValue", "ObjectValue")
        return cls(
            subject=subject,
            relationship=str(_get(data, "relationship", "Relationship", default="")),
            object=obj,
            certainty=_as_number(_get(data, "certainty", "Certainty")),
            fact_id=_get(data, "factID", "FactID", "factId"),
            relationship_type=_get(data, "relationshipType", "RelationshipType"),
            subject_metadata=_as_dict(_get(data, "subjectMetadata", "SubjectMetadata")),
            object_metadata=_as_dict(_get(data, "objectMetadata", "ObjectMetadata")),
            raw=data,
        )

    def __str__(self) -> str:
        cert = f" ({self.certainty}%)" if self.certainty is not None else ""
        parts = [_part(self.subject), self.relationship, _part(self.object)]
        return " ".join(p for p in parts if p) + cert


@dataclass(frozen=True)
class ResultResponse(_RawMixin):
    """The engine reached a decision. ``results`` may be empty if nothing could be inferred."""

    results: list[ResultItem]
    sid: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    is_question: Literal[False] = field(default=False, init=False, repr=False)
    is_result: Literal[True] = field(default=True, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: JSON) -> ResultResponse:
        return cls(
            results=[ResultItem.from_dict(_as_dict(r)) for r in _as_list(_get(data, "result"))],
            sid=_get(data, "sid", "sessionID"),
            raw=data,
        )

    @property
    def question(self) -> None:
        """Always ``None`` on a result response (mirrors :class:`QuestionResponse`)."""
        return None

    @property
    def best(self) -> ResultItem | None:
        """The result with the highest certainty, or ``None`` when there are no results."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.certainty if r.certainty is not None else -1)

    @property
    def fact_ids(self) -> list[str]:
        return [r.fact_id for r in self.results if r.fact_id]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[ResultItem]:
        return iter(self.results)


QueryResponse = QuestionResponse | ResultResponse
"""What ``query``, ``respond``, ``answer``, ``skip`` and ``undo`` return."""


def parse_query_response(data: Any) -> QueryResponse:
    """Classify a decisions-endpoint body as a question or a result.

    Raises :class:`~rainbird_ai.UnexpectedResponseError` (with the body on ``.body``) when it
    is neither, so callers can surface the raw payload instead of guessing.
    """
    if not isinstance(data, dict):
        raise UnexpectedResponseError(
            f"unexpected response body type: {type(data).__name__}", body=data
        )
    if isinstance(data.get("question"), dict):
        return QuestionResponse.from_dict(data)
    if "result" in data:
        return ResultResponse.from_dict(data)
    keys = sorted(data.keys())
    raise UnexpectedResponseError(
        f"response is neither a questionResponse nor a resultResponse (keys: {keys})", body=data
    )


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class FactValue(_RawMixin):
    """One side (subject or object) of a stored fact."""

    value: Any
    concept: str | None = None
    data_type: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> FactValue:
        if not isinstance(data, dict):
            return cls(value=data, raw={"value": data})
        return cls(
            value=_get(data, "value"),
            concept=_get(data, "concept", "type"),
            data_type=_get(data, "dataType"),
            raw=data,
        )

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SessionFact(_RawMixin):
    """A fact held in a session (from ``/analysis/session?filter=facts`` or evidence)."""

    subject: FactValue
    relationship: str
    object: FactValue
    certainty: int | float | None
    id: str | None = None
    source: FactSource | str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> SessionFact:
        rel = _get(data, "relationship")
        if isinstance(rel, dict):
            rel = _get(rel, "type", "name", "value")
        return cls(
            id=_get(data, "id", "factID"),
            subject=FactValue.from_dict(_get(data, "subject")),
            relationship=str(rel if rel is not None else ""),
            object=FactValue.from_dict(_get(data, "object")),
            certainty=_as_number(_get(data, "certainty")),
            source=FactSource.parse(_get(data, "source")),
            raw=data,
        )

    def __str__(self) -> str:
        cert = f" ({self.certainty}%)" if self.certainty is not None else ""
        parts = [_part(self.subject.value), self.relationship, _part(self.object.value)]
        return " ".join(p for p in parts if p) + cert


@dataclass(frozen=True)
class SessionFacts(_RawMixin):
    """Facts visible to a session, grouped by scope."""

    global_: list[SessionFact] = field(default_factory=list)
    local: list[SessionFact] = field(default_factory=list)
    context: list[SessionFact] = field(default_factory=list)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> SessionFacts:
        body = _as_dict(_get(data, "facts")) or data
        return cls(
            global_=[SessionFact.from_dict(_as_dict(f)) for f in _as_list(body.get("global"))],
            local=[SessionFact.from_dict(_as_dict(f)) for f in _as_list(body.get("local"))],
            context=[SessionFact.from_dict(_as_dict(f)) for f in _as_list(body.get("context"))],
            raw=data,
        )

    @property
    def all(self) -> list[SessionFact]:
        return [*self.global_, *self.context, *self.local]

    def __len__(self) -> int:
        return len(self.all)

    def __iter__(self) -> Iterator[SessionFact]:
        return iter(self.all)


@dataclass(frozen=True)
class KnowledgeMapVersion(_RawMixin):
    """Knowledge map + version used by a session (``/analysis/session?filter=version``)."""

    id: str | None
    name: str | None
    version_id: str | None
    version_status: str | None
    version_created: _dt.datetime | None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> KnowledgeMapVersion:
        km = _as_dict(_get(data, "km")) or data
        return cls(
            id=_get(km, "id"),
            name=_get(km, "name"),
            version_id=_get(km, "versionID", "versionId"),
            version_status=_get(km, "versionStatus"),
            version_created=_parse_datetime(_get(km, "versionCreated")),
            raw=data,
        )

    @property
    def is_draft(self) -> bool:
        return (self.version_status or "").lower() == "draft"

    @property
    def version_number(self) -> int | None:
        """The explicit ``versionNumber`` when the API sends one, else ``3`` for
        ``versionStatus == "Version3"`` and ``None`` for drafts."""
        km = _as_dict(_get(self.raw, "km")) or self.raw
        explicit = _get(km, "versionNumber")
        if isinstance(explicit, bool):
            explicit = None
        if isinstance(explicit, int):
            return explicit
        if isinstance(explicit, float) and explicit.is_integer():
            return int(explicit)
        if isinstance(explicit, str) and explicit.strip().isdigit():
            return int(explicit.strip())
        status = (self.version_status or "").lower()
        digits = "".join(ch for ch in status if ch.isdigit())
        return int(digits) if status.startswith("version") and digits else None


@dataclass(frozen=True)
class Condition(_RawMixin):
    """A relationship condition of a rule, and the fact that satisfied it."""

    subject: Any
    relationship: str
    object: Any
    certainty: int | float | None = None
    salience: int | float | None = None
    impact: int | float | None = None
    fact_id: str | None = None
    fact_key: str | None = None
    object_type: str | None = None
    alt: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> Condition:
        return cls(
            subject=_get(data, "subject"),
            relationship=str(_get(data, "relationship", default="")),
            object=_get(data, "object"),
            certainty=_as_number(_get(data, "certainty")),
            salience=_as_number(_get(data, "salience")),
            impact=_as_number(_get(data, "impact")),
            fact_id=_get(data, "factID", "factId"),
            fact_key=None if _get(data, "factKey") is None else str(_get(data, "factKey")),
            object_type=_get(data, "objectType"),
            alt=_get(data, "alt"),
            raw=data,
        )


@dataclass(frozen=True)
class ExpressionCondition(_RawMixin):
    """An expression condition of a rule (``%AGE is greater than 18``) and its evaluation."""

    text: str | None
    value: str | None
    was_met: bool | None = None
    functions: JSON = field(default_factory=dict)
    salience: int | float | None = None
    impact: int | float | None = None
    alt: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> ExpressionCondition:
        expr = _as_dict(_get(data, "expression"))
        was_met = _get(data, "wasMet")
        return cls(
            text=_get(expr, "text"),
            value=_get(expr, "value"),
            was_met=None if was_met is None else _as_bool(was_met),
            functions=_as_dict(_get(expr, "functions")),
            salience=_as_number(_get(data, "salience")),
            impact=_as_number(_get(data, "impact")),
            alt=_get(data, "alt"),
            raw=data,
        )

    @property
    def fact_ids(self) -> list[str]:
        """Fact IDs referenced by functions evaluated in this expression."""
        ids: list[str] = []
        for call in self.functions.values():
            for fact in _as_list(_as_dict(call).get("facts")):
                fid = _as_dict(fact).get("factID")
                if fid:
                    ids.append(str(fid))
        return ids


@dataclass(frozen=True)
class Rule(_RawMixin):
    """The rule that inferred a fact: variable bindings and evaluated conditions."""

    bindings: JSON = field(default_factory=dict)
    conditions: list[Condition | ExpressionCondition] = field(default_factory=list)
    max_certainty: int | float | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> Rule:
        conditions: list[Condition | ExpressionCondition] = []
        for item in _as_list(_get(data, "conditions")):
            cond = _as_dict(item)
            if "expression" in cond:
                conditions.append(ExpressionCondition.from_dict(cond))
            else:
                conditions.append(Condition.from_dict(cond))
        return cls(
            bindings=_as_dict(_get(data, "bindings")),
            conditions=conditions,
            max_certainty=_as_number(_get(data, "ruleMaxCertainty")),
            raw=data,
        )

    @property
    def relationship_conditions(self) -> list[Condition]:
        return [c for c in self.conditions if isinstance(c, Condition)]

    @property
    def expression_conditions(self) -> list[ExpressionCondition]:
        return [c for c in self.conditions if isinstance(c, ExpressionCondition)]


@dataclass(frozen=True)
class Evidence(_RawMixin):
    """Why a fact holds: its source and, for inferred facts, the rule that produced it."""

    fact_id: str
    source: FactSource | str | None
    fact: SessionFact
    time: int | float | None = None
    rule: Rule | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> Evidence:
        fact_data = _as_dict(_get(data, "fact"))
        rule_data = _get(data, "rule")
        # The envelope carries the fact's id and source; copy both onto the nested fact so a
        # SessionFact from evidence looks like one from ``/analysis/session?filter=facts``.
        fact_payload = {**fact_data, "id": _get(data, "factID")}
        if fact_payload.get("source") is None and _get(data, "source") is not None:
            fact_payload["source"] = _get(data, "source")
        return cls(
            fact_id=str(_get(data, "factID", "factId", "id", default="")),
            source=FactSource.parse(_get(data, "source")),
            fact=SessionFact.from_dict(fact_payload),
            time=_as_number(_get(data, "time")),
            rule=Rule.from_dict(rule_data) if isinstance(rule_data, dict) else None,
            raw=data,
        )

    @property
    def timestamp(self) -> _dt.datetime | None:
        return _parse_datetime(self.time)

    @property
    def is_inferred(self) -> bool:
        return self.rule is not None

    @property
    def child_fact_ids(self) -> list[str]:
        """Fact IDs one level down the evidence tree (empty at a leaf)."""
        if self.rule is None:
            return []
        ids: list[str] = []
        for cond in self.rule.conditions:
            if isinstance(cond, Condition):
                if cond.fact_id:
                    ids.append(cond.fact_id)
            else:
                ids.extend(cond.fact_ids)
        seen: set[str] = set()
        unique: list[str] = []
        for fid in ids:
            if fid not in seen and fid != self.fact_id:
                seen.add(fid)
                unique.append(fid)
        return unique


@dataclass(frozen=True)
class InteractionEvent(_RawMixin):
    """One entry from the interaction log (``start``, ``query``, ``question``, ``answer``...)."""

    event: str
    created: _dt.datetime | None
    values: JSON = field(default_factory=dict)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> InteractionEvent:
        return cls(
            event=str(_get(data, "event", default="")),
            created=_parse_datetime(_get(data, "created")),
            values=_as_dict(_get(data, "values")),
            raw=data,
        )


# --------------------------------------------------------------------------- natural language


@dataclass(frozen=True)
class NLError(_RawMixin):
    """Error object returned inside ``/nl/interact`` and ``/nl/explain`` bodies."""

    code: int | None
    message: str
    suggested_chat_response: str | None = None
    status_code: str | int | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> NLError:
        code = _as_number(_get(data, "code"))
        return cls(
            code=int(code) if code is not None else None,
            message=str(_get(data, "message", "error", default="")),
            suggested_chat_response=_get(data, "suggestedChatResponse"),
            status_code=_get(data, "statusCode"),
            raw=data,
        )


@dataclass(frozen=True)
class NLMetadata(_RawMixin):
    """Timing and token usage reported by the natural language endpoints."""

    endpoint: str | None
    session_id: str | None
    query_submitted_at: _dt.datetime | None
    language_engine_response_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    api_version: str | None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> NLMetadata:
        tokens = _as_dict(_get(data, "llmTokens"))

        def _int(value: Any) -> int | None:
            num = _as_number(value)
            return int(num) if num is not None else None

        return cls(
            endpoint=_get(data, "endpoint"),
            session_id=_get(data, "sessionID"),
            query_submitted_at=_parse_datetime(_get(data, "querySubmittedAt")),
            language_engine_response_ms=_int(_get(data, "languageEngineResponseTime")),
            prompt_tokens=_int(_get(tokens, "promptTokens")),
            completion_tokens=_int(_get(tokens, "completionTokens")),
            total_tokens=_int(_get(tokens, "totalTokens")),
            api_version=_get(data, "apiVersion"),
            raw=data,
        )


@dataclass(frozen=True)
class InteractResponse(_RawMixin):
    """Response to ``POST /nl/interact`` (beta).

    ``response_type`` is ``"question"``, ``"result"`` or ``"error"``. An error is a
    legitimate outcome (for example the prompt did not contain a recognisable
    query), so it is returned rather than raised; call :meth:`raise_for_error`
    if you prefer an exception.
    """

    response_type: str
    query: JSON | None = None
    questions: list[Question] = field(default_factory=list)
    results: list[ResultItem] = field(default_factory=list)
    error: NLError | None = None
    metadata: NLMetadata | None = None
    injected_facts: list[JSON] = field(default_factory=list)
    invalid_facts: list[JSON] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> InteractResponse:
        facts = _as_dict(_get(data, "facts"))
        error = _get(data, "error")
        metadata = _get(data, "metadata")
        response_type = str(_get(data, "responseType", default="") or "")
        if not response_type:
            response_type = (
                "error"
                if isinstance(error, dict)
                else "question"
                if _as_list(_get(data, "questions"))
                else "result"
                if _get(data, "results") is not None
                else "unknown"
            )
        return cls(
            response_type=response_type,
            query=_as_dict(_get(data, "query")) or None,
            questions=[Question.from_dict(_as_dict(q)) for q in _as_list(_get(data, "questions"))],
            results=[ResultItem.from_dict(_as_dict(r)) for r in _as_list(_get(data, "results"))],
            error=NLError.from_dict(error) if isinstance(error, dict) else None,
            metadata=NLMetadata.from_dict(metadata) if isinstance(metadata, dict) else None,
            injected_facts=[_as_dict(f) for f in _as_list(facts.get("injected"))],
            invalid_facts=[_as_dict(f) for f in _as_list(facts.get("invalid"))],
            unmatched=[str(u) for u in _as_list(facts.get("unmatched"))],
            raw=data,
        )

    @property
    def is_question(self) -> bool:
        return self.response_type == "question"

    @property
    def is_result(self) -> bool:
        return self.response_type == "result"

    @property
    def is_error(self) -> bool:
        return self.response_type == "error" or (
            self.error is not None and not self.questions and not self.results
        )

    @property
    def question(self) -> Question | None:
        return self.questions[0] if self.questions else None

    @property
    def prompt(self) -> str | None:
        """Text to show a user: the question prompt, or the suggested chat response on error."""
        if self.question is not None:
            return self.question.prompt
        if self.error is not None:
            return self.error.suggested_chat_response or self.error.message
        return None

    def raise_for_error(self) -> InteractResponse:
        """Raise :class:`~rainbird_ai.errors.NaturalLanguageError` on an error response."""
        if self.is_error:
            err = self.error
            raise NaturalLanguageError(
                err.message if err else "interact returned an error response",
                code=err.code if err else None,
                suggested_chat_response=err.suggested_chat_response if err else None,
                status_code=err.status_code if err else None,
                body=self.raw,
            )
        return self


@dataclass(frozen=True)
class ExplainResponse(_RawMixin):
    """Response to ``POST /nl/explain`` (beta): a natural-language explanation of a fact."""

    explanation: str | None
    error: NLError | None = None
    metadata: NLMetadata | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON) -> ExplainResponse:
        error = _get(data, "error")
        metadata = _get(data, "metadata")
        explanation = _get(data, "explanation")
        return cls(
            explanation=None if explanation is None else str(explanation),
            error=NLError.from_dict(error) if isinstance(error, dict) else None,
            metadata=NLMetadata.from_dict(metadata) if isinstance(metadata, dict) else None,
            raw=data,
        )

    def __str__(self) -> str:
        return self.explanation or ""


# ------------------------------------------------------------------ knowledge maps (undocumented)


@dataclass(frozen=True)
class KnowledgeMapSource(_RawMixin):
    """RBLang source and summary of a knowledge map (``GET /analysis/file/{kmID}``).

    This endpoint is not part of the public OpenAPI specification and may change.
    """

    km_id: str
    rblang: str
    concepts: list[Any] = field(default_factory=list)
    relationships: list[Any] = field(default_factory=list)
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSON, *, km_id: str = "") -> KnowledgeMapSource:
        rblang = _get(data, "rblang", "rbl", "xml", default="")
        return cls(
            km_id=str(_get(data, "kmID", "kmid", "id", default=km_id) or km_id),
            rblang=str(rblang or ""),
            concepts=list(_as_list(_get(data, "concepts"))),
            relationships=list(_as_list(_get(data, "rels", "relationships"))),
            raw=data,
        )


@dataclass(frozen=True)
class CreateMapResult(_RawMixin):
    """Result of ``POST /maps`` (create a knowledge map from RBLang).

    This endpoint is not part of the public OpenAPI specification and may change.
    ``km_id`` is the GUID to use with ``start()``; ``id`` is Rainbird's numeric id. ``name``
    and ``description`` come from the response when it echoes them, else from the request.
    """

    km_id: str
    name: str | None
    id: str | int | None = None
    description: str | None = None
    raw: JSON = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(
        cls, data: JSON, *, name: str | None = None, description: str | None = None
    ) -> CreateMapResult:
        km_id = _get(data, "kmID", "kmid", "kmId", "mapId", "knowledge_map_id")
        numeric_id = _get(data, "id")
        if km_id is None:
            km_id = numeric_id
        if km_id is None:
            raise ValueError("create map response did not include a knowledge map id")
        return cls(
            km_id=str(km_id),
            name=_get(data, "name", default=name),
            id=numeric_id,
            description=_get(data, "description", default=description),
            raw=data,
        )
