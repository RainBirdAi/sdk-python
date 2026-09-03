"""Build and validate ``inject`` and ``response`` payloads.

This module is pure (no I/O) and shared by the sync and async clients. It
encodes the rules of the Rainbird question/answer protocol:

* **First Form** questions ("Does Tom speak French?") are answered with the
  question's own triple plus ``"answer": "yes" | "no"``.
* **Second Form Object** questions ("What does Tom speak?") put the value in
  ``object``; **Second Form Subject** questions ("Who speaks French?") put it in
  ``subject``.
* ``plural`` questions accept several values, sent as several answers.
* A question can be skipped with ``"unanswered": true`` when ``allowUnknown``.
* ``certainty`` is 0-100 (the API documents 1-100; 100 = certain).
* Values are coerced for the wire: ``date``/``datetime`` become ``YYYY-MM-DD``
  (Rainbird dates are calendar dates or epoch milliseconds), ``Decimal`` becomes
  a number, enums use their value. ``None``, dicts and lists are rejected.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .errors import InvalidAnswerError, InvalidFactError
from .models import Answer, Fact, Question, QuestionForm

__all__ = [
    "SKIP",
    "build_answers",
    "build_skip",
    "coerce_value",
    "normalize_answers",
    "normalize_facts",
    "validate_certainty",
]

_YES = {"yes", "y", "true", "t", "1"}
_NO = {"no", "n", "false", "f", "0"}
_ANSWER_KEYS = {"subject", "relationship", "object", "certainty", "answer", "unanswered", "cf"}
_FACT_KEYS = {"subject", "relationship", "object", "certainty", "cf"}
_SCALAR = (str, int, float, bool)


class _Skip:
    """Sentinel returned by an answerer to skip the current question."""

    _instance: _Skip | None = None

    def __new__(cls) -> _Skip:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIP"

    def __bool__(self) -> bool:
        return False


SKIP = _Skip()
"""Return this from an answerer callback to skip a question (needs ``allowUnknown``)."""


def validate_certainty(certainty: Any) -> int | float:
    """Return ``certainty`` as a number in ``[0, 100]`` or raise :class:`InvalidFactError`."""
    if isinstance(certainty, bool):
        raise InvalidFactError("certainty must be a number between 0 and 100, not a boolean")
    if isinstance(certainty, str):
        try:
            certainty = float(certainty)
        except ValueError as exc:
            raise InvalidFactError(
                f"certainty must be a number between 0 and 100, got {certainty!r}"
            ) from exc
    if isinstance(certainty, decimal.Decimal):
        certainty = float(certainty)
    if not isinstance(certainty, (int, float)):
        raise InvalidFactError(
            f"certainty must be a number between 0 and 100, got {type(certainty).__name__}"
        )
    if math.isnan(certainty) or not 0 <= certainty <= 100:
        raise InvalidFactError(f"certainty must be between 0 and 100, got {certainty!r}")
    if isinstance(certainty, float) and certainty.is_integer():
        return int(certainty)
    return certainty


def coerce_value(value: Any) -> str | int | float | bool:
    """Convert a Python value into something the API accepts for subject/object.

    Strings, ints, finite floats and bools pass through. Enum members use their
    ``value``. ``date`` and ``datetime`` become ``YYYY-MM-DD`` strings (Rainbird
    ``date`` values are calendar dates or epoch milliseconds; the time of day of a
    ``datetime`` is dropped, pass an ``int`` of epoch milliseconds if you need an
    instant). ``Decimal`` becomes an ``int`` when integral, otherwise a ``float``.
    Anything else (``None``, dicts, lists, NaN/infinity) raises
    :class:`InvalidAnswerError`.
    """
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAnswerError(f"value must be a finite number, got {value!r}")
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise InvalidAnswerError(f"value must be a finite number, got {value!r}")
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    raise InvalidAnswerError(
        f"unsupported value type {type(value).__name__}; use a string, number, bool or date"
    )


def _text(value: Any) -> str:
    """``None`` -> ``""``; everything else -> ``str``."""
    return "" if value is None else str(value)


def _is_many(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset)) or (
        isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))
    )


def _truth_of(value: Any) -> bool | None:
    """Interpret a yes/no/true/false-like value as a bool, or ``None`` if it is not one."""
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _YES:
        return True
    if text in _NO:
        return False
    return None


def _to_yes_no(value: Any) -> str:
    truth = _truth_of(value)
    if truth is None:
        raise InvalidAnswerError(f"First Form questions take yes/no (or True/False); got {value!r}")
    return "yes" if truth else "no"


def _unanswered_flag(value: Any) -> bool:
    if value is None:
        return False
    truth = _truth_of(value)
    if truth is None or (not isinstance(value, bool) and isinstance(value, (int, float))):
        raise InvalidFactError(f"unanswered must be a boolean, got {value!r}")
    return truth


def _same_option(value: Any, option: Any, *, numeric: bool) -> bool:
    """String equality, plus numeric equality when allowed (5.0 == "5")."""
    if str(value) == str(option):
        return True
    if not numeric:
        return False
    try:
        return float(value) == float(option)
    except (TypeError, ValueError, OverflowError):
        return False


def _check_option(question: Question, value: Any, strict: bool) -> Any:
    """Validate ``value`` against ``concepts`` when the question does not allow new values.

    Returns the value to send. An exact textual match keeps the caller's value; a
    purely numeric match (``5.0`` for the option ``"5"``) returns the option's own
    spelling so the wire value is the concept instance the engine knows.
    """
    if not strict or question.can_add:
        return value
    options = question.options
    if not options:  # nothing usable to check against; leave validation to the API
        return value
    if question.is_truth and question.form is QuestionForm.SECOND_OBJECT:
        allowed = {t for t in (_truth_of(opt) for opt in options) if t is not None}
        if not allowed or _truth_of(value) in allowed:
            return value
    else:
        for opt in options:
            if _same_option(value, opt, numeric=False):
                return value
        value_is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        question_is_numeric = question.form is QuestionForm.SECOND_OBJECT and (
            question.data_type or ""
        ).lower() in {"number", "integer", "float"}
        for opt in options:
            option_is_number = isinstance(opt, (int, float)) and not isinstance(opt, bool)
            if (value_is_number or question_is_numeric or option_is_number) and _same_option(
                value, opt, numeric=True
            ):
                return opt
    opts = ", ".join(sorted(str(opt) for opt in options))
    raise InvalidAnswerError(
        f"{value!r} is not one of the allowed answers for {question.relationship!r} "
        f"and the question does not allow new values. Options: {opts}"
    )


def _check_truth(question: Question, value: Any) -> Any:
    """Normalise truth-typed *object* values to real booleans."""
    if not question.is_truth:
        return value
    truth = _truth_of(value)
    if truth is None:
        raise InvalidAnswerError(
            f"{question.relationship!r} expects a truth value (true/false); got {value!r}"
        )
    return truth


def build_answers(
    question: Question,
    value: Any,
    *,
    certainty: int | float = 100,
    strict: bool = True,
) -> list[Answer]:
    """Turn a Python value into the answer(s) for ``question``.

    Args:
        question: The question being answered (usually ``session.pending_question``).
        value: For First Form questions a yes/no or bool. For Second Form questions a
            single value, or a list/tuple/set of values when the question is ``plural``.
            Pass :data:`SKIP` to skip (requires ``allowUnknown``).
        certainty: 0-100 confidence in the answer. Applied to every generated answer.
            Only meaningful when the question has ``allowCF``; sent regardless.
        strict: When ``True`` (default) reject values outside ``question.options`` if
            ``canAdd`` is false, multiple values for non-plural questions, and skips when
            ``allowUnknown`` is false. Set ``False`` to let the API be the judge.

    Raises:
        InvalidAnswerError: if the value cannot be mapped onto the question.
    """
    if value is SKIP:
        return build_skip(question, strict=strict)
    cert = validate_certainty(certainty)
    form = question.form

    if form is QuestionForm.FIRST:
        if _is_many(value):
            items = list(value)
            if len(items) != 1:
                raise InvalidAnswerError("First Form questions take exactly one yes/no answer")
            value = items[0]
        if question.object is None:
            raise InvalidAnswerError("First Form question is missing its 'object'; cannot answer")
        return [
            Answer(
                subject=_text(question.subject),
                relationship=question.relationship,
                object=question.object,
                certainty=cert,
                answer=_to_yes_no(value),
            )
        ]

    if form is QuestionForm.UNKNOWN:
        raise InvalidAnswerError(
            f"unsupported question type {question.type!r}; build Answer objects manually "
            "and call respond()"
        )

    values: list[Any] = list(value) if _is_many(value) else [value]
    if not values:
        raise InvalidAnswerError("at least one value is required to answer the question")
    if strict and len(values) > 1 and not question.plural:
        raise InvalidAnswerError(
            f"{question.relationship!r} accepts a single answer; got {len(values)} values"
        )

    answers: list[Answer] = []
    for item in values:
        if form is QuestionForm.SECOND_SUBJECT:
            # dataType describes the object side; subjects are always strings.
            coerced = _check_option(question, coerce_value(item), strict)
            answers.append(
                Answer(
                    subject=_text(coerced),
                    relationship=question.relationship,
                    object=question.object if question.object is not None else "",
                    certainty=cert,
                )
            )
        else:
            coerced = _check_option(question, coerce_value(_check_truth(question, item)), strict)
            answers.append(
                Answer(
                    subject=_text(question.subject),
                    relationship=question.relationship,
                    object=coerced,
                    certainty=cert,
                )
            )
    return answers


def build_skip(question: Question, *, strict: bool = True) -> list[Answer]:
    """Build the ``unanswered`` payload that skips ``question``.

    The unknown side is sent as ``""``; First Form questions keep their full triple
    because both sides are already known.
    """
    if strict and not question.allow_unknown:
        raise InvalidAnswerError(
            f"{question.relationship!r} cannot be skipped: "
            "the question does not allow unknown answers"
        )
    if question.form is QuestionForm.SECOND_SUBJECT:
        return [
            Answer(
                subject="",
                relationship=question.relationship,
                object=question.object if question.object is not None else "",
                certainty=100,
                unanswered=True,
            )
        ]
    return [
        Answer(
            subject=_text(question.subject),
            relationship=question.relationship,
            object="" if question.form is not QuestionForm.FIRST else question.object,
            certainty=100,
            unanswered=True,
        )
    ]


def _normalize_answer(answer: Answer) -> Answer:
    """Return a copy of ``answer`` with wire-ready values (coerced, validated)."""
    return dataclasses.replace(
        answer,
        subject=_text(answer.subject),
        relationship=_text(answer.relationship),
        object=coerce_value(answer.object),
        certainty=validate_certainty(answer.certainty),
        answer=None if answer.answer is None else _to_yes_no(answer.answer),
        unanswered=_unanswered_flag(answer.unanswered),
    )


def _answer_from_mapping(data: Mapping[str, Any]) -> Answer:
    unknown = set(data) - _ANSWER_KEYS
    if unknown:
        raise InvalidFactError(f"answer has unexpected field(s): {', '.join(sorted(unknown))}")
    missing = [k for k in ("subject", "relationship", "object") if k not in data]
    if missing:
        raise InvalidFactError(f"answer is missing required field(s): {', '.join(missing)}")
    certainty = data.get("certainty", data.get("cf", 100))
    return _normalize_answer(
        Answer(
            subject=_text(data["subject"]),
            relationship=_text(data["relationship"]),
            object=data["object"],
            certainty=100 if certainty is None else certainty,
            answer=data.get("answer"),
            unanswered=data.get("unanswered", False),
        )
    )


def normalize_answers(answers: Any) -> list[dict[str, Any]]:
    """Accept ``Answer``/dict or a sequence of them; return the ``answers`` payload list."""
    items: Sequence[Any]
    if isinstance(answers, (Answer, Mapping)):
        items = [answers]
    elif _is_many(answers):
        items = list(answers)
    else:
        raise InvalidFactError(
            f"answers must be an Answer, a dict, or a list of those; got {type(answers).__name__}"
        )
    if not items:
        raise InvalidFactError("answers must not be empty")
    payload: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Answer):
            payload.append(_normalize_answer(item).to_payload())
        elif isinstance(item, Mapping):
            payload.append(_answer_from_mapping(item).to_payload())
        else:
            raise InvalidFactError(
                f"each answer must be an Answer or a dict, got {type(item).__name__}"
            )
    return payload


def _fact_from_any(item: Any) -> Fact:
    if isinstance(item, Fact):
        return Fact(
            _text(item.subject),
            _text(item.relationship),
            coerce_value(item.object),
            validate_certainty(item.certainty),
        )
    if isinstance(item, Mapping):
        unknown = set(item) - _FACT_KEYS
        if unknown:
            raise InvalidFactError(f"fact has unexpected field(s): {', '.join(sorted(unknown))}")
        missing = [k for k in ("subject", "relationship", "object") if k not in item]
        if missing:
            raise InvalidFactError(f"fact is missing required field(s): {', '.join(missing)}")
        certainty = item.get("certainty", item.get("cf", 100))
        return Fact(
            subject=_text(item["subject"]),
            relationship=_text(item["relationship"]),
            object=coerce_value(item["object"]),
            certainty=validate_certainty(100 if certainty is None else certainty),
        )
    if isinstance(item, (tuple, list)) and 3 <= len(item) <= 4:
        subject, relationship, obj = item[0], item[1], item[2]
        certainty = item[3] if len(item) == 4 else 100
        return Fact(
            _text(subject), _text(relationship), coerce_value(obj), validate_certainty(certainty)
        )
    raise InvalidFactError(
        "each fact must be a Fact, a dict with subject/relationship/object[/certainty], "
        f"or a (subject, relationship, object[, certainty]) tuple; got {type(item).__name__}"
    )


def _looks_like_single_fact(value: Any) -> bool:
    """``("Tom", "lives in", "Germany"[, 90])`` rather than a list of facts."""
    return (
        isinstance(value, (tuple, list))
        and 3 <= len(value) <= 4
        and not any(isinstance(x, (Fact, Mapping, list, tuple, set, frozenset)) for x in value)
    )


def normalize_facts(facts: Any) -> list[dict[str, Any]]:
    """Accept one fact or many (``Fact``, dict or tuple); return the inject payload list."""
    if isinstance(facts, (Fact, Mapping)) or _looks_like_single_fact(facts):
        items: list[Any] = [facts]
    elif _is_many(facts):
        items = list(facts)
    else:
        raise InvalidFactError(
            f"facts must be a Fact, dict, tuple or a list of those; got {type(facts).__name__}"
        )
    if not items:
        raise InvalidFactError("facts must not be empty")
    return [_fact_from_any(item).to_payload() for item in items]
