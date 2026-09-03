"""Tests for ``rainbird_ai._answers``: answer/skip building and payload normalisation.

Pure-function tests use canned questions from ``conftest``; a handful of round-trip
tests drive ``Session`` against the in-memory fake and assert the exact wire payload.
"""

from __future__ import annotations

import copy
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any

import pytest

from rainbird_ai import (
    SKIP,
    Answer,
    Fact,
    InvalidAnswerError,
    InvalidFactError,
    Question,
    QuestionForm,
    Rainbird,
    Session,
    build_answers,
    build_skip,
)
from rainbird_ai._answers import (
    _Skip,
    coerce_value,
    normalize_answers,
    normalize_facts,
    validate_certainty,
)
from rainbird_ai.aio import AsyncRainbird
from tests.conftest import (
    KM_ID,
    QUESTION_FIRST_FORM,
    QUESTION_PLURAL_LOSS,
    QUESTION_SECOND_OBJECT,
    QUESTION_SECOND_SUBJECT,
    FakeRainbird,
)

# --------------------------------------------------------------------------- helpers


def question(base: dict[str, Any], **overrides: Any) -> Question:
    """Build a :class:`Question` from a canned payload with raw keys overridden."""
    return Question.from_dict({**base, **overrides})


def payloads(answers: list[Answer]) -> list[dict[str, Any]]:
    return [a.to_payload() for a in answers]


def ask(client: Rainbird, fake: FakeRainbird, raw: dict[str, Any]) -> Session:
    """Start a session and make ``raw`` its pending question."""
    fake.queue_question(raw)
    session = client.start(KM_ID)
    response = session.query(
        raw["relationship"], subject=raw.get("subject"), object=raw.get("object")
    )
    assert response.is_question
    return session


FIRST = question(QUESTION_FIRST_FORM)
SECOND_OBJECT = question(QUESTION_SECOND_OBJECT)  # canAdd, allowCF, not plural, no allowUnknown
SECOND_SUBJECT = question(QUESTION_SECOND_SUBJECT)  # plural, allowUnknown, canAdd False
PLURAL_LOSS = question(QUESTION_PLURAL_LOSS)  # plural, canAdd False, no allowUnknown
SKIPPABLE_OBJECT = question(QUESTION_SECOND_OBJECT, allowUnknown=True)

TRUTH_RAW: dict[str, Any] = {
    **QUESTION_SECOND_OBJECT,
    "subject": "Tom",
    "relationship": "is tall",
    "dataType": "truth",
    "prompt": "Is Tom tall?",
    "concepts": [],
}
TRUTH = Question.from_dict(TRUTH_RAW)

NUMBER_OPTIONS = question(
    QUESTION_SECOND_OBJECT,
    subject="Tom",
    relationship="age",
    dataType="number",
    canAdd=False,
    concepts=[
        {"conceptType": "age", "name": "5", "type": "number", "value": 5},
        {"conceptType": "age", "name": "10", "type": "number", "value": 10},
    ],
)


class Language(enum.Enum):
    FRENCH = "French"
    GERMAN = "German"


class Loss(enum.Enum):
    THEFT = "theft"


class Country(enum.Enum):
    DE = "Germany"


class StrLanguage(str, enum.Enum):
    FRENCH = "French"


# --------------------------------------------------------------------------- SKIP sentinel


class TestSkipSentinel:
    def test_is_falsy(self) -> None:
        assert not SKIP
        assert bool(SKIP) is False

    def test_repr_and_str(self) -> None:
        assert repr(SKIP) == "SKIP"
        assert str(SKIP) == "SKIP"

    def test_is_a_singleton(self) -> None:
        assert _Skip() is SKIP
        assert type(SKIP)() is SKIP

    def test_copy_and_pickle_preserve_identity(self) -> None:
        assert copy.copy(SKIP) is SKIP
        assert copy.deepcopy(SKIP) is SKIP
        assert pickle.loads(pickle.dumps(SKIP)) is SKIP

    def test_is_distinct_from_other_falsy_values(self) -> None:
        assert SKIP not in (None, False, 0, "", [])


# --------------------------------------------------------------------------- validate_certainty


class TestValidateCertainty:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0),
            (1, 1),
            (100, 100),
            (50.5, 50.5),
            (99.9, 99.9),
            ("80", 80),
            (" 80 ", 80),
            ("12.5", 12.5),
            (decimal.Decimal("70"), 70),
            (decimal.Decimal("12.5"), 12.5),
            (90.0, 90),
            (100.0, 100),
            (0.0, 0),
        ],
    )
    def test_accepts_valid_values(self, value: Any, expected: Any) -> None:
        assert validate_certainty(value) == expected

    def test_integral_float_becomes_int(self) -> None:
        result = validate_certainty(90.0)
        assert result == 90
        assert type(result) is int

    def test_integral_string_becomes_int(self) -> None:
        assert type(validate_certainty("80")) is int

    def test_fractional_string_becomes_float(self) -> None:
        result = validate_certainty("12.5")
        assert result == 12.5
        assert type(result) is float

    def test_decimal_becomes_plain_number(self) -> None:
        assert type(validate_certainty(decimal.Decimal("12.5"))) is float
        assert type(validate_certainty(decimal.Decimal("70"))) is int

    def test_fractional_float_is_kept(self) -> None:
        assert validate_certainty(50.5) == 50.5

    def test_int_is_returned_as_is(self) -> None:
        assert type(validate_certainty(75)) is int

    @pytest.mark.parametrize("value", [True, False])
    def test_rejects_booleans(self, value: bool) -> None:
        with pytest.raises(InvalidFactError, match="boolean"):
            validate_certainty(value)

    @pytest.mark.parametrize(
        "value",
        [
            101,
            -1,
            100.0001,
            -0.5,
            float("nan"),
            float("inf"),
            float("-inf"),
            decimal.Decimal("NaN"),
            "nan",
            "inf",
            "abc",
            "",
            "100%",
            None,
            [50],
            {"certainty": 50},
        ],
    )
    def test_rejects_invalid_values(self, value: Any) -> None:
        with pytest.raises(InvalidFactError):
            validate_certainty(value)

    def test_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            validate_certainty(101)

    def test_error_message_mentions_the_value(self) -> None:
        with pytest.raises(InvalidFactError, match="101"):
            validate_certainty(101)
        with pytest.raises(InvalidFactError, match="'abc'"):
            validate_certainty("abc")
        with pytest.raises(InvalidFactError, match="NoneType"):
            validate_certainty(None)


# --------------------------------------------------------------------------- coerce_value


class TestCoerceValue:
    @pytest.mark.parametrize("value", ["French", "", 5, 0, -3, 2.5, True, False])
    def test_scalars_pass_through_unchanged(self, value: Any) -> None:
        result = coerce_value(value)
        assert result == value
        assert type(result) is type(value)

    def test_date_becomes_iso_string(self) -> None:
        assert coerce_value(dt.date(2024, 3, 1)) == "2024-03-01"

    def test_naive_datetime_becomes_date_only_string(self) -> None:
        assert coerce_value(dt.datetime(2024, 3, 1, 12, 30)) == "2024-03-01"

    def test_datetime_time_of_day_is_dropped(self) -> None:
        assert coerce_value(dt.datetime(2024, 3, 1, 0, 0)) == "2024-03-01"
        assert coerce_value(dt.datetime(2024, 3, 1, 23, 59, 59, 999999)) == "2024-03-01"

    def test_aware_datetime_uses_its_own_calendar_date(self) -> None:
        # The calendar date is taken as written; there is no conversion to UTC or local time.
        late = dt.datetime(2024, 3, 1, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
        early = dt.datetime(2024, 3, 1, 0, 30, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        assert coerce_value(late) == "2024-03-01"
        assert coerce_value(early) == "2024-03-01"
        assert coerce_value(dt.datetime(2024, 3, 1, 12, tzinfo=dt.timezone.utc)) == "2024-03-01"

    def test_datetime_and_date_coerce_to_the_same_string(self) -> None:
        assert coerce_value(dt.datetime(2024, 3, 1, 12, 30)) == coerce_value(dt.date(2024, 3, 1))

    def test_fractional_decimal_becomes_float(self) -> None:
        result = coerce_value(decimal.Decimal("12.50"))
        assert result == 12.5
        assert type(result) is float

    @pytest.mark.parametrize(
        ("raw", "expected"), [("70", 70), ("70.0", 70), ("7E1", 70), ("-3", -3), ("0", 0)]
    )
    def test_integral_decimal_becomes_int(self, raw: str, expected: int) -> None:
        result = coerce_value(decimal.Decimal(raw))
        assert result == expected
        assert type(result) is int

    def test_integral_decimal_is_json_encoded_without_a_fraction(self) -> None:
        assert json.dumps(coerce_value(decimal.Decimal("70.0"))) == "70"
        assert json.dumps(coerce_value(decimal.Decimal("70.5"))) == "70.5"

    def test_plain_enum_uses_its_value(self) -> None:
        result = coerce_value(Language.FRENCH)
        assert result == "French"
        assert type(result) is str

    def test_enum_with_int_value(self) -> None:
        class Level(enum.Enum):
            HIGH = 3

        assert coerce_value(Level.HIGH) == 3

    def test_str_enum_is_json_compatible(self) -> None:
        result = coerce_value(StrLanguage.FRENCH)
        assert result == "French"
        assert isinstance(result, str)
        assert json.dumps(result) == '"French"'

    def test_enum_with_non_scalar_value_is_rejected(self) -> None:
        class Weird(enum.Enum):
            PAIR = (1, 2)

        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            coerce_value(Weird.PAIR)

    @pytest.mark.parametrize("value", [None, {"a": 1}, [1, 2], (1, 2), {1}, b"bytes", object()])
    def test_rejects_unsupported_types(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match=type(value).__name__):
            coerce_value(value)

    def test_class_with_value_attribute_is_not_an_enum_member(self) -> None:
        class HasValue:
            value = "x"

        with pytest.raises(InvalidAnswerError):
            coerce_value(HasValue)

    def test_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            coerce_value(None)


# --------------------------------------------------------------------------- First Form


class TestFirstForm:
    @pytest.mark.parametrize(
        "value", [True, "yes", "Yes", "YES", "y", "Y", "true", "True", "t", 1, "1", 1.0, " yes "]
    )
    def test_yes_values(self, value: Any) -> None:
        [answer] = build_answers(FIRST, value)
        assert answer.answer == "yes"

    @pytest.mark.parametrize(
        "value", [False, "no", "No", "n", "N", "false", "FALSE", "f", 0, "0", 0.0, " no "]
    )
    def test_no_values(self, value: Any) -> None:
        [answer] = build_answers(FIRST, value)
        assert answer.answer == "no"

    def test_payload_is_the_question_triple_plus_answer(self) -> None:
        assert payloads(build_answers(FIRST, True)) == [
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "answer": "yes",
            }
        ]

    def test_payload_has_no_unanswered_key(self) -> None:
        [answer] = build_answers(FIRST, False)
        assert answer.unanswered is False
        assert "unanswered" not in answer.to_payload()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [([True], "yes"), (("no",), "no"), ({"yes"}, "yes"), (frozenset({False}), "no")],
    )
    def test_single_element_collections_are_unwrapped(self, value: Any, expected: str) -> None:
        [answer] = build_answers(FIRST, value)
        assert answer.answer == expected

    def test_single_element_generator_is_unwrapped(self) -> None:
        [answer] = build_answers(FIRST, (v for v in [False]))
        assert answer.answer == "no"

    def test_multiple_values_are_rejected(self) -> None:
        with pytest.raises(InvalidAnswerError, match="exactly one"):
            build_answers(FIRST, [True, False])

    def test_multiple_values_are_rejected_even_when_not_strict(self) -> None:
        with pytest.raises(InvalidAnswerError, match="exactly one"):
            build_answers(FIRST, ["yes", "yes"], strict=False)

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(InvalidAnswerError, match="exactly one"):
            build_answers(FIRST, [])

    @pytest.mark.parametrize("value", ["maybe", "", "yep", 2, -1, 0.5, None, object()])
    def test_non_boolean_values_are_rejected(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="yes/no"):
            build_answers(FIRST, value)

    def test_strict_false_does_not_relax_yes_no(self) -> None:
        with pytest.raises(InvalidAnswerError, match="yes/no"):
            build_answers(FIRST, "maybe", strict=False)

    def test_missing_object_is_rejected(self) -> None:
        with pytest.raises(InvalidAnswerError, match="missing its 'object'"):
            build_answers(question(QUESTION_FIRST_FORM, object=None), True)

    def test_missing_subject_becomes_empty_string(self) -> None:
        [answer] = build_answers(question(QUESTION_FIRST_FORM, subject=None), True)
        assert answer.subject == ""

    def test_object_is_kept_exactly_as_asked(self) -> None:
        [answer] = build_answers(question(QUESTION_FIRST_FORM, object=42), "yes")
        assert answer.object == 42

    def test_subject_is_stringified(self) -> None:
        [answer] = build_answers(question(QUESTION_FIRST_FORM, subject=7), "yes")
        assert answer.subject == "7"

    def test_default_certainty_is_100(self) -> None:
        [answer] = build_answers(FIRST, True)
        assert answer.certainty == 100

    @pytest.mark.parametrize(
        ("certainty", "expected"),
        [(60, 60), ("75", 75), (90.0, 90), (decimal.Decimal("33.5"), 33.5), (0, 0)],
    )
    def test_certainty_is_applied(self, certainty: Any, expected: Any) -> None:
        [answer] = build_answers(FIRST, True, certainty=certainty)
        assert answer.certainty == expected
        assert answer.to_payload()["certainty"] == expected

    @pytest.mark.parametrize("certainty", [101, -1, True, "abc", float("nan"), None])
    def test_invalid_certainty_is_rejected(self, certainty: Any) -> None:
        with pytest.raises(InvalidFactError):
            build_answers(FIRST, True, certainty=certainty)

    def test_type_matching_is_case_insensitive(self) -> None:
        [answer] = build_answers(question(QUESTION_FIRST_FORM, type="first form"), "y")
        assert answer.answer == "yes"

    def test_options_are_not_checked_for_first_form(self) -> None:
        # canAdd is False on the canned question and "yes" is not a concept; still fine.
        assert FIRST.can_add is False
        assert build_answers(FIRST, "yes")[0].answer == "yes"

    def test_session_answer_sends_exact_payload(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_FIRST_FORM)
        session.answer(True, certainty=80)
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Tom",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 80,
                    "answer": "yes",
                }
            ]
        }

    def test_session_answer_no(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_FIRST_FORM)
        session.answer("n")
        assert fake.last.json["answers"][0]["answer"] == "no"


# --------------------------------------------------------------------------- Second Form Object


class TestSecondFormObject:
    def test_value_goes_in_object_and_subject_is_copied(self) -> None:
        assert build_answers(SECOND_OBJECT, "German") == [
            Answer(subject="Germany", relationship="national language", object="German")
        ]

    def test_payload_has_only_the_four_core_keys(self) -> None:
        assert payloads(build_answers(SECOND_OBJECT, "German")) == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "German",
                "certainty": 100,
            }
        ]

    def test_no_answer_or_unanswered_flags(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, "German")
        assert answer.answer is None
        assert answer.unanswered is False

    @pytest.mark.parametrize("value", [42, 3.5, True, False, 0, ""])
    def test_scalars_keep_their_type(self, value: Any) -> None:
        [answer] = build_answers(SECOND_OBJECT, value)
        assert answer.object == value
        assert type(answer.object) is type(value)

    def test_date_is_coerced_to_iso(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, dt.date(2024, 3, 1))
        assert answer.object == "2024-03-01"

    def test_datetime_is_coerced_to_date_only(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, dt.datetime(2024, 3, 1, 9, 5))
        assert answer.object == "2024-03-01"

    def test_fractional_decimal_is_coerced_to_float(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, decimal.Decimal("1.5"))
        assert answer.object == 1.5
        assert type(answer.object) is float

    def test_integral_decimal_is_coerced_to_int(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, decimal.Decimal("2.0"))
        assert answer.object == 2
        assert type(answer.object) is int

    def test_enum_is_coerced_to_its_value(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, Language.GERMAN)
        assert answer.object == "German"
        assert type(answer.object) is str

    @pytest.mark.parametrize("value", [None, {"a": 1}, object()])
    def test_unsupported_values_are_rejected(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            build_answers(SECOND_OBJECT, value)

    def test_missing_subject_becomes_empty_string(self) -> None:
        [answer] = build_answers(question(QUESTION_SECOND_OBJECT, subject=None), "German")
        assert answer.subject == ""

    def test_subject_is_stringified(self) -> None:
        [answer] = build_answers(question(QUESTION_SECOND_OBJECT, subject=7), "German")
        assert answer.subject == "7"

    def test_certainty_is_applied(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, "German", certainty=55.5)
        assert answer.certainty == 55.5

    def test_invalid_certainty_is_rejected_before_building(self) -> None:
        with pytest.raises(InvalidFactError):
            build_answers(SECOND_OBJECT, "German", certainty=150)

    def test_session_answer_sends_exact_payload(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        session.answer("German", certainty=80)
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "German",
                    "certainty": 80,
                }
            ]
        }

    @pytest.mark.anyio
    async def test_async_session_answer_sends_exact_payload(
        self, aclient: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        session = await aclient.start(KM_ID)
        await session.query("national language", subject="Germany")
        await session.answer(dt.date(2024, 3, 1), certainty="90")
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "2024-03-01",
                    "certainty": 90,
                }
            ]
        }


# --------------------------------------------------------------------------- Second Form Subject


class TestSecondFormSubject:
    def test_value_goes_in_subject_and_object_is_copied(self) -> None:
        assert build_answers(SECOND_SUBJECT, "Tom") == [
            Answer(subject="Tom", relationship="speaks", object="French")
        ]

    def test_payload_shape(self) -> None:
        assert payloads(build_answers(SECOND_SUBJECT, "Ana")) == [
            {"subject": "Ana", "relationship": "speaks", "object": "French", "certainty": 100}
        ]

    def test_numeric_subject_is_stringified(self) -> None:
        [answer] = build_answers(question(QUESTION_SECOND_SUBJECT, canAdd=True), 5)
        assert answer.subject == "5"

    def test_date_subject_becomes_iso_string(self) -> None:
        [answer] = build_answers(
            question(QUESTION_SECOND_SUBJECT, canAdd=True), dt.date(2024, 3, 1)
        )
        assert answer.subject == "2024-03-01"

    def test_datetime_subject_becomes_date_only_string(self) -> None:
        [answer] = build_answers(
            question(QUESTION_SECOND_SUBJECT, canAdd=True), dt.datetime(2024, 3, 1, 12, 30)
        )
        assert answer.subject == "2024-03-01"

    def test_integral_decimal_subject_has_no_fraction(self) -> None:
        [answer] = build_answers(
            question(QUESTION_SECOND_SUBJECT, canAdd=True), decimal.Decimal("5.0")
        )
        assert answer.subject == "5"

    def test_enum_subject_uses_its_value(self) -> None:
        class Person(enum.Enum):
            TOM = "Tom"

        [answer] = build_answers(SECOND_SUBJECT, Person.TOM)
        assert answer.subject == "Tom"

    @pytest.mark.parametrize("value", [None, {"a": 1}, object()])
    def test_unsupported_subject_values_are_rejected(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            build_answers(SECOND_SUBJECT, value)

    def test_missing_object_becomes_empty_string(self) -> None:
        [answer] = build_answers(question(QUESTION_SECOND_SUBJECT, object=None), "Tom")
        assert answer.object == ""

    def test_object_is_kept_exactly_as_asked(self) -> None:
        [answer] = build_answers(question(QUESTION_SECOND_SUBJECT, object=42), "Tom")
        assert answer.object == 42

    def test_certainty_is_applied(self) -> None:
        [answer] = build_answers(SECOND_SUBJECT, "Tom", certainty=20)
        assert answer.certainty == 20

    def test_legacy_second_form_type_infers_subject_from_the_missing_side(self) -> None:
        legacy = question(QUESTION_SECOND_SUBJECT, type="Second Form")
        assert legacy.form is QuestionForm.SECOND_SUBJECT
        [answer] = build_answers(legacy, "Tom")
        assert (answer.subject, answer.object) == ("Tom", "French")

    def test_legacy_second_form_type_defaults_to_object(self) -> None:
        legacy = question(QUESTION_SECOND_OBJECT, type="Second Form")
        assert legacy.form is QuestionForm.SECOND_OBJECT
        [answer] = build_answers(legacy, "German")
        assert (answer.subject, answer.object) == ("Germany", "German")

    def test_session_answer_sends_exact_payload(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_SECOND_SUBJECT)
        session.answer("Tom")
        assert fake.last.json == {
            "answers": [
                {"subject": "Tom", "relationship": "speaks", "object": "French", "certainty": 100}
            ]
        }


# --------------------------------------------------------------------------- plural questions


class TestPlural:
    def test_list_produces_one_answer_per_value_in_order(self) -> None:
        answers = build_answers(SECOND_SUBJECT, ["Tom", "Ana"])
        assert [a.subject for a in answers] == ["Tom", "Ana"]
        assert all(a.object == "French" for a in answers)

    def test_tuple_is_accepted(self) -> None:
        answers = build_answers(PLURAL_LOSS, ("theft", "water damage"))
        assert [a.object for a in answers] == ["theft", "water damage"]

    def test_set_is_accepted(self) -> None:
        answers = build_answers(PLURAL_LOSS, {"theft", "water damage"})
        assert len(answers) == 2
        assert {a.object for a in answers} == {"theft", "water damage"}

    def test_frozenset_is_accepted(self) -> None:
        answers = build_answers(PLURAL_LOSS, frozenset({"theft", "cosmetic damage"}))
        assert {a.object for a in answers} == {"theft", "cosmetic damage"}

    def test_generator_is_accepted(self) -> None:
        answers = build_answers(PLURAL_LOSS, (v for v in ["theft", "water damage"]))
        assert [a.object for a in answers] == ["theft", "water damage"]

    def test_single_scalar_on_plural_question(self) -> None:
        [answer] = build_answers(PLURAL_LOSS, "theft")
        assert answer.object == "theft"

    def test_every_answer_repeats_the_known_side(self) -> None:
        answers = build_answers(PLURAL_LOSS, ["theft", "water damage"])
        assert {(a.subject, a.relationship) for a in answers} == {("Ben", "suffered")}

    def test_certainty_is_applied_to_every_answer(self) -> None:
        answers = build_answers(PLURAL_LOSS, ["theft", "water damage"], certainty=40)
        assert [a.certainty for a in answers] == [40, 40]

    def test_two_values_on_non_plural_question_are_rejected_when_strict(self) -> None:
        with pytest.raises(InvalidAnswerError, match="accepts a single answer; got 2 values"):
            build_answers(SECOND_OBJECT, ["English", "French"])

    def test_two_values_on_non_plural_question_are_allowed_when_not_strict(self) -> None:
        answers = build_answers(SECOND_OBJECT, ["English", "French"], strict=False)
        assert [a.object for a in answers] == ["English", "French"]

    def test_single_element_list_on_non_plural_question_is_fine(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, ["German"])
        assert answer.object == "German"

    @pytest.mark.parametrize("empty", [[], (), set(), frozenset(), iter(())])
    def test_empty_collection_is_rejected(self, empty: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="at least one value"):
            build_answers(PLURAL_LOSS, empty)

    def test_empty_collection_is_rejected_even_when_not_strict(self) -> None:
        with pytest.raises(InvalidAnswerError, match="at least one value"):
            build_answers(SECOND_OBJECT, [], strict=False)

    def test_each_value_is_coerced_independently(self) -> None:
        free_text = question(QUESTION_PLURAL_LOSS, canAdd=True)
        values = [
            "theft",
            5,
            dt.date(2024, 1, 2),
            dt.datetime(2024, 1, 3, 9, 30),
            decimal.Decimal("2.0"),
            Loss.THEFT,
        ]
        answers = build_answers(free_text, values)
        assert [a.object for a in answers] == ["theft", 5, "2024-01-02", "2024-01-03", 2, "theft"]

    def test_session_answer_sends_one_answer_per_value(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_PLURAL_LOSS)
        session.answer(["theft", "water damage"])
        assert fake.last.json == {
            "answers": [
                {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 100},
                {
                    "subject": "Ben",
                    "relationship": "suffered",
                    "object": "water damage",
                    "certainty": 100,
                },
            ]
        }


# --------------------------------------------------------------------------- option validation


class TestOptionValidation:
    def test_value_outside_options_is_rejected_when_can_add_is_false(self) -> None:
        with pytest.raises(InvalidAnswerError) as info:
            build_answers(PLURAL_LOSS, "fire")
        message = str(info.value)
        assert "'fire'" in message
        assert "'suffered'" in message
        assert message.endswith("Options: cosmetic damage, theft, water damage")

    def test_value_inside_options_is_accepted(self) -> None:
        [answer] = build_answers(PLURAL_LOSS, "theft")
        assert answer.object == "theft"

    def test_options_are_listed_sorted(self) -> None:
        with pytest.raises(InvalidAnswerError, match=r"Options: Ana, Tom$"):
            build_answers(SECOND_SUBJECT, "Bob")

    def test_can_add_true_accepts_values_outside_options(self) -> None:
        assert SECOND_OBJECT.options == ["English", "French"]
        [answer] = build_answers(SECOND_OBJECT, "German")
        assert answer.object == "German"

    def test_strict_false_accepts_values_outside_options(self) -> None:
        [answer] = build_answers(PLURAL_LOSS, "fire", strict=False)
        assert answer.object == "fire"

    def test_no_concepts_means_free_text(self) -> None:
        free_text = question(QUESTION_PLURAL_LOSS, concepts=[])
        assert free_text.can_add is False
        assert build_answers(free_text, "fire")[0].object == "fire"

    def test_one_invalid_value_in_a_list_rejects_the_whole_call(self) -> None:
        with pytest.raises(InvalidAnswerError, match="'fire'"):
            build_answers(PLURAL_LOSS, ["theft", "fire"])

    def test_invalid_response_concepts_are_not_options(self) -> None:
        q = question(
            QUESTION_PLURAL_LOSS,
            concepts=[
                {"conceptType": "loss", "name": "theft", "type": "string", "value": "theft"},
                {
                    "conceptType": "loss",
                    "name": "Other",
                    "type": "string",
                    "value": "Other",
                    "invalidResponse": True,
                },
            ],
        )
        assert q.options == ["theft"]
        with pytest.raises(InvalidAnswerError, match=r"Options: theft$"):
            build_answers(q, "Other")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            ("5", "5"),  # exact textual match keeps the caller's value
            (5.0, 5),  # numeric-only match is sent as the option's own value
            ("5.0", 5),
            (10, 10),
            ("10.00", 10),
            (decimal.Decimal("10.0"), 10),
            (decimal.Decimal("5"), 5),
        ],
    )
    def test_numbers_are_compared_numerically(self, value: Any, expected: Any) -> None:
        [answer] = build_answers(NUMBER_OPTIONS, value)
        assert answer.object == expected
        assert type(answer.object) is type(expected)

    def test_numeric_match_sends_the_option_value(self) -> None:
        # 5.0 matches the option 5 numerically; the wire value is the concept the engine knows.
        [answer] = build_answers(NUMBER_OPTIONS, 5.0)
        assert answer.to_payload()["object"] == 5
        assert type(answer.to_payload()["object"]) is int

    @pytest.mark.parametrize("value", [7, "7", 5.5, "5.5", 50, 0.5, "five", "", "5x"])
    def test_numbers_outside_options_are_rejected(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match=r"Options: 10, 5$"):
            build_answers(NUMBER_OPTIONS, value)

    @pytest.mark.parametrize("value", [5, 5.0, decimal.Decimal("5")])
    def test_string_options_match_a_numeric_answer(self, value: Any) -> None:
        q = question(
            QUESTION_SECOND_OBJECT,
            canAdd=False,
            concepts=[{"conceptType": "n", "name": "5", "type": "string", "value": "5"}],
        )
        [answer] = build_answers(q, value)
        # An exact textual match keeps the caller's value; a numeric-only match is sent
        # as the option's own spelling so the engine receives its known concept instance.
        assert str(answer.object) == "5"

    @pytest.mark.parametrize("value", ["5.0", "05", "5e0"])
    def test_string_question_does_not_match_numeric_spellings_of_a_string_option(
        self, value: str
    ) -> None:
        # Numeric matching only applies to numeric answers or number-typed questions; on a
        # string question the concept instance "5" is only matched by the exact text "5".
        q = question(
            QUESTION_SECOND_OBJECT,
            canAdd=False,
            concepts=[{"conceptType": "n", "name": "5", "type": "string", "value": "5"}],
        )
        with pytest.raises(InvalidAnswerError, match="Options: 5$"):
            build_answers(q, value)

    def test_numeric_options_match_a_string_answer(self) -> None:
        [answer] = build_answers(NUMBER_OPTIONS, "10.0")
        assert answer.object == 10  # canonicalised to the option's own value

    def test_non_numeric_options_use_string_equality(self) -> None:
        q = question(
            QUESTION_SECOND_OBJECT,
            canAdd=False,
            concepts=[{"conceptType": "n", "name": "ten", "type": "string", "value": "ten"}],
        )
        assert build_answers(q, "ten")[0].object == "ten"
        with pytest.raises(InvalidAnswerError, match=r"Options: ten$"):
            build_answers(q, 10)
        with pytest.raises(InvalidAnswerError, match=r"Options: ten$"):
            build_answers(q, "Ten")

    def test_numeric_option_check_applies_to_subject_questions(self) -> None:
        q = question(
            QUESTION_SECOND_SUBJECT,
            concepts=[{"conceptType": "n", "name": "5", "type": "number", "value": 5}],
        )
        assert q.can_add is False
        [answer] = build_answers(q, 5.0)
        assert answer.subject == "5"  # option's value, as text (subjects are strings)
        with pytest.raises(InvalidAnswerError, match=r"Options: 5$"):
            build_answers(q, 6)

    def test_option_check_applies_to_subject_questions(self) -> None:
        with pytest.raises(InvalidAnswerError, match="'Bob'"):
            build_answers(SECOND_SUBJECT, "Bob")
        [answer] = build_answers(SECOND_SUBJECT, "Ana")
        assert answer.subject == "Ana"

    def test_option_check_uses_the_coerced_value(self) -> None:
        [answer] = build_answers(PLURAL_LOSS, Loss.THEFT)
        assert answer.object == "theft"

    def test_option_check_is_case_sensitive(self) -> None:
        with pytest.raises(InvalidAnswerError):
            build_answers(PLURAL_LOSS, "Theft")

    def test_validation_happens_before_any_request(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_PLURAL_LOSS)
        with pytest.raises(InvalidAnswerError):
            session.answer("fire")
        assert fake.calls("/response") == []
        assert session.pending_question is not None

    def test_session_answer_strict_false_sends_unknown_value(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_PLURAL_LOSS)
        session.answer("fire", strict=False)
        assert fake.last.json["answers"][0]["object"] == "fire"


# --------------------------------------------------------------------------- truth dataType


class TestTruthNormalisation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("yes", True),
            ("Yes", True),
            (" true ", True),
            ("t", True),
            ("1", True),
            (1, True),
            (True, True),
            ("no", False),
            ("false", False),
            ("FALSE", False),
            ("n", False),
            ("0", False),
            (0, False),
            (False, False),
        ],
    )
    def test_values_become_real_booleans(self, value: Any, expected: bool) -> None:
        [answer] = build_answers(TRUTH, value)
        assert answer.object is expected

    def test_payload_carries_a_json_boolean(self) -> None:
        [answer] = build_answers(TRUTH, "yes")
        assert answer.to_payload()["object"] is True
        assert json.dumps(answer.to_payload()["object"]) == "true"

    @pytest.mark.parametrize("value", ["maybe", "", "yep", 2, None])
    def test_invalid_truth_values_are_rejected(self, value: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="expects a truth value"):
            build_answers(TRUTH, value)

    def test_error_names_the_relationship_and_value(self) -> None:
        with pytest.raises(InvalidAnswerError, match=r"'is tall'.*'maybe'"):
            build_answers(TRUTH, "maybe")

    def test_strict_false_does_not_relax_truth_parsing(self) -> None:
        with pytest.raises(InvalidAnswerError):
            build_answers(TRUTH, "maybe", strict=False)

    def test_data_type_match_is_case_insensitive(self) -> None:
        q = question(TRUTH_RAW, dataType="Truth")
        assert build_answers(q, "yes")[0].object is True

    def test_non_truth_questions_keep_strings(self) -> None:
        [answer] = build_answers(SECOND_OBJECT, "yes")
        assert answer.object == "yes"

    def test_second_form_subject_truth_question_sends_plain_strings(self) -> None:
        # dataType describes the object side; a subject answer is never a boolean.
        q = question(TRUTH_RAW, type="Second Form Subject", subject=None, object=True)
        assert q.form is QuestionForm.SECOND_SUBJECT
        assert q.is_truth is True
        [answer] = build_answers(q, "Tom")
        assert answer.subject == "Tom"
        assert answer.object is True

    @pytest.mark.parametrize("value", ["yes", "true", "1", "maybe"])
    def test_second_form_subject_truth_question_does_not_parse_truth_values(
        self, value: str
    ) -> None:
        q = question(TRUTH_RAW, type="Second Form Subject", subject=None, object=True)
        [answer] = build_answers(q, value)
        assert answer.subject == value
        assert type(answer.subject) is str

    def test_first_form_ignores_truth_data_type(self) -> None:
        q = question(QUESTION_FIRST_FORM, dataType="truth", object=True)
        [answer] = build_answers(q, "yes")
        assert answer.object is True
        assert answer.answer == "yes"

    def test_plural_truth_values(self) -> None:
        q = question(TRUTH_RAW, plural=True)
        answers = build_answers(q, ["yes", "no"])
        assert [a.object for a in answers] == [True, False]

    def test_truth_with_boolean_options_passes_the_option_check(self) -> None:
        q = question(
            TRUTH_RAW,
            canAdd=False,
            concepts=[
                {"conceptType": "t", "name": "true", "type": "truth", "value": True},
                {"conceptType": "t", "name": "false", "type": "truth", "value": False},
            ],
        )
        assert build_answers(q, "yes")[0].object is True
        assert build_answers(q, "no")[0].object is False

    def test_truth_with_string_options_compares_normalised_booleans(self) -> None:
        q = question(
            TRUTH_RAW,
            canAdd=False,
            concepts=[
                {"conceptType": "t", "name": "true", "type": "truth", "value": "true"},
                {"conceptType": "t", "name": "false", "type": "truth", "value": "false"},
            ],
        )
        assert build_answers(q, "yes")[0].object is True
        assert build_answers(q, 0)[0].object is False
        assert build_answers(q, True)[0].object is True

    def test_truth_value_outside_options_is_rejected(self) -> None:
        q = question(
            TRUTH_RAW,
            canAdd=False,
            concepts=[{"conceptType": "t", "name": "true", "type": "truth", "value": True}],
        )
        assert build_answers(q, "yes")[0].object is True
        with pytest.raises(
            InvalidAnswerError, match=r"not one of the allowed answers.*Options: True$"
        ):
            build_answers(q, "no")

    def test_truth_value_outside_options_is_allowed_when_not_strict(self) -> None:
        q = question(
            TRUTH_RAW,
            canAdd=False,
            concepts=[{"conceptType": "t", "name": "true", "type": "truth", "value": True}],
        )
        assert build_answers(q, "no", strict=False)[0].object is False

    def test_truth_with_non_truth_options_is_left_to_the_api(self) -> None:
        q = question(
            TRUTH_RAW,
            canAdd=False,
            concepts=[{"conceptType": "t", "name": "maybe", "type": "string", "value": "maybe"}],
        )
        assert build_answers(q, "yes")[0].object is True
        assert build_answers(q, "no")[0].object is False

    def test_truth_subject_question_validates_subjects_as_strings(self) -> None:
        # dataType "truth" describes the object side; the options here are subjects, so the
        # strict check uses plain string matching rather than truth normalisation.
        q = question(
            TRUTH_RAW,
            type="Second Form Subject",
            subject=None,
            object=True,
            canAdd=False,
            concepts=[{"conceptType": "person", "name": "Tom", "type": "string", "value": "Tom"}],
        )
        assert build_answers(q, "Tom")[0].subject == "Tom"
        with pytest.raises(InvalidAnswerError, match="'Ana' is not one of the allowed answers"):
            build_answers(q, "Ana")
        assert build_answers(q, "Ana", strict=False)[0].subject == "Ana"


# --------------------------------------------------------------------------- unknown form


class TestUnknownForm:
    @pytest.mark.parametrize("kind", ["Third Form", "", "question", "Form"])
    def test_unknown_type_cannot_be_answered_automatically(self, kind: str) -> None:
        q = question(QUESTION_SECOND_OBJECT, type=kind)
        assert q.form is QuestionForm.UNKNOWN
        with pytest.raises(InvalidAnswerError, match="unsupported question type"):
            build_answers(q, "German")

    def test_error_names_the_type_and_points_to_respond(self) -> None:
        q = question(QUESTION_SECOND_OBJECT, type="Third Form")
        with pytest.raises(InvalidAnswerError, match=r"'Third Form'.*respond\(\)"):
            build_answers(q, "German")

    def test_unknown_form_rejects_even_when_not_strict(self) -> None:
        q = question(QUESTION_SECOND_OBJECT, type="Third Form")
        with pytest.raises(InvalidAnswerError, match="unsupported question type"):
            build_answers(q, "German", strict=False)

    def test_unknown_form_can_still_be_skipped(self) -> None:
        q = question(QUESTION_SECOND_OBJECT, type="Third Form", allowUnknown=True)
        assert payloads(build_skip(q)) == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        ]


# --------------------------------------------------------------------------- skipping


class TestSkip:
    def test_second_form_object_payload_blanks_the_object(self) -> None:
        assert payloads(build_skip(SKIPPABLE_OBJECT)) == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    def test_second_form_subject_payload_blanks_the_subject(self) -> None:
        assert payloads(build_skip(SECOND_SUBJECT)) == [
            {
                "subject": "",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    def test_first_form_payload_keeps_the_full_triple(self) -> None:
        assert payloads(build_skip(FIRST)) == [
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    @pytest.mark.parametrize("q", [FIRST, SECOND_SUBJECT, SKIPPABLE_OBJECT])
    def test_skip_payload_never_has_an_answer_key(self, q: Question) -> None:
        [answer] = build_skip(q)
        assert answer.answer is None
        assert answer.unanswered is True
        assert "answer" not in answer.to_payload()

    @pytest.mark.parametrize("q", [FIRST, SECOND_SUBJECT, SKIPPABLE_OBJECT])
    def test_build_answers_with_skip_equals_build_skip(self, q: Question) -> None:
        assert build_answers(q, SKIP) == build_skip(q)

    def test_skip_certainty_is_always_100(self) -> None:
        [answer] = build_answers(FIRST, SKIP, certainty=30)
        assert answer.certainty == 100

    @pytest.mark.parametrize("q", [SECOND_OBJECT, PLURAL_LOSS])
    def test_skip_is_rejected_when_allow_unknown_is_false(self, q: Question) -> None:
        assert q.allow_unknown is False
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            build_skip(q)
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            build_answers(q, SKIP)

    def test_skip_error_names_the_relationship(self) -> None:
        with pytest.raises(InvalidAnswerError, match="'national language'"):
            build_skip(SECOND_OBJECT)

    def test_skip_is_allowed_when_not_strict(self) -> None:
        expected = [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        ]
        assert payloads(build_skip(SECOND_OBJECT, strict=False)) == expected
        assert payloads(build_answers(SECOND_OBJECT, SKIP, strict=False)) == expected

    def test_first_form_skip_with_allow_unknown_false_when_not_strict(self) -> None:
        q = question(QUESTION_FIRST_FORM, allowUnknown=False)
        [answer] = build_skip(q, strict=False)
        assert (answer.object, answer.unanswered) == ("French", True)

    def test_subject_skip_with_missing_object(self) -> None:
        [answer] = build_skip(question(QUESTION_SECOND_SUBJECT, object=None))
        assert answer.object == ""
        assert answer.subject == ""

    def test_object_skip_with_missing_subject(self) -> None:
        [answer] = build_skip(question(QUESTION_SECOND_OBJECT, subject=None, allowUnknown=True))
        assert answer.subject == ""
        assert answer.object == ""

    def test_skip_is_a_single_answer_even_for_plural_questions(self) -> None:
        assert SECOND_SUBJECT.plural is True
        assert len(build_skip(SECOND_SUBJECT)) == 1

    def test_skip_survives_normalize_answers(self) -> None:
        assert normalize_answers(build_skip(SECOND_SUBJECT)) == payloads(build_skip(SECOND_SUBJECT))

    def test_session_skip_sends_exact_payload(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_SECOND_SUBJECT)
        session.skip()
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 100,
                    "unanswered": True,
                }
            ]
        }

    def test_session_answer_with_skip_sentinel(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = ask(client, fake, QUESTION_FIRST_FORM)
        session.answer(SKIP)
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Tom",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 100,
                    "unanswered": True,
                }
            ]
        }

    def test_session_skip_is_rejected_locally_when_not_allowed(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            session.skip()
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            session.answer(SKIP)
        assert fake.calls("/response") == []

    def test_session_skip_strict_false_sends_anyway(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        session.skip(strict=False)
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "",
                    "certainty": 100,
                    "unanswered": True,
                }
            ]
        }

    def test_run_answerer_returning_skip(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        session = client.start(KM_ID)
        result = session.run("speaks", object="French", answerer=lambda _q: SKIP)
        assert result.is_result
        [call] = fake.calls("/response")
        assert call.json["answers"] == [
            {
                "subject": "",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]


# --------------------------------------------------------------------------- normalize_answers

ANSWER = Answer("Germany", "national language", "German")
ANSWER_PAYLOAD = {
    "subject": "Germany",
    "relationship": "national language",
    "object": "German",
    "certainty": 100,
}
ANSWER_DICT = {"subject": "Germany", "relationship": "national language", "object": "German"}


class TestNormalizeAnswers:
    def test_single_answer(self) -> None:
        assert normalize_answers(ANSWER) == [ANSWER_PAYLOAD]

    def test_list_of_answers_preserves_order(self) -> None:
        second = Answer("Germany", "national language", "English", certainty=20)
        assert normalize_answers([ANSWER, second]) == [
            ANSWER_PAYLOAD,
            {**ANSWER_PAYLOAD, "object": "English", "certainty": 20},
        ]

    def test_tuple_of_answers(self) -> None:
        assert normalize_answers((ANSWER, ANSWER)) == [ANSWER_PAYLOAD, ANSWER_PAYLOAD]

    def test_generator_of_answers(self) -> None:
        assert normalize_answers(a for a in [ANSWER]) == [ANSWER_PAYLOAD]

    def test_single_dict(self) -> None:
        assert normalize_answers(ANSWER_DICT) == [ANSWER_PAYLOAD]

    def test_list_of_dicts(self) -> None:
        assert normalize_answers([ANSWER_DICT, ANSWER_DICT]) == [ANSWER_PAYLOAD, ANSWER_PAYLOAD]

    def test_mixed_answers_and_dicts(self) -> None:
        assert normalize_answers([ANSWER, ANSWER_DICT]) == [ANSWER_PAYLOAD, ANSWER_PAYLOAD]

    def test_dict_default_certainty_is_100(self) -> None:
        assert normalize_answers(ANSWER_DICT)[0]["certainty"] == 100

    def test_cf_alias_is_mapped_to_certainty(self) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "cf": 60})
        assert payload["certainty"] == 60
        assert "cf" not in payload

    def test_certainty_wins_over_cf(self) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "certainty": 70, "cf": 60})
        assert payload["certainty"] == 70

    def test_certainty_none_defaults_to_100(self) -> None:
        assert normalize_answers({**ANSWER_DICT, "certainty": None})[0]["certainty"] == 100

    def test_cf_none_defaults_to_100(self) -> None:
        assert normalize_answers({**ANSWER_DICT, "cf": None})[0]["certainty"] == 100

    def test_dict_certainty_is_normalised(self) -> None:
        assert type(normalize_answers({**ANSWER_DICT, "certainty": "80"})[0]["certainty"]) is int
        assert normalize_answers({**ANSWER_DICT, "certainty": "80"})[0]["certainty"] == 80
        assert type(normalize_answers({**ANSWER_DICT, "certainty": 90.0})[0]["certainty"]) is int

    def test_dict_invalid_certainty_is_rejected(self) -> None:
        with pytest.raises(InvalidFactError):
            normalize_answers({**ANSWER_DICT, "certainty": 150})
        with pytest.raises(InvalidFactError):
            normalize_answers({**ANSWER_DICT, "cf": "abc"})

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("yes", "yes"),
            ("Y", "yes"),
            ("true", "yes"),
            (True, "yes"),
            (1, "yes"),
            ("no", "no"),
            ("N", "no"),
            ("false", "no"),
            (False, "no"),
            (0, "no"),
        ],
    )
    def test_answer_field_is_normalised_to_yes_no(self, raw: Any, expected: str) -> None:
        assert normalize_answers({**ANSWER_DICT, "answer": raw})[0]["answer"] == expected

    def test_answer_none_is_omitted(self) -> None:
        assert "answer" not in normalize_answers({**ANSWER_DICT, "answer": None})[0]

    def test_invalid_answer_field_is_rejected(self) -> None:
        with pytest.raises((InvalidFactError, InvalidAnswerError), match="yes/no"):
            normalize_answers({**ANSWER_DICT, "answer": "maybe"})

    def test_unanswered_true_is_kept(self) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "object": "", "unanswered": True})
        assert payload["unanswered"] is True
        assert payload["object"] == ""

    def test_unanswered_false_is_omitted(self) -> None:
        assert "unanswered" not in normalize_answers({**ANSWER_DICT, "unanswered": False})[0]

    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", "yes", "Y", "1", " t "])
    def test_unanswered_true_like_strings_become_true(self, raw: str) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "unanswered": raw})
        assert payload["unanswered"] is True

    @pytest.mark.parametrize("raw", ["false", "False", "no", "N", "0", "f"])
    def test_unanswered_false_like_strings_are_omitted(self, raw: str) -> None:
        assert "unanswered" not in normalize_answers({**ANSWER_DICT, "unanswered": raw})[0]

    def test_unanswered_none_is_omitted(self) -> None:
        assert "unanswered" not in normalize_answers({**ANSWER_DICT, "unanswered": None})[0]

    @pytest.mark.parametrize("raw", [1, 0, 1.0, 0.0, 2, -1])
    def test_unanswered_numbers_are_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidFactError, match="unanswered must be a boolean"):
            normalize_answers({**ANSWER_DICT, "unanswered": raw})

    @pytest.mark.parametrize("raw", ["maybe", "", "yep", [True], {"unanswered": True}])
    def test_unanswered_other_values_are_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidFactError, match="unanswered must be a boolean"):
            normalize_answers({**ANSWER_DICT, "unanswered": raw})

    def test_unanswered_error_mentions_the_value(self) -> None:
        with pytest.raises(InvalidFactError, match="got 'maybe'"):
            normalize_answers({**ANSWER_DICT, "unanswered": "maybe"})
        with pytest.raises(InvalidFactError, match="got 1"):
            normalize_answers({**ANSWER_DICT, "unanswered": 1})

    def test_subject_and_relationship_are_stringified(self) -> None:
        [payload] = normalize_answers({"subject": 5, "relationship": 7, "object": "x"})
        assert (payload["subject"], payload["relationship"]) == ("5", "7")

    @pytest.mark.parametrize("value", [5, 2.5, True, "German", ""])
    def test_object_keeps_its_type(self, value: Any) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "object": value})
        assert payload["object"] == value
        assert type(payload["object"]) is type(value)

    @pytest.mark.parametrize(
        ("data", "missing"),
        [
            ({"relationship": "r", "object": "o"}, "subject"),
            ({"subject": "s", "object": "o"}, "relationship"),
            ({"subject": "s", "relationship": "r"}, "object"),
            ({"subject": "s"}, "relationship, object"),
            ({}, "subject, relationship, object"),
        ],
    )
    def test_missing_fields_are_rejected(self, data: dict[str, Any], missing: str) -> None:
        with pytest.raises(InvalidFactError, match=rf"missing required field\(s\): {missing}$"):
            normalize_answers(data)

    def test_unexpected_fields_are_rejected_sorted(self) -> None:
        with pytest.raises(InvalidFactError, match=r"unexpected field\(s\): bar, foo$"):
            normalize_answers({**ANSWER_DICT, "foo": 1, "bar": 2})

    def test_unexpected_fields_are_reported_before_missing_ones(self) -> None:
        with pytest.raises(InvalidFactError, match="unexpected"):
            normalize_answers({"foo": 1})

    @pytest.mark.parametrize("empty", [[], (), iter(())])
    def test_empty_is_rejected(self, empty: Any) -> None:
        with pytest.raises(InvalidFactError, match="must not be empty"):
            normalize_answers(empty)

    @pytest.mark.parametrize("bad", ["German", 42, None, 3.5, object()])
    def test_wrong_type_is_rejected(self, bad: Any) -> None:
        with pytest.raises(InvalidFactError, match="answers must be"):
            normalize_answers(bad)

    def test_wrong_item_type_is_rejected(self) -> None:
        with pytest.raises(InvalidFactError, match="each answer must be .* got int"):
            normalize_answers([ANSWER, 42])

    def test_tuple_items_are_not_answers(self) -> None:
        with pytest.raises(InvalidFactError, match="each answer must be"):
            normalize_answers([("Germany", "national language", "German")])

    @pytest.mark.parametrize("certainty", [150, -1, True, "abc"])
    def test_answer_object_certainty_is_validated(self, certainty: Any) -> None:
        with pytest.raises(InvalidFactError):
            normalize_answers(Answer("s", "r", "o", certainty=certainty))

    def test_answer_object_flags_are_carried_through(self) -> None:
        assert normalize_answers(Answer("s", "r", "o", answer="yes")) == [
            {"subject": "s", "relationship": "r", "object": "o", "certainty": 100, "answer": "yes"}
        ]
        assert normalize_answers(Answer("s", "r", "", unanswered=True)) == [
            {
                "subject": "s",
                "relationship": "r",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    def test_input_dict_is_not_mutated(self) -> None:
        data = {**ANSWER_DICT, "cf": 60, "answer": "Y"}
        snapshot = dict(data)
        normalize_answers(data)
        assert data == snapshot

    def test_dict_object_is_coerced_like_facts(self) -> None:
        [payload] = normalize_answers(
            {"subject": "Tom", "relationship": "born on", "object": dt.date(2000, 1, 1)}
        )
        assert payload["object"] == "2000-01-01"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (dt.datetime(2000, 1, 1, 8, 30), "2000-01-01"),
            (decimal.Decimal("1.5"), 1.5),
            (decimal.Decimal("2"), 2),
            (Language.GERMAN, "German"),
        ],
    )
    def test_dict_object_values_are_coerced(self, raw: Any, expected: Any) -> None:
        [payload] = normalize_answers({**ANSWER_DICT, "object": raw})
        assert payload["object"] == expected
        assert type(payload["object"]) is type(expected)

    @pytest.mark.parametrize("raw", [None, {"a": 1}, [1, 2], object()])
    def test_dict_unsupported_object_is_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            normalize_answers({**ANSWER_DICT, "object": raw})

    def test_dict_payload_is_json_serialisable(self) -> None:
        payload = normalize_answers(
            {
                **ANSWER_DICT,
                "object": dt.datetime(2000, 1, 1, 8, 30),
                "certainty": decimal.Decimal("90"),
                "unanswered": "false",
            }
        )
        assert json.loads(json.dumps(payload)) == payload

    # -- Answer objects are normalised exactly like dicts

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (dt.date(2000, 1, 1), "2000-01-01"),
            (dt.datetime(2000, 1, 1, 8, 30), "2000-01-01"),
            (decimal.Decimal("1.5"), 1.5),
            (decimal.Decimal("2"), 2),
            (Language.GERMAN, "German"),
        ],
    )
    def test_answer_object_value_is_coerced(self, raw: Any, expected: Any) -> None:
        [payload] = normalize_answers(Answer("Tom", "born on", raw))
        assert payload["object"] == expected
        assert type(payload["object"]) is type(expected)

    @pytest.mark.parametrize("raw", [None, {"a": 1}, [1, 2], object()])
    def test_answer_object_unsupported_value_is_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            normalize_answers(Answer("s", "r", raw))

    def test_answer_object_subject_and_relationship_are_stringified(self) -> None:
        [payload] = normalize_answers(Answer(5, 7, "o"))  # type: ignore[arg-type]
        assert (payload["subject"], payload["relationship"]) == ("5", "7")

    def test_answer_object_certainty_is_normalised(self) -> None:
        [payload] = normalize_answers(Answer("s", "r", "o", certainty="80"))  # type: ignore[arg-type]
        assert payload["certainty"] == 80
        assert type(payload["certainty"]) is int
        assert type(normalize_answers(Answer("s", "r", "o", certainty=90.0))[0]["certainty"]) is int

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("yes", "yes"), ("Y", "yes"), (True, "yes"), (1, "yes"), ("false", "no"), (0, "no")],
    )
    def test_answer_object_answer_field_is_normalised(self, raw: Any, expected: str) -> None:
        assert normalize_answers(Answer("s", "r", "o", answer=raw))[0]["answer"] == expected

    def test_answer_object_invalid_answer_field_is_rejected(self) -> None:
        with pytest.raises(InvalidAnswerError, match="yes/no"):
            normalize_answers(Answer("s", "r", "o", answer="maybe"))

    @pytest.mark.parametrize("raw", [True, "true", "yes", "1"])
    def test_answer_object_unanswered_true_like_values(self, raw: Any) -> None:
        [payload] = normalize_answers(Answer("s", "r", "", unanswered=raw))
        assert payload["unanswered"] is True

    @pytest.mark.parametrize("raw", [False, None, "false", "no", "0"])
    def test_answer_object_unanswered_false_like_values_are_omitted(self, raw: Any) -> None:
        assert "unanswered" not in normalize_answers(Answer("s", "r", "o", unanswered=raw))[0]

    @pytest.mark.parametrize("raw", [1, 0, 1.0, "maybe", ""])
    def test_answer_object_invalid_unanswered_is_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidFactError, match="unanswered must be a boolean"):
            normalize_answers(Answer("s", "r", "o", unanswered=raw))

    def test_answer_object_is_not_mutated(self) -> None:
        answer = Answer("s", "r", dt.date(2000, 1, 1), certainty=90.0, answer="Y")
        normalize_answers(answer)
        assert answer.object == dt.date(2000, 1, 1)
        assert answer.certainty == 90.0
        assert answer.answer == "Y"

    def test_answer_object_and_dict_produce_identical_payloads(self) -> None:
        date = dt.datetime(2000, 1, 1, 8, 30)
        from_object = normalize_answers(Answer("Tom", "born on", date, certainty="70", answer=1))  # type: ignore[arg-type]
        from_dict = normalize_answers(
            {"subject": "Tom", "relationship": "born on", "object": date, "cf": "70", "answer": 1}
        )
        assert from_object == from_dict
        assert from_object == [
            {
                "subject": "Tom",
                "relationship": "born on",
                "object": "2000-01-01",
                "certainty": 70,
                "answer": "yes",
            }
        ]

    def test_session_respond_with_dict_sends_exact_payload(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        session.respond({**ANSWER_DICT, "cf": 70})
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {"answers": [{**ANSWER_PAYLOAD, "certainty": 70}]}

    def test_session_respond_with_answer_objects(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_PLURAL_LOSS)
        session.respond(
            [Answer("Ben", "suffered", "theft"), Answer("Ben", "suffered", "water damage", 50)]
        )
        assert fake.last.json == {
            "answers": [
                {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 100},
                {
                    "subject": "Ben",
                    "relationship": "suffered",
                    "object": "water damage",
                    "certainty": 50,
                },
            ]
        }

    def test_session_respond_rejects_bad_payload_locally(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        with pytest.raises(InvalidFactError):
            session.respond({"subject": "Germany"})
        with pytest.raises(InvalidFactError):
            session.respond({**ANSWER_DICT, "unanswered": 1})
        with pytest.raises(InvalidAnswerError):
            session.respond(Answer("Germany", "national language", None))
        assert fake.calls("/response") == []

    def test_session_respond_normalises_answer_objects_before_sending(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        session.respond(
            Answer("Germany", "national language", Language.GERMAN, certainty="70")  # type: ignore[arg-type]
        )
        assert fake.last.json == {"answers": [{**ANSWER_PAYLOAD, "certainty": 70}]}

    def test_session_respond_normalises_dicts_before_sending(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = ask(client, fake, QUESTION_SECOND_OBJECT)
        session.respond(
            {
                **ANSWER_DICT,
                "object": dt.datetime(2024, 3, 1, 12, 0),
                "certainty": decimal.Decimal("80"),
                "unanswered": "false",
            }
        )
        assert fake.last.json == {
            "answers": [{**ANSWER_PAYLOAD, "object": "2024-03-01", "certainty": 80}]
        }


# --------------------------------------------------------------------------- normalize_facts

FACT = Fact("Tom", "lives in", "Germany")
FACT_PAYLOAD = {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
FACT_DICT = {"subject": "Tom", "relationship": "lives in", "object": "Germany"}


class TestNormalizeFacts:
    def test_single_fact(self) -> None:
        assert normalize_facts(FACT) == [FACT_PAYLOAD]

    def test_fact_with_certainty(self) -> None:
        assert normalize_facts(Fact("Tom", "lives in", "Germany", 60))[0]["certainty"] == 60

    @pytest.mark.parametrize("certainty", [150, -1, True, "abc"])
    def test_fact_certainty_is_validated(self, certainty: Any) -> None:
        with pytest.raises(InvalidFactError):
            normalize_facts(Fact("Tom", "lives in", "Germany", certainty))

    def test_fact_certainty_is_normalised(self) -> None:
        [payload] = normalize_facts(Fact("Tom", "lives in", "Germany", 90.0))
        assert payload["certainty"] == 90
        assert type(payload["certainty"]) is int

    def test_single_dict(self) -> None:
        assert normalize_facts(FACT_DICT) == [FACT_PAYLOAD]

    def test_dict_cf_alias(self) -> None:
        [payload] = normalize_facts({**FACT_DICT, "cf": 60})
        assert payload["certainty"] == 60
        assert "cf" not in payload

    def test_dict_certainty_wins_over_cf(self) -> None:
        assert normalize_facts({**FACT_DICT, "certainty": 70, "cf": 60})[0]["certainty"] == 70

    def test_dict_certainty_none_defaults_to_100(self) -> None:
        assert normalize_facts({**FACT_DICT, "certainty": None})[0]["certainty"] == 100

    def test_dict_certainty_string_is_normalised(self) -> None:
        [payload] = normalize_facts({**FACT_DICT, "certainty": "80"})
        assert payload["certainty"] == 80
        assert type(payload["certainty"]) is int

    def test_dict_object_is_coerced(self) -> None:
        assert (
            normalize_facts({**FACT_DICT, "object": dt.date(2000, 1, 1)})[0]["object"]
            == "2000-01-01"
        )
        assert normalize_facts({**FACT_DICT, "object": decimal.Decimal("1.5")})[0]["object"] == 1.5
        assert normalize_facts({**FACT_DICT, "object": Country.DE})[0]["object"] == "Germany"

    @pytest.mark.parametrize("value", [42, 2.5, True, "Germany"])
    def test_dict_scalar_object_keeps_its_type(self, value: Any) -> None:
        [payload] = normalize_facts({**FACT_DICT, "object": value})
        assert payload["object"] == value
        assert type(payload["object"]) is type(value)

    def test_dict_subject_and_relationship_are_stringified(self) -> None:
        [payload] = normalize_facts({"subject": 5, "relationship": 7, "object": "x"})
        assert (payload["subject"], payload["relationship"]) == ("5", "7")

    def test_dict_none_object_is_rejected(self) -> None:
        with pytest.raises((InvalidFactError, InvalidAnswerError)):
            normalize_facts({**FACT_DICT, "object": None})

    @pytest.mark.parametrize(
        ("data", "missing"),
        [
            ({"relationship": "r", "object": "o"}, "subject"),
            ({"subject": "s", "object": "o"}, "relationship"),
            ({"subject": "s", "relationship": "r"}, "object"),
            ({}, "subject, relationship, object"),
        ],
    )
    def test_dict_missing_fields_are_rejected(self, data: dict[str, Any], missing: str) -> None:
        with pytest.raises(
            InvalidFactError, match=rf"fact is missing required field\(s\): {missing}$"
        ):
            normalize_facts(data)

    def test_dict_unexpected_fields_are_rejected_sorted(self) -> None:
        with pytest.raises(InvalidFactError, match=r"fact has unexpected field\(s\): bar, foo$"):
            normalize_facts({**FACT_DICT, "foo": 1, "bar": 2})

    @pytest.mark.parametrize("key", ["answer", "unanswered"])
    def test_answer_only_keys_are_not_valid_for_facts(self, key: str) -> None:
        with pytest.raises(InvalidFactError, match=f"unexpected field\\(s\\): {key}$"):
            normalize_facts({**FACT_DICT, key: True})

    def test_triple_tuple(self) -> None:
        assert normalize_facts(("Tom", "lives in", "Germany")) == [FACT_PAYLOAD]

    def test_quad_tuple(self) -> None:
        assert normalize_facts(("Tom", "lives in", "Germany", 60)) == [
            {**FACT_PAYLOAD, "certainty": 60}
        ]

    def test_list_triple_is_a_single_fact(self) -> None:
        assert normalize_facts(["Tom", "lives in", "Germany"]) == [FACT_PAYLOAD]

    def test_tuple_with_numeric_object(self) -> None:
        [payload] = normalize_facts(("Tom", "age", 42))
        assert payload["object"] == 42
        assert type(payload["object"]) is int

    def test_tuple_with_bool_object(self) -> None:
        assert normalize_facts(("Tom", "is tall", True))[0]["object"] is True

    def test_tuple_with_date_object(self) -> None:
        assert normalize_facts(("Tom", "born on", dt.date(2000, 1, 1)))[0]["object"] == "2000-01-01"

    def test_tuple_with_datetime_object_is_date_only(self) -> None:
        [payload] = normalize_facts(("Tom", "joined", dt.datetime(2000, 1, 1, 8, 0)))
        assert payload["object"] == "2000-01-01"

    def test_tuple_with_integral_decimal_object(self) -> None:
        [payload] = normalize_facts(("Tom", "age", decimal.Decimal("42.0")))
        assert payload["object"] == 42
        assert type(payload["object"]) is int

    def test_tuple_with_decimal_object_and_certainty(self) -> None:
        [payload] = normalize_facts(
            ("Tom", "height", decimal.Decimal("1.5"), decimal.Decimal("50"))
        )
        assert payload["object"] == 1.5
        assert payload["certainty"] == 50

    def test_tuple_with_str_enum_object(self) -> None:
        [payload] = normalize_facts(("Tom", "speaks", StrLanguage.FRENCH))
        assert payload["object"] == "French"
        assert json.dumps(payload["object"]) == '"French"'

    def test_tuple_subject_and_relationship_are_stringified(self) -> None:
        [payload] = normalize_facts((5, 7, "x"))
        assert (payload["subject"], payload["relationship"]) == ("5", "7")

    @pytest.mark.parametrize("certainty", [150, -1, "abc", True])
    def test_tuple_invalid_certainty_is_rejected(self, certainty: Any) -> None:
        with pytest.raises(InvalidFactError):
            normalize_facts(("Tom", "lives in", "Germany", certainty))

    def test_list_of_tuples(self) -> None:
        assert normalize_facts([("Tom", "lives in", "Germany"), ("Tom", "age", 42, 90)]) == [
            FACT_PAYLOAD,
            {"subject": "Tom", "relationship": "age", "object": 42, "certainty": 90},
        ]

    def test_tuple_of_tuples(self) -> None:
        assert normalize_facts(
            (("Tom", "lives in", "Germany"), ("Tom", "lives in", "Germany"))
        ) == [
            FACT_PAYLOAD,
            FACT_PAYLOAD,
        ]

    def test_list_of_lists(self) -> None:
        assert normalize_facts([["Tom", "lives in", "Germany"]]) == [FACT_PAYLOAD]

    def test_generator_of_tuples(self) -> None:
        assert normalize_facts(t for t in [("Tom", "lives in", "Germany")]) == [FACT_PAYLOAD]

    def test_list_of_facts_and_dicts(self) -> None:
        assert normalize_facts([FACT, FACT_DICT]) == [FACT_PAYLOAD, FACT_PAYLOAD]

    def test_mixed_list(self) -> None:
        assert normalize_facts([FACT, FACT_DICT, ("Tom", "lives in", "Germany")]) == [
            FACT_PAYLOAD,
            FACT_PAYLOAD,
            FACT_PAYLOAD,
        ]

    def test_list_of_one_tuple_with_enum_object_works(self) -> None:
        assert normalize_facts([("Tom", "lives in", Country.DE)]) == [FACT_PAYLOAD]

    @pytest.mark.parametrize("bad", [("Tom", "lives in"), ("a", "b", "c", 50, "extra"), ("Tom",)])
    def test_wrong_arity_tuple_is_rejected(self, bad: Any) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be"):
            normalize_facts(bad)

    def test_wrong_arity_tuple_inside_list_is_rejected(self) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be .* got tuple"):
            normalize_facts([("Tom", "lives in")])

    @pytest.mark.parametrize("empty", [[], (), set(), iter(())])
    def test_empty_is_rejected(self, empty: Any) -> None:
        with pytest.raises(InvalidFactError, match="must not be empty"):
            normalize_facts(empty)

    @pytest.mark.parametrize("bad", ["Tom lives in Germany", 42, None, 3.5, object()])
    def test_wrong_type_is_rejected(self, bad: Any) -> None:
        with pytest.raises(InvalidFactError, match="facts must be"):
            normalize_facts(bad)

    def test_wrong_item_type_is_rejected(self) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be .* got int"):
            normalize_facts([FACT, 42])

    def test_none_object_in_listed_tuple_is_rejected(self) -> None:
        with pytest.raises((InvalidFactError, InvalidAnswerError)):
            normalize_facts([("Tom", "lives in", None)])

    def test_answer_objects_are_not_facts(self) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be .* got Answer"):
            normalize_facts([ANSWER])

    def test_input_dict_is_not_mutated(self) -> None:
        data = {**FACT_DICT, "cf": 60}
        snapshot = dict(data)
        normalize_facts(data)
        assert data == snapshot

    # -- structural single-fact detection

    def test_single_tuple_with_enum_object(self) -> None:
        assert normalize_facts(("Tom", "lives in", Country.DE)) == [FACT_PAYLOAD]

    def test_single_tuple_with_enum_object_and_certainty(self) -> None:
        assert normalize_facts(("Tom", "lives in", Country.DE, 60)) == [
            {**FACT_PAYLOAD, "certainty": 60}
        ]

    def test_single_tuple_with_none_object_reports_the_none(self) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type NoneType"):
            normalize_facts(("Tom", "lives in", None))

    def test_single_tuple_with_arbitrary_object_reports_its_type(self) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type object"):
            normalize_facts(("Tom", "lives in", object()))

    def test_single_tuple_with_string_certainty(self) -> None:
        assert normalize_facts(("Tom", "lives in", "Germany", "60"))[0]["certainty"] == 60

    def test_single_tuple_with_none_certainty_is_rejected(self) -> None:
        with pytest.raises(InvalidFactError, match="certainty"):
            normalize_facts(("Tom", "lives in", "Germany", None))

    def test_single_list_with_date_object_and_certainty(self) -> None:
        assert normalize_facts(["Tom", "born on", dt.date(2000, 1, 1), 90]) == [
            {"subject": "Tom", "relationship": "born on", "object": "2000-01-01", "certainty": 90}
        ]

    def test_three_tuples_are_three_facts(self) -> None:
        triple = ("Tom", "lives in", "Germany")
        assert normalize_facts((triple, triple, triple)) == [FACT_PAYLOAD] * 3
        assert normalize_facts([triple, triple, triple, triple]) == [FACT_PAYLOAD] * 4

    def test_three_lists_are_three_facts(self) -> None:
        triple = ["Tom", "lives in", "Germany"]
        assert normalize_facts([triple, triple, triple]) == [FACT_PAYLOAD] * 3

    def test_three_dicts_are_three_facts(self) -> None:
        assert normalize_facts((FACT_DICT, FACT_DICT, FACT_DICT)) == [FACT_PAYLOAD] * 3

    def test_three_fact_objects_are_three_facts(self) -> None:
        assert normalize_facts([FACT, FACT, FACT]) == [FACT_PAYLOAD] * 3

    def test_mixed_three_item_sequence_containing_a_tuple_is_many_facts(self) -> None:
        # One nested tuple means "list of facts", so the bare strings are rejected as facts.
        with pytest.raises(InvalidFactError, match="each fact must be .* got str"):
            normalize_facts(["Tom", "lives in", ("Tom", "lives in", "Germany")])

    def test_three_item_sequence_containing_a_set_is_many_facts(self) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be .* got str"):
            normalize_facts(("Tom", "lives in", {"Germany"}))

    def test_three_item_sequence_containing_a_dict_is_many_facts(self) -> None:
        with pytest.raises(InvalidFactError, match="each fact must be .* got str"):
            normalize_facts(("Tom", "lives in", {"object": "Germany"}))

    # -- Fact objects are coerced like dicts and tuples

    def test_fact_instance_object_is_coerced(self) -> None:
        [payload] = normalize_facts(Fact("Tom", "born on", dt.date(2000, 1, 1)))
        assert payload["object"] == "2000-01-01"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (dt.datetime(2000, 1, 1, 8, 0), "2000-01-01"),
            (decimal.Decimal("1.5"), 1.5),
            (decimal.Decimal("42"), 42),
            (Country.DE, "Germany"),
        ],
    )
    def test_fact_instance_object_values_are_coerced(self, raw: Any, expected: Any) -> None:
        [payload] = normalize_facts(Fact("Tom", "r", raw))
        assert payload["object"] == expected
        assert type(payload["object"]) is type(expected)

    @pytest.mark.parametrize("raw", [None, {"a": 1}, [1, 2], object()])
    def test_fact_instance_unsupported_object_is_rejected(self, raw: Any) -> None:
        with pytest.raises(InvalidAnswerError, match="unsupported value type"):
            normalize_facts(Fact("Tom", "lives in", raw))

    def test_fact_instance_subject_and_relationship_are_stringified(self) -> None:
        [payload] = normalize_facts(Fact(5, 7, "x"))  # type: ignore[arg-type]
        assert (payload["subject"], payload["relationship"]) == ("5", "7")

    def test_fact_instance_str_enum_object_stays_json_compatible(self) -> None:
        [payload] = normalize_facts(Fact("Tom", "speaks", StrLanguage.FRENCH))
        assert payload["object"] == "French"
        assert isinstance(payload["object"], str)
        assert json.dumps(payload["object"]) == '"French"'

    def test_fact_instance_in_a_list_is_coerced(self) -> None:
        payload = normalize_facts([Fact("Tom", "born on", dt.datetime(2000, 1, 1, 8, 0)), FACT])
        assert payload[0]["object"] == "2000-01-01"
        assert payload[1] == FACT_PAYLOAD

    def test_fact_instance_is_not_mutated(self) -> None:
        fact = Fact("Tom", "born on", dt.date(2000, 1, 1), 90.0)
        normalize_facts(fact)
        assert fact.object == dt.date(2000, 1, 1)
        assert fact.certainty == 90.0

    def test_fact_dict_and_tuple_produce_identical_payloads(self) -> None:
        date = dt.datetime(2000, 1, 1, 8, 0)
        expected = [
            {"subject": "Tom", "relationship": "born on", "object": "2000-01-01", "certainty": 90}
        ]
        assert normalize_facts(("Tom", "born on", date, 90.0)) == expected
        assert normalize_facts(Fact("Tom", "born on", date, 90.0)) == expected
        assert (
            normalize_facts(
                {"subject": "Tom", "relationship": "born on", "object": date, "cf": 90.0}
            )
            == expected
        )

    def test_fact_payload_is_json_serialisable(self) -> None:
        payload = normalize_facts(
            [
                Fact("Tom", "born on", dt.datetime(2000, 1, 1, 8, 0), decimal.Decimal("90")),
                ("Tom", "lives in", Country.DE),
            ]
        )
        assert json.loads(json.dumps(payload)) == payload

    def test_session_inject_sends_bare_list(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        session.inject(("Tom", "lives in", "Germany"))
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/inject"
        assert fake.last.json == [FACT_PAYLOAD]

    def test_session_inject_many_shapes(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        session.inject(
            [
                FACT,
                {"subject": "Tom", "relationship": "age", "object": 42, "cf": 90},
                ("Tom", "born on", dt.date(2000, 1, 1)),
            ]
        )
        assert fake.last.json == [
            FACT_PAYLOAD,
            {"subject": "Tom", "relationship": "age", "object": 42, "certainty": 90},
            {"subject": "Tom", "relationship": "born on", "object": "2000-01-01", "certainty": 100},
        ]

    def test_session_inject_fact_helper(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        session.inject_fact("Tom", "lives in", "Germany", certainty=80)
        assert fake.last.json == [{**FACT_PAYLOAD, "certainty": 80}]

    def test_session_inject_coerces_fact_objects(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        session.inject(Fact("Tom", "born on", dt.datetime(2000, 1, 1, 8, 0), 90.0))
        assert fake.last.json == [
            {"subject": "Tom", "relationship": "born on", "object": "2000-01-01", "certainty": 90}
        ]

    def test_session_inject_single_tuple_with_enum_object(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        session.inject(("Tom", "lives in", Country.DE))
        assert fake.last.json == [FACT_PAYLOAD]

    def test_session_inject_rejects_bad_payload_locally(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        with pytest.raises(InvalidFactError):
            session.inject([])
        with pytest.raises(InvalidFactError):
            session.inject(("Tom", "lives in"))
        with pytest.raises(InvalidAnswerError, match="NoneType"):
            session.inject(("Tom", "lives in", None))
        assert fake.calls("/inject") == []
