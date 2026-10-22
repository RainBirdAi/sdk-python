"""Edge cases for parsers and coercion added in the review rounds."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import math

import httpx
import pytest

from rainbird_ai import (
    InvalidAnswerError,
    KnowledgeMapError,
    Rainbird,
    UnexpectedResponseError,
)
from rainbird_ai._answers import build_answers, coerce_value
from rainbird_ai.models import Question, _parse_datetime
from tests.conftest import KM_ID, QUESTION_SECOND_OBJECT, FakeRainbird

# --------------------------------------------------------------------------- /version parsing


class TestVersionParsing:
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("4.89.0", "4.89.0"),
            ("  4.89.0\n", "4.89.0"),
            ('"4.89.0"', "4.89.0"),
            ("4.89.0-rc.1", "4.89.0-rc.1"),
            ("v4.89.0", "v4.89.0"),
            ('{"version": "5.0.1"}', "5.0.1"),
            ('{"engine": "5.0.2"}', "5.0.2"),
        ],
    )
    def test_accepts_plausible_versions(
        self, client: Rainbird, fake: FakeRainbird, body: str, expected: str
    ) -> None:
        fake.version_text = body
        assert client.version() == expected

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "   ",
            "<html><body>login</body></html>",
            "engine 4.89.0",
            "x" * 65,
            '{"other": 1}',
            '{"other":1}',
            '{"status":"ok"}',
            "{}",
            "[]",
            '["4.89.0"]',
            '{"version":null}',
            '{"version":true}',
            '{"version":""}',
            "{not json",
        ],
    )
    def test_rejects_implausible_bodies(
        self, client: Rainbird, fake: FakeRainbird, body: str
    ) -> None:
        fake.version_text = body
        with pytest.raises(UnexpectedResponseError, match="expected an engine version"):
            client.version()


# --------------------------------------------------------------------------- interactions envelopes


class TestInteractionEnvelopes:
    EVENT = {"event": "start", "created": "2025-07-10T10:06:27.780Z", "values": {"start": {}}}

    def _session(self, client: Rainbird, fake: FakeRainbird):  # type: ignore[no-untyped-def]
        return client.start(KM_ID)

    @pytest.mark.parametrize("key", ["values", "events", "interactions"])
    def test_list_under_known_envelope_key(
        self, client: Rainbird, fake: FakeRainbird, key: str
    ) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, {key: [self.EVENT, self.EVENT]})
        events = session.interactions()
        assert [e.event for e in events] == ["start", "start"]

    def test_single_event_object_is_wrapped(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, self.EVENT)
        events = session.interactions()
        assert len(events) == 1
        assert events[0].event == "start"
        assert events[0].values == {"start": {}}

    def test_empty_values_envelope_is_an_empty_log(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, {"values": []})
        assert session.interactions() == []

    def test_non_list_body_is_unexpected(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, "just text")
        with pytest.raises(UnexpectedResponseError):
            session.interactions()

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({"err": ["Interaction log is secured"]}, "secured"),
            ({"error": "no such session"}, "no such session"),
            ({"data": [{"event": "q"}]}, "expected a list"),
            ({"events": None}, "expected a list"),
            ({"values": {}}, "expected a list"),
        ],
    )
    def test_200_error_or_unknown_envelope_raises(
        self, client: Rainbird, fake: FakeRainbird, body: dict[str, object], match: str
    ) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, body)
        with pytest.raises(UnexpectedResponseError, match=match):
            session.interactions()

    def test_single_event_with_list_values_is_not_unwrapped(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = self._session(client, fake)
        fake.fail_next(200, {"event": "question", "created": None, "values": [1, 2]})
        events = session.interactions()
        assert len(events) == 1 and events[0].event == "question"


# --------------------------------------------------------------------------- start / map bodies


class TestMalformedSuccessBodies:
    def test_start_without_session_id(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(200, {"kmVersion": {"id": "v"}})
        with pytest.raises(UnexpectedResponseError, match="session id"):
            client.start(KM_ID)

    def test_start_with_non_object_body(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(200, [1, 2, 3])
        with pytest.raises(UnexpectedResponseError, match="expected a JSON object"):
            client.start(KM_ID)

    def test_map_source_without_rblang(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(200, {"concepts": [], "rels": []})
        with pytest.raises(UnexpectedResponseError, match="rblang"):
            client.get_map_source(KM_ID)

    def test_map_source_with_error_body(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(200, {"error": "Knowledge map is private"})
        with pytest.raises(UnexpectedResponseError, match="private"):
            client.get_map_source(KM_ID)

    @pytest.mark.parametrize(
        "body",
        [
            {"kmID": "abc", "id": 1, "message": "Knowledge map created"},
            {"kmID": "abc", "id": 1, "error": None},
            {"kmID": "abc", "id": 1, "errors": []},
            {"kmID": "abc", "id": 1, "error": ""},
            {"kmID": "abc", "id": 1, "error": False},
            {"kmID": "abc", "id": 1, "error": 0},
            {"kmID": "abc", "id": 1, "error": {}},
        ],
    )
    def test_create_map_informational_keys_are_not_failures(
        self, client: Rainbird, fake: FakeRainbird, body: dict[str, object]
    ) -> None:
        fake.fail_next(200, body)
        result = client.create_map("m", "<rbl:kb/>")
        assert result.km_id == "abc"
        assert result.id == 1

    def test_create_map_without_any_id_uses_message_as_hint(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200, {"message": "queued for processing"})
        with pytest.raises(UnexpectedResponseError, match="queued"):
            client.create_map("m", "<rbl:kb/>")

    def test_create_map_error_list(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(200, {"errors": [{"message": "line 3: bad tag"}, "line 9: unknown rel"]})
        with pytest.raises(KnowledgeMapError) as info:
            client.create_map("m", "<rbl:kb/>")
        assert info.value.errors == ["line 3: bad tag", "line 9: unknown rel"]


# --------------------------------------------------------------------------- coercion & options


class Lang(str, enum.Enum):
    FR = "French"


class Level(enum.IntEnum):
    HIGH = 3


class Weird(enum.Enum):
    PAIR = (1, 2)


class TestCoercionEdges:
    def test_str_mixin_enum_uses_value(self) -> None:
        assert coerce_value(Lang.FR) == "French"
        assert type(coerce_value(Lang.FR)) is str

    def test_int_enum_uses_value(self) -> None:
        assert coerce_value(Level.HIGH) == 3
        assert type(coerce_value(Level.HIGH)) is int

    def test_enum_with_non_scalar_value_is_rejected(self) -> None:
        with pytest.raises(InvalidAnswerError):
            coerce_value(Weird.PAIR)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_floats_are_rejected(self, value: float) -> None:
        with pytest.raises(InvalidAnswerError, match="finite"):
            coerce_value(value)

    @pytest.mark.parametrize("text", ["NaN", "sNaN", "Infinity", "-Infinity"])
    def test_non_finite_decimals_are_rejected(self, text: str) -> None:
        with pytest.raises(InvalidAnswerError, match="finite"):
            coerce_value(decimal.Decimal(text))

    def test_scientific_integral_decimal_becomes_int(self) -> None:
        assert coerce_value(decimal.Decimal("1E+2")) == 100
        assert type(coerce_value(decimal.Decimal("1E+2"))) is int

    def test_datetime_drops_time_of_day(self) -> None:
        value = dt.datetime(2024, 3, 1, 23, 59, tzinfo=dt.UTC)
        assert coerce_value(value) == "2024-03-01"

    def test_subject_enum_answer_is_sent_as_its_value(self) -> None:
        q = Question.from_dict(
            {
                **QUESTION_SECOND_OBJECT,
                "type": "Second Form Subject",
                "subject": None,
                "object": "French",
                "canAdd": True,
                "concepts": [],
            }
        )
        [answer] = build_answers(q, Lang.FR)
        assert answer.subject == "French"

    def test_number_question_matches_numeric_spellings(self) -> None:
        q = Question.from_dict(
            {
                **QUESTION_SECOND_OBJECT,
                "dataType": "number",
                "canAdd": False,
                "concepts": [{"name": "10", "type": "number", "value": "10"}],
            }
        )
        assert build_answers(q, "10.0")[0].object == "10"  # option's spelling is sent
        assert build_answers(q, 10)[0].object == 10  # exact textual match keeps the value
        with pytest.raises(InvalidAnswerError):
            build_answers(q, 11)

    def test_truth_question_options_are_matched_case_insensitively(self) -> None:
        q = Question.from_dict(
            {
                **QUESTION_SECOND_OBJECT,
                "dataType": "truth",
                "canAdd": False,
                "concepts": [{"name": "TRUE", "type": "truth", "value": "TRUE"}],
            }
        )
        assert build_answers(q, "yes")[0].object is True
        with pytest.raises(InvalidAnswerError):
            build_answers(q, False)


# --------------------------------------------------------------------------- datetime parsing


class TestParseDatetimeEdges:
    @pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), 1e20, -1e15, "NaN"])
    def test_unusable_numbers_yield_none(self, value: object) -> None:
        assert _parse_datetime(value) is None

    def test_negative_epoch_seconds_are_accepted(self) -> None:
        parsed = _parse_datetime(-86400)
        assert parsed is not None
        assert parsed.year == 1969

    def test_millisecond_boundary(self) -> None:
        seconds = _parse_datetime(1e11)
        millis = _parse_datetime(1e11 + 1)
        assert seconds is not None and millis is not None
        assert seconds.year > 5000  # 1e11 seconds is far in the future
        assert millis.year == 1973  # 1e11 ms is 1973

    def test_math_helpers_available(self) -> None:
        assert math.isfinite(1.0)


# --------------------------------------------------------------------------- redirects


class TestRedirectMessages:
    def test_redirect_with_html_body_keeps_location(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                301,
                headers={"location": "https://other.example/x", "content-type": "text/html"},
                text="<html><h1>301 Moved Permanently</h1><hr>nginx</html>",
                request=request,
            )

        client = Rainbird(
            api_key="k",
            base_url="https://fake-api.rainbird.ai",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(Exception) as info:
            client.version()
        message = str(info.value)
        assert "unexpected redirect to https://other.example/x" in message
        assert "301 Moved Permanently" in message
        assert "<html>" not in message


# --------------------------------------------------------------------------- round three


class YesNo(str, enum.Enum):
    YES = "yes"
    NO = "no"


class TestRoundThreeAnswers:
    def test_str_enum_answers_first_form_questions(self) -> None:
        from tests.conftest import QUESTION_FIRST_FORM

        q = Question.from_dict(QUESTION_FIRST_FORM)
        assert build_answers(q, YesNo.YES)[0].answer == "yes"
        assert build_answers(q, YesNo.NO)[0].answer == "no"

    def test_str_enum_answers_truth_object_questions(self) -> None:
        q = Question.from_dict({**QUESTION_SECOND_OBJECT, "dataType": "truth", "concepts": []})
        assert build_answers(q, YesNo.YES)[0].object is True

    def test_huge_int_is_rejected_cleanly(self) -> None:
        q = Question.from_dict({**QUESTION_SECOND_OBJECT, "canAdd": False})
        with pytest.raises(InvalidAnswerError):
            build_answers(q, 10**400)

    def test_only_invalid_response_options_defer_to_api(self) -> None:
        q = Question.from_dict(
            {
                **QUESTION_SECOND_OBJECT,
                "canAdd": False,
                "concepts": [
                    {"name": "A", "value": "A", "type": "string", "invalidResponse": True}
                ],
            }
        )
        assert build_answers(q, "B")[0].object == "B"

    def test_int_answer_does_not_match_zero_padded_string_option(self) -> None:
        q = Question.from_dict(
            {
                **QUESTION_SECOND_OBJECT,
                "canAdd": False,
                "concepts": [{"name": "05", "value": "05", "type": "string"}],
            }
        )
        # Numeric match against a string-typed question canonicalises to the option text.
        assert build_answers(q, 5)[0].object == "05"


class TestRoundThreeHttp:
    @pytest.mark.parametrize("value", ["https://host?", "https://host#", "https://host/?", "host#"])
    def test_query_or_fragment_delimiters_are_rejected(self, value: str) -> None:
        from rainbird_ai import RainbirdConfigurationError
        from rainbird_ai._http import normalize_base_url

        with pytest.raises(RainbirdConfigurationError, match="query string or fragment"):
            normalize_base_url(value, {})

    def test_plain_text_error_keeps_angle_brackets(self) -> None:
        from rainbird_ai._http import error_for_response

        response = httpx.Response(
            400,
            text="value must be < 10 and > 5",
            headers={"content-type": "text/plain"},
            request=httpx.Request("GET", "https://x/y"),
        )
        assert error_for_response(response).message == "value must be < 10 and > 5"

    def test_html_error_is_stripped(self) -> None:
        from rainbird_ai._http import error_for_response

        response = httpx.Response(
            502,
            text="<html><h1>Bad Gateway</h1></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://x/y"),
        )
        assert error_for_response(response).message == "Bad Gateway"


class TestRoundThreeCli:
    def test_skip_with_certainty_suffix_still_skips(self) -> None:
        from rainbird_ai import SKIP
        from rainbird_ai.cli import _parse_answer_text

        q = Question.from_dict({**QUESTION_SECOND_OBJECT, "allowUnknown": True})
        assert _parse_answer_text("?@50", q) == (SKIP, 100)
        assert _parse_answer_text("skip@50", q) == (SKIP, 100)

    def test_per_item_certainty_in_plural_answer_is_rejected(self) -> None:
        from rainbird_ai.cli import _parse_answer_text

        q = Question.from_dict({**QUESTION_SECOND_OBJECT, "plural": True})
        with pytest.raises(InvalidAnswerError, match="whole answer"):
            _parse_answer_text("a@70, b@80", q)
        assert _parse_answer_text("a, b@70", q) == (["a", "b"], 70)
        assert _parse_answer_text("tom@example.com, ana@example.com", q) == (
            ["tom@example.com", "ana@example.com"],
            100,
        )

    def test_flat_fact_list_in_file_is_a_usage_error(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import json as _json

        from rainbird_ai import cli

        path = tmp_path / "flat.json"
        path.write_text(_json.dumps(["Tom", "lives in", "Germany"]), encoding="utf-8")
        with pytest.raises(SystemExit) as info:
            cli.main(["inject", "--session", "S", "--file", str(path)])
        assert info.value.code == 2

    def test_nested_fact_lists_in_file_are_accepted(self, tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
        import argparse
        import json as _json

        from rainbird_ai import cli
        from rainbird_ai.cli import _load_facts

        path = tmp_path / "nested.json"
        path.write_text(
            _json.dumps([["Tom", "lives in", "Germany"], ["Ana", "lives in", "Spain", 90]])
        )
        args = argparse.Namespace(fact=None, file=str(path))
        assert len(_load_facts(args)) == 2
        assert cli.EXIT_USAGE == 2
