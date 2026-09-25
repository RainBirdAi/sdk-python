"""Tests for the ``rainbird`` command-line interface (``rainbird_ai.cli``).

Every test drives ``rainbird_ai.cli.main([...])`` end to end against the in-memory
``FakeRainbird`` from ``conftest.py``. ``rainbird_ai.cli.Rainbird`` is monkeypatched
so the client the CLI builds talks to the fake through ``httpx.MockTransport``, and
``rainbird_ai.cli.engine_version`` (used by the keyless ``version`` command) is routed
through the same transport; all other CLI code (argument parsing, ``_client``, output
formatting) is real.
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from rainbird_ai import Rainbird, cli
from rainbird_ai._version import __version__
from rainbird_ai.client import engine_version as real_engine_version

from .conftest import (
    API_KEY,
    BASE_URL,
    EVIDENCE_LEAVES,
    EVIDENCE_ROOT,
    EXPLAIN_OK,
    INTERACT_ERROR,
    INTERACT_QUESTION,
    INTERACT_RESULT,
    INTERACTIONS,
    KM_ID,
    KM_VERSION_ID,
    MAP_SOURCE,
    QUESTION_FIRST_FORM,
    QUESTION_PLURAL_LOSS,
    QUESTION_SECOND_OBJECT,
    QUESTION_SECOND_SUBJECT,
    RBLANG,
    RESULT_TOM_SPEAKS_GERMAN,
    SESSION_FACTS,
    SESSION_VERSION,
    FakeRainbird,
    leaf_evidence,
)

SID = "sess-0001"
ROOT_FACT_ID: str = EVIDENCE_ROOT["factID"]
LEAF_FACT_IDS = list(EVIDENCE_LEAVES)
RESULT_ITEM = RESULT_TOM_SPEAKS_GERMAN["result"][0]
RESULT_FACT_ID: str = RESULT_ITEM["factID"]
NEW_KM_ID = "99999999-8888-7777-6666-555555555555"


# --------------------------------------------------------------------------- helpers


@dataclass
class Run:
    code: int
    out: str
    err: str

    @property
    def json(self) -> Any:
        return json.loads(self.out)


Runner = Callable[..., Run]


def seed_session(fake: FakeRainbird, sid: str = SID) -> str:
    """Register ``sid`` with the fake so session endpoints accept it (no /start call)."""
    fake.sessions[sid] = {"km": KM_ID, "params": {}, "facts": [], "answers": []}
    return sid


def answers_payload(*answers: dict[str, Any]) -> dict[str, Any]:
    return {"answers": list(answers)}


class _Tty(io.StringIO):
    """stdin stand-in that claims to be interactive."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def run(
    fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Iterator[Runner]:
    """Return ``run(*argv) -> Run`` executing the CLI against the fake server."""
    http = httpx.Client(transport=httpx.MockTransport(fake.handle))

    def factory(api_key: str | None = None, **kwargs: Any) -> Rainbird:
        # Same call the real ``_client`` makes, plus the fake transport. The test key is
        # used unless the test passes --api-key or sets RAINBIRD_API_KEY explicitly.
        if api_key is None and "RAINBIRD_API_KEY" not in os.environ:
            api_key = API_KEY
        if kwargs.get("base_url") is None:
            kwargs["base_url"] = BASE_URL
        kwargs["http_client"] = http
        return Rainbird(api_key, **kwargs)

    monkeypatch.setattr(cli, "Rainbird", factory)

    def fake_engine_version(base_url: str | None = None, **kwargs: Any) -> str:
        # ``rainbird version`` without a key calls the module-level helper directly;
        # route it through the fake too.
        kwargs["http_client"] = http
        return real_engine_version(base_url if base_url is not None else BASE_URL, **kwargs)

    monkeypatch.setattr(cli, "engine_version", fake_engine_version)

    def _run(*argv: str) -> Run:
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        out, err = capsys.readouterr()
        return Run(code=code, out=out, err=err)

    yield _run
    http.close()


def _subparsers(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.ArgumentParser]]:
    found: list[tuple[str, argparse.ArgumentParser]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            found.extend(action.choices.items())
    return found


# --------------------------------------------------------------------------- parser


class TestParser:
    def test_build_parser_returns_argument_parser(self) -> None:
        parser = cli.build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "rainbird"

    def test_every_subcommand_has_func_default(self) -> None:
        parser = cli.build_parser()
        leaves: list[tuple[str, argparse.ArgumentParser]] = []
        for name, sub in _subparsers(parser):
            nested = _subparsers(sub)
            if nested:
                leaves.extend((f"{name} {inner}", p) for inner, p in nested)
            else:
                leaves.append((name, sub))
        assert {name for name, _ in leaves} == {
            "version",
            "start",
            "inject",
            "query",
            "decide",
            "respond",
            "undo",
            "evidence",
            "explain",
            "interact",
            "facts",
            "session",
            "interactions",
            "map get",
            "map create",
        }
        for name, sub in leaves:
            func = sub.get_default("func")
            assert callable(func), f"{name} has no func default"

    def test_global_option_defaults(self) -> None:
        args = cli.build_parser().parse_args(["version"])
        assert args.command == "version"
        assert args.json is False
        assert args.api_key is None
        assert args.base_url is None
        assert args.timeout == 30.0
        assert args.func is cli.cmd_version

    def test_query_option_defaults(self) -> None:
        args = cli.build_parser().parse_args(["query", "-r", "speaks"])
        assert args.relationship == "speaks"
        assert args.subject is None
        assert args.answers is None
        assert args.no_input is False
        assert args.no_strict is False
        assert args.fact is None
        assert args.file is None

    def test_unknown_command_exits_2(self, run: Runner) -> None:
        result = run("bogus")
        assert result.code == 2
        assert "invalid choice: 'bogus'" in result.err

    def test_missing_command_exits_2(self, run: Runner) -> None:
        result = run()
        assert result.code == 2
        assert "required" in result.err.lower()

    def test_sdk_version_flag(self, run: Runner) -> None:
        result = run("-V")
        assert result.code == 0
        assert result.out.strip() == f"rainbird-ai {__version__}"

    def test_help_lists_commands(self, run: Runner) -> None:
        result = run("--help")
        assert result.code == 0
        for name in ("version", "start", "query", "inject", "evidence", "interact", "map"):
            assert name in result.out

    def test_query_requires_relationship(self, run: Runner) -> None:
        result = run("query", "--km-id", KM_ID)
        assert result.code == 2
        assert "--relationship" in result.err

    def test_inject_requires_session(self, run: Runner) -> None:
        result = run("inject", "--fact", "Tom", "lives in", "Germany")
        assert result.code == 2
        assert "--session" in result.err

    def test_interactions_format_is_restricted(self, run: Runner) -> None:
        result = run("interactions", "--session", SID, "--format", "xml")
        assert result.code == 2
        assert "invalid choice" in result.err

    def test_map_requires_action(self, run: Runner) -> None:
        result = run("map")
        assert result.code == 2

    def test_json_flag_accepted_after_subcommand(self, run: Runner) -> None:
        result = run("version", "--json")
        assert result.code == 0
        assert result.json["sdk"] == __version__

    def test_version_help_says_no_api_key_needed(self, run: Runner) -> None:
        result = run("--help")
        assert result.code == 0
        assert "(no API key needed)" in result.out

    def test_module_docstring_documents_exit_codes(self) -> None:
        doc = cli.__doc__ or ""
        assert "Exit codes" in doc
        for code in ("0 success", "2 usage error", "3 a question is still pending", "130"):
            assert code in doc


class TestGlobalOptionPlacement:
    """Global options are accepted before *or* after the subcommand (and after ``map``
    actions); a value given after the subcommand overrides one given before it."""

    LEAVES = [
        ["version"],
        ["start"],
        ["inject", "--session", SID],
        ["query", "-r", "speaks"],
        ["decide", "-r", "speaks"],
        ["respond", "--session", SID, "[]"],
        ["undo", "--session", SID],
        ["evidence", "--session", SID, "--fact-id", "F"],
        ["explain", "--session", SID, "--fact-id", "F"],
        ["interact", "hi"],
        ["facts", "--session", SID],
        ["session", "--session", SID],
        ["interactions", "--session", SID],
        ["map", "get", KM_ID],
        ["map", "create", "--name", "n", "--file", "f"],
    ]

    @pytest.mark.parametrize("leaf", LEAVES, ids=lambda leaf: " ".join(leaf[:2]))
    def test_every_leaf_accepts_global_options_after_it(self, leaf: list[str]) -> None:
        argv = [
            *leaf,
            "--api-key",
            "k",
            "--base-url",
            "https://x.example",
            "--engine",
            "next",
            "--evidence-key",
            "ev",
            "--interaction-key",
            "ix",
            "--timeout",
            "7.5",
            "--json",
        ]
        args = cli.build_parser().parse_args(argv)
        assert args.api_key == "k"
        assert args.base_url == "https://x.example"
        assert args.engine == "next"
        assert args.evidence_key == "ev"
        assert args.interaction_key == "ix"
        assert args.timeout == 7.5
        assert args.json is True

    @pytest.mark.parametrize("leaf", LEAVES, ids=lambda leaf: " ".join(leaf[:2]))
    def test_root_defaults_survive_when_not_repeated(self, leaf: list[str]) -> None:
        args = cli.build_parser().parse_args(leaf)
        assert args.api_key is None
        assert args.base_url is None
        assert args.engine is None
        assert args.evidence_key is None
        assert args.interaction_key is None
        assert args.timeout == 30.0
        assert args.json is False

    def test_value_after_subcommand_overrides_value_before_it(self) -> None:
        args = cli.build_parser().parse_args(
            ["--timeout", "5", "--api-key", "a", "version", "--timeout", "9", "--api-key", "b"]
        )
        assert args.timeout == 9.0
        assert args.api_key == "b"

    def test_before_and_after_can_be_mixed(self) -> None:
        args = cli.build_parser().parse_args(
            ["--json", "start", "--km-id", KM_ID, "--timeout", "2"]
        )
        assert args.json is True
        assert args.timeout == 2.0
        assert args.km_id == KM_ID

    def test_json_after_start(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("start", "--km-id", KM_ID, "--json")
        assert result.code == 0, result.err
        assert set(result.json) == {"session_id", "km_version_id", "km_id"}
        assert result.json["session_id"] in fake.sessions

    def test_json_after_query(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "--answers", "German", "--json")
        assert result.code == 0, result.err
        assert result.json["result"] == [RESULT_ITEM]
        assert "Q:" not in result.out

    def test_api_key_after_subcommand_is_sent(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("start", "--km-id", KM_ID, "--api-key", "wrong")
        assert result.code == 1
        assert fake.last.headers["x-api-key"] == "wrong"
        assert "HTTP 401" in result.err

    def test_engine_after_subcommand_sets_header(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("version", "--engine", "next").code == 0
        assert fake.last.headers["x-rainbird-engine"] == "next"

    def test_evidence_key_after_subcommand(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "ev-secret"
        result = run(
            "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--evidence-key", "ev-secret"
        )
        assert result.code == 0, result.err
        assert fake.last.headers["x-evidence-key"] == "ev-secret"

    def test_interaction_key_after_subcommand(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "ix-secret"
        result = run("interactions", "--session", SID, "--interaction-key", "ix-secret")
        assert result.code == 0, result.err
        assert fake.last.headers["x-interaction-key"] == "ix-secret"

    def test_base_url_after_subcommand(self, run: Runner) -> None:
        result = run(
            "evidence",
            "--session",
            SID,
            "--fact-id",
            ROOT_FACT_ID,
            "--url",
            "--base-url",
            "https://enterprise-api.rainbird.ai",
        )
        assert result.code == 0, result.err
        assert result.out.splitlines()[-1] == (
            f"https://enterprise.rainbird.ai/evidence?id={ROOT_FACT_ID}"
            f"&api=https://enterprise-api.rainbird.ai&sid={SID}"
        )

    def test_json_after_map_get(self, run: Runner) -> None:
        result = run("map", "get", KM_ID, "--json")
        assert result.code == 0, result.err
        assert result.json == MAP_SOURCE

    def test_json_after_map_create(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        path = tmp_path / "map.rbl"
        path.write_text(RBLANG, encoding="utf-8")
        result = run("map", "create", "--name", "My map", "--file", str(path), "--json")
        assert result.code == 0, result.err
        assert result.json == fake.map_create_body


# --------------------------------------------------------------------------- version


class TestVersion:
    def test_text_output(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("version")
        assert result.code == 0
        assert result.out == f"engine 4.89.0\nsdk    {__version__}\n"
        assert result.err == ""
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "version")
        assert result.code == 0
        assert result.json == {"engine": "4.89.0", "sdk": __version__}

    def test_custom_engine_version_text(self, run: Runner, fake: FakeRainbird) -> None:
        fake.version_text = "5.0.1"
        assert run("version").out.splitlines()[0] == "engine 5.0.1"

    def test_works_without_api_key(self, run: Runner, fake: FakeRainbird) -> None:
        # No --api-key and no RAINBIRD_API_KEY (autouse _clean_env): GET /version is public.
        result = run("version")
        assert result.code == 0, result.err
        assert result.out.splitlines()[0] == "engine 4.89.0"
        assert len(fake.requests) == 1
        assert fake.last.path == "/version"
        assert "x-api-key" not in fake.last.headers

    def test_uses_api_key_flag_when_given(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("--api-key", API_KEY, "version")
        assert result.code == 0, result.err
        assert fake.last.path == "/version"
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_uses_api_key_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", API_KEY)
        assert run("version").code == 0
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_api_error_without_key_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        # (A 5xx on GET would be retried and then succeed; 404 is not retried.)
        fake.fail_next(404, "<html><body>Not Found</body></html>")
        result = run("version")
        assert result.code == 1
        assert result.out == ""
        assert "HTTP 404" in result.err
        assert "Not Found" in result.err  # HTML body is tag-stripped in the message
        assert "<html>" not in result.err


# --------------------------------------------------------------------------- start


class TestStart:
    def test_prints_session_id(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("start", "--km-id", KM_ID)
        assert result.code == 0
        sid = result.out.strip()
        assert sid in fake.sessions
        assert result.err == ""
        calls = fake.calls("/start/")
        assert len(calls) == 1
        assert calls[0].method == "GET"
        assert calls[0].path == f"/start/{KM_ID}"
        assert calls[0].params == {}
        assert calls[0].headers["x-api-key"] == API_KEY

    def test_json_shape(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("--json", "start", "--km-id", KM_ID)
        assert result.code == 0
        data = result.json
        assert set(data) == {"session_id", "km_version_id", "km_id"}
        assert data["session_id"] in fake.sessions
        assert data["km_version_id"] == KM_VERSION_ID
        assert data["km_id"] == KM_ID

    def test_draft_sends_use_draft(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("start", "--km-id", KM_ID, "--draft").code == 0
        assert fake.last.params == {"useDraft": "true"}

    def test_version_sends_version(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("start", "--km-id", KM_ID, "--version", "3").code == 0
        assert fake.last.params == {"version": "3"}

    def test_context_id_sends_contextid(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("start", "--km-id", KM_ID, "--context-id", "ctx-42").code == 0
        assert fake.last.params == {"contextid": "ctx-42"}

    def test_draft_and_version_are_mutually_exclusive(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        result = run("start", "--km-id", KM_ID, "--draft", "--version", "2")
        assert result.code == 2
        assert "not allowed with argument" in result.err
        assert fake.requests == []

    def test_version_must_be_an_integer(self, run: Runner) -> None:
        result = run("start", "--km-id", KM_ID, "--version", "latest")
        assert result.code == 2
        assert "invalid int value" in result.err

    def test_version_must_be_positive(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("start", "--km-id", KM_ID, "--version", "0")
        assert result.code == 1
        assert "positive integer" in result.err
        assert fake.requests == []

    def test_km_id_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", KM_ID)
        result = run("start")
        assert result.code == 0
        assert fake.last.path == f"/start/{KM_ID}"

    def test_missing_km_id_is_an_error(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("start")
        assert result.code == 1
        assert result.err.startswith("rainbird: ")
        assert "km_id is required" in result.err
        assert "RAINBIRD_KM_ID" in result.err
        assert fake.requests == []


# --------------------------------------------------------------------------- query


class TestQueryScripted:
    def test_second_form_then_first_form_then_result(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(RESULT_ITEM)

        result = run(
            "query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--answers", "German", "yes"
        )

        assert result.code == 0, result.err
        assert result.err == ""
        sid = next(iter(fake.sessions))
        lines = result.out.splitlines()
        assert lines[0] == f"session {sid}"
        assert "Q: What is the national language of Germany?" in lines
        assert "A: German" in lines
        assert "Q: Does Tom speak French?" in lines
        assert "A: yes" in lines
        assert "Tom  speaks  German    certainty=75%" in lines
        assert f"    factID: {RESULT_FACT_ID}" in lines

        assert fake.calls("/query")[0].json == {"relationship": "speaks", "subject": "Tom"}
        responses = fake.calls("/response")
        assert len(responses) == 2
        assert responses[0].json == answers_payload(
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "German",
                "certainty": 100,
            }
        )
        assert responses[1].json == answers_payload(
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "answer": "yes",
            }
        )

    def test_first_form_no_answer(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "-a", "no").code == 0
        assert fake.calls("/response")[0].json["answers"][0]["answer"] == "no"

    def test_json_output(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)

        result = run(
            "--json", "query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "-a", "German"
        )

        assert result.code == 0, result.err
        data = result.json  # stdout must be pure JSON: no "session"/"Q:"/"A:" lines
        sid = next(iter(fake.sessions))
        assert data["session_id"] == sid
        assert data["sid"] == sid
        assert data["result"] == [RESULT_ITEM]
        assert "Q:" not in result.out

    def test_query_object_is_sent(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-o", "German").code == 0
        assert fake.calls("/query")[0].json == {"relationship": "speaks", "object": "German"}

    def test_immediate_result_needs_no_answers(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 0
        assert fake.calls("/response") == []
        assert "Tom  speaks  German" in result.out

    def test_empty_result_prints_message(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result()
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 0
        assert "No result: the engine could not infer anything." in result.out

    def test_plural_comma_separated_answers(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(RESULT_ITEM)

        result = run(
            "query",
            "--km-id",
            KM_ID,
            "-r",
            "can claim",
            "-s",
            "Ben",
            "--answers",
            "cosmetic damage, theft",
        )

        assert result.code == 0, result.err
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "Ben",
                "relationship": "suffered",
                "object": "cosmetic damage",
                "certainty": 100,
            },
            {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 100},
        )

    def test_comma_in_answer_to_non_plural_question_is_literal(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # plural False, canAdd True
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "Low, Saxon").code == 0
        answers = fake.calls("/response")[0].json["answers"]
        assert len(answers) == 1
        assert answers[0]["object"] == "Low, Saxon"

    def test_certainty_suffix(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "-a", "German@80")

        assert result.code == 0, result.err
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "German",
                "certainty": 80,
            }
        )
        assert "A: German@80" in result.out

    def test_certainty_suffix_applies_to_every_plural_value(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "x", "-a", "theft,water damage@60").code == 0
        answers = fake.calls("/response")[0].json["answers"]
        assert [a["object"] for a in answers] == ["theft", "water damage"]
        assert [a["certainty"] for a in answers] == [60, 60]

    def test_float_certainty_suffix(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German@80.5").code == 0
        assert fake.calls("/response")[0].json["answers"][0]["certainty"] == 80.5

    def test_integral_float_certainty_suffix_is_sent_as_int(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German@80.0").code == 0
        assert fake.calls("/response")[0].json["answers"][0]["certainty"] == 80

    @pytest.mark.parametrize("suffix", ["150", "100.5"])
    def test_out_of_range_certainty_suffix_exits_1(
        self, run: Runner, fake: FakeRainbird, suffix: str
    ) -> None:
        # A scripted answer that the SDK rejects is an InvalidAnswerError -> exit 1 (not a
        # usage error): the value came from --answers, not from the command line syntax.
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", f"German@{suffix}")
        assert result.code == 1
        assert result.err.startswith("rainbird: ")
        assert f"invalid certainty {suffix!r}" in result.err
        assert "use a number 0-100" in result.err
        assert fake.calls("/response") == []

    def test_email_address_answer_is_a_plain_value(self, run: Runner, fake: FakeRainbird) -> None:
        # "@" only separates a certainty when what follows is numeric.
        fake.queue_question(QUESTION_SECOND_OBJECT)  # canAdd True
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "tom@example.com")
        assert result.code == 0, result.err
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "tom@example.com",
                "certainty": 100,
            }
        )
        assert "A: tom@example.com" in result.out

    def test_non_numeric_at_suffix_is_part_of_the_value(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # canAdd True
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German@lots").code == 0
        answer = fake.calls("/response")[0].json["answers"][0]
        assert answer["object"] == "German@lots"
        assert answer["certainty"] == 100

    def test_email_answer_with_certainty_suffix(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "tom@example.com@60").code == 0
        answer = fake.calls("/response")[0].json["answers"][0]
        assert answer["object"] == "tom@example.com"
        assert answer["certainty"] == 60

    def test_skip_first_form_question(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)  # allowUnknown True
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "-a", "?")

        assert result.code == 0, result.err
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        )

    def test_skip_second_form_subject_question_blanks_subject(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)  # allowUnknown True
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-o", "French", "-a", "?").code == 0
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        )

    @pytest.mark.parametrize("token", ["skip", "SKIP", " ? "])
    def test_skip_aliases(self, run: Runner, fake: FakeRainbird, token: str) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "-a", token).code == 0
        assert fake.calls("/response")[0].json["answers"][0]["unanswered"] is True

    def test_skip_not_allowed_is_rejected_client_side(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown False
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "?", "--no-input")
        assert result.code == 1
        assert "cannot be skipped" in result.err
        assert fake.calls("/response") == []

    def test_no_strict_sends_skip_anyway(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown False
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "?", "--no-strict")
        assert result.code == 0, result.err
        assert fake.calls("/response")[0].json == answers_payload(
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        )

    def test_answer_outside_options_rejected_when_can_add_false(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)  # canAdd False
        result = run("query", "--km-id", KM_ID, "-r", "x", "-a", "arson", "--no-input")
        assert result.code == 1
        assert "'arson' is not one of the allowed answers" in result.err
        assert fake.calls("/response") == []

    def test_answers_beyond_the_result_are_ignored(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German", "yes")
        assert result.code == 0
        assert fake.calls("/response") == []

    def test_invalid_last_scripted_answer_exits_1_without_tty(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO())  # isatty() is False
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown False
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "?")
        assert result.code == 1
        assert result.err.startswith("rainbird: ")
        assert "cannot be skipped" in result.err
        assert "cannot be skipped" not in result.out
        assert "waiting for an answer" not in result.err
        assert fake.calls("/response") == []

    def test_invalid_last_scripted_answer_exits_1_with_tty(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even when a terminal is available a rejected --answers value is fatal: the CLI
        # never falls back to prompting for a value the user scripted wrongly.
        monkeypatch.setattr(sys, "stdin", _Tty())
        monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("prompted"))
        fake.queue_question(QUESTION_PLURAL_LOSS)  # canAdd False
        result = run("query", "--km-id", KM_ID, "-r", "x", "-a", "arson")
        assert result.code == 1
        assert "'arson' is not one of the allowed answers" in result.err
        assert fake.calls("/response") == []


class TestQueryNoInput:
    def test_remaining_question_exits_3(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--no-input")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        assert result.out.splitlines() == [
            f"session {sid}",
            "What is the national language of Germany?",
            "    options: English, French",
            "    append @<certainty> e.g. German@80",
        ]
        assert result.err == (
            f"rainbird: session {sid} is waiting for an answer; resume with --session {sid}\n"
        )
        assert fake.calls("/response") == []

    def test_remaining_question_after_scripted_answers(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German", "--no-input")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        assert len(fake.calls("/response")) == 1
        assert "A: German" in result.out
        assert "Does Tom speak French?" in result.out
        assert "yes/no" in result.out
        assert f"session {sid} is waiting for an answer" in result.err
        assert f"--session {sid}" in result.err

    def test_json_emits_question_response(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("--json", "query", "--km-id", KM_ID, "-r", "speaks", "--no-input")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        data = result.json  # stdout is pure JSON; the hint goes to stderr
        assert data["session_id"] == sid
        assert data["question"] == QUESTION_SECOND_OBJECT
        assert data["sid"] == sid
        assert result.err == (
            f"rainbird: session {sid} is waiting for an answer; resume with --session {sid}\n"
        )

    def test_non_tty_stdin_behaves_like_no_input(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO())
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("query", "--km-id", KM_ID, "-r", "speaks")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        assert "What is the national language of Germany?" in result.out
        assert result.err == (
            f"rainbird: session {sid} is waiting for an answer; resume with --session {sid}\n"
        )
        assert fake.calls("/response") == []

    def test_reports_the_reused_session_id(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO())
        seed_session(fake)
        fake.queue_question(QUESTION_FIRST_FORM)
        result = run("query", "--session", SID, "-r", "speaks")
        assert result.code == 3
        assert result.out.splitlines()[0] == "Does Tom speak French?"
        assert f"session {SID} is waiting for an answer" in result.err
        assert f"resume with --session {SID}" in result.err


class TestQueryInteractive:
    @pytest.fixture
    def typed(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], list[str]]:
        """Make stdin look like a tty and feed ``input()`` from a list."""

        def _typed(values: list[str]) -> list[str]:
            monkeypatch.setattr(sys, "stdin", _Tty())

            def fake_input(prompt: str = "") -> str:
                if not values:
                    raise EOFError
                return values.pop(0)

            monkeypatch.setattr(builtins, "input", fake_input)
            return values

        return _typed

    def test_prompts_and_sends_typed_answers(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed(["German@90", "y"])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")

        assert result.code == 0, result.err
        assert "What is the national language of Germany?" in result.out
        assert "    options: English, French" in result.out
        assert "Does Tom speak French?" in result.out
        assert "yes/no; type '?' to skip; append @<certainty>" in result.out
        responses = fake.calls("/response")
        assert responses[0].json["answers"][0] == {
            "subject": "Germany",
            "relationship": "national language",
            "object": "German",
            "certainty": 90,
        }
        assert responses[1].json["answers"][0]["answer"] == "yes"
        assert "Tom  speaks  German    certainty=75%" in result.out

    def test_scripted_answers_are_used_before_prompting(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed(["no"])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-a", "German")
        assert result.code == 0, result.err
        assert "A: German" in result.out
        assert fake.calls("/response")[1].json["answers"][0]["answer"] == "no"

    def test_invalid_typed_answer_reprompts(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed(["arson", "theft"])
        fake.queue_question(QUESTION_PLURAL_LOSS)  # canAdd False
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--km-id", KM_ID, "-r", "can claim", "-s", "Ben")

        assert result.code == 0
        assert "  ! 'arson' is not one of the allowed answers" in result.err
        assert "arson" not in result.out
        assert result.out.count("What type of loss has Ben suffered?") == 2
        responses = fake.calls("/response")
        assert len(responses) == 1
        assert responses[0].json["answers"][0]["object"] == "theft"

    def test_out_of_range_certainty_reprompts(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed(["German@150", "German@80"])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")

        assert result.code == 0
        assert "  ! invalid certainty '150'; use a number 0-100" in result.err
        assert result.out.count("What is the national language of Germany?") == 2
        responses = fake.calls("/response")
        assert len(responses) == 1
        assert responses[0].json["answers"][0]["certainty"] == 80

    def test_typed_email_address_is_a_plain_value(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed(["tom@example.com"])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(RESULT_ITEM)
        result = run("query", "--km-id", KM_ID, "-r", "speaks")
        assert result.code == 0, result.err
        assert result.err == ""
        answer = fake.calls("/response")[0].json["answers"][0]
        assert answer["object"] == "tom@example.com"
        assert answer["certainty"] == 100

    def test_eof_exits_3(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed([])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("query", "--km-id", KM_ID, "-r", "speaks")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        assert fake.calls("/response") == []
        # The question is shown once when prompting and not repeated after Ctrl-D.
        assert result.out.count("What is the national language of Germany?") == 1
        assert result.err.startswith("\n")  # newline after the interrupted prompt
        assert f"rainbird: session {sid} is waiting for an answer" in result.err

    def test_eof_json_emits_question_response(
        self, run: Runner, fake: FakeRainbird, typed: Callable[[list[str]], list[str]]
    ) -> None:
        typed([])
        fake.queue_question(QUESTION_FIRST_FORM)
        result = run("--json", "query", "--km-id", KM_ID, "-r", "speaks")
        sid = next(iter(fake.sessions))
        assert result.code == 3
        # Text mode prints the question before prompting; JSON mode does too (stdout is
        # then not pure JSON), so only the last JSON document is checked here.
        body = result.out[result.out.index("{") :]
        data = json.loads(body)
        assert data["session_id"] == sid
        assert data["question"] == QUESTION_FIRST_FORM


class TestQuerySessionAndFacts:
    def test_session_reuses_existing_session(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)

        result = run("query", "--session", SID, "-r", "speaks", "-s", "Tom")

        assert result.code == 0, result.err
        assert fake.calls("/start/") == []
        assert fake.calls("/query")[0].path == f"/{SID}/query"
        assert "session " not in result.out  # no "session <id>" banner when resuming
        assert "Tom  speaks  German" in result.out

    def test_session_json_reports_the_reused_id(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)
        result = run("--json", "query", "--session", SID, "-r", "speaks")
        assert result.code == 0
        assert result.json["session_id"] == SID

    def test_unknown_session_is_an_error(self, run: Runner) -> None:
        result = run("query", "--session", "nope", "-r", "speaks")
        assert result.code == 1
        assert "HTTP 404" in result.err
        assert "Session not found" in result.err

    def test_fact_is_injected_before_querying(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)

        result = run(
            "query",
            "--km-id",
            KM_ID,
            "-r",
            "speaks",
            "-s",
            "Tom",
            "--fact",
            "Tom",
            "lives in",
            "Germany",
        )

        assert result.code == 0, result.err
        paths = [r.path.rsplit("/", 1)[-1] for r in fake.requests]
        assert paths == [KM_ID, "inject", "query"]
        assert fake.calls("/inject")[0].json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
        ]
        sid = next(iter(fake.sessions))
        assert fake.sessions[sid]["facts"] == fake.calls("/inject")[0].json

    def test_fact_with_certainty_and_multiple_facts(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        argv = ["query", "--km-id", KM_ID, "-r", "speaks"]
        argv += ["--fact", "Tom", "lives in", "Germany", "80"]
        argv += ["--fact", "Germany", "national language", "German"]
        assert run(*argv).code == 0
        assert fake.calls("/inject")[0].json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 80},
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "German",
                "certainty": 100,
            },
        ]

    def test_fact_with_wrong_arity_exits_2(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "--fact", "Tom", "speaks")
        assert result.code == 2
        assert "--fact needs SUBJECT RELATIONSHIP OBJECT [CERTAINTY]" in result.err
        assert fake.calls("/inject") == []

    def test_file_facts_injected(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        facts = [{"subject": "Tom", "relationship": "lives in", "object": "Germany"}]
        path = tmp_path / "facts.json"
        path.write_text(json.dumps(facts), encoding="utf-8")
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks", "--file", str(path)).code == 0
        assert fake.calls("/inject")[0].json == [{**facts[0], "certainty": 100}]

    def test_no_facts_means_no_inject_call(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "-r", "speaks").code == 0
        assert fake.calls("/inject") == []

    def test_draft_flag_forwarded_to_start(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "--draft", "-r", "speaks").code == 0
        assert fake.calls("/start/")[0].params == {"useDraft": "true"}

    def test_version_flag_forwarded_to_start(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        assert run("query", "--km-id", KM_ID, "--version", "2", "-r", "speaks").code == 0
        assert fake.calls("/start/")[0].params == {"version": "2"}


# --------------------------------------------------------------------------- inject


class TestInject:
    def test_multiple_fact_flags(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        argv = ["inject", "--session", SID]
        argv += ["--fact", "Tom", "lives in", "Germany"]
        argv += ["--fact", "Germany", "national language", "German", "90"]
        result = run(*argv)
        assert result.code == 0, result.err
        assert result.out == "injected 2 fact(s): OK\n"
        calls = fake.calls("/inject")
        assert len(calls) == 1
        assert calls[0].method == "POST"
        assert calls[0].path == f"/{SID}/inject"
        assert calls[0].json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100},
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "German",
                "certainty": 90,
            },
        ]

    def test_file_with_list(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        seed_session(fake)
        facts = [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100},
            {"subject": "Tom", "relationship": "age", "object": 42, "certainty": 70},
        ]
        path = tmp_path / "facts.json"
        path.write_text(json.dumps(facts), encoding="utf-8")

        result = run("inject", "--session", SID, "--file", str(path))

        assert result.code == 0, result.err
        assert result.out == "injected 2 fact(s): OK\n"
        assert fake.calls("/inject")[0].json == facts

    def test_file_with_single_object(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        seed_session(fake)
        path = tmp_path / "fact.json"
        path.write_text(
            json.dumps({"subject": "Tom", "relationship": "lives in", "object": "Germany"}),
            encoding="utf-8",
        )
        assert run("inject", "--session", SID, "--file", str(path)).code == 0
        assert fake.calls("/inject")[0].json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
        ]

    def test_fact_flags_come_before_file_facts(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        seed_session(fake)
        path = tmp_path / "facts.json"
        path.write_text(
            json.dumps([{"subject": "B", "relationship": "r", "object": "o"}]), encoding="utf-8"
        )
        result = run("inject", "--session", SID, "--file", str(path), "--fact", "A", "r", "o")
        assert result.code == 0
        assert result.out == "injected 2 fact(s): OK\n"
        assert [f["subject"] for f in fake.calls("/inject")[0].json] == ["A", "B"]

    def test_json_output(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        result = run("--json", "inject", "--session", SID, "--fact", "Tom", "lives in", "Germany")
        assert result.code == 0
        assert result.json == {"result": "OK"}

    def test_nothing_to_inject_exits_2(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("inject", "--session", SID)
        assert result.code == 2
        assert "nothing to inject" in result.err
        assert fake.requests == []

    def test_unknown_session_exits_1(self, run: Runner) -> None:
        result = run("inject", "--session", "nope", "--fact", "Tom", "lives in", "Germany")
        assert result.code == 1
        assert "Session not found" in result.err

    def test_missing_file_exits_1(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        result = run("inject", "--session", SID, "--file", str(tmp_path / "missing.json"))
        assert result.code == 1
        assert result.err.startswith("rainbird: ")
        assert "missing.json" in result.err
        assert fake.requests == []

    def test_large_file_is_sent_in_batches_of_250(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        seed_session(fake)
        facts = [{"subject": f"P{i}", "relationship": "likes", "object": "tea"} for i in range(300)]
        path = tmp_path / "many.json"
        path.write_text(json.dumps(facts), encoding="utf-8")

        result = run("inject", "--session", SID, "--file", str(path))

        assert result.code == 0, result.err
        assert result.out == "injected 300 fact(s): OK\n"
        batches = fake.calls("/inject")
        assert [len(b.json) for b in batches] == [250, 50]
        assert len(fake.sessions[SID]["facts"]) == 300

    def test_invalid_fact_in_file_is_a_usage_error(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        seed_session(fake)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"subject": "Tom", "object": "Germany"}]), encoding="utf-8")
        result = run("inject", "--session", SID, "--file", str(path))
        assert result.code == 2  # validated locally, before any request
        assert result.err.startswith("rainbird: invalid fact: ")
        assert "missing required field(s): relationship" in result.err
        assert fake.requests == []

    def test_malformed_json_file_is_a_usage_error(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        seed_session(fake)
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        result = run("inject", "--session", SID, "--file", str(path))
        assert result.code == 2
        assert result.out == ""
        assert result.err.startswith("rainbird: ")
        assert "broken.json" in result.err
        assert "invalid JSON" in result.err
        assert fake.requests == []

    def test_non_numeric_fact_certainty_is_a_usage_error(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        seed_session(fake)
        result = run("inject", "--session", SID, "--fact", "Tom", "lives in", "Germany", "high")
        assert result.code == 2
        assert result.err.startswith("rainbird: ")
        assert "--fact certainty must be a number 0-100, got 'high'" in result.err
        assert fake.requests == []

    def test_float_fact_certainty(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        argv = ["inject", "--session", SID]
        argv += ["--fact", "Tom", "lives in", "Germany", "80.5"]
        argv += ["--fact", "Tom", "likes", "tea", "90.0"]
        result = run(*argv)
        assert result.code == 0, result.err
        assert [f["certainty"] for f in fake.calls("/inject")[0].json] == [80.5, 90]

    def test_out_of_range_fact_certainty_is_a_usage_error(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        seed_session(fake)
        result = run("inject", "--session", SID, "--fact", "Tom", "lives in", "Germany", "150")
        assert result.code == 2  # validated locally, before any request
        assert "certainty must be between 0 and 100" in result.err
        assert fake.requests == []

    def test_query_validates_facts_before_starting_a_session(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        result = run("query", "--km-id", KM_ID, "-r", "speaks", "--fact", "Tom", "r", "o", "500")
        assert result.code == 2
        assert fake.calls("/start/") == []


# --------------------------------------------------------------------------- respond


class TestRespond:
    ANSWER = {
        "subject": "Germany",
        "relationship": "national language",
        "object": "German",
        "certainty": 100,
    }

    def test_sends_raw_answers_and_prints_next_question(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        seed_session(fake)
        fake.queue_question(QUESTION_FIRST_FORM)

        result = run("respond", "--session", SID, json.dumps([self.ANSWER]))

        assert result.code == 0, result.err
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{SID}/response"
        assert fake.last.json == answers_payload(self.ANSWER)
        assert result.out.splitlines() == [
            "Does Tom speak French?",
            "    yes/no; type '?' to skip; append @<certainty> e.g. German@80",
        ]

    def test_prints_result(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)
        result = run("respond", "--session", SID, json.dumps([self.ANSWER]))
        assert result.code == 0
        assert result.out == (
            f"Tom  speaks  German    certainty=75%\n    factID: {RESULT_FACT_ID}\n"
        )

    def test_json_output(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        result = run("--json", "respond", "--session", SID, json.dumps([self.ANSWER]))
        assert result.code == 0
        assert result.json == {"question": QUESTION_SECOND_SUBJECT, "sid": SID}

    def test_single_object_is_accepted(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)
        assert run("respond", "--session", SID, json.dumps(self.ANSWER)).code == 0
        assert fake.last.json == answers_payload(self.ANSWER)

    def test_first_form_answer_passthrough(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)
        answer = {"subject": "Tom", "relationship": "speaks", "object": "French", "answer": "yes"}
        assert run("respond", "--session", SID, json.dumps([answer])).code == 0
        assert fake.last.json == answers_payload({**answer, "certainty": 100})

    def test_invalid_answer_shape_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        result = run("respond", "--session", SID, json.dumps([{"subject": "Tom"}]))
        assert result.code == 1
        assert "missing required field(s)" in result.err
        assert fake.requests == []

    def test_empty_list_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        result = run("respond", "--session", SID, "[]")
        assert result.code == 1
        assert "must not be empty" in result.err

    def test_malformed_json_is_a_usage_error(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        result = run("respond", "--session", SID, "{oops")
        assert result.code == 2
        assert result.out == ""
        assert result.err.startswith("rainbird: answers must be valid JSON: ")
        assert fake.requests == []


# --------------------------------------------------------------------------- undo


class TestUndo:
    def test_prints_previous_question(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_question(QUESTION_SECOND_OBJECT)

        result = run("undo", "--session", SID)

        assert result.code == 0, result.err
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{SID}/undo"
        assert fake.last.json == {}
        assert result.out.splitlines() == [
            "What is the national language of Germany?",
            "    options: English, French",
            "    append @<certainty> e.g. German@80",
        ]

    def test_plural_question_flags(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        result = run("undo", "--session", SID)
        assert result.out.splitlines() == [
            "Who speaks French?",
            "    options: Tom, Ana",
            "    multiple values allowed (comma-separated); type '?' to skip",
        ]

    def test_prints_result(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_result(RESULT_ITEM)
        result = run("undo", "--session", SID)
        assert result.code == 0
        assert "Tom  speaks  German    certainty=75%" in result.out

    def test_json_output(self, run: Runner, fake: FakeRainbird) -> None:
        seed_session(fake)
        fake.queue_question(QUESTION_FIRST_FORM)
        result = run("--json", "undo", "--session", SID)
        assert result.code == 0
        assert result.json == {"question": QUESTION_FIRST_FORM, "sid": SID}

    def test_unknown_session_exits_1(self, run: Runner) -> None:
        result = run("undo", "--session", "nope")
        assert result.code == 1
        assert "Session not found" in result.err


# --------------------------------------------------------------------------- evidence


class TestEvidence:
    def test_single_node_text(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID)

        assert result.code == 0, result.err
        assert len(fake.requests) == 1
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/evidence/{ROOT_FACT_ID}/{SID}"
        assert "x-evidence-key" not in fake.last.headers
        lines = result.out.splitlines()
        assert lines[0] == f"Tom speaks German (75%)  [rule]  {ROOT_FACT_ID}"
        assert "  - Tom lives in Germany (certainty 100, impact 37.5)" in lines
        assert "  - Germany national language German (certainty 100, impact 37.5)" in lines
        expr = [line for line in lines if line.startswith("  - expr ")]
        assert len(expr) == 1
        assert "countRelationshipInstances" in expr[0]
        assert expr[0].endswith("-> '%N' met=True")

    def test_leaf_node_has_no_conditions(self, run: Runner) -> None:
        leaf = LEAF_FACT_IDS[0]
        result = run("evidence", "--session", SID, "--fact-id", leaf)
        assert result.code == 0
        assert result.out == f"Tom lives in Germany (100%)  [injection]  {leaf}\n"

    def test_tree_fetches_every_node(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--tree")

        assert result.code == 0, result.err
        fetched = [r.path.split("/")[3] for r in fake.calls("/analysis/evidence/")]
        assert fetched == [ROOT_FACT_ID, *LEAF_FACT_IDS]
        lines = result.out.splitlines()
        fact_lines = [line for line in lines if not line.lstrip().startswith("- ")]
        assert [line.strip() for line in fact_lines] == [
            f"Tom speaks German (75%)  [rule]  {ROOT_FACT_ID}",
            f"Tom lives in Germany (100%)  [injection]  {LEAF_FACT_IDS[0]}",
            f"Germany national language German (100%)  [answer]  {LEAF_FACT_IDS[1]}",
        ]
        # Nested: root at column 0, its child evidence nodes indented two spaces.
        assert fact_lines[0] == fact_lines[0].lstrip()
        assert all(line.startswith("  ") and not line.startswith("   ") for line in fact_lines[1:])
        # Conditions resolved to fetched nodes are not repeated as "- ..." lines.
        assert not any("(certainty 100, impact 37.5)" in line for line in lines)
        expr = [line for line in lines if line.lstrip().startswith("- expr ")]
        assert len(expr) == 1
        assert expr[0].startswith("  - expr ")  # beneath the root, same indent as its children

    def test_tree_nests_deeper_levels(self, run: Runner, fake: FakeRainbird) -> None:
        deep_id = "WA:KF:deep0000000000000000000000000000000000000000000000000000000000000000"
        leaf0 = LEAF_FACT_IDS[0]
        fake.evidence[deep_id] = leaf_evidence(
            deep_id, "Germany", "is in", "Europe", "knowledgemap"
        )
        fake.evidence[leaf0] = {
            **fake.evidence[leaf0],
            "source": "rule",
            "rule": {
                "bindings": {},
                "conditions": [
                    {
                        "certainty": 100,
                        "factID": deep_id,
                        "impact": 50,
                        "object": "Europe",
                        "objectType": "string",
                        "relationship": "is in",
                        "salience": 100,
                        "subject": "Germany",
                    },
                    {
                        "certainty": 100,
                        "impact": 50,
                        "object": "Berlin",
                        "objectType": "string",
                        "relationship": "capital",
                        "salience": 100,
                        "subject": "Germany",
                    },
                ],
                "ruleMaxCertainty": 100,
            },
        }

        result = run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--tree")

        assert result.code == 0, result.err
        assert result.out.splitlines() == [
            f"Tom speaks German (75%)  [rule]  {ROOT_FACT_ID}",
            f"  Tom lives in Germany (100%)  [rule]  {leaf0}",
            f"    Germany is in Europe (100%)  [knowledgemap]  {deep_id}",
            "    - Germany capital Berlin (certainty 100, impact 50)",
            f"  Germany national language German (100%)  [answer]  {LEAF_FACT_IDS[1]}",
            "  - expr \"countRelationshipInstances( 'Tom', 'lives in', *)\" -> '%N' met=True",
        ]

    def test_tree_max_depth_zero_fetches_only_root(self, run: Runner, fake: FakeRainbird) -> None:
        result = run(
            "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--tree", "--max-depth", "0"
        )
        assert result.code == 0
        assert len(fake.calls("/analysis/evidence/")) == 1
        # Unfetched children are shown inline as conditions beneath the root.
        lines = result.out.splitlines()
        assert lines[0] == f"Tom speaks German (75%)  [rule]  {ROOT_FACT_ID}"
        assert "  - Tom lives in Germany (certainty 100, impact 37.5)" in lines
        assert "  - Germany national language German (certainty 100, impact 37.5)" in lines

    def test_url_flag_prints_studio_link(self, run: Runner) -> None:
        result = run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--url")
        assert result.code == 0
        assert result.out.splitlines()[-1] == (
            f"https://fake.rainbird.ai/evidence?id={ROOT_FACT_ID}&api={BASE_URL}&sid={SID}"
        )

    def test_url_follows_base_url(self, run: Runner) -> None:
        result = run(
            "--base-url",
            "https://enterprise-api.rainbird.ai",
            "evidence",
            "--session",
            SID,
            "--fact-id",
            ROOT_FACT_ID,
            "--url",
        )
        assert result.code == 0
        assert result.out.splitlines()[-1] == (
            f"https://enterprise.rainbird.ai/evidence?id={ROOT_FACT_ID}"
            f"&api=https://enterprise-api.rainbird.ai&sid={SID}"
        )

    def test_json_single(self, run: Runner) -> None:
        result = run("--json", "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID)
        assert result.code == 0
        assert result.json == EVIDENCE_ROOT

    def test_json_tree_is_a_list(self, run: Runner) -> None:
        result = run("--json", "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--tree")
        assert result.code == 0
        data = result.json
        assert isinstance(data, list)
        assert [node["factID"] for node in data] == [ROOT_FACT_ID, *LEAF_FACT_IDS]
        assert data[0] == EVIDENCE_ROOT

    def test_json_with_url_appends_link_line(self, run: Runner) -> None:
        result = run("--json", "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID, "--url")
        assert result.code == 0
        body, link = result.out.rstrip("\n").rsplit("\n", 1)
        assert json.loads(body) == EVIDENCE_ROOT
        assert link.startswith("https://fake.rainbird.ai/evidence?id=")

    def test_evidence_key_header(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "ev-secret"
        result = run(
            "--evidence-key", "ev-secret", "evidence", "--session", SID, "--fact-id", ROOT_FACT_ID
        )
        assert result.code == 0, result.err
        assert fake.last.headers["x-evidence-key"] == "ev-secret"

    def test_evidence_key_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake.require_evidence_key = "ev-secret"
        monkeypatch.setenv("RAINBIRD_EVIDENCE_KEY", "ev-secret")
        assert run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID).code == 0
        assert fake.last.headers["x-evidence-key"] == "ev-secret"

    def test_missing_evidence_key_is_403(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "ev-secret"
        result = run("evidence", "--session", SID, "--fact-id", ROOT_FACT_ID)
        assert result.code == 1
        assert "HTTP 403" in result.err
        assert "provide x-evidence-key" in result.err

    def test_unknown_fact_is_404(self, run: Runner) -> None:
        result = run("evidence", "--session", SID, "--fact-id", "WA:RF:unknown")
        assert result.code == 1
        assert "HTTP 404" in result.err
        assert "Fact not found" in result.err

    def test_fact_id_is_required(self, run: Runner) -> None:
        result = run("evidence", "--session", SID)
        assert result.code == 2
        assert "--fact-id" in result.err


# --------------------------------------------------------------------------- explain


class TestExplain:
    def test_prints_explanation(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("explain", "--session", SID, "--fact-id", ROOT_FACT_ID)

        assert result.code == 0, result.err
        assert result.out == EXPLAIN_OK["explanation"] + "\n"
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/explain"
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.json == {"language": "en", "sessionID": SID, "factID": ROOT_FACT_ID}

    def test_language_and_instructions(self, run: Runner, fake: FakeRainbird) -> None:
        result = run(
            "explain",
            "--session",
            SID,
            "--fact-id",
            ROOT_FACT_ID,
            "--language",
            "de",
            "--instructions",
            "one sentence",
        )
        assert result.code == 0
        assert fake.last.json == {
            "language": "de",
            "sessionID": SID,
            "factID": ROOT_FACT_ID,
            "instructions": "one sentence",
        }

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "explain", "--session", SID, "--fact-id", ROOT_FACT_ID)
        assert result.code == 0
        assert result.json == EXPLAIN_OK

    def test_error_body_exits_1(self, run: Runner) -> None:
        result = run("explain", "--session", SID, "--fact-id", "not-a-fact")
        assert result.code == 1
        assert result.out == ""
        assert "Fact not found" in result.err


# --------------------------------------------------------------------------- interact


class TestInteract:
    def test_question_response(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_QUESTION)

        result = run("interact", "--km-id", KM_ID, "Can Ben claim?")

        assert result.code == 0, result.err
        sid = next(iter(fake.sessions))
        assert fake.calls("/start/")[0].path == f"/start/{KM_ID}"
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/interact"
        assert fake.last.headers["version"] == "v1"
        assert fake.last.json == {"sessionID": sid, "userPrompt": "Can Ben claim?"}
        assert result.out.splitlines() == [
            f"session {sid}",
            "[1] Which policy does Ben hold?",
            "    options: Gold, Silver",
            "    append @<certainty> e.g. German@80",
            "",
            f'Answer with: rainbird interact --session {sid} "<your answer>"',
        ]

    def test_result_response(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_RESULT)
        result = run("interact", "--km-id", KM_ID, "Ben holds Gold, can he claim?")
        assert result.code == 0, result.err
        assert result.out.splitlines()[1:] == ["Ben can claim repair (100%)"]

    def test_error_response_exits_1_with_suggested_reply(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.append(INTERACT_ERROR)
        result = run("interact", "--km-id", KM_ID, "hello there")
        assert result.code == 1
        assert result.err == (
            "rainbird: I couldn't work out what you want to know. Could you rephrase?\n"
        )

    def test_error_without_suggestion_uses_message(self, run: Runner, fake: FakeRainbird) -> None:
        body = {**INTERACT_ERROR, "error": {"code": 1001, "message": "Could not detect a query"}}
        fake.interact_bodies.append(body)
        result = run("interact", "--km-id", KM_ID, "hello")
        assert result.code == 1
        assert "Could not detect a query" in result.err

    def test_json_output_includes_session_id(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_QUESTION)
        result = run("--json", "interact", "--km-id", KM_ID, "Can Ben claim?")
        assert result.code == 0
        data = result.json
        sid = next(iter(fake.sessions))
        assert data["session_id"] == sid
        assert data["responseType"] == "question"
        assert data["questions"] == INTERACT_QUESTION["questions"]

    def test_json_error_is_returned_not_raised(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_ERROR)
        result = run("--json", "interact", "--km-id", KM_ID, "hello")
        assert result.code == 0
        assert result.err == ""
        assert result.json["responseType"] == "error"
        assert result.json["error"] == INTERACT_ERROR["error"]

    def test_session_flag_reuses_session(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_RESULT)
        result = run("interact", "--session", SID, "Gold")
        assert result.code == 0, result.err
        assert fake.calls("/start/") == []
        assert fake.last.json == {"sessionID": SID, "userPrompt": "Gold"}
        assert result.out == "Ben can claim repair (100%)\n"

    def test_prompt_may_precede_options(self, run: Runner, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_RESULT)
        assert run("interact", "Gold", "--session", SID).code == 0
        assert fake.last.json["userPrompt"] == "Gold"

    def test_km_id_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", KM_ID)
        assert run("interact", "Can Ben claim?").code == 0
        assert fake.calls("/start/")[0].path == f"/start/{KM_ID}"

    def test_missing_km_id_and_session_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("interact", "Can Ben claim?")
        assert result.code == 1
        assert "km_id is required" in result.err
        assert fake.requests == []


# --------------------------------------------------------------------------- facts / session / interactions


class TestFacts:
    def test_text_output(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("facts", "--session", SID)

        assert result.code == 0, result.err
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/session/{SID}"
        assert fake.last.params == {"filter": "facts"}
        assert result.out.splitlines() == [
            "global (1)",
            "  France national language French (100%)  [knowledgemap]",
            "local (1)",
            "  Tom lives in Germany (100%)  [injection]",
        ]

    def test_relationships_filter(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("facts", "--session", SID, "--relationships", "lives in", "speaks")
        assert result.code == 0
        assert fake.last.params == {"filter": "facts", "relationships": "lives in,speaks"}
        assert "global" not in result.out
        assert result.out.splitlines()[0] == "local (1)"

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "facts", "--session", SID)
        assert result.code == 0
        assert result.json == SESSION_FACTS


class TestSessionInfo:
    def test_text_output(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("session", "--session", SID)
        assert result.code == 0, result.err
        assert fake.last.path == f"/analysis/session/{SID}"
        assert fake.last.params == {"filter": "version"}
        assert result.out == (
            f"knowledge map: Languages ({KM_ID})\nversion: Version3 ({KM_VERSION_ID})\n"
        )

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "session", "--session", SID)
        assert result.code == 0
        assert result.json == SESSION_VERSION


class TestInteractions:
    def test_text_output(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("interactions", "--session", SID)

        assert result.code == 0, result.err
        assert fake.last.path == f"/analysis/interactions/{SID}"
        assert fake.last.params == {"format": "json"}
        lines = result.out.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("2025-07-10T10:06:27.780000+00:00  start      ")
        assert json.loads(lines[0].split("  start      ", 1)[1]) == INTERACTIONS[0]["values"]
        assert lines[1].split()[1] == "query"
        assert lines[2].split()[1] == "result"

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "interactions", "--session", SID)
        assert result.code == 0
        assert result.json == INTERACTIONS

    def test_csv_output(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("interactions", "--session", SID, "--format", "csv")
        assert result.code == 0
        assert fake.last.params == {"format": "csv"}
        assert result.out == "event,created\nstart,2025-07-10T10:06:27.780Z\n\n"

    def test_csv_ignores_json_flag(self, run: Runner) -> None:
        result = run("--json", "interactions", "--session", SID, "--format", "csv")
        assert result.code == 0
        assert result.out.startswith("event,created\n")

    def test_interaction_key_header(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "ix-secret"
        result = run("--interaction-key", "ix-secret", "interactions", "--session", SID)
        assert result.code == 0, result.err
        assert fake.last.headers["x-interaction-key"] == "ix-secret"

    def test_missing_interaction_key_is_403(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "ix-secret"
        result = run("interactions", "--session", SID)
        assert result.code == 1
        assert "HTTP 403" in result.err
        assert "Interaction log is secured" in result.err


# --------------------------------------------------------------------------- map (undocumented endpoints)


class TestMapGet:
    def test_prints_rblang(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("map", "get", KM_ID)
        assert result.code == 0, result.err
        assert result.out == RBLANG + "\n"
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/file/{KM_ID}"
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_out_writes_file(self, run: Runner, tmp_path: Path) -> None:
        target = tmp_path / "map.rbl"
        result = run("map", "get", KM_ID, "--out", str(target))
        assert result.code == 0, result.err
        assert target.read_text(encoding="utf-8") == RBLANG
        assert result.out == f"wrote {len(RBLANG)} chars of RBLang to {target}\n"

    def test_json_output(self, run: Runner) -> None:
        result = run("--json", "map", "get", KM_ID)
        assert result.code == 0
        assert result.json == MAP_SOURCE

    def test_json_with_out_writes_file_and_prints_json_only(
        self, run: Runner, tmp_path: Path
    ) -> None:
        target = tmp_path / "map.rbl"
        result = run("--json", "map", "get", KM_ID, "--out", str(target))
        assert result.code == 0
        assert target.read_text(encoding="utf-8") == RBLANG
        assert result.json == MAP_SOURCE  # no "wrote ..." line mixed into the JSON

    def test_unknown_map_is_404(self, run: Runner) -> None:
        result = run("map", "get", "00000000-0000-0000-0000-000000000000")
        assert result.code == 1
        assert "HTTP 404" in result.err
        assert "Knowledge map not found" in result.err

    def test_out_in_missing_directory_exits_1(self, run: Runner, tmp_path: Path) -> None:
        result = run("map", "get", KM_ID, "--out", str(tmp_path / "nope" / "map.rbl"))
        assert result.code == 1
        assert result.err.startswith("rainbird: ")


class TestMapCreate:
    def test_from_file_prints_km_id(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        path = tmp_path / "map.rbl"
        path.write_text(RBLANG, encoding="utf-8")

        result = run("map", "create", "--name", "My map", "--file", str(path))

        assert result.code == 0, result.err
        assert result.out == NEW_KM_ID + "\n"
        assert fake.last.method == "POST"
        assert fake.last.path == "/maps"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["version"] == "v1"
        assert fake.last.json == {"rblang": RBLANG, "name": "My map"}

    def test_dash_reads_stdin(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(RBLANG))
        result = run("map", "create", "--name", "From stdin", "--file", "-")
        assert result.code == 0, result.err
        assert result.out == NEW_KM_ID + "\n"
        assert fake.last.json == {"rblang": RBLANG, "name": "From stdin"}

    def test_description_is_sent_when_given(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        path = tmp_path / "map.rbl"
        path.write_text(RBLANG, encoding="utf-8")
        result = run(
            "map",
            "create",
            "--name",
            "My map",
            "--description",
            "Who speaks what",
            "--file",
            str(path),
        )
        assert result.code == 0, result.err
        assert result.out == NEW_KM_ID + "\n"
        assert fake.last.json == {
            "rblang": RBLANG,
            "name": "My map",
            "description": "Who speaks what",
        }

    def test_json_output(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        path = tmp_path / "map.rbl"
        path.write_text(RBLANG, encoding="utf-8")
        result = run("--json", "map", "create", "--name", "My map", "--file", str(path))
        assert result.code == 0
        assert result.json == fake.map_create_body

    def test_rblang_rejected_with_http_200_exits_1(
        self, run: Runner, fake: FakeRainbird, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.rbl"
        path.write_text("<kb>not rblang</kb>", encoding="utf-8")
        result = run("map", "create", "--name", "Bad", "--file", str(path))
        assert result.code == 1
        assert result.out == ""
        assert result.err == "rainbird: RBLang parse error: missing rbl:kb root\n"
        assert fake.last.path == "/maps"

    def test_empty_file_exits_1(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        path = tmp_path / "empty.rbl"
        path.write_text("", encoding="utf-8")
        result = run("map", "create", "--name", "Empty", "--file", str(path))
        assert result.code == 1
        assert "rblang must not be empty" in result.err
        assert fake.requests == []

    def test_missing_file_exits_1(self, run: Runner, fake: FakeRainbird, tmp_path: Path) -> None:
        result = run("map", "create", "--name", "x", "--file", str(tmp_path / "missing.rbl"))
        assert result.code == 1
        assert "missing.rbl" in result.err
        assert fake.requests == []

    def test_name_is_required(self, run: Runner, tmp_path: Path) -> None:
        result = run("map", "create", "--file", str(tmp_path / "map.rbl"))
        assert result.code == 2
        assert "--name" in result.err


# --------------------------------------------------------------------------- errors, auth, global options


class TestErrorsAndAuth:
    def test_api_error_exits_1_with_message_on_stderr(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.fail_next(401, {"err": ["Your API key is invalid"]})
        result = run("start", "--km-id", KM_ID)
        assert result.code == 1
        assert result.out == ""
        assert result.err == (
            f"rainbird: HTTP 401 calling GET {BASE_URL}/start/{KM_ID}: Your API key is invalid\n"
        )

    def test_api_error_with_error_key_shape(self, run: Runner, fake: FakeRainbird) -> None:
        fake.fail_next(403, {"error": "Not permitted"})
        result = run("version")
        assert result.code == 1
        assert "HTTP 403" in result.err
        assert "Not permitted" in result.err

    def test_api_error_without_body_uses_default_message(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        fake.fail_next(404)
        result = run("session", "--session", SID)
        assert result.code == 1
        assert "HTTP 404" in result.err
        assert "could not be found" in result.err

    def test_missing_api_key_exits_1_with_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        # No env (autouse _clean_env), no --api-key, real ``_client``: fails before any request.
        # (``version`` needs no key, so use a command that does.)
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["start", "--km-id", KM_ID])
        assert excinfo.value.code == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert err.startswith("rainbird: ")
        assert "API key is required" in err
        assert "RAINBIRD_API_KEY" in err
        assert "--api-key" in err or "api_key" in err

    def test_api_key_flag_is_sent(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("--api-key", API_KEY, "start", "--km-id", KM_ID)
        assert result.code == 0
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_wrong_api_key_flag_is_rejected(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("--api-key", "wrong", "start", "--km-id", KM_ID)
        assert result.code == 1
        assert fake.last.headers["x-api-key"] == "wrong"
        assert "HTTP 401" in result.err

    def test_api_key_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", API_KEY)
        assert run("start", "--km-id", KM_ID).code == 0
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_engine_flag_sets_header(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("--engine", "next", "version").code == 0
        assert fake.last.headers["x-rainbird-engine"] == "next"

    def test_no_engine_header_by_default(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("version").code == 0
        assert "x-rainbird-engine" not in fake.last.headers

    def test_user_agent_identifies_sdk(self, run: Runner, fake: FakeRainbird) -> None:
        assert run("version").code == 0
        assert fake.last.headers["user-agent"].startswith(f"rainbird-ai-python/{__version__}")

    def test_keyboard_interrupt_returns_130(
        self, run: Runner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(args: argparse.Namespace) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "cmd_version", interrupted)
        result = run("version")
        assert result.code == 130
        assert result.out == ""
        assert result.err == "\n"

    def test_success_returns_0_from_main(self, run: Runner) -> None:
        # main() returns rather than raising SystemExit on success
        assert cli.main(["version"]) == 0


# --------------------------------------------------------------------------- decide


class TestDecide:
    def test_text_output_for_a_result(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run(
            "decide",
            "--km-id",
            KM_ID,
            "-r",
            "speaks",
            "-s",
            "Tom",
            "--fact",
            "Tom",
            "lives in",
            "Germany",
        )
        assert result.code == 0, result.err
        assert result.err == ""
        sid = next(iter(fake.sessions))
        lines = result.out.splitlines()
        assert lines[0] == "status     result"
        assert lines[1] == "decision   German"
        assert lines[2] == "certainty  75%"
        assert lines[3] == f"session    {sid}"
        assert lines[4] == f"fact_id    {RESULT_FACT_ID}"
        assert lines[5].startswith("evidence   https://fake.rainbird.ai/evidence?id=")
        assert len(lines) == 6
        assert [r.path.rsplit("/", 1)[-1] for r in fake.requests] == [KM_ID, "inject", "query"]
        assert fake.calls("/inject")[0].json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
        ]
        assert fake.calls("/query")[0].json == {"relationship": "speaks", "subject": "Tom"}

    def test_json_output(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run("--json", "decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 0, result.err
        data = result.json
        sid = next(iter(fake.sessions))
        assert data["status"] == "result"
        assert data["decision"] == "German"
        assert data["certainty"] == 75
        assert data["session_id"] == sid
        assert data["fact_id"] == RESULT_FACT_ID
        assert data["question"] is None
        assert data["results"] == [RESULT_ITEM]
        assert data["evidence"] == []
        assert data["km_id"] == KM_ID

    def test_question_prints_decision_and_exits_3(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 3
        sid = next(iter(fake.sessions))
        assert result.out.splitlines()[:2] == ["status     question", f"session    {sid}"]
        assert "What is the national language of Germany?" in result.out
        assert f"rainbird query --session {sid} -r speaks" in result.err
        assert fake.calls("/response") == []

    def test_question_json_exits_3(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        result = run("decide", "--json", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 3
        data = result.json
        assert data["status"] == "question"
        assert data["question"] == QUESTION_SECOND_OBJECT
        assert data["decision"] is None

    def test_no_result(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result()
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 0
        lines = result.out.splitlines()
        assert lines[0] == "status     no_result"
        assert lines[1] == "decision   -"
        assert lines[2] == "certainty  -"
        assert lines[4] == "fact_id    -"
        assert not any(line.startswith("evidence") for line in lines)

    def test_both_sides_prints_boolean(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "-o", "German")
        assert result.code == 0
        assert "decision   True" in result.out
        assert fake.calls("/query")[0].json == {
            "relationship": "speaks",
            "subject": "Tom",
            "object": "German",
        }

    def test_several_results_are_listed(self, run: Runner, fake: FakeRainbird) -> None:
        low = dict(RESULT_ITEM, object="English", certainty=40, factID="WA:RF:low")
        fake.queue_result(low, RESULT_ITEM)
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom")
        assert result.code == 0
        assert "decision   German" in result.out
        assert "results:" in result.out
        assert "Tom  speaks  English    certainty=40%" in result.out
        assert "Tom  speaks  German    certainty=75%" in result.out

    def test_evidence_tree(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--evidence", "tree")
        assert result.code == 0, result.err
        assert "evidence tree:" in result.out
        assert f"Tom speaks German (75%)  [rule]  {ROOT_FACT_ID}" in result.out
        assert "  Tom lives in Germany (100%)  [injection]" in result.out
        assert len(fake.calls("/analysis/evidence/")) == 3

    def test_evidence_root_json_and_max_depth(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run(
            "--json", "decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--evidence", "root"
        )
        assert result.code == 0
        assert result.json["evidence"] == [EVIDENCE_ROOT]
        fake.queue_result(RESULT_ITEM)
        result = run(
            "--json",
            "decide",
            "--km-id",
            KM_ID,
            "-r",
            "speaks",
            "-s",
            "Tom",
            "--evidence",
            "tree",
            "--max-depth",
            "0",
        )
        assert result.code == 0
        assert [n["factID"] for n in result.json["evidence"]] == [ROOT_FACT_ID]

    def test_evidence_key_is_forwarded(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "secret"
        fake.queue_result(RESULT_ITEM)
        result = run(
            "decide",
            "--km-id",
            KM_ID,
            "-r",
            "speaks",
            "-s",
            "Tom",
            "--evidence",
            "root",
            "--evidence-key",
            "secret",
        )
        assert result.code == 0, result.err
        assert fake.last.headers["x-evidence-key"] == "secret"

    def test_missing_evidence_key_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "secret"
        fake.queue_result(RESULT_ITEM)
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--evidence", "root")
        assert result.code == 1
        assert "403" in result.err

    def test_invalid_evidence_choice_is_a_usage_error(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--evidence", "all")
        assert result.code == 2
        assert fake.requests == []

    def test_relationship_is_required(self, run: Runner) -> None:
        result = run("decide", "--km-id", KM_ID, "-s", "Tom")
        assert result.code == 2

    def test_bad_fact_is_a_usage_error_before_any_request(
        self, run: Runner, fake: FakeRainbird
    ) -> None:
        result = run("decide", "--km-id", KM_ID, "-r", "speaks", "--fact", "Tom", "lives in")
        assert result.code == 2
        assert fake.requests == []

    def test_start_options_are_forwarded(self, run: Runner, fake: FakeRainbird) -> None:
        fake.queue_result(RESULT_ITEM)
        result = run(
            "decide", "--km-id", KM_ID, "-r", "speaks", "-s", "Tom", "--draft", "--context-id", "c1"
        )
        assert result.code == 0
        sid = next(iter(fake.sessions))
        assert fake.sessions[sid]["params"] == {"useDraft": "true", "contextid": "c1"}

    def test_km_id_from_environment(
        self, run: Runner, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", KM_ID)
        fake.queue_result(RESULT_ITEM)
        assert run("decide", "-r", "speaks", "-s", "Tom").code == 0
        assert fake.requests[0].path == f"/start/{KM_ID}"

    def test_missing_km_id_exits_1(self, run: Runner, fake: FakeRainbird) -> None:
        result = run("decide", "-r", "speaks", "-s", "Tom")
        assert result.code == 1
        assert "RAINBIRD_KM_ID" in result.err
        assert fake.requests == []

    def test_help_mentions_one_shot(self, run: Runner) -> None:
        result = run("--help")
        assert result.code == 0
        assert "decide" in result.out
