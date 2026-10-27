"""``rainbird`` command-line interface.

Examples::

    export RAINBIRD_API_KEY=... RAINBIRD_KM_ID=...
    rainbird version                                            # no API key needed
    rainbird query --relationship speaks --subject Tom          # interactive Q&A
    rainbird query --relationship speaks --subject Tom --answers German --json
    rainbird decide --relationship speaks --subject Tom --fact Tom "lives in" Germany
    rainbird inject --session SID --fact Tom "lives in" Germany
    rainbird evidence --session SID --fact-id WA:RF:... --tree
    rainbird interact --km-id KM "Can Ben claim?"
    rainbird map get KM_ID --out map.rbl
    rainbird map create --name "My map" --file map.rbl
    rainbird map validate --file map.rbl
    rainbird map create-tests KM_ID --file tests.json
    rainbird map run-tests KM_ID

Global options (``--api-key``, ``--base-url``, ``--json``, ...) may be given before
or after the subcommand.

Exit codes: 0 success; 1 API, SDK or file error (message on stderr); 2 usage error;
3 a question is still pending and no input is available (``--no-input``,
non-interactive stdin or Ctrl-D), or a test run has not finished (``map test-run``,
``map run-tests`` after ``--max-wait``); 4 a check failed (``map run-tests`` or
``map test-run`` found failing tests, ``map validate`` found an RBLang error);
130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, NoReturn, TextIO

from ._answers import SKIP, normalize_facts
from ._http import ENV_API_KEY
from ._ops import suite_payload
from ._version import __version__
from .client import Rainbird, Session, engine_version
from .errors import (
    BadRequestError,
    InvalidAnswerError,
    InvalidFactError,
    RainbirdConfigurationError,
    RainbirdError,
    TestRunFailedError,
    TestRunTimeoutError,
)
from .models import (
    Condition,
    Decision,
    DecisionStatus,
    Environment,
    Evidence,
    QueryResponse,
    Question,
    QuestionResponse,
    ResultResponse,
    TestRun,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_PENDING = 3
EXIT_FAILED = 4
_SKIP_TOKENS = {"?", "skip", "SKIP"}


def _die(message: str, code: int = EXIT_ERROR) -> NoReturn:
    print(f"rainbird: {message}", file=sys.stderr)
    raise SystemExit(code)


def _source_name(source: Any) -> str:
    return str(source.value) if hasattr(source, "value") else str(source or "-")


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "raw") and isinstance(value.raw, dict):
        return value.raw
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _emit_json(value: Any) -> None:
    print(json.dumps(_to_jsonable(value), indent=2, ensure_ascii=False, default=str))


def _client(args: argparse.Namespace) -> Rainbird:
    return Rainbird(
        api_key=args.api_key,
        base_url=args.base_url,
        engine=args.engine,
        evidence_key=args.evidence_key,
        interaction_key=args.interaction_key,
        timeout=args.timeout,
    )


def _print_question(
    question: Question, index: int | None = None, *, file: TextIO | None = None
) -> None:
    """Show a question; ``file=None`` means the current ``sys.stdout``."""
    prefix = f"[{index}] " if index is not None else ""
    print(f"{prefix}{question.prompt}", file=file)
    if question.options:
        print("    options: " + ", ".join(str(o) for o in question.options), file=file)
    flags = []
    if question.is_first_form:
        flags.append("yes/no")
    if question.plural:
        flags.append("multiple values allowed (comma-separated)")
    if question.allow_unknown:
        flags.append("type '?' to skip")
    if question.allow_cf:
        flags.append("append @<certainty> e.g. German@80")
    if flags:
        print("    " + "; ".join(flags), file=file)


def _numeric_suffix(text: str) -> tuple[str, float | None]:
    """Split ``"German@80"`` into ``("German", 80.0)``; a non-numeric suffix is not split."""
    head, sep, tail = text.rpartition("@")
    suffix = tail.strip()
    if not sep or not suffix or "_" in suffix or suffix.lower() in {"nan", "inf", "infinity"}:
        return text, None
    try:
        return head.strip(), float(suffix)
    except ValueError:
        return text, None


def _parse_answer_text(text: str, question: Question) -> tuple[Any, int | float]:
    """Parse a typed answer.

    ``"German"`` -> ("German", 100); ``"a, b@70"`` -> (["a", "b"], 70); ``"?"`` -> (SKIP, 100).
    An ``@`` only counts as the certainty separator when what follows parses as a
    number, so values such as e-mail addresses pass through untouched; a numeric
    suffix outside 0-100 is an error rather than part of the value. Certainty applies
    to the whole answer, so ``"a@70, b@80"`` is rejected.
    """
    text = text.strip()
    if text in _SKIP_TOKENS:
        return SKIP, 100
    if not text:
        hint = " (type '?' to skip)" if question.allow_unknown else ""
        raise InvalidAnswerError(f"answer must not be empty{hint}")
    certainty: int | float = 100
    head, parsed = _numeric_suffix(text)
    if parsed is not None:
        if not math.isfinite(parsed) or not 0 <= parsed <= 100:
            raise InvalidAnswerError(
                f"invalid certainty {text.rpartition('@')[2]!r}; use a number 0-100"
            )
        certainty = int(parsed) if parsed.is_integer() else parsed
        text = head
        if text in _SKIP_TOKENS:  # "?@50": skipping has no certainty
            return SKIP, 100
        if not text:
            raise InvalidAnswerError("answer must not be empty")
    if question.plural and "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        for part in parts:
            if _numeric_suffix(part)[1] is not None:
                raise InvalidAnswerError(
                    "certainty applies to the whole answer: use 'a, b@70' rather than 'a@70, b@80'"
                )
        return parts, certainty
    return text, certainty


def _print_results(response: ResultResponse) -> None:
    if not response.results:
        print("No result: the engine could not infer anything.")
        return
    for item in response.results:
        cert = f"{item.certainty}%" if item.certainty is not None else "-"
        print(f"{item.subject}  {item.relationship}  {item.object}    certainty={cert}")
        if item.fact_id:
            print(f"    factID: {item.fact_id}")


def _pending_exit(
    session: Session,
    response: QuestionResponse,
    args: argparse.Namespace,
    *,
    show_question: bool = True,
) -> NoReturn:
    if args.json:
        _emit_json({"session_id": session.id, **response.raw})
    elif show_question:
        _print_question(response.question)
    print(
        f"rainbird: session {session.id} is waiting for an answer; resume with "
        f"--session {session.id}",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_PENDING)


def _drive(session: Session, response: QueryResponse, args: argparse.Namespace) -> QueryResponse:
    """Answer questions from ``--answers`` first, then interactively, until a result.

    With ``--json`` the interactive prompts go to stderr so stdout stays a single JSON
    document.
    """
    scripted = list(args.answers or [])
    interactive = not args.no_input and sys.stdin is not None and sys.stdin.isatty()
    prompt_stream = sys.stderr if args.json else None
    while isinstance(response, QuestionResponse):
        question = response.question
        from_script = bool(scripted)
        if from_script:
            raw = scripted.pop(0)
            if not args.json:
                print(f"Q: {question.prompt}\nA: {raw}")
        elif not interactive:
            _pending_exit(session, response, args)
        else:
            _print_question(question, file=prompt_stream)
            try:
                if args.json:
                    sys.stderr.write("> ")
                    sys.stderr.flush()
                    raw = input()
                else:
                    raw = input("> ")
            except EOFError:
                print(file=sys.stderr)
                _pending_exit(session, response, args, show_question=False)
        try:
            value, certainty = _parse_answer_text(raw, question)
            response = session.answer(value, certainty=certainty, strict=not args.no_strict)
        except (InvalidAnswerError, InvalidFactError, BadRequestError) as exc:
            # Only answer-level problems are worth re-prompting for; auth, network and
            # missing-session errors propagate to main() and exit 1.
            if from_script or not interactive:
                raise
            print(f"  ! {exc}", file=sys.stderr)
    return response


# --------------------------------------------------------------------------- commands


def cmd_version(args: argparse.Namespace) -> None:
    key = args.api_key if args.api_key is not None else os.environ.get(ENV_API_KEY)
    if key:
        with _client(args) as rb:
            version = rb.version()
    else:  # GET /version is public; do not demand a key for a connectivity check
        version = engine_version(args.base_url, engine=args.engine, timeout=args.timeout)
    if args.json:
        _emit_json({"engine": version, "sdk": __version__})
    else:
        print(f"engine {version}\nsdk    {__version__}")


def cmd_start(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        session = rb.start(
            args.km_id,
            use_draft=args.draft or None,
            version=args.version,
            context_id=args.context_id,
        )
    if args.json:
        _emit_json(
            {
                "session_id": session.id,
                "km_version_id": session.km_version_id,
                "km_id": session.km_id,
            }
        )
    else:
        print(session.id)


def _load_facts(args: argparse.Namespace) -> list[Any]:
    """Collect and validate facts from ``--fact``/``--file`` before any network call."""
    facts: list[Any] = []
    for fact in args.fact or []:
        if len(fact) not in (3, 4):
            _die("--fact needs SUBJECT RELATIONSHIP OBJECT [CERTAINTY]", EXIT_USAGE)
        if len(fact) == 4:
            try:
                certainty = float(fact[3])
            except ValueError:
                _die(f"--fact certainty must be a number 0-100, got {fact[3]!r}", EXIT_USAGE)
            facts.append(
                (fact[0], fact[1], fact[2], int(certainty) if certainty.is_integer() else certainty)
            )
        else:
            facts.append((fact[0], fact[1], fact[2]))
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            try:
                loaded = json.load(fh)
            except ValueError as exc:
                _die(f"{args.file}: invalid JSON: {exc}", EXIT_USAGE)
        if isinstance(loaded, dict):
            loaded = [loaded]
        if not isinstance(loaded, list) or not all(isinstance(f, (dict, list)) for f in loaded):
            _die(
                f"{args.file}: expected a JSON list of fact objects "
                '({"subject", "relationship", "object", "certainty"}) or [s, r, o, c] lists',
                EXIT_USAGE,
            )
        facts.extend(loaded)
    if facts:
        try:
            normalize_facts(facts)
        except (InvalidFactError, InvalidAnswerError) as exc:
            _die(f"invalid fact: {exc}", EXIT_USAGE)
    return facts


def cmd_inject(args: argparse.Namespace) -> None:
    facts = _load_facts(args)
    if not facts:
        _die("nothing to inject: use --fact or --file", EXIT_USAGE)
    with _client(args) as rb:
        session = rb.resume(args.session)
        result = session.inject(facts)
    if args.json:
        _emit_json(result)
    else:
        print(f"injected {len(facts)} fact(s): {result.result}")


def cmd_query(args: argparse.Namespace) -> None:
    facts = _load_facts(args)  # validate local input before touching the network
    with _client(args) as rb:
        session = (
            rb.resume(args.session)
            if args.session
            else rb.start(args.km_id, use_draft=args.draft or None, version=args.version)
        )
        if facts:
            session.inject(facts)
        if not args.json and not args.session:
            print(f"session {session.id}")
        response = session.query(args.relationship, subject=args.subject, object=args.object)
        response = _drive(session, response, args)
    if args.json:
        _emit_json({"session_id": session.id, **response.raw})
    else:
        assert isinstance(response, ResultResponse)
        _print_results(response)


def _print_decision(decision: Decision) -> None:
    rows: list[tuple[str, Any]] = [("status", decision.status.value)]
    if decision.status is DecisionStatus.QUESTION:
        rows.append(("session", decision.session_id))
    else:
        cert = f"{decision.certainty}%" if decision.certainty is not None else "-"
        rows += [
            ("decision", decision.decision if decision.decision is not None else "-"),
            ("certainty", cert),
            ("session", decision.session_id),
            ("fact_id", decision.fact_id or "-"),
        ]
        if decision.evidence_url:
            rows.append(("evidence", decision.evidence_url))
    for label, value in rows:
        print(f"{label:<10} {value}")
    if decision.question is not None:
        _print_question(decision.question)
    if len(decision.results) > 1 and isinstance(decision.response, ResultResponse):
        print("results:")
        _print_results(decision.response)
    if decision.evidence:
        print("evidence tree:")
        _print_evidence_tree(decision.evidence)


def cmd_decide(args: argparse.Namespace) -> None:
    facts = _load_facts(args)  # validate local input before touching the network
    with _client(args) as rb:
        decision = rb.decide(
            args.relationship,
            km_id=args.km_id,
            subject=args.subject,
            object=args.object,
            facts=facts or None,
            evidence=args.evidence,
            evidence_depth=args.max_depth,
            use_draft=args.draft or None,
            version=args.version,
            context_id=args.context_id,
        )
    if args.json:
        _emit_json(decision.to_dict())
    else:
        _print_decision(decision)
    if decision.status is DecisionStatus.QUESTION:
        print(
            f"rainbird: session {decision.session_id} is waiting for an answer; continue with "
            f"rainbird query --session {decision.session_id} -r {args.relationship} ...",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_PENDING)


def cmd_respond(args: argparse.Namespace) -> None:
    try:
        answers = json.loads(args.answers_json)
    except ValueError as exc:
        _die(f"answers must be valid JSON: {exc}", EXIT_USAGE)
    with _client(args) as rb:
        session = rb.resume(args.session)
        response = session.respond(answers)
    if args.json:
        _emit_json(response)
    elif isinstance(response, ResultResponse):
        _print_results(response)
    else:
        _print_question(response.question)


def cmd_undo(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        response = rb.resume(args.session).undo()
    if args.json:
        _emit_json(response)
    elif isinstance(response, ResultResponse):
        _print_results(response)
    else:
        _print_question(response.question)


def _print_evidence_tree(nodes: list[Evidence]) -> None:
    """Print fetched evidence as an indented tree, following ``factID`` links.

    Relationship conditions whose fact was fetched are shown as child nodes; other
    conditions are listed inline. Facts referenced from expression functions are
    shown beneath the expression. Anything fetched but unreachable is appended.
    """
    by_id = {node.fact_id: node for node in nodes}
    seen: set[str] = set()

    def walk(fact_id: str, indent: int) -> None:
        node = by_id.get(fact_id)
        if node is None or fact_id in seen:
            return
        seen.add(fact_id)
        pad = "  " * indent
        print(f"{pad}{node.fact}  [{_source_name(node.source)}]  {node.fact_id}")
        if not node.rule:
            return
        for cond in node.rule.conditions:
            if isinstance(cond, Condition):
                if cond.fact_id and cond.fact_id in by_id and cond.fact_id not in seen:
                    walk(cond.fact_id, indent + 1)
                else:
                    print(
                        f"{pad}  - {cond.subject} {cond.relationship} {cond.object} "
                        f"(certainty {cond.certainty}, impact {cond.impact})"
                    )
            else:
                print(f"{pad}  - expr {cond.text!r} -> {cond.value!r} met={cond.was_met}")
                for fid in cond.fact_ids:
                    walk(fid, indent + 2)

    if nodes:
        walk(nodes[0].fact_id, 0)
    for node in nodes:  # never silently drop something we fetched
        if node.fact_id not in seen:
            walk(node.fact_id, 0)


def cmd_evidence(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        session = rb.resume(args.session)
        if args.tree:
            nodes = session.evidence_tree(args.fact_id, max_depth=args.max_depth)
            if args.json:
                _emit_json(nodes)
            else:
                _print_evidence_tree(nodes)
        else:
            node = session.evidence(args.fact_id)
            if args.json:
                _emit_json(node)
            else:
                _print_evidence_tree([node])
        if args.url:
            print(session.evidence_url(args.fact_id))


def cmd_explain(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        result = rb.resume(args.session).explain(
            args.fact_id, language=args.language, instructions=args.instructions
        )
    if args.json:
        _emit_json(result)
    else:
        print(result.explanation or "")


def cmd_interact(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        session = rb.resume(args.session) if args.session else rb.start(args.km_id)
        if not args.session and not args.json:
            print(f"session {session.id}")
        result = session.interact(args.prompt)
    if args.json:
        _emit_json({"session_id": session.id, **result.raw})
        return
    if result.is_error:
        _die(result.prompt or "interact returned an error")
    if result.is_question:
        for i, q in enumerate(result.questions, 1):
            _print_question(q, i)
        print(f'\nAnswer with: rainbird interact --session {session.id} "<your answer>"')
    else:
        for item in result.results:
            print(item)


def cmd_facts(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        facts = rb.resume(args.session).facts(relationships=args.relationships)
    if args.json:
        _emit_json(facts)
        return
    for scope, items in (
        ("global", facts.global_),
        ("context", facts.context),
        ("local", facts.local),
    ):
        if items:
            print(f"{scope} ({len(items)})")
            for fact in items:
                print(f"  {fact}  [{_source_name(fact.source)}]")


def cmd_session(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        info = rb.resume(args.session).knowledge_map()
    if args.json:
        _emit_json(info)
    else:
        print(f"knowledge map: {info.name} ({info.id})")
        print(f"version: {info.version_status} ({info.version_id})")


def cmd_interactions(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        log = rb.resume(args.session).interactions(format=args.format)
    if isinstance(log, str):
        print(log)
    elif args.json:
        _emit_json(log)
    else:
        for event in log:
            when = event.created.isoformat() if event.created else "-"
            print(f"{when}  {event.event:<10} {json.dumps(event.values, ensure_ascii=False)}")


def cmd_map_get(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        source = rb.get_map_source(args.km_id)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(source.rblang)
        if not args.json:
            print(f"wrote {len(source.rblang)} chars of RBLang to {args.out}")
    if args.json:
        _emit_json(source)
    elif not args.out:
        print(source.rblang)


def _read_text(path: str) -> str:
    """The contents of ``path``, or of stdin when ``path`` is ``-``."""
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def cmd_map_create(args: argparse.Namespace) -> None:
    rblang = _read_text(args.file)
    with _client(args) as rb:
        result = rb.create_map(args.name, rblang, args.description)
    if args.json:
        _emit_json(result)
    else:
        print(result.km_id)


def cmd_map_validate(args: argparse.Namespace) -> None:
    rblang = _read_text(args.file)
    with _client(args) as rb:
        result = rb.validate_rblang(rblang)
    if args.json:
        _emit_json(result)
    else:
        print("valid" if result.valid else "invalid")
        for message in result.errors:
            print(f"  error    {message}")
        for message in result.warnings:
            print(f"  warning  {message}")
    if not result.valid:
        raise SystemExit(EXIT_FAILED)


def cmd_map_add_readme(args: argparse.Namespace) -> None:
    readme = _read_text(args.file)
    with _client(args) as rb:
        result = rb.add_readme(args.km_id, readme)
    if args.json:
        _emit_json(result)
    else:
        print(f"added a readme to the draft of {result.km_id}")


def cmd_map_get_tests(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        suite = rb.get_tests(args.km_id)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(suite.tests, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        if not args.json:
            print(f"wrote {len(suite.tests)} test(s) to {args.out}")
    if args.json:
        _emit_json(suite)
    elif not args.out:
        print(f"{suite.test_count} test(s) on {suite.km_id}")
        for index, test in enumerate(suite.tests):
            entry = test if isinstance(test, dict) else {}
            steps = entry.get("steps")
            count = len(steps) if isinstance(steps, list) else 0
            print(f"  [{index}] {entry.get('name') or '?'} ({count} step(s))")


def _load_suite(path: str) -> list[Any]:
    """Read and check a test suite file before any network call (usage errors exit 2)."""
    text = _read_text(path)
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        _die(f"{path}: invalid JSON: {exc}", EXIT_USAGE)
    try:
        return suite_payload(loaded)
    except (RainbirdConfigurationError, InvalidFactError) as exc:
        _die(f"{path}: {exc}", EXIT_USAGE)


def cmd_map_create_tests(args: argparse.Namespace) -> None:
    tests = _load_suite(args.file)
    with _client(args) as rb:
        result = rb.create_tests(args.km_id, tests)
    if args.json:
        _emit_json(result)
    else:
        count = result.test_count if result.test_count is not None else len(tests)
        print(f"created {count} test(s) on {result.km_id}")


def _print_test_run(run: TestRun) -> None:
    summary = run.summary
    print(
        f"run {run.run_id} {run.status}: {summary.passed} of {summary.total} passed "
        f"({summary.failed} failed, {summary.errored} errored, {summary.not_run} not run)"
    )
    for result in run.failures:
        index = result.index if result.index is not None else "?"
        failed_at = result.failed_step_index
        step = f" at step {failed_at}" if failed_at is not None else ""
        detail = ": " + "; ".join(result.errors) if result.errors else ""
        print(f"  {result.status!s:<6} [{index}] {result.name}{step}{detail}")
    if run.error is not None:
        print(f"  error: {run.error}")


def _report_test_run(run: TestRun, args: argparse.Namespace) -> None:
    """Print a run; exit 1 if it failed, 3 if it is unfinished, 4 if any test failed."""
    if args.json:
        _emit_json(run)
    else:
        _print_test_run(run)
    if run.is_failed:
        _die(f"test run {run.run_id} failed" + (f": {run.error}" if run.error else ""))
    if not run.is_finished:
        print(
            f"rainbird: test run {run.run_id} is still {run.status}; check it with "
            f"rainbird map test-run {run.run_id}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_PENDING)
    if not run.all_passed:
        raise SystemExit(EXIT_FAILED)


def cmd_map_run_tests(args: argparse.Namespace) -> None:
    if args.poll_interval <= 0:
        _die("--poll-interval must be a positive number of seconds", EXIT_USAGE)
    if args.max_wait < 0:
        _die("--max-wait must not be negative", EXIT_USAGE)
    with _client(args) as rb:
        started = rb.start_test_run(args.km_id)
        if args.no_wait:
            if args.json:
                _emit_json(started)
            else:
                print(started.run_id)
            return
        print(f"rainbird: started test run {started.run_id}", file=sys.stderr)
        try:
            run = rb.wait_for_test_run(
                started.run_id, poll_interval=args.poll_interval, timeout=args.max_wait
            )
        except (TestRunFailedError, TestRunTimeoutError) as exc:
            if exc.run is None:
                raise
            run = exc.run
    _report_test_run(run, args)


def cmd_map_test_run(args: argparse.Namespace) -> None:
    with _client(args) as rb:
        run = rb.get_test_run(args.run_id)
    _report_test_run(run, args)


# --------------------------------------------------------------------------- parser


def _add_global_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Global options. On the root they carry defaults; on subcommands they are
    ``SUPPRESS``ed so a value given after the subcommand overrides the root default."""

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument(
        "--api-key", default=default(None), help="API key (default: $RAINBIRD_API_KEY)"
    )
    parser.add_argument(
        "--base-url",
        default=default(None),
        help=f"API host (default: $RAINBIRD_API_URL or {Environment.COMMUNITY.value}); "
        f"enterprise: {Environment.ENTERPRISE.value}",
    )
    parser.add_argument(
        "--engine", default=default(None), help="alternate engine name (x-rainbird-engine header)"
    )
    parser.add_argument(
        "--evidence-key", default=default(None), help="x-evidence-key for evidence calls"
    )
    parser.add_argument(
        "--interaction-key", default=default(None), help="x-interaction-key for interaction logs"
    )
    parser.add_argument(
        "--timeout", type=float, default=default(30.0), help="request timeout in seconds"
    )
    parser.add_argument(
        "--json", action="store_true", default=default(False), help="print raw JSON instead of text"
    )


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - one statement per CLI option
    parser = argparse.ArgumentParser(
        prog="rainbird",
        description="Command-line client for the Rainbird decision intelligence API.",
        epilog="Environment: RAINBIRD_API_KEY, RAINBIRD_API_URL, RAINBIRD_KM_ID, "
        "RAINBIRD_EVIDENCE_KEY, RAINBIRD_INTERACTION_KEY, RAINBIRD_ENGINE. "
        "Global options may also be given after the subcommand.",
    )
    # Top-level only: after a subcommand, ``--version N`` is the knowledge map version.
    parser.add_argument(
        "-V",
        "--version",
        "--sdk-version",
        action="version",
        version=f"rainbird-ai {__version__}",
    )
    _add_global_options(parser, suppress=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress=True)

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common])

    p = command("version", "engine and SDK versions (no API key needed)")
    p.set_defaults(func=cmd_version)

    def add_km(p: argparse.ArgumentParser) -> None:
        p.add_argument("--km-id", help="knowledge map id (default: $RAINBIRD_KM_ID)")
        group = p.add_mutually_exclusive_group()
        group.add_argument("--draft", action="store_true", help="use the draft version")
        group.add_argument("--version", type=int, help="use a specific version number")

    p = command("start", "start a session and print its id")
    add_km(p)
    p.add_argument("--context-id", help="share context-scoped facts between sessions")
    p.set_defaults(func=cmd_start)

    def add_facts(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--fact",
            nargs="+",
            action="append",
            metavar="S R O [C]",
            help="a fact to inject (repeatable)",
        )
        p.add_argument(
            "--file", help="JSON file with a list of {subject, relationship, object, certainty}"
        )

    p = command("inject", "inject facts into a session")
    p.add_argument("--session", required=True, help="session id")
    add_facts(p)
    p.set_defaults(func=cmd_inject)

    p = command("query", "query a relationship and answer questions until a result")
    p.add_argument("--relationship", "-r", required=True)
    p.add_argument("--subject", "-s")
    p.add_argument("--object", "-o")
    p.add_argument("--session", help="reuse an existing session instead of starting one")
    add_km(p)
    add_facts(p)
    p.add_argument(
        "--answers",
        "-a",
        nargs="+",
        help="scripted answers in order; '?' skips, 'a,b' for plural, 'x@80' for certainty",
    )
    p.add_argument(
        "--no-input", action="store_true", help="never prompt; exit 3 if a question remains"
    )
    p.add_argument("--no-strict", action="store_true", help="skip client-side answer validation")
    p.set_defaults(func=cmd_query)

    p = command("decide", "one-shot decision: start, inject facts, query, summarise")
    p.add_argument("--relationship", "-r", required=True)
    p.add_argument("--subject", "-s")
    p.add_argument("--object", "-o")
    add_km(p)
    p.add_argument("--context-id", help="share context-scoped facts between sessions")
    add_facts(p)
    p.add_argument(
        "--evidence",
        choices=["none", "root", "tree"],
        default="none",
        help="also fetch the best result's evidence node or whole tree (needs the evidence key)",
    )
    p.add_argument("--max-depth", type=int, help="depth limit for --evidence tree")
    p.set_defaults(func=cmd_decide)

    p = command("respond", "send raw answers JSON to a session")
    p.add_argument("--session", required=True)
    p.add_argument(
        "answers_json",
        help='JSON list, e.g. \'[{"subject":"Tom","relationship":"speaks","object":"German"}]\'',
    )
    p.set_defaults(func=cmd_respond)

    p = command("undo", "undo the last answers in a session")
    p.add_argument("--session", required=True)
    p.set_defaults(func=cmd_undo)

    p = command("evidence", "evidence for a fact")
    p.add_argument("--session", required=True)
    p.add_argument("--fact-id", required=True)
    p.add_argument("--tree", action="store_true", help="fetch the whole evidence tree")
    p.add_argument("--max-depth", type=int)
    p.add_argument("--url", action="store_true", help="also print the Studio evidence URL")
    p.set_defaults(func=cmd_evidence)

    p = command("explain", "natural-language explanation of a fact (beta)")
    p.add_argument("--session", required=True)
    p.add_argument("--fact-id", required=True)
    p.add_argument("--language", default="en")
    p.add_argument("--instructions")
    p.set_defaults(func=cmd_explain)

    p = command("interact", "natural-language interaction (beta)")
    p.add_argument("prompt")
    p.add_argument("--session", help="continue an existing session")
    p.add_argument("--km-id", help="start a new session on this map (default: $RAINBIRD_KM_ID)")
    p.set_defaults(func=cmd_interact)

    p = command("facts", "facts visible to a session")
    p.add_argument("--session", required=True)
    p.add_argument("--relationships", nargs="+")
    p.set_defaults(func=cmd_facts)

    p = command("session", "knowledge map and version used by a session")
    p.add_argument("--session", required=True)
    p.set_defaults(func=cmd_session)

    p = command("interactions", "interaction log of a session")
    p.add_argument("--session", required=True)
    p.add_argument("--format", choices=["json", "csv"], default="json")
    p.set_defaults(func=cmd_interactions)

    p = sub.add_parser(
        "map",
        help="knowledge maps: source, readme, tests, RBLang validation (undocumented endpoints)",
    )
    msub = p.add_subparsers(dest="map_command", required=True, metavar="ACTION")

    def map_command(name: str, help_text: str) -> argparse.ArgumentParser:
        return msub.add_parser(name, help=help_text, parents=[common])

    mp = map_command("get", "download RBLang for a knowledge map")
    mp.add_argument("km_id")
    mp.add_argument("--out", help="write RBLang to this file")
    mp.set_defaults(func=cmd_map_get)
    mp = map_command("create", "create a knowledge map from an RBLang file")
    mp.add_argument("--name", required=True)
    mp.add_argument("--description", help="optional description stored with the map")
    mp.add_argument("--file", required=True, help="RBLang XML file, or '-' for stdin")
    mp.set_defaults(func=cmd_map_create)
    mp = map_command("validate", "check RBLang without saving it (exit 4 if invalid)")
    mp.add_argument("--file", required=True, help="RBLang XML file, or '-' for stdin")
    mp.set_defaults(func=cmd_map_validate)
    mp = map_command("add-readme", "add a Markdown readme to a map's draft (if it has none)")
    mp.add_argument("km_id")
    mp.add_argument("--file", required=True, help="Markdown file, or '-' for stdin")
    mp.set_defaults(func=cmd_map_add_readme)
    mp = map_command("get-tests", "list a map's test suite (Studio's Unit Tester tests)")
    mp.add_argument("km_id")
    mp.add_argument("--out", help="write the suite to this file as JSON (for create-tests)")
    mp.set_defaults(func=cmd_map_get_tests)
    mp = map_command("create-tests", "save a test suite on a map that has none")
    mp.add_argument("km_id")
    mp.add_argument(
        "--file",
        required=True,
        help="JSON list of tests or a get-tests --json body, or '-' for stdin",
    )
    mp.set_defaults(func=cmd_map_create_tests)
    mp = map_command(
        "run-tests", "run a map's tests on its draft and wait (exit 4 if a test fails)"
    )
    mp.add_argument("km_id")
    mp.add_argument("--no-wait", action="store_true", help="print the run id and exit")
    mp.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="time between status checks (default: 1)",
    )
    mp.add_argument(
        "--max-wait",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="stop waiting after this long and exit 3; the run carries on (default: 300)",
    )
    mp.set_defaults(func=cmd_map_run_tests)
    mp = map_command("test-run", "status and results of a test run")
    mp.add_argument("run_id")
    mp.set_defaults(func=cmd_map_test_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RainbirdError as exc:
        _die(str(exc))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except OSError as exc:
        _die(str(exc))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
