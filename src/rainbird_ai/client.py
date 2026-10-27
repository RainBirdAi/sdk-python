"""Synchronous client: :class:`Rainbird` (account level) and :class:`Session` (one session).

Typical use::

    from rainbird_ai import Rainbird

    with Rainbird(api_key="...") as rb:
        session = rb.start("your-km-id")
        session.inject_fact("Tom", "lives in", "Germany")
        response = session.query("speaks", subject="Tom")
        while response.is_question:
            print(response.question.prompt)
            response = session.answer("German")
        for result in response.results:
            print(result)
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, overload

import httpx

from . import _ops
from ._answers import SKIP, build_answers, build_skip, normalize_answers, normalize_facts
from ._http import (
    ENV_KM_ID,
    UNSET,
    ClientConfig,
    Request,
    RetryPolicy,
    SyncTransport,
    Unset,
)
from .errors import (
    NoPendingQuestionError,
    RainbirdConfigurationError,
    RainbirdError,
    TestRunFailedError,
    TestRunTimeoutError,
)
from .models import (
    Answer,
    CreateMapResult,
    Decision,
    Evidence,
    ExplainResponse,
    InjectResponse,
    InteractionEvent,
    InteractResponse,
    KnowledgeMapSource,
    KnowledgeMapVersion,
    QueryResponse,
    Question,
    QuestionResponse,
    ReadmeAdded,
    ResultResponse,
    SessionFacts,
    StartResponse,
    TestRun,
    TestRunStarted,
    TestSuite,
    TestSuiteCreated,
    ValidationResult,
    studio_url_for,
)

__all__ = ["Answerer", "Rainbird", "Session", "engine_version"]

Answerer = Callable[[Question], Any]
"""Callback used by :meth:`Session.run`: receives a :class:`Question`, returns the answer value,
a list of values (plural questions), an :class:`Answer`/list of answers, or :data:`SKIP`."""

MAX_INJECT_BATCH = 250


class Rainbird:
    """Entry point for the Rainbird API.

    Args:
        api_key: Your API key (Rainbird Studio > knowledge map > Publish). Falls back to
            the ``RAINBIRD_API_KEY`` environment variable.
        base_url: API host. Defaults to ``RAINBIRD_API_URL`` or the Community host
            ``https://api.rainbird.ai``. Use :class:`~rainbird_ai.Environment` or a
            private host such as ``https://acme-api.rainbird.ai``.
        evidence_key: Default ``x-evidence-key`` for evidence calls (``RAINBIRD_EVIDENCE_KEY``).
        interaction_key: Default ``x-interaction-key`` for interaction logs
            (``RAINBIRD_INTERACTION_KEY``).
        engine: Optional alternate engine name sent as ``x-rainbird-engine`` (``RAINBIRD_ENGINE``).
        timeout: Seconds (or an ``httpx.Timeout``) per request. Defaults to 30 for the
            SDK's own connection pool; when you pass ``http_client`` its own timeout applies
            unless you set this explicitly, in which case it is applied per request.
        retry: :class:`~rainbird_ai.RetryPolicy`. Defaults to two retries for safe requests.
        default_headers: Extra headers sent with every request.
        http_client: Bring your own ``httpx.Client`` (proxies, custom transports, tests).
            You remain responsible for closing it. Redirects are never followed by the
            SDK's own client; keep ``follow_redirects=False`` on yours too so the API key
            is not forwarded to a third-party host.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        evidence_key: str | None = None,
        interaction_key: str | None = None,
        engine: str | None = None,
        timeout: float | httpx.Timeout | Unset | None = UNSET,
        retry: RetryPolicy | None = None,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = ClientConfig.resolve(
            api_key=api_key,
            base_url=str(base_url) if base_url is not None else None,
            evidence_key=evidence_key,
            interaction_key=interaction_key,
            engine=engine,
            timeout=timeout,
            retry=retry,
            default_headers=default_headers,
        )
        self._transport = SyncTransport(self.config, http_client)

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Resolved API host this client talks to (no trailing slash)."""
        return self.config.base_url

    @property
    def studio_url(self) -> str:
        """Rainbird Studio host matching :attr:`base_url` (used for evidence links)."""
        return studio_url_for(self.config.base_url)

    @property
    def http(self) -> httpx.Client:
        """The underlying ``httpx.Client``."""
        return self._transport.http

    def close(self) -> None:
        """Close the HTTP connection pool (no-op for a user-supplied ``http_client``)."""
        self._transport.close()

    def __enter__(self) -> Rainbird:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Rainbird(base_url={self.config.base_url!r})"

    def _send(self, request: Request[Any], extra_headers: Mapping[str, str] | None = None) -> Any:
        return self._transport.send(request, extra_headers=extra_headers)

    # -- sessions ----------------------------------------------------------------------

    def start(
        self,
        km_id: str | None = None,
        *,
        use_draft: bool | None = None,
        version: int | None = None,
        context_id: str | None = None,
    ) -> Session:
        """Start a reasoning session against a knowledge map.

        By default the live version is used (or the draft if nothing is live). Pass
        ``use_draft=True`` or ``version=N`` to override (not both). ``context_id``
        lets sessions share facts on context-scoped relationships.
        """
        resolved = km_id if km_id is not None else os.environ.get(ENV_KM_ID)
        if not resolved or not str(resolved).strip():
            raise RainbirdConfigurationError(
                "km_id is required: pass it to start() or set RAINBIRD_KM_ID"
            )
        response: StartResponse = self._send(
            _ops.start(str(resolved), use_draft=use_draft, version=version, context_id=context_id)
        )
        return Session(
            self,
            response.session_id,
            km_version_id=response.km_version_id,
            km_id=str(resolved),
            context_id=context_id,
            start_response=response,
        )

    def resume(self, session_id: str, *, km_id: str | None = None) -> Session:
        """Attach to an existing session by ID (no request is made)."""
        if not session_id or not str(session_id).strip():
            raise RainbirdConfigurationError("session_id must not be empty")
        return Session(self, str(session_id), km_id=km_id)

    def decide(
        self,
        relationship: str,
        *,
        km_id: str | None = None,
        subject: Any = None,
        object: Any = None,
        facts: Any = None,
        answerer: Answerer | None = None,
        max_questions: int = 100,
        evidence: Literal["none", "root", "tree"] = "none",
        evidence_depth: int | None = None,
        evidence_key: str | None = None,
        use_draft: bool | None = None,
        version: int | None = None,
        context_id: str | None = None,
    ) -> Decision:
        """Get a decision in one call: start a session, inject ``facts``, query, summarise.

        Runs ``GET /start/{kmID}``, ``POST /{sid}/inject`` (only when ``facts`` is given; any
        shape :meth:`Session.inject` accepts) and ``POST /{sid}/query`` for ``relationship``
        with the given ``subject`` and/or ``object``, then returns a flat
        :class:`~rainbird_ai.Decision`. ``km_id`` falls back to ``RAINBIRD_KM_ID``;
        ``use_draft``, ``version`` and ``context_id`` are passed to :meth:`start`.

        Without an ``answerer`` the first question the engine asks ends the call with
        ``status == DecisionStatus.QUESTION``; continue it with ``decision.session.answer(...)``,
        or later with ``resume(decision.session_id).answer(..., question=decision.question)``
        (a resumed session does not know the pending question). With an ``answerer`` the
        questions are answered through :meth:`Session.run` (at most ``max_questions``) so the
        result is always final.

        ``evidence="root"`` also fetches the evidence node of the best result and ``"tree"``
        its whole evidence tree (``evidence_depth`` limits it); both need the map's evidence
        key (``evidence_key`` or the client default) and add one request per node.
        ``Decision.evidence_url`` is always filled when there is a ``fact_id``.
        """
        _check_evidence_mode(evidence)
        # Validate the query and the facts before starting a session that would be left unused.
        query = _ops.query_body(relationship, subject=subject, object=object)
        payload = normalize_facts(facts) if facts else None
        session = self.start(km_id, use_draft=use_draft, version=version, context_id=context_id)
        if payload:
            session.inject(payload)
        if answerer is None:
            response = session.query(relationship, subject=subject, object=object)
        else:
            response = session.run(
                relationship,
                subject=subject,
                object=object,
                answerer=answerer,
                max_questions=max_questions,
            )
        best = response.best if isinstance(response, ResultResponse) else None
        fact_id = best.fact_id if best is not None else None
        nodes: list[Evidence] = []
        if fact_id and evidence == "root":
            nodes = [session.evidence(fact_id, evidence_key=evidence_key)]
        elif fact_id and evidence == "tree":
            nodes = session.evidence_tree(
                fact_id, max_depth=evidence_depth, evidence_key=evidence_key
            )
        return Decision.from_response(
            response,
            session_id=session.id,
            relationship=relationship,
            subject=query.get("subject", subject),  # the coerced value, so to_dict() stays JSON
            object=query.get("object", object),
            evidence=nodes,
            evidence_url=session.evidence_url(fact_id) if fact_id else None,
            km_id=session.km_id,
            km_version_id=session.km_version_id,
            session=session,
        )

    # -- platform ----------------------------------------------------------------------

    def version(self) -> str:
        """Engine version (semver string) from ``GET /version``.

        The endpoint itself is unauthenticated; use :func:`engine_version` to call it
        without constructing a keyed client.
        """
        result: str = self._send(_ops.version())
        return result

    # -- evidence (session-independent forms) ------------------------------------------

    def evidence(
        self, session_id: str, fact_id: str, *, evidence_key: str | None = None
    ) -> Evidence:
        """Evidence for ``fact_id`` produced in ``session_id``."""
        key = evidence_key if evidence_key is not None else self.config.evidence_key
        result: Evidence = self._send(_ops.evidence(fact_id, session_id, key))
        return result

    def evidence_url(self, session_id: str, fact_id: str, *, studio_url: str | None = None) -> str:
        """Link to the Evidence Tree UI in Rainbird Studio for a fact."""
        studio = (studio_url or self.studio_url).rstrip("/")
        return f"{studio}/evidence?id={fact_id}&api={self.config.base_url}&sid={session_id}"

    # -- knowledge maps (undocumented endpoints) ---------------------------------------

    def get_map_source(self, km_id: str) -> KnowledgeMapSource:
        """RBLang source of a knowledge map via ``GET /analysis/file/{kmID}``.

        This endpoint is not in the public OpenAPI specification and may change.
        """
        result: KnowledgeMapSource = self._send(_ops.map_source(km_id))
        return result

    def create_map(self, name: str, rblang: str, description: str | None = None) -> CreateMapResult:
        """Create a knowledge map from RBLang via ``POST /maps``.

        The body is ``{"rblang", "name", "description"}``; ``description`` is optional and
        sent only when given. Raises :class:`~rainbird_ai.errors.KnowledgeMapError` when
        Rainbird rejects the RBLang (it replies HTTP 200 with an ``error`` key in that case), and
        :class:`~rainbird_ai.BadRequestError` ``NAME_ERROR`` for a name it does not accept (one
        containing ``:``, for example).
        This endpoint is not in the public OpenAPI specification and may change.
        """
        result: CreateMapResult = self._send(_ops.create_map(name, rblang, description))
        return result

    # -- knowledge map tests, readme and validation (undocumented endpoints) -----------

    def create_tests(
        self, km_id: str, tests: Sequence[Mapping[str, Any]] | Mapping[str, Any] | TestSuite
    ) -> TestSuiteCreated:
        """Save a test suite on a knowledge map via ``POST /maps/{kmID}/tests``.

        ``tests`` is a list in the format of Studio's Unit Tester (``[{"name": ..., "steps":
        [...]}]``; tests created here appear there), or a :class:`~rainbird_ai.TestSuite` /
        ``{"tests": [...]}`` body from :meth:`get_tests` to copy a suite between maps. Test
        steps write certainty as ``cf``: a fact giving ``certainty`` raises
        :class:`~rainbird_ai.InvalidFactError` before anything is sent, because the test runner
        would ignore it. A map that already has tests raises
        :class:`~rainbird_ai.ConflictError` (409); suites cannot be replaced or deleted through
        the API. This endpoint is not in the public OpenAPI specification and may change.
        """
        result: TestSuiteCreated = self._send(_ops.create_tests(km_id, tests))
        return result

    def get_tests(self, km_id: str) -> TestSuite:
        """A knowledge map's test suite via ``GET /maps/{kmID}/tests``.

        Raises :class:`~rainbird_ai.NotFoundError` when the map has no tests and
        :class:`~rainbird_ai.ConflictError` when its stored suite is in a format the API cannot
        read. This endpoint is not in the public OpenAPI specification and may change.
        """
        result: TestSuite = self._send(_ops.get_tests(km_id))
        return result

    def start_test_run(self, km_id: str) -> TestRunStarted:
        """Start running a knowledge map's tests via ``POST /maps/{kmID}/runs``.

        Returns at once with ``run_id``; poll :meth:`get_test_run`, or call :meth:`run_tests`
        to start and wait in one go. A run always tests the map's draft and covers the whole
        suite. It is accepted even when the map has no tests (the run then fails with category
        ``suite_load``). Like every ``POST`` it is retried on 429/503, which could start a
        second, harmless run. This endpoint is not in the public OpenAPI specification and may
        change.
        """
        result: TestRunStarted = self._send(_ops.start_test_run(km_id))
        return result

    def get_test_run(self, run_id: str) -> TestRun:
        """Status and results of a test run via ``GET /runs/{runID}``.

        :class:`~rainbird_ai.NotFoundError` means the run id is unknown or belongs to another
        user's map, not that the run is still starting. This endpoint is not in the public
        OpenAPI specification and may change.
        """
        result: TestRun = self._send(_ops.get_test_run(run_id))
        return result

    def wait_for_test_run(
        self, run_id: str, *, poll_interval: float = 1.0, timeout: float | None = 300.0
    ) -> TestRun:
        """Wait for a test run to finish: :meth:`get_test_run` every ``poll_interval`` seconds.

        Returns the ``complete`` :class:`~rainbird_ai.TestRun`. Tests can still have failed in
        a complete run: check ``run.all_passed``, ``run.summary`` and ``run.failures``.

        Raises :class:`~rainbird_ai.TestRunFailedError` when the run ends ``failed`` (``.run``,
        ``.category``) and :class:`~rainbird_ai.TestRunTimeoutError`, a
        :class:`~rainbird_ai.RainbirdTimeoutError`, when it is still unfinished after
        ``timeout`` seconds (``None`` waits indefinitely; the server abandons a run after 15
        minutes without progress). A timeout does not stop the run: call this again with
        ``exc.run_id`` to keep waiting. ``timeout`` is the overall wait, not the per-request HTTP
        timeout set on the client.
        """
        _check_polling(poll_interval, timeout)
        started = time.monotonic()
        while True:
            run = self.get_test_run(run_id)
            if run.is_finished:
                return _final_run(run)
            time.sleep(_poll_delay(run, started, poll_interval, timeout))

    def run_tests(
        self, km_id: str, *, poll_interval: float = 1.0, timeout: float | None = 300.0
    ) -> TestRun:
        """Run a knowledge map's tests and wait for the outcome.

        :meth:`start_test_run` followed by :meth:`wait_for_test_run` (``timeout`` counts from
        the start of the run); see there for the result and the exceptions.
        """
        _check_polling(poll_interval, timeout)  # before starting a run that nobody waits for
        run_id = self.start_test_run(km_id).run_id
        return self.wait_for_test_run(run_id, poll_interval=poll_interval, timeout=timeout)

    def add_readme(self, km_id: str, readme: str) -> ReadmeAdded:
        """Add a Markdown readme to a knowledge map's draft via ``POST /maps/{kmID}/readme``.

        Only a draft without a readme is changed (blank text or Studio's cleared
        ``<p><br></p>`` count as none); otherwise :class:`~rainbird_ai.ConflictError` (409).
        Published versions keep their own readme, so the new one reaches users with the next
        published version. The server rejects a readme over 100,000 characters, or one that
        starts with ``<`` (Studio would open it as HTML), with
        :class:`~rainbird_ai.BadRequestError`. This endpoint is not in the public OpenAPI
        specification and may change.
        """
        result: ReadmeAdded = self._send(_ops.add_readme(km_id, readme))
        return result

    def validate_rblang(self, rblang: str) -> ValidationResult:
        """Compile RBLang without saving it via ``POST /validate``.

        Invalid RBLang is a normal result, not an exception: check ``result.valid``,
        ``result.errors`` (at most one; compilation stops at the first) and
        ``result.warnings`` (reported only for valid RBLang), each with a ``message`` and an
        optional ``line``. The server allows up to 5 minutes for very large maps, so you may
        need a client with a longer ``timeout``. This endpoint is newer than the others and not
        in the public OpenAPI specification: a host whose engine predates it answers 404
        (:class:`~rainbird_ai.NotFoundError`).
        """
        result: ValidationResult = self._send(_ops.validate_rblang(rblang))
        return result


class Session:
    """One reasoning session. Create with :meth:`Rainbird.start` or :meth:`Rainbird.resume`.

    The session remembers the last question the engine asked (:attr:`pending_question`)
    so :meth:`answer` and :meth:`skip` can build the correct payload for you.
    """

    def __init__(
        self,
        client: Rainbird,
        session_id: str,
        *,
        km_version_id: str | None = None,
        km_id: str | None = None,
        context_id: str | None = None,
        start_response: StartResponse | None = None,
    ) -> None:
        self.client = client
        self.id = session_id
        self.km_version_id = km_version_id
        self.km_id = km_id
        self.context_id = context_id
        self.start_response = start_response
        self.last_response: QueryResponse | None = None

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, km_id={self.km_id!r})"

    # -- state -------------------------------------------------------------------------

    @property
    def pending(self) -> QuestionResponse | None:
        """The last :class:`QuestionResponse`, or ``None`` after a result / before a query."""
        if isinstance(self.last_response, QuestionResponse):
            return self.last_response
        return None

    @property
    def pending_question(self) -> Question | None:
        """The question currently awaiting an answer, if any."""
        return self.pending.question if self.pending else None

    @property
    def result(self) -> ResultResponse | None:
        """The last :class:`ResultResponse`, or ``None`` if the engine is still asking."""
        if isinstance(self.last_response, ResultResponse):
            return self.last_response
        return None

    def _track(self, response: QueryResponse) -> QueryResponse:
        self.last_response = response
        return response

    def _send(self, request: Request[Any], extra_headers: Mapping[str, str] | None = None) -> Any:
        return self.client._send(request, extra_headers)

    # -- decisions ---------------------------------------------------------------------

    def inject(
        self,
        facts: Any,
        *,
        batch_size: int = MAX_INJECT_BATCH,
        headers: Mapping[str, str] | None = None,
    ) -> InjectResponse:
        """Inject facts. Accepts a :class:`Fact`, a dict, a ``(s, r, o[, certainty])`` tuple,
        or a list of any of those. Large lists are sent in batches of ``batch_size``
        (Rainbird recommends at most 250 per request). Returns the last batch's response.
        A fact may carry ``metadata`` (``{"source": "system" | "user"}``), forwarded as-is.
        ``headers`` are extra HTTP headers for these requests only.
        """
        payload = normalize_facts(facts)
        if batch_size < 1:
            raise RainbirdConfigurationError("batch_size must be at least 1")
        response: InjectResponse | None = None
        for start in range(0, len(payload), batch_size):
            response = self._send(
                _ops.inject(self.id, payload[start : start + batch_size]), headers
            )
        assert response is not None  # payload is never empty (normalize_facts guarantees it)
        return response

    def inject_fact(
        self,
        subject: str,
        relationship: str,
        object: Any,
        certainty: int | float = 100,
        *,
        source: str | None = None,
    ) -> InjectResponse:
        """Inject a single fact. ``source`` (``"system"`` or ``"user"``) is sent as
        ``metadata.source`` and records where the fact came from."""
        fact: dict[str, Any] = {
            "subject": subject,
            "relationship": relationship,
            "object": object,
            "certainty": certainty,
        }
        if source is not None:
            fact["metadata"] = {"source": source}
        return self.inject(fact)

    def query(
        self,
        relationship: str,
        *,
        subject: Any = None,
        object: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> QueryResponse:
        """Ask the engine about ``relationship``. Give the subject, the object, or both.

        Returns a :class:`QuestionResponse` (check ``.is_question``) when more information
        is needed, or a :class:`ResultResponse` with the decision. ``headers`` are extra HTTP
        headers for this request only (``Rainbird(default_headers=...)`` applies to all).
        """
        return self._track(
            self._send(_ops.query(self.id, relationship, subject=subject, object=object), headers)
        )

    def respond(self, answers: Any, *, headers: Mapping[str, str] | None = None) -> QueryResponse:
        """Send prebuilt answers (:class:`Answer` objects or dicts) to the engine.
        ``headers`` are extra HTTP headers for this request only."""
        return self._track(self._send(_ops.respond(self.id, normalize_answers(answers)), headers))

    def answer(
        self,
        value: Any,
        *,
        certainty: int | float = 100,
        question: Question | None = None,
        strict: bool = True,
    ) -> QueryResponse:
        """Answer the pending question (or ``question``) with a plain Python value.

        - First Form: ``True``/``False`` or ``"yes"``/``"no"``.
        - Second Form: the value asked for; a list of values when the question is plural.
        - :data:`~rainbird_ai.SKIP` skips the question (needs ``allowUnknown``).

        ``strict`` validates the value against the question's options/plurality before
        sending; set ``False`` to defer all validation to the API. Options are enforced
        only when the map sets ``canAdd="false"`` (RBLang defaults it to true), so a
        misspelt value is otherwise sent and becomes a new concept instance.
        """
        target = question or self.pending_question
        if target is None:
            raise NoPendingQuestionError(
                "there is no pending question to answer; call query() first or pass question="
            )
        if value is SKIP:
            return self.respond(build_skip(target, strict=strict))
        return self.respond(build_answers(target, value, certainty=certainty, strict=strict))

    def skip(self, *, question: Question | None = None, strict: bool = True) -> QueryResponse:
        """Skip the pending question with ``unanswered: true`` (question must allow unknown)."""
        target = question or self.pending_question
        if target is None:
            raise NoPendingQuestionError("there is no pending question to skip")
        return self.respond(build_skip(target, strict=strict))

    def undo(self, *, headers: Mapping[str, str] | None = None) -> QueryResponse:
        """Roll back the last answers; returns the question (or result) that preceded them.
        ``headers`` are extra HTTP headers for this request only."""
        return self._track(self._send(_ops.undo(self.id), headers))

    def run(
        self,
        relationship: str,
        *,
        subject: Any = None,
        object: Any = None,
        answerer: Answerer,
        max_questions: int = 100,
    ) -> ResultResponse:
        """Query and keep answering questions via ``answerer`` until a result is reached.

        ``answerer(question)`` may return a value, a list of values, an :class:`Answer`
        (or list of them), or :data:`~rainbird_ai.SKIP`. Raises
        :class:`~rainbird_ai.RainbirdError` after ``max_questions`` questions.
        """
        response = self.query(relationship, subject=subject, object=object)
        asked = 0
        while isinstance(response, QuestionResponse):
            asked += 1
            if asked > max_questions:
                raise RainbirdError(
                    f"engine asked more than {max_questions} questions; giving up "
                    f"(last: {response.question.prompt!r})"
                )
            response = self._apply_answer(response.question, answerer(response.question))
        return response

    def _apply_answer(self, question: Question, value: Any) -> QueryResponse:
        if value is SKIP:
            return self.skip(question=question)
        if isinstance(value, Answer) or (
            isinstance(value, (list, tuple)) and value and all(isinstance(v, Answer) for v in value)
        ):
            return self.respond(value)
        return self.answer(value, question=question)

    # -- evidence & analysis -----------------------------------------------------------

    def evidence(self, fact_id: str, *, evidence_key: str | None = None) -> Evidence:
        """Why a fact holds: its source and the rule/conditions that inferred it."""
        return self.client.evidence(self.id, fact_id, evidence_key=evidence_key)

    def evidence_tree(
        self,
        fact_id: str,
        *,
        max_depth: int | None = None,
        evidence_key: str | None = None,
    ) -> list[Evidence]:
        """Walk the whole evidence tree breadth-first, following ``factID`` references.

        Returns every :class:`Evidence` node reached (root first), each fetched once.
        """
        return list(
            self.iter_evidence_tree(fact_id, max_depth=max_depth, evidence_key=evidence_key)
        )

    def iter_evidence_tree(
        self,
        fact_id: str,
        *,
        max_depth: int | None = None,
        evidence_key: str | None = None,
    ) -> Iterator[Evidence]:
        """Lazy version of :meth:`evidence_tree`."""
        seen: set[str] = set()
        frontier: list[tuple[str, int]] = [(fact_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self.evidence(current, evidence_key=evidence_key)
            yield node
            if max_depth is not None and depth >= max_depth:
                continue
            frontier.extend(
                (child, depth + 1) for child in node.child_fact_ids if child not in seen
            )

    def evidence_url(self, fact_id: str, *, studio_url: str | None = None) -> str:
        """Link to the Evidence Tree UI in Rainbird Studio for ``fact_id``."""
        return self.client.evidence_url(self.id, fact_id, studio_url=studio_url)

    @overload
    def interactions(
        self, *, format: Literal["json"] = ..., interaction_key: str | None = ...
    ) -> list[InteractionEvent]: ...

    @overload
    def interactions(self, *, format: Literal["csv"], interaction_key: str | None = ...) -> str: ...

    def interactions(
        self,
        *,
        format: Literal["json", "csv"] = "json",
        interaction_key: str | None = None,
    ) -> list[InteractionEvent] | str:
        """Interaction log (must be enabled per knowledge-map version in Studio).

        Returns a list of :class:`InteractionEvent` for ``format="json"`` or the raw CSV
        text for ``format="csv"``.
        """
        key = interaction_key if interaction_key is not None else self.client.config.interaction_key
        if format.lower() == "csv":
            csv_text: str = self._send(_ops.interactions_csv(self.id, key))
            return csv_text
        events: list[InteractionEvent] = self._send(_ops.interactions_json(self.id, key))
        return events

    def knowledge_map(self) -> KnowledgeMapVersion:
        """Knowledge map id/name and the version this session is using."""
        result: KnowledgeMapVersion = self._send(_ops.session_version(self.id))
        return result

    def facts(self, *, relationships: str | Iterable[str] | None = None) -> SessionFacts:
        """All facts visible to the session (global, context and local).

        ``relationships`` filters to one relationship name or several.
        """
        result: SessionFacts = self._send(_ops.session_facts(self.id, relationships))
        return result

    # -- natural language (beta) -------------------------------------------------------

    def interact(self, prompt: str) -> InteractResponse:
        """Natural-language interaction (beta): inject facts, query and answer in plain text.

        The response is a question, a result, or an error (``response_type``); errors are
        returned, not raised, because "I could not find a query in that" is a normal
        outcome. Use ``.raise_for_error()`` if you want an exception.
        """
        result: InteractResponse = self._send(_ops.interact(self.id, prompt))
        return result

    def explain(
        self, fact_id: str, *, language: str = "en", instructions: str | None = None
    ) -> ExplainResponse:
        """Natural-language explanation (beta) of why ``fact_id`` holds."""
        result: ExplainResponse = self._send(
            _ops.explain(self.id, fact_id, language=language, instructions=instructions)
        )
        return result


def _check_evidence_mode(evidence: str) -> None:
    if evidence not in ("none", "root", "tree"):
        raise RainbirdConfigurationError(
            f"evidence must be 'none', 'root' or 'tree', got {evidence!r}"
        )


def _seconds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_polling(poll_interval: float, timeout: float | None) -> None:
    if not _seconds(poll_interval) or poll_interval <= 0:
        raise RainbirdConfigurationError(
            f"poll_interval must be a positive number of seconds, got {poll_interval!r}"
        )
    if timeout is not None and (not _seconds(timeout) or timeout < 0):
        raise RainbirdConfigurationError(
            f"timeout must be None or a number of seconds >= 0, got {timeout!r}"
        )


def _final_run(run: TestRun) -> TestRun:
    """A finished run, or :class:`TestRunFailedError` when its status is ``failed``."""
    if run.is_failed:
        detail = f": {run.error}" if run.error else ""
        raise TestRunFailedError(f"test run {run.run_id} failed{detail}", run=run)
    return run


def _poll_delay(run: TestRun, started: float, poll_interval: float, timeout: float | None) -> float:
    """Seconds to wait before polling ``run`` again; :class:`TestRunTimeoutError` once
    ``timeout`` seconds have passed since ``started`` (a ``time.monotonic()`` reading)."""
    if timeout is None:
        return poll_interval
    remaining = started + timeout - time.monotonic()
    if remaining <= 0:
        raise TestRunTimeoutError(
            f"test run {run.run_id} did not finish within {timeout:g} seconds "
            f"(status: {run.status})",
            run=run,
        )
    return min(poll_interval, remaining)


def engine_version(
    base_url: str | None = None,
    *,
    engine: str | None = None,
    timeout: float | httpx.Timeout | Unset | None = UNSET,
    http_client: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
) -> str:
    """Engine version from ``GET /version`` without an API key (the endpoint is public).

    ``base_url`` defaults to ``RAINBIRD_API_URL`` or the Community host. ``retry`` defaults
    to :class:`RetryPolicy` (pass :data:`NO_RETRY` for a single attempt).
    """
    config = ClientConfig.resolve(
        api_key="",
        base_url=base_url,
        evidence_key=None,
        interaction_key=None,
        engine=engine,
        timeout=timeout,
        retry=retry,
        default_headers=None,
        require_api_key=False,
    )
    transport = SyncTransport(config, http_client)
    try:
        result: str = transport.send(_ops.version())
        return result
    finally:
        transport.close()
