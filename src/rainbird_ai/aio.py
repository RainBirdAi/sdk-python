"""Asynchronous client: :class:`AsyncRainbird` and :class:`AsyncSession`.

Same API as :mod:`rainbird_ai.client`, with ``await``::

    from rainbird_ai.aio import AsyncRainbird

    async with AsyncRainbird(api_key="...") as rb:
        session = await rb.start("your-km-id")
        response = await session.query("speaks", subject="Tom")
        while response.is_question:
            response = await session.answer("German")
"""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, overload

import httpx

from . import _ops
from ._answers import SKIP, build_answers, build_skip, normalize_answers, normalize_facts
from ._http import (
    ENV_KM_ID,
    UNSET,
    AsyncTransport,
    ClientConfig,
    Request,
    RetryPolicy,
    Unset,
    async_sleep,
)
from .client import (
    MAX_INJECT_BATCH,
    _check_evidence_mode,
    _check_polling,
    _final_run,
    _poll_delay,
)
from .errors import NoPendingQuestionError, RainbirdConfigurationError, RainbirdError
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

__all__ = ["AsyncAnswerer", "AsyncRainbird", "AsyncSession", "async_engine_version"]

AsyncAnswerer = Callable[[Question], Any | Awaitable[Any]]
"""Callback for :meth:`AsyncSession.run`; may be a plain function or a coroutine function."""


class AsyncRainbird:
    """Async entry point for the Rainbird API. See :class:`rainbird_ai.Rainbird` for arguments."""

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
        http_client: httpx.AsyncClient | None = None,
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
        self._transport = AsyncTransport(self.config, http_client)

    @property
    def base_url(self) -> str:
        """Resolved API host this client talks to (no trailing slash)."""
        return self.config.base_url

    @property
    def studio_url(self) -> str:
        """Rainbird Studio host matching :attr:`base_url` (used for evidence links)."""
        return studio_url_for(self.config.base_url)

    @property
    def http(self) -> httpx.AsyncClient:
        """The underlying ``httpx.AsyncClient``."""
        return self._transport.http

    async def aclose(self) -> None:
        """Close the HTTP connection pool (no-op for a user-supplied ``http_client``)."""
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncRainbird:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncRainbird(base_url={self.config.base_url!r})"

    async def _send(
        self, request: Request[Any], extra_headers: Mapping[str, str] | None = None
    ) -> Any:
        return await self._transport.send(request, extra_headers=extra_headers)

    # -- sessions ----------------------------------------------------------------------

    async def start(
        self,
        km_id: str | None = None,
        *,
        use_draft: bool | None = None,
        version: int | None = None,
        context_id: str | None = None,
    ) -> AsyncSession:
        """Start a reasoning session. See :meth:`rainbird_ai.Rainbird.start`."""
        resolved = km_id if km_id is not None else os.environ.get(ENV_KM_ID)
        if not resolved or not str(resolved).strip():
            raise RainbirdConfigurationError(
                "km_id is required: pass it to start() or set RAINBIRD_KM_ID"
            )
        response: StartResponse = await self._send(
            _ops.start(str(resolved), use_draft=use_draft, version=version, context_id=context_id)
        )
        return AsyncSession(
            self,
            response.session_id,
            km_version_id=response.km_version_id,
            km_id=str(resolved),
            context_id=context_id,
            start_response=response,
        )

    def resume(self, session_id: str, *, km_id: str | None = None) -> AsyncSession:
        """Attach to an existing session by ID (no request is made)."""
        if not session_id or not str(session_id).strip():
            raise RainbirdConfigurationError("session_id must not be empty")
        return AsyncSession(self, str(session_id), km_id=km_id)

    async def decide(
        self,
        relationship: str,
        *,
        km_id: str | None = None,
        subject: Any = None,
        object: Any = None,
        facts: Any = None,
        answerer: AsyncAnswerer | None = None,
        max_questions: int = 100,
        evidence: Literal["none", "root", "tree"] = "none",
        evidence_depth: int | None = None,
        evidence_key: str | None = None,
        use_draft: bool | None = None,
        version: int | None = None,
        context_id: str | None = None,
    ) -> Decision:
        """Async version of :meth:`rainbird_ai.Rainbird.decide` (``answerer`` may be async)."""
        _check_evidence_mode(evidence)
        # Validate the query and the facts before starting a session that would be left unused.
        query = _ops.query_body(relationship, subject=subject, object=object)
        payload = normalize_facts(facts) if facts else None
        session = await self.start(
            km_id, use_draft=use_draft, version=version, context_id=context_id
        )
        if payload:
            await session.inject(payload)
        if answerer is None:
            response = await session.query(relationship, subject=subject, object=object)
        else:
            response = await session.run(
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
            nodes = [await session.evidence(fact_id, evidence_key=evidence_key)]
        elif fact_id and evidence == "tree":
            nodes = await session.evidence_tree(
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

    async def version(self) -> str:
        """Async version of :meth:`rainbird_ai.Rainbird.version`."""
        result: str = await self._send(_ops.version())
        return result

    # -- evidence ----------------------------------------------------------------------

    async def evidence(
        self, session_id: str, fact_id: str, *, evidence_key: str | None = None
    ) -> Evidence:
        """Async version of :meth:`rainbird_ai.Rainbird.evidence`."""
        key = evidence_key if evidence_key is not None else self.config.evidence_key
        result: Evidence = await self._send(_ops.evidence(fact_id, session_id, key))
        return result

    def evidence_url(self, session_id: str, fact_id: str, *, studio_url: str | None = None) -> str:
        """Link to the Evidence Tree UI in Rainbird Studio for a fact."""
        studio = (studio_url or self.studio_url).rstrip("/")
        return f"{studio}/evidence?id={fact_id}&api={self.config.base_url}&sid={session_id}"

    # -- knowledge maps (undocumented endpoints) ---------------------------------------

    async def get_map_source(self, km_id: str) -> KnowledgeMapSource:
        """RBLang source via ``GET /analysis/file/{kmID}`` (undocumented endpoint)."""
        result: KnowledgeMapSource = await self._send(_ops.map_source(km_id))
        return result

    async def create_map(
        self, name: str, rblang: str, description: str | None = None
    ) -> CreateMapResult:
        """Async version of :meth:`rainbird_ai.Rainbird.create_map` (undocumented endpoint)."""
        result: CreateMapResult = await self._send(_ops.create_map(name, rblang, description))
        return result

    # -- knowledge map tests, readme and validation (undocumented endpoints) -----------

    async def create_tests(
        self, km_id: str, tests: Sequence[Mapping[str, Any]] | Mapping[str, Any] | TestSuite
    ) -> TestSuiteCreated:
        """Async version of :meth:`rainbird_ai.Rainbird.create_tests` (undocumented endpoint)."""
        result: TestSuiteCreated = await self._send(_ops.create_tests(km_id, tests))
        return result

    async def get_tests(self, km_id: str) -> TestSuite:
        """Async version of :meth:`rainbird_ai.Rainbird.get_tests` (undocumented endpoint)."""
        result: TestSuite = await self._send(_ops.get_tests(km_id))
        return result

    async def start_test_run(self, km_id: str) -> TestRunStarted:
        """Async version of :meth:`rainbird_ai.Rainbird.start_test_run` (undocumented endpoint)."""
        result: TestRunStarted = await self._send(_ops.start_test_run(km_id))
        return result

    async def get_test_run(self, run_id: str) -> TestRun:
        """Async version of :meth:`rainbird_ai.Rainbird.get_test_run` (undocumented endpoint)."""
        result: TestRun = await self._send(_ops.get_test_run(run_id))
        return result

    async def wait_for_test_run(
        self, run_id: str, *, poll_interval: float = 1.0, timeout: float | None = 300.0
    ) -> TestRun:
        """Async version of :meth:`rainbird_ai.Rainbird.wait_for_test_run` (``anyio.sleep``)."""
        _check_polling(poll_interval, timeout)
        started = time.monotonic()
        while True:
            run = await self.get_test_run(run_id)
            if run.is_finished:
                return _final_run(run)
            await async_sleep(_poll_delay(run, started, poll_interval, timeout))

    async def run_tests(
        self, km_id: str, *, poll_interval: float = 1.0, timeout: float | None = 300.0
    ) -> TestRun:
        """Async version of :meth:`rainbird_ai.Rainbird.run_tests`."""
        _check_polling(poll_interval, timeout)  # before starting a run that nobody waits for
        run_id = (await self.start_test_run(km_id)).run_id
        return await self.wait_for_test_run(run_id, poll_interval=poll_interval, timeout=timeout)

    async def add_readme(self, km_id: str, readme: str) -> ReadmeAdded:
        """Async version of :meth:`rainbird_ai.Rainbird.add_readme` (undocumented endpoint)."""
        result: ReadmeAdded = await self._send(_ops.add_readme(km_id, readme))
        return result

    async def validate_rblang(self, rblang: str) -> ValidationResult:
        """Async version of :meth:`rainbird_ai.Rainbird.validate_rblang` (undocumented endpoint)."""
        result: ValidationResult = await self._send(_ops.validate_rblang(rblang))
        return result


class AsyncSession:
    """One reasoning session (async). See :class:`rainbird_ai.Session` for semantics."""

    def __init__(
        self,
        client: AsyncRainbird,
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
        return f"AsyncSession(id={self.id!r}, km_id={self.km_id!r})"

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

    async def _send(
        self, request: Request[Any], extra_headers: Mapping[str, str] | None = None
    ) -> Any:
        return await self.client._send(request, extra_headers)

    # -- decisions ---------------------------------------------------------------------

    async def inject(
        self,
        facts: Any,
        *,
        batch_size: int = MAX_INJECT_BATCH,
        headers: Mapping[str, str] | None = None,
    ) -> InjectResponse:
        """Async version of :meth:`rainbird_ai.Session.inject`."""
        payload = normalize_facts(facts)
        if batch_size < 1:
            raise RainbirdConfigurationError("batch_size must be at least 1")
        response: InjectResponse | None = None
        for start in range(0, len(payload), batch_size):
            response = await self._send(
                _ops.inject(self.id, payload[start : start + batch_size]), headers
            )
        assert response is not None
        return response

    async def inject_fact(
        self,
        subject: str,
        relationship: str,
        object: Any,
        certainty: int | float = 100,
        *,
        source: str | None = None,
    ) -> InjectResponse:
        """Async version of :meth:`rainbird_ai.Session.inject_fact`."""
        fact: dict[str, Any] = {
            "subject": subject,
            "relationship": relationship,
            "object": object,
            "certainty": certainty,
        }
        if source is not None:
            fact["metadata"] = {"source": source}
        return await self.inject(fact)

    async def query(
        self,
        relationship: str,
        *,
        subject: Any = None,
        object: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> QueryResponse:
        """Async version of :meth:`rainbird_ai.Session.query`."""
        return self._track(
            await self._send(
                _ops.query(self.id, relationship, subject=subject, object=object), headers
            )
        )

    async def respond(
        self, answers: Any, *, headers: Mapping[str, str] | None = None
    ) -> QueryResponse:
        """Async version of :meth:`rainbird_ai.Session.respond`."""
        return self._track(
            await self._send(_ops.respond(self.id, normalize_answers(answers)), headers)
        )

    async def answer(
        self,
        value: Any,
        *,
        certainty: int | float = 100,
        question: Question | None = None,
        strict: bool = True,
    ) -> QueryResponse:
        """Async version of :meth:`rainbird_ai.Session.answer`."""
        target = question or self.pending_question
        if target is None:
            raise NoPendingQuestionError(
                "there is no pending question to answer; call query() first or pass question="
            )
        if value is SKIP:
            return await self.respond(build_skip(target, strict=strict))
        return await self.respond(build_answers(target, value, certainty=certainty, strict=strict))

    async def skip(self, *, question: Question | None = None, strict: bool = True) -> QueryResponse:
        """Async version of :meth:`rainbird_ai.Session.skip`."""
        target = question or self.pending_question
        if target is None:
            raise NoPendingQuestionError("there is no pending question to skip")
        return await self.respond(build_skip(target, strict=strict))

    async def undo(self, *, headers: Mapping[str, str] | None = None) -> QueryResponse:
        """Async version of :meth:`rainbird_ai.Session.undo`."""
        return self._track(await self._send(_ops.undo(self.id), headers))

    async def run(
        self,
        relationship: str,
        *,
        subject: Any = None,
        object: Any = None,
        answerer: AsyncAnswerer,
        max_questions: int = 100,
    ) -> ResultResponse:
        """Query and keep answering via ``answerer`` (sync or async) until a result is reached."""
        response = await self.query(relationship, subject=subject, object=object)
        asked = 0
        while isinstance(response, QuestionResponse):
            asked += 1
            if asked > max_questions:
                raise RainbirdError(
                    f"engine asked more than {max_questions} questions; giving up "
                    f"(last: {response.question.prompt!r})"
                )
            value = answerer(response.question)
            if inspect.isawaitable(value):
                value = await value
            response = await self._apply_answer(response.question, value)
        return response

    async def _apply_answer(self, question: Question, value: Any) -> QueryResponse:
        if value is SKIP:
            return await self.skip(question=question)
        if isinstance(value, Answer) or (
            isinstance(value, (list, tuple)) and value and all(isinstance(v, Answer) for v in value)
        ):
            return await self.respond(value)
        return await self.answer(value, question=question)

    # -- evidence & analysis -----------------------------------------------------------

    async def evidence(self, fact_id: str, *, evidence_key: str | None = None) -> Evidence:
        """Async version of :meth:`rainbird_ai.Session.evidence`."""
        return await self.client.evidence(self.id, fact_id, evidence_key=evidence_key)

    async def evidence_tree(
        self,
        fact_id: str,
        *,
        max_depth: int | None = None,
        evidence_key: str | None = None,
    ) -> list[Evidence]:
        """Async version of :meth:`rainbird_ai.Session.evidence_tree`."""
        return [
            node
            async for node in self.iter_evidence_tree(
                fact_id, max_depth=max_depth, evidence_key=evidence_key
            )
        ]

    async def iter_evidence_tree(
        self,
        fact_id: str,
        *,
        max_depth: int | None = None,
        evidence_key: str | None = None,
    ) -> AsyncIterator[Evidence]:
        """Async generator version of :meth:`evidence_tree`."""
        seen: set[str] = set()
        frontier: list[tuple[str, int]] = [(fact_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = await self.evidence(current, evidence_key=evidence_key)
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
    async def interactions(
        self, *, format: Literal["json"] = ..., interaction_key: str | None = ...
    ) -> list[InteractionEvent]: ...

    @overload
    async def interactions(
        self, *, format: Literal["csv"], interaction_key: str | None = ...
    ) -> str: ...

    async def interactions(
        self,
        *,
        format: Literal["json", "csv"] = "json",
        interaction_key: str | None = None,
    ) -> list[InteractionEvent] | str:
        """Async version of :meth:`rainbird_ai.Session.interactions`."""
        key = interaction_key if interaction_key is not None else self.client.config.interaction_key
        if format.lower() == "csv":
            csv_text: str = await self._send(_ops.interactions_csv(self.id, key))
            return csv_text
        events: list[InteractionEvent] = await self._send(_ops.interactions_json(self.id, key))
        return events

    async def knowledge_map(self) -> KnowledgeMapVersion:
        """Async version of :meth:`rainbird_ai.Session.knowledge_map`."""
        result: KnowledgeMapVersion = await self._send(_ops.session_version(self.id))
        return result

    async def facts(self, *, relationships: str | Iterable[str] | None = None) -> SessionFacts:
        """Async version of :meth:`rainbird_ai.Session.facts`."""
        result: SessionFacts = await self._send(_ops.session_facts(self.id, relationships))
        return result

    # -- natural language (beta) -------------------------------------------------------

    async def interact(self, prompt: str) -> InteractResponse:
        """Async version of :meth:`rainbird_ai.Session.interact`."""
        result: InteractResponse = await self._send(_ops.interact(self.id, prompt))
        return result

    async def explain(
        self, fact_id: str, *, language: str = "en", instructions: str | None = None
    ) -> ExplainResponse:
        """Async version of :meth:`rainbird_ai.Session.explain`."""
        result: ExplainResponse = await self._send(
            _ops.explain(self.id, fact_id, language=language, instructions=instructions)
        )
        return result


async def async_engine_version(
    base_url: str | None = None,
    *,
    engine: str | None = None,
    timeout: float | httpx.Timeout | Unset | None = UNSET,
    http_client: httpx.AsyncClient | None = None,
    retry: RetryPolicy | None = None,
) -> str:
    """Async version of :func:`rainbird_ai.engine_version` (no API key needed)."""
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
    transport = AsyncTransport(config, http_client)
    try:
        result: str = await transport.send(_ops.version())
        return result
    finally:
        await transport.aclose()
