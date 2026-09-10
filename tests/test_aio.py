"""Tests for the asynchronous client (``rainbird_ai.aio``).

Every behaviour of :class:`AsyncRainbird` / :class:`AsyncSession` is exercised against the
in-memory fake API from ``conftest.py``; where it matters, the exact request method, path,
headers, query parameters and JSON payload are asserted. The final class checks structural
parity with the synchronous client (``rainbird_ai.client``).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest

import rainbird_ai
from rainbird_ai import (
    NO_RETRY,
    SKIP,
    Answer,
    AuthenticationError,
    BadRequestError,
    CreateMapResult,
    Evidence,
    ExplainResponse,
    Fact,
    FactSource,
    InjectResponse,
    InteractionEvent,
    InteractResponse,
    InvalidAnswerError,
    InvalidFactError,
    KnowledgeMapError,
    KnowledgeMapSource,
    KnowledgeMapVersion,
    NaturalLanguageError,
    NoPendingQuestionError,
    NotFoundError,
    PermissionDeniedError,
    Question,
    QuestionForm,
    QuestionResponse,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTimeoutError,
    RateLimitError,
    ResultResponse,
    RetryPolicy,
    ServerError,
    SessionFacts,
    UnexpectedResponseError,
)
from rainbird_ai import aio as aio_module
from rainbird_ai.aio import AsyncRainbird, AsyncSession, async_engine_version
from rainbird_ai.client import MAX_INJECT_BATCH, Rainbird, Session, engine_version
from tests.conftest import (
    API_KEY,
    BASE_URL,
    EVIDENCE_LEAVES,
    EVIDENCE_ROOT,
    INTERACT_ERROR,
    INTERACT_QUESTION,
    KM_ID,
    KM_VERSION_ID,
    QUESTION_FIRST_FORM,
    QUESTION_PLURAL_LOSS,
    QUESTION_SECOND_OBJECT,
    QUESTION_SECOND_SUBJECT,
    RBLANG,
    RESULT_TOM_SPEAKS_GERMAN,
    FakeRainbird,
)

pytestmark = pytest.mark.anyio

ROOT_FACT: str = EVIDENCE_ROOT["factID"]
IF_FACT, AF_FACT = list(
    EVIDENCE_LEAVES
)  # "Tom lives in Germany", "Germany national language German"
STUDIO_URL = "https://fake.rainbird.ai"  # studio host derived from BASE_URL
UTC = dt.timezone.utc


# --------------------------------------------------------------------------- helpers / fixtures


@pytest.fixture
async def rb(aclient: AsyncRainbird) -> AsyncIterator[AsyncRainbird]:
    """The shared ``aclient`` fixture, entered as an async context manager."""
    async with aclient:
        yield aclient
    # aclose() is a no-op for the user-supplied httpx client; release it ourselves.
    await aclient.http.aclose()


@pytest.fixture
async def session(rb: AsyncRainbird) -> AsyncSession:
    return await rb.start(KM_ID)


@asynccontextmanager
async def make_client(
    fake: FakeRainbird, *, transport: httpx.AsyncBaseTransport | None = None, **kwargs: Any
) -> AsyncIterator[AsyncRainbird]:
    """An ``AsyncRainbird`` wired to ``fake`` with extra constructor options."""
    kwargs.setdefault("api_key", API_KEY)
    kwargs.setdefault("base_url", BASE_URL)
    async with httpx.AsyncClient(transport=transport or httpx.MockTransport(fake.handle)) as http:
        async with AsyncRainbird(http_client=http, **kwargs) as client:
            yield client


def raising_transport(exc: Exception) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def recording_transport(fake: FakeRainbird, seen: list[httpx.Request]) -> httpx.MockTransport:
    """``fake`` behind a transport that also keeps every raw ``httpx.Request`` in ``seen``.

    ``FakeRainbird.requests`` records method/path/headers only; the raw request is needed to
    inspect the full URL and the per-request timeout (``request.extensions["timeout"]``).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return fake.handle(request)

    return httpx.MockTransport(handler)


def evidence_calls(fake: FakeRainbird) -> list[str]:
    return [c.path for c in fake.calls("/analysis/evidence/")]


# --------------------------------------------------------------------------- client lifecycle


class TestClientLifecycle:
    async def test_repr(self, aclient: AsyncRainbird) -> None:
        assert repr(aclient) == f"AsyncRainbird(base_url={BASE_URL!r})"

    async def test_base_url_and_studio_url(self, rb: AsyncRainbird) -> None:
        assert rb.base_url == BASE_URL
        assert rb.studio_url == STUDIO_URL

    async def test_http_exposes_underlying_async_client(self, rb: AsyncRainbird) -> None:
        assert isinstance(rb.http, httpx.AsyncClient)

    async def test_aenter_returns_self(self, aclient: AsyncRainbird) -> None:
        async with aclient as entered:
            assert entered is aclient
        await aclient.http.aclose()

    async def test_aexit_does_not_close_user_supplied_http_client(
        self, aclient: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        http = aclient.http
        async with aclient:
            pass
        assert not http.is_closed
        # still usable afterwards: the caller owns the httpx client
        assert await aclient.version() == "4.89.0"
        assert fake.last.path == "/version"
        await http.aclose()
        assert http.is_closed

    async def test_aclose_is_noop_for_user_supplied_http_client(
        self, aclient: AsyncRainbird
    ) -> None:
        await aclient.aclose()
        assert not aclient.http.is_closed
        await aclient.http.aclose()

    async def test_aexit_closes_owned_http_client(self) -> None:
        client = AsyncRainbird(api_key=API_KEY, base_url=BASE_URL)
        http = client.http
        assert isinstance(http, httpx.AsyncClient)
        async with client:
            assert not http.is_closed
        assert http.is_closed

    async def test_aclose_twice_on_owned_client_is_safe(self) -> None:
        client = AsyncRainbird(api_key=API_KEY, base_url=BASE_URL)
        await client.aclose()
        await client.aclose()
        assert client.http.is_closed

    async def test_owned_client_uses_configured_timeout(self) -> None:
        client = AsyncRainbird(api_key=API_KEY, base_url=BASE_URL, timeout=7)
        try:
            assert client.http.timeout == httpx.Timeout(7)
            assert client.config.timeout == 7 and client.config.timeout_explicit is True
        finally:
            await client.aclose()

    async def test_owned_client_defaults_to_30_seconds_and_no_redirects(self) -> None:
        client = AsyncRainbird(api_key=API_KEY, base_url=BASE_URL)
        try:
            assert client.config.timeout == 30.0
            assert client.config.timeout_explicit is False
            assert client.http.timeout == httpx.Timeout(30.0)
            assert client.http.follow_redirects is False
        finally:
            await client.aclose()

    async def test_timeout_defaults_to_unset_sentinel(self, rb: AsyncRainbird) -> None:
        assert rb.config.timeout == 30.0
        assert rb.config.timeout_explicit is False
        async with AsyncRainbird(
            api_key=API_KEY, base_url=BASE_URL, timeout=rainbird_ai.UNSET
        ) as explicit_unset:
            assert explicit_unset.config.timeout == 30.0
            assert explicit_unset.config.timeout_explicit is False

    async def test_user_client_keeps_its_own_timeout_by_default(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        async with httpx.AsyncClient(
            transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0)
        ) as http:
            async with AsyncRainbird(
                api_key=API_KEY, base_url=BASE_URL, http_client=http
            ) as client:
                session = await client.start(KM_ID)
                await session.query("speaks", subject="Tom")
        assert len(seen) == 2
        assert all(r.extensions["timeout"] == httpx.Timeout(9.0).as_dict() for r in seen)

    async def test_explicit_timeout_is_applied_per_request_with_user_client(
        self, fake: FakeRainbird
    ) -> None:
        seen: list[httpx.Request] = []
        async with httpx.AsyncClient(
            transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0)
        ) as http:
            async with AsyncRainbird(
                api_key=API_KEY, base_url=BASE_URL, http_client=http, timeout=5
            ) as client:
                session = await client.start(KM_ID)
                await session.query("speaks", subject="Tom")
                await client.version()
            assert http.timeout == httpx.Timeout(9.0)  # the caller's client itself is untouched
        assert len(seen) == 3
        assert all(r.extensions["timeout"] == httpx.Timeout(5).as_dict() for r in seen)

    @pytest.mark.parametrize("timeout", [None, httpx.Timeout(5.0, connect=1.0)])
    async def test_explicit_none_or_httpx_timeout_applied_per_request(
        self, fake: FakeRainbird, timeout: Any
    ) -> None:
        seen: list[httpx.Request] = []
        async with make_client(
            fake, transport=recording_transport(fake, seen), timeout=timeout
        ) as client:
            assert client.config.timeout_explicit is True
            await client.version()
        assert seen[-1].extensions["timeout"] == httpx.Timeout(timeout).as_dict()

    async def test_missing_api_key_raises(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="API key"):
            AsyncRainbird(base_url=BASE_URL)

    async def test_api_key_from_environment(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", API_KEY)
        async with make_client(fake, api_key=None) as client:
            assert await client.version() == "4.89.0"
            assert client.config.api_key == API_KEY

    async def test_default_headers_sent_on_every_request(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        await rb.version()
        headers = fake.last.headers
        assert headers["x-api-key"] == API_KEY
        assert headers["user-agent"].startswith("rainbird-ai-python/")
        assert "x-rainbird-engine" not in headers

    async def test_engine_and_custom_headers(self, fake: FakeRainbird) -> None:
        async with make_client(
            fake, engine="alt-engine", default_headers={"X-Trace": "abc123"}
        ) as client:
            await client.version()
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"
        assert fake.last.headers["x-trace"] == "abc123"


# --------------------------------------------------------------------------- start / resume


class TestStart:
    async def test_start_returns_session(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        session = await rb.start(KM_ID)
        assert isinstance(session, AsyncSession)
        assert session.client is rb
        assert session.id in fake.sessions
        assert session.km_id == KM_ID
        assert session.km_version_id == KM_VERSION_ID
        assert session.context_id is None
        assert session.start_response is not None
        assert session.start_response.session_id == session.id
        assert session.start_response.km_version_id == KM_VERSION_ID

    async def test_start_request(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        await rb.start(KM_ID)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {}
        assert fake.last.json is None
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["accept"] == "application/json"

    async def test_start_use_draft(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        await rb.start(KM_ID, use_draft=True)
        assert fake.last.params == {"useDraft": "true"}

    async def test_start_use_draft_false_is_sent_explicitly(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        await rb.start(KM_ID, use_draft=False)
        assert fake.last.params == {"useDraft": "false"}

    async def test_start_version(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        await rb.start(KM_ID, version=3)
        assert fake.last.params == {"version": "3"}

    async def test_start_context_id(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        session = await rb.start(KM_ID, context_id="ctx-42")
        assert fake.last.params == {"contextid": "ctx-42"}
        assert session.context_id == "ctx-42"

    async def test_start_use_draft_and_version_are_exclusive(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="cannot be used together"):
            await rb.start(KM_ID, use_draft=True, version=2)
        assert fake.requests == []

    async def test_start_rejects_non_positive_version(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="positive integer"):
            await rb.start(KM_ID, version=0)
        assert fake.requests == []

    async def test_start_without_km_id_raises(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="km_id is required"):
            await rb.start()
        with pytest.raises(RainbirdConfigurationError):
            await rb.start("   ")
        assert fake.requests == []

    async def test_start_km_id_from_environment(
        self, rb: AsyncRainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", KM_ID)
        session = await rb.start()
        assert session.km_id == KM_ID
        assert fake.last.path == f"/start/{KM_ID}"

    async def test_start_with_invalid_key_is_authentication_error(self, fake: FakeRainbird) -> None:
        async with make_client(fake, api_key="wrong") as client:
            with pytest.raises(AuthenticationError) as info:
                await client.start(KM_ID)
        assert info.value.status_code == 401
        assert info.value.errors == ["Your API key is invalid"]

    async def test_new_session_has_no_state(self, session: AsyncSession) -> None:
        assert session.last_response is None
        assert session.pending is None
        assert session.pending_question is None
        assert session.result is None

    async def test_session_repr(self, session: AsyncSession) -> None:
        assert repr(session) == f"AsyncSession(id={session.id!r}, km_id={KM_ID!r})"


class TestResume:
    async def test_resume_makes_no_request(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        session = rb.resume("existing-session", km_id=KM_ID)
        assert fake.requests == []
        assert isinstance(session, AsyncSession)
        assert session.id == "existing-session"
        assert session.km_id == KM_ID
        assert session.km_version_id is None
        assert session.start_response is None
        assert session.pending is None

    async def test_resume_is_synchronous(self) -> None:
        assert not inspect.iscoroutinefunction(AsyncRainbird.resume)

    async def test_resume_without_km_id(self, rb: AsyncRainbird) -> None:
        session = rb.resume("sid-1")
        assert session.km_id is None
        assert repr(session) == "AsyncSession(id='sid-1', km_id=None)"

    async def test_resume_rejects_empty_id(self, rb: AsyncRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="session_id"):
            rb.resume("")
        with pytest.raises(RainbirdConfigurationError):
            rb.resume("   ")

    async def test_resumed_session_can_query(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        fake.sessions["known"] = {"km": KM_ID, "params": {}, "facts": [], "answers": []}
        session = rb.resume("known")
        response = await session.query("speaks", subject="Tom")
        assert response.is_result
        assert fake.last.path == "/known/query"

    async def test_resumed_unknown_session_is_not_found(self, rb: AsyncRainbird) -> None:
        session = rb.resume("no-such-session")
        with pytest.raises(NotFoundError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.status_code == 404
        assert info.value.message == "Session not found"


# --------------------------------------------------------------------------- inject


class TestInject:
    async def test_inject_fact_payload(self, session: AsyncSession, fake: FakeRainbird) -> None:
        response = await session.inject_fact("Tom", "lives in", "Germany")
        assert isinstance(response, InjectResponse)
        assert response.ok
        assert response.result == "OK"
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/inject"
        assert fake.last.json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100}
        ]

    async def test_inject_fact_with_certainty(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.inject_fact("Tom", "lives in", "Germany", certainty=60)
        assert fake.last.json[0]["certainty"] == 60

    async def test_inject_single_tuple(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.inject(("Tom", "lives in", "Germany", 80))
        assert fake.last.json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 80}
        ]

    async def test_inject_mixed_list_in_one_request(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.inject(
            [
                Fact("A", "rel", "B"),
                {"subject": "C", "relationship": "rel", "object": "D", "certainty": 50},
                ("E", "rel", 3),
            ]
        )
        assert len(fake.calls("inject")) == 1
        assert fake.last.json == [
            {"subject": "A", "relationship": "rel", "object": "B", "certainty": 100},
            {"subject": "C", "relationship": "rel", "object": "D", "certainty": 50},
            {"subject": "E", "relationship": "rel", "object": 3, "certainty": 100},
        ]

    async def test_inject_batches_in_chunks_of_250_by_default(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        assert MAX_INJECT_BATCH == 250
        facts = [("Tom", "knows", f"thing-{i}") for i in range(501)]
        response = await session.inject(facts)
        calls = fake.calls("inject")
        assert [len(c.json) for c in calls] == [250, 250, 1]
        assert all(c.path == f"/{session.id}/inject" for c in calls)
        stored = fake.sessions[session.id]["facts"]
        assert [f["object"] for f in stored] == [f"thing-{i}" for i in range(501)]
        assert response.ok

    async def test_inject_custom_batch_size_preserves_order(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = [("S", "r", f"o{i}") for i in range(5)]
        await session.inject(facts, batch_size=2)
        calls = fake.calls("inject")
        assert [[f["object"] for f in c.json] for c in calls] == [
            ["o0", "o1"],
            ["o2", "o3"],
            ["o4"],
        ]

    async def test_inject_exact_multiple_of_batch_size(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.inject([("S", "r", f"o{i}") for i in range(4)], batch_size=2)
        assert [len(c.json) for c in fake.calls("inject")] == [2, 2]

    async def test_inject_batch_size_below_one_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="batch_size"):
            await session.inject(("S", "r", "o"), batch_size=0)
        assert fake.calls("inject") == []

    async def test_inject_empty_list_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(InvalidFactError, match="must not be empty"):
            await session.inject([])
        assert fake.calls("inject") == []

    async def test_inject_malformed_fact_raises_before_sending(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(InvalidFactError, match="missing required field"):
            await session.inject({"subject": "Tom", "relationship": "lives in"})
        with pytest.raises(InvalidFactError, match="certainty"):
            await session.inject(("Tom", "lives in", "Germany", 101))
        assert fake.calls("inject") == []


# --------------------------------------------------------------------------- query


class TestQuery:
    async def test_query_result(self, session: AsyncSession, fake: FakeRainbird) -> None:
        response = await session.query("speaks", subject="Tom")
        assert isinstance(response, ResultResponse)
        assert response.is_result and not response.is_question
        assert response.sid == session.id
        assert len(response.results) == 1
        item = response.results[0]
        assert (item.subject, item.relationship, item.object) == ("Tom", "speaks", "German")
        assert item.certainty == 75
        assert item.fact_id == ROOT_FACT
        assert response.best is item
        assert response.fact_ids == [ROOT_FACT]

    async def test_query_request_subject_only(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.query("speaks", subject="Tom")
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/query"
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom"}

    async def test_query_request_object_only(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.query("speaks", object="French")
        assert fake.last.json == {"relationship": "speaks", "object": "French"}

    async def test_query_request_both_sides(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.query("speaks", subject="Tom", object="French")
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom", "object": "French"}

    async def test_query_relationship_only(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.query("speaks")
        assert fake.last.json == {"relationship": "speaks"}

    async def test_query_returns_question(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        response = await session.query("speaks", subject="Tom")
        assert isinstance(response, QuestionResponse)
        assert response.is_question and not response.is_result
        assert response.results == []
        assert response.question.prompt == "What is the national language of Germany?"
        assert response.question.form is QuestionForm.SECOND_OBJECT
        assert response.question.options == ["English", "French"]
        assert response.sid == session.id

    async def test_query_with_extra_questions(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT, QUESTION_FIRST_FORM)
        response = await session.query("speaks", subject="Tom")
        assert isinstance(response, QuestionResponse)
        assert len(response.extra_questions) == 1
        assert [q.prompt for q in response.questions] == [
            "What is the national language of Germany?",
            "Does Tom speak French?",
        ]

    async def test_query_empty_relationship_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="relationship"):
            await session.query("")
        assert fake.calls("query") == []

    async def test_query_unexpected_body_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue({"weird": True})
        with pytest.raises(UnexpectedResponseError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.body == {"weird": True, "sid": session.id}
        assert info.value.status_code == 200
        assert session.last_response is None


# --------------------------------------------------------------------------- answer / respond


class TestAnswer:
    async def test_second_form_object(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        response = await session.answer("German")  # canAdd: values outside options are fine
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "German",
                    "certainty": 100,
                }
            ]
        }
        assert isinstance(response, ResultResponse)

    async def test_second_form_object_with_certainty(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        await session.answer("English", certainty=80)
        assert fake.last.json["answers"] == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "English",
                "certainty": 80,
            }
        ]

    async def test_second_form_subject(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        response = await session.query("speaks", object="French")
        assert response.question.form is QuestionForm.SECOND_SUBJECT
        await session.answer("Tom")
        assert fake.last.json == {
            "answers": [
                {"subject": "Tom", "relationship": "speaks", "object": "French", "certainty": 100}
            ]
        }

    async def test_second_form_subject_plural_sends_one_answer_per_value(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        await session.query("speaks", object="French")
        await session.answer(["Tom", "Ana"])
        assert fake.last.json["answers"] == [
            {"subject": "Tom", "relationship": "speaks", "object": "French", "certainty": 100},
            {"subject": "Ana", "relationship": "speaks", "object": "French", "certainty": 100},
        ]

    async def test_second_form_subject_rejects_unknown_option_when_can_add_false(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        await session.query("speaks", object="French")
        with pytest.raises(InvalidAnswerError, match="not one of the allowed answers"):
            await session.answer("Zed")
        assert fake.calls("response") == []
        assert session.pending is not None  # still waiting

    async def test_non_strict_defers_validation_to_api(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        await session.query("speaks", object="French")
        await session.answer("Zed", strict=False)
        assert fake.last.json["answers"][0]["subject"] == "Zed"

    async def test_first_form_yes(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        response = await session.query("speaks", subject="Tom")
        assert response.question.is_first_form
        await session.answer(True)
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Tom",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 100,
                    "answer": "yes",
                }
            ]
        }

    @pytest.mark.parametrize("value", [False, "no", "No", "n", 0])
    async def test_first_form_no_variants(
        self, session: AsyncSession, fake: FakeRainbird, value: Any
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        await session.query("speaks", subject="Tom")
        await session.answer(value)
        assert fake.last.json["answers"][0]["answer"] == "no"

    async def test_first_form_rejects_non_boolean(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        await session.query("speaks", subject="Tom")
        with pytest.raises(InvalidAnswerError, match="yes/no"):
            await session.answer("maybe")
        assert fake.calls("response") == []

    async def test_plural_object_question(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        await session.query("can claim", subject="Ben")
        await session.answer(["theft", "water damage"])
        assert fake.last.json["answers"] == [
            {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 100},
            {
                "subject": "Ben",
                "relationship": "suffered",
                "object": "water damage",
                "certainty": 100,
            },
        ]

    async def test_multiple_values_for_singular_question_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        with pytest.raises(InvalidAnswerError, match="single answer"):
            await session.answer(["English", "French"])
        assert fake.calls("response") == []

    async def test_answer_without_pending_question_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(NoPendingQuestionError):
            await session.answer("German")
        assert fake.calls("response") == []

    async def test_answer_after_result_raises(self, session: AsyncSession) -> None:
        await session.query("speaks", subject="Tom")  # result
        assert session.result is not None
        with pytest.raises(NoPendingQuestionError):
            await session.answer("German")

    async def test_answer_explicit_question_without_pending(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        question = Question.from_dict(QUESTION_FIRST_FORM)
        response = await session.answer("yes", question=question)
        assert fake.last.json["answers"][0]["answer"] == "yes"
        assert session.result is response

    async def test_answer_with_skip_sentinel(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        await session.query("speaks", subject="Tom")
        await session.answer(SKIP)
        assert fake.last.json["answers"] == [
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    async def test_chained_questions(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])
        first = await session.query("speaks", subject="Tom")
        assert session.pending is first
        second = await session.answer("German")
        assert isinstance(second, QuestionResponse)
        assert session.pending is second
        assert session.pending_question.prompt == "Does Tom speak French?"
        final = await session.answer(True)
        assert isinstance(final, ResultResponse)
        assert session.pending is None
        assert session.result is final
        assert len(fake.calls("response")) == 2

    async def test_respond_with_answer_objects(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        response = await session.respond(
            [
                Answer("Ben", "suffered", "theft", certainty=90),
                Answer("Ben", "suffered", "water damage"),
            ]
        )
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 90},
                {
                    "subject": "Ben",
                    "relationship": "suffered",
                    "object": "water damage",
                    "certainty": 100,
                },
            ]
        }
        assert session.last_response is response

    async def test_respond_with_dicts(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.respond(
            {"subject": "Tom", "relationship": "speaks", "object": "French", "answer": "yes"}
        )
        assert fake.last.json["answers"] == [
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "answer": "yes",
            }
        ]

    async def test_respond_rejects_malformed_answer(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(InvalidFactError):
            await session.respond({"subject": "Tom", "relationship": "speaks"})
        with pytest.raises(InvalidFactError):
            await session.respond([])
        assert fake.calls("response") == []


# --------------------------------------------------------------------------- skip


class TestSkip:
    async def test_skip_second_form_subject(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)  # allowUnknown: True
        await session.query("speaks", object="French")
        response = await session.skip()
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
        assert isinstance(response, ResultResponse)
        assert session.result is response
        assert session.pending is None

    async def test_skip_first_form_keeps_triple(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)  # allowUnknown: True
        await session.query("speaks", subject="Tom")
        await session.skip()
        assert fake.last.json["answers"] == [
            {
                "subject": "Tom",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    async def test_skip_not_allowed_raises_locally(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown: False
        await session.query("speaks", subject="Tom")
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            await session.skip()
        assert fake.calls("response") == []
        assert session.pending is not None

    async def test_skip_not_allowed_non_strict_sends_unanswered_object(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        await session.skip(strict=False)
        assert fake.last.json["answers"] == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    async def test_skip_explicit_question(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.skip(question=Question.from_dict(QUESTION_SECOND_SUBJECT))
        assert fake.last.json["answers"][0]["unanswered"] is True

    async def test_skip_without_pending_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(NoPendingQuestionError):
            await session.skip()
        assert fake.calls("response") == []

    async def test_skip_rejected_by_api_is_bad_request(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        fake.fail_next(400, {"err": ["Please provide an expected boolean value for unanswered."]})
        with pytest.raises(BadRequestError) as info:
            await session.skip(strict=False)
        assert info.value.status_code == 400
        assert info.value.errors == ["Please provide an expected boolean value for unanswered."]
        assert info.value.message == "Please provide an expected boolean value for unanswered."
        assert session.pending is not None  # the failed request did not change state


# --------------------------------------------------------------------------- undo


class TestUndo:
    async def test_undo_request_and_tracking(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        await session.answer("German")
        assert session.result is not None
        undone = await session.undo()
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/undo"
        assert fake.last.json == {}
        assert isinstance(undone, QuestionResponse)
        assert session.pending is undone
        assert session.result is None
        # the restored question can be answered again
        await session.answer("English")
        assert fake.last.json["answers"][0]["object"] == "English"

    async def test_undo_can_return_result(self, session: AsyncSession, fake: FakeRainbird) -> None:
        response = await session.undo()
        assert isinstance(response, ResultResponse)
        assert session.result is response


# --------------------------------------------------------------------------- pending tracking


class TestPendingTracking:
    async def test_question_sets_pending(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        response = await session.query("speaks", subject="Tom")
        assert session.last_response is response
        assert session.pending is response
        assert session.pending_question is response.question
        assert session.result is None

    async def test_result_clears_pending(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await session.query("speaks", subject="Tom")
        result = await session.answer("German")
        assert session.pending is None
        assert session.pending_question is None
        assert session.result is result
        assert session.last_response is result

    async def test_new_query_replaces_pending(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        first = await session.query("speaks", subject="Tom")
        second = await session.query("speaks", subject="Ana")
        assert session.pending is second
        assert session.pending is not first

    async def test_failed_request_leaves_state_untouched(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        pending = await session.query("speaks", subject="Tom")
        fake.fail_next(500, {"error": "boom"})
        with pytest.raises(ServerError):
            await session.answer("German")
        assert session.pending is pending

    async def test_sessions_track_state_independently(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        first = await rb.start(KM_ID)
        second = await rb.start(KM_ID)
        fake.queue_question(QUESTION_SECOND_OBJECT)
        await first.query("speaks", subject="Tom")
        assert first.pending is not None
        assert second.pending is None
        assert first.id != second.id


# --------------------------------------------------------------------------- run()


def _default_answerer(question: Question) -> Any:
    if question.is_first_form:
        return True
    if question.plural:
        return question.options[:2]
    return "German"


class TestRun:
    def _script_two_questions(self, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])

    async def test_run_with_sync_answerer(self, session: AsyncSession, fake: FakeRainbird) -> None:
        self._script_two_questions(fake)
        seen: list[Question] = []

        def answerer(question: Question) -> Any:
            seen.append(question)
            return _default_answerer(question)

        result = await session.run("speaks", subject="Tom", answerer=answerer)
        assert isinstance(result, ResultResponse)
        assert result.best is not None and result.best.object == "German"
        assert [q.prompt for q in seen] == [
            "What is the national language of Germany?",
            "Does Tom speak French?",
        ]
        assert [c.json for c in fake.calls("query")] == [
            {"relationship": "speaks", "subject": "Tom"}
        ]
        assert [c.json for c in fake.calls("response")] == [
            {
                "answers": [
                    {
                        "subject": "Germany",
                        "relationship": "national language",
                        "object": "German",
                        "certainty": 100,
                    }
                ]
            },
            {
                "answers": [
                    {
                        "subject": "Tom",
                        "relationship": "speaks",
                        "object": "French",
                        "certainty": 100,
                        "answer": "yes",
                    }
                ]
            },
        ]
        assert session.result is result
        assert session.pending is None

    async def test_run_with_async_answerer(self, session: AsyncSession, fake: FakeRainbird) -> None:
        self._script_two_questions(fake)
        seen: list[str] = []

        async def answerer(question: Question) -> Any:
            await asyncio.sleep(0)
            seen.append(question.prompt)
            return _default_answerer(question)

        result = await session.run("speaks", subject="Tom", answerer=answerer)
        assert isinstance(result, ResultResponse)
        assert seen == ["What is the national language of Germany?", "Does Tom speak French?"]
        payloads = [c.json["answers"][0] for c in fake.calls("response")]
        assert payloads[0]["object"] == "German"
        assert payloads[1]["answer"] == "yes"

    async def test_run_with_skip_from_sync_answerer(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])
        result = await session.run("speaks", subject="Tom", answerer=lambda q: SKIP)
        assert isinstance(result, ResultResponse)
        assert [c.json for c in fake.calls("response")] == [
            {
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
        ]

    async def test_run_with_skip_from_async_answerer(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])

        async def answerer(question: Question) -> Any:
            return SKIP

        await session.run("speaks", object="French", answerer=answerer)
        assert fake.calls("response")[0].json["answers"] == [
            {
                "subject": "",
                "relationship": "speaks",
                "object": "French",
                "certainty": 100,
                "unanswered": True,
            }
        ]

    async def test_run_skip_not_allowed_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown: False
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            await session.run("speaks", subject="Tom", answerer=lambda q: SKIP)
        assert fake.calls("response") == []

    async def test_run_with_answer_object(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])

        def answerer(question: Question) -> Answer:
            return Answer(question.subject or "", question.relationship, "Bavarian", certainty=42)

        await session.run("speaks", subject="Tom", answerer=answerer)
        assert fake.last.json["answers"] == [
            {
                "subject": "Germany",
                "relationship": "national language",
                "object": "Bavarian",
                "certainty": 42,
            }
        ]

    async def test_run_with_list_of_answer_objects(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])

        async def answerer(question: Question) -> list[Answer]:
            return [Answer("Ben", "suffered", "theft"), Answer("Ben", "suffered", "fire", 30)]

        await session.run("can claim", subject="Ben", answerer=answerer)
        assert [a["object"] for a in fake.last.json["answers"]] == ["theft", "fire"]
        assert fake.last.json["answers"][1]["certainty"] == 30

    async def test_run_with_plural_list_of_values(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(*RESULT_TOM_SPEAKS_GERMAN["result"])
        await session.run("can claim", subject="Ben", answerer=_default_answerer)
        assert [a["object"] for a in fake.last.json["answers"]] == [
            "cosmetic damage",
            "water damage",
        ]

    async def test_run_without_questions_never_calls_answerer(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        calls: list[Question] = []

        def answerer(question: Question) -> Any:
            calls.append(question)
            return True

        result = await session.run("speaks", subject="Tom", object="German", answerer=answerer)
        assert isinstance(result, ResultResponse)
        assert calls == []
        assert fake.calls("response") == []
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom", "object": "German"}

    async def test_run_max_questions_exceeded(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        for _ in range(3):
            fake.queue_question(QUESTION_FIRST_FORM)
        with pytest.raises(RainbirdError, match="more than 2 questions") as info:
            await session.run("speaks", subject="Tom", answerer=lambda q: True, max_questions=2)
        assert "Does Tom speak French?" in str(info.value)
        assert len(fake.calls("response")) == 2
        assert session.pending is not None  # the third question is still pending

    async def test_run_answerer_exception_propagates(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)

        async def answerer(question: Question) -> Any:
            raise LookupError("no answer available")

        with pytest.raises(LookupError, match="no answer available"):
            await session.run("speaks", subject="Tom", answerer=answerer)
        assert fake.calls("response") == []
        assert session.pending_question is not None

    async def test_run_invalid_answer_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        with pytest.raises(InvalidAnswerError):
            await session.run("speaks", subject="Tom", answerer=lambda q: "perhaps")
        assert fake.calls("response") == []


# --------------------------------------------------------------------------- evidence


class TestEvidence:
    async def test_evidence_node(self, session: AsyncSession, fake: FakeRainbird) -> None:
        node = await session.evidence(ROOT_FACT)
        assert isinstance(node, Evidence)
        assert node.fact_id == ROOT_FACT
        assert node.source is FactSource.RULE
        assert node.is_inferred
        assert node.fact.subject.value == "Tom"
        assert node.fact.subject.concept == "person"
        assert node.fact.relationship == "speaks"
        assert node.fact.object.value == "German"
        assert node.fact.certainty == 75
        assert node.time == 1751644730474
        assert node.timestamp == dt.datetime(2025, 7, 4, 15, 58, 50, 474000, tzinfo=UTC)
        assert node.rule is not None
        assert node.rule.bindings == {"COUNTRY": "Germany", "O": "German", "S": "Tom"}
        assert node.rule.max_certainty == 75
        assert len(node.rule.relationship_conditions) == 2
        assert len(node.rule.expression_conditions) == 1
        assert node.rule.expression_conditions[0].was_met is True
        assert node.child_fact_ids == [IF_FACT, AF_FACT]

    async def test_evidence_request(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.evidence(ROOT_FACT)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/evidence/{ROOT_FACT}/{session.id}"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert "x-evidence-key" not in fake.last.headers

    async def test_evidence_key_header(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "ev-secret"
        node = await session.evidence(ROOT_FACT, evidence_key="ev-secret")
        assert node.fact_id == ROOT_FACT
        assert fake.last.headers["x-evidence-key"] == "ev-secret"

    async def test_evidence_key_from_client_default(self, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "default-ev"
        async with make_client(fake, evidence_key="default-ev") as client:
            session = await client.start(KM_ID)
            await session.evidence(ROOT_FACT)
            assert fake.last.headers["x-evidence-key"] == "default-ev"
            # an explicit key overrides the default
            with pytest.raises(PermissionDeniedError):
                await session.evidence(ROOT_FACT, evidence_key="other")
            assert fake.last.headers["x-evidence-key"] == "other"

    async def test_evidence_missing_key_is_permission_denied(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.require_evidence_key = "ev-secret"
        with pytest.raises(PermissionDeniedError) as info:
            await session.evidence(ROOT_FACT)
        assert info.value.status_code == 403
        assert info.value.message == "Evidence is secured; provide x-evidence-key"

    async def test_evidence_unknown_fact_is_not_found(self, session: AsyncSession) -> None:
        with pytest.raises(NotFoundError) as info:
            await session.evidence("WA:RF:doesnotexist")
        assert info.value.status_code == 404

    async def test_client_level_evidence(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        node = await rb.evidence("some-session", ROOT_FACT)
        assert node.fact_id == ROOT_FACT
        assert fake.last.path == f"/analysis/evidence/{ROOT_FACT}/some-session"

    async def test_leaf_evidence_has_no_children(self, session: AsyncSession) -> None:
        leaf = await session.evidence(IF_FACT)
        assert leaf.source is FactSource.INJECTION
        assert leaf.rule is None
        assert not leaf.is_inferred
        assert leaf.child_fact_ids == []

    async def test_evidence_tree_fetches_each_node_once(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        nodes = await session.evidence_tree(ROOT_FACT)
        assert [n.fact_id for n in nodes] == [ROOT_FACT, IF_FACT, AF_FACT]
        assert nodes[0].is_inferred
        assert [n.source for n in nodes[1:]] == [FactSource.INJECTION, FactSource.ANSWER]
        # IF_FACT is referenced twice by the root (condition + expression) but fetched once
        assert evidence_calls(fake) == [
            f"/analysis/evidence/{fid}/{session.id}" for fid in (ROOT_FACT, IF_FACT, AF_FACT)
        ]

    async def test_evidence_tree_max_depth_zero_is_root_only(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        nodes = await session.evidence_tree(ROOT_FACT, max_depth=0)
        assert [n.fact_id for n in nodes] == [ROOT_FACT]
        assert len(evidence_calls(fake)) == 1

    async def test_evidence_tree_max_depth_one(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        nodes = await session.evidence_tree(ROOT_FACT, max_depth=1)
        assert [n.fact_id for n in nodes] == [ROOT_FACT, IF_FACT, AF_FACT]
        assert len(evidence_calls(fake)) == 3

    async def test_evidence_tree_from_leaf(self, session: AsyncSession, fake: FakeRainbird) -> None:
        nodes = await session.evidence_tree(AF_FACT)
        assert [n.fact_id for n in nodes] == [AF_FACT]
        assert len(evidence_calls(fake)) == 1

    async def test_evidence_tree_passes_key_to_every_request(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.require_evidence_key = "ev-secret"
        nodes = await session.evidence_tree(ROOT_FACT, evidence_key="ev-secret")
        assert len(nodes) == 3
        assert all(
            c.headers["x-evidence-key"] == "ev-secret" for c in fake.calls("/analysis/evidence/")
        )

    async def test_evidence_tree_propagates_errors(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        del fake.evidence[AF_FACT]
        with pytest.raises(NotFoundError):
            await session.evidence_tree(ROOT_FACT)
        assert len(evidence_calls(fake)) == 3

    async def test_iter_evidence_tree_is_async_generator(self, session: AsyncSession) -> None:
        assert inspect.isasyncgenfunction(AsyncSession.iter_evidence_tree)
        agen = session.iter_evidence_tree(ROOT_FACT)
        assert inspect.isasyncgen(agen)
        await agen.aclose()

    async def test_iter_evidence_tree_is_lazy(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        agen = session.iter_evidence_tree(ROOT_FACT)
        assert evidence_calls(fake) == []  # nothing fetched until iteration starts
        first = await agen.__anext__()
        assert first.fact_id == ROOT_FACT
        assert len(evidence_calls(fake)) == 1
        rest = [node async for node in agen]
        assert [n.fact_id for n in rest] == [IF_FACT, AF_FACT]
        assert len(evidence_calls(fake)) == 3

    async def test_iter_evidence_tree_break_stops_fetching(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        agen = session.iter_evidence_tree(ROOT_FACT)
        async for node in agen:
            assert node.fact_id == ROOT_FACT
            break
        await agen.aclose()
        assert len(evidence_calls(fake)) == 1

    async def test_iter_evidence_tree_matches_evidence_tree(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        eager = await session.evidence_tree(ROOT_FACT)
        lazy = [n async for n in session.iter_evidence_tree(ROOT_FACT)]
        assert [n.fact_id for n in eager] == [n.fact_id for n in lazy]

    async def test_evidence_url(self, session: AsyncSession, rb: AsyncRainbird) -> None:
        expected = f"{STUDIO_URL}/evidence?id={ROOT_FACT}&api={BASE_URL}&sid={session.id}"
        assert session.evidence_url(ROOT_FACT) == expected
        assert rb.evidence_url(session.id, ROOT_FACT) == expected
        assert not inspect.iscoroutinefunction(AsyncSession.evidence_url)
        assert not inspect.iscoroutinefunction(AsyncRainbird.evidence_url)

    async def test_evidence_url_custom_studio(self, session: AsyncSession) -> None:
        url = session.evidence_url(ROOT_FACT, studio_url="https://acme.rainbird.ai/")
        assert (
            url
            == f"https://acme.rainbird.ai/evidence?id={ROOT_FACT}&api={BASE_URL}&sid={session.id}"
        )


# --------------------------------------------------------------------------- interactions


class TestInteractions:
    async def test_interactions_json(self, session: AsyncSession, fake: FakeRainbird) -> None:
        events = await session.interactions()
        assert isinstance(events, list)
        assert len(events) == 3
        assert all(isinstance(e, InteractionEvent) for e in events)
        assert [e.event for e in events] == ["start", "query", "result"]
        assert events[1].values == {"query": {"subject": "Tom", "relationship": "speaks"}}
        assert events[0].created == dt.datetime(2025, 7, 10, 10, 6, 27, 780000, tzinfo=UTC)
        assert events[2].raw["values"]["results"][0]["object"] == "German"

    async def test_interactions_json_request(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.interactions(format="json")
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/interactions/{session.id}"
        assert fake.last.params == {"format": "json"}
        assert fake.last.headers["accept"] == "application/json"
        assert "x-interaction-key" not in fake.last.headers

    async def test_interactions_csv(self, session: AsyncSession, fake: FakeRainbird) -> None:
        text = await session.interactions(format="csv")
        assert isinstance(text, str)
        assert text.splitlines()[0] == "event,created"
        assert fake.last.path == f"/analysis/interactions/{session.id}"
        assert fake.last.params == {"format": "csv"}
        assert fake.last.headers["accept"] == "text/csv, text/plain, */*"

    async def test_interactions_key_header(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "ik-secret"
        events = await session.interactions(interaction_key="ik-secret")
        assert len(events) == 3
        assert fake.last.headers["x-interaction-key"] == "ik-secret"
        text = await session.interactions(format="csv", interaction_key="ik-secret")
        assert text.startswith("event,created")
        assert fake.last.headers["x-interaction-key"] == "ik-secret"

    async def test_interactions_key_from_client_default(self, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "default-ik"
        async with make_client(fake, interaction_key="default-ik") as client:
            session = await client.start(KM_ID)
            events = await session.interactions()
            assert len(events) == 3
            assert fake.last.headers["x-interaction-key"] == "default-ik"

    async def test_interactions_missing_key_is_permission_denied(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.require_interaction_key = "ik-secret"
        with pytest.raises(PermissionDeniedError) as info:
            await session.interactions()
        assert info.value.status_code == 403
        assert info.value.message == "Interaction log is secured"


# --------------------------------------------------------------------------- session analysis


class TestSessionAnalysis:
    async def test_knowledge_map(self, session: AsyncSession, fake: FakeRainbird) -> None:
        km = await session.knowledge_map()
        assert isinstance(km, KnowledgeMapVersion)
        assert km.id == KM_ID
        assert km.name == "Languages"
        assert km.version_id == KM_VERSION_ID
        assert km.version_status == "Version3"
        assert km.version_number == 3
        assert not km.is_draft
        assert km.version_created == dt.datetime(2025, 7, 9, 14, 27, 22, tzinfo=UTC)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/session/{session.id}"
        assert fake.last.params == {"filter": "version"}

    async def test_facts_all(self, session: AsyncSession, fake: FakeRainbird) -> None:
        facts = await session.facts()
        assert isinstance(facts, SessionFacts)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/session/{session.id}"
        assert fake.last.params == {"filter": "facts"}
        assert len(facts.global_) == 1
        assert len(facts.local) == 1
        assert facts.context == []
        assert len(facts) == 2
        assert [str(f.subject) for f in facts] == ["France", "Tom"]
        local = facts.local[0]
        assert local.id == "WA:IF:local1"
        assert (local.subject.value, local.relationship, local.object.value) == (
            "Tom",
            "lives in",
            "Germany",
        )
        assert local.subject.concept == "person"
        assert local.source is FactSource.INJECTION
        assert facts.global_[0].source is FactSource.KNOWLEDGE_MAP

    async def test_facts_filtered_by_relationship(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = await session.facts(relationships=["lives in"])
        assert fake.last.params == {"filter": "facts", "relationships": "lives in"}
        assert facts.global_ == []
        assert [f.relationship for f in facts.local] == ["lives in"]

    async def test_facts_filtered_by_several_relationships(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = await session.facts(relationships=("lives in", "national language"))
        assert fake.last.params == {
            "filter": "facts",
            "relationships": "lives in,national language",
        }
        assert len(facts) == 2

    async def test_facts_accepts_any_iterable(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.facts(relationships=(r for r in ["national language"]))
        assert fake.last.params["relationships"] == "national language"

    async def test_facts_accepts_single_relationship_string(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = await session.facts(relationships="speaks")
        assert fake.last.params == {"filter": "facts", "relationships": "speaks"}
        assert len(facts) == 0  # nothing in the fake uses "speaks"

    async def test_facts_single_string_is_not_split_into_characters(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = await session.facts(relationships="lives in")
        assert fake.last.params["relationships"] == "lives in"
        assert [f.relationship for f in facts] == ["lives in"]

    async def test_facts_empty_filter_means_no_filter(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        facts = await session.facts(relationships=[])
        assert fake.last.params == {"filter": "facts"}
        assert len(facts) == 2


# --------------------------------------------------------------------------- natural language


class TestNaturalLanguage:
    async def test_interact_result(self, session: AsyncSession, fake: FakeRainbird) -> None:
        prompt = "Ben holds a Gold policy and he likes tea. What can Ben claim?"
        response = await session.interact(prompt)
        assert isinstance(response, InteractResponse)
        assert response.response_type == "result"
        assert response.is_result and not response.is_question and not response.is_error
        assert response.error is None
        assert response.query == {"subject": "Ben", "relationship": "can claim", "object": None}
        item = response.results[0]  # capitalised keys in the payload
        assert (item.subject, item.relationship, item.object) == ("Ben", "can claim", "repair")
        assert item.certainty == 100
        assert (
            item.fact_id == "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea"
        )
        assert response.injected_facts[0]["object"] == "Gold"
        assert response.invalid_facts == []
        assert response.unmatched == ["and he likes tea"]
        assert response.metadata is not None and response.metadata.endpoint == "interact"
        assert response.raise_for_error() is response

    async def test_interact_request(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.interact("What can Ben claim?")
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/interact"
        assert fake.last.json == {"sessionID": session.id, "userPrompt": "What can Ben claim?"}
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["x-api-key"] == API_KEY

    async def test_interact_question(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_QUESTION)
        response = await session.interact("Can Ben claim?")
        assert response.is_question
        assert response.question is not None
        assert response.question.prompt == "Which policy does Ben hold?"
        assert response.prompt == "Which policy does Ben hold?"
        assert response.question.form is QuestionForm.SECOND_OBJECT
        assert response.question.options == ["Gold", "Silver"]
        assert response.results == []
        assert response.metadata is not None
        assert response.metadata.total_tokens == 3169
        assert response.metadata.language_engine_response_ms == 2173
        assert response.metadata.query_submitted_at is not None
        assert response.metadata.query_submitted_at.year == 2025

    async def test_interact_question_can_be_answered_via_response_endpoint(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.append(INTERACT_QUESTION)
        response = await session.interact("Can Ben claim?")
        assert session.pending is None  # interact does not track pending questions
        result = await session.answer("Gold", question=response.question)
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {
            "answers": [
                {"subject": "Ben", "relationship": "holds", "object": "Gold", "certainty": 100}
            ]
        }
        assert isinstance(result, ResultResponse)

    async def test_interact_error_is_returned_not_raised(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.append(INTERACT_ERROR)
        response = await session.interact("hello there")
        assert response.is_error
        assert response.error is not None
        assert response.error.code == 1001
        assert response.error.message == "Could not detect a query in the prompt"
        assert response.error.status_code == "400"
        assert response.prompt == response.error.suggested_chat_response
        with pytest.raises(NaturalLanguageError) as info:
            response.raise_for_error()
        assert info.value.code == 1001
        assert info.value.message == "Could not detect a query in the prompt"
        assert info.value.suggested_chat_response == (
            "I couldn't work out what you want to know. Could you rephrase?"
        )
        assert info.value.status_code == "400"
        assert info.value.body == INTERACT_ERROR

    async def test_interact_bodies_are_consumed_in_order(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.extend([INTERACT_QUESTION, INTERACT_ERROR])
        assert (await session.interact("one")).is_question
        assert (await session.interact("two")).is_error
        assert (await session.interact("three")).is_result

    async def test_interact_empty_prompt_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="user_prompt"):
            await session.interact("   ")
        assert fake.calls("interact") == []

    async def test_explain(self, session: AsyncSession, fake: FakeRainbird) -> None:
        response = await session.explain(ROOT_FACT)
        assert isinstance(response, ExplainResponse)
        assert response.explanation is not None
        assert response.explanation.startswith("Tom speaks German because")
        assert str(response) == response.explanation
        assert response.error is None
        assert response.metadata is not None
        assert response.metadata.endpoint == "explain"
        assert response.metadata.completion_tokens == 174
        assert response.metadata.prompt_tokens == 726
        assert response.metadata.total_tokens == 900

    async def test_explain_request(self, session: AsyncSession, fake: FakeRainbird) -> None:
        await session.explain(ROOT_FACT)
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/explain"
        assert fake.last.json == {"language": "en", "sessionID": session.id, "factID": ROOT_FACT}
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["x-api-key"] == API_KEY

    async def test_explain_language_and_instructions(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        await session.explain(ROOT_FACT, language="es", instructions="Keep it to one sentence.")
        assert fake.last.json == {
            "language": "es",
            "sessionID": session.id,
            "factID": ROOT_FACT,
            "instructions": "Keep it to one sentence.",
        }

    async def test_explain_error_raises_natural_language_error(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(NaturalLanguageError) as info:
            await session.explain("not-a-fact")
        assert info.value.message == "Fact not found"
        assert info.value.code == 2001
        assert info.value.status_code == "404"
        assert info.value.suggested_chat_response is None
        assert info.value.body == {
            "error": {"code": 2001, "message": "Fact not found", "statusCode": "404"}
        }
        assert not isinstance(info.value, RainbirdAPIError)
        assert isinstance(info.value, RainbirdError)

    async def test_explain_empty_fact_id_raises(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="fact_id"):
            await session.explain("")
        assert fake.calls("explain") == []


# --------------------------------------------------------------------------- version


class TestVersion:
    async def test_version(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        assert await rb.version() == "4.89.0"
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"
        assert fake.last.json is None
        assert fake.last.headers["accept"] == "text/plain, application/json, */*"

    async def test_version_reflects_server_text(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.version_text = "  5.1.0\n"
        assert await rb.version() == "5.1.0"

    async def test_version_sends_api_key_when_called_on_a_client(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        await rb.version()
        assert fake.last.headers["x-api-key"] == API_KEY


class TestEngineVersion:
    """``rainbird_ai.async_engine_version()``: ``GET /version`` with no API key."""

    @pytest.fixture
    async def http(self, fake: FakeRainbird) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as client:
            yield client

    def test_exported_at_package_and_module_level(self) -> None:
        assert rainbird_ai.async_engine_version is async_engine_version
        assert aio_module.async_engine_version is async_engine_version
        assert "async_engine_version" in rainbird_ai.__all__
        assert inspect.iscoroutinefunction(async_engine_version)

    async def test_returns_engine_version_without_api_key(
        self, http: httpx.AsyncClient, fake: FakeRainbird
    ) -> None:
        assert await async_engine_version(BASE_URL, http_client=http) == "4.89.0"
        assert len(fake.requests) == 1
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"
        assert fake.last.json is None
        assert "x-api-key" not in fake.last.headers
        assert fake.last.headers["accept"] == "text/plain, application/json, */*"
        assert fake.last.headers["user-agent"].startswith("rainbird-ai-python/")
        assert "x-rainbird-engine" not in fake.last.headers

    async def test_does_not_send_an_api_key_from_the_environment(
        self, http: httpx.AsyncClient, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", API_KEY)
        assert await async_engine_version(BASE_URL, http_client=http) == "4.89.0"
        assert "x-api-key" not in fake.last.headers

    async def test_base_url_falls_back_to_env_then_community(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[httpx.Request] = []
        async with httpx.AsyncClient(transport=recording_transport(fake, seen)) as http:
            await async_engine_version(http_client=http)
            assert str(seen[-1].url) == "https://api.rainbird.ai/version"
            monkeypatch.setenv("RAINBIRD_API_URL", BASE_URL + "/")
            await async_engine_version(http_client=http)
            assert str(seen[-1].url) == f"{BASE_URL}/version"
            await async_engine_version("", http_client=http)  # blank behaves like "not given"
            assert str(seen[-1].url) == f"{BASE_URL}/version"
            await async_engine_version(base_url=BASE_URL, http_client=http)
            assert str(seen[-1].url) == f"{BASE_URL}/version"

    async def test_invalid_base_url_raises_before_any_request(
        self, http: httpx.AsyncClient, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="has no host"):
            await async_engine_version("https://:443", http_client=http)
        assert fake.requests == []

    async def test_engine_header_from_argument_and_env(
        self, http: httpx.AsyncClient, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await async_engine_version(BASE_URL, engine="alt-engine", http_client=http)
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"
        monkeypatch.setenv("RAINBIRD_ENGINE", "env-engine")
        await async_engine_version(BASE_URL, http_client=http)
        assert fake.last.headers["x-rainbird-engine"] == "env-engine"

    async def test_parses_the_same_body_variants_as_version(
        self, http: httpx.AsyncClient, fake: FakeRainbird
    ) -> None:
        fake.version_text = "  4.90.1\n"
        assert await async_engine_version(BASE_URL, http_client=http) == "4.90.1"
        fake.version_text = '{"version": "5.0.1"}'
        assert await async_engine_version(BASE_URL, http_client=http) == "5.0.1"
        fake.version_text = '"5.0.2"'
        assert await async_engine_version(BASE_URL, http_client=http) == "5.0.2"

    async def test_user_client_is_left_open(
        self, http: httpx.AsyncClient, fake: FakeRainbird
    ) -> None:
        await async_engine_version(BASE_URL, http_client=http)
        assert http.is_closed is False
        assert await async_engine_version(BASE_URL, http_client=http) == "4.89.0"  # still usable
        assert len(fake.requests) == 2

    async def test_user_client_timeout_is_used_by_default(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        async with httpx.AsyncClient(
            transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0)
        ) as http:
            await async_engine_version(BASE_URL, http_client=http)
        assert seen[-1].extensions["timeout"] == httpx.Timeout(9.0).as_dict()

    async def test_explicit_timeout_is_applied_per_request_with_user_client(
        self, fake: FakeRainbird
    ) -> None:
        seen: list[httpx.Request] = []
        async with httpx.AsyncClient(
            transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0)
        ) as http:
            await async_engine_version(BASE_URL, timeout=2.5, http_client=http)
            assert seen[-1].extensions["timeout"] == httpx.Timeout(2.5).as_dict()
            await async_engine_version(BASE_URL, timeout=None, http_client=http)
            assert seen[-1].extensions["timeout"] == httpx.Timeout(None).as_dict()
            assert http.timeout == httpx.Timeout(9.0)

    async def test_owned_client_is_created_with_defaults_and_closed(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[httpx.AsyncClient] = []
        real_client = httpx.AsyncClient

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            client = real_client(transport=httpx.MockTransport(fake.handle), **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        assert await async_engine_version(BASE_URL) == "4.89.0"
        (client,) = created
        assert client.timeout == httpx.Timeout(30.0)
        assert client.follow_redirects is False
        assert client.is_closed
        assert "x-api-key" not in fake.last.headers

    async def test_owned_client_uses_explicit_timeout_and_is_closed_after_an_error(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[httpx.AsyncClient] = []
        real_client = httpx.AsyncClient

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            client = real_client(transport=httpx.MockTransport(fake.handle), **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        fake.fail_next(404, {"err": ["gone"]})
        with pytest.raises(NotFoundError):
            await async_engine_version(BASE_URL, timeout=7)
        (client,) = created
        assert client.timeout == httpx.Timeout(7)
        assert client.is_closed

    async def test_api_errors_carry_method_and_url(
        self, http: httpx.AsyncClient, fake: FakeRainbird
    ) -> None:
        fake.fail_next(404, {"err": ["gone"]})
        with pytest.raises(NotFoundError) as info:
            await async_engine_version(BASE_URL, http_client=http)
        assert info.value.errors == ["gone"]
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/version"

    async def test_transport_errors_are_retried_then_mapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(anyio, "sleep", fake_sleep)  # rainbird_ai._http backs off via anyio
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("refused")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(RainbirdConnectionError, match="could not reach GET") as info:
                await async_engine_version(BASE_URL, http_client=http)
        assert attempts == 3  # GET /version follows the default policy: two retries
        assert len(sleeps) == 2
        assert info.value.url == f"{BASE_URL}/version"


# --------------------------------------------------------------------------- knowledge maps (undocumented)


class TestKnowledgeMaps:
    async def test_get_map_source(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        source = await rb.get_map_source(KM_ID)
        assert isinstance(source, KnowledgeMapSource)
        assert source.km_id == KM_ID
        assert source.rblang == RBLANG
        assert '<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">' in source.rblang
        assert [c["name"] for c in source.concepts] == ["Person", "Language"]
        assert source.relationships == [
            {"name": "speaks", "subject": "Person", "object": "Language"}
        ]
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/file/{KM_ID}"
        assert fake.last.headers["x-api-key"] == API_KEY

    async def test_get_map_source_unknown_map(self, rb: AsyncRainbird) -> None:
        with pytest.raises(NotFoundError) as info:
            await rb.get_map_source("00000000-0000-0000-0000-000000000000")
        assert info.value.message == "Knowledge map not found"

    async def test_get_map_source_empty_id_raises(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="km_id"):
            await rb.get_map_source("")
        assert fake.requests == []

    async def test_get_map_source_error_body_is_unexpected_response(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.map_source_body = {"error": "map export is disabled"}
        with pytest.raises(UnexpectedResponseError, match="map export is disabled") as info:
            await rb.get_map_source(KM_ID)
        assert info.value.body == {"error": "map export is disabled"}

    async def test_create_map(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        result = await rb.create_map("New map", RBLANG)
        assert isinstance(result, CreateMapResult)
        assert result.km_id == "99999999-8888-7777-6666-555555555555"
        assert result.id == 4242
        assert result.name == "New map"
        assert result.raw == fake.map_create_body

    async def test_create_map_request(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        await rb.create_map("New map", RBLANG)
        assert fake.last.method == "POST"
        assert fake.last.path == "/maps"
        assert fake.last.json == {"rblang": RBLANG, "name": "New map"}
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["x-api-key"] == API_KEY

    async def test_create_map_with_description(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        result = await rb.create_map("New map", RBLANG, description="Who speaks what")
        assert fake.last.json == {
            "rblang": RBLANG,
            "name": "New map",
            "description": "Who speaks what",
        }
        assert result.description == "Who speaks what"

    async def test_create_map_parse_failure_raises_knowledge_map_error(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(KnowledgeMapError) as info:
            await rb.create_map("Broken", "<kb>not rblang</kb>")
        assert info.value.message == "RBLang parse error: missing rbl:kb root"
        assert info.value.errors == ["RBLang parse error: missing rbl:kb root"]
        assert info.value.body == {"error": "RBLang parse error: missing rbl:kb root"}
        assert not isinstance(info.value, RainbirdAPIError)  # the HTTP status was 200
        assert isinstance(info.value, RainbirdError)
        assert len(fake.calls("maps")) == 1

    async def test_create_map_falls_back_to_numeric_id(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.map_create_body = {"id": 77}
        result = await rb.create_map("Numeric", RBLANG)
        assert result.km_id == "77"
        assert result.id == 77
        assert result.name == "Numeric"

    async def test_create_map_without_any_id_is_unexpected_response(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.map_create_body = {"status": "queued"}
        with pytest.raises(UnexpectedResponseError):
            await rb.create_map("Nameless", RBLANG)

    async def test_create_map_validates_arguments(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="name"):
            await rb.create_map("", RBLANG)
        with pytest.raises(RainbirdConfigurationError, match="rblang"):
            await rb.create_map("Name", "  ")
        assert fake.requests == []


# --------------------------------------------------------------------------- error mapping


class TestErrorMapping:
    async def test_401_authentication_error(self, rb: AsyncRainbird, fake: FakeRainbird) -> None:
        fake.fail_next(401, {"err": ["Your API key is invalid"]})
        with pytest.raises(AuthenticationError) as info:
            await rb.start(KM_ID)
        exc = info.value
        assert exc.status_code == 401
        assert exc.message == "Your API key is invalid"
        assert exc.errors == ["Your API key is invalid"]
        assert exc.body == {"err": ["Your API key is invalid"]}
        assert exc.method == "GET"
        assert exc.url == f"{BASE_URL}/start/{KM_ID}"
        assert exc.retry_after is None
        assert str(exc) == f"HTTP 401 calling GET {BASE_URL}/start/{KM_ID}: Your API key is invalid"
        assert len(fake.calls("start")) == 1  # 401 is never retried

    async def test_404_not_found_error(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.fail_next(404, {"err": ["Session not found"]})
        with pytest.raises(NotFoundError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.status_code == 404
        assert info.value.message == "Session not found"
        assert info.value.method == "POST"
        assert info.value.url == f"{BASE_URL}/{session.id}/query"
        assert session.last_response is None

    async def test_429_rate_limit_error(self, fake: FakeRainbird) -> None:
        async with make_client(fake, retry=NO_RETRY) as client:
            fake.fail_next(429, {"message": "Too many requests"}, headers={"Retry-After": "1.5"})
            with pytest.raises(RateLimitError) as info:
                await client.version()
        assert info.value.status_code == 429
        assert info.value.message == "Too many requests"
        assert info.value.retry_after == 1.5
        assert len(fake.calls("version")) == 1

    async def test_429_is_retried_with_retry_after(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.fail_next(429, {"message": "slow down"}, headers={"Retry-After": "0"})
        assert await rb.version() == "4.89.0"
        assert len(fake.calls("version")) == 2

    async def test_429_gives_up_after_max_retries(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        for _ in range(3):
            fake.fail_next(429, {"message": "slow down"}, headers={"Retry-After": "0"})
        with pytest.raises(RateLimitError) as info:
            await rb.version()
        assert info.value.retry_after == 0.0
        assert len(fake.calls("version")) == 3  # 1 attempt + 2 retries

    async def test_500_server_error_on_post_is_not_retried(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.fail_next(500, {"error": "engine exploded"})
        with pytest.raises(ServerError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.status_code == 500
        assert info.value.message == "engine exploded"
        assert info.value.errors == ["engine exploded"]
        assert len(fake.calls("query")) == 1

    async def test_500_on_get_is_retried(self, fake: FakeRainbird) -> None:
        async with make_client(fake, retry=RetryPolicy(max_retries=2, backoff_factor=0)) as client:
            fake.fail_next(500, {"error": "flaky"})
            assert await client.version() == "4.89.0"
        assert len(fake.calls("version")) == 2

    async def test_503_on_post_is_retried(self, fake: FakeRainbird) -> None:
        async with make_client(fake, retry=RetryPolicy(max_retries=2, backoff_factor=0)) as client:
            session = await client.start(KM_ID)
            fake.fail_next(503, {"message": "maintenance"})
            response = await session.query("speaks", subject="Tom")
        assert response.is_result
        assert len(fake.calls("query")) == 2

    async def test_500_with_text_body(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.fail_next(500, "upstream exploded")
        with pytest.raises(ServerError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.message == "upstream exploded"
        assert info.value.errors == ["upstream exploded"]  # plain-text bodies count as one error
        assert info.value.body == "upstream exploded"

    async def test_500_without_body_uses_default_message(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.fail_next(500)
        with pytest.raises(ServerError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.message == "Rainbird server error"
        assert info.value.body is None

    async def test_401_without_body_uses_default_message(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.fail_next(401)
        with pytest.raises(AuthenticationError) as info:
            await rb.start(KM_ID)
        assert info.value.message == "Your API key is invalid"

    async def test_403_permission_denied(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.fail_next(403, {"err": ["Your API key does not permit access to this resource"]})
        with pytest.raises(PermissionDeniedError) as info:
            await session.knowledge_map()
        assert info.value.status_code == 403

    async def test_400_bad_request(self, session: AsyncSession, fake: FakeRainbird) -> None:
        fake.fail_next(400, {"err": ["relationship is required", "subject is invalid"]})
        with pytest.raises(BadRequestError) as info:
            await session.query("speaks", subject="Tom")
        assert info.value.errors == ["relationship is required", "subject is invalid"]
        assert info.value.message == "relationship is required; subject is invalid"

    async def test_unmapped_status_is_generic_api_error(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.fail_next(418, {"error": "teapot"})
        with pytest.raises(RainbirdAPIError) as info:
            await session.query("speaks", subject="Tom")
        assert type(info.value) is RainbirdAPIError
        assert info.value.status_code == 418

    async def test_error_hierarchy(self) -> None:
        for cls in (
            AuthenticationError,
            NotFoundError,
            RateLimitError,
            ServerError,
            BadRequestError,
            PermissionDeniedError,
        ):
            assert issubclass(cls, RainbirdAPIError)
            assert issubclass(cls, RainbirdError)

    async def test_connection_error_on_post(self, fake: FakeRainbird) -> None:
        transport = raising_transport(httpx.ConnectError("connection refused"))
        async with make_client(fake, transport=transport) as client:
            session = client.resume("sid")
            with pytest.raises(RainbirdConnectionError) as info:
                await session.query("speaks", subject="Tom")
        assert info.value.method == "POST"
        assert info.value.url == f"{BASE_URL}/sid/query"
        assert "connection refused" in str(info.value)
        assert isinstance(info.value, RainbirdError)

    async def test_timeout_error_on_get_without_retries(self, fake: FakeRainbird) -> None:
        transport = raising_transport(httpx.ReadTimeout("timed out"))
        async with make_client(fake, retry=NO_RETRY, transport=transport) as client:
            with pytest.raises(RainbirdTimeoutError) as info:
                await client.version()
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/version"

    async def test_timeout_on_get_is_retried(self, fake: FakeRainbird) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("timed out")
            return fake.handle(request)

        policy = RetryPolicy(max_retries=1, backoff_factor=0)
        async with make_client(
            fake, retry=policy, transport=httpx.MockTransport(handler)
        ) as client:
            assert await client.version() == "4.89.0"
        assert attempts == 2

    async def test_non_json_success_body_is_unexpected_response_error(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200, "<html>login</html>")
        with pytest.raises(UnexpectedResponseError) as info:
            await session.query("speaks", subject="Tom")
        assert type(info.value) is UnexpectedResponseError  # not a RainbirdAPIError subclass
        assert str(info.value) == (
            f"expected a JSON body from POST {BASE_URL}/{session.id}/query "
            "but got: '<html>login</html>'"
        )
        assert info.value.status_code == 200
        assert info.value.body == "<html>login</html>"
        assert session.pending_question is None and session.last_response is None

    async def test_empty_success_body_is_unexpected_response_error(
        self, session: AsyncSession, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200)
        with pytest.raises(UnexpectedResponseError, match="expected a JSON body") as info:
            await session.query("speaks", subject="Tom")
        assert info.value.body == ""
        assert info.value.status_code == 200


# --------------------------------------------------------------------------- sync/async parity


def _public_names(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def _shared_functions(sync_cls: type, async_cls: type) -> list[str]:
    shared = _public_names(sync_cls) & _public_names(async_cls)
    return sorted(
        name
        for name in shared | {"__init__"}
        if inspect.isfunction(getattr(sync_cls, name))
        and inspect.isfunction(getattr(async_cls, name))
    )


def _shape(fn: Any) -> list[tuple[str, Any, Any]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


SESSION_FUNCTIONS = _shared_functions(Session, AsyncSession)
CLIENT_FUNCTIONS = _shared_functions(Rainbird, AsyncRainbird)

# methods that are deliberately plain (non-awaitable) on the async classes
SYNC_ON_ASYNC_SESSION = {"evidence_url", "__init__"}
SYNC_ON_ASYNC_CLIENT = {"resume", "evidence_url", "__init__"}


class TestSyncAsyncParity:
    def test_session_public_names_match(self) -> None:
        assert _public_names(Session) == _public_names(AsyncSession)

    def test_client_public_names_match_except_close(self) -> None:
        sync_names = _public_names(Rainbird)
        async_names = _public_names(AsyncRainbird)
        assert "close" in sync_names and "aclose" not in sync_names
        assert "aclose" in async_names and "close" not in async_names
        assert sync_names - {"close"} == async_names - {"aclose"}

    def test_context_manager_protocols(self) -> None:
        assert hasattr(Rainbird, "__enter__") and hasattr(Rainbird, "__exit__")
        assert hasattr(AsyncRainbird, "__aenter__") and hasattr(AsyncRainbird, "__aexit__")
        assert not hasattr(AsyncRainbird, "__enter__")
        assert not hasattr(Rainbird, "__aenter__")
        assert inspect.iscoroutinefunction(AsyncRainbird.__aenter__)
        assert inspect.iscoroutinefunction(AsyncRainbird.__aexit__)
        assert inspect.iscoroutinefunction(AsyncRainbird.aclose)
        assert _shape(Rainbird.close) == _shape(AsyncRainbird.aclose)

    def test_shared_function_lists_are_complete(self) -> None:
        assert set(SESSION_FUNCTIONS) >= {
            "__init__",
            "answer",
            "evidence",
            "evidence_tree",
            "evidence_url",
            "explain",
            "facts",
            "inject",
            "inject_fact",
            "interact",
            "interactions",
            "iter_evidence_tree",
            "knowledge_map",
            "query",
            "respond",
            "run",
            "skip",
            "undo",
        }
        assert set(CLIENT_FUNCTIONS) >= {
            "__init__",
            "create_map",
            "evidence",
            "evidence_url",
            "get_map_source",
            "resume",
            "start",
            "version",
        }

    @pytest.mark.parametrize("name", SESSION_FUNCTIONS)
    def test_session_signatures_identical(self, name: str) -> None:
        assert _shape(getattr(Session, name)) == _shape(getattr(AsyncSession, name))

    @pytest.mark.parametrize("name", CLIENT_FUNCTIONS)
    def test_client_signatures_identical(self, name: str) -> None:
        assert _shape(getattr(Rainbird, name)) == _shape(getattr(AsyncRainbird, name))

    @pytest.mark.parametrize("pair", [(Session, AsyncSession), (Rainbird, AsyncRainbird)])
    def test_properties_match(self, pair: tuple[type, type]) -> None:
        sync_cls, async_cls = pair
        for name in _public_names(sync_cls) & _public_names(async_cls):
            sync_is_prop = isinstance(getattr(sync_cls, name), property)
            async_is_prop = isinstance(getattr(async_cls, name), property)
            assert sync_is_prop == async_is_prop, name

    @pytest.mark.parametrize("name", SESSION_FUNCTIONS)
    def test_async_session_methods_are_awaitable(self, name: str) -> None:
        fn = getattr(AsyncSession, name)
        if name == "iter_evidence_tree":
            assert inspect.isasyncgenfunction(fn)
        elif name in SYNC_ON_ASYNC_SESSION:
            assert not inspect.iscoroutinefunction(fn)
            assert not inspect.isasyncgenfunction(fn)
        else:
            assert inspect.iscoroutinefunction(fn), name

    @pytest.mark.parametrize("name", CLIENT_FUNCTIONS)
    def test_async_client_methods_are_awaitable(self, name: str) -> None:
        fn = getattr(AsyncRainbird, name)
        if name in SYNC_ON_ASYNC_CLIENT:
            assert not inspect.iscoroutinefunction(fn)
        else:
            assert inspect.iscoroutinefunction(fn), name

    @pytest.mark.parametrize("cls", [Session, Rainbird])
    def test_sync_methods_are_plain_functions(self, cls: type) -> None:
        for name in _public_names(cls):
            attr = getattr(cls, name)
            if inspect.isfunction(attr):
                assert not inspect.iscoroutinefunction(attr), name
                assert not inspect.isasyncgenfunction(attr), name

    def test_inject_batch_default_is_shared(self) -> None:
        sync_default = inspect.signature(Session.inject).parameters["batch_size"].default
        async_default = inspect.signature(AsyncSession.inject).parameters["batch_size"].default
        assert sync_default == async_default == MAX_INJECT_BATCH == 250

    def test_module_exports(self) -> None:
        assert aio_module.__all__ == [
            "AsyncAnswerer",
            "AsyncRainbird",
            "AsyncSession",
            "async_engine_version",
        ]
        assert rainbird_ai.AsyncRainbird is AsyncRainbird
        assert rainbird_ai.AsyncSession is AsyncSession
        assert rainbird_ai.AsyncAnswerer is aio_module.AsyncAnswerer
        assert rainbird_ai.async_engine_version is async_engine_version
        for name in aio_module.__all__:
            assert name in rainbird_ai.__all__, name
        assert not hasattr(rainbird_ai, "NoActiveSessionError")  # removed: it was never raised

    def test_engine_version_helpers_share_a_signature(self) -> None:
        assert inspect.iscoroutinefunction(async_engine_version)
        assert not inspect.iscoroutinefunction(engine_version)
        assert _shape(engine_version) == _shape(async_engine_version)
        params = inspect.signature(async_engine_version).parameters
        assert params["timeout"].default is rainbird_ai.UNSET
        assert params["base_url"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(
            params[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in ("engine", "timeout", "http_client")
        )

    async def test_reprs_mirror_each_other(self, client: Rainbird, rb: AsyncRainbird) -> None:
        assert repr(client) == f"Rainbird(base_url={BASE_URL!r})"
        assert repr(rb) == f"AsyncRainbird(base_url={BASE_URL!r})"
        assert repr(client.resume("s1", km_id=KM_ID)) == f"Session(id='s1', km_id={KM_ID!r})"
        assert repr(rb.resume("s1", km_id=KM_ID)) == f"AsyncSession(id='s1', km_id={KM_ID!r})"


# ------------------------------------------------------------ additions from the SDK parity round


@pytest.mark.anyio
class TestSiblingParityAdditionsAsync:
    """Async twins of the features added after porting the Go and JS SDK test suites."""

    async def test_inject_fact_source_is_sent_as_metadata(
        self, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        session = await rb.start(KM_ID)
        await session.inject_fact("Tom", "lives in", "Germany", source="system")
        assert fake.last.json[0]["metadata"] == {"source": "system"}
        await session.inject(Fact("A", "r", "x", metadata={"source": "user"}))
        assert fake.last.json[0]["metadata"] == {"source": "user"}
        with pytest.raises(InvalidFactError, match="metadata must be a dict"):
            await session.inject(
                {"subject": "A", "relationship": "r", "object": "x", "metadata": 1}
            )

    @pytest.mark.parametrize("method", ["query", "respond", "undo", "inject"])
    async def test_per_call_headers_apply_to_that_request_only(
        self, method: str, rb: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        session = await rb.start(KM_ID)
        headers = {"x-api-source": "natlang"}
        if method == "query":
            await session.query("speaks", subject="Tom", headers=headers)
        elif method == "respond":
            await session.respond(
                {"subject": "Tom", "relationship": "speaks", "object": "German"}, headers=headers
            )
        elif method == "undo":
            await session.undo(headers=headers)
        else:
            await session.inject(("Tom", "lives in", "Germany"), headers=headers)
        assert fake.last.headers["x-api-source"] == "natlang"
        assert fake.last.headers["x-api-key"] == API_KEY
        await session.query("speaks", subject="Tom")
        assert "x-api-source" not in fake.last.headers

    async def test_async_engine_version_retry_no_retry_makes_one_attempt(
        self, fake: FakeRainbird
    ) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as http:
            fake.fail_next(500, body="Foobarbaz")
            with pytest.raises(ServerError) as ei:
                await async_engine_version(BASE_URL, http_client=http, retry=NO_RETRY)
            assert ei.value.status_code == 500
            assert len(fake.requests) == 1

    async def test_async_engine_version_default_policy_still_retries(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(anyio, "sleep", no_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as http:
            fake.fail_next(500)
            assert await async_engine_version(BASE_URL, http_client=http) == "4.89.0"
            assert len(fake.requests) == 2
