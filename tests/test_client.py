"""End-to-end tests for the synchronous client (``Rainbird`` / ``Session``) against the fake API.

Every test drives the real SDK code path (ops builders, transport, parsers) through the
``FakeRainbird`` served by ``httpx.MockTransport`` in ``conftest.py`` and asserts the exact
HTTP requests the SDK produced.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest

import rainbird_ai
import rainbird_ai._http as http_mod
from rainbird_ai import (
    NO_RETRY,
    SKIP,
    UNSET,
    Answer,
    AuthenticationError,
    BadRequestError,
    Environment,
    Evidence,
    ExplainResponse,
    Fact,
    FactSource,
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
    Rainbird,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RateLimitError,
    ResultResponse,
    ServerError,
    Session,
    SessionFacts,
    UnexpectedResponseError,
    __version__,
    engine_version,
)
from rainbird_ai import client as client_module
from tests.conftest import (
    API_KEY,
    BASE_URL,
    EVIDENCE_LEAVES,
    EVIDENCE_ROOT,
    EXPLAIN_OK,
    INTERACT_ERROR,
    INTERACT_QUESTION,
    INTERACT_RESULT,
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

ROOT_FACT_ID: str = EVIDENCE_ROOT["factID"]
LEAF_IF_ID, LEAF_AF_ID = list(EVIDENCE_LEAVES)
UTC = dt.timezone.utc

# --------------------------------------------------------------------------- local helpers


def q(data: dict[str, Any]) -> Question:
    return Question.from_dict(data)


def triple(subject: str, relationship: str, obj: Any, certainty: int = 100, **extra: Any) -> dict:
    return {
        "subject": subject,
        "relationship": relationship,
        "object": obj,
        "certainty": certainty,
        **extra,
    }


def recording_transport(fake: FakeRainbird, seen: list[httpx.Request]) -> httpx.MockTransport:
    """``fake`` behind a transport that also keeps every raw ``httpx.Request`` in ``seen``.

    ``FakeRainbird.requests`` records method/path/headers only; the raw request is needed to
    inspect the full URL and the per-request timeout (``request.extensions["timeout"]``).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return fake.handle(request)

    return httpx.MockTransport(handler)


@pytest.fixture
def make_client(fake: FakeRainbird):
    """Factory for extra ``Rainbird`` clients wired to the same fake (closed on teardown)."""
    owned: list[httpx.Client] = []

    def _make(**kwargs: Any) -> Rainbird:
        http = httpx.Client(transport=httpx.MockTransport(fake.handle))
        owned.append(http)
        kwargs.setdefault("api_key", API_KEY)
        kwargs.setdefault("base_url", BASE_URL)
        return Rainbird(http_client=http, **kwargs)

    yield _make
    for http in owned:
        http.close()


@pytest.fixture
def noretry(make_client) -> Rainbird:
    """A client that never retries, so 429/5xx surface immediately."""
    return make_client(retry=NO_RETRY)


@pytest.fixture
def session(client: Rainbird) -> Session:
    return client.start(KM_ID)


# =========================================================================== Rainbird construction


class TestConstruction:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", "env-key")
        with Rainbird() as rb:
            assert rb.config.api_key == "env-key"

    def test_explicit_api_key_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", "env-key")
        with Rainbird(api_key="explicit") as rb:
            assert rb.config.api_key == "explicit"

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(RainbirdConfigurationError, match="API key is required"):
            Rainbird()

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_api_key_raises(self, bad: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            Rainbird(api_key=bad)

    def test_base_url_defaults_to_community(self) -> None:
        with Rainbird(api_key="k") as rb:
            assert rb.base_url == "https://api.rainbird.ai"

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_API_URL", "https://acme-api.rainbird.ai/")
        with Rainbird(api_key="k") as rb:
            assert rb.base_url == "https://acme-api.rainbird.ai"

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            (Environment.COMMUNITY, "https://api.rainbird.ai"),
            (Environment.ENTERPRISE, "https://enterprise-api.rainbird.ai"),
            (Environment.ENTERPRISE_SOUTH_AMERICA, "https://sa-enterprise-api.rainbird.ai"),
        ],
    )
    def test_base_url_from_environment_enum(self, env: Environment, expected: str) -> None:
        with Rainbird(api_key="k", base_url=env) as rb:
            assert rb.base_url == expected

    def test_base_url_adds_scheme_and_strips_trailing_slash(self) -> None:
        with Rainbird(api_key="k", base_url="acme-api.rainbird.ai/") as rb:
            assert rb.base_url == "https://acme-api.rainbird.ai"

    def test_injected_http_client_is_used(self, fake: FakeRainbird) -> None:
        http = httpx.Client(transport=httpx.MockTransport(fake.handle))
        rb = Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http)
        assert rb.http is http
        rb.start(KM_ID)
        assert fake.last.path == f"/start/{KM_ID}"
        http.close()

    def test_context_manager_does_not_close_injected_client(self, client: Rainbird) -> None:
        with client as rb:
            assert rb is client
        assert client.http.is_closed is False

    def test_context_manager_closes_owned_client(self) -> None:
        with Rainbird(api_key="k") as rb:
            assert rb.http.is_closed is False
        assert rb.http.is_closed is True

    def test_close_is_idempotent_for_injected_client(self, client: Rainbird) -> None:
        client.close()
        client.close()
        assert client.http.is_closed is False

    def test_repr(self, client: Rainbird) -> None:
        assert repr(client) == f"Rainbird(base_url={BASE_URL!r})"

    @pytest.mark.parametrize(
        ("base", "studio"),
        [
            (BASE_URL, "https://fake.rainbird.ai"),
            ("https://api.rainbird.ai", "https://app.rainbird.ai"),
            ("https://enterprise-api.rainbird.ai", "https://enterprise.rainbird.ai"),
            ("https://sa-enterprise-api.rainbird.ai", "https://sa-enterprise.rainbird.ai"),
            ("https://acme-api.rainbird.ai", "https://acme.rainbird.ai"),
        ],
    )
    def test_studio_url(self, make_client, base: str, studio: str) -> None:
        assert make_client(base_url=base).studio_url == studio

    def test_environment_enum_studio_url(self) -> None:
        assert Environment.ENTERPRISE.studio_url == "https://enterprise.rainbird.ai"

    def test_evidence_and_interaction_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_EVIDENCE_KEY", "ev-env")
        monkeypatch.setenv("RAINBIRD_INTERACTION_KEY", "ik-env")
        with Rainbird(api_key="k") as rb:
            assert rb.config.evidence_key == "ev-env"
            assert rb.config.interaction_key == "ik-env"

    def test_engine_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAINBIRD_ENGINE", "alt-engine")
        with Rainbird(api_key="k") as rb:
            assert rb.config.engine == "alt-engine"

    # -- timeout: UNSET sentinel, owned client vs user-supplied client ------------------

    def test_timeout_defaults_to_unset_sentinel(self, client: Rainbird) -> None:
        assert UNSET is rainbird_ai.UNSET is http_mod.UNSET
        assert isinstance(UNSET, http_mod.Unset)
        assert not UNSET and repr(UNSET) == "UNSET"
        assert client.config.timeout == 30.0
        assert client.config.timeout_explicit is False

    def test_passing_unset_explicitly_is_the_default(self, make_client) -> None:
        rb = make_client(timeout=UNSET)
        assert rb.config.timeout == 30.0
        assert rb.config.timeout_explicit is False

    @pytest.mark.parametrize("timeout", [5, 2.5, None, httpx.Timeout(5.0, connect=1.0)])
    def test_explicit_timeout_is_recorded_on_config(self, make_client, timeout: Any) -> None:
        rb = make_client(timeout=timeout)
        assert rb.config.timeout == timeout
        assert rb.config.timeout_explicit is True

    def test_owned_client_defaults_to_30_seconds_and_no_redirects(self) -> None:
        with Rainbird(api_key="k") as rb:
            assert rb.http.timeout == httpx.Timeout(30.0)
            assert rb.http.follow_redirects is False

    def test_owned_client_uses_explicit_timeout(self) -> None:
        with Rainbird(api_key="k", timeout=7) as rb:
            assert rb.http.timeout == httpx.Timeout(7)

    def test_user_client_keeps_its_own_timeout_by_default(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0))
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http) as rb:
            rb.start(KM_ID).query("speaks", subject="Tom")
        assert len(seen) == 2
        assert all(r.extensions["timeout"] == httpx.Timeout(9.0).as_dict() for r in seen)
        http.close()

    def test_explicit_timeout_is_applied_per_request_with_user_client(
        self, fake: FakeRainbird
    ) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0))
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http, timeout=5) as rb:
            rb.start(KM_ID).query("speaks", subject="Tom")
            rb.version()
        assert len(seen) == 3
        assert all(r.extensions["timeout"] == httpx.Timeout(5).as_dict() for r in seen)
        assert http.timeout == httpx.Timeout(9.0)  # the caller's client itself is untouched
        http.close()

    def test_explicit_httpx_timeout_object_is_applied_per_request(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        timeout = httpx.Timeout(5.0, connect=1.0)
        http = httpx.Client(transport=recording_transport(fake, seen))
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http, timeout=timeout) as rb:
            rb.version()
        assert seen[-1].extensions["timeout"] == timeout.as_dict()
        http.close()

    def test_explicit_none_timeout_disables_timeouts_per_request(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0))
        with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http, timeout=None) as rb:
            rb.version()
        assert seen[-1].extensions["timeout"] == httpx.Timeout(None).as_dict()
        http.close()


# =========================================================================== start / resume


class TestStart:
    def test_sends_api_key_and_user_agent(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID)
        assert fake.last.method == "GET"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["user-agent"].startswith(f"rainbird-ai-python/{__version__}")
        assert fake.last.headers["accept"] == "application/json"

    def test_path_and_no_params_by_default(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID)
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {}
        assert fake.last.json is None

    def test_use_draft_param(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID, use_draft=True)
        assert fake.last.params == {"useDraft": "true"}

    def test_use_draft_false_param(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID, use_draft=False)
        assert fake.last.params == {"useDraft": "false"}

    def test_version_param(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID, version=3)
        assert fake.last.params == {"version": "3"}

    def test_context_id_param(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID, context_id="ctx-1")
        assert fake.last.params == {"contextid": "ctx-1"}
        assert session.context_id == "ctx-1"

    def test_version_and_context_id_together(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start(KM_ID, version=2, context_id="ctx-1")
        assert fake.last.params == {"version": "2", "contextid": "ctx-1"}

    def test_use_draft_and_version_conflict(self, client: Rainbird, fake: FakeRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="cannot be used together"):
            client.start(KM_ID, use_draft=True, version=2)
        assert fake.requests == []

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5])
    def test_non_positive_or_non_int_version_rejected(
        self, client: Rainbird, fake: FakeRainbird, bad: Any
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="positive integer"):
            client.start(KM_ID, version=bad)
        assert fake.requests == []

    def test_km_id_from_env(
        self, client: Rainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", "env-km")
        session = client.start()
        assert fake.last.path == "/start/env-km"
        assert session.km_id == "env-km"

    def test_explicit_km_id_wins_over_env(
        self, client: Rainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_KM_ID", "env-km")
        client.start(KM_ID)
        assert fake.last.path == f"/start/{KM_ID}"

    def test_missing_km_id_raises(self, client: Rainbird, fake: FakeRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="km_id is required"):
            client.start()
        assert fake.requests == []

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_km_id_raises(self, client: Rainbird, fake: FakeRainbird, bad: str) -> None:
        with pytest.raises(RainbirdConfigurationError):
            client.start(bad)
        assert fake.requests == []

    def test_km_id_is_url_encoded(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.start("odd km/id")
        # httpx decodes the path for us; the wire form is percent-encoded
        assert fake.last.path == "/start/odd km/id"

    def test_returns_session(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID, context_id="ctx")
        assert isinstance(session, Session)
        assert session.id in fake.sessions
        assert session.km_version_id == KM_VERSION_ID
        assert session.km_id == KM_ID
        assert session.context_id == "ctx"
        assert session.client is client
        assert session.start_response is not None
        assert session.start_response.session_id == session.id
        assert session.start_response.km_version_id == KM_VERSION_ID
        assert session.start_response.raw == {"id": session.id, "kmVersion": {"id": KM_VERSION_ID}}
        assert session.last_response is None

    def test_session_repr(self, session: Session) -> None:
        assert repr(session) == f"Session(id={session.id!r}, km_id={KM_ID!r})"

    def test_each_start_creates_a_new_session(self, client: Rainbird) -> None:
        assert client.start(KM_ID).id != client.start(KM_ID).id

    def test_invalid_api_key_is_authentication_error(self, make_client, fake: FakeRainbird) -> None:
        with pytest.raises(AuthenticationError) as info:
            make_client(api_key="wrong").start(KM_ID)
        assert info.value.status_code == 401
        assert info.value.errors == ["Your API key is invalid"]


class TestResume:
    def test_makes_no_request(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.resume("existing-sid")
        assert fake.requests == []
        assert session.id == "existing-sid"
        assert session.km_id is None
        assert session.km_version_id is None
        assert session.start_response is None
        assert session.pending is None and session.result is None

    def test_km_id_passthrough(self, client: Rainbird) -> None:
        assert client.resume("sid", km_id=KM_ID).km_id == KM_ID

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_empty_session_id_raises(self, client: Rainbird, bad: str) -> None:
        with pytest.raises(RainbirdConfigurationError, match="session_id"):
            client.resume(bad)

    def test_resumed_session_uses_id_in_paths(self, client: Rainbird, fake: FakeRainbird) -> None:
        sid = client.start(KM_ID).id  # make the fake aware of it
        client.resume(sid).query("speaks", subject="Tom")
        assert fake.last.path == f"/{sid}/query"


# =========================================================================== inject


class TestInject:
    def test_fact_object(self, session: Session, fake: FakeRainbird) -> None:
        response = session.inject(Fact("Tom", "lives in", "Germany"))
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/inject"
        assert fake.last.json == [triple("Tom", "lives in", "Germany")]
        assert response.ok is True
        assert response.result == "OK"
        assert response.raw == {"result": "OK"}

    def test_dict(self, session: Session, fake: FakeRainbird) -> None:
        session.inject({"subject": "Tom", "relationship": "lives in", "object": "Germany"})
        assert fake.last.json == [triple("Tom", "lives in", "Germany")]

    def test_dict_with_cf_alias(self, session: Session, fake: FakeRainbird) -> None:
        session.inject(
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "cf": 40}
        )
        assert fake.last.json == [triple("Tom", "lives in", "Germany", 40)]

    def test_tuple_with_certainty(self, session: Session, fake: FakeRainbird) -> None:
        session.inject(("Tom", "lives in", "Germany", 80))
        assert fake.last.json == [triple("Tom", "lives in", "Germany", 80)]

    def test_triple_without_certainty_defaults_to_100(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.inject(("Tom", "lives in", "Germany"))
        assert fake.last.json == [triple("Tom", "lives in", "Germany", 100)]

    def test_mixed_list_in_one_request(self, session: Session, fake: FakeRainbird) -> None:
        session.inject(
            [
                Fact("Tom", "lives in", "Germany"),
                {"subject": "Tom", "relationship": "age", "object": 30, "certainty": 90},
                ("Ana", "lives in", "France", 70),
            ]
        )
        assert len(fake.calls("/inject")) == 1
        assert fake.last.json == [
            triple("Tom", "lives in", "Germany"),
            triple("Tom", "age", 30, 90),
            triple("Ana", "lives in", "France", 70),
        ]

    def test_fact_values_are_coerced(self, session: Session, fake: FakeRainbird) -> None:
        session.inject(("Tom", "born on", dt.date(2000, 1, 2)))
        assert fake.last.json == [triple("Tom", "born on", "2000-01-02")]

    def test_batches_of_250(self, session: Session, fake: FakeRainbird) -> None:
        facts = [("s", "rel", i) for i in range(600)]
        session.inject(facts)
        calls = fake.calls("/inject")
        assert [len(c.json) for c in calls] == [250, 250, 100]
        assert calls[0].json[0] == triple("s", "rel", 0)
        assert calls[2].json[-1] == triple("s", "rel", 599)
        assert len(fake.sessions[session.id]["facts"]) == 600

    def test_exactly_250_is_one_request(self, session: Session, fake: FakeRainbird) -> None:
        session.inject([("s", "rel", i) for i in range(250)])
        assert [len(c.json) for c in fake.calls("/inject")] == [250]

    def test_custom_batch_size(self, session: Session, fake: FakeRainbird) -> None:
        session.inject([("s", "rel", i) for i in range(7)], batch_size=3)
        assert [len(c.json) for c in fake.calls("/inject")] == [3, 3, 1]

    def test_returns_last_batch_response(self, session: Session) -> None:
        response = session.inject([("s", "rel", i) for i in range(5)], batch_size=2)
        assert response.ok

    @pytest.mark.parametrize("bad", [0, -5])
    def test_batch_size_below_one_raises(
        self, session: Session, fake: FakeRainbird, bad: int
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="batch_size"):
            session.inject([("s", "rel", "o")], batch_size=bad)
        assert fake.calls("/inject") == []

    def test_inject_fact(self, session: Session, fake: FakeRainbird) -> None:
        response = session.inject_fact("Tom", "lives in", "Germany", 90)
        assert fake.last.json == [triple("Tom", "lives in", "Germany", 90)]
        assert response.ok

    def test_inject_fact_default_certainty(self, session: Session, fake: FakeRainbird) -> None:
        session.inject_fact("Tom", "lives in", "Germany")
        assert fake.last.json == [triple("Tom", "lives in", "Germany", 100)]

    @pytest.mark.parametrize(
        "bad",
        [42, [], {"subject": "Tom", "relationship": "x"}, ("Tom", "x"), ("Tom", "x", "y", 101)],
    )
    def test_invalid_facts_raise_before_any_request(
        self, session: Session, fake: FakeRainbird, bad: Any
    ) -> None:
        with pytest.raises(InvalidFactError):
            session.inject(bad)
        assert fake.calls("/inject") == []

    def test_facts_are_recorded_by_fake_session(self, session: Session, fake: FakeRainbird) -> None:
        session.inject_fact("Tom", "lives in", "Germany")
        assert fake.sessions[session.id]["facts"] == [triple("Tom", "lives in", "Germany")]


# =========================================================================== query


class TestQuery:
    def test_body_with_subject(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="Tom")
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/query"
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom"}

    def test_body_with_object(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", object="German")
        assert fake.last.json == {"relationship": "speaks", "object": "German"}

    def test_body_with_subject_and_object(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="Tom", object="German")
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom", "object": "German"}

    def test_body_relationship_only(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks")
        assert fake.last.json == {"relationship": "speaks"}

    def test_empty_strings_are_omitted(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="", object="")
        assert fake.last.json == {"relationship": "speaks"}

    def test_numeric_subject_is_kept(self, session: Session, fake: FakeRainbird) -> None:
        session.query("is even", subject=0)
        assert fake.last.json == {"relationship": "is even", "subject": 0}

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_relationship_raises(
        self, session: Session, fake: FakeRainbird, bad: Any
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="relationship"):
            session.query(bad, subject="Tom")
        assert fake.calls("/query") == []

    def test_returns_result_response(self, session: Session) -> None:
        response = session.query("speaks", subject="Tom")
        assert isinstance(response, ResultResponse)
        assert response.is_result and not response.is_question
        assert response.question is None
        assert response.sid == session.id
        assert len(response) == 1
        item = response.results[0]
        assert (item.subject, item.relationship, item.object) == ("Tom", "speaks", "German")
        assert item.certainty == 75
        assert item.fact_id == ROOT_FACT_ID
        assert response.best is item
        assert response.fact_ids == [ROOT_FACT_ID]

    def test_returns_question_response(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        response = session.query("national language", subject="Germany")
        assert isinstance(response, QuestionResponse)
        assert response.is_question and not response.is_result
        assert response.results == []
        assert response.sid == session.id
        question = response.question
        assert question.prompt == "What is the national language of Germany?"
        assert question.form is QuestionForm.SECOND_OBJECT
        assert question.options == ["English", "French"]
        assert question.can_add is True and question.allow_unknown is False

    def test_extra_questions(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT, QUESTION_FIRST_FORM)
        response = session.query("national language", subject="Germany")
        assert isinstance(response, QuestionResponse)
        assert [x.type for x in response.questions] == ["Second Form Object", "First Form"]
        assert response.extra_questions[0].is_first_form

    def test_empty_result_list(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_result()
        response = session.query("speaks", subject="Nobody")
        assert isinstance(response, ResultResponse)
        assert response.results == [] and response.best is None

    def test_unexpected_body_raises(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue({"weird": 1})
        with pytest.raises(UnexpectedResponseError) as info:
            session.query("speaks", subject="Tom")
        assert info.value.body == {"weird": 1, "sid": session.id}
        assert info.value.status_code == 200
        assert "neither" in str(info.value)

    def test_unexpected_body_leaves_state_untouched(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        session.query("national language", subject="Germany")
        fake.queue({"weird": 1})
        with pytest.raises(UnexpectedResponseError):
            session.answer("German")
        assert session.pending_question is not None
        assert session.pending_question.relationship == "national language"

    def test_non_json_2xx_body_raises_unexpected_response_error(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200, "<html>oops</html>")
        with pytest.raises(UnexpectedResponseError) as info:
            session.query("speaks", subject="Tom")
        assert type(info.value) is UnexpectedResponseError  # not a RainbirdAPIError subclass
        assert str(info.value) == (
            f"expected a JSON body from POST {BASE_URL}/{session.id}/query "
            "but got: '<html>oops</html>'"
        )
        assert info.value.status_code == 200
        assert info.value.body == "<html>oops</html>"
        assert session.pending_question is None and session.last_response is None

    def test_empty_2xx_body_raises_unexpected_response_error(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200)
        with pytest.raises(UnexpectedResponseError, match="expected a JSON body") as info:
            session.query("speaks", subject="Tom")
        assert info.value.body == ""
        assert info.value.status_code == 200

    def test_non_json_2xx_snippet_is_capped_at_200_chars(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.fail_next(200, "x" * 500)
        with pytest.raises(UnexpectedResponseError) as info:
            session.query("speaks", subject="Tom")
        assert str(info.value).endswith("but got: '" + "x" * 200 + "'")
        assert info.value.body == "x" * 500  # the full text stays available on the exception


# =========================================================================== pending state


class TestPendingState:
    def test_initial_state(self, session: Session) -> None:
        assert session.last_response is None
        assert session.pending is None
        assert session.pending_question is None
        assert session.result is None

    def test_question_then_answer_then_result(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*_result_items())
        asked = session.query("national language", subject="Germany")
        assert session.pending is asked
        assert session.pending_question is asked.question
        assert session.result is None
        assert session.last_response is asked

        answered = session.answer("German")
        assert isinstance(answered, ResultResponse)
        assert session.pending is None
        assert session.pending_question is None
        assert session.result is answered
        assert session.last_response is answered

    def test_undo_restores_pending(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*_result_items())
        fake.queue_question(QUESTION_SECOND_OBJECT)
        session.query("national language", subject="Germany")
        session.answer("German")
        assert session.result is not None
        undone = session.undo()
        assert isinstance(undone, QuestionResponse)
        assert session.pending is undone
        assert session.result is None
        assert session.pending_question.prompt == "What is the national language of Germany?"

    def test_undo_sends_empty_object(self, session: Session, fake: FakeRainbird) -> None:
        session.undo()
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/undo"
        assert fake.last.json == {}

    def test_undo_may_return_result(self, session: Session, fake: FakeRainbird) -> None:
        response = session.undo()
        assert isinstance(response, ResultResponse)
        assert session.result is response

    def test_result_after_question_chain(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(*_result_items())
        session.query("speaks", subject="Tom")
        session.answer("German")
        assert session.pending_question.is_first_form
        final = session.answer(True)
        assert session.result is final
        assert [r.path.rsplit("/", 1)[1] for r in fake.requests[1:]] == [
            "query",
            "response",
            "response",
        ]


def _result_items() -> list[dict[str, Any]]:
    return list(RESULT_TOM_SPEAKS_GERMAN["result"])


# =========================================================================== respond


class TestRespond:
    def test_raw_dict(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(triple("Germany", "national language", "German"))
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {"answers": [triple("Germany", "national language", "German")]}

    def test_answer_object(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(Answer("Germany", "national language", "German", certainty=80))
        assert fake.last.json == {"answers": [triple("Germany", "national language", "German", 80)]}

    def test_list_of_dicts_and_answers(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(
            [
                Answer("Ben", "suffered", "theft"),
                {"subject": "Ben", "relationship": "suffered", "object": "water damage"},
            ]
        )
        assert fake.last.json == {
            "answers": [
                triple("Ben", "suffered", "theft"),
                triple("Ben", "suffered", "water damage"),
            ]
        }

    def test_first_form_dict_normalises_answer(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(triple("Tom", "speaks", "French", answer=True))
        assert fake.last.json == {"answers": [triple("Tom", "speaks", "French", answer="yes")]}

    def test_unanswered_dict(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(triple("Germany", "national language", "", unanswered=True))
        assert fake.last.json == {
            "answers": [triple("Germany", "national language", "", unanswered=True)]
        }

    def test_tracks_response(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        response = session.respond(triple("Germany", "national language", "German"))
        assert session.pending is response

    @pytest.mark.parametrize(
        "bad", [{"subject": "a", "relationship": "b"}, [], "German", [{"foo": 1}]]
    )
    def test_invalid_answers_raise_before_request(
        self, session: Session, fake: FakeRainbird, bad: Any
    ) -> None:
        with pytest.raises(InvalidFactError):
            session.respond(bad)
        assert fake.calls("/response") == []

    def test_fake_records_answers(self, session: Session, fake: FakeRainbird) -> None:
        session.respond(Answer("Germany", "national language", "German"))
        assert fake.sessions[session.id]["answers"] == [
            {"answers": [triple("Germany", "national language", "German")]}
        ]


# =========================================================================== answer / skip


class TestAnswer:
    def _ask(self, session: Session, fake: FakeRainbird, question: dict[str, Any]) -> None:
        fake.queue_question(question)
        session.query(question["relationship"], subject=question.get("subject"))

    # -- second form object ----------------------------------------------------------

    def test_second_form_object_exact_json(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        response = session.answer("German")
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {"answers": [triple("Germany", "national language", "German")]}
        assert isinstance(response, ResultResponse)

    def test_second_form_object_option_value(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        session.answer("French")
        assert fake.last.json == {"answers": [triple("Germany", "national language", "French")]}

    def test_second_form_object_numeric_value_kept(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, {**QUESTION_SECOND_OBJECT, "dataType": "number", "concepts": []})
        session.answer(42)
        assert fake.last.json == {"answers": [triple("Germany", "national language", 42)]}

    def test_multiple_values_for_non_plural_raises(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        with pytest.raises(InvalidAnswerError, match="single answer"):
            session.answer(["German", "French"])
        assert fake.calls("/response") == []

    # -- second form subject ---------------------------------------------------------

    def test_second_form_subject_exact_json(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.answer("Tom")
        assert fake.last.json == {"answers": [triple("Tom", "speaks", "French")]}

    def test_second_form_subject_plural_list(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.answer(["Tom", "Ana"])
        assert fake.last.json == {
            "answers": [triple("Tom", "speaks", "French"), triple("Ana", "speaks", "French")]
        }

    def test_value_outside_options_when_can_add_false_raises(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        with pytest.raises(InvalidAnswerError, match="not one of the allowed answers"):
            session.answer("Bob")
        assert fake.calls("/response") == []

    # -- first form ------------------------------------------------------------------

    @pytest.mark.parametrize("value", [True, "yes", "YES", "y", 1])
    def test_first_form_yes(self, session: Session, fake: FakeRainbird, value: Any) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        session.answer(value)
        assert fake.last.json == {"answers": [triple("Tom", "speaks", "French", answer="yes")]}

    @pytest.mark.parametrize("value", [False, "no", "n", 0])
    def test_first_form_no(self, session: Session, fake: FakeRainbird, value: Any) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        session.answer(value)
        assert fake.last.json == {"answers": [triple("Tom", "speaks", "French", answer="no")]}

    def test_first_form_invalid_value_raises(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        with pytest.raises(InvalidAnswerError, match="yes/no"):
            session.answer("maybe")
        assert fake.calls("/response") == []

    # -- plural --------------------------------------------------------------------

    def test_plural_list_sends_one_answer_per_value(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_PLURAL_LOSS)
        session.answer(["water damage", "theft"])
        assert fake.last.json == {
            "answers": [
                triple("Ben", "suffered", "water damage"),
                triple("Ben", "suffered", "theft"),
            ]
        }

    def test_plural_single_value(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_PLURAL_LOSS)
        session.answer("theft")
        assert fake.last.json == {"answers": [triple("Ben", "suffered", "theft")]}

    def test_plural_empty_list_raises(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_PLURAL_LOSS)
        with pytest.raises(InvalidAnswerError, match="at least one value"):
            session.answer([])
        assert fake.calls("/response") == []

    # -- certainty -----------------------------------------------------------------

    def test_certainty(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        session.answer("German", certainty=60)
        assert fake.last.json == {"answers": [triple("Germany", "national language", "German", 60)]}

    def test_certainty_applies_to_every_plural_answer(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_PLURAL_LOSS)
        session.answer(["theft", "water damage"], certainty=55)
        assert [a["certainty"] for a in fake.last.json["answers"]] == [55, 55]

    @pytest.mark.parametrize("bad", [101, -1, "high", True])
    def test_invalid_certainty_raises(self, session: Session, fake: FakeRainbird, bad: Any) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        with pytest.raises(InvalidFactError):
            session.answer("German", certainty=bad)
        assert fake.calls("/response") == []

    # -- skipping ------------------------------------------------------------------

    def test_answer_skip_first_form_exact_json(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        session.answer(SKIP)
        assert fake.last.json == {
            "answers": [triple("Tom", "speaks", "French", 100, unanswered=True)]
        }
        assert "answer" not in fake.last.json["answers"][0]

    def test_skip_second_form_subject_exact_json(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.skip()
        assert fake.last.json == {"answers": [triple("", "speaks", "French", 100, unanswered=True)]}

    def test_skip_second_form_object_exact_json(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, {**QUESTION_SECOND_OBJECT, "allowUnknown": True})
        session.skip()
        assert fake.last.json == {
            "answers": [triple("Germany", "national language", "", 100, unanswered=True)]
        }

    def test_skip_and_answer_skip_are_identical(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.skip()
        via_skip = fake.last.json
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.answer(SKIP)
        assert fake.last.json == via_skip

    def test_skip_not_allowed_raises_and_sends_nothing(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)  # allowUnknown False
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            session.skip()
        with pytest.raises(InvalidAnswerError, match="cannot be skipped"):
            session.answer(SKIP)
        assert fake.calls("/response") == []

    def test_skip_strict_false_sends_anyway(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        session.skip(strict=False)
        assert fake.last.json == {
            "answers": [triple("Germany", "national language", "", 100, unanswered=True)]
        }

    def test_skip_rejected_by_api_maps_to_bad_request(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        fake.fail_next(400, {"err": ["Please provide an expected boolean value for unanswered."]})
        with pytest.raises(BadRequestError) as info:
            session.skip(strict=False)
        assert info.value.errors == ["Please provide an expected boolean value for unanswered."]
        assert info.value.status_code == 400

    def test_skip_returns_next_response(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        fake.queue_question(QUESTION_SECOND_OBJECT)
        response = session.skip()
        assert isinstance(response, QuestionResponse)
        assert session.pending_question.relationship == "national language"

    # -- nothing pending -----------------------------------------------------------

    def test_answer_without_pending_raises(self, session: Session, fake: FakeRainbird) -> None:
        with pytest.raises(NoPendingQuestionError):
            session.answer("German")
        assert fake.calls("/response") == []

    def test_answer_skip_without_pending_raises(self, session: Session) -> None:
        with pytest.raises(NoPendingQuestionError):
            session.answer(SKIP)

    def test_skip_without_pending_raises(self, session: Session) -> None:
        with pytest.raises(NoPendingQuestionError):
            session.skip()

    def test_answer_after_result_raises(self, session: Session) -> None:
        session.query("speaks", subject="Tom")  # default fake body is a result
        assert session.result is not None
        with pytest.raises(NoPendingQuestionError):
            session.answer("German")

    # -- question override ---------------------------------------------------------

    def test_question_override_without_pending(self, session: Session, fake: FakeRainbird) -> None:
        response = session.answer("German", question=q(QUESTION_SECOND_OBJECT))
        assert fake.last.json == {"answers": [triple("Germany", "national language", "German")]}
        assert isinstance(response, ResultResponse)

    def test_question_override_takes_precedence_over_pending(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        session.answer(True, question=q(QUESTION_FIRST_FORM))
        assert fake.last.json == {"answers": [triple("Tom", "speaks", "French", answer="yes")]}

    def test_skip_with_question_override(self, session: Session, fake: FakeRainbird) -> None:
        session.skip(question=q(QUESTION_SECOND_SUBJECT))
        assert fake.last.json == {"answers": [triple("", "speaks", "French", 100, unanswered=True)]}

    # -- strict=False ----------------------------------------------------------------

    def test_strict_false_allows_value_outside_options(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_SUBJECT)
        session.answer("Bob", strict=False)
        assert fake.last.json == {"answers": [triple("Bob", "speaks", "French")]}

    def test_strict_false_allows_multiple_values_for_non_plural(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        session.answer(["German", "French"], strict=False)
        assert fake.last.json == {
            "answers": [
                triple("Germany", "national language", "German"),
                triple("Germany", "national language", "French"),
            ]
        }

    def test_strict_false_still_rejects_bad_yes_no(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        self._ask(session, fake, QUESTION_FIRST_FORM)
        with pytest.raises(InvalidAnswerError):
            session.answer("maybe", strict=False)
        assert fake.calls("/response") == []

    # -- tracking ------------------------------------------------------------------

    def test_answer_tracks_next_question(self, session: Session, fake: FakeRainbird) -> None:
        self._ask(session, fake, QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        response = session.answer("German")
        assert isinstance(response, QuestionResponse)
        assert session.pending is response
        assert session.pending_question.prompt == "Does Tom speak French?"


# =========================================================================== run


class TestRun:
    def test_drives_to_result_with_values(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_FIRST_FORM)
        fake.queue_result(*_result_items())
        seen: list[Question] = []

        def answerer(question: Question) -> Any:
            seen.append(question)
            return {"national language": "German", "speaks": True}[question.relationship]

        result = session.run("speaks", subject="Tom", answerer=answerer)
        assert isinstance(result, ResultResponse)
        assert result.results[0].object == "German"
        assert [x.prompt for x in seen] == [
            "What is the national language of Germany?",
            "Does Tom speak French?",
        ]
        assert fake.calls("/query")[0].json == {"relationship": "speaks", "subject": "Tom"}
        assert [c.json for c in fake.calls("/response")] == [
            {"answers": [triple("Germany", "national language", "German")]},
            {"answers": [triple("Tom", "speaks", "French", answer="yes")]},
        ]
        assert session.result is result

    def test_returns_immediately_when_first_response_is_result(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        calls = 0

        def answerer(question: Question) -> Any:
            nonlocal calls
            calls += 1
            return "x"

        result = session.run("speaks", subject="Tom", object="German", answerer=answerer)
        assert isinstance(result, ResultResponse)
        assert calls == 0
        assert fake.calls("/response") == []
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom", "object": "German"}

    def test_skip(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)  # allowUnknown True
        fake.queue_result(*_result_items())
        result = session.run("speaks", subject="Tom", answerer=lambda _q: SKIP)
        assert isinstance(result, ResultResponse)
        assert fake.calls("/response")[0].json == {
            "answers": [triple("Tom", "speaks", "French", 100, unanswered=True)]
        }

    def test_skip_not_allowed_raises(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)  # allowUnknown False
        with pytest.raises(InvalidAnswerError):
            session.run("national language", subject="Germany", answerer=lambda _q: SKIP)
        assert fake.calls("/response") == []

    def test_answer_object(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*_result_items())
        custom = Answer("Germany", "national language", "Bavarian", certainty=42)
        session.run("national language", subject="Germany", answerer=lambda _q: custom)
        assert fake.calls("/response")[0].json == {
            "answers": [triple("Germany", "national language", "Bavarian", 42)]
        }

    def test_list_of_answers(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(*_result_items())
        answers = [
            Answer("Ben", "suffered", "theft"),
            Answer("Ben", "suffered", "water damage", 70),
        ]
        session.run("can claim", subject="Ben", answerer=lambda _q: answers)
        assert fake.calls("/response")[0].json == {
            "answers": [
                triple("Ben", "suffered", "theft"),
                triple("Ben", "suffered", "water damage", 70),
            ]
        }

    def test_list_of_values_for_plural_question(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        fake.queue_result(*_result_items())
        session.run("can claim", subject="Ben", answerer=lambda _q: ["theft", "cosmetic damage"])
        assert fake.calls("/response")[0].json == {
            "answers": [
                triple("Ben", "suffered", "theft"),
                triple("Ben", "suffered", "cosmetic damage"),
            ]
        }

    def test_max_questions_exceeded_raises(self, session: Session, fake: FakeRainbird) -> None:
        for _ in range(3):
            fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*_result_items())
        with pytest.raises(RainbirdError, match="more than 2 questions") as info:
            session.run(
                "national language",
                subject="Germany",
                answerer=lambda _q: "German",
                max_questions=2,
            )
        assert not isinstance(info.value, RainbirdAPIError)
        assert len(fake.calls("/response")) == 2
        assert session.pending_question is not None  # the third question is still pending

    def test_max_questions_boundary_allows_exact_count(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_question(QUESTION_SECOND_OBJECT)
        fake.queue_result(*_result_items())
        result = session.run(
            "national language", subject="Germany", answerer=lambda _q: "German", max_questions=2
        )
        assert isinstance(result, ResultResponse)
        assert len(fake.calls("/response")) == 2

    def test_invalid_answer_propagates(self, session: Session, fake: FakeRainbird) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        with pytest.raises(InvalidAnswerError):
            session.run("speaks", subject="Tom", answerer=lambda _q: "maybe")
        assert fake.calls("/response") == []


# =========================================================================== evidence


class TestEvidence:
    def test_request_path_and_no_key_header_by_default(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.evidence(ROOT_FACT_ID)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/evidence/{ROOT_FACT_ID}/{session.id}"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert "x-evidence-key" not in fake.last.headers

    def test_parses_root_evidence(self, session: Session) -> None:
        ev = session.evidence(ROOT_FACT_ID)
        assert isinstance(ev, Evidence)
        assert ev.fact_id == ROOT_FACT_ID
        assert ev.source is FactSource.RULE
        assert ev.is_inferred
        assert (ev.fact.subject.value, ev.fact.relationship, ev.fact.object.value) == (
            "Tom",
            "speaks",
            "German",
        )
        assert ev.fact.certainty == 75
        assert ev.time == 1751644730474
        assert ev.timestamp == dt.datetime.fromtimestamp(1751644730.474, tz=UTC)
        assert ev.rule is not None
        assert ev.rule.bindings == {"COUNTRY": "Germany", "O": "German", "S": "Tom"}
        assert ev.rule.max_certainty == 75
        assert len(ev.rule.relationship_conditions) == 2
        assert len(ev.rule.expression_conditions) == 1
        assert ev.rule.expression_conditions[0].was_met is True
        assert ev.child_fact_ids == [LEAF_IF_ID, LEAF_AF_ID]
        assert ev.raw == EVIDENCE_ROOT

    def test_parses_leaf_evidence(self, session: Session) -> None:
        ev = session.evidence(LEAF_IF_ID)
        assert ev.source is FactSource.INJECTION
        assert ev.rule is None and not ev.is_inferred
        assert ev.child_fact_ids == []

    def test_client_default_key_header(self, make_client, fake: FakeRainbird) -> None:
        rb = make_client(evidence_key="default-key")
        rb.start(KM_ID).evidence(ROOT_FACT_ID)
        assert fake.last.headers["x-evidence-key"] == "default-key"

    def test_key_from_env(self, make_client, fake: FakeRainbird, monkeypatch) -> None:
        monkeypatch.setenv("RAINBIRD_EVIDENCE_KEY", "env-key")
        make_client().start(KM_ID).evidence(ROOT_FACT_ID)
        assert fake.last.headers["x-evidence-key"] == "env-key"

    def test_explicit_key_overrides_client_default(self, make_client, fake: FakeRainbird) -> None:
        rb = make_client(evidence_key="default-key")
        rb.start(KM_ID).evidence(ROOT_FACT_ID, evidence_key="explicit-key")
        assert fake.last.headers["x-evidence-key"] == "explicit-key"

    def test_explicit_key_without_default(self, session: Session, fake: FakeRainbird) -> None:
        session.evidence(ROOT_FACT_ID, evidence_key="only-explicit")
        assert fake.last.headers["x-evidence-key"] == "only-explicit"

    def test_secured_evidence_without_key_is_permission_denied(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.require_evidence_key = "secret"
        with pytest.raises(PermissionDeniedError) as info:
            session.evidence(ROOT_FACT_ID)
        assert info.value.status_code == 403
        assert info.value.errors == ["Evidence is secured; provide x-evidence-key"]

    def test_secured_evidence_with_key_succeeds(self, session: Session, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "secret"
        assert session.evidence(ROOT_FACT_ID, evidence_key="secret").fact_id == ROOT_FACT_ID

    def test_secured_evidence_with_wrong_key_is_permission_denied(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.require_evidence_key = "secret"
        with pytest.raises(PermissionDeniedError):
            session.evidence(ROOT_FACT_ID, evidence_key="wrong")

    def test_unknown_fact_is_not_found(self, session: Session) -> None:
        with pytest.raises(NotFoundError) as info:
            session.evidence("WA:RF:doesnotexist")
        assert info.value.status_code == 404

    def test_client_level_evidence(
        self, client: Rainbird, session: Session, fake: FakeRainbird
    ) -> None:
        ev = client.evidence(session.id, ROOT_FACT_ID, evidence_key="k")
        assert ev.fact_id == ROOT_FACT_ID
        assert fake.last.path == f"/analysis/evidence/{ROOT_FACT_ID}/{session.id}"
        assert fake.last.headers["x-evidence-key"] == "k"

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_empty_fact_id_raises(self, session: Session, fake: FakeRainbird, bad: str) -> None:
        with pytest.raises(RainbirdConfigurationError, match="fact_id"):
            session.evidence(bad)
        assert fake.calls("/analysis") == []


class TestEvidenceTree:
    def test_fetches_root_and_leaves_once_each(self, session: Session, fake: FakeRainbird) -> None:
        tree = session.evidence_tree(ROOT_FACT_ID)
        assert [e.fact_id for e in tree] == [ROOT_FACT_ID, LEAF_IF_ID, LEAF_AF_ID]
        calls = fake.calls("/analysis/evidence")
        assert len(calls) == 3
        assert [c.path for c in calls] == [
            f"/analysis/evidence/{fid}/{session.id}"
            for fid in (ROOT_FACT_ID, LEAF_IF_ID, LEAF_AF_ID)
        ]

    def test_root_first_and_all_evidence_objects(self, session: Session) -> None:
        tree = session.evidence_tree(ROOT_FACT_ID)
        assert all(isinstance(e, Evidence) for e in tree)
        assert tree[0].is_inferred and not tree[1].is_inferred and not tree[2].is_inferred

    def test_max_depth_zero_returns_only_root(self, session: Session, fake: FakeRainbird) -> None:
        tree = session.evidence_tree(ROOT_FACT_ID, max_depth=0)
        assert [e.fact_id for e in tree] == [ROOT_FACT_ID]
        assert len(fake.calls("/analysis/evidence")) == 1

    def test_max_depth_one_reaches_leaves(self, session: Session, fake: FakeRainbird) -> None:
        tree = session.evidence_tree(ROOT_FACT_ID, max_depth=1)
        assert len(tree) == 3
        assert len(fake.calls("/analysis/evidence")) == 3

    def test_starting_at_a_leaf(self, session: Session, fake: FakeRainbird) -> None:
        tree = session.evidence_tree(LEAF_AF_ID)
        assert [e.fact_id for e in tree] == [LEAF_AF_ID]
        assert len(fake.calls("/analysis/evidence")) == 1

    def test_cycles_are_not_refetched(self, session: Session, fake: FakeRainbird) -> None:
        # make the IF leaf point back at the root
        looped = {**EVIDENCE_LEAVES[LEAF_IF_ID], "rule": {"conditions": [{"factID": ROOT_FACT_ID}]}}
        fake.evidence[LEAF_IF_ID] = looped
        tree = session.evidence_tree(ROOT_FACT_ID)
        assert [e.fact_id for e in tree] == [ROOT_FACT_ID, LEAF_IF_ID, LEAF_AF_ID]
        assert len(fake.calls("/analysis/evidence")) == 3

    def test_passes_evidence_key(self, session: Session, fake: FakeRainbird) -> None:
        fake.require_evidence_key = "secret"
        tree = session.evidence_tree(ROOT_FACT_ID, evidence_key="secret")
        assert len(tree) == 3
        assert all(
            c.headers["x-evidence-key"] == "secret" for c in fake.calls("/analysis/evidence")
        )

    def test_iter_is_lazy(self, session: Session, fake: FakeRainbird) -> None:
        iterator = iter(session.iter_evidence_tree(ROOT_FACT_ID))
        assert fake.calls("/analysis/evidence") == []
        first = next(iterator)
        assert first.fact_id == ROOT_FACT_ID
        assert len(fake.calls("/analysis/evidence")) == 1
        second = next(iterator)
        assert second.fact_id == LEAF_IF_ID
        assert len(fake.calls("/analysis/evidence")) == 2
        assert [e.fact_id for e in iterator] == [LEAF_AF_ID]
        assert len(fake.calls("/analysis/evidence")) == 3

    def test_iter_can_be_abandoned_early(self, session: Session, fake: FakeRainbird) -> None:
        for ev in session.iter_evidence_tree(ROOT_FACT_ID):
            if ev.fact_id == ROOT_FACT_ID:
                break
        assert len(fake.calls("/analysis/evidence")) == 1


class TestEvidenceUrl:
    def test_session_evidence_url_format(self, session: Session) -> None:
        assert session.evidence_url(ROOT_FACT_ID) == (
            f"https://fake.rainbird.ai/evidence?id={ROOT_FACT_ID}&api={BASE_URL}&sid={session.id}"
        )

    def test_client_evidence_url_format(self, client: Rainbird) -> None:
        assert client.evidence_url("S1", "F1") == (
            f"https://fake.rainbird.ai/evidence?id=F1&api={BASE_URL}&sid=S1"
        )

    def test_custom_studio_url_trailing_slash_stripped(self, session: Session) -> None:
        url = session.evidence_url("F1", studio_url="https://studio.example.com/")
        assert url == f"https://studio.example.com/evidence?id=F1&api={BASE_URL}&sid={session.id}"

    def test_community_host(self, make_client) -> None:
        rb = make_client(base_url=Environment.COMMUNITY)
        assert rb.evidence_url("S", "F") == (
            "https://app.rainbird.ai/evidence?id=F&api=https://api.rainbird.ai&sid=S"
        )


# =========================================================================== interactions


class TestInteractions:
    def test_json_returns_events(self, session: Session) -> None:
        events = session.interactions()
        assert isinstance(events, list)
        assert all(isinstance(e, InteractionEvent) for e in events)
        assert [e.event for e in events] == ["start", "query", "result"]
        assert events[0].created == dt.datetime(2025, 7, 10, 10, 6, 27, 780000, tzinfo=UTC)
        assert events[1].values == {"query": {"subject": "Tom", "relationship": "speaks"}}
        assert events[2].values["results"][0]["object"] == "German"

    def test_json_request(self, session: Session, fake: FakeRainbird) -> None:
        session.interactions()
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/interactions/{session.id}"
        assert fake.last.params == {"format": "json"}
        assert fake.last.headers["accept"] == "application/json"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert "x-interaction-key" not in fake.last.headers

    def test_json_explicit_format(self, session: Session, fake: FakeRainbird) -> None:
        events = session.interactions(format="json")
        assert isinstance(events, list) and len(events) == 3
        assert fake.last.params == {"format": "json"}

    def test_csv_returns_text_and_sends_format_param(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        csv = session.interactions(format="csv")
        assert isinstance(csv, str)
        assert csv.startswith("event,created\n")
        assert "start,2025-07-10T10:06:27.780Z" in csv
        assert fake.last.params == {"format": "csv"}
        assert fake.last.path == f"/analysis/interactions/{session.id}"

    def test_csv_accept_header(self, session: Session, fake: FakeRainbird) -> None:
        session.interactions(format="csv")
        assert "text/csv" in fake.last.headers["accept"]

    def test_explicit_interaction_key_header(self, session: Session, fake: FakeRainbird) -> None:
        session.interactions(interaction_key="ik-explicit")
        assert fake.last.headers["x-interaction-key"] == "ik-explicit"

    def test_client_default_interaction_key(self, make_client, fake: FakeRainbird) -> None:
        make_client(interaction_key="ik-default").start(KM_ID).interactions(format="csv")
        assert fake.last.headers["x-interaction-key"] == "ik-default"

    def test_explicit_key_overrides_default(self, make_client, fake: FakeRainbird) -> None:
        make_client(interaction_key="ik-default").start(KM_ID).interactions(
            interaction_key="ik-explicit"
        )
        assert fake.last.headers["x-interaction-key"] == "ik-explicit"

    def test_secured_log_without_key_is_permission_denied(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.require_interaction_key = "secret"
        with pytest.raises(PermissionDeniedError) as info:
            session.interactions()
        assert info.value.status_code == 403

    def test_secured_log_with_key_succeeds(self, session: Session, fake: FakeRainbird) -> None:
        fake.require_interaction_key = "secret"
        events = session.interactions(interaction_key="secret")
        assert len(events) == 3


# =========================================================================== session analysis


class TestSessionAnalysis:
    def test_knowledge_map(self, session: Session) -> None:
        km = session.knowledge_map()
        assert isinstance(km, KnowledgeMapVersion)
        assert km.id == KM_ID
        assert km.name == "Languages"
        assert km.version_id == KM_VERSION_ID
        assert km.version_status == "Version3"
        assert km.version_number == 3
        assert km.is_draft is False
        assert km.version_created == dt.datetime(2025, 7, 9, 14, 27, 22, tzinfo=UTC)

    def test_knowledge_map_request(self, session: Session, fake: FakeRainbird) -> None:
        session.knowledge_map()
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/session/{session.id}"
        assert fake.last.params == {"filter": "version"}
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_facts_all_scopes(self, session: Session) -> None:
        facts = session.facts()
        assert isinstance(facts, SessionFacts)
        assert len(facts) == 2
        assert len(facts.global_) == 1 and len(facts.local) == 1 and facts.context == []
        g = facts.global_[0]
        assert (g.subject.value, g.relationship, g.object.value) == (
            "France",
            "national language",
            "French",
        )
        assert g.subject.concept == "country"
        assert g.source is FactSource.KNOWLEDGE_MAP
        assert facts.local[0].id == "WA:IF:local1"
        assert facts.local[0].source is FactSource.INJECTION
        assert [f.id for f in facts] == ["WA:KF:global1", "WA:IF:local1"]

    def test_facts_request_without_relationships(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.facts()
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/session/{session.id}"
        assert fake.last.params == {"filter": "facts"}

    def test_facts_relationships_joined_with_commas(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.facts(relationships=["lives in", "speaks"])
        assert fake.last.params == {"filter": "facts", "relationships": "lives in,speaks"}

    def test_facts_relationships_accepts_any_iterable(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.facts(relationships=(r for r in ("a", "b")))
        assert fake.last.params["relationships"] == "a,b"

    def test_facts_relationships_accepts_single_string(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        facts = session.facts(relationships="speaks")
        assert fake.last.params == {"filter": "facts", "relationships": "speaks"}
        assert len(facts) == 0  # nothing in the fake uses "speaks"

    def test_facts_single_string_is_not_split_into_characters(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        facts = session.facts(relationships="lives in")
        assert fake.last.params["relationships"] == "lives in"
        assert [f.relationship for f in facts] == ["lives in"]

    def test_facts_filtered_result(self, session: Session) -> None:
        facts = session.facts(relationships=["lives in"])
        assert facts.global_ == []
        assert [f.relationship for f in facts.local] == ["lives in"]
        assert len(facts) == 1


# =========================================================================== natural language


class TestNaturalLanguage:
    def test_interact_sends_body_and_headers(self, session: Session, fake: FakeRainbird) -> None:
        session.interact("Ben holds Gold, can he claim?")
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/interact"
        assert fake.last.json == {
            "sessionID": session.id,
            "userPrompt": "Ben holds Gold, can he claim?",
        }
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["accept"] == "application/json"
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_interact_question(self, session: Session, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_QUESTION)
        r = session.interact("Can Ben claim?")
        assert isinstance(r, InteractResponse)
        assert r.response_type == "question" and r.is_question
        assert not r.is_result and not r.is_error
        assert r.question is not None
        assert r.question.prompt == "Which policy does Ben hold?"
        assert r.question.options == ["Gold", "Silver"]
        assert r.prompt == "Which policy does Ben hold?"
        assert r.query == {"subject": "Ben", "relationship": "can claim", "object": None}
        assert r.results == []
        assert r.metadata is not None and r.metadata.total_tokens == 3169
        assert r.metadata.endpoint == "interact"

    def test_interact_result(self, session: Session, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_RESULT)
        r = session.interact("Ben holds Gold and he likes tea. Can he claim?")
        assert r.is_result and r.response_type == "result"
        assert len(r.results) == 1
        item = r.results[0]
        assert (item.subject, item.relationship, item.object) == ("Ben", "can claim", "repair")
        assert item.certainty == 100
        assert (
            item.fact_id == "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea"
        )
        assert r.injected_facts[0]["object"] == "Gold"
        assert r.invalid_facts == []
        assert r.unmatched == ["and he likes tea"]
        assert r.error is None
        assert r.raise_for_error() is r

    def test_interact_error_is_returned_not_raised(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.append(INTERACT_ERROR)
        r = session.interact("hello")
        assert r.is_error and r.response_type == "error"
        assert r.error is not None
        assert r.error.code == 1001
        assert r.error.message == "Could not detect a query in the prompt"
        assert r.error.status_code == "400"
        assert r.prompt == "I couldn't work out what you want to know. Could you rephrase?"
        assert r.questions == [] and r.results == []

    def test_interact_error_raise_for_error(self, session: Session, fake: FakeRainbird) -> None:
        fake.interact_bodies.append(INTERACT_ERROR)
        with pytest.raises(NaturalLanguageError) as info:
            session.interact("hello").raise_for_error()
        assert info.value.code == 1001
        assert info.value.suggested_chat_response.startswith("I couldn't work out")
        assert info.value.body == INTERACT_ERROR

    def test_interact_bodies_are_consumed_in_order(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.interact_bodies.extend([INTERACT_QUESTION, INTERACT_ERROR, INTERACT_RESULT])
        kinds = [session.interact(f"prompt {i}").response_type for i in range(3)]
        assert kinds == ["question", "error", "result"]

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_interact_empty_prompt_raises(
        self, session: Session, fake: FakeRainbird, bad: str
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="user_prompt"):
            session.interact(bad)
        assert fake.calls("/nl") == []

    def test_explain_ok(self, session: Session, fake: FakeRainbird) -> None:
        r = session.explain(ROOT_FACT_ID)
        assert isinstance(r, ExplainResponse)
        assert r.explanation == EXPLAIN_OK["explanation"]
        assert str(r) == EXPLAIN_OK["explanation"]
        assert r.error is None
        assert r.metadata is not None
        assert r.metadata.total_tokens == 900
        assert r.metadata.language_engine_response_ms == 5538
        assert r.metadata.endpoint == "explain"
        assert fake.last.method == "POST"
        assert fake.last.path == "/nl/explain"
        assert fake.last.json == {"language": "en", "sessionID": session.id, "factID": ROOT_FACT_ID}

    def test_explain_headers(self, session: Session, fake: FakeRainbird) -> None:
        session.explain(ROOT_FACT_ID)
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["accept"] == "application/json"
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_explain_language_and_instructions(self, session: Session, fake: FakeRainbird) -> None:
        session.explain(ROOT_FACT_ID, language="es", instructions="Keep it short")
        assert fake.last.json == {
            "language": "es",
            "sessionID": session.id,
            "factID": ROOT_FACT_ID,
            "instructions": "Keep it short",
        }

    def test_explain_error_raises(self, session: Session) -> None:
        with pytest.raises(NaturalLanguageError) as info:
            session.explain("bogus-fact")
        assert info.value.code == 2001
        assert info.value.message == "Fact not found"
        assert info.value.status_code == "404"
        assert info.value.body == {
            "error": {"code": 2001, "message": "Fact not found", "statusCode": "404"}
        }

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_explain_empty_fact_id_raises(
        self, session: Session, fake: FakeRainbird, bad: str
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="fact_id"):
            session.explain(bad)
        assert fake.calls("/nl") == []


# =========================================================================== version


class TestVersion:
    def test_text_plain(self, client: Rainbird, fake: FakeRainbird) -> None:
        assert client.version() == "4.89.0"
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"

    def test_strips_whitespace(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.version_text = "  4.90.1\n"
        assert client.version() == "4.90.1"

    def test_json_object_variant(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.version_text = '{"version": "5.0.1"}'
        assert client.version() == "5.0.1"

    def test_json_string_variant(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.version_text = '"5.0.2"'
        assert client.version() == "5.0.2"

    def test_accept_header_allows_text_and_json(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.version()
        accept = fake.last.headers["accept"]
        assert "text/plain" in accept and "application/json" in accept

    def test_sends_api_key_when_called_on_a_client(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        client.version()
        assert fake.last.headers["x-api-key"] == API_KEY


class TestEngineVersion:
    """``rainbird_ai.engine_version()``: ``GET /version`` without constructing a keyed client."""

    @pytest.fixture
    def http(self, fake: FakeRainbird):
        client = httpx.Client(transport=httpx.MockTransport(fake.handle))
        yield client
        client.close()

    def test_exported_at_package_and_module_level(self) -> None:
        assert rainbird_ai.engine_version is engine_version is client_module.engine_version
        assert "engine_version" in rainbird_ai.__all__
        assert "UNSET" in rainbird_ai.__all__
        assert client_module.__all__ == ["Answerer", "Rainbird", "Session", "engine_version"]
        assert not hasattr(rainbird_ai, "NoActiveSessionError")  # removed: it was never raised

    def test_returns_engine_version_without_api_key(
        self, http: httpx.Client, fake: FakeRainbird
    ) -> None:
        assert engine_version(BASE_URL, http_client=http) == "4.89.0"
        assert len(fake.requests) == 1
        assert fake.last.method == "GET"
        assert fake.last.path == "/version"
        assert fake.last.json is None
        assert "x-api-key" not in fake.last.headers
        assert fake.last.headers["accept"] == "text/plain, application/json, */*"
        assert fake.last.headers["user-agent"] == http_mod.USER_AGENT
        assert "x-rainbird-engine" not in fake.last.headers

    def test_does_not_send_an_api_key_from_the_environment(
        self, http: httpx.Client, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAINBIRD_API_KEY", API_KEY)
        assert engine_version(BASE_URL, http_client=http) == "4.89.0"
        assert "x-api-key" not in fake.last.headers

    def test_base_url_may_be_passed_positionally_or_by_keyword(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen))
        assert engine_version(BASE_URL, http_client=http) == "4.89.0"
        assert engine_version(base_url=BASE_URL + "/", http_client=http) == "4.89.0"
        assert [str(r.url) for r in seen] == [f"{BASE_URL}/version"] * 2
        http.close()

    def test_base_url_falls_back_to_env_then_community(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen))
        engine_version(http_client=http)
        assert str(seen[-1].url) == "https://api.rainbird.ai/version"
        monkeypatch.setenv("RAINBIRD_API_URL", BASE_URL)
        engine_version(http_client=http)
        assert str(seen[-1].url) == f"{BASE_URL}/version"
        engine_version("   ", http_client=http)  # blank behaves like "not given"
        assert str(seen[-1].url) == f"{BASE_URL}/version"
        http.close()

    def test_base_url_accepts_environment_enum_and_bare_host(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen))
        engine_version(Environment.ENTERPRISE, http_client=http)
        assert str(seen[-1].url) == "https://enterprise-api.rainbird.ai/version"
        engine_version("acme-api.rainbird.ai/", http_client=http)
        assert str(seen[-1].url) == "https://acme-api.rainbird.ai/version"
        http.close()

    def test_invalid_base_url_raises_before_any_request(
        self, http: httpx.Client, fake: FakeRainbird
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="has no host"):
            engine_version("https://:443", http_client=http)
        assert fake.requests == []

    def test_engine_header_from_argument_and_env(
        self, http: httpx.Client, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine_version(BASE_URL, engine="alt-engine", http_client=http)
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"
        monkeypatch.setenv("RAINBIRD_ENGINE", "env-engine")
        engine_version(BASE_URL, http_client=http)
        assert fake.last.headers["x-rainbird-engine"] == "env-engine"
        engine_version(BASE_URL, engine="alt-engine", http_client=http)
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"  # argument wins

    def test_parses_the_same_body_variants_as_rainbird_version(
        self, http: httpx.Client, fake: FakeRainbird
    ) -> None:
        fake.version_text = "  4.90.1\n"
        assert engine_version(BASE_URL, http_client=http) == "4.90.1"
        fake.version_text = '{"version": "5.0.1"}'
        assert engine_version(BASE_URL, http_client=http) == "5.0.1"
        fake.version_text = '"5.0.2"'
        assert engine_version(BASE_URL, http_client=http) == "5.0.2"

    def test_user_client_is_left_open(self, http: httpx.Client, fake: FakeRainbird) -> None:
        engine_version(BASE_URL, http_client=http)
        assert http.is_closed is False
        assert engine_version(BASE_URL, http_client=http) == "4.89.0"  # still usable
        assert len(fake.requests) == 2

    def test_user_client_timeout_is_used_by_default(self, fake: FakeRainbird) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0))
        engine_version(BASE_URL, http_client=http)
        assert seen[-1].extensions["timeout"] == httpx.Timeout(9.0).as_dict()
        http.close()

    def test_explicit_timeout_is_applied_per_request_with_user_client(
        self, fake: FakeRainbird
    ) -> None:
        seen: list[httpx.Request] = []
        http = httpx.Client(transport=recording_transport(fake, seen), timeout=httpx.Timeout(9.0))
        engine_version(BASE_URL, timeout=2.5, http_client=http)
        assert seen[-1].extensions["timeout"] == httpx.Timeout(2.5).as_dict()
        engine_version(BASE_URL, timeout=None, http_client=http)
        assert seen[-1].extensions["timeout"] == httpx.Timeout(None).as_dict()
        assert http.timeout == httpx.Timeout(9.0)
        http.close()

    def test_owned_client_is_created_with_defaults_and_closed(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[httpx.Client] = []
        real_client = httpx.Client

        def factory(**kwargs: Any) -> httpx.Client:
            client = real_client(transport=httpx.MockTransport(fake.handle), **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", factory)
        assert engine_version(BASE_URL) == "4.89.0"
        (client,) = created
        assert client.timeout == httpx.Timeout(30.0)
        assert client.follow_redirects is False
        assert client.is_closed
        assert "x-api-key" not in fake.last.headers

    def test_owned_client_uses_explicit_timeout_and_is_closed_after_an_error(
        self, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[httpx.Client] = []
        real_client = httpx.Client

        def factory(**kwargs: Any) -> httpx.Client:
            client = real_client(transport=httpx.MockTransport(fake.handle), **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", factory)
        fake.fail_next(404, {"err": ["gone"]})
        with pytest.raises(NotFoundError):
            engine_version(BASE_URL, timeout=7)
        (client,) = created
        assert client.timeout == httpx.Timeout(7)
        assert client.is_closed

    def test_api_errors_carry_method_and_url(self, http: httpx.Client, fake: FakeRainbird) -> None:
        fake.fail_next(404, {"err": ["gone"]})
        with pytest.raises(NotFoundError) as info:
            engine_version(BASE_URL, http_client=http)
        assert info.value.errors == ["gone"]
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/version"
        assert str(info.value) == f"HTTP 404 calling GET {BASE_URL}/version: gone"

    def test_transport_errors_are_retried_then_mapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(http_mod.time, "sleep", sleeps.append)
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("refused")

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(RainbirdConnectionError, match="could not reach GET") as info:
            engine_version(BASE_URL, http_client=http)
        http.close()
        assert attempts == 3  # GET /version follows the default policy: two retries
        assert len(sleeps) == 2
        assert info.value.url == f"{BASE_URL}/version"


# =========================================================================== knowledge maps (undocumented)


class TestKnowledgeMaps:
    def test_get_map_source(self, client: Rainbird) -> None:
        source = client.get_map_source(KM_ID)
        assert isinstance(source, KnowledgeMapSource)
        assert source.km_id == KM_ID
        assert source.rblang == RBLANG
        assert '<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">' in source.rblang
        assert [c["name"] for c in source.concepts] == ["Person", "Language"]
        assert source.relationships == [
            {"name": "speaks", "subject": "Person", "object": "Language"}
        ]

    def test_get_map_source_request(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.get_map_source(KM_ID)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/analysis/file/{KM_ID}"
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.json is None

    def test_get_map_source_not_found(self, client: Rainbird) -> None:
        with pytest.raises(NotFoundError) as info:
            client.get_map_source("00000000-0000-0000-0000-000000000000")
        assert info.value.status_code == 404
        assert info.value.errors == ["Knowledge map not found"]

    def test_get_map_source_error_body_on_200(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.map_source_body = {"error": "Map is private"}
        with pytest.raises(UnexpectedResponseError, match="Map is private"):
            client.get_map_source(KM_ID)

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_get_map_source_empty_id_raises(
        self, client: Rainbird, fake: FakeRainbird, bad: str
    ) -> None:
        with pytest.raises(RainbirdConfigurationError, match="km_id"):
            client.get_map_source(bad)
        assert fake.requests == []

    def test_create_map_success(self, client: Rainbird) -> None:
        created = client.create_map("New map", RBLANG)
        assert created.km_id == "99999999-8888-7777-6666-555555555555"
        assert created.id == 4242
        assert created.name == "New map"
        assert created.raw["kmID"] == created.km_id

    def test_create_map_request(self, client: Rainbird, fake: FakeRainbird) -> None:
        client.create_map("New map", RBLANG)
        assert fake.last.method == "POST"
        assert fake.last.path == "/maps"
        assert fake.last.json == {"rblang": RBLANG, "name": "New map"}
        assert fake.last.headers["x-api-key"] == API_KEY
        assert fake.last.headers["version"] == "v1"
        assert fake.last.headers["accept"] == "application/json"

    def test_create_map_name_falls_back_to_argument(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.map_create_body = {"kmID": "abc", "id": 1}
        assert client.create_map("Given name", RBLANG).name == "Given name"

    def test_create_map_http_200_error_body_raises(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        bad_rblang = '<?xml version="1.0"?><foo/>'
        with pytest.raises(KnowledgeMapError) as info:
            client.create_map("Broken", bad_rblang)
        assert info.value.message == "RBLang parse error: missing rbl:kb root"
        assert info.value.errors == ["RBLang parse error: missing rbl:kb root"]
        assert info.value.body == {"error": "RBLang parse error: missing rbl:kb root"}
        assert not isinstance(info.value, RainbirdAPIError)
        assert fake.last.path == "/maps"

    def test_create_map_not_found(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(404, {"err": ["Not found"]})
        with pytest.raises(NotFoundError):
            client.create_map("x", RBLANG)

    def test_create_map_empty_name_raises(self, client: Rainbird, fake: FakeRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="name"):
            client.create_map("", RBLANG)
        assert fake.requests == []

    def test_create_map_empty_rblang_raises(self, client: Rainbird, fake: FakeRainbird) -> None:
        with pytest.raises(RainbirdConfigurationError, match="rblang"):
            client.create_map("x", "  ")
        assert fake.requests == []

    def test_create_map_response_without_id_is_unexpected(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.map_create_body = {"ok": True}
        with pytest.raises(UnexpectedResponseError):
            client.create_map("x", RBLANG)


# =========================================================================== error mapping


class TestErrorMapping:
    def test_400_err_list(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        session = noretry.start(KM_ID)
        fake.fail_next(400, {"err": ["relationship is required"]})
        with pytest.raises(BadRequestError) as info:
            session.query("speaks", subject="Tom")
        err = info.value
        assert err.status_code == 400
        assert err.message == "relationship is required"
        assert err.errors == ["relationship is required"]
        assert err.body == {"err": ["relationship is required"]}
        assert err.method == "POST"
        assert err.url == f"{BASE_URL}/{session.id}/query"
        assert err.retry_after is None
        assert (
            str(err)
            == f"HTTP 400 calling POST {BASE_URL}/{session.id}/query: relationship is required"
        )

    def test_400_multiple_errors_joined(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(400, {"err": ["first", "second"]})
        with pytest.raises(BadRequestError) as info:
            noretry.start(KM_ID)
        assert info.value.message == "first; second"
        assert info.value.errors == ["first", "second"]

    def test_401(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(401, {"err": ["Your API key is invalid"]})
        with pytest.raises(AuthenticationError) as info:
            noretry.start(KM_ID)
        assert info.value.status_code == 401
        assert info.value.message == "Your API key is invalid"

    def test_403(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(403, {"err": ["Forbidden"]})
        with pytest.raises(PermissionDeniedError) as info:
            noretry.start(KM_ID)
        assert info.value.status_code == 403

    def test_404(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(404, {"err": ["Session not found"]})
        with pytest.raises(NotFoundError) as info:
            noretry.resume("gone").query("speaks")
        assert info.value.status_code == 404
        assert info.value.message == "Session not found"

    def test_404_for_unknown_session_from_fake(self, noretry: Rainbird) -> None:
        with pytest.raises(NotFoundError):
            noretry.resume("not-a-session").inject_fact("a", "b", "c")

    def test_429_with_retry_after_header(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(429, {"message": "slow down"}, headers={"Retry-After": "7"})
        with pytest.raises(RateLimitError) as info:
            noretry.start(KM_ID)
        assert info.value.status_code == 429
        assert info.value.message == "slow down"
        assert info.value.retry_after == 7.0

    def test_429_without_retry_after(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(429)
        with pytest.raises(RateLimitError) as info:
            noretry.start(KM_ID)
        assert info.value.retry_after is None
        assert info.value.message == "Too many requests"

    def test_500_text_body(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(500, "Internal Server Error")
        with pytest.raises(ServerError) as info:
            noretry.start(KM_ID)
        assert info.value.status_code == 500
        assert info.value.message == "Internal Server Error"
        assert info.value.body == "Internal Server Error"

    def test_503_empty_body_default_message(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(503)
        with pytest.raises(ServerError) as info:
            noretry.start(KM_ID)
        assert info.value.message == "Rainbird server error"
        assert info.value.body is None

    def test_error_key_shape(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(400, {"error": "bad thing"})
        with pytest.raises(BadRequestError) as info:
            noretry.start(KM_ID)
        assert info.value.message == "bad thing"

    def test_nested_error_message_shape(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(403, {"error": {"message": "nope", "code": 7}})
        with pytest.raises(PermissionDeniedError) as info:
            noretry.start(KM_ID)
        assert info.value.message == "nope"

    def test_message_key_shape(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(404, {"message": "missing"})
        with pytest.raises(NotFoundError) as info:
            noretry.start(KM_ID)
        assert info.value.message == "missing"

    def test_unknown_4xx_is_generic_api_error(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(418, {"err": ["teapot"]})
        with pytest.raises(RainbirdAPIError) as info:
            noretry.start(KM_ID)
        assert type(info.value) is RainbirdAPIError
        assert info.value.status_code == 418

    def test_api_errors_are_rainbird_errors(self, noretry: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(400, {"err": ["x"]})
        with pytest.raises(RainbirdError):
            noretry.start(KM_ID)

    def test_default_client_does_not_retry_4xx(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.fail_next(400, {"err": ["x"]})
        with pytest.raises(BadRequestError):
            client.start(KM_ID)
        assert len(fake.requests) == 1

    def test_default_client_retries_429_honouring_retry_after(
        self, client: Rainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(http_mod.time, "sleep", slept.append)
        fake.fail_next(429, headers={"Retry-After": "3"})
        session = client.start(KM_ID)
        assert session.id in fake.sessions
        assert len(fake.calls("/start")) == 2
        assert slept == [3.0]

    def test_default_client_gives_up_after_max_retries(
        self, client: Rainbird, fake: FakeRainbird, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)
        for _ in range(3):
            fake.fail_next(503)
        with pytest.raises(ServerError):
            client.start(KM_ID)
        assert len(fake.calls("/start")) == 3

    def test_post_decision_is_not_retried_on_500(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        fake.fail_next(500, {"err": ["boom"]})
        with pytest.raises(ServerError):
            session.query("speaks", subject="Tom")
        assert len(fake.calls("/query")) == 1


# =========================================================================== headers


class TestHeaders:
    def test_engine_header_sent_when_configured(self, make_client, fake: FakeRainbird) -> None:
        rb = make_client(engine="alt-engine")
        session = rb.start(KM_ID)
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"
        session.query("speaks", subject="Tom")
        assert fake.last.headers["x-rainbird-engine"] == "alt-engine"

    def test_engine_header_absent_by_default(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="Tom")
        assert all("x-rainbird-engine" not in r.headers for r in fake.requests)

    def test_engine_header_from_env(self, make_client, fake: FakeRainbird, monkeypatch) -> None:
        monkeypatch.setenv("RAINBIRD_ENGINE", "env-engine")
        make_client().version()
        assert fake.last.headers["x-rainbird-engine"] == "env-engine"

    def test_default_headers_sent_with_every_request(self, make_client, fake: FakeRainbird) -> None:
        rb = make_client(default_headers={"X-Trace-Id": "trace-1"})
        rb.start(KM_ID).query("speaks", subject="Tom")
        assert all(r.headers["x-trace-id"] == "trace-1" for r in fake.requests)

    def test_default_headers_can_override_user_agent(self, make_client, fake: FakeRainbird) -> None:
        make_client(default_headers={"User-Agent": "custom/1.0"}).version()
        assert fake.last.headers["user-agent"] == "custom/1.0"

    def test_api_key_on_session_endpoints(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="Tom")
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_nl_endpoints_send_version_v1_and_json_accept(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.interact("hi")
        session.explain(ROOT_FACT_ID)
        nl_calls = fake.calls("/nl/")
        assert len(nl_calls) == 2
        assert all(c.headers["version"] == "v1" for c in nl_calls)
        assert all(c.headers["accept"] == "application/json" for c in nl_calls)

    def test_decision_endpoints_do_not_send_nl_version(
        self, session: Session, fake: FakeRainbird
    ) -> None:
        session.query("speaks", subject="Tom")
        assert "version" not in fake.last.headers

    def test_json_content_type_on_posts(self, session: Session, fake: FakeRainbird) -> None:
        session.query("speaks", subject="Tom")
        assert fake.last.headers["content-type"] == "application/json"
