"""Public API surface of ``rainbird_ai``.

Covers what a user sees before making a single request: the names exported by
the package, versioning, the exception hierarchy, packaging metadata (typed
marker, console script, dependencies), docstrings and type annotations on the
public classes, and a handful of wire-level smoke tests proving the exported
types are the ones the clients actually produce and raise.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.resources
import inspect
import re
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

import rainbird_ai
from rainbird_ai import (
    DEFAULT_BASE_URL,
    NO_RETRY,
    SKIP,
    UNSET,
    AsyncRainbird,
    AsyncSession,
    AuthenticationError,
    BadRequestError,
    Environment,
    InvalidAnswerError,
    InvalidFactError,
    KnowledgeMapError,
    NaturalLanguageError,
    NoPendingQuestionError,
    NotFoundError,
    PermissionDeniedError,
    Question,
    QuestionResponse,
    Rainbird,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTimeoutError,
    RainbirdTransportError,
    RateLimitError,
    ResultItem,
    ResultResponse,
    RetryPolicy,
    ServerError,
    Session,
    UnexpectedResponseError,
    __version__,
    _answers,
    _http,
    _version,
    aio,
    async_engine_version,
    build_answers,
    cli,
    client,
    engine_version,
    errors,
    models,
    studio_url_for,
)
from tests.conftest import (
    API_KEY,
    BASE_URL,
    KM_ID,
    QUESTION_FIRST_FORM,
    QUESTION_SECOND_SUBJECT,
    FakeRainbird,
)

# --------------------------------------------------------------------------- helpers

PUBLIC_MODULES = (models, errors, client, aio)
CLIENT_CLASSES = (Rainbird, Session, AsyncRainbird, AsyncSession)
SYNC_CLIENT_CLASSES = (Rainbird, Session)
ASYNC_CLIENT_CLASSES = (AsyncRainbird, AsyncSession)

API_ERRORS: dict[int, type[RainbirdAPIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    500: ServerError,
}
VALUE_ERRORS = (InvalidAnswerError, InvalidFactError, RainbirdConfigurationError)
TRANSPORT_ERRORS = (RainbirdConnectionError, RainbirdTimeoutError)
OTHER_ERRORS = (
    NoPendingQuestionError,
    KnowledgeMapError,
    NaturalLanguageError,
    UnexpectedResponseError,
)

# Canonical PEP 440 pattern (Appendix B of the specification).
_PEP440 = re.compile(
    r"""
    ^\s*v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?
        (?P<release>[0-9]+(?:\.[0-9]+)*)
        (?P<pre>[-_\.]?(?P<pre_l>a|b|c|rc|alpha|beta|pre|preview)[-_\.]?(?P<pre_n>[0-9]+)?)?
        (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_\.]?(?P<post_l>post|rev|r)[-_\.]?(?P<post_n2>[0-9]+)?))?
        (?P<dev>[-_\.]?(?P<dev_l>dev)[-_\.]?(?P<dev_n>[0-9]+)?)?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)
# Normalised form: no leading v, "."-separated release, a/b/rc, .postN, .devN.
_PEP440_CANONICAL = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
)


def _unwrap(member: Any) -> Any:
    """Return the plain function behind a property / classmethod / staticmethod."""
    if isinstance(member, property):
        return member.fget
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    return member


def _short(module: types.ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


def _public_methods(cls: type) -> dict[str, Callable[..., Any]]:
    """Public (non-underscore) methods declared directly on ``cls`` (properties excluded)."""
    return {
        name: _unwrap(member)
        for name, member in vars(cls).items()
        if not name.startswith("_")
        and not isinstance(member, property)
        and inspect.isfunction(_unwrap(member))
    }


def _public_properties(cls: type) -> dict[str, property]:
    return {
        name: member
        for name, member in vars(cls).items()
        if not name.startswith("_") and isinstance(member, property)
    }


def _public_functions(module: types.ModuleType) -> list[tuple[str, Callable[..., Any]]]:
    """Module-level public functions plus public methods of public classes defined in ``module``."""
    found: list[tuple[str, Callable[..., Any]]] = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            found.append((f"{_short(module)}.{name}", obj))
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for attr, member in vars(obj).items():
                if attr.startswith("_"):
                    continue
                func = _unwrap(member)
                if inspect.isfunction(func):
                    found.append((f"{_short(module)}.{name}.{attr}", func))
    return found


def _has_doc(obj: Any) -> bool:
    return bool((getattr(obj, "__doc__", None) or "").strip())


def _method_ids(classes: tuple[type, ...]) -> list[str]:
    return [f"{cls.__name__}.{name}" for cls in classes for name in _public_methods(cls)]


def _property_ids(classes: tuple[type, ...]) -> list[str]:
    return [f"{cls.__name__}.{name}" for cls in classes for name in _public_properties(cls)]


def _class_named(name: str) -> type:
    return {c.__name__: c for c in CLIENT_CLASSES}[name]


def _exported_classes() -> list[type]:
    return [
        obj
        for obj in (getattr(rainbird_ai, n) for n in rainbird_ai.__all__)
        if inspect.isclass(obj)
    ]


def _exported_functions() -> list[Callable[..., Any]]:
    return [
        obj
        for obj in (getattr(rainbird_ai, n) for n in rainbird_ai.__all__)
        if inspect.isfunction(obj)
    ]


ALL_PUBLIC_FUNCTIONS = [item for module in PUBLIC_MODULES for item in _public_functions(module)]

# Third-party packages the SDK may import at runtime (must match the declared dependencies).
RUNTIME_DEPENDENCIES = {"httpx", "anyio"}


@pytest.fixture
def noretry(fake: FakeRainbird) -> Iterator[Rainbird]:
    """A sync client on the fake with retries disabled (so 5xx/429 fail fast)."""
    http = httpx.Client(transport=httpx.MockTransport(fake.handle))
    with Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http, retry=NO_RETRY) as rb:
        yield rb
    http.close()


def _client_raising(make_exc: Callable[[httpx.Request], Exception]) -> Rainbird:
    def handler(request: httpx.Request) -> httpx.Response:
        raise make_exc(request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http, retry=NO_RETRY)


# --------------------------------------------------------------------------- __all__


class TestDunderAll:
    def test_all_is_a_list_of_unique_strings(self) -> None:
        assert isinstance(rainbird_ai.__all__, list)
        assert all(isinstance(name, str) for name in rainbird_ai.__all__)
        assert len(set(rainbird_ai.__all__)) == len(rainbird_ai.__all__)

    @pytest.mark.parametrize("name", rainbird_ai.__all__)
    def test_every_entry_resolves(self, name: str) -> None:
        assert getattr(rainbird_ai, name) is not None

    @pytest.mark.parametrize("name", rainbird_ai.__all__)
    def test_every_entry_is_defined_inside_the_package(self, name: str) -> None:
        obj = getattr(rainbird_ai, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            assert obj.__module__.startswith("rainbird_ai.")

    def test_star_import_exposes_exactly_all(self) -> None:
        namespace: dict[str, Any] = {}
        exec("from rainbird_ai import *", namespace)  # noqa: S102
        exported = {k for k in namespace if not k.startswith("__")}
        assert exported == {n for n in rainbird_ai.__all__ if not n.startswith("__")}

    def test_public_classes_and_functions_in_namespace_are_listed(self) -> None:
        unlisted = [
            name
            for name, obj in vars(rainbird_ai).items()
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and obj.__module__.startswith("rainbird_ai")
            and name not in rainbird_ai.__all__
        ]
        assert unlisted == []

    @pytest.mark.parametrize("module", PUBLIC_MODULES, ids=_short)
    def test_public_classes_of_module_are_reexported(self, module: types.ModuleType) -> None:
        classes = {n for n in module.__all__ if inspect.isclass(getattr(module, n))}
        assert classes <= set(rainbird_ai.__all__)

    @pytest.mark.parametrize("module", PUBLIC_MODULES, ids=_short)
    def test_reexports_are_the_same_objects(self, module: types.ModuleType) -> None:
        for name in module.__all__:
            if name in rainbird_ai.__all__:
                assert getattr(rainbird_ai, name) is getattr(module, name), name

    @pytest.mark.parametrize("module", PUBLIC_MODULES, ids=_short)
    def test_whole_module_all_is_reexported(self, module: types.ModuleType) -> None:
        assert set(module.__all__) <= set(rainbird_ai.__all__)

    @pytest.mark.parametrize("module", PUBLIC_MODULES, ids=_short)
    def test_module_all_entries_resolve(self, module: types.ModuleType) -> None:
        for name in module.__all__:
            assert hasattr(module, name), name

    @pytest.mark.parametrize("module", PUBLIC_MODULES, ids=_short)
    def test_module_all_lists_every_public_class_and_function(
        self, module: types.ModuleType
    ) -> None:
        defined = {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", None) == module.__name__
        }
        assert defined <= set(module.__all__)

    def test_client_and_aio_export_their_classes_aliases_and_version_helpers(self) -> None:
        assert client.__all__ == ["Answerer", "Rainbird", "Session", "engine_version"]
        assert aio.__all__ == [
            "AsyncAnswerer",
            "AsyncRainbird",
            "AsyncSession",
            "async_engine_version",
        ]

    def test_models_all_includes_rule_and_studio_url_for(self) -> None:
        assert {"Rule", "studio_url_for"} <= set(models.__all__)

    def test_no_active_session_error_is_gone(self) -> None:
        # Removed from the package because nothing ever raised it.
        assert "NoActiveSessionError" not in rainbird_ai.__all__
        assert "NoActiveSessionError" not in errors.__all__
        assert not hasattr(errors, "NoActiveSessionError")
        assert not hasattr(rainbird_ai, "NoActiveSessionError")


# --------------------------------------------------------------------------- constants


class TestExportedConstants:
    def test_retry_policy_is_the_http_module_class(self) -> None:
        assert RetryPolicy is _http.RetryPolicy
        assert inspect.isclass(RetryPolicy)

    def test_no_retry_is_a_retry_policy_with_zero_retries(self) -> None:
        assert NO_RETRY is _http.NO_RETRY
        assert isinstance(NO_RETRY, RetryPolicy)
        assert NO_RETRY.max_retries == 0
        assert NO_RETRY.should_retry_status("GET", 503, 0) is False
        assert NO_RETRY.should_retry_transport("GET", 0) is False

    def test_retry_policy_defaults_are_conservative_for_stateful_posts(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.post_retry_statuses == frozenset({429, 503})
        for status in (429, 500, 502, 503, 504):
            assert policy.should_retry_status("GET", status, 0) is True, status
        for status in (429, 503):
            assert policy.should_retry_status("POST", status, 0) is True, status
        # A gateway may report 502/504 after the engine already applied the request.
        for status in (500, 502, 504):
            assert policy.should_retry_status("POST", status, 0) is False, status
        assert policy.should_retry_transport("GET", 0) is True
        assert policy.should_retry_transport("POST", 0) is False

    def test_retry_policy_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            NO_RETRY.max_retries = 3  # type: ignore[misc]

    def test_default_base_url_is_the_community_host(self) -> None:
        assert DEFAULT_BASE_URL is _http.DEFAULT_BASE_URL
        assert DEFAULT_BASE_URL == "https://api.rainbird.ai"
        assert DEFAULT_BASE_URL == Environment.COMMUNITY.value

    def test_skip_is_a_falsy_singleton_sentinel(self) -> None:
        assert SKIP is _answers.SKIP
        assert not SKIP
        assert repr(SKIP) == "SKIP"
        assert type(SKIP)() is SKIP

    def test_unset_is_a_falsy_singleton_sentinel(self) -> None:
        assert UNSET is _http.UNSET
        assert isinstance(UNSET, _http.Unset)
        assert not UNSET
        assert repr(UNSET) == "UNSET"
        assert type(UNSET)() is UNSET

    def test_engine_version_helpers_are_reexported(self) -> None:
        assert rainbird_ai.engine_version is client.engine_version
        assert rainbird_ai.async_engine_version is aio.async_engine_version
        assert inspect.isfunction(engine_version)
        assert inspect.iscoroutinefunction(async_engine_version)

    def test_answer_builders_are_reexported(self) -> None:
        assert rainbird_ai.build_answers is _answers.build_answers
        assert rainbird_ai.build_skip is _answers.build_skip

    def test_environment_hosts_match_the_documentation(self) -> None:
        assert Environment.COMMUNITY.value == "https://api.rainbird.ai"
        assert Environment.ENTERPRISE.value == "https://enterprise-api.rainbird.ai"
        assert Environment.ENTERPRISE_SOUTH_AMERICA.value == "https://sa-enterprise-api.rainbird.ai"
        assert str(Environment.ENTERPRISE) == Environment.ENTERPRISE.value
        assert isinstance(Environment.COMMUNITY, str)

    @pytest.mark.parametrize(
        ("api", "studio"),
        [
            ("https://api.rainbird.ai", "https://app.rainbird.ai"),
            ("https://enterprise-api.rainbird.ai", "https://enterprise.rainbird.ai"),
            ("https://sa-enterprise-api.rainbird.ai", "https://sa-enterprise.rainbird.ai"),
            ("https://acme-api.rainbird.ai", "https://acme.rainbird.ai"),
        ],
    )
    def test_studio_url_for_maps_api_hosts_to_studio_hosts(self, api: str, studio: str) -> None:
        assert studio_url_for(api) == studio

    def test_environment_studio_url_property(self) -> None:
        assert Environment.COMMUNITY.studio_url == "https://app.rainbird.ai"
        assert Environment.ENTERPRISE.studio_url == "https://enterprise.rainbird.ai"


# --------------------------------------------------------------------------- version


class TestVersion:
    def test_dunder_version_is_exported_and_matches_version_module(self) -> None:
        assert "__version__" in rainbird_ai.__all__
        assert rainbird_ai.__version__ == _version.__version__
        assert __version__ == _version.__version__

    def test_version_is_a_non_empty_string(self) -> None:
        assert isinstance(__version__, str)
        assert __version__.strip() == __version__ != ""

    def test_version_is_pep440(self) -> None:
        assert _PEP440.match(__version__), __version__

    def test_version_is_canonical_pep440(self) -> None:
        assert _PEP440_CANONICAL.match(__version__), __version__

    def test_version_matches_installed_distribution(self) -> None:
        assert importlib.metadata.version("rainbird-ai") == __version__

    def test_version_module_only_defines_version(self) -> None:
        public = [n for n in vars(_version) if not n.startswith("_") or n == "__version__"]
        assert public == ["__version__"]

    def test_user_agent_carries_the_version(self, fake: FakeRainbird, client: Rainbird) -> None:
        client.version()
        assert fake.last.path == "/version"
        assert fake.last.headers["user-agent"] == (
            f"rainbird-ai-python/{__version__} httpx/{httpx.__version__}"
        )

    def test_cli_sdk_version_flag_prints_the_package_version(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as info:
            cli.main(["--sdk-version"])
        assert info.value.code == 0
        assert capsys.readouterr().out.strip() == f"rainbird-ai {__version__}"


# --------------------------------------------------------------------------- errors


class TestErrorHierarchy:
    def test_base_error_is_an_exception(self) -> None:
        assert issubclass(RainbirdError, Exception)
        assert RainbirdError.__bases__ == (Exception,)

    @pytest.mark.parametrize("name", errors.__all__)
    def test_every_error_derives_from_rainbird_error(self, name: str) -> None:
        exc_type = getattr(errors, name)
        assert inspect.isclass(exc_type)
        assert issubclass(exc_type, RainbirdError)
        assert issubclass(exc_type, Exception)

    @pytest.mark.parametrize("name", errors.__all__)
    def test_every_error_is_exported_at_top_level(self, name: str) -> None:
        assert name in rainbird_ai.__all__
        assert getattr(rainbird_ai, name) is getattr(errors, name)

    def test_errors_module_all_is_complete(self) -> None:
        defined = {
            n
            for n, obj in vars(errors).items()
            if inspect.isclass(obj)
            and issubclass(obj, BaseException)
            and obj.__module__ == errors.__name__
        }
        assert defined == set(errors.__all__)

    @pytest.mark.parametrize(("status", "exc_type"), sorted(API_ERRORS.items()))
    def test_http_errors_subclass_api_error(
        self, status: int, exc_type: type[RainbirdAPIError]
    ) -> None:
        assert exc_type.__bases__ == (RainbirdAPIError,)
        assert issubclass(exc_type, RainbirdError)
        assert str(status) in (exc_type.__doc__ or "") or status >= 500

    def test_api_error_subclasses_rainbird_error_directly(self) -> None:
        assert RainbirdAPIError.__bases__ == (RainbirdError,)

    @pytest.mark.parametrize("exc_type", VALUE_ERRORS)
    def test_input_errors_are_also_value_errors(self, exc_type: type[RainbirdError]) -> None:
        assert issubclass(exc_type, RainbirdError)
        assert issubclass(exc_type, ValueError)
        with pytest.raises(ValueError):
            raise exc_type("bad input")

    @pytest.mark.parametrize("exc_type", TRANSPORT_ERRORS)
    def test_transport_errors_subclass_transport_error(
        self, exc_type: type[RainbirdTransportError]
    ) -> None:
        assert exc_type.__bases__ == (RainbirdTransportError,)
        assert issubclass(exc_type, RainbirdError)
        assert not issubclass(exc_type, RainbirdAPIError)

    def test_transport_error_subclasses_rainbird_error_directly(self) -> None:
        assert RainbirdTransportError.__bases__ == (RainbirdError,)

    @pytest.mark.parametrize("exc_type", OTHER_ERRORS)
    def test_non_http_errors_are_not_api_errors(self, exc_type: type[RainbirdError]) -> None:
        assert issubclass(exc_type, RainbirdError)
        assert not issubclass(exc_type, RainbirdAPIError)
        assert not issubclass(exc_type, RainbirdTransportError)

    @pytest.mark.parametrize("exc_type", [*VALUE_ERRORS, *OTHER_ERRORS, *TRANSPORT_ERRORS])
    def test_non_http_errors_take_a_single_message(self, exc_type: type[RainbirdError]) -> None:
        exc = exc_type("boom")
        assert str(exc) == "boom"
        assert exc.args == ("boom",)

    def test_api_errors_are_not_value_errors(self) -> None:
        for exc_type in API_ERRORS.values():
            assert not issubclass(exc_type, ValueError)

    def test_api_error_str_and_attributes(self) -> None:
        exc = BadRequestError(
            "bad",
            status_code=400,
            errors=["bad"],
            body={"err": ["bad"]},
            method="POST",
            url="https://api.rainbird.ai/S/response",
            retry_after=None,
        )
        assert str(exc) == "HTTP 400 calling POST https://api.rainbird.ai/S/response: bad"
        assert repr(exc) == "BadRequestError(status_code=400, message='bad', errors=['bad'])"
        assert (exc.status_code, exc.message, exc.errors, exc.body) == (
            400,
            "bad",
            ["bad"],
            {"err": ["bad"]},
        )
        assert exc.retry_after is None

    def test_api_error_str_without_request_context(self) -> None:
        assert str(ServerError("down", status_code=503)) == "HTTP 503: down"

    def test_transport_error_carries_method_and_url(self) -> None:
        exc = RainbirdTimeoutError("slow", method="GET", url="https://x/version")
        assert (exc.method, exc.url, str(exc)) == ("GET", "https://x/version", "slow")

    def test_single_except_clause_catches_everything(self) -> None:
        for name in errors.__all__:
            exc_type = getattr(errors, name)
            kwargs = {"status_code": 500} if issubclass(exc_type, RainbirdAPIError) else {}
            with pytest.raises(RainbirdError):
                raise exc_type("x", **kwargs)


class TestErrorsRaisedOnTheWire:
    @pytest.mark.parametrize(("status", "exc_type"), sorted(API_ERRORS.items()))
    def test_status_code_maps_to_exported_error_class(
        self, fake: FakeRainbird, noretry: Rainbird, status: int, exc_type: type[RainbirdAPIError]
    ) -> None:
        fake.fail_next(status, {"err": ["nope"]})
        with pytest.raises(exc_type) as info:
            noretry.version()
        assert type(info.value) is exc_type
        assert info.value.status_code == status
        assert info.value.errors == ["nope"]
        assert info.value.message == "nope"
        assert info.value.method == "GET"
        assert info.value.url == f"{BASE_URL}/version"
        assert fake.last.path == "/version"

    @pytest.mark.parametrize("status", [502, 503, 504])
    def test_other_5xx_are_server_errors(
        self, fake: FakeRainbird, noretry: Rainbird, status: int
    ) -> None:
        fake.fail_next(status, "upstream unavailable")
        with pytest.raises(ServerError) as info:
            noretry.version()
        assert info.value.status_code == status
        assert info.value.message == "upstream unavailable"

    def test_unmapped_status_is_a_plain_api_error(
        self, fake: FakeRainbird, noretry: Rainbird
    ) -> None:
        fake.fail_next(418, {"error": "teapot"})
        with pytest.raises(RainbirdAPIError) as info:
            noretry.version()
        assert type(info.value) is RainbirdAPIError
        assert info.value.status_code == 418
        assert info.value.message == "teapot"

    def test_rate_limit_error_exposes_retry_after(
        self, fake: FakeRainbird, noretry: Rainbird
    ) -> None:
        fake.fail_next(429, {"err": ["slow down"]}, headers={"Retry-After": "7"})
        with pytest.raises(RateLimitError) as info:
            noretry.version()
        assert info.value.retry_after == 7.0

    def test_400_body_shape_from_spec_is_parsed(
        self, fake: FakeRainbird, noretry: Rainbird
    ) -> None:
        fake.fail_next(400, {"err": ["Please provide an expected boolean value for unanswered."]})
        with pytest.raises(BadRequestError) as info:
            noretry.resume("S").respond(
                {"subject": "Tom", "relationship": "speaks", "object": "", "unanswered": True}
            )
        assert info.value.errors == ["Please provide an expected boolean value for unanswered."]
        assert fake.last.method == "POST"
        assert fake.last.path == "/S/response"

    def test_connect_error_becomes_connection_error(self) -> None:
        rb = _client_raising(lambda req: httpx.ConnectError("refused", request=req))
        with pytest.raises(RainbirdConnectionError) as info:
            rb.version()
        assert isinstance(info.value, RainbirdTransportError)
        assert isinstance(info.value.__cause__, httpx.ConnectError)
        assert (info.value.method, info.value.url) == ("GET", f"{BASE_URL}/version")

    def test_timeout_becomes_timeout_error(self) -> None:
        rb = _client_raising(lambda req: httpx.ReadTimeout("slow", request=req))
        with pytest.raises(RainbirdTimeoutError) as info:
            rb.version()
        assert isinstance(info.value, RainbirdTransportError)
        assert isinstance(info.value.__cause__, httpx.TimeoutException)

    def test_missing_api_key_is_configuration_error_and_value_error(self) -> None:
        with pytest.raises(RainbirdConfigurationError) as info:
            Rainbird()
        assert isinstance(info.value, ValueError)
        assert "RAINBIRD_API_KEY" in str(info.value)

    def test_empty_facts_raise_invalid_fact_error_before_any_request(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        with pytest.raises(InvalidFactError) as info:
            client.resume("S").inject([])
        assert isinstance(info.value, ValueError)
        assert fake.requests == []

    def test_disallowed_option_raises_invalid_answer_error(self) -> None:
        question = Question.from_dict(QUESTION_SECOND_SUBJECT)
        with pytest.raises(InvalidAnswerError) as info:
            build_answers(question, "Zed")
        assert isinstance(info.value, ValueError)

    def test_answer_without_question_raises_no_pending_question_error(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        with pytest.raises(NoPendingQuestionError):
            client.resume("S").answer("German")
        assert fake.requests == []

    @pytest.mark.anyio
    async def test_async_client_raises_the_same_exported_errors(
        self, fake: FakeRainbird, aclient: AsyncRainbird
    ) -> None:
        fake.fail_next(401, {"err": ["Your API key is invalid"]})
        async with aclient as rb:
            with pytest.raises(AuthenticationError) as info:
                await rb.start(KM_ID)
        assert info.value.status_code == 401
        assert info.value.errors == ["Your API key is invalid"]


# --------------------------------------------------------------------------- exported types on the wire


class TestExportedTypesOnTheWire:
    def test_start_returns_the_exported_session_class(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        session = client.start(KM_ID)
        assert type(session) is Session
        assert session.client is client
        assert fake.last.method == "GET"
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.headers["x-api-key"] == API_KEY

    def test_query_returns_exported_result_types(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        session = client.start(KM_ID)
        response = session.query("speaks", subject="Tom")
        assert type(response) is ResultResponse
        assert response.is_result and not response.is_question
        assert all(type(item) is ResultItem for item in response.results)
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom"}

    def test_question_response_and_question_are_exported_types(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        session = client.start(KM_ID)
        fake.queue_question(QUESTION_FIRST_FORM)
        response = session.query("speaks", subject="Tom")
        assert type(response) is QuestionResponse
        assert type(response.question) is Question
        assert session.pending_question is response.question

    def test_skip_sentinel_sends_unanswered_payload(
        self, fake: FakeRainbird, client: Rainbird
    ) -> None:
        session = client.start(KM_ID)
        fake.queue_question(QUESTION_FIRST_FORM)
        session.query("speaks", subject="Tom")
        session.answer(SKIP)
        assert fake.last.path == f"/{session.id}/response"
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

    def test_environment_enum_is_accepted_as_base_url(self) -> None:
        rb = Rainbird(api_key=API_KEY, base_url=Environment.ENTERPRISE)  # type: ignore[arg-type]
        try:
            assert rb.base_url == "https://enterprise-api.rainbird.ai"
            assert rb.studio_url == "https://enterprise.rainbird.ai"
            assert rb.evidence_url("SID", "WA:RF:1") == (
                "https://enterprise.rainbird.ai/evidence?id=WA:RF:1"
                "&api=https://enterprise-api.rainbird.ai&sid=SID"
            )
        finally:
            rb.close()

    def test_default_base_url_is_used_when_nothing_is_configured(self) -> None:
        rb = Rainbird(api_key=API_KEY)
        try:
            assert rb.base_url == DEFAULT_BASE_URL
        finally:
            rb.close()

    @pytest.mark.anyio
    async def test_async_start_returns_the_exported_async_session_class(
        self, fake: FakeRainbird, aclient: AsyncRainbird
    ) -> None:
        async with aclient as rb:
            session = await rb.start(KM_ID)
        assert type(session) is AsyncSession
        assert type(rb) is AsyncRainbird
        assert fake.last.path == f"/start/{KM_ID}"

    def test_aio_module_and_top_level_export_the_same_async_classes(self) -> None:
        assert rainbird_ai.AsyncRainbird is aio.AsyncRainbird
        assert rainbird_ai.AsyncSession is aio.AsyncSession
        assert rainbird_ai.Rainbird is client.Rainbird
        assert rainbird_ai.Session is client.Session


# --------------------------------------------------------------------------- packaging


class TestPackaging:
    def test_py_typed_marker_is_shipped(self) -> None:
        assert importlib.resources.files("rainbird_ai").joinpath("py.typed").is_file()

    def test_distribution_is_named_rainbird_ai(self) -> None:
        dist = importlib.metadata.distribution("rainbird-ai")
        assert dist.metadata["Name"] == "rainbird-ai"
        assert dist.version == __version__

    def test_requires_python_is_3_10_or_newer(self) -> None:
        dist = importlib.metadata.distribution("rainbird-ai")
        assert dist.metadata["Requires-Python"] == ">=3.10"

    def test_httpx_and_anyio_are_the_only_runtime_dependencies(self) -> None:
        dist = importlib.metadata.distribution("rainbird-ai")
        runtime = [req for req in (dist.requires or []) if "extra ==" not in req.partition(";")[2]]
        names = {re.match(r"^[A-Za-z0-9_.-]+", req).group(0) for req in runtime}  # type: ignore[union-attr]
        assert names == RUNTIME_DEPENDENCIES
        assert any(re.match(r"^anyio\s*>=\s*4\b", req) for req in runtime), runtime

    def test_dev_extras_are_marked_as_extras(self) -> None:
        dist = importlib.metadata.distribution("rainbird-ai")
        extras = [req for req in (dist.requires or []) if "extra ==" in req]
        assert all("extra == 'dev'" in req for req in extras)
        assert any(req.startswith("pytest") for req in extras)

    def test_typed_classifier_is_declared(self) -> None:
        dist = importlib.metadata.distribution("rainbird-ai")
        assert "Typing :: Typed" in dist.metadata.get_all("Classifier", [])

    def test_console_script_entry_point_resolves_to_cli_main(self) -> None:
        eps = list(
            importlib.metadata.entry_points().select(group="console_scripts", name="rainbird")
        )
        assert len(eps) == 1
        assert eps[0].value == "rainbird_ai.cli:main"
        assert eps[0].load() is cli.main

    def test_cli_main_signature(self) -> None:
        sig = inspect.signature(cli.main)
        assert list(sig.parameters) == ["argv"]
        assert sig.parameters["argv"].default is None

    def test_no_third_party_imports_besides_declared_dependencies(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        offenders: set[str] = set()
        for module in (rainbird_ai, models, errors, client, aio, cli, _answers, _http, _version):
            source = inspect.getsource(module)
            for match in re.finditer(r"^(?:from|import)\s+([A-Za-z_][\w]*)", source, re.M):
                top = match.group(1)
                if (
                    top not in stdlib
                    and top not in RUNTIME_DEPENDENCIES
                    and top != "rainbird_ai"
                    and top != "__future__"
                ):
                    offenders.add(top)
        assert offenders == set()

    def test_anyio_is_only_imported_by_the_http_layer(self) -> None:
        # anyio backs the async back-off sleep; nothing else should depend on it.
        pattern = re.compile(r"^(?:from|import)\s+anyio\b", re.M)
        assert pattern.search(inspect.getsource(_http))
        for module in (rainbird_ai, models, errors, client, aio, cli, _answers, _version):
            assert not pattern.search(inspect.getsource(module)), module.__name__


# --------------------------------------------------------------------------- docstrings


class TestDocstrings:
    @pytest.mark.parametrize("module", [rainbird_ai, *PUBLIC_MODULES], ids=lambda m: m.__name__)
    def test_public_modules_have_docstrings(self, module: types.ModuleType) -> None:
        assert _has_doc(module)

    @pytest.mark.parametrize("cls", CLIENT_CLASSES, ids=lambda c: c.__name__)
    def test_client_classes_have_docstrings(self, cls: type) -> None:
        assert _has_doc(cls)

    @pytest.mark.parametrize("cls", _exported_classes(), ids=lambda c: c.__name__)
    def test_every_exported_class_has_a_docstring(self, cls: type) -> None:
        assert _has_doc(cls)

    @pytest.mark.parametrize("func", _exported_functions(), ids=lambda f: f.__name__)
    def test_every_exported_function_has_a_docstring(self, func: Callable[..., Any]) -> None:
        assert _has_doc(func)

    @pytest.mark.parametrize("qualname", _method_ids(CLIENT_CLASSES))
    def test_public_methods_have_docstrings(self, qualname: str) -> None:
        cls_name, name = qualname.split(".")
        assert _has_doc(_public_methods(_class_named(cls_name))[name]), qualname

    @pytest.mark.parametrize("qualname", _property_ids(CLIENT_CLASSES))
    def test_public_properties_have_docstrings(self, qualname: str) -> None:
        cls_name, name = qualname.split(".")
        assert _has_doc(_public_properties(_class_named(cls_name))[name]), qualname

    def test_docstring_checks_cover_the_whole_client_surface(self) -> None:
        methods, properties = set(_method_ids(CLIENT_CLASSES)), set(_property_ids(CLIENT_CLASSES))
        assert {"Rainbird.start", "AsyncRainbird.aclose", "Session.answer", "AsyncSession.run"} <= (
            methods
        )
        assert {
            "Rainbird.base_url",
            "AsyncRainbird.base_url",
            "Session.pending_question",
            "AsyncSession.pending_question",
        } <= properties

    def test_version_docstring_points_at_engine_version_instead_of_claiming_no_key(self) -> None:
        sync_doc = (Rainbird.version.__doc__ or "").lower()
        assert "engine_version" in sync_doc
        assert "no api key needed" not in sync_doc
        async_doc = (AsyncRainbird.version.__doc__ or "").lower()
        assert "rainbird.version" in async_doc or "engine_version" in async_doc
        assert "no api key needed" not in async_doc
        assert "api key" in (engine_version.__doc__ or "").lower()
        assert "api key" in (async_engine_version.__doc__ or "").lower()


# --------------------------------------------------------------------------- sync / async parity


class TestSyncAsyncParity:
    def test_rainbird_and_async_rainbird_expose_the_same_public_names(self) -> None:
        sync_names = set(_public_methods(Rainbird)) | set(_public_properties(Rainbird))
        async_names = set(_public_methods(AsyncRainbird)) | set(_public_properties(AsyncRainbird))
        assert sync_names - async_names == {"close"}
        assert async_names - sync_names == {"aclose"}

    def test_session_and_async_session_expose_the_same_public_names(self) -> None:
        sync_names = set(_public_methods(Session)) | set(_public_properties(Session))
        async_names = set(_public_methods(AsyncSession)) | set(_public_properties(AsyncSession))
        assert sync_names == async_names

    @pytest.mark.parametrize(
        ("sync_cls", "async_cls"), [(Rainbird, AsyncRainbird), (Session, AsyncSession)]
    )
    def test_shared_methods_have_identical_parameters(
        self, sync_cls: type, async_cls: type
    ) -> None:
        sync_methods, async_methods = _public_methods(sync_cls), _public_methods(async_cls)
        for name in sorted(set(sync_methods) & set(async_methods)):
            sync_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(sync_methods[name]).parameters.values()
            ]
            async_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(async_methods[name]).parameters.values()
            ]
            assert sync_params == async_params, name

    def test_engine_version_helpers_have_identical_parameters(self) -> None:
        def params(func: Callable[..., Any]) -> list[tuple[str, Any, Any]]:
            return [
                (p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()
            ]

        assert params(engine_version) == params(async_engine_version)
        assert [p.name for p in inspect.signature(engine_version).parameters.values()] == [
            "base_url",
            "engine",
            "timeout",
            "http_client",
            "retry",
        ]
        assert inspect.signature(engine_version).parameters["retry"].default is None
        assert inspect.signature(engine_version).parameters["timeout"].default is UNSET
        assert inspect.signature(engine_version).return_annotation in ("str", str)
        assert not inspect.iscoroutinefunction(engine_version)
        assert inspect.iscoroutinefunction(async_engine_version)

    @pytest.mark.parametrize(
        ("sync_cls", "async_cls"), [(Rainbird, AsyncRainbird), (Session, AsyncSession)]
    )
    def test_constructors_have_identical_parameters(self, sync_cls: type, async_cls: type) -> None:
        def params(cls: type) -> list[tuple[str, Any, Any]]:
            return [
                (p.name, p.kind, p.default)
                for p in inspect.signature(cls.__init__).parameters.values()
            ]

        assert params(sync_cls) == params(async_cls)

    @pytest.mark.parametrize(
        ("qualname", "kind"),
        [
            ("AsyncRainbird.start", "coroutine"),
            ("AsyncRainbird.version", "coroutine"),
            ("AsyncRainbird.evidence", "coroutine"),
            ("AsyncRainbird.get_map_source", "coroutine"),
            ("AsyncRainbird.create_map", "coroutine"),
            ("AsyncRainbird.aclose", "coroutine"),
            ("AsyncRainbird.resume", "plain"),
            ("AsyncRainbird.evidence_url", "plain"),
            ("AsyncSession.query", "coroutine"),
            ("AsyncSession.answer", "coroutine"),
            ("AsyncSession.run", "coroutine"),
            ("AsyncSession.evidence_tree", "coroutine"),
            ("AsyncSession.iter_evidence_tree", "async generator"),
            ("AsyncSession.evidence_url", "plain"),
        ],
    )
    def test_async_methods_have_the_expected_flavour(self, qualname: str, kind: str) -> None:
        cls_name, name = qualname.split(".")
        func = _public_methods(_class_named(cls_name))[name]
        flavour = (
            "async generator"
            if inspect.isasyncgenfunction(func)
            else "coroutine"
            if inspect.iscoroutinefunction(func)
            else "plain"
        )
        assert flavour == kind

    def test_every_io_method_on_async_session_is_awaitable(self) -> None:
        plain = {
            name
            for name, func in _public_methods(AsyncSession).items()
            if not (inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func))
        }
        assert plain == {"evidence_url"}

    def test_sync_classes_have_no_coroutine_methods(self) -> None:
        for cls in SYNC_CLIENT_CLASSES:
            for name, func in _public_methods(cls).items():
                assert not inspect.iscoroutinefunction(func), f"{cls.__name__}.{name}"
                assert not inspect.isasyncgenfunction(func), f"{cls.__name__}.{name}"

    def test_context_manager_protocols(self) -> None:
        assert callable(Rainbird.__enter__) and callable(Rainbird.__exit__)
        assert callable(AsyncRainbird.__aenter__) and callable(AsyncRainbird.__aexit__)
        assert not hasattr(AsyncRainbird, "__enter__")
        assert not hasattr(Rainbird, "__aenter__")


# --------------------------------------------------------------------------- type annotations


class TestTypeAnnotations:
    @pytest.mark.parametrize(
        "func", [f for _, f in ALL_PUBLIC_FUNCTIONS], ids=[n for n, _ in ALL_PUBLIC_FUNCTIONS]
    )
    def test_public_function_parameters_are_annotated(self, func: Callable[..., Any]) -> None:
        sig = inspect.signature(func)
        missing = [
            p.name
            for p in sig.parameters.values()
            if p.name not in {"self", "cls"}
            and p.kind is not inspect.Parameter.VAR_POSITIONAL
            and p.annotation is inspect.Parameter.empty
        ]
        assert missing == []

    @pytest.mark.parametrize(
        "func", [f for _, f in ALL_PUBLIC_FUNCTIONS], ids=[n for n, _ in ALL_PUBLIC_FUNCTIONS]
    )
    def test_public_function_return_is_annotated(self, func: Callable[..., Any]) -> None:
        assert inspect.signature(func).return_annotation is not inspect.Signature.empty

    def test_collected_public_functions_cover_the_client_surface(self) -> None:
        names = {n for n, _ in ALL_PUBLIC_FUNCTIONS}
        assert {"client.Rainbird.start", "client.Session.answer", "aio.AsyncSession.run"} <= names
        assert {
            "models.Question.from_dict",
            "models.parse_query_response",
            "models.studio_url_for",
        } <= names
        assert len(names) == len(ALL_PUBLIC_FUNCTIONS)

    @pytest.mark.parametrize("cls", CLIENT_CLASSES, ids=lambda c: c.__name__)
    def test_client_constructors_are_fully_annotated(self, cls: type) -> None:
        sig = inspect.signature(cls.__init__)
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            assert param.annotation is not inspect.Parameter.empty, param.name
        assert sig.return_annotation is not inspect.Signature.empty

    @pytest.mark.parametrize("cls", CLIENT_CLASSES, ids=lambda c: c.__name__)
    def test_client_properties_have_return_annotations(self, cls: type) -> None:
        for name, prop in _public_properties(cls).items():
            assert prop.fget is not None, name
            assert inspect.signature(prop.fget).return_annotation is not inspect.Signature.empty, (
                name
            )

    def test_dataclass_models_declare_field_annotations(self) -> None:
        for cls in _exported_classes():
            if dataclasses.is_dataclass(cls):
                for f in dataclasses.fields(cls):
                    assert f.type not in (None, ""), f"{cls.__name__}.{f.name}"
