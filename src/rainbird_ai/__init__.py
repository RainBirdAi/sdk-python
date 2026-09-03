"""Rainbird Python SDK.

Quick start::

    from rainbird_ai import Rainbird

    rb = Rainbird(api_key="...")                 # or set RAINBIRD_API_KEY
    session = rb.start("knowledge-map-id")       # or set RAINBIRD_KM_ID
    response = session.query("speaks", subject="Tom")
    while response.is_question:
        print(response.question.prompt, response.question.options)
        response = session.answer("German")
    print(response.results)

Async: ``from rainbird_ai.aio import AsyncRainbird``. Engine version without a key:
``rainbird_ai.engine_version()`` / ``rainbird_ai.async_engine_version()``.
"""

from __future__ import annotations

from ._answers import SKIP, build_answers, build_skip
from ._http import DEFAULT_BASE_URL, NO_RETRY, UNSET, RetryPolicy
from ._version import __version__
from .aio import AsyncAnswerer, AsyncRainbird, AsyncSession, async_engine_version
from .client import Answerer, Rainbird, Session, engine_version
from .errors import (
    AuthenticationError,
    BadRequestError,
    InvalidAnswerError,
    InvalidFactError,
    KnowledgeMapError,
    NaturalLanguageError,
    NoPendingQuestionError,
    NotFoundError,
    PermissionDeniedError,
    RainbirdAPIError,
    RainbirdConfigurationError,
    RainbirdConnectionError,
    RainbirdError,
    RainbirdTimeoutError,
    RainbirdTransportError,
    RateLimitError,
    ServerError,
    UnexpectedResponseError,
)
from .models import (
    Answer,
    Concept,
    Condition,
    CreateMapResult,
    Environment,
    Evidence,
    ExplainResponse,
    ExpressionCondition,
    Fact,
    FactSource,
    FactValue,
    InjectResponse,
    InteractionEvent,
    InteractResponse,
    KnowledgeMapSource,
    KnowledgeMapVersion,
    NLError,
    NLMetadata,
    QueryResponse,
    Question,
    QuestionForm,
    QuestionResponse,
    ResultItem,
    ResultResponse,
    Rule,
    SessionFact,
    SessionFacts,
    StartResponse,
    parse_query_response,
    studio_url_for,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "NO_RETRY",
    "SKIP",
    "UNSET",
    "Answer",
    "Answerer",
    "AsyncAnswerer",
    "AsyncRainbird",
    "AsyncSession",
    "AuthenticationError",
    "BadRequestError",
    "Concept",
    "Condition",
    "CreateMapResult",
    "Environment",
    "Evidence",
    "ExplainResponse",
    "ExpressionCondition",
    "Fact",
    "FactSource",
    "FactValue",
    "InjectResponse",
    "InteractResponse",
    "InteractionEvent",
    "InvalidAnswerError",
    "InvalidFactError",
    "KnowledgeMapError",
    "KnowledgeMapSource",
    "KnowledgeMapVersion",
    "NLError",
    "NLMetadata",
    "NaturalLanguageError",
    "NoPendingQuestionError",
    "NotFoundError",
    "PermissionDeniedError",
    "QueryResponse",
    "Question",
    "QuestionForm",
    "QuestionResponse",
    "Rainbird",
    "RainbirdAPIError",
    "RainbirdConfigurationError",
    "RainbirdConnectionError",
    "RainbirdError",
    "RainbirdTimeoutError",
    "RainbirdTransportError",
    "RateLimitError",
    "ResultItem",
    "ResultResponse",
    "RetryPolicy",
    "Rule",
    "ServerError",
    "Session",
    "SessionFact",
    "SessionFacts",
    "StartResponse",
    "UnexpectedResponseError",
    "__version__",
    "async_engine_version",
    "build_answers",
    "build_skip",
    "engine_version",
    "parse_query_response",
    "studio_url_for",
]
