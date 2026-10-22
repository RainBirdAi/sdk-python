"""Exception hierarchy for the Rainbird SDK.

Every exception raised by this package derives from :class:`RainbirdError`, so a
single ``except RainbirdError`` catches everything the SDK can raise on purpose.

Hierarchy::

    RainbirdError
    ├── RainbirdConfigurationError      missing API key / base URL, bad arguments
    ├── NoPendingQuestionError          answer() with nothing to answer
    ├── InvalidAnswerError              answer does not fit the pending question
    ├── InvalidFactError                fact payload is malformed
    ├── KnowledgeMapError               /maps accepted the request but rejected the RBLang
    ├── NaturalLanguageError            /nl/* endpoint returned an error object
    ├── RainbirdTransportError
    │   ├── RainbirdConnectionError     DNS / TCP / TLS failure
    │   └── RainbirdTimeoutError        request timed out
    ├── UnexpectedResponseError         2xx body that is not the documented shape
    └── RainbirdAPIError                non-2xx HTTP response
        ├── BadRequestError             400
        ├── AuthenticationError         401
        ├── PermissionDeniedError       403
        ├── NotFoundError               404
        ├── RateLimitError              429
        └── ServerError                 5xx
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AuthenticationError",
    "BadRequestError",
    "InvalidAnswerError",
    "InvalidFactError",
    "KnowledgeMapError",
    "NaturalLanguageError",
    "NoPendingQuestionError",
    "NotFoundError",
    "PermissionDeniedError",
    "RainbirdAPIError",
    "RainbirdConfigurationError",
    "RainbirdConnectionError",
    "RainbirdError",
    "RainbirdTimeoutError",
    "RainbirdTransportError",
    "RateLimitError",
    "ServerError",
    "UnexpectedResponseError",
]


class RainbirdError(Exception):
    """Base class for every exception raised by ``rainbird_ai``."""


class RainbirdConfigurationError(RainbirdError, ValueError):
    """The client was configured incorrectly (for example no API key was supplied)."""


class NoPendingQuestionError(RainbirdError):
    """``answer()``/``skip()`` was called but the engine has not asked a question."""


class InvalidAnswerError(RainbirdError, ValueError):
    """An answer cannot be built for, or does not satisfy, the question being answered."""


class InvalidFactError(RainbirdError, ValueError):
    """A fact or answer payload is missing required fields or has invalid values."""


class RainbirdTransportError(RainbirdError):
    """Base class for network-level failures (no HTTP response was received)."""

    def __init__(self, message: str, *, method: str | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.method = method
        self.url = url


class RainbirdConnectionError(RainbirdTransportError):
    """The API host could not be reached (DNS, TCP or TLS failure)."""


class RainbirdTimeoutError(RainbirdTransportError):
    """The request did not complete within the configured timeout."""


class RainbirdAPIError(RainbirdError):
    """The API returned a non-2xx HTTP status.

    Attributes:
        status_code: HTTP status code.
        message: Best human-readable message extracted from the response body.
        errors: All error strings found in the body (Rainbird returns ``{"err": [...]}``
            for 400s; other shapes such as ``{"error": ...}`` and ``{"message": ...}``
            are also recognised). Empty if the body had none.
        body: Parsed JSON body if the response was JSON, otherwise the raw text.
        method: HTTP method of the failed request.
        url: Full URL of the failed request.
        retry_after: Seconds suggested by a ``Retry-After`` header, when present.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        errors: list[str] | None = None,
        body: Any = None,
        method: str | None = None,
        url: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.errors = list(errors or [])
        self.body = body
        self.method = method
        self.url = url
        self.retry_after = retry_after

    def __str__(self) -> str:
        where = f" calling {self.method} {self.url}" if self.method and self.url else ""
        return f"HTTP {self.status_code}{where}: {self.message}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, message={self.message!r}, "
            f"errors={self.errors!r})"
        )


class BadRequestError(RainbirdAPIError):
    """400: the request is missing a required parameter or a value is invalid."""


class AuthenticationError(RainbirdAPIError):
    """401: the API key is missing or invalid."""


class PermissionDeniedError(RainbirdAPIError):
    """403: the API key does not permit access to this resource."""


class NotFoundError(RainbirdAPIError):
    """404: the session, knowledge map or fact could not be found."""


class RateLimitError(RainbirdAPIError):
    """429: too many requests. ``retry_after`` holds the server's hint when provided."""


class ServerError(RainbirdAPIError):
    """5xx: the Rainbird platform failed to process the request."""


class KnowledgeMapError(RainbirdError):
    """``POST /maps`` accepted the request (HTTP 200) but rejected the RBLang.

    Attributes:
        errors: Parser/validation messages returned by Rainbird.
        body: The full response body for inspection.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = list(errors or [])
        self.body = body


class NaturalLanguageError(RainbirdError):
    """A ``/nl/interact`` or ``/nl/explain`` call returned an error object.

    Attributes:
        code: Rainbird's internal error code.
        message: Description of the error.
        suggested_chat_response: A user-friendly message suitable for chat UIs.
        status_code: HTTP-style status code reported *inside* the body (string per spec).
        body: The full response body.
    """

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        suggested_chat_response: str | None = None,
        status_code: str | int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.suggested_chat_response = suggested_chat_response
        self.status_code = status_code
        self.body = body


class UnexpectedResponseError(RainbirdError):
    """The API returned 2xx but the body was not in the documented shape.

    Attributes:
        body: The parsed body (or raw text) so callers can inspect it.
        status_code: HTTP status of the response.
    """

    def __init__(self, message: str, *, body: Any = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body
        self.status_code = status_code
