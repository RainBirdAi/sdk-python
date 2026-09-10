# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Closes the gaps found by porting the Go and JavaScript SDK test suites to this package
(`tests/parity/`, see its README for the full comparison).

### Added

- Injected facts may carry `metadata` (the API uses `{"source": "system" | "user"}` to record
  where a fact came from): `Fact(metadata=...)`, a `metadata` key in fact dicts, and
  `inject_fact(..., source="user")`. A `metadata` that is not a dict raises `InvalidFactError`.
- `headers=` on `Session.query()`, `respond()`, `undo()` and `inject()` (and the async twins)
  for extra HTTP headers on one request; `Rainbird(default_headers=...)` still applies to all.
- `retry=` on `engine_version()` / `async_engine_version()` (default `RetryPolicy()`; pass
  `NO_RETRY` for a single attempt), matching `Rainbird(retry=...)`.
- `ENGINE_CORE` (`"Core"`) and `ENGINE_BETA` (`"Experimental (Beta)"`), the engine names the
  JavaScript SDK exports for the `x-rainbird-engine` header.
- `create_map(name, rblang, description=None)` (sync and async) sends an optional
  `"description"` in the undocumented `POST /maps` body; `CreateMapResult.description` carries
  it back. The CLI gains `rainbird map create --description`.

### Changed

- `parse_query_response()` raises `UnexpectedResponseError` (with the payload on `.body`)
  instead of a bare `ValueError` for a body that is neither a question nor a result.
- `str()` of a `Question` without a prompt, a `ResultItem` or a `SessionFact` renders a missing
  subject or object as `?` instead of the literal `None`, and no longer leaves double spaces.
- `KnowledgeMapVersion.version_number` returns the API's explicit `versionNumber` when present,
  falling back to parsing `versionStatus` (`"Version3"` -> 3, drafts -> `None`) as before.

### Fixed

- `Evidence.fact.source` is now populated from the evidence envelope (as `fact.id` already
  was), so a `SessionFact` taken from evidence carries the same fields as one from
  `session.facts()`.

## [1.0.0] - 2026-09-03

Complete rewrite of the SDK. The `rainbird_python_sdk` package and its `RainbirdAPIClient`
class are gone; the new package is `rainbird-ai` on PyPI with import name `rainbird_ai`.

### Added

- `Rainbird` client (account level) and `Session` (one reasoning session) replacing the single
  `RainbirdAPIClient`. `Rainbird.start(km_id, use_draft=, version=, context_id=)` returns a
  `Session`; `Rainbird.resume(session_id)` attaches to an existing session without a request.
  API key, host and knowledge-map id fall back to `RAINBIRD_API_KEY`, `RAINBIRD_API_URL` and
  `RAINBIRD_KM_ID`; `RAINBIRD_EVIDENCE_KEY`, `RAINBIRD_INTERACTION_KEY` and `RAINBIRD_ENGINE`
  set the default `x-evidence-key`, `x-interaction-key` and `x-rainbird-engine` headers. A blank
  `base_url` falls back the same way, a bare host gets `https://`, and a non-`http(s)` scheme, a
  scheme without a host, any `?` or `#` (a query string or fragment, even an empty trailing one),
  or an otherwise invalid URL raises `RainbirdConfigurationError`.
- Async client with the same surface: `rainbird_ai.aio.AsyncRainbird` / `AsyncSession`
  (`async with`, `await session.query(...)`, `AsyncSession.run` accepts sync or coroutine
  answerers).
- `httpx` transport (sync and async) replacing `requests`; bring your own `httpx.Client` /
  `httpx.AsyncClient` via `http_client=` (never closed by the SDK; an explicit `timeout` is then
  applied per request instead of being ignored); `default_headers` (merged case-insensitively, so
  `{"x-api-key": ...}` replaces the SDK's `X-API-Key`); `timeout` (30 s by default, with the
  exported `UNSET` sentinel marking "not given"); `Environment` enum for the Community /
  Enterprise / South America hosts; private `<client>-api.rainbird.ai` hosts via `base_url`. The
  SDK's own clients never follow redirects: a 3xx raises
  `RainbirdAPIError("unexpected redirect to ...")` instead of forwarding the API key.
- Typed, frozen dataclass models for every payload: `StartResponse`, `InjectResponse`,
  `Question` (with `form`, `is_first_form`, `asks_for_subject`, `asks_for_object`, `options`,
  `allow_unknown`, `plural`...), `Concept`, `QuestionResponse`, `ResultItem`, `ResultResponse`
  (`best`, `fact_ids`), `Evidence`, `Rule`, `Condition`, `ExpressionCondition`, `SessionFact`,
  `FactValue`, `SessionFacts`, `KnowledgeMapVersion`, `InteractionEvent`, `InteractResponse`,
  `ExplainResponse`, `NLError`, `NLMetadata`, `KnowledgeMapSource`, `CreateMapResult`, plus the
  input models `Fact` and `Answer`. Every model keeps the original JSON on `.raw` and exposes
  `to_dict()`; `parse_query_response()` classifies a decisions body as question or result.
- Question-aware answering: `Session.answer(value, certainty=, question=, strict=)` builds the
  correct payload for First Form (`True`/`False`, `"yes"`/`"no"`, or an enum member with such
  a value), Second Form Object and Second Form Subject questions, sends several answers for
  `plural` questions, validates the value against `concepts` when `canAdd` is false (truth-typed
  Second Form Object answers compare as booleans; an exact string match keeps your value; a
  purely numeric match such as `5.0` for the option `5` is accepted only for a numeric answer, a
  number-typed question or a numeric option and then sends the option's own value; an `int` too
  large for a `float` raises `InvalidAnswerError` rather than `OverflowError`; when every concept
  is `invalidResponse` validation is left to the API), and coerces `date`/`datetime` (to
  `YYYY-MM-DD`), `Decimal` (integral ones to `int`) and enums (`.value`, for yes/no and truth
  answers too); `None`, dicts, lists and non-finite numbers (NaN/infinity) raise
  `InvalidAnswerError`. `Session.skip()` (or
  `answer(SKIP)`) sends `unanswered: true` and refuses when `allowUnknown` is false unless
  `strict=False`. `Session.respond()` still accepts raw `Answer` objects or dicts; both are
  normalised before sending (values coerced, `certainty` validated, `answer` normalised to
  `"yes"`/`"no"`, `unanswered` must be a bool or a yes/no/true/false/1/0 string, otherwise
  `InvalidFactError`). `Session.pending_question`, `pending`, `result` and
  `last_response` track engine state; `build_answers()` / `build_skip()` are exported for
  building payloads without a session.
- `Session.run(relationship, subject=, object=, answerer=, max_questions=100)`: query and keep
  answering through a callback until a `ResultResponse` is reached.
- `Session.inject()` accepts a `Fact`, a dict, a `(subject, relationship, object[, certainty])`
  tuple (any 3/4-item tuple or list of scalar values, enums included) or a list of any of those,
  coerces values like answers and sends batches of 250 (`batch_size=`); `Session.inject_fact()`
  for a single triple.
- Evidence: `Session.evidence(fact_id)`, `Session.evidence_tree(fact_id, max_depth=)` and the
  lazy `iter_evidence_tree()` walk the whole tree breadth-first by following condition
  `factID`s (including facts referenced from expression functions), each node fetched once.
  `Session.evidence_url()` / `Rainbird.evidence_url()` build the Studio Evidence Tree link;
  `studio_url_for()` maps an API host to its Studio host.
- Analysis: `Session.interactions(format="json"|"csv", interaction_key=)` (typed overloads:
  `list[InteractionEvent]` for json, `str` for csv), `Session.knowledge_map()`
  (`/analysis/session?filter=version`) and `Session.facts(relationships=)` (a single
  relationship name or an iterable; `/analysis/session?filter=facts`, grouped into
  `global_`/`context`/`local`). `interactions()` raises `UnexpectedResponseError` (carrying any
  error text found, e.g. a secured log answered 200 `{"err": [...]}`) when a 200 body is neither
  a list, an envelope with a list under `values`/`events`/`interactions`, nor a single event
  object.
- Natural language (beta): `Session.interact(prompt)` returns an `InteractResponse`
  (`response_type`, `questions`, `results`, `error`, `prompt`, `raise_for_error()`; errors are
  returned, not raised) and `Session.explain(fact_id, language=, instructions=)` returns an
  `ExplainResponse` (raises `NaturalLanguageError` when the body only contains an error).
- `Rainbird.version()` returns the engine semver string from `GET /version` on the configured
  host. A JSON body must carry a non-empty string or number under `version`, `engine` or
  `semver` (`{}`, `[]` and `{"version": null}` raise `UnexpectedResponseError`); a plain-text
  body must be non-empty, without whitespace or `<`, and at most 64 characters.
  `rainbird_ai.engine_version(base_url=None, *, engine=, timeout=, http_client=)` and
  `rainbird_ai.async_engine_version(...)` call the same public endpoint without an API key.
- Retries: `RetryPolicy` with exponential backoff and jitter that honours `Retry-After`; by
  default GETs retry on timeouts, network/protocol/proxy errors and 429, 500, 502, 503 and 504
  (`retry_statuses`) while POSTs retry only on 429 and 503 (the decision endpoints are stateful,
  and a gateway may report 502/504 after the engine has already applied the request). `NO_RETRY`
  disables retries. The async client waits with `anyio.sleep`.
- Error hierarchy rooted at `RainbirdError`: `RainbirdAPIError` with `status_code`, `message`,
  `errors`, `body`, `method`, `url`, `retry_after` and subclasses `BadRequestError` (400),
  `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404),
  `RateLimitError` (429), `ServerError` (5xx); `RainbirdConnectionError` /
  `RainbirdTimeoutError`; `UnexpectedResponseError`; `RainbirdConfigurationError`;
  `NoPendingQuestionError`; `InvalidAnswerError`; `InvalidFactError`; `KnowledgeMapError`;
  `NaturalLanguageError`. Error messages are extracted from `{"err": [...]}`, `{"error": ...}`
  and `{"message": ...}` bodies; text bodies are whitespace-collapsed and truncated to 300
  characters for `message`/`errors`, with HTML tags stripped only when the content type is HTML
  or the body starts with `<` (plain text keeps its `<`/`>`; `body` keeps the raw text). A 2xx
  response with a non-JSON or empty body raises `UnexpectedResponseError` naming the request;
  an unusable `base_url` raises `RainbirdConfigurationError` rather than an `httpx` exception.
- `rainbird` command-line tool (`rainbird_ai.cli:main`) with `version` (no API key needed),
  `start`, `inject`, `query` (interactive or scripted `--answers`, `--json`), `respond`, `undo`,
  `evidence` (`--tree` prints the nested evidence tree, including facts reached only through
  expression functions and anything fetched but unreachable; `--url`), `explain`, `interact`,
  `facts`, `session`, `interactions` and `map get` / `map create`. Global options (`--api-key`,
  `--base-url`, `--engine`, `--evidence-key`, `--interaction-key`, `--timeout`, `--json`) may be
  given before or after the subcommand; with `--json` interactive prompts go to stderr so stdout
  is pure JSON. Exit codes: 0 success; 1 API/SDK/file error (including a rejected `--answers`
  value, which is only parsed when its question arrives); 2 usage error (including an invalid
  `--fact` certainty or malformed JSON input, checked before any request is sent); 3 a question
  is still pending and no input is available (`--no-input`, non-interactive stdin or Ctrl-D: the
  question is printed once and the session can be resumed with `--session`); 130 interrupted.
  Typed answers use `value@certainty`; `@` is only a separator when the suffix is numeric, so
  `tom@example.com` is a plain value, while a numeric suffix outside 0-100 and an empty answer
  are rejected. `?@50` still skips (certainty dropped); certainty applies to the whole answer, so
  `a@70, b@80` is rejected in favour of `a, b@70`. Interactive re-prompting happens only for
  `InvalidAnswerError`/`InvalidFactError`/`BadRequestError`; any other error (401, session not
  found, network) exits 1 immediately. `--file` must be a JSON object, a list of fact objects or
  a list of `[subject, relationship, object, certainty?]` lists, otherwise exit 2 (a flat
  `["Tom", "lives in", "Germany"]` is rejected rather than miscounted).
- Undocumented knowledge-map endpoints (not in the public OpenAPI spec, may change):
  `Rainbird.get_map_source(km_id)` (`GET /analysis/file/{kmID}` -> `KnowledgeMapSource` with the
  RBLang) and `Rainbird.create_map(name, rblang)` (`POST /maps` -> `CreateMapResult`), which
  raises `KnowledgeMapError` when Rainbird answers HTTP 200 with a truthy `err`/`errors`/`error`
  value for invalid RBLang (`{"error": false}`, `0`, `""`, `[]`, `{}`, `null` and an
  informational `message` on a success body are not errors).
- `py.typed` marker; the package is fully typed and checked with strict mypy.
- Test suite built on an in-memory fake of the Rainbird API (`tests/conftest.py`, served
  through `httpx.MockTransport`) plus opt-in integration tests (`-m integration`).

### Changed

- Distribution renamed to `rainbird-ai`; import name changed from `rainbird_python_sdk` to
  `rainbird_ai`. Build backend is now hatchling with the version read from
  `src/rainbird_ai/_version.py`.
- Minimum Python version raised from 3.8 to 3.10.
- Runtime dependencies are now `httpx>=0.27,<1.0` and `anyio>=4` (the latter only for async
  retry back-off) instead of `requests`.
- Every method now returns a typed model instead of a raw `dict`. The original JSON is still
  available on `.raw` / `.to_dict()`.
- `query()` takes `relationship` as its first positional argument; `subject` and `object` are
  keyword-only.
- `version()` calls `GET /version` on the configured `base_url` (the old client always called
  the Community host) and returns a string; `engine_version()` does the same without a client
  or API key.
- The interaction log defaults to `format="json"` (was `"csv"`) and returns
  `list[InteractionEvent]`; pass `format="csv"` for the raw CSV text.
- HTTP and validation failures raise `RainbirdError` subclasses instead of
  `requests.HTTPError` / `ValueError`.

### Removed

- The `rainbird_python_sdk` package and `RainbirdAPIClient` class, including `response()`,
  `answer_question()`, `answer_first_form()`, `answer_second_form()`, `interaction_log()`,
  `session_info()` and the `session_id` / `last_question` attributes.
- Dependency on `requests`.

#### Migration from `RainbirdAPIClient`

| `rainbird_python_sdk` (0.x)                                  | `rainbird_ai` (1.0)                                                                 |
|--------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `RainbirdAPIClient(api_host, api_key, km_id)`                | `rb = Rainbird(api_key, base_url=api_host)`; the km id moves to `rb.start(km_id)`   |
| `client.start(use_draft, version)` -> dict                   | `session = rb.start(km_id, use_draft=, version=, context_id=)` -> `Session`         |
| `client.session_id`                                          | `session.id` (reattach later with `rb.resume(session_id)`)                          |
| `client.inject(facts)`                                       | `session.inject(facts)` (Fact / dict / tuple / list; batched by 250)                |
| `client.inject_fact(subject, relationship, object_value, certainty)` | `session.inject_fact(subject, relationship, object, certainty=100)`         |
| `client.query(subject, relationship, object)` -> dict        | `session.query(relationship, subject=, object=)` -> `QuestionResponse` or `ResultResponse` |
| `'question' in result`                                       | `response.is_question` / `response.is_result`                                       |
| `client.last_question`                                       | `session.pending_question` (a `Question`)                                           |
| `client.response(answers)`                                   | `session.respond(answers)` (`Answer` objects or dicts)                              |
| `client.answer(value_or_values, certainty)`                  | `session.answer(value, certainty=100)`; pass a list for plural questions            |
| `client.answer_question(question, value, certainty)`         | `session.answer(value, question=question)` or `build_answers(question, value)`      |
| `client.answer_first_form(s, r, o, "yes"/"no", certainty)`   | `session.answer(True/False)` or `Answer(s, r, o, certainty, answer="yes")`          |
| `client.answer_second_form(s, r, value, certainty)`          | `session.answer(value)` or `Answer(s, r, value, certainty)`                         |
| (not available)                                              | `session.skip()` / `session.answer(SKIP)` when `question.allow_unknown`             |
| `client.undo()`                                              | `session.undo()`                                                                    |
| `client.interact(user_prompt)` -> dict                       | `session.interact(prompt)` -> `InteractResponse`                                    |
| `client.explain(fact_id, language)` -> dict                  | `session.explain(fact_id, language=, instructions=)` -> `ExplainResponse`           |
| `client.evidence(fact_id, evidence_key)` -> dict             | `session.evidence(fact_id, evidence_key=)` -> `Evidence`; `session.evidence_tree()` |
| `client.interaction_log(interaction_key, format="csv")`      | `session.interactions(format="csv", interaction_key=)`; default is now `"json"`     |
| `client.session_info(filters=["version"])`                   | `session.knowledge_map()` -> `KnowledgeMapVersion`                                  |
| `client.session_info(filters=["facts"], relationships=[...])`| `session.facts(relationships=[...])` -> `SessionFacts`                              |
| `client.version()` -> dict                                   | `rb.version()` -> `str`, or `rainbird_ai.engine_version(base_url)` without a key     |
| `client.close()` / `with RainbirdAPIClient(...) as client:`  | `rb.close()` / `with Rainbird(...) as rb:` (`async with AsyncRainbird(...)`)        |
| `ValueError("Session not started. Call start() first.")`     | gone: session methods live on the `Session` returned by `rb.start()` / `rb.resume()` |
| `except requests.HTTPError` / `except ValueError`            | `except RainbirdAPIError` / `except RainbirdError` (`RainbirdConfigurationError`, `InvalidAnswerError` and `InvalidFactError` still subclass `ValueError`) |

[Unreleased]: https://github.com/juliodt-ai/rainbird-python-sdk/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/juliodt-ai/rainbird-python-sdk/releases/tag/v1.0.0
