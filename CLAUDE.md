# CLAUDE.md - development guide for rainbird-python-sdk

This file is for people (and Claude Code) working *on* the SDK. End-user docs live in
`README.md`. Keep this file accurate: every command and name below is checked against the
source, so update it when you change the code.

## What this is

`rainbird-ai` (import name `rainbird_ai`) is the Python SDK for the Rainbird
decision-intelligence REST API: start a session on a knowledge map, inject facts, query a
relationship, answer the questions the reasoning engine asks until it returns a result, then
inspect the evidence behind that result. It also covers the interaction log, session facts,
the natural-language `/nl/*` endpoints (beta) and two undocumented knowledge-map endpoints.

Hard constraints:

- Python >= 3.10. Use `from __future__ import annotations`, `X | None`, `list[str]`.
- Runtime dependencies are `httpx` (`>=0.27,<1.0`) and `anyio` (`>=4`; imported only by
  `_http.py`, where `AsyncTransport` awaits `anyio.sleep` between retries). No pydantic, no
  requests, no attrs. Models are stdlib `dataclasses`. `tests/test_public_api.py` pins both the
  dependency list and the set of third-party imports, so adding a dependency means updating it.
- Strict mypy on `src/`, ruff (check + format, line length 100) and pytest must stay green. All
  of them are green after the 1.0.0 review round; do not hand over work that turns one red.
- The Python environment is `.venv/` (managed with `uv`). Ignore the stale `venv/` directory.
- Do not commit or push unless explicitly asked.

## Layout

```
pyproject.toml            hatchling build, deps, ruff/mypy/pytest config, `rainbird` console script
src/rainbird_ai/
  __init__.py             public surface: re-exports + __all__ (add new public names here)
  _version.py             __version__ = "x.y.z"; single source of truth, read by hatchling
  _http.py                UNSET sentinel (class Unset), ENGINE_CORE/ENGINE_BETA, normalize_base_url(),
                          merge_headers(),
                          ClientConfig.resolve (env resolution, require_api_key=), RetryPolicy/
                          NO_RETRY, Request[T].with_headers, error-body parsing,
                          error_for_response(), json_body(), _BaseTransport(config, owns_client)
                          -> SyncTransport/AsyncTransport
  _ops.py                 ONE builder per endpoint, each returns Request[T]; the only place URLs,
                          params, headers, bodies and response parsers are defined
  _answers.py             pure question/answer protocol logic: SKIP, build_answers, build_skip,
                          normalize_answers, normalize_facts (a fact's `metadata` dict is kept),
                          coerce_value, validate_certainty (no I/O)
  models.py               frozen dataclasses (raw + from_dict), enums Environment/QuestionForm/
                          FactSource, parse_query_response(), studio_url_for()
  client.py               sync Rainbird (account level) + Session (one reasoning session), Answerer,
                          module-level engine_version() (GET /version without an API key)
  aio.py                  AsyncRainbird + AsyncSession: same surface with await, AsyncAnswerer,
                          async_engine_version()
  cli.py                  argparse `rainbird` CLI (entry point rainbird_ai.cli:main); exit codes
                          0 ok, 1 API/SDK/file error, 2 usage, 3 question pending, 130 interrupted
  errors.py               exception hierarchy rooted at RainbirdError
  py.typed                PEP 561 marker
tests/
  conftest.py             FakeRainbird (in-memory API via httpx.MockTransport), fixtures, canned payloads
  test_*.py               unit tests against the fake; test_integration.py is opt-in (real API)
  parity/                 tests ported from the Go and JS SDK suites (support.py helper, README.md with
                          the case mapping and the behavioural differences they pin down)
examples/                 runnable example scripts
```

### Design rules

1. **Endpoint definitions live only in `_ops.py`.** A builder such as `_ops.query(session_id,
   relationship, subject=..., object=...)` validates arguments, builds a
   `Request(method, path, params, json, headers, accept, parse)` and returns it. Nothing in
   `client.py`/`aio.py` knows a URL, a header name or a JSON key.
2. **Sync and async clients are thin executors over `Request` objects.** Every client method is
   `self._send(_ops.something(...))` (sync) or `await self._send(_ops.something(...))` (async) plus
   a little local state (`Session.last_response`); the per-call `headers=` of `query`/`respond`/
   `undo`/`inject` is passed straight through as `_send(request, headers)` -> `extra_headers`. The transports in `_http.py` build the header
   set with `merge_headers(config.base_headers(), {"Accept": ...}, request.headers,
   extra_headers)` (case-insensitive, later layer wins, so `default_headers={"x-api-key": ...}`
   replaces `X-API-Key` instead of duplicating it; `base_headers()` is `User-Agent`, `X-API-Key`
   when a key is set, optional `x-rainbird-engine`, then `default_headers`), run the retry loop,
   call `request.parse(response)` on 2xx and raise `error_for_response(response)` otherwise.
   `_BaseTransport.__init__(config, owns_client)`: the SDK creates its own `httpx` client
   (`timeout=config.timeout`, `follow_redirects=False`, closed by `close()`/`aclose()`) only when
   the caller did not pass `http_client=`; a user-supplied client is never closed, and an explicit
   `timeout` is then applied per request (`timeout=` in the request kwargs) rather than ignored.
   If you find yourself writing HTTP logic in a client class, it belongs in `_ops.py` or
   `_http.py`.
3. **`_answers.py` is pure.** It turns Python values into `Answer`/`Fact` payloads according to
   the Rainbird protocol (First Form -> question triple + `answer: "yes"|"no"`; Second Form
   Object -> value in `object`; Second Form Subject -> value in `subject`; `plural` -> several
   answers; skip -> `unanswered: true`, only when `allowUnknown`). It must never import httpx or
   touch a client, so it is trivially unit-testable and shared by both clients.
4. **Never lose API data.** Every response model keeps the original JSON on `.raw`
   (`to_dict()` returns a deep copy). Unknown fields are preserved, not dropped.
5. **Configuration is resolved once, in `ClientConfig.resolve`.** Explicit argument > environment
   variable > default. `normalize_base_url(value, env)` treats `None`, `""` and whitespace as
   "not given" (falls back to `RAINBIRD_API_URL`, then `https://api.rainbird.ai`), strips trailing
   slashes, prepends `https://` to a bare host and raises `RainbirdConfigurationError` for a
   non-http(s) scheme, a scheme without a host (`"https://"`), URLs `httpx.URL` rejects, and
   any URL containing `?` or `#` (checked on the text, because httpx reports an empty trailing
   query/fragment as `""`). `timeout` defaults to the `UNSET` sentinel
   (`rainbird_ai.UNSET`, singleton of `_http.Unset`, falsy, `repr` `UNSET`) so `config.timeout ==
   30.0` with `timeout_explicit=False` unless the caller passed a value (`None` included).
   `require_api_key=False` (used by `engine_version()`/`async_engine_version()`) allows an empty
   key, in which case no `X-API-Key` header is sent. Pass `env={...}` in tests instead of touching
   `os.environ`.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

Dev extras: pytest, pytest-cov, ruff, mypy (`anyio` is a runtime dependency and also supplies
the pytest plugin behind `@pytest.mark.anyio`). Always run tools through the venv:
`.venv/bin/python`, `.venv/bin/ruff`, `.venv/bin/mypy`.

## Commands (all must be green before opening a PR)

```bash
.venv/bin/python -m pytest -q                 # unit tests against the fake server (no network)
.venv/bin/ruff check src tests                # lint (add --fix to auto-fix)
.venv/bin/ruff format src tests               # format (--check to verify without writing)
.venv/bin/mypy                                # strict type check; files=["src"] comes from pyproject
uv build                                      # sdist + wheel into dist/ (or: python -m build)
```

Opt-in integration tests against the real API (skipped when the variables are missing):

```bash
RAINBIRD_API_KEY=... RAINBIRD_KM_ID=... .venv/bin/python -m pytest -m integration -q
```

`pytest` runs with `--strict-markers`; the only registered marker is `integration`.

## Conventions

- **Models**: `@dataclass(frozen=True)` with a `raw: JSON` field and a `from_dict(data)`
  classmethod. Pythonic names on the model (`fact_id`, `km_version_id`, `allow_unknown`), API
  names only inside `raw` (`factID`, `kmVersion`, `allowUnknown`). Use the `_get(data, "factID",
  "factId")` helper to tolerate casing variants; never raise on a missing optional field.
  Input models (`Fact`, `Answer`) expose `to_payload()`.
- **Public API**: every public class, method, property and function has a docstring that says
  what the API call does and which endpoint it hits (async ones may just point at the sync
  docstring). New public names must be added to `src/rainbird_ai/__init__.py` (`__all__`) and,
  for models, to `models.__all__`; `client.__all__` is exactly `Answerer, Rainbird, Session,
  engine_version` and `aio.__all__` is `AsyncAnswerer, AsyncRainbird, AsyncSession,
  async_engine_version`. `tests/test_public_api.py` pins every `__all__`, so extend it in the
  same change. `UNSET` is the exported "argument not given" sentinel for parameters where `None`
  is itself meaningful (`timeout=None` means no timeout in httpx).
- **Sync/async parity**: `AsyncRainbird`/`AsyncSession` must expose exactly the methods of
  `Rainbird`/`Session` with the same parameters. This is enforced by
  `tests/test_public_api.py` and `tests/test_aio.py`; when you add a method to one client, add it
  to the other in the same change. The same goes for the module-level pair
  `engine_version()` (`client.py`) / `async_engine_version()` (`aio.py`).
- **Errors**: everything raised on purpose derives from `RainbirdError`. HTTP failures map to
  `RainbirdAPIError` subclasses by status (400 `BadRequestError`, 401 `AuthenticationError`,
  403 `PermissionDeniedError`, 404 `NotFoundError`, 429 `RateLimitError`, 5xx `ServerError`;
  anything else, including a 3xx because the SDK never follows redirects, is a plain
  `RainbirdAPIError` such as `"unexpected redirect to <location>"`). Text error bodies are
  whitespace-collapsed and truncated to 300 chars for `.message`/`.errors[0]`; `_clean_text`
  strips tags only when the content type is HTML or the body starts with `<`
  (`"<html>Bad Gateway</html>"` -> `"Bad Gateway"`), so plain text keeps its `<`/`>`; `.body`
  keeps the raw text; JSON messages are truncated to 300 chars too. Network failures map to
  `RainbirdConnectionError`/`RainbirdTimeoutError`, but `httpx.InvalidURL`/
  `httpx.UnsupportedProtocol` are configuration mistakes and become
  `RainbirdConfigurationError`. A 2xx body in the wrong shape is
  `UnexpectedResponseError`: `json_body()` raises it for non-JSON or empty bodies (`"expected a
  JSON body from METHOD URL but got: '<200-char snippet>'"`, the `from ...` part omitted when
  the response has no request attached), `parse_query_response()` for a decisions body that is
  neither a question nor a result (`_ops._decision` re-raises it with the status code and any
  error text from the body), `_ops.start` for a 200 without a session id,
  `_ops.interactions_json` for a 200 body that is neither a list, an envelope with a list under
  `values`/`events`/`interactions`, nor a single object with an `event` key (the message carries
  any error text found, e.g. a secured log returned as 200 `{"err": [...]}`), and `_ops.version`
  for a `/version` body that is not a plausible version: a JSON body must carry a non-empty
  string or number under `version`, `engine` or `semver` (`{}`, `[]`, `{"version": null}` raise),
  a plain-text body must be non-empty with no whitespace or `<` and at most 64 chars.
  `error_for_response()`/`json_body()` must keep working on an `httpx.Response` with no attached
  request (tests build those by hand). Bad local input raises `RainbirdConfigurationError`,
  `InvalidAnswerError`, `InvalidFactError` or `NoPendingQuestionError`. Do not raise bare
  `ValueError`/`KeyError` from public code paths.
- **Retries**: `RetryPolicy` defaults retry GET/HEAD/OPTIONS on retryable transport errors and
  on 429, 500, 502, 503 and 504 (`retry_statuses`), and POSTs only on 429 and 503
  (`post_retry_statuses`): the decision endpoints are stateful, a replayed `POST /response`
  could answer twice, and a gateway can report 502/504 *after* the engine applied the request,
  so those are deliberately not retried for POST. POST transport errors are retried only with
  `retry_post_on_transport_error=True`. "Retryable" transport errors are exactly
  `is_retryable_transport_error()`: `httpx.TimeoutException`, `NetworkError`,
  `RemoteProtocolError`, `ProxyError`; anything else is raised at once. `SyncTransport` waits
  with `time.sleep`, `AsyncTransport` with `anyio.sleep`. Keep that asymmetry; the `RetryPolicy`
  docstring explains it to users.
- **Headers** are merged case-insensitively everywhere (`merge_headers(*layers)`, later layer
  wins and keeps its spelling; `Request.with_headers()` uses it too). Never build a header dict
  by `{**a, **b}`.
- **Parameter names** mirror the API (`subject`, `relationship`, `object`) even where they shadow
  builtins; that is intentional and the ruff config allows it.
- **Environment variables** (constants `ENV_*` in `_http.py`): `RAINBIRD_API_KEY`,
  `RAINBIRD_API_URL`, `RAINBIRD_EVIDENCE_KEY`, `RAINBIRD_INTERACTION_KEY` and `RAINBIRD_ENGINE`
  are read by `ClientConfig.resolve`; `RAINBIRD_KM_ID` by `Rainbird.start`/`AsyncRainbird.start`.
  Explicit arguments always win (an explicit `base_url=""` counts as not given).
- No runtime dependencies beyond `httpx` and `anyio`. Python 3.10 floor (no `typing.Self`, `tomllib`, `StrEnum` or
  `ExceptionGroup`, all 3.11+). ruff line length 100; run
  `.venv/bin/ruff check --fix <file> && .venv/bin/ruff format <file>` on anything you write.
- **CLI** (`cli.py`): the global options (`--api-key`, `--base-url`, `--engine`,
  `--evidence-key`, `--interaction-key`, `--timeout`, `--json`) are defined once in
  `_add_global_options()` and attached twice: to the root parser with real defaults and, through
  a `parents=[common]` parser whose defaults are `argparse.SUPPRESS`, to every leaf subcommand
  (including `map get`/`map create`). That is what lets them appear before or after the
  subcommand with the later value winning; add new global options there and new subcommands via
  `command(name, help)`. `cmd_version` uses the `Rainbird` client only when `--api-key` or
  `RAINBIRD_API_KEY` is set and otherwise calls `engine_version()`, so `rainbird version` needs
  no key. `_load_facts` validates `--fact` (arity, numeric certainty, then the full
  `normalize_facts` check including the 0-100 range) and `--file` JSON (an object, a list of
  fact objects or a list of `[s, r, o, c?]` lists; a flat `["Tom", "lives in", "Germany"]` is
  rejected rather than miscounted), and `cmd_respond` the `respond` JSON, before any request is
  sent; those usage errors exit 2 through
  `_die(msg, EXIT_USAGE)`. `--answers` values (including `value@certainty`) are parsed by
  `_parse_answer_text` in `_drive` only when the matching question arrives, i.e. after
  start/query, so a rejected scripted answer propagates as a `RainbirdError` and exits 1.
  `_parse_answer_text` rejects an empty answer (`answer must not be empty`) and a numeric `@`
  suffix outside 0-100 (negative included); a non-numeric suffix is part of the value. `?@50` /
  `skip@50` still skip (certainty dropped); in a plural answer certainty applies to the whole
  answer, so `a@70, b@80` is rejected with a hint to write `a, b@70`.
  `RainbirdError`/`OSError` exit 1, and a pending question with no input source (`--no-input`,
  non-tty stdin, EOF) prints the question (or `{"session_id": ..., ...question body}` with
  `--json`) to stdout, `rainbird: session SID is waiting for an answer; resume with --session
  SID` to stderr and exits 3 (`EXIT_PENDING`); on EOF the question is not printed a second time
  (it was already shown as the prompt). With `--json`, interactive questions and prompts go to
  stderr so stdout stays a single JSON document. `_drive` re-prompts only for
  `InvalidAnswerError`, `InvalidFactError` and `BadRequestError` (the message goes to stderr as
  `  ! <message>`); any other error (401, 404 session not found, network) propagates to `main()`
  and exits 1, and a rejected `--answers` value always propagates (exit 1).

## The fake server (tests/conftest.py)

Unit tests never touch the network. `FakeRainbird` is a small stateful dataclass whose
`handle(request) -> httpx.Response` is plugged into `httpx.MockTransport`, so the real SDK code
(URL building, headers, retries, parsing) runs end to end.

Fixtures:

| fixture         | what you get                                                                  |
|-----------------|-------------------------------------------------------------------------------|
| `fake`          | a fresh `FakeRainbird`                                                        |
| `client`        | `Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=...)` wired to `fake` |
| `aclient`       | `AsyncRainbird(...)` wired to the same `fake`                                 |
| `anyio_backend` | `"asyncio"`; decorate async tests with `@pytest.mark.anyio`                   |
| `_clean_env`    | autouse; deletes all `RAINBIRD_*` variables so tests are hermetic             |

Canned payloads (import with `from tests.conftest import ...`): `API_KEY`, `BASE_URL`, `KM_ID`,
`KM_VERSION_ID`, `QUESTION_SECOND_OBJECT`, `QUESTION_SECOND_SUBJECT` (plural, allowUnknown),
`QUESTION_FIRST_FORM`, `QUESTION_PLURAL_LOSS`, `RESULT_TOM_SPEAKS_GERMAN`, `EVIDENCE_ROOT` +
`EVIDENCE_LEAVES` (a 3-node evidence tree), `SESSION_VERSION`, `SESSION_FACTS`, `INTERACTIONS`,
`INTERACT_QUESTION`, `INTERACT_RESULT`, `INTERACT_ERROR`, `EXPLAIN_OK`, `RBLANG`, `MAP_SOURCE`.

How routes behave:

- `GET /version` -> `fake.version_text` (`"4.89.0"`) as `text/plain`; no auth.
- `start`, `analysis/*` and `maps` require `X-API-Key == API_KEY`, else 401 `{"err": [...]}`.
- `GET /start/{km}` creates a session (random UUID) in `fake.sessions[sid]` and returns
  `{"id": sid, "kmVersion": {"id": KM_VERSION_ID}}`; `fake.sessions[sid]["params"]` records the
  query string (`useDraft`, `version`, `contextid`).
- `POST /{sid}/inject` appends to `fake.sessions[sid]["facts"]`, returns `{"result": "OK"}`.
- `POST /{sid}/query|response|undo` pop the next body from `fake.decisions`; when the queue is
  empty they return `RESULT_TOM_SPEAKS_GERMAN`. Dict bodies get `"sid"` added. Every `/response`
  body is appended to `fake.sessions[sid]["answers"]` so you can assert on the exact payload.
  Unknown `sid` -> 404.
- `GET /analysis/evidence/{factID}/{sid}` looks up `fake.evidence[factID]` (root + two leaves by
  default; add your own), 404 otherwise. If `fake.require_evidence_key` is set the
  `x-evidence-key` header must match or you get 403.
- `GET /analysis/interactions/{sid}` -> `INTERACTIONS` (json) or a two-line CSV;
  `fake.require_interaction_key` works like the evidence key.
- `GET /analysis/session/{sid}` -> `SESSION_VERSION` for `filter=version`, `SESSION_FACTS`
  (filtered by `relationships=a,b`) for `filter=facts`.
- `GET /analysis/file/{km}` -> `fake.map_source_body`; 404 unless `km` is `KM_ID` or the `kmID`
  in `fake.map_create_body`, so a map created through `POST /maps` can be read back.
- `POST /maps` -> HTTP 200 `{"error": "RBLang parse error..."}` when the body's `rblang` lacks
  `<rbl:kb`, else `fake.map_create_body` (`{"kmID": ..., "id": 4242, "name": ...}`).
- `POST /nl/interact` pops `fake.interact_bodies`, falling back to `INTERACT_RESULT`.
- `POST /nl/explain` -> `EXPLAIN_OK` when `factID` starts with `WA:`, else a 200 with an
  `error` object (which the SDK turns into `NaturalLanguageError`).

Scripting helpers:

```python
fake.queue(body, ...)                      # raw bodies for the next query/response/undo calls
fake.queue_question(question, *extra)      # {"question": ..., "extraQuestions": [...]}
fake.queue_result(*items)                  # {"result": [...], "queryProfile": []}; no items = empty result
fake.fail_next(status, body=None, headers=None)  # next request on ANY endpoint fails
fake.interact_bodies.append(INTERACT_ERROR)      # script /nl/interact
fake.require_evidence_key = "secret"             # evidence needs x-evidence-key
fake.evidence["WA:RF:..."] = {...}               # add evidence nodes
fake.requests / fake.last / fake.calls("inject") # Recorded(method, path, headers, params, json)
```

Gotchas:

- `fail_next` is consumed *before* routing and only once per request. The default `RetryPolicy`
  retries GETs on 429/500/502/503/504 (`retry_statuses`), so a single `fail_next(500)` on
  `rb.version()` is transparently retried and the call succeeds. To assert on an error either
  use a non-retried status (400/401/403/404), queue `max_retries + 1` failures (3 by default),
  or build a client with `retry=NO_RETRY` / `RetryPolicy(backoff_factor=0)` (the fixture client
  sleeps real backoff delays otherwise). POSTs are retried only on 429/503, so `fail_next(502)`
  on `session.inject(...)` raises `ServerError` immediately.
- `AsyncTransport` waits with `anyio.sleep`, not `asyncio.sleep`. To record or skip async
  back-off patch `anyio.sleep` (`monkeypatch.setattr(anyio, "sleep", fake)`;
  `rainbird_ai._http.anyio` is the same module object). Patching `asyncio.sleep` happens to
  work on the asyncio backend only. `SyncTransport` uses `time.sleep`.
- `rainbird version` without `--api-key`/`RAINBIRD_API_KEY` bypasses `cli.Rainbird` and calls
  `cli.engine_version`, so the CLI `run` fixture in `tests/test_cli.py` monkeypatches both:
  `rainbird_ai.cli.Rainbird` with a factory that injects
  `http_client=httpx.Client(transport=httpx.MockTransport(fake.handle))` and `BASE_URL`, and
  `rainbird_ai.cli.engine_version` with a wrapper that calls the real
  `rainbird_ai.client.engine_version(base_url or BASE_URL, http_client=<that client>, **kwargs)`.
  Without the second patch the test would hit the real Community host.
- Headers in `Recorded.headers` are lower-cased (`fake.last.headers["x-api-key"]`).

Typical tests:

```python
import pytest
from tests.conftest import KM_ID, QUESTION_FIRST_FORM, QUESTION_SECOND_OBJECT


def test_answers_second_form_object(client, fake):
    session = client.start(KM_ID)
    fake.queue_question(QUESTION_SECOND_OBJECT)
    fake.queue_result()
    response = session.query("national language", subject="Germany")
    assert response.is_question and response.question.asks_for_object
    response = session.answer("German", certainty=80)
    assert response.is_result
    sent = fake.sessions[session.id]["answers"][-1]["answers"][0]
    assert sent == {"subject": "Germany", "relationship": "national language",
                    "object": "German", "certainty": 80}


@pytest.mark.anyio
async def test_first_form_async(aclient, fake):
    session = await aclient.start(KM_ID)
    fake.queue_question(QUESTION_FIRST_FORM)
    fake.queue_result()
    await session.query("speaks", subject="Tom")
    await session.answer(True)
    assert fake.sessions[session.id]["answers"][-1]["answers"][0]["answer"] == "yes"
```

## Adding a new endpoint

1. **Read the spec first** (see references below). Note the exact path, required headers
   (`X-API-Key`, `x-evidence-key`, `x-interaction-key`, `Version: v1` for `/nl/*`), the request
   body and every response shape, including error bodies.
2. **Model** (`models.py`): add a frozen dataclass with `raw` + `from_dict`, Pythonic field names,
   tolerant parsing via `_get`/`_as_*`. Export it in `models.__all__`.
3. **Builder** (`_ops.py`): one function returning `Request[YourModel]`. Validate inputs with
   `RainbirdConfigurationError`, encode path pieces with `path_segment()`, parse with a closure
   that raises `UnexpectedResponseError` for unexpected 2xx bodies.
4. **Clients**: add the method to `Session`/`Rainbird` in `client.py` *and* to
   `AsyncSession`/`AsyncRainbird` in `aio.py` with identical names and parameters; both just
   call `_send(_ops.your_builder(...))`. Docstrings on both (the async one may just point to the
   sync one); `tests/test_public_api.py` checks that every public member is documented.
5. **Public surface**: export new names in `src/rainbird_ai/__init__.py`.
6. **Fake route** (`tests/conftest.py`): add a canned payload constant and handle the path in
   `FakeRainbird.handle`/`_analysis`, honouring the same auth rules as the real API.
7. **Tests**: sync + async happy path, error path (`fail_next`), request assertions
   (`fake.last.path/params/headers/json`), model parsing from the canned payload. Update the
   parity tests if they enumerate methods.
8. **Docs**: README.md (usage) and, if present, AGENTS.md / llms.txt; add a line under
   `Unreleased` in CHANGELOG.md. Consider a CLI subcommand in `cli.py` if it is user-facing.
9. Run the full command set above.

## Authoritative references

- OpenAPI 3.1 spec "Rainbird API Documentation" (v1.2.0), rendered at
  https://rainbird.redoc.ly/ . This is the ground truth for paths, headers, bodies and error
  shapes. When the code and the spec disagree, the spec wins, except for the undocumented
  endpoints below.
- Developer docs: https://docs.rainbird.ai/rainbird/developer-docs/overview
- API request flow (start/inject/query/response/undo):
  https://docs.rainbird.ai/rainbird/developer-docs/api-guide/api-request-flow
- Example API requests (payloads):
  https://docs.rainbird.ai/rainbird/developer-docs/api-guide/example-api-requests
- API keys / publishing:
  https://docs.rainbird.ai/rainbird/knowledge-modelling/publishing-your-km/api-management
- Hosts: Community `https://api.rainbird.ai` (Studio `app.rainbird.ai`), Enterprise
  `https://enterprise-api.rainbird.ai` (`enterprise.rainbird.ai`), South America
  `https://sa-enterprise-api.rainbird.ai` (`sa-enterprise.rainbird.ai`), private
  `https://<client>-api.rainbird.ai` (`<client>.rainbird.ai`). `studio_url_for()` encodes this
  mapping; the evidence UI link is `{studio}/evidence?id={factID}&api={apiHost}&sid={sessionID}`.

### Undocumented endpoints - do not "fix" them to match the spec

`GET /analysis/file/{kmID}` (`Rainbird.get_map_source`) and `POST /maps`
(`Rainbird.create_map`, sends `Version: v1` and `{"rblang": ..., "name": ...}` plus
`"description"` when one is given) are **not** in
the OpenAPI spec. Their behaviour was supplied by the maintainer from the live platform:
`/analysis/file` returns `{"concepts": [...], "rels": [...], "rblang": "<?xml ..."}`; `/maps`
returns `{"kmID": "...", "id": 123, ...}` on success and **HTTP 200 with a singular `error`
key** when the RBLang fails to parse (the SDK raises `KnowledgeMapError`; only truthy `err`,
`errors` and `error` values count as failures, so `{"error": false}`, `0`, `""`, `[]`, `{}`,
`null` and an informational `message` on a success body do not).
Keep them, keep them labelled "undocumented / may change" in docstrings, README and CHANGELOG,
and do not remove or reshape them because a spec search comes up empty.

## Release process

1. Make sure `Unreleased` in `CHANGELOG.md` lists every user-visible change.
2. Bump `__version__` in `src/rainbird_ai/_version.py` (semver; hatchling reads it, and it also
   feeds the `User-Agent` header and `rainbird --sdk-version`).
3. Rename `Unreleased` to `x.y.z - YYYY-MM-DD` in `CHANGELOG.md`, add a fresh empty
   `Unreleased` section and update the link references at the bottom.
4. Run the full check set (pytest, ruff check, ruff format --check, mypy) and `uv build`;
   inspect `dist/`.
5. Commit, tag `vx.y.z`, push the branch and the tag, then upload the `dist/` artefacts to PyPI
   (for example `uv publish`).
