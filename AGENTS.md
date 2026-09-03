# AGENTS.md - using the `rainbird-ai` Python SDK

Reference for coding agents that call Rainbird from another project through this SDK.
Everything below is derived from the SDK source (`src/rainbird_ai/`) and the Rainbird
OpenAPI 3.1 spec. Do not hand-roll HTTP calls with `requests`; use the SDK.

## 1. What Rainbird is

Rainbird is a decision-intelligence platform. Domain experts build a **knowledge map**
(concepts, relationships, rules, written in RBLang XML) in Rainbird Studio and publish it.
Client code then:

1. **starts a session** against a knowledge map (a `kmID` GUID),
2. optionally **injects facts** it already knows (`subject relationship object` triples with a
   certainty 1-100),
3. **queries** a relationship (`"speaks"` with `subject="Tom"`),
4. **answers the questions** the reasoning engine asks until it returns
5. **results**: triples with a `certainty` (0-100) and a `fact_id`, which unlocks the
   **evidence tree** (the rule and conditions that produced the fact) and natural-language
   explanations.

Every `query`, `respond`, `answer`, `skip` and `undo` call returns either a
`QuestionResponse` (more information needed) or a `ResultResponse` (a decision).

## 2. Install and import

```bash
pip install rainbird-ai        # Python >= 3.10, runtime dependencies: httpx, anyio
```

```python
from rainbird_ai import Rainbird, SKIP, Answer, Fact, RainbirdError   # sync
from rainbird_ai.aio import AsyncRainbird                             # async
from rainbird_ai import engine_version, async_engine_version, UNSET   # keyless version check, timeout sentinel
```

Console script: `rainbird --help` (`rainbird query -r speaks -s Tom --answers German --json`).
CLI notes for scripting:

- Global options `--api-key --base-url --engine --evidence-key --interaction-key --timeout --json`
  are accepted before or after the subcommand (including `map get` / `map create`).
- `rainbird version` needs no API key: without `--api-key`/`RAINBIRD_API_KEY` it calls
  `engine_version()`.
- `--answers`: `?` skips, `a,b` answers a plural question, `value@80` sets the certainty (`@` counts
  only when the suffix is numeric, so `tom@example.com` is a plain value; a numeric suffix outside
  0-100, negative included, is rejected like any other bad answer, and so is an empty answer:
  `answer must not be empty`). `?@50` / `skip@50` still skip (the certainty is dropped). Certainty
  applies to the whole answer: `a, b@70` is valid, `a@70, b@80` is rejected with a message saying
  so. Scripted answers are parsed only when the matching question
  arrives, i.e. after the session has started and the query was sent. `--fact S R O [CERTAINTY]`
  accepts a float certainty, validated (0-100) before any request like `--file` JSON, which must be
  a JSON object, a list of fact objects or a list of `[subject, relationship, object, certainty?]`
  lists (a flat `["Tom", "lives in", "Germany"]` is a usage error, not one fact).
- `--json`: stdout is a single JSON document; interactive questions and prompts go to stderr.
- Exit codes: `0` success; `1` API/SDK/file error or a rejected `--answers` value
  (`rainbird: <message>` on stderr); `2` usage error, including an invalid `--fact` (non-numeric or
  out-of-range certainty), malformed `--file` JSON or malformed `respond` JSON (all checked before
  any network call); `3` a
  question is still pending and no input is available (`--no-input`, non-tty stdin or Ctrl-D): the
  question goes to stdout (text mode) or `{"session_id": ..., ...questionResponse}` (`--json`), then
  `rainbird: session <SID> is waiting for an answer; resume with --session <SID>` to stderr; `130`
  interrupted. Interactive rejections re-prompt and print `  ! <message>` to stderr, but only for
  answer-level problems (`InvalidAnswerError`, `InvalidFactError`, `BadRequestError`); a 401, a 404
  (session not found), a network error or any other failure exits 1 immediately.
- `evidence --tree` prints a nested tree: root at column 0, each child evidence node indented two
  more spaces, inline conditions as `- ...` lines beneath their node, facts referenced only from
  expression functions beneath their expression, and any fetched node not reachable from the root
  appended at the end.

## 3. Canonical decision loop

```python
from rainbird_ai import Rainbird

rb = Rainbird()                                        # RAINBIRD_API_KEY / RAINBIRD_API_URL from env
session = rb.start("<knowledge-map-id>")               # GET /start/{kmID} -> Session
session.inject_fact("Tom", "lives in", "Germany")      # POST /{sid}/inject (optional, known facts)
response = session.query("speaks", subject="Tom")      # POST /{sid}/query -> Question or Result
while response.is_question:                            # Literal-typed: narrows to QuestionResponse under mypy
    q = response.question                              # q.prompt, q.form, q.options, q.plural ...
    value = True if q.is_first_form else (q.options[0] if q.options else "German")
    response = session.answer(value)                   # POST /{sid}/response -> Question or Result
for item in response.results:                          # ResultResponse (may be empty)
    print(item.subject, item.relationship, item.object, item.certainty, item.fact_id)
if response.best:                                      # highest-certainty result or None
    evidence = session.evidence(response.best.fact_id) # GET /analysis/evidence/{factID}/{sid}
rb.close()                                             # or: with Rainbird() as rb: ...
```

Fully automated variant: `result = session.run("speaks", subject="Tom", answerer=fn)` (section 9).

## 4. Configuration

`Rainbird(api_key=None, *, base_url=None, evidence_key=None, interaction_key=None, engine=None,
timeout=UNSET, retry=None, default_headers=None, http_client=None)`. `AsyncRainbird` has the same
signature with `http_client: httpx.AsyncClient`. Explicit arguments win over env vars.

| Argument          | Env var                    | Meaning |
|-------------------|----------------------------|---------|
| `api_key`         | `RAINBIRD_API_KEY`         | Sent as `X-API-Key` (Studio > knowledge map > Publish). Required; missing -> `RainbirdConfigurationError` (only the module-level `engine_version()` works without a key). |
| `base_url`        | `RAINBIRD_API_URL`         | API host, default `https://api.rainbird.ai`. A bare host gets `https://`, trailing `/` stripped; `""`/whitespace falls back to the env var, then the default; a non-`http(s)` scheme, a scheme without a host (`"https://"`), a query string or fragment (any `?` or `#`, even an empty trailing one), or an otherwise invalid URL -> `RainbirdConfigurationError`. |
| `evidence_key`    | `RAINBIRD_EVIDENCE_KEY`    | Default `x-evidence-key` for evidence calls (evidence is secured by default). |
| `interaction_key` | `RAINBIRD_INTERACTION_KEY` | Default `x-interaction-key` for interaction logs. |
| `engine`          | `RAINBIRD_ENGINE`          | Optional alternate engine, sent as `x-rainbird-engine`. |
| (start km_id)     | `RAINBIRD_KM_ID`           | Fallback for `start(km_id=None)`. |
| `timeout`         | -                          | Seconds, an `httpx.Timeout`, or `None` (no timeout). Default is the `UNSET` sentinel = 30 s on the SDK's own client; with `http_client=` the client's own timeout applies unless you pass `timeout=` explicitly, which is then applied per request. |
| `retry`           | -                          | `RetryPolicy` (section 11); `NO_RETRY` disables retries. |
| `default_headers` | -                          | Extra headers on every request, merged case-insensitively (later wins): `{"x-api-key": ...}` replaces `X-API-Key` instead of duplicating it. |
| `http_client`     | -                          | Bring your own `httpx.Client`; you close it. The SDK's own clients use `follow_redirects=False` (a 3xx raises `RainbirdAPIError("unexpected redirect to <location>")`); keep that on yours so the API key is never forwarded to another host. |

Hosts (`rainbird_ai.Environment`, a `str` enum usable directly as `base_url`):

| Account type            | `base_url`                              | Studio (`.studio_url`)               |
|-------------------------|-----------------------------------------|--------------------------------------|
| `Environment.COMMUNITY` | `https://api.rainbird.ai`               | `https://app.rainbird.ai`            |
| `Environment.ENTERPRISE`| `https://enterprise-api.rainbird.ai`    | `https://enterprise.rainbird.ai`     |
| `Environment.ENTERPRISE_SOUTH_AMERICA` | `https://sa-enterprise-api.rainbird.ai` | `https://sa-enterprise.rainbird.ai` |
| Private deployment      | `https://<client>-api.rainbird.ai`      | `https://<client>.rainbird.ai`       |

The `Version: v1` header required by `/nl/*` and `/maps` is added by the SDK. Auth headers are
added by the SDK. `studio_url_for(api_url)` derives the Studio host from any API host.

## 5. Public API reference (sync)

`AsyncRainbird` / `AsyncSession` (module `rainbird_ai.aio`) expose the same names; every method
that performs I/O is `async` and must be awaited (`session = await rb.start(...)`,
`await session.answer(...)`), `close()` becomes `await aclose()`, use
`async with AsyncRainbird() as rb`. `resume()`, `evidence_url()` and the properties stay
synchronous; `iter_evidence_tree()` returns an `AsyncIterator` (`async for`).
`rainbird_ai.aio.async_engine_version()` is the awaitable twin of `engine_version()`.

### `Rainbird`

- `start(km_id=None, *, use_draft=None, version=None, context_id=None) -> Session` - `GET /start/{kmID}`. Live version by default (draft if nothing is live). `use_draft=True` or `version=N` (positive int), never both (`RainbirdConfigurationError`). `context_id` shares context-scoped facts between sessions.
- `resume(session_id, *, km_id=None) -> Session` - attach to an existing session id; no request; `pending` is `None` until you query.
- `version() -> str` - `GET /version`, engine semver such as `"4.89.0"`. Needs a configured client (so an API key); the endpoint itself is public, use module-level `engine_version()` when you have no key.
- `evidence(session_id, fact_id, *, evidence_key=None) -> Evidence` - session-independent evidence lookup.
- `evidence_url(session_id, fact_id, *, studio_url=None) -> str` - `{studio}/evidence?id={factID}&api={base_url}&sid={sid}` link to the Studio evidence UI.
- `get_map_source(km_id) -> KnowledgeMapSource` - UNDOCUMENTED `GET /analysis/file/{kmID}` (section 12).
- `create_map(name, rblang) -> CreateMapResult` - UNDOCUMENTED `POST /maps` (section 12).
- `close()`; context manager; properties `base_url`, `studio_url`, `http` (the `httpx.Client`), `config`.

### `Session` (from `rb.start()` / `rb.resume()`)

State: `id` (session id), `km_id`, `km_version_id`, `context_id`, `start_response`,
`last_response: QueryResponse | None`, `client`. Derived: `pending -> QuestionResponse | None`,
`pending_question -> Question | None`, `result -> ResultResponse | None`. State is updated only
after a successful `query/respond/answer/skip/undo`.

- `inject(facts, *, batch_size=250) -> InjectResponse` - `POST /{sid}/inject`. Accepts one or many of: `Fact`, `{"subject","relationship","object"[,"certainty"]}`, `("s","r","o"[,certainty])`. Objects in all three forms are coerced like answers (section 8: `date`/`datetime` -> `"YYYY-MM-DD"`, `Decimal` -> number, enum -> `.value`; `None`/dict/list/NaN/infinity -> `InvalidAnswerError`) and certainty is validated (`InvalidFactError`). A 3/4-item tuple or list whose items are none of `Fact`/dict/list/tuple/set is one fact, so `("Tom", "lives in", Country.DE)` works. Lists are chunked into batches of `batch_size` (Rainbird recommends <= 250); returns the last batch's response (`.ok`).
- `inject_fact(subject, relationship, object, certainty=100) -> InjectResponse`.
- `query(relationship, *, subject=None, object=None) -> QueryResponse` - `POST /{sid}/query`. Give subject, object or both; `relationship` is required (`RainbirdConfigurationError` if empty).
- `answer(value, *, certainty=100, question=None, strict=True) -> QueryResponse` - builds the correct payload for `question` (default `pending_question`, else `NoPendingQuestionError`) from a plain Python value (section 8) and sends `POST /{sid}/response`.
- `skip(*, question=None, strict=True) -> QueryResponse` - sends `unanswered: true` for the pending question. Same as `answer(SKIP)`.
- `respond(answers) -> QueryResponse` - send prebuilt `Answer` objects/dicts (one or a list). Use for `extra_questions` or `QuestionForm.UNKNOWN`. Dict keys allowed: subject, relationship, object, certainty (or cf), answer, unanswered. `Answer` objects and dicts are both normalised before sending: object coerced (section 8), certainty validated, `answer` normalised to `"yes"`/`"no"` (anything else -> `InvalidAnswerError`), `unanswered` must be a bool or a `"true"/"false"/"yes"/"no"/"1"/"0"`-like string (other values, including numbers, -> `InvalidFactError`).
- `undo() -> QueryResponse` - `POST /{sid}/undo`: rolls back the last answers array and returns the question that preceded it (or the same question when nothing is left to undo).
- `run(relationship, *, subject=None, object=None, answerer, max_questions=100) -> ResultResponse` - query, then loop calling `answerer(question)` until a result (section 9). Raises `RainbirdError` after `max_questions` questions.
- `evidence(fact_id, *, evidence_key=None) -> Evidence` - one node.
- `evidence_tree(fact_id, *, max_depth=None, evidence_key=None) -> list[Evidence]` - breadth-first walk over `child_fact_ids`, root first, each node fetched once (one GET per node). `iter_evidence_tree(...) -> Iterator[Evidence]` is the lazy form.
- `evidence_url(fact_id, *, studio_url=None) -> str`.
- `interactions(*, format="json", interaction_key=None) -> list[InteractionEvent] | str` - `GET /analysis/interactions/{sid}`; overloaded so `format="json"` (default) is typed `list[InteractionEvent]` and `format="csv"` is typed `str` (raw CSV text). Logging must be enabled per map version in Studio before the session starts. A 200 body that is neither a list, an envelope with a list under `values`/`events`/`interactions`, nor a single event object (`event` key) raises `UnexpectedResponseError` carrying any error text found (a secured log may come back as 200 `{"err": [...]}`).
- `knowledge_map() -> KnowledgeMapVersion` - `GET /analysis/session/{sid}?filter=version`.
- `facts(*, relationships: str | Iterable[str] | None = None) -> SessionFacts` - `GET /analysis/session/{sid}?filter=facts[&relationships=a,b]`; a single `str` is sent as-is (`relationships="speaks"` -> `relationships=speaks`), an iterable is comma-joined.
- `interact(prompt) -> InteractResponse` - `POST /nl/interact` (beta). Errors are returned, not raised.
- `explain(fact_id, *, language="en", instructions=None) -> ExplainResponse` - `POST /nl/explain` (beta). Raises `NaturalLanguageError` when the body has `error` and no `explanation`.

### Module-level helpers (`rainbird_ai`)

`SKIP` (sentinel, falsy), `build_answers(question, value, *, certainty=100, strict=True) -> list[Answer]`,
`build_skip(question, *, strict=True) -> list[Answer]`, `parse_query_response(dict) -> QueryResponse`,
`studio_url_for(api_url) -> str`, `RetryPolicy`, `NO_RETRY`, `DEFAULT_BASE_URL`, `__version__`,
type aliases `Answerer = Callable[[Question], Any]` and `AsyncAnswerer` (also in `rainbird_ai.aio`).

- `engine_version(base_url=None, *, engine=None, timeout=UNSET, http_client=None) -> str` -
  `GET /version` with **no API key** (no `X-API-Key` header is sent). `base_url` falls back to
  `RAINBIRD_API_URL`, then the Community host. Opens and closes its own connection unless you pass
  `http_client` (then you close it). Use it for connectivity/health checks.
  `async_engine_version(...)` (also exported from `rainbird_ai.aio`) is the awaitable twin and takes
  an `httpx.AsyncClient`.
- `UNSET` - falsy sentinel meaning "argument not given"; the default for `timeout=` in `Rainbird`,
  `AsyncRainbird`, `engine_version` and `async_engine_version`. Distinct from `None`, which disables
  the timeout. Test with `x is UNSET`.

## 6. Models (frozen dataclasses; every response model has `.raw` (original JSON) and `.to_dict()`)

Inputs (have `.to_payload()`):
- `Fact(subject: str, relationship: str, object: Any, certainty=100)`.
- `Answer(subject: str, relationship: str, object: Any, certainty=100, answer: str | None = None, unanswered=False)`. `answer` is `"yes"`/`"no"` for First Form only; `unanswered=True` skips.

Decisions:
- `StartResponse(session_id, km_version_id)`.
- `InjectResponse(result)`; `.ok` (true for `"OK"`).
- `Question(subject, relationship, object, type, prompt, data_type, object_type, plural, allow_cf, allow_unknown, can_add, known_answers, concepts: list[Concept], object_metadata)`. Properties: `form -> QuestionForm`, `is_first_form`, `is_second_form`, `asks_for_subject`, `asks_for_object`, `options` (concept values with `invalid_response` false), `is_truth` (`data_type == "truth"`). `str(q)` is the prompt.
- `Concept(name, value, concept_type, data_type, invalid_response, fsid)`.
- `QuestionForm`: `FIRST`, `SECOND_OBJECT`, `SECOND_SUBJECT`, `UNKNOWN` (derived from `Question.type`; plain `"Second Form"` is inferred from which side is missing).
- `QuestionResponse(question, extra_questions, sid)`; `is_question: Literal[True]`, `is_result: Literal[False]` (runtime `True`/`False`), `questions` (main + extra), `results == []`.
- `ResultItem(subject, relationship, object, certainty, fact_id, relationship_type, subject_metadata, object_metadata)`; also parses the capitalised keys returned by `/nl/interact`.
- `ResultResponse(results: list[ResultItem], sid)`; `is_question: Literal[False]`, `is_result: Literal[True]`, `question is None`, `best` (max certainty or `None`), `fact_ids`, `len()`, iterating yields `ResultItem`.
- `QueryResponse = QuestionResponse | ResultResponse` (a type alias, not a class). Because `is_question`/`is_result` are `Literal`-typed, `if response.is_question:` (or `while response.is_question:`) narrows the union to `QuestionResponse` under mypy and the `else` branch to `ResultResponse`, exactly like `isinstance`; `if response.is_result:` narrows the other way.

Evidence and analysis:
- `Evidence(fact_id, source: FactSource | str | None, fact: SessionFact, time (epoch ms), rule: Rule | None)`; `timestamp` (aware datetime), `is_inferred` (`rule is not None`), `child_fact_ids` (fact IDs one level down, deduplicated).
- `Rule(bindings: dict, conditions: list[Condition | ExpressionCondition], max_certainty)`; `relationship_conditions`, `expression_conditions`.
- `Condition(subject, relationship, object, certainty, salience, impact, fact_id, fact_key, object_type, alt)`.
- `ExpressionCondition(text, value, was_met, functions, salience, impact, alt)`; `fact_ids` referenced by evaluated functions.
- `SessionFact(subject: FactValue, relationship, object: FactValue, certainty, id, source)`; `FactValue(value, concept, data_type)`.
- `SessionFacts(global_, local, context)`; `all`, `len()`, iterating yields `SessionFact`.
- `FactSource`: `KNOWLEDGE_MAP="knowledgemap"`, `RULE`, `ANSWER`, `INJECTION`, `DATASOURCE`, `SYNTHESIS` (unknown strings pass through as `str`). `str()` of a `FactSource`, `QuestionForm` or `Environment` member is its `.value` (`str(FactSource.RULE) == "rule"`).
- `KnowledgeMapVersion(id, name, version_id, version_status, version_created: datetime | None)`; `is_draft`, `version_number` (`3` for `"Version3"`).
- `InteractionEvent(event, created: datetime | None, values: dict)`.

Natural language (beta):
- `InteractResponse(response_type: "question"|"result"|"error", query, questions: list[Question], results: list[ResultItem], error: NLError | None, metadata: NLMetadata | None, injected_facts, invalid_facts, unmatched: list[str])`; `is_question`, `is_result`, `is_error`, `question` (first or `None`), `prompt` (question prompt or suggested chat response), `raise_for_error()` (raises `NaturalLanguageError`, else returns self).
- `ExplainResponse(explanation: str | None, error, metadata)`; `str()` is the explanation.
- `NLError(code, message, suggested_chat_response, status_code)`; `NLMetadata(endpoint, session_id, query_submitted_at, language_engine_response_ms, prompt_tokens, completion_tokens, total_tokens, api_version)`.

Knowledge maps (undocumented endpoints):
- `KnowledgeMapSource(km_id, rblang: str, concepts: list, relationships: list)`.
- `CreateMapResult(km_id: str (GUID for start()), name, id (Rainbird numeric id))`.

## 7. Exceptions (all subclass `RainbirdError`; `except RainbirdError` catches everything the SDK raises on purpose)

| Exception | Raised when |
|-----------|-------------|
| `RainbirdConfigurationError` (also `ValueError`) | No API key; `base_url` without a host, with a non-`http(s)` scheme, any `?` or `#` (query string or fragment, even empty), or otherwise invalid (also `httpx.InvalidURL`/`UnsupportedProtocol` at request time); empty km_id/session_id/relationship/prompt; `use_draft` with `version`; bad `version`/`batch_size`. |
| `NoPendingQuestionError` | `answer()`/`skip()` with no pending question and no `question=`. |
| `InvalidAnswerError` (also `ValueError`) | Value does not fit the question: not yes/no for First Form, list for non-plural, not in `options` when `can_add` is false, skip when `allow_unknown` is false, non-boolean value for a truth object, `QuestionForm.UNKNOWN`; also an unsupported value (`None`, dict, list outside plural, NaN or infinity) wherever a value is coerced, including objects passed to `inject()`/`respond()`. |
| `InvalidFactError` (also `ValueError`) | Malformed fact/answer payload (missing keys, unknown keys, certainty outside 0-100 or non-numeric, `unanswered` not bool-like, empty list). |
| `KnowledgeMapError` | `create_map()` got HTTP 200 whose body carries a truthy `err`/`errors`/`error`; `.errors`, `.body`. |
| `NaturalLanguageError` | `explain()` returned an error object; `interact(...).raise_for_error()`. `.code`, `.message`, `.suggested_chat_response`, `.status_code`, `.body`. |
| `RainbirdTransportError` > `RainbirdConnectionError` / `RainbirdTimeoutError` | No HTTP response (DNS/TCP/TLS failure, proxy or remote-protocol error, or timeout); `.method`, `.url`. |
| `UnexpectedResponseError` | 2xx body not in the documented shape, including a non-JSON or empty 2xx body (`expected a JSON body from GET <url> but got: '<snippet>'`), a start response without a session id, an interaction log that is not a list of events (see `interactions()`) and a `GET /version` body that is not a plausible version (JSON must carry a non-empty string or number under `version`/`engine`/`semver`, so `{}`, `[]` and `{"version": null}` raise; plain text must be non-empty, without whitespace or `<`, at most 64 chars); `.body`, `.status_code`. |
| `RainbirdAPIError` | Any non-2xx, including 3xx (`unexpected redirect to <location>`; redirects are never followed); `.status_code`, `.message`, `.errors` (from `{"err": [...]}`, `{"error": ...}`, `{"message": ...}`), `.body`, `.method`, `.url`, `.retry_after`. Text bodies are whitespace-collapsed and truncated to 300 chars for `.message`/`.errors[0]`; tags are stripped only when the body is HTML (HTML content type or a body starting with `<`: `<html>Bad Gateway</html>` -> `Bad Gateway`), plain text keeps its `<`/`>`; JSON messages are truncated to 300 chars too; `.body` keeps the raw text/JSON. |
| `BadRequestError` 400 | Missing/invalid parameter, e.g. skipping a question without `allowUnknown` -> `"Please provide an expected boolean value for unanswered."`. |
| `AuthenticationError` 401 | Invalid/missing API key. |
| `PermissionDeniedError` 403 | Key not permitted (e.g. secured evidence without `x-evidence-key`). |
| `NotFoundError` 404 | Unknown session, knowledge map or fact. |
| `RateLimitError` 429 | Too many requests; `.retry_after`. |
| `ServerError` 5xx | Platform failure. |

`NoActiveSessionError` no longer exists (it was never raised); do not import it. A `Session` from
`resume()` is always usable; the only "no session state" error is `NoPendingQuestionError`.

## 8. Question protocol (what `session.answer(value)` sends)

Decide from `q = session.pending_question`:

| `q.form` | `q.type` contains | Ask | `answer(value)` builds |
|----------|-------------------|-----|------------------------|
| `FIRST` | `"First Form"` | "Does Tom speak French?" | the question's own triple + `answer: "yes"/"no"`. `value`: `True/False`, `"yes"/"no"`, `"y"/"n"`, `1/0`, or an enum member whose `.value` is one of those. |
| `SECOND_OBJECT` | `"Second Form Object"` | "What does Tom speak?" | `subject=q.subject, object=value`. |
| `SECOND_SUBJECT` | `"Second Form Subject"` | "Who speaks French?" | `subject=value, object=q.object`. |
| `UNKNOWN` | anything else | - | `InvalidAnswerError`; build `Answer` objects and call `respond()`. |

Rules:
- **Options**: `q.options` lists the candidate values from `q.concepts`. If `q.can_add` is `False` and options exist, the value must be one of them (strict mode raises `InvalidAnswerError`; the API would reject it anyway). If `can_add` is `True` any value is accepted. Empty options means free text/number/date input. Matching: for a truth-typed Second Form **Object** question the answer and the options are normalised to booleans (`"yes"` matches option `True`). Otherwise an exact string match wins and your value is sent unchanged; a purely numeric match (`5.0` or `"10.0"` against the options `5` / `"10"`) is accepted only when the answer is an `int`/`float`, the Second Form Object question's `data_type` is `number`/`integer`/`float`, or the option itself is a number, and then the **option's own value** is sent (the concept instance the engine knows), so `answer(5.0)` sends `5`. Everything else (Second Form Subject answers included) must equal an option exactly as a string; an `int` too large for a `float` simply fails to match (`InvalidAnswerError`, never `OverflowError`). If every concept is flagged `invalidResponse`, `q.options` is empty and validation is left to the API.
- **plural**: `q.plural=True` accepts a list/tuple/set of values -> one answer per value in a single request. A list of more than one value on a non-plural question raises `InvalidAnswerError` (strict).
- **allow_unknown**: `q.allow_unknown=True` allows skipping: `session.skip()` or `answer(SKIP)` sends `{"unanswered": true, certainty: 100}` with `object: ""` (subject questions: `subject: ""`; First Form keeps its object). When `allow_unknown` is `False`, strict mode raises `InvalidAnswerError`; if you bypass it the API answers 400.
- **allow_cf**: when `True` the engine honours `certainty=` (0-100; API documents 1-100; 100 = certain). The SDK always sends `certainty` (default 100); values outside 0-100 raise `InvalidFactError`.
- **data_type** (describes the object side): `"string"` -> str; `"number"` -> int/float; `"truth"` -> Python `bool` for Second Form **Object** answers (`True/False`, `1/0`, `"yes"/"y"/"true"/"t"/"1"`, `"no"/"n"/"false"/"f"/"0"`, or an enum member with one of those values, are converted; the payload carries a JSON boolean; anything else -> `InvalidAnswerError`). Second Form **Subject** answers are always sent as plain strings, whatever `data_type` says. `"date"` -> pass `datetime.date`/`datetime.datetime`, sent as `"YYYY-MM-DD"` (the time of day of a `datetime` is dropped; pass an `int` of epoch milliseconds if you need an instant), or an ISO string. `Decimal` becomes `int` when integral, else `float`; enums (including `str`/`int`/`bool`-valued ones) use `.value`, for First Form and truth answers too; `None`, dicts, lists (outside plural) and non-finite numbers (NaN/infinity, as `float` or `Decimal`) raise `InvalidAnswerError`.
- `certainty` on the answer applies to every generated answer of a plural response.
- `q.known_answers` lists values already known for this question. `response.extra_questions` are optional additional questions; `answer()` only handles `response.question`, so to answer several at once use `session.respond(build_answers(q1, v1) + build_answers(q2, v2))`.
- `strict=False` on `answer()`/`skip()`/`build_answers()` skips client-side validation and lets the API judge.

## 9. Answerer contract for `session.run(...)`

```python
from rainbird_ai import SKIP, Question

def answerer(q: Question):
    if q.is_first_form:            # yes/no
        return False
    if q.is_truth:                 # truth-typed value question
        return True
    if q.options:                  # choose among concepts (respect q.can_add / q.plural)
        return q.options[:2] if q.plural else q.options[0]
    if q.allow_unknown:            # nothing to say and skipping is allowed
        return SKIP
    return "German"                # free value: str / int / float / date

result = session.run("speaks", subject="Tom", answerer=answerer, max_questions=50)
```

Return values accepted: a single value, a list of values (plural), `SKIP`, an `Answer`, or a
list of `Answer` (sent verbatim via `respond()`). Returning a value for a question the value does
not fit raises `InvalidAnswerError` from inside `run()`. `AsyncSession.run` accepts a sync or
`async def` answerer. The loop stops with `RainbirdError` after `max_questions` (default 100).
`run()` never returns a `QuestionResponse`.

## 10. Reading results, evidence and explanations

```python
result = session.result                     # ResultResponse after the loop, or response itself
for item in result:                         # ResultItem
    item.subject, item.relationship, item.object, item.certainty, item.fact_id
best = result.best                          # None when result.results is empty (nothing inferred)
if best and best.fact_id:
    ev = session.evidence(best.fact_id)     # Evidence: ev.source, ev.fact, ev.rule, ev.child_fact_ids
    if ev.rule:
        ev.rule.bindings                    # {"S": "Tom", "COUNTRY": "Germany", ...}
        for cond in ev.rule.conditions:     # Condition (has .fact_id) | ExpressionCondition (.text, .was_met)
            ...
    tree = session.evidence_tree(best.fact_id, max_depth=3)   # list[Evidence], root first
    url = session.evidence_url(best.fact_id)                  # Studio evidence-tree link
    text = session.explain(best.fact_id, language="en").explanation   # beta, LLM generated
```

`fact_id` values are opaque strings (typically `WA:RF:<hash>` for rule-inferred facts, `WA:IF:`
injected, `WA:AF:` answered); never parse them, pass them to `evidence()`/`explain()`. An empty
`results` list is a valid decision ("nothing could be inferred"), not an error. Evidence is secured by default: pass `evidence_key=` (or `RAINBIRD_EVIDENCE_KEY`) or a
403 `PermissionDeniedError` is raised.

Natural language (beta, results vary): `r = session.interact("Ben lives in Germany. What does he speak?")`;
`r.response_type` is `"question"`, `"result"` or `"error"`. Check `r.is_error` / `r.prompt` /
`r.error.suggested_chat_response`; call `r.raise_for_error()` only if you want an exception.
`r.injected_facts`, `r.invalid_facts`, `r.unmatched` report what the language layer extracted.

## 11. Retries, timeouts and idempotency

Default `RetryPolicy(max_retries=2, backoff_factor=0.5, max_backoff=20.0,
retry_statuses={429,500,502,503,504}, post_retry_statuses={429,503},
retry_post_on_transport_error=False)`, exponential backoff with jitter, honours `Retry-After`.

- `GET` (start, evidence, analysis, version) retries on connection errors, timeouts and 429, 500,
  502, 503 and 504 (`retry_statuses`).
- `POST` (inject, query, response, undo, nl/*, maps) retries only on 429/503 (the service refused
  the request before processing it) and **never on 502/504, timeouts or connection errors**: a
  gateway can report 502/504 after the engine has already applied the request, and a
  `POST /{sid}/response` that timed out may already have been applied server-side, so retrying
  could answer twice.
- Only `httpx.TimeoutException`, `NetworkError`, `RemoteProtocolError` and `ProxyError` count as
  retryable transport errors; `httpx.InvalidURL`/`UnsupportedProtocol` raise
  `RainbirdConfigurationError` at once. Redirects are never followed (3xx -> `RainbirdAPIError`).
  Back-off sleeps with `time.sleep` (sync) / `anyio.sleep` (async).
- After a `RainbirdTimeoutError` from `answer()/respond()/skip()`, `session.pending` still holds
  the question you were answering (state is only updated on success), so you cannot tell whether
  the answer landed. Recover by re-issuing `session.query(...)` with the same arguments: the
  session keeps every fact already accepted and the engine re-asks only what it still needs.
  Alternatively start a fresh session and re-inject known facts. Do not blindly call `undo()`.
- `retry=NO_RETRY` disables retries; `RetryPolicy(retry_post_on_transport_error=True)` opts into
  POST retries on transport errors and `RetryPolicy(post_retry_statuses={429, 502, 503, 504})`
  restores gateway-status retries for POST; both only if your flow tolerates duplicate answers.
- Reuse one `Rainbird` client (connection pool) for many sessions; a `Session` object holds
  mutable state (`last_response`) and must not be shared across concurrent tasks. Rainbird
  recommends a new session per query unless several queries must share facts.

## 12. Undocumented knowledge-map endpoints (unstable)

Not in the OpenAPI spec; supplied by Rainbird maintainers and may change without notice.

- `rb.get_map_source(km_id) -> KnowledgeMapSource` - `GET /analysis/file/{kmID}` with `X-API-Key`; returns `{"concepts": [...], "rels": [...], "rblang": "<?xml ..."}` -> `.rblang`, `.concepts`, `.relationships`.
- `rb.create_map(name, rblang) -> CreateMapResult` - `POST /maps` with `X-API-Key` and `Version: v1`, body `{"rblang": xml, "name": name}` -> `{"kmID": "...", "id": 123, ...}`. On RBLang parse failure Rainbird replies **HTTP 200 with an `error` key**; the SDK raises `KnowledgeMapError` (`.errors`). Only truthy `err`/`errors`/`error` values are treated as failures (`{"error": false}`, `null`, `""`, `0`, `[]` and `{}` are success); an informational `message` on a success body is ignored. Use `result.km_id` with `rb.start()`.

Minimal RBLang: `<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang"><concept name="Person" type="string"/><concept name="Language" type="string"/><rel name="speaks" subject="Person" object="Language"/></rbl:kb>`
(concept types: `string|number|date|truth`; instances via `<concinst>`/`<relinst>`).

## 13. Common mistakes

1. Putting a `Second Form Subject` answer in `object` (or vice versa). Use `session.answer(value)`; it reads `q.form`. When building `Answer` by hand, check `q.asks_for_subject`.
2. Skipping when `q.allow_unknown` is `False`: `InvalidAnswerError` locally, 400 from the API. Provide a value from `q.options` instead.
3. Answering a First Form question with the object value (`"French"`) instead of yes/no.
4. Sending several values to a non-plural question, or a value outside `q.options` when `q.can_add` is `False`.
5. Passing a truth answer as the string `"true"` in a hand-built `Answer` (send a JSON boolean; `answer()` converts for you) or a date as a non-ISO string.
6. Hardcoding `https://api.rainbird.ai` for Enterprise/private accounts: use `Environment.ENTERPRISE` or `https://<client>-api.rainbird.ai`; a 401 `AuthenticationError` with a valid key usually means the wrong host.
7. Adding `Version: v1`, `X-API-Key` or `x-rainbird-engine` headers yourself, or using `requests`: the SDK sets all headers.
8. Treating `interact()` errors as exceptions: they are returned in `InteractResponse` (`is_error`); only `raise_for_error()` raises.
9. Reading `response.results` on a `QuestionResponse` (always `[]`) or `response.question` on a `ResultResponse` (always `None`) without branching on `is_question`.
10. Calling `answer()` on a `resume()`d session before a `query()`: no pending question -> `NoPendingQuestionError`; pass `question=` or query first.
11. Expecting `result.best` to be non-`None`; empty results are legitimate.
12. Retrying a timed-out `answer()` in your own code (duplicate answers). Re-query instead (section 11).
13. Using `use_draft=True` together with `version=N`, or forgetting that the default is the live version (draft changes are invisible until published or `use_draft=True`).
14. Fetching evidence without an evidence key on a secured map (403), or reading `interactions()` on a session started before logging was enabled (empty).
15. Forgetting `rb.close()` / `await rb.aclose()` (or the context manager) in long-running processes.
16. Importing `NoActiveSessionError`: it was removed (never raised). Catch `NoPendingQuestionError` for `answer()`/`skip()` without a question and `RainbirdConfigurationError` for bad arguments.
17. Passing a `datetime` and expecting the time of day to survive: only `YYYY-MM-DD` is sent. Use an `int` of epoch milliseconds for instants.
18. Constructing `Rainbird()` (which demands a key) just to call `version()` for a health check: `engine_version()` / `async_engine_version()` need no key.
19. Retrying `POST` on 502/504 yourself, or setting `post_retry_statuses` to include them, without accepting that the answer may be applied twice (section 11).

## 14. References

- Rainbird docs index for LLMs: https://docs.rainbird.ai/llms.txt
- Developer docs: https://docs.rainbird.ai/rainbird/developer-docs/overview
- API request flow: https://docs.rainbird.ai/rainbird/developer-docs/api-guide/api-request-flow
- Example API requests: https://docs.rainbird.ai/rainbird/developer-docs/api-guide/example-api-requests
- API keys / publishing: https://docs.rainbird.ai/rainbird/knowledge-modelling/publishing-your-km/api-management
- Knowledge modelling (RBLang, concepts, rules): https://docs.rainbird.ai/rainbird/knowledge-modelling/overview
- OpenAPI 3.1 reference (rendered spec, downloadable): https://rainbird.redoc.ly/
- SDK source of truth: `src/rainbird_ai/client.py` (sync), `aio.py` (async), `models.py`, `_answers.py` (answer rules), `errors.py`.
