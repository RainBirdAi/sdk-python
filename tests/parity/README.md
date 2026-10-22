# Parity tests: the Python SDK driven by the Go and JavaScript SDK test suites

## Purpose

The Rainbird REST API has three first-party SDKs: Go (`gitlab.com/rainbird-ai/sdk-go`),
JavaScript (`@rainbird/sdk`, the `packages/base` package of the JS monorepo) and this Python
package (`rainbird-ai`). The Go and JS repositories ship unit tests that run each SDK call
against a fake HTTP server (`httptest.NewServer` in Go, a mocked `global.fetch` in JS) and
assert on the exact request sent and the value returned.

This directory ports every one of those cases to pytest so that the Python SDK is exercised
by the same inputs, canned server bodies and request expectations as its siblings. Where the
Python SDK deliberately behaves differently (header names, payload keys, typed exceptions,
typed models) the test asserts Python's real behaviour and the divergence is marked in the
source with a `# DIFFERENCE(go)` / `# DIFFERENCE(js)` comment, collected in
[Behavioural differences](#behavioural-differences) below. Where the sibling behaviour looks
like something Python should also do but does not, the test asserts the sibling behaviour and
is marked `xfail(strict=True)`, collected in
[Possible gaps or bugs in the Python SDK](#possible-gaps-or-bugs-in-the-python-sdk).

The port itself changed nothing in `src/`. The gaps it surfaced were then fixed in the SDK
(shipped as 0.2.0): the former strict xfails now pass as ordinary
tests and the entries below are marked **fixed**; the remaining entries are still open.

## How to run

```bash
.venv/bin/python -m pytest tests/parity -q      # from the repository root
```

Current result: **180 tests, 180 passed, 0 xfailed**, no network access. The parity tests
are part of the normal suite (`.venv/bin/python -m pytest -q` from the repo root runs them
too) and the root `tests/conftest.py` fixtures (`_clean_env` autouse, which removes every
`RAINBIRD_*` variable) apply to them.

To see the xfail reasons:

```bash
.venv/bin/python -m pytest tests/parity -q -rx
```

## Layout

| file | ported from |
|------|-------------|
| `support.py` | shared helper (see below) |
| `test_go_client.py` | Go `client_test.go`: `TestClientNewSessionValidation`, `TestClientNewSessionStartCalls`, `TestClientHTTP`, `TestClientVersion` |
| `test_go_evidence.py` | Go `client_test.go`: `TestEvidence` |
| `test_go_interactions.py` | Go `client_test.go`: `TestInteractionLog` |
| `test_go_session_analysis.py` | Go `client_test.go`: `TestSessionKmVersion`, `TestSessionFacts` |
| `test_go_inject_query.py` | Go `session_test.go`: `TestSessionInject`, `TestSessionQuery` |
| `test_go_response_undo.py` | Go `session_test.go`: `TestSessionResponse`, `TestSessionUndo` |
| `test_go_stringers.py` | Go `answer_test.go` (`TestAnswerToString`), `question_test.go` (`TestQuestionToString`) |
| `test_js_start.py` | JS `__tests__/start.test.js`, `index.test.js` (constants) |
| `test_js_inject.py` | JS `__tests__/inject.test.js` |
| `test_js_query.py` | JS `__tests__/query.test.js`, `response_types.test.js` |
| `test_js_response.py` | JS `__tests__/response.test.js` |
| `test_js_undo.py` | JS `__tests__/undo.test.js` |
| `test_js_evidence.py` | JS `__tests__/evidence.test.js`, `full_evidence.test.js` |
| `test_js_analysis.py` | JS `__tests__/interaction_log.test.js`, `session.test.js` (`knowledgeMap`, `facts`) |

### The helper (`support.py`)

`make_client(handler, api_key=..., base_url=..., **rainbird_kwargs) -> (Rainbird, Recorder)`
builds a `Rainbird` whose `httpx.Client` uses an `httpx.MockTransport` running `handler`
(`httpx.Request -> httpx.Response`) and records every request as a `Call` (`.method`,
`.path`, `.query`, `.headers` (case-insensitive), `.json`, `.content`). `Recorder.last` /
`Recorder.calls[i]` / `len(recorder)` give the Go `expectCalls` and jest
`toHaveBeenCalledTimes` assertions. `respond(status, body=None, text=None, headers=None)`
is the one-line constant handler; `raise_transport_error(exc)` makes the transport fail (jest's
rejecting `fetch`). Retries are disabled (`retry=NO_RETRY`) unless a test overrides it, because
the Go and JS SDKs never retry and Python's default `RetryPolicy` would otherwise turn a
single GET 500 into three requests. Defaults: `api_key="test-api-key"`,
`base_url="https://api.test"`. `make_async_client` is the async twin (currently unused, the
sibling suites have no async cases).

### Status labels used in the tables

- **ported**: same inputs, same assertions, same outcome.
- **adapted**: same inputs and canned bodies; some assertions rewritten to Python's real
  behaviour (typed exception instead of an error string, `X-API-Key` instead of
  `Authorization`, model attribute instead of raw JSON, ...). Every such rewrite carries a
  `DIFFERENCE` comment in the test.
- **xfail**: the test asserts the sibling behaviour, Python does not have it;
  `@pytest.mark.xfail(strict=True)` so it flips to a failure the day the SDK changes. All ten
  original xfails flipped with the 0.2.0 fixes and are now labelled "(fixed)" in the tables.
- **extra**: no sibling table entry; derived from the sibling implementation or added to
  cover a Python-only feature that the sibling case touches (client-wide keys, `resume()`,
  `engine_version()`).

Line numbers below refer to the checked-out sibling sources at the time of the port
(`/tmp/rb-sdk-go`, `/tmp/rb-sdk-js/packages/base/lib`). Those checkouts were Go SDK
`v1.9.0` and JS `@rainbird/sdk` `1.5.9`; the same versions are available offline in the Go
module cache (`$(go env GOMODCACHE)/gitlab.com/rainbird-ai/sdk-go@v1.9.0`) and in any
`node_modules/@rainbird/sdk/lib/__tests__` installed from npm (the package ships its tests),
and every line number in the tables was re-checked against them on 2026-09-22.

## Mapping tables

### Go `client_test.go` lines 1-326 -> `test_go_client.py` (20 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `TestClientNewSessionValidation/"Missing API Key"` (:28) | `test_new_session_validation_missing_api_key` | adapted | Go: `ErrClientMissingAPIKey` from `NewSession()`. Python: `RainbirdConfigurationError` from `Rainbird(api_key=None|"")`, so `start()` is never reached; no request in either. |
| `"Missing EnvironmentURL"` (:36) | `test_new_session_validation_missing_environment_url` | adapted | Go: `ErrClientMissingEnvironmentURL`. Python has no such error: `base_url=None|""` falls back to `RAINBIRD_API_URL` then `https://api.rainbird.ai` (= Go `EnvCommunity`) and the request is sent there. |
| `"Missing KMID"` (:44) | `test_new_session_validation_missing_kmid` | adapted | Go: sentinel `ErrNewSessionInvalidKMID`. Python: `RainbirdConfigurationError("km_id is required ...")` for `""`, `None` (after `RAINBIRD_KM_ID`) and whitespace; 0 requests. |
| `TestClientNewSessionStartCalls/"Missing API Key"` (:83) | `test_start_calls_missing_api_key` | adapted | Fails at construction; handler `pytest.fail`s if contacted (Go `expectCalls 0`). |
| `"Missing KMID"` (:88) | `test_start_calls_missing_kmid` | adapted | `start("")` raises before any request. |
| `"Simple success"` (:93) | `test_start_calls_simple_success` | adapted | GET `/start/<kmid>`, `Accept: application/json`, empty body, `{"id": "success-id"}` -> `session.id`. Go asserts `Authorization: Basic ...`; Python sends `X-API-Key` and no `Authorization`. |
| `"Correctly request context"` (:107) | `test_start_calls_correctly_request_context` | adapted | `?contextid=foo` identical to Go; auth header differs. |
| `"Handle 400"` (:122) | `test_start_calls_handle_400` | adapted | Go: `"API returned an error 400: Bad request"`. Python: `BadRequestError` (`.status_code`, `.message`, `.body`, `.errors`, `.method`, `.url`). |
| `"Handle good code but no ID"` (:136) | `test_start_calls_handle_good_code_but_no_id` | adapted | Go: `"API returned no error but no ID either"`. Python: `UnexpectedResponseError("start response did not include a session id")`. |
| `"Success with useDraft"` (:150) | `test_start_calls_success_with_use_draft` | adapted | `?useDraft=true` identical; auth header differs. |
| `"Success with version"` (:165) | `test_start_calls_success_with_version` | adapted | `?version=1` identical; auth header differs. |
| `client.go:126-136` URL building | `test_start_combined_params_build_one_query_string` | extra | Go appends every parameter with its own `?` (`?contextid=foo?useDraft=true`); Python builds one `&`-joined query. |
| `client.go:130-136` useDraft + version | `test_start_rejects_use_draft_together_with_version` | extra | Python rejects `use_draft` with `version`, and `version < 1`, locally (`RainbirdConfigurationError`, 0 requests); Go sends whatever it is given. |
| `TestClientHTTP/"Default"` (:238) | `test_http_default` | adapted | Go: shared `http.DefaultClient`. Python: private per-instance `httpx.Client(timeout=30, follow_redirects=False)`, closed by `close()`. |
| `TestClientHTTP/"Custom"` (:243) | `test_http_custom` | ported | Plus: a user-supplied client is never closed by the SDK. |
| `TestClientVersion/"Simple success"` (:273) | `test_version_simple_success` | adapted | Go's keyless `Version()` sets no headers; `Rainbird.version()` needs a key and sends `X-API-Key` + `Accept: text/plain, application/json, */*`. |
| `TestClientVersion/"Error code"` (:284) | `test_version_error_code` | adapted | Go: `"API returned error code 500: Foobarbaz"`. Python: `ServerError`, exactly one call (NO_RETRY). |
| `TestClientVersion/"Simple success"` (keyless twin) | `test_engine_version_simple_success` | ported | `engine_version(BASE_URL, http_client=...)`: no `X-API-Key`, no `Authorization`, one GET; the closest match to Go's keyless `Version()`. |
| `TestClientVersion/"Error code"` (keyless twin) | `test_engine_version_error_code` | adapted | `engine_version()` has no `retry=` parameter: default policy retries the GET 500 twice (3 calls, `time.sleep` patched) before `ServerError`. |
| `client.go:188 ResumeSession` | `test_resume_session` | extra | Both send nothing; Go accepts `""`, Python `resume("")` raises `RainbirdConfigurationError`. |

### Go `client_test.go` lines 327-546 (`TestEvidence`) -> `test_go_evidence.py` (14 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `"Bad request"` (:340) | `test_bad_request` | adapted | Go: `"API returned error 400: Foo bar baz"`. Python: `BadRequestError`. Go asserts the literal path `/analysis/evidence/WA:RF:1234/1234`; Python percent-encodes (`WA%3ARF%3A1234`), decoded path identical. Go sends `Content-Type` and no auth on the GET; Python sends `X-API-Key` + `Accept`. |
| `"Internal server error"` (:347) | `test_internal_server_error` | adapted | `ServerError`, one request (NO_RETRY). |
| `"Returns correct evidence for rule"` (:354) | `test_returns_correct_evidence_for_rule` | adapted | Every Go expected field asserted under its Python name (`Fact.ID` -> `fact_id`, `ConceptInstance.Type/Value/DataType` -> `FactValue.concept/value/data_type`, `Conditioner.Type()` -> `isinstance(Condition|ExpressionCondition)`, bindings `100.0` -> `100`). `evidence.fact.source is FactSource.RULE` (the envelope source is copied onto the fact, like its id). |
| `"Returns correct evidence for rule"` / `"... for datasource"` (`Fact.Source` only) | `test_fact_source_is_propagated_to_fact[rule|datasource]` | ported (fixed) | Go copies the envelope `source` onto `Fact.Source`; Python's `Evidence.from_dict` does the same since 0.2.0 (it was a strict xfail before). |
| `"Returns correct evidence for datasource"` (:470) | `test_returns_correct_evidence_for_datasource` | adapted | `rule is None`, `is_inferred False`, `child_fact_ids []`, object value stays `"290.00"`. |
| `client.go:252-254` 404 -> `ErrFactNotFound` | `test_not_found_with_body_maps_to_not_found_error` | extra | Go discards the body for the sentinel; Python `NotFoundError` keeps `"Foo bar baz"` as message/body. |
| `client.go:252-254` 404, empty body | `test_not_found_with_empty_body_maps_to_not_found_error` | extra | Python falls back to the fixed message `"The resource (session, knowledge map or fact) could not be found"`, `errors == []`, `body None`. |
| `client.go:236-238` `x-evidence-key` set | `test_evidence_key_header_is_sent` | extra | `rb.evidence(..., evidence_key="secret")` sends the header; full Go request assertion set. |
| same, via `Session` | `test_evidence_key_header_is_sent_by_session` | extra | `rb.resume(sid).evidence(fid, evidence_key=...)`, `resume` sends nothing. Go has only the client-level call. |
| `client.go:236-238` nil key | `test_evidence_key_header_is_absent_when_not_given[none|empty]` | extra | With no client-wide default, `None` and `""` both omit the header. |
| client-wide key (Python only) | `test_client_wide_evidence_key_is_default_and_per_call_wins` | extra | `Rainbird(evidence_key=...)` is the default; per-call `None` -> default, non-empty -> wins, `""` -> no header despite the default. |
| `evidence.go:154-177` `asConditions` | `test_unsupported_condition_type_is_kept_not_rejected` | extra | Go fails the call with `"unsupported condition type"`; Python keeps a `Condition(relationship="")` with `raw` intact. A condition with both `relationship` and `expression` is a relationship condition in Go, an `ExpressionCondition` in Python. |

### Go `client_test.go` lines 548-939 (`TestInteractionLog`) -> `test_go_interactions.py` (13 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `"Correct interaction log for Start returned"` (:560) | `test_correct_interaction_log_for_start_returned` | adapted | Go: int enum `StartEvent` + typed `Start{}`. Python: `event == "start"`, `created` aware UTC datetime, fields via `values["start"][...]`. Go asserts the bare RequestURI `/analysis/interactions/1234-5678`; Python appends `?format=json`. |
| `"... Query, Questions returned"` (:589) | `test_correct_interaction_log_for_query_questions_returned` | adapted | Dict access plus `Question.from_dict` on each entry for the typed view. |
| `"... Injects returned"` (:683) | `test_correct_interaction_log_for_injects_returned` | adapted | Logged `cf` stays the string the API sent; `certainty`/`metadata` absent. No typed model for logged facts in Python. |
| `"... Answers returned"` (:742) | `test_correct_interaction_log_for_answers_returned` | adapted | `answer` present only on the first; `certainty`/`unanswered` absent (Go zero values). |
| `"... Datasources, Result returned"` (:805) | `test_correct_interaction_log_for_datasources_result_returned` | adapted | Results via `ResultItem.from_dict`; `str()` is `"Dan lives in England (100%)"` vs Go `"Dan - lives in - England [100%]"` (Go pads to 3 columns: `[ 90%]`). |
| `"Bad request"` (:894) | `test_bad_request` | adapted | `BadRequestError`, url includes `?format=json`. |
| `"Internal server error"` (:901) | `test_internal_server_error` | adapted | `ServerError`, one request (NO_RETRY; the default policy would make three). |
| `client.go:288-290` key set / nil | `test_x_interaction_key_header_only_when_given` | extra | Header present when given, absent for `None`; plus `Rainbird(interaction_key=...)` default and per-call override. |
| `client.go:288-290` pointer to `""` | `test_empty_interaction_key_sends_no_header` | extra | Go would send an empty header; Python sends none and `""` also suppresses the client default. |
| `client.go:287` request headers | `test_request_headers_differ_from_go` | extra | Go: `Content-Type`, no credentials. Python: `X-API-Key`, `Accept`, `User-Agent`, no `Content-Type`. |
| `interaction.go:75-77` unknown event | `test_unknown_event_name_is_kept_not_rejected` | extra | Go fails the call (`"unsupported event: ..."`); Python keeps the entry. |
| `interaction.go:203-208` `parseEventTime` | `test_unparseable_created_yields_none[garbage|empty]` | extra | Go returns the parse error; Python yields `created=None`, original text on `.raw`. |

### Go `client_test.go` lines 940-1251 (`TestSessionKmVersion`, `TestSessionFacts`) -> `test_go_session_analysis.py` (6 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `TestSessionKmVersion/"Bad request"` (:952) | `test_session_km_version_bad_request` | adapted | `rb.resume("1234").knowledge_map()` -> `BadRequestError`; GET `/analysis/session/1234?filter=version`, `X-API-Key` (Go: Basic auth). |
| `"Internal server error"` (:959) | `test_session_km_version_internal_server_error` | adapted | `ServerError`, one request. |
| `"Returns version info"` (:966) | `test_session_km_version_returns_version_info` | adapted | Same values as `expectKm`; `version_created` is a tz-aware datetime; `version_number` is derived from `versionStatus` (`None` for Draft, Go `VersionNumber` nil), `is_draft True`. |
| `TestSessionFacts/"Bad request"` (:1028) | `test_session_facts_bad_request` | adapted | `session.facts()` -> `BadRequestError`; query exactly `{filter: facts}`. |
| `"Internal server error"` (:1035) | `test_session_facts_internal_server_error` | adapted | `ServerError`. |
| `"Returns facts info"` (:1042) | `test_session_facts_returns_facts_info` | adapted | Five local facts field by field; Go `Flatten()` -> `.all`; certainty pinned as a JSON integer on the wire and `int` on the model (Go `Certainty int`); `source` is the `FactSource` str-enum; JSON ints stay `int` (Go float64). |

### Go `session_test.go` lines 1-345 (`TestSessionInject`, `TestSessionQuery`) -> `test_go_inject_query.py` (13 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `TestSessionInject/Empty` (:36) | `test_inject_empty` | adapted | Go POSTs `[]` (2 calls). Python raises `InvalidFactError("facts must not be empty")` locally (1 call, the `/start`). |
| `Some facts` (:44), original certainties | `test_inject_some_facts_original_certainties_are_rejected` | adapted | Go sends `cf: "121"/"122"` opaquely; Python validates 0-100 and raises `InvalidFactError`, no request. |
| `Some facts` (:44) | `test_inject_some_facts` | adapted | Certainties brought to 21/22, server answers `{"result": "OK"}`. Go body: `"cf": "121"` (string) + `"metadata": {}` per fact; Python: numeric `"certainty"`, no `cf`, no `metadata`. |
| `Some facts` (:44), bare 200 | `test_inject_some_facts_empty_200_body` | adapted | Go treats a bodiless 200 as success; Python raises `UnexpectedResponseError(status_code=200, body="")` after sending (2 calls). |
| `Some facts with metadata` (:65) | `test_inject_some_facts_with_metadata` | adapted (fixed) | Go sends `metadata.source` (`"system"`/`"user"`) per fact; Python forwards a fact's `metadata` dict since 0.2.0 (a strict xfail before; certainties kept in range and spelled `certainty`). |
| `Bad request` 400 (:92) | `test_inject_bad_request` | adapted | Go POSTs `[]`; Python needs one fact to reach the server. `BadRequestError`, `body None`, `errors []`, default message. |
| `Bad request` 500 (:100) | `test_inject_bad_request_internal_server_error` | adapted | `ServerError("Rainbird server error")`, one inject request. |
| `TestSessionQuery/Empty` (:184) | `test_query_empty` | adapted | Go: `ErrQueryBlankRelationship`. Python: `RainbirdConfigurationError("relationship must not be empty")`; no request in either. |
| `John - lives in - ? leading to answer` (:193) | `test_query_john_lives_in_leading_to_answer` | adapted | Go body `{"subject":"John","relationship":"lives in","object":""}`; Python omits `object` when `""`/`None`. Go returns `([]Question{}, []Answer, nil)`; Python one `ResultResponse`. |
| `Handle bad request` (:224) | `test_query_handle_bad_request` | adapted | `BadRequestError`, `str()` `"HTTP 400 calling POST https://api.test/success-id/query: Foo bar baz"`. |
| `Handle internal server error` (:239) | `test_query_handle_internal_server_error` | adapted | Go error text carries the raw body; Python parses the JSON body: `message "some JSON returned"`, `body {"error": ...}`. |
| `check the optional headers propagate` (:254) | `test_query_check_the_optional_headers_propagate` | adapted | Go `AddHeaders(...)` on one `Query` call; Python has no per-call header option, so `Rainbird(default_headers={"x-api-source": "natlang"})` is used (also present on `/start`). |

### Go `session_test.go` lines 346-743 (`TestSessionResponse`, `TestSessionUndo`) -> `test_go_response_undo.py` (8 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `"Simple response that returns a question"` (:361) | `test_response_simple_response_that_returns_a_question` | adapted | Go body `"cf": "100"` (string); Python `"certainty": 100`, same key order, no `cf`. Go returns `([]Question, nil, nil)`; Python `QuestionResponse`. Also asserts `Answer(...).to_payload()`. |
| `"... returns extra questions"` (:408) | `test_response_simple_response_that_returns_extra_questions` | adapted | Main question first, then `extraQuestions`, via `.questions` / `.question` / `.extra_questions`. |
| `"... question with float object"` (:481) | `test_response_simple_response_that_returns_a_question_with_float_object` | adapted | Go marshals `float64(12)` as `12`; Python writes `12.0` on the wire (both decode to 12). Parsed `question.object` is `"12"` in both. |
| `"Bad request"` (:530) | `test_response_bad_request` | adapted | `BadRequestError`, session state untouched. |
| `"Internal server error"` (:549) | `test_response_internal_server_error` | adapted | `ServerError`, POST not retried, two calls. |
| `TestSessionUndo/"Success back to question"` (:635) | `test_undo_success_back_to_question` | adapted | Body exactly `{}` (raw bytes). Go `Undo` sends no `Accept`; Python sends `Accept: application/json`. |
| `"Bad request"` (:672) | `test_undo_bad_request` | adapted | `BadRequestError`, body `{}` asserted. |
| `"Internal server error"` (:681) | `test_undo_internal_server_error` | adapted | `ServerError`. |

### Go `answer_test.go`, `question_test.go` -> `test_go_stringers.py` (14 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `TestAnswerToString/"Empty"` (:16) | `test_answer_to_string_empty` | adapted | Go: `" -  -  [  0%]"`. Python `ResultItem.from_dict({})`: `"None  None"` (no certainty appended when absent); zero-like payload: `"? ? (0%)"`. |
| `"John speaks English 50%"` (:21) | `test_answer_to_string_john_speaks_english_50` | adapted | Go `"John - speaks - English [ 50%]"`; Python `"John speaks English (50%)"`. |
| `"Dan speaks German 100%"` (:31) | `test_answer_to_string_dan_speaks_german_100` | adapted | Python `"Dan speaks German (100%)"`. |
| `"John favourite number 137 35%"` (:41) | `test_answer_to_string_john_favourite_number_137_35` | adapted | Python `"John favourite number 137 (35%)"`; numeric object kept as `int`. |
| `answer.go:26` nil object | `test_answer_to_string_missing_parts_do_not_render_none` | adapted (fixed) | Go prints `""` for a nil object; Python prints `?` since 0.2.0 (it interpolated the literal `None` before). |
| `TestQuestionToString/"Empty"` (:16) | `test_question_to_string_empty` | adapted | Go `" (? -  - ?)"`; Python with empty prompt falls back to the triple: `"? ?"`. |
| `"Unknown object"` (:21) | `test_question_to_string_unknown_object` | adapted | Go `"Which language does John speak? (John - speaks - ?)"`; Python is the prompt alone. |
| `"Unknown subject"` (:30) | `test_question_to_string_unknown_subject` | adapted | Prompt alone; prompt-less fallback `"? lives in England"`. |
| `"Unknown object as empty string"` (:39) | `test_question_to_string_unknown_object_as_empty_string` | adapted | Go normalises `""` to `?`; Python keeps `question.object == ""`. |
| `question.go:69-80` `?` placeholders | `test_question_to_string_missing_parts_do_not_render_none[empty|unknown-object|unknown-subject|unknown-object-as-empty-string]` | adapted (fixed) | Go prints `?` for an unknown subject/object; Python does too since 0.2.0 (strict xfail before). |
| `question.go:80` non-string object | `test_question_to_string_numeric_object_is_kept` | extra | Go's `objectStr()` renders a non-string object (137) as `""` (a Go quirk); Python keeps `137`. |

### JS `start.test.js`, `index.test.js` -> `test_js_start.py` (13 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should call appropriate endpoint and parse response` (:11) | `test_should_call_appropriate_endpoint_and_parse_response` | adapted | JS: `fetch("thisistheurl/start/thisisthekmid", {headers: {Content-Type, Authorization: Basic ...}})` -> `{sessionID, kmVersionID}`. Python: GET `https://thisistheurl/start/thisisthekmid` (bare host normalised), `X-API-Key`, no `Authorization`, no `Content-Type` on the GET; returns a `Session` (`.id`, `.km_version_id`, `.start_response`). |
| `should handle engine option` (:38) | `test_should_handle_engine_option` | adapted | JS per-call `{engine}`; Python `Rainbird(engine=...)`, `x-rainbird-engine` asserted on the request. |
| `should handle context option` (:71) | `test_should_handle_context_option` | adapted | JS `?contextId=thisisthecontext&useDraft=true`; Python `?useDraft=true&contextid=thisisthecontext` (lower-case key, matches Go/API docs). |
| `should handle useDraft option` (:103) | `test_should_handle_useDraft_option` | adapted | `?useDraft=true` identical. |
| `should handle context and useDraft option` (:135) | `test_should_handle_context_and_useDraft_option` | adapted | The JS case passes only `context`; `contextId` vs `contextid`. |
| `should throw fetch-generated errors` (:167) | `test_should_throw_fetch_generated_errors` | adapted | JS re-throws the fetch error; Python `RainbirdConnectionError("could not reach GET <url>: This was generated by fetch")`, `__cause__` set, `.url` without query string. |
| `should handle 404` (:179) | `test_should_handle_404` | adapted | JS `Error("API error code: 404")`; Python `NotFoundError`, default message, `body {}`, `errors []`. |
| `should handle 500` (:193) | `test_should_handle_500` | adapted | `ServerError`, one request (NO_RETRY). |
| `should handle 400` (:207) | `test_should_handle_400` | adapted | `BadRequestError`. |
| `index.test.js` `URL_COMMUNITY` (:12) | `test_should_export_constant_URL_COMMUNITY` | adapted | JS bare string; Python `Environment.COMMUNITY` (str enum) and `DEFAULT_BASE_URL`, both in `__all__`. |
| `URL_ENTERPRISE` (:12) | `test_should_export_constant_URL_ENTERPRISE` | adapted | `Environment.ENTERPRISE == "https://enterprise-api.rainbird.ai"`. |
| `ENGINE_CORE` (:12) | `test_should_export_constant_ENGINE_CORE` | ported (fixed) | JS exports `ENGINE_CORE = "Core"`; `rainbird_ai.ENGINE_CORE` has the same value since 0.2.0 (strict xfail before). |
| `ENGINE_BETA` (:12) | `test_should_export_constant_ENGINE_BETA` | ported (fixed) | JS exports `ENGINE_BETA = "Experimental (Beta)"`; `rainbird_ai.ENGINE_BETA` matches since 0.2.0 and is asserted on the `x-rainbird-engine` header. |
| `index.ts:9-13` `RESPONSE_TYPE_*` re-exports | (none) | not applicable | See [Not ported](#not-ported). |

### JS `inject.test.js` -> `test_js_inject.py` (11 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should assert a new fact within a session` (:9) | `test_should_assert_a_new_fact_within_a_session` | adapted | JS body carries `"cf": "99"` verbatim; Python sends `"certainty": 99`. Returns `InjectResponse` (`.raw == {"response": {"ok": true}}`, `.result None`, `.ok True`). |
| `should assert facts within a session` (:41) | `test_should_assert_facts_within_a_session` | adapted | Five facts, one request; `cf` strings 98..94 -> numeric `certainty`. |
| `should assert a facts for specific engine within a session` (:121) | `test_should_assert_a_facts_for_specific_engine_within_a_session` | adapted | `Rainbird(engine="thisIsTheEngine")` -> `x-rainbird-engine`. |
| `should throw fetch-generated errors` (:161) | `test_should_throw_fetch_generated_errors[connect-error|remote-protocol-error|read-timeout]` | adapted | JS one generic error; Python maps `ConnectError`/`RemoteProtocolError` -> `RainbirdConnectionError`, `ReadTimeout` -> `RainbirdTimeoutError`. JS sends `[]`; Python rejects it locally, so one fact is used. |
| `should handle 404` (:171) | `test_should_handle_404` | adapted | `NotFoundError`; `[]` rejected locally first (asserted). |
| `should handle 500` (:183) | `test_should_handle_500` | adapted | `ServerError`. |
| `should handle 400` (:195) | `test_should_handle_400` | adapted | `BadRequestError`. |
| `inject.ts:8` `InjectResponse {result, warningLog?}` | `test_inject_response_interface_result_and_warning_log` | extra | `{"result": "OK", "warningLog": [...]}` -> `.result "OK"`, `.ok True`; `warningLog` only on `.raw`. |
| `inject.test.js:13` canned body | `test_inject_response_ok_ignores_nested_ok` | extra | `{"response": {"ok": false}}` still gives `.ok True` (only top-level `result`/`ok`/`status` are inspected); pinned as current behaviour. |

### JS `query.test.js`, `response_types.test.js` -> `test_js_query.py` (14 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should make a simple query for object` (:15) | `test_query_should_make_a_simple_query_for_object` | adapted | JS actually sends `"object": null`; Python omits `None`. `ResultResponse` instead of `{type, data}`; the non-list `result: {foo: bar}` is wrapped into one empty `ResultItem`. |
| `should make a simple query for subject` (:46) | `test_query_should_make_a_simple_query_for_subject` | adapted | `"subject": null` omitted by Python. |
| `should make a simple query with both subject and object` (:77) | `test_query_should_make_a_simple_query_with_both_subject_and_object` | adapted | Exact body asserted. |
| `should handle a question response` (:109) | `test_query_should_handle_a_question_response` | adapted | `QuestionResponse` wrapping `Question` models; session records it as pending. |
| `should make a query for specific engine` (:144) | `test_query_should_make_a_query_for_specific_engine` | adapted | `Rainbird(engine=...)`. |
| `should handle a question response with extra questions for grouped relationships` (:179) | `test_query_should_handle_a_question_response_with_extra_questions_for_grouped_relationships` | adapted | `.extra_questions` / `.questions` as `Question` models, raw dicts preserved. |
| `should throw fetch-generated errors` (:222) | `test_query_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError`, `__cause__` set. |
| `should handle 404` (:238) | `test_query_should_handle_404` | adapted | `NotFoundError`. |
| `should handle 500` (:256) | `test_query_should_handle_500` | adapted | `ServerError`, one attempt. |
| `should handle 400` (:274) | `test_query_should_handle_400` | adapted | `BadRequestError`. |
| `response_types.test.js` `should handle a question` (:12) | `test_parse_should_handle_a_question` | adapted | `parse_query_response` -> `QuestionResponse`, `extra_questions == []`. |
| `should handle extra questions` (:19) | `test_parse_should_handle_extra_questions` | adapted | `Question` models with `.raw` equal to the originals. |
| `should handle a result` (:28) | `test_parse_should_handle_a_result` | adapted | JS `data == {foo: bar}`; Python `ResultResponse` with one empty `ResultItem`. |
| `should handle a question` (duplicate name, `RESPONSE_TYPE_UNKNOWN`) (:35) | `test_parse_should_handle_a_question_unknown_body` | adapted | JS returns `{type: UNKNOWN, data}`; `parse_query_response` raises `UnexpectedResponseError` with the body (a bare `ValueError` before 0.2.0); `Session.query` re-raises it with the HTTP status. |

### JS `response.test.js` -> `test_js_response.py` (12 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should make a response call containing a single answer` (:15) | `test_should_make_a_response_call_containing_a_single_answer` | adapted | Python adds the protocol default `"certainty": 100`; `ResultResponse` recorded on `session.result`. |
| `should handle a question response` (:51) | `test_should_handle_a_question_response` | adapted | `QuestionResponse`, `session.pending` set. |
| `should pass certainty when specified` (:87) | `test_should_pass_certainty_when_specified` | adapted | Body identical to JS (`certainty: 80`). Python-only sub-case: `"cf": 80` on input is sent as `"certainty": 80`. |
| `should pass yes as an answer` (:126) | `test_should_pass_yes_as_an_answer` | adapted | Body identical; base URL `"url"` normalised to `https://url`. |
| `should pass no as an answer` (:155) | `test_should_pass_no_as_an_answer` | adapted | Body identical. |
| `should make a response call containing multiple answers` (:185) | `test_should_make_a_response_call_containing_multiple_answers` | adapted | Both answers in order, each with `certainty: 100`. |
| `should make a response call for a specific engine` (:231) | `test_should_make_a_response_call_for_a_specific_engine` | adapted | `Rainbird(engine=...)`. |
| `should handle a question response with extra questions for grouped relationships` (:275) | `test_should_handle_a_question_response_with_extra_questions_for_grouped_relationships` | adapted | `questions == [main, *extra]` (4). |
| `should throw fetch-generated errors` (:323) | `test_should_throw_fetch_generated_errors` | adapted | JS sends `[]`; Python raises `InvalidFactError("answers must not be empty")` locally, so one answer is used; `RainbirdConnectionError`; session state untouched. |
| `should handle 404` (:333) | `test_should_handle_404` | adapted | `NotFoundError`. |
| `should handle 500` (:345) | `test_should_handle_500` | adapted | `ServerError`, one request. |
| `should handle 400` (:357) | `test_should_handle_400` | adapted | `BadRequestError`. |

### JS `undo.test.js` -> `test_js_undo.py` (8 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should make a undo call and parse result` (:13) | `test_should_make_a_undo_call_and_parse_result` | adapted | JS posts `{"data": {}}`; Python posts `{}` (`content == b"{}"`). Dict `result` wrapped into one empty `ResultItem`. |
| `should make a undo call for a specific engine` (:33) | `test_should_make_a_undo_call_for_a_specific_engine` | adapted | `Rainbird(engine=...)`. |
| `should handle a question response` (:62) | `test_should_handle_a_question_response` | adapted | `QuestionResponse`; `session.pending` set, `session.result None`. |
| `should handle a question response with extra questions for grouped relationships` (:88) | `test_should_handle_a_question_response_with_extra_questions_for_grouped_relationships` | adapted | Body copied verbatim; `Question` models. |
| `should throw fetch-generated errors` (:128) | `test_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError`. |
| `should handle 404` (:138) | `test_should_handle_404` | adapted | `NotFoundError`. |
| `should handle 500` (:150) | `test_should_handle_500` | adapted | `ServerError`. |
| `should handle 400` (:162) | `test_should_handle_400` | adapted | `BadRequestError`. |

### JS `evidence.test.js`, `full_evidence.test.js` -> `test_js_evidence.py` (13 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should perform simple GET for evidence` (:11) | `test_should_perform_simple_get_for_evidence` | adapted | JS returns `{result: {foo: "bar"}}` untouched; Python returns an `Evidence` whose `.raw` equals it and parses it as an empty node (`fact_id ""`, `rule None`). `Session.evidence` sends the identical request. |
| `should GET evidence for alternate engines` (:35) | `test_should_get_evidence_for_alternate_engines` | adapted | `Rainbird(engine=rainbird_ai.ENGINE_BETA)`, the same string as JS `ENGINE_BETA`. |
| `should throw fetch-generated errors` (:65) | `test_should_throw_fetch_generated_errors` | adapted | JS passes `{}` as fact id (quirk); a string is used. `RainbirdConnectionError`. |
| `should handle 404` (:75) | `test_should_handle_404` | adapted | `NotFoundError`. |
| `should handle 500` (:87) | `test_should_handle_500` | adapted | `ServerError`, one request. |
| `should handle 400` (:99) | `test_should_handle_400` | adapted | `BadRequestError`; session id `"thisisthekey"` kept verbatim. |
| `should call the evidence endpoint with the evidence key header` (:111) | `test_should_call_the_evidence_endpoint_with_the_evidence_key_header` | adapted | `x-evidence-key` sent exactly as JS; client and session entry points. |
| (no JS counterpart) | `test_evidence_key_client_default_and_per_call_override` | extra | `Rainbird(evidence_key=...)` default, per-call override, `""` suppresses. |
| `full_evidence.test.js` `should perform a simple fullEvidence retrieval` (:623) | `test_should_perform_a_simple_fullevidence_retrieval` | adapted | JS: 10 fetches, bodies served by `json()` call order, falsy leaf bodies dropped, 4 nodes. Python cannot skip a node, so the walk is bounded with `evidence_tree(fid, max_depth=1)`: 4 GETs in BFS order, node set equals `normalResp`, root first (JS Map order differs). |
| same (the six leaf fetches) | `test_fullevidence_unbounded_walk_cannot_skip_null_leaf_bodies` | extra | Unbounded `evidence_tree()` raises `UnexpectedResponseError` on the first `200 null` leaf after 5 requests; `iter_evidence_tree` yields the 4 canned nodes first. |
| `should include fact that does not have a rule` (:646) | `test_should_include_fact_that_does_not_have_a_rule` | adapted | JS expects 5 nodes because `full_evidence.ts` keeps a module-level `Map` that leaks across tests; Python returns exactly `[noRule]`, 1 GET, no cache (second call re-fetches). |
| `should throw fetch-generated errors` (:667) | `test_fullevidence_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError` from the root fetch. |
| `should handle 404` (:676) | `test_fullevidence_should_handle_404` | adapted | `NotFoundError` from the root fetch. |

### JS `interaction_log.test.js`, `session.test.js` -> `test_js_analysis.py` (21 tests)

| source case | Python test | status | note |
|---|---|---|---|
| `should perform simple GET for interactionLog` (:11) | `test_should_perform_simple_get_for_interaction_log` | adapted | JS resolves `{result: {foo: "bar"}}` untouched; Python's `interactions()` validates the shape and raises `UnexpectedResponseError("expected a list of interaction events")`, `.body` == that JSON, `.status_code 200`. Python adds `?format=json`, sends `X-API-Key`, no `Content-Type`. |
| same (companion) | `test_should_perform_simple_get_for_interaction_log_parses_events` | extra | Body in the list-of-events shape Python expects: `list[InteractionEvent]`, `.event`, `.created`, `.values`, `to_dict()`. |
| `should throw fetch-generated errors` (:31) | `test_interaction_log_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError`; `.url` has no query string (unlike `RainbirdAPIError.url`). |
| `should handle 404` (:41) | `test_interaction_log_should_handle_404` | adapted | `NotFoundError`, url with `?format=json`. |
| `should handle 500` (:53) | `test_interaction_log_should_handle_500` | adapted | `ServerError`. |
| `should handle 400` (:65) | `test_interaction_log_should_handle_400` | adapted | `BadRequestError`. |
| `should call the evidence endpoint with the evidence key header` (:76, misnamed: it is the interaction key) | `test_should_call_the_evidence_endpoint_with_the_evidence_key_header` | adapted | `x-interaction-key` sent as in JS; the verbatim body again raises `UnexpectedResponseError`. |
| same (companion) | `test_interaction_key_default_from_client_and_per_call_override` | extra | `Rainbird(interaction_key=...)` default, per-call override, `""` sends no header. |
| `session.test.js` `knowledgeMap should perform simple GET` (:11) | `test_knowledge_map_should_perform_simple_get` | adapted | GET `/analysis/session/thisisthesessionid?filter=version`; JS sends `Authorization: Basic dGhpc2lzdGhlYXBpa2V5Og==`, Python `X-API-Key: thisistheapikey`. `KnowledgeMapVersion` with every field `None`, JSON on `.raw`. |
| same (companion) | `test_knowledge_map_parses_js_km_shape` | extra | Body in the `Km` interface shape from `domain.ts`; `version_number` honours the explicit `versionNumber` (3, also for a `"Draft"` body) since 0.2.0 and still derives it from `versionStatus` when the field is absent. |
| `knowledgeMap should throw fetch-generated errors` (:35) | `test_knowledge_map_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError`. |
| `knowledgeMap should handle 404` (:45) | `test_knowledge_map_should_handle_404` | adapted | `NotFoundError`. |
| `knowledgeMap should handle 500` (:57) | `test_knowledge_map_should_handle_500` | adapted | `ServerError`. |
| `knowledgeMap should handle 400` (:69) | `test_knowledge_map_should_handle_400` | adapted | `BadRequestError`. |
| `facts should perform simple GET` (:87) | `test_facts_should_perform_simple_get` | adapted | GET `...?filter=facts`; empty `SessionFacts`, JSON on `.raw`. |
| same (companion) | `test_facts_parses_js_facts_shape` | extra | Body in the `Facts` interface shape (`{global, context, local}`); `.all` order, `FactSource` enum, `str(fact) == "Tom speaks German (75%)"`. `Facts.global` is `SessionFacts.global_`. |
| same (companion) | `test_facts_relationships_filter_param` | extra | Python-only `relationships=` filter: string, comma-joined list, `None`. |
| `facts should throw fetch-generated errors` (:111) | `test_facts_should_throw_fetch_generated_errors` | adapted | `RainbirdConnectionError`. |
| `facts should handle 404` (:121) | `test_facts_should_handle_404` | adapted | `NotFoundError`. |
| `facts should handle 500` (:133) | `test_facts_should_handle_500` | adapted | `ServerError`. |
| `facts should handle 400` (:145) | `test_facts_should_handle_400` | adapted | `BadRequestError`. |

## Behavioural differences

Deduplicated across all units. "Go" / "JS" names the sibling(s) that differ from Python.

### Authentication and standard headers

- **Credential header** (Go, JS): Go and JS send `Authorization: Basic base64(apiKey + ":")`
  on `/start` and the `/analysis/session` calls only, and nothing at all on `/inject`,
  `/query`, `/response`, `/undo`, `/analysis/evidence`, `/analysis/interactions` and
  `/version`. Python sends `X-API-Key: <key>` on every request and never sends an
  `Authorization` header. `engine_version()` is the only keyless call.
- **Content-Type on GET** (Go, JS): both siblings set `Content-Type: application/json` on
  body-less GETs; Python sends no `Content-Type` on a GET.
- **Accept / User-Agent** (Go, JS): Python sends `Accept: application/json` (Go's `Undo` sends
  no `Accept`; `version()` uses `text/plain, application/json, */*`) and
  `User-Agent: rainbird-ai-python/<version>` on every request; the siblings send neither.
- **Missing API key** (Go): Go fails lazily in `NewSession()` (`ErrClientMissingAPIKey`);
  Python fails eagerly in `Rainbird(...)` with `RainbirdConfigurationError` (after consulting
  `RAINBIRD_API_KEY`). Neither contacts the server.

### Payload keys and bodies

- **Certainty key** (Go, JS): JS forwards facts/answers verbatim (its tests carry
  `"cf": "99"` strings, although the JS `Fact` type declares `certainty?: number`); Go sends
  `"cf": "<string>"` (plus an optional numeric `"certainty"` when `CertFactor` is set) and an
  always-present `"metadata": {}` on inject. Python accepts `cf` as an input alias, validates it
  as a number in 0-100 and sends a numeric `"certainty"` (never `cf`); `metadata` is sent only
  when the fact carries one.
- **Default certainty on `/response`** (JS): JS sends an answer without a certainty key when
  none is given; Python always sends `"certainty": 100` by default.
- **Float encoding** (Go): Go marshals `float64(12)` as `12`; Python writes `12.0` (both decode
  to 12).
- **Undo body** (JS): JS posts `{"data": {}}`; Go and Python post `{}`.
- **Null / empty query fields** (Go, JS): Go serialises `"object": ""`, JS sends
  `"subject": null` / `"object": null` when given `null`; Python omits the key for `None` and
  `""`.
- **Empty lists** (Go, JS): both siblings POST `[]` to `/inject` (and JS `{"answers": []}` to
  `/response`); Python raises `InvalidFactError("facts must not be empty")` /
  `InvalidFactError("answers must not be empty")` before sending.
- **Inject metadata** (Go): Go always sends a `metadata` object (`{"source": "system"}` by
  default); Python sends it only when given (`Fact(metadata=...)`, `inject_fact(source=...)`)
  and forwards the dict as-is (fixed in 0.2.0; it was rejected before).
- **Certainty range** (Go, JS): siblings forward any value (`"121"`); Python validates 0-100
  locally and raises `InvalidFactError`.

### Error handling

- **Error type** (Go, JS): Go returns `error` values (`"API returned error 400: <body>"`,
  `"API returned an error 400: ..."`, `"API returned error code 500: ..."`), JS throws
  `Error("API error code: N")`. Python raises `RainbirdAPIError` subclasses by status
  (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`,
  `RateLimitError`, `ServerError`) with `.status_code`, `.message`, `.body`, `.errors`,
  `.method`, `.url`; `str()` is `"HTTP N calling METHOD <url>: <message>"`. Text bodies are
  kept as the message; JSON error bodies are parsed (`{"error": "x"}` -> `message "x"`); an
  empty or `{}` body gets a status-specific default message and `errors == []`.
- **404 on evidence** (Go): Go returns the body-independent sentinel `ErrFactNotFound`;
  Python's `NotFoundError` keeps the body text.
- **Transport failures** (Go, JS): JS re-throws the `fetch` rejection unchanged; Python wraps
  `httpx.TimeoutException` in `RainbirdTimeoutError("request to METHOD <url> timed out: ...")`
  and other transport errors in `RainbirdConnectionError("could not reach METHOD <url>: ...")`,
  with `__cause__` and `.method`/`.url` set.
- **Local validation errors** (Go): Go uses sentinel values (`ErrNewSessionInvalidKMID`,
  `ErrQueryBlankRelationship`); Python raises `RainbirdConfigurationError` (km id, blank
  relationship, `use_draft` with `version`, `version < 1`, `resume("")`).
- **Retries** (Go, JS): neither sibling retries. Python's default `RetryPolicy` retries GETs on
  429/500/502/503/504 (three requests for a single 500) and POSTs on 429/503; the parity helper
  uses `NO_RETRY`. `engine_version()` always uses the default policy.
- **Unexpected 2xx bodies** (Go, JS): siblings return whatever came back (Go `Inject` never
  reads the body; JS returns `response.json()`). Python raises `UnexpectedResponseError` for
  a `/start` body without an id, an empty or non-JSON inject body, an interactions body that is
  not a list of events, a `/version` body that does not look like a version, and a
  question/result body that is neither (`RESPONSE_TYPE_UNKNOWN` in JS; `parse_query_response`
  itself raises `UnexpectedResponseError` since 0.2.0). Python also fails the
  whole `evidence_tree()` walk on the first bad node, where JS `fullEvidence` drops falsy
  bodies.
- **Tolerance where the siblings fail** (Go): Go fails the interactions call on an unknown
  event name or an unparseable `created`, and the evidence call on a condition with neither
  `relationship` nor `expression.text`; Python keeps the data (`event` verbatim,
  `created=None`, `Condition(relationship="")`).

### URL building and query parameters

- **Base URL** (Go, JS): siblings concatenate the string as given (`"thisistheurl/start/..."`);
  Python normalises a bare host to `https://<host>`, requires an http(s) URL and falls back to
  `RAINBIRD_API_URL` then `https://api.rainbird.ai` when `base_url` is `None`/`""` (Go returns
  `ErrClientMissingEnvironmentURL` instead).
- **Context parameter** (JS): JS sends `?contextId=`; Go and Python send `contextid`. JS orders
  `contextId` before `useDraft`; Python orders `useDraft` first.
- **Combined start parameters** (Go): Go appends each parameter with its own `?`
  (`?contextid=foo?useDraft=true`); Python emits one `&`-joined query string.
- **Path encoding** (Go): Go writes `WA:RF:1234` literally; Python percent-encodes path
  segments (`WA%3ARF%3A1234`).
- **Interactions query string** (Go, JS): siblings GET `/analysis/interactions/{sid}` with no
  query; Python always appends `?format=json` (or `?format=csv` for `format="csv"`).
- **Relationships filter** (Go, JS): Python's `Session.facts(relationships=...)` adds a
  comma-joined `relationships` parameter the siblings do not have (absent when not used).
- **Fact id type** (JS): JS error tests pass `{}` as a fact/session id (`"[object Object]"` in
  the URL); Python stringifies to `{}` and sends `%7B%7D` rather than rejecting it.

### Parsing and typing

- **Return shapes** (Go, JS): Go returns `([]Question, []Answer, error)` from
  `Query`/`Response`/`Undo`; JS returns `{type, data}` or the raw JSON. Python returns one
  `QuestionResponse` / `ResultResponse` (`.is_question`, `.question`, `.questions`,
  `.extra_questions`, `.results`, `.raw`) and records it on `session.last_response` /
  `pending_question` / `result`; `start()` returns a `Session`; `evidence()` returns an
  `Evidence`; `interactions()` a `list[InteractionEvent]`; `knowledge_map()` a
  `KnowledgeMapVersion`; `facts()` a `SessionFacts`; `inject()` an `InjectResponse`.
- **Non-list `result`** (JS): JS hands back `result: {foo: "bar"}` untouched; Python wraps the
  dict into one `ResultItem` with every field empty.
- **Interaction log** (Go): Go maps events to an int enum and decodes `values` into typed
  structs; Python keeps `event` as the API string and `values` as the raw dict.
- **`Fact.Source` on evidence** (Go): both copy the envelope `source` onto the fact (Python
  since 0.2.0; it also keeps it on `Evidence.source`, and a `source` inside `fact` wins).
- **Condition classification** (Go): Go checks `relationship != ""` first; Python classifies
  on the presence of the `expression` key.
- **Numbers** (Go): Go decodes JSON numbers as float64 (`100.0`); Python keeps `int`.
  `SessionFact.certainty` in Python also tolerates `100.0`, `"100"` and `true` where Go's
  `int` field would fail to decode.
- **`FactSource` enum** (Go, JS): Python parses `source` into a `str` enum (still `== "answer"`).
- **`versionNumber`** (Go, JS): both siblings declare a `versionNumber` field; Python's
  `KnowledgeMapVersion.version_number` returns it when present (since 0.2.0) and otherwise
  derives it from `versionStatus` (`"Version3"` -> 3, drafts -> `None`), which the siblings
  do not do.
- **Question fields** (Go): Go gives `[]KnownAnswer{}` for `knownAnswers: []` and `nil` for
  absent `concepts`; Python gives `[]` for both, and `None` for an absent `object`. Go
  normalises `object: ""` to `?`; Python keeps `""` distinct from `None`.
- **`InjectResponse`** (JS): Python's `.ok` is `True` for `{"response": {"ok": true}}` because
  only top-level `result`/`ok`/`status` are inspected; `warningLog` is on `.raw` only.
- **Evidence tree** (JS): JS `fullEvidence` recurses concurrently, keys nodes by the body's
  `factID` in a module-level `Map` that is never cleared (so a later test sees earlier nodes),
  and orders by promise resolution. Python walks breadth-first keyed by the requested id, one
  fresh walk per call (no cache), root first then `child_fact_ids` in condition order.
- **`/version` body** (Go): Go returns any 2xx body; Python validates it looks like a version.

### String formats

- **`ResultItem`** (Go): Go `Answer.String()` is `"Subject - relationship - object [ 50%]"`
  (dash separators, certainty right-aligned to 3 columns, `[  0%]` for the zero value);
  Python `str(ResultItem)` is `"Subject relationship object (50%)"` and appends nothing when
  certainty is absent.
- **`Question`** (Go): Go `Question.String()` is `"prompt (subject - relationship - object)"`
  with `?` for unknown parts; Python `str(Question)` is the prompt alone and falls back to
  `"subject relationship object"` (stripped) only when the prompt is empty.
- **Missing parts** (Go): Go prints `?` (question) or `""` (answer); Python prints `?` for
  both a missing (`None`) and an empty (`""`) subject or object and joins only the non-empty
  parts (since 0.2.0; it interpolated the literal `None` before).
- **`SessionFact`**: Python `str(fact)` is `"Dan might speak English (100%)"`; Go's
  `Fact.String()` would be `"WA:RF:1234 (rule): Dan, might speak, English (100)"`.

### Per-call options versus client configuration

- **Engine** (Go, JS): siblings pass `engine` per call; Python sets `Rainbird(engine=...)` (or
  `RAINBIRD_ENGINE`) once and sends `x-rainbird-engine` on every request. Both JS and Python
  (since 0.2.0) export `ENGINE_CORE` / `ENGINE_BETA` constants for the known engine names.
- **Evidence / interaction keys** (Go, JS): siblings take the key per call only (Go's `*string`
  pointer to `""` would send an empty header). Python also has `Rainbird(evidence_key=...)` /
  `Rainbird(interaction_key=...)` (and the `RAINBIRD_*_KEY` variables) as defaults; a per-call
  `None` falls back to the default, a non-empty value wins, and an explicit `""` suppresses
  the header even when a default is configured.
- **Extra headers** (Go): Go's `AddHeaders(...)` applies to a single `Query` call; Python has
  `headers=` on `query()`, `respond()`, `undo()` and `inject()` (since 0.2.0) plus
  `Rainbird(default_headers=...)` for every request; a per-call header overrides a default one
  case-insensitively.
- **HTTP client** (Go): Go's `HTTP()` falls back to the shared `http.DefaultClient`; Python
  creates a private `httpx.Client` per `Rainbird` and closes it in `close()`; a user-supplied
  `http_client` is never closed by the SDK (same as Go's custom case).
- **Session handle** (Go, JS): siblings expose free functions / client methods taking a session
  id; Python exposes `inject`/`query`/`respond`/`undo`/`evidence`/`interactions`/
  `knowledge_map`/`facts` on a `Session` obtained from `start()` or `resume(sid)` (`resume`
  sends no request and rejects `""`).

### Observation about the Go sources (not a Python difference)

`constants.go` in the Go SDK has `EnvEnterprise = "https://api-enterprise.rainbird.ai"`, while
the JS SDK's `URL_ENTERPRISE` and Python's `Environment.ENTERPRISE` are
`https://enterprise-api.rainbird.ai` (the value documented in `CLAUDE.md`). The Go constant may
be stale; not asserted anywhere here.

## Possible gaps or bugs in the Python SDK

Each entry names the test that pins it. Strict xfails will flip to a failure when the SDK
changes; "pinned" tests assert the current behaviour and will need updating instead. Entries
marked **fixed** were addressed after the port (shipped as 0.2.0); their
former xfails now pass as regular tests and their pinned tests assert the new behaviour.

1. **fixed: No inject metadata support.** Go sends `metadata.source` (`"system"`/`"user"`) with
   each fact; `normalize_facts` rejected the `metadata` key. Now `Fact(metadata=...)`, a
   `metadata` key and `inject_fact(source=...)` are forwarded as-is (a non-dict raises
   `InvalidFactError`). Test: `test_go_inject_query.py::test_inject_some_facts_with_metadata`.
2. **fixed: `Evidence.from_dict` did not propagate the envelope `source` onto `fact.source`**
   (it did propagate `factID` onto `fact.id`). Now both are copied; a `source` inside `fact`
   wins. Test: `test_go_evidence.py::test_fact_source_is_propagated_to_fact[rule|datasource]`.
3. **fixed: `str(ResultItem)` / `str(Question)` / `str(SessionFact)` rendered a missing subject
   or object as the literal `None`** (`"None  None"`, `"John speaks ?"`). Now `?`, with only
   the non-empty parts joined (`"? ?"`, `"John speaks ?"`). Tests:
   `test_go_stringers.py::test_answer_to_string_missing_parts_do_not_render_none`,
   `test_question_to_string_missing_parts_do_not_render_none[4 ids]`.
4. **fixed: No `ENGINE_CORE` / `ENGINE_BETA` constants** for the `x-rainbird-engine` header.
   Now exported from `rainbird_ai` with the JS values. Tests:
   `test_js_start.py::test_should_export_constant_ENGINE_CORE`, `..._ENGINE_BETA`.
5. **fixed: `parse_query_response` raised a bare `ValueError`** for a body that is neither
   question nor result. Now `UnexpectedResponseError` with the body on `.body`;
   `Session.query/respond/undo` re-raise it with the HTTP status. Pinned:
   `test_js_query.py::test_parse_should_handle_a_question_unknown_body`.
6. **A non-list `result` value is silently accepted**: `_as_list` wraps
   `{"result": {"foo": "bar"}}` into one `ResultItem` with every field empty instead of
   raising `UnexpectedResponseError`. Low severity (the API returns a list). Pinned in
   `test_js_query.py`, `test_js_response.py`, `test_js_undo.py` (result cases).
7. **`RainbirdConnectionError.url` omits the query string while `RainbirdAPIError.url`
   includes it** for the same call (`SyncTransport.send` passes `build_url(base, path)` to
   `transport_error()`, `_where()` uses `str(request.url)`). Pinned:
   `test_js_analysis.py::test_interaction_log_should_throw_fetch_generated_errors` (no `?`)
   versus `test_interaction_log_should_handle_404` (`?format=json`).
8. **An explicit empty `evidence_key=""` / `interaction_key=""` silently suppresses the
   client-wide default** (`client.py` treats `""` as given, `_ops` adds the header only for a
   truthy key). Undocumented opt-out. Pinned:
   `test_go_interactions.py::test_empty_interaction_key_sends_no_header`,
   `test_js_analysis.py::test_interaction_key_default_from_client_and_per_call_override`,
   `test_go_evidence.py::test_client_wide_evidence_key_is_default_and_per_call_wins`.
9. **fixed: `KnowledgeMapVersion.version_number` ignored an explicit `versionNumber` field**
   (declared by both siblings' types). Now the field wins when present; the `"VersionN"`
   `versionStatus` fallback remains. Pinned:
   `test_js_analysis.py::test_knowledge_map_parses_js_km_shape` (Draft branch now 3).
10. **fixed: `engine_version()` / `async_engine_version()` had no `retry=` parameter.** Now
    `retry=` (default `RetryPolicy()`, `NO_RETRY` for one attempt). Tests:
    `test_go_client.py::test_engine_version_error_code` (default, 3 calls) and
    `test_engine_version_error_code_no_retry` (1 call).
11. **fixed: No per-call extra-headers option** on `Session.query()/respond()/undo()/inject()`
    (Go's `AddHeaders`). Now `headers=` on all four (sync and async); the Go case is still
    ported with `default_headers` in
    `test_go_inject_query.py::test_query_check_the_optional_headers_propagate`.
12. **`Session.inject()` raises `UnexpectedResponseError` on a 2xx with an empty body**, where Go
    treats any status < 400 as success. Probably intentional (spec body is `{"result": "OK"}`).
    Pinned: `test_go_inject_query.py::test_inject_some_facts_empty_200_body`.
13. **`InjectResponse.ok` only inspects top-level keys**, so `{"response": {"ok": false}}` reports
    `ok=True`; and there is no typed accessor for `warningLog` (declared in `inject.ts`).
    Pinned: `test_js_inject.py::test_inject_response_ok_ignores_nested_ok`,
    `test_inject_response_interface_result_and_warning_log`.
14. **`evidence_tree()` / `iter_evidence_tree()` abort on the first node with a non-object 2xx
    body or a 404**: `evidence_tree()` discards the nodes already fetched (only a caller
    driving `iter_evidence_tree()` sees them first) and there is no lenient/partial mode
    (JS `fullEvidence` skips falsy bodies). Pinned:
    `test_js_evidence.py::test_fullevidence_unbounded_walk_cannot_skip_null_leaf_bodies`.
15. **`InteractionEvent` has no typed accessors** for the per-event payloads (start / query /
    questions / facts / answers / datasources / results) that Go decodes; and
    `_ops.interactions_json` silently drops non-dict list items, at odds with "never lose API
    data". Not pinned (no failing assertion to write; documented in
    `test_go_interactions.py`).
16. **`?format=json` on `GET /analysis/interactions/{sid}`** is sent by Python only; neither
    sibling sends a `format` parameter. Worth confirming against the OpenAPI spec that it is an
    accepted parameter. Not pinned beyond the request assertions.
17. **`path_segment` accepts any non-string with a non-empty `str()`** as a fact or session id
    (a dict becomes `%7B%7D`) instead of raising `RainbirdConfigurationError`. Not asserted.
18. **`_answers._normalize_answer` coerces the answer subject to `str`** while the object keeps
    its type; JS `ConceptValue` allows `string | boolean | number` subjects. Not exercised by
    the sibling cases (every subject is a string); may be intentional.
19. **`SessionFact.from_dict` leniently coerces certainty** (`"100"` -> 100, `true` -> 1) where
    Go's `int` field would reject the wire value. Not exercised; the Go port pins only the
    integer case.

## Not ported

### `not_applicable` cases

| source | reason |
|---|---|
| JS `index.ts:9-13` re-exports of `RESPONSE_TYPE_QUESTION` / `RESPONSE_TYPE_RESULT` / `RESPONSE_TYPE_UNKNOWN` | Not in `index.test.js`'s constants list (only re-exported). Python has no type-tag constants: `parse_query_response` returns a `QuestionResponse` or `ResultResponse` discriminated by `.is_question` / `.is_result` and raises for an unknown shape instead of returning an `UNKNOWN` tag, so there is no export to assert. |

### Assertions with no httpx analogue (documented in the tests, not silently dropped)

- jest's `Object.keys(fetchInit).sort() == ["body", "headers", "method"]` (no extra fetch
  options) in `inject.test.js` / `response.test.js` / `undo.test.js`; the ports assert method,
  headers and body individually (`call.query == {}` stands in where used).
- jest's `toHaveBeenCalledTimes(10)` in `full_evidence.test.js` relies on the mock answering
  `undefined` for six leaf fetches, which Python cannot swallow; the 10 fetches are covered
  across `test_should_perform_a_simple_fullevidence_retrieval` (4, `max_depth=1`) and
  `test_fullevidence_unbounded_walk_cannot_skip_null_leaf_bodies` (5, then
  `UnexpectedResponseError`).
- Go `TestClientVersion` lists `expectCalls` but never asserts it; the ports assert the count.
- Go `TestSessionInject` compares the raw body as a string (`expectBody`); the ports compare
  parsed JSON (key order asserted only in `test_inject_some_facts`), since Python's body
  deliberately differs.

### Out of scope

- `@rainbird/sdk-react` (`packages/react`) and `@rainbird/sdk-react-material`
  (`packages/react-material`): React component and hook tests (UI rendering), nothing the
  Python SDK could be exercised by.
- `tests/test_integration.py` style real-API runs: the sibling suites are offline-only and so
  are these ports.
- Async twins: neither sibling suite has async cases, so only the sync `Rainbird` is driven
  here (`support.make_async_client` exists for future use). The existing `tests/test_aio.py`
  covers sync/async parity of the Python SDK itself.
