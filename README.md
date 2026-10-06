# Rainbird Python SDK

`rainbird-ai` is the official Python client for the [Rainbird](https://rainbird.ai) decision
intelligence REST API. Start a reasoning session against a knowledge map, inject the facts you
already know, ask a question, answer whatever the engine needs to know, and get back a decision
with a certainty score and a fact ID you can turn into a full evidence tree. Sync and async clients
share one API; every response is a typed, frozen dataclass that keeps the original JSON on `.raw`.

[![PyPI](https://img.shields.io/pypi/v/rainbird-ai.svg)](https://pypi.org/project/rainbird-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/rainbird-ai.svg)](https://pypi.org/project/rainbird-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status: alpha.** The package is usable and tested, but it is pre-1.0: the public API may
> change between minor versions (`0.x`) and every such change is listed in
> [CHANGELOG.md](CHANGELOG.md). Pin a minor version (`rainbird-ai~=0.3.0`) if you need
> stability.

## Installation

```bash
pip install rainbird-ai            # or: python -m pip install "rainbird-ai"
uv add rainbird-ai
```

Requires Python 3.11 or newer. Runtime dependencies are [httpx](https://www.python-httpx.org/) and
[anyio](https://anyio.readthedocs.io/) (the async client sleeps through it between retries).

## Getting your API key and knowledge map ID

1. Open your knowledge map in [Rainbird Studio](https://app.rainbird.ai) and go to **Publish**.
   The **API key** is shown there (it is also on your Account page). The same page has the
   access controls for the optional **evidence key** and **interaction key**.
2. The **knowledge map ID** is a GUID: open the map details menu and choose
   **View Knowledge Map ID**.

Export them as `RAINBIRD_API_KEY` and `RAINBIRD_KM_ID` and every example below works unchanged.

## Quick start

```python
from rainbird_ai import Rainbird

with Rainbird(api_key="YOUR_API_KEY") as rb:      # or omit and set RAINBIRD_API_KEY
    session = rb.start("YOUR_KM_ID")               # or omit and set RAINBIRD_KM_ID
    session.inject_fact("Tom", "lives in", "Germany")

    response = session.query("speaks", subject="Tom")
    while response.is_question:
        question = response.question
        print(question.prompt)
        if question.options:
            print("options:", ", ".join(map(str, question.options)))
        response = session.answer(input("> "))

    for result in response.results:
        print(result)                              # Tom speaks German (75%)
    if response.best:
        print("Evidence:", session.evidence_url(response.best.fact_id))
```

`query()`, `answer()`, `skip()`, `respond()` and `undo()` all return either a `QuestionResponse`
(`.is_question`) or a `ResultResponse` (`.is_result`, `.results`, `.best`, `.fact_ids`).

## The question loop explained

A `Question` tells you what the engine needs and how it must be answered:

| Form (`question.form`)         | Helper                     | Example prompt                  | What `answer()` sends                       |
| ------------------------------ | -------------------------- | ------------------------------- | ------------------------------------------- |
| `QuestionForm.FIRST`           | `question.is_first_form`   | "Does Tom speak French?"        | the question's own triple + `answer: yes/no` |
| `QuestionForm.SECOND_OBJECT`   | `question.asks_for_object` | "What language does Tom speak?" | your value in `object`                       |
| `QuestionForm.SECOND_SUBJECT`  | `question.asks_for_subject`| "Who speaks French?"            | your value in `subject`                      |

```python
q = response.question
q.prompt, q.relationship, q.subject, q.object   # the triple being asked about
q.data_type                                     # "string" | "number" | "truth" | "date"
q.options, q.can_add                            # values from q.concepts; may you add a new one?
q.plural, q.allow_unknown, q.allow_cf           # several values? skippable? certainty meaningful?
q.known_answers, response.questions             # values already known; main + extraQuestions
```

`session.answer(value, *, certainty=100, question=None, strict=True)` builds the right payload
from a plain Python value and remembers the pending question for you:

```python
session.answer(True)                       # First Form: True/False or "yes"/"no"
session.answer("German")                   # Second Form Object
session.answer("Tom")                      # Second Form Subject
session.answer(["theft", "water damage"])  # plural question: several answers in one response
session.answer(42)                         # number questions take int/float
session.answer(date(2024, 1, 31))          # date questions take date/datetime (sent as "2024-01-31")
session.answer("German", certainty=80)     # when question.allow_cf is True
session.skip()                             # only when question.allow_unknown is True
session.undo()                             # roll back the last answers; returns the previous question
```

A `datetime` is sent as its calendar date only (`YYYY-MM-DD`, the time of day is dropped); pass an
`int` of epoch milliseconds if you need an instant. Integral `Decimal`s become `int`, enums
(including `str`/`int` mixin enums) use their `.value`, also for First Form yes/no answers and
truth values (an enum member whose value is `True`, `"yes"`, `"false"`, ... works); `None`, dicts,
lists (outside plural questions) and non-finite numbers (NaN or infinity, as `float` or `Decimal`)
raise `InvalidAnswerError`.

With `strict=True` (the default) the SDK raises `InvalidAnswerError` before making a request when
the value is not in `options` and `can_add` is false, when you pass several values to a
non-plural question, or when you skip a question that does not `allow_unknown` (the API would
otherwise reply 400 `Please provide an expected boolean value for unanswered.`). Options are
matched by exact text and your value is sent as-is; a purely numeric match (`5.0` or `"10.0"` for
the options `5` / `"10"`) is accepted only for a numeric answer, a number-typed question or a
numeric option, and then the option's own value is what gets sent, since that is the concept
instance the engine knows. Pass `strict=False` to let the API be the judge. Calling `answer()` with
no pending question raises `NoPendingQuestionError`.

Option checking is not a spell-checker. `canAdd` defaults to true in RBLang, so unless the map
author writes `canAdd="false"` on the relationship, `can_add` is true, any value is sent, and the
engine creates a new concept instance for it. Set `canAdd="false"` in the map when you want
misspelt answers rejected client-side.

For full control build `Answer` objects (or dicts with the API's field names) and send them with
`session.respond(answers)`:

```python
from rainbird_ai import Answer

session.respond([Answer("Tom", "speaks", "French", certainty=100, answer="yes")])
session.respond({"subject": "Germany", "relationship": "national language", "object": "German",
                 "certainty": 100})
```

`session.pending_question`, `session.result` and `session.last_response` expose the current state.
`query()`, `respond()`, `undo()` and `inject()` take `headers={...}` for extra HTTP headers on that
request only; `Rainbird(default_headers=...)` applies to every request.

## Automating the loop with `session.run()`

`run()` queries and keeps calling your `answerer(question)` until the engine returns a result.
The callback may return a value, a list of values (plural questions), an `Answer` or list of
`Answer`s, or `SKIP`.

```python
from rainbird_ai import SKIP, Rainbird

known = {"lives in": "Germany", "national language": "German"}

def answerer(question):
    if question.is_first_form:
        return False
    if question.relationship in known:
        return known[question.relationship]
    if question.allow_unknown:
        return SKIP
    raise RuntimeError(f"no answer for: {question.prompt}")

with Rainbird() as rb:
    session = rb.start()
    result = session.run("speaks", subject="Tom", answerer=answerer, max_questions=50)
    print(result.best)
```

`run()` raises `RainbirdError` once more than `max_questions` (default 100) questions have been asked.

## One call: `rb.decide()`

`decide()` does the whole start / inject / query round trip in one call and returns a flat
`Decision`, which is what scripts, connectors and LLM tools usually want:

```python
with Rainbird() as rb:
    d = rb.decide("speaks", km_id="YOUR_KM_ID", subject="Tom",
                  facts=[("Tom", "lives in", "Germany", 90)], evidence="root")

d.status          # DecisionStatus.RESULT | NO_RESULT | QUESTION
d.decision        # "German": the unknown side of the triple (object here); True/False when both were given
d.certainty       # 75
d.session_id      # the session it created
d.fact_id         # fact ID of the best result, for evidence()/explain()
d.results         # every ResultItem, not just the best
d.evidence        # [Evidence] root node (evidence="root") or whole tree (evidence="tree"); [] by default
d.evidence_url    # Studio link for fact_id (no request needed)
d.question        # the Question when status is QUESTION, else None
d.to_dict()       # flat JSON-serialisable dict of all of the above
```

Facts are injected only when given (any shape `inject()` accepts). Without an `answerer` the
first question ends the call with `status == DecisionStatus.QUESTION`; `d.session` is the
`Session` with that question pending, so `d.session.answer("German")` continues it (or resume
later with `rb.resume(d.session_id)`). Pass `answerer=` to have questions answered through
`session.run()` (`max_questions` applies), which makes the decision always final. An empty
result is `NO_RESULT` (`decision` is `None`, or `False` for a subject + object check), not an
error; API and network failures raise as usual. Evidence fetching is off by default because it
costs one request per node and needs the map's evidence key (`evidence_key=` or
`RAINBIRD_EVIDENCE_KEY`). `use_draft`, `version` and `context_id` are forwarded to `start()`.
The async twin is `await AsyncRainbird.decide(...)`, and the CLI has `rainbird decide`.

## Injecting facts

`inject()` accepts a `Fact`, a dict, a `(subject, relationship, object[, certainty])` tuple, or a
list mixing any of those. Certainty is 0-100 (100 = certain) and defaults to 100. A fact may carry
`metadata`; the API uses `{"source": "system" | "user"}` to record whether it came from another
system (the default) or from a person, and `inject_fact(..., source="user")` is the shorthand.

```python
from rainbird_ai import Fact

session.inject_fact("Tom", "lives in", "Germany")                 # one fact, certainty 100
session.inject_fact("Tom", "lives in", "Germany", source="user")   # metadata: {"source": "user"}
session.inject(Fact("Tom", "lives in", "Germany", certainty=90))
session.inject(Fact("Tom", "works in", "Berlin", metadata={"source": "system"}))
session.inject({"subject": "Ana", "relationship": "lives in", "object": "France", "certainty": 80})
session.inject(("Ben", "lives in", "Spain"))
response = session.inject([Fact("A", "r", "x"), {"subject": "B", "relationship": "r", "object": 1},
                           ("C", "r", "y", 70)])
response.ok   # True when the API replied {"result": "OK"}
```

Rainbird recommends at most 250 facts per request, so long lists are sent in batches of 250
automatically (`batch_size=` overrides this) and the last batch's `InjectResponse` is returned.
Malformed facts are rejected before anything is sent: `InvalidFactError` for a missing or unknown
key, a `metadata` that is not a dict, a certainty outside 0-100 or an empty list;
`InvalidAnswerError` for an unsupported object value (`None`, a dict, a list, NaN or infinity).

## Evidence

Every `ResultItem.fact_id` can be explained. Evidence is secured by default: pass the evidence key
from the Publish page as `Rainbird(evidence_key=...)`, via `RAINBIRD_EVIDENCE_KEY`, or per call.

```python
fact_id = response.best.fact_id

node = session.evidence(fact_id)                  # one Evidence node
node.fact, node.source                            # SessionFact (node.fact.source mirrors node.source);
                                                  #   FactSource.RULE / ANSWER / INJECTION / ...
if node.is_inferred:
    node.rule.bindings                            # {"S": "Tom", "O": "German", ...}
    for cond in node.rule.conditions:             # Condition (.subject/.relationship/.object/.certainty/
        print(cond)                               #   .impact/.fact_id) or ExpressionCondition (.text/.value/.was_met)
node.child_fact_ids                               # fact IDs one level down

tree = session.evidence_tree(fact_id, max_depth=2, evidence_key="...")  # breadth-first, root first
for node in session.iter_evidence_tree(fact_id):  # lazy variant, one request per node
    ...
session.evidence_url(fact_id)
# https://app.rainbird.ai/evidence?id=<factID>&api=https://api.rainbird.ai&sid=<sessionID>
```

The Studio host for `evidence_url()` is derived from `base_url` (`api.` becomes `app.`,
`acme-api.rainbird.ai` becomes `acme.rainbird.ai`); override it with `studio_url=`. For sessions you
did not start here use `rb.evidence(session_id, fact_id)` and `rb.evidence_url(session_id, fact_id)`.

## Analysis

```python
info = session.knowledge_map()          # KnowledgeMapVersion
info.km_id, info.name, info.version_id, info.version_status, info.version_created   # km_id == id
info.is_draft, info.version_number      # versionNumber from the API, else "Version3" -> 3; None for drafts

facts = session.facts()                 # SessionFacts, grouped by scope
facts.global_, facts.context, facts.local, facts.all
facts = session.facts(relationships="speaks")                  # filter by one relationship
facts = session.facts(relationships=["speaks", "lives in"])    # or several
for fact in facts:                      # SessionFact: .subject.value, .relationship, .object.value,
    print(fact, fact.source)            #              .certainty, .id, .source

events = session.interactions()         # list[InteractionEvent]: .event, .created, .values
csv_text = session.interactions(format="csv")
```

Interaction logging is **off by default**. Enable it in Rainbird Studio per knowledge map version;
only sessions started after that are recorded. If the log is secured, pass the interaction key as
`Rainbird(interaction_key=...)`, `RAINBIRD_INTERACTION_KEY`, or `interactions(interaction_key=...)`.
A 200 body that is not a list of events (or an envelope with the list under `values`, `events` or
`interactions`, or a single event object) raises `UnexpectedResponseError` carrying any error text
it contains, for example a secured log answered with 200 `{"err": [...]}`.

## Natural language (beta)

`interact()` lets a user inject facts, query and answer in plain text within a session. The
response type is `"question"`, `"result"` or `"error"`. An error is a normal outcome (for example
the text contained no recognisable query), so it is returned rather than raised.

```python
r = session.interact("Tom lives in Germany. What language does Tom speak?")

if r.is_question:                          # r.response_type == "question"
    print(r.prompt)                        # first question's prompt
    r = session.interact("German")         # answer in natural language, same session
elif r.is_result:
    for item in r.results:                 # ResultItem, same shape as a query result
        print(item, item.fact_id)
else:
    print(r.error.code, r.error.message)
    print(r.error.suggested_chat_response) # user-friendly text for chat UIs

r.raise_for_error()                        # raises NaturalLanguageError instead, if you prefer
r.injected_facts, r.invalid_facts, r.unmatched, r.metadata.total_tokens

explanation = session.explain(fact_id, language="en", instructions="Use a formal tone.")
print(explanation.explanation)
```

`explain()` raises `NaturalLanguageError` when the API returns an error object instead of an
explanation. Both endpoints send the required `Version: v1` header for you.

## Async usage

`rainbird_ai.aio.AsyncRainbird` and `AsyncSession` mirror the sync API method for method.
`run()` accepts a plain or `async` answerer and `iter_evidence_tree()` is an async iterator.
Retry back-off waits with `anyio.sleep` (not `asyncio.sleep`), which is why `anyio` is a runtime
dependency; `async_engine_version()` is the async twin of `engine_version()` (see below).

```python
import asyncio

from rainbird_ai.aio import AsyncRainbird


async def main() -> None:
    async with AsyncRainbird() as rb:                 # reads RAINBIRD_API_KEY
        session = await rb.start("YOUR_KM_ID")
        await session.inject_fact("Tom", "lives in", "Germany")

        response = await session.query("speaks", subject="Tom")
        while response.is_question:
            print(response.question.prompt)
            response = await session.answer("German")

        for result in response.results:
            print(result)
        if response.best:
            print(await session.evidence_tree(response.best.fact_id))


asyncio.run(main())
```

## Configuration

Explicit constructor arguments win over environment variables.

| Argument (`Rainbird(...)` / `AsyncRainbird(...)`) | Environment variable       | Default                    | Purpose                                                          |
| ------------------------------------------------- | -------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `api_key`                                         | `RAINBIRD_API_KEY`         | required                   | sent as `X-API-Key`                                              |
| `base_url`                                        | `RAINBIRD_API_URL`         | `https://api.rainbird.ai`  | an `Environment` member or any host string                       |
| `evidence_key`                                    | `RAINBIRD_EVIDENCE_KEY`    | `None`                     | `x-evidence-key` for secured evidence                            |
| `interaction_key`                                 | `RAINBIRD_INTERACTION_KEY` | `None`                     | `x-interaction-key` for secured interaction logs                 |
| `engine`                                          | `RAINBIRD_ENGINE`          | `None`                     | `x-rainbird-engine`; `ENGINE_CORE` / `ENGINE_BETA` are the known names |
| `timeout`                                         |                            | `30.0`                     | seconds or an `httpx.Timeout`; `None` disables it; see below     |
| `retry`                                           |                            | `RetryPolicy()`            | see below; `NO_RETRY` turns retries off                          |
| `default_headers`                                 |                            | `{}`                       | extra headers merged (case-insensitively) into every request     |
| `http_client`                                     |                            | a new `httpx.Client`       | bring your own client (proxies, transports, tests); you close it |
| `start(km_id)`                                    | `RAINBIRD_KM_ID`           | required                   | knowledge map to start a session on                              |

```python
import httpx
from rainbird_ai import Environment, Rainbird, RetryPolicy

rb = Rainbird(base_url=Environment.ENTERPRISE)               # https://enterprise-api.rainbird.ai
rb = Rainbird(base_url=Environment.ENTERPRISE_SOUTH_AMERICA) # https://sa-enterprise-api.rainbird.ai
rb = Rainbird(base_url="https://acme-api.rainbird.ai")       # private deployment (Studio: acme.rainbird.ai)
rb = Rainbird(timeout=10, retry=RetryPolicy(max_retries=4, backoff_factor=1.0))
rb = Rainbird(default_headers={"X-Request-Source": "billing-service"})
rb = Rainbird(http_client=httpx.Client(proxy="http://proxy.internal:3128", timeout=60))

session = rb.start("KM_ID", use_draft=True)        # draft instead of the live version
session = rb.start("KM_ID", version=3)             # a specific version (not with use_draft)
session = rb.start("KM_ID", context_id="case-42")  # share context-scoped facts between sessions
session = rb.resume("existing-session-id")         # attach to a session; no request is made
```

`Environment.COMMUNITY`, `.ENTERPRISE` and `.ENTERPRISE_SOUTH_AMERICA` each expose `.studio_url`;
`rb.studio_url` gives the Studio host for whatever `base_url` is in use. A bare host such as
`acme-api.rainbird.ai` gets `https://`; an empty string falls back to `RAINBIRD_API_URL` / the
default; a non-`http(s)` scheme, a scheme without a host (`"https://"`) or a URL containing `?`
or `#` (a query string or fragment, even an empty trailing one) raises
`RainbirdConfigurationError`.

`GET /version` is public, so you can check connectivity or the engine version without a key:

```python
from rainbird_ai import NO_RETRY, async_engine_version, engine_version

engine_version()                                            # "4.89.0"; host from RAINBIRD_API_URL or default
engine_version("https://acme-api.rainbird.ai", timeout=5)   # engine=, timeout=, http_client=, retry= keyword-only
engine_version(retry=NO_RETRY)                              # one attempt; the default RetryPolicy retries 5xx
version = await async_engine_version()                      # inside a coroutine
rb.version()                                                # same endpoint through a configured client
```

**Timeouts.** The SDK's own connection pool is created with `timeout` (default 30 s). When you
bring your own `http_client`, its timeout applies unless you also pass `timeout=` explicitly, in
which case the SDK sends that value with every request so it is never silently ignored.

**RetryPolicy.** Rainbird's decision endpoints are stateful: a `POST /response` that reached the
engine but whose reply was lost would answer twice if replayed. The default policy
(`max_retries=2`, `backoff_factor=0.5`, `max_backoff=20.0`) therefore retries `GET` requests on
connection errors, timeouts and 429, 500, 502, 503 and 504 (`retry_statuses`), but retries `POST`
requests only on 429 and 503, where the service refused the request before processing it. 502
and 504 are deliberately *not* retried
for `POST`: a gateway can report them after the engine has already applied the request. `POST` is
never retried on connection errors or timeouts unless you set
`retry_post_on_transport_error=True`; `post_retry_statuses=` changes the status list. Only genuine
network failures (timeouts, connection/protocol/proxy errors) count as retryable; a malformed URL
raises `RainbirdConfigurationError` at once. Delays grow exponentially with jitter and honour
`Retry-After`.

**Redirects.** The SDK never follows redirects (its own `httpx` clients are created with
`follow_redirects=False`, so your API key is not forwarded to another host); a 3xx surfaces as
`RainbirdAPIError` with the message `unexpected redirect to <location>`, followed by the body's
error text if there is one. Keep `follow_redirects=False` on any `http_client` you supply.

## Error handling

Everything the SDK raises derives from `RainbirdError`:

```text
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
├── UnexpectedResponseError         2xx body that is not JSON or not the documented shape
└── RainbirdAPIError                non-2xx HTTP response
    ├── BadRequestError             400
    ├── AuthenticationError         401
    ├── PermissionDeniedError       403
    ├── NotFoundError               404
    ├── RateLimitError              429
    └── ServerError                 5xx
```

```python
from rainbird_ai import (
    AuthenticationError, BadRequestError, RainbirdAPIError, RainbirdError, RateLimitError,
)

try:
    response = session.query("speaks", subject="Tom")
except AuthenticationError:
    print("check RAINBIRD_API_KEY")
except BadRequestError as exc:
    print(exc.errors)              # ["..."] from the API's {"err": [...]} body
except RateLimitError as exc:
    print("retry in", exc.retry_after, "seconds")
except RainbirdAPIError as exc:
    print(exc.status_code, exc.message, exc.body, exc.method, exc.url)
except RainbirdError as exc:       # transport errors, invalid answers, configuration, ...
    print("SDK error:", exc)
```

`RainbirdAPIError` exposes `.status_code`, `.message`, `.errors` (every error string found in the
body, whatever its shape), `.body` (parsed JSON or raw text), `.method`, `.url` and `.retry_after`
(seconds from a `Retry-After` header, when present). For text bodies `.message` is
whitespace-collapsed and truncated to 300 characters; tags are stripped only when the body is HTML
(an HTML content type, or a body starting with `<`: `<html>Bad Gateway</html>` becomes
`Bad Gateway`), so a plain-text message keeps its `<`/`>`; `.body` keeps the raw text.

A 2xx response whose body is empty or not JSON raises `UnexpectedResponseError`, not
`RainbirdAPIError`, with a message such as
`expected a JSON body from GET https://api.rainbird.ai/start/KM but got: '<html>...'`; `.body`
holds the raw text and `.status_code` the HTTP status. The same exception covers a JSON body that
lacks a required field (for example a `start` reply without a session id, or an interaction log
that is not a list of events) and a `GET /version` body that is not a plausible version: a JSON
body must carry a non-empty string or number under `version`, `engine` or `semver` (`{}`, `[]` and
`{"version": null}` are rejected), and a plain-text body must be non-empty, contain no whitespace
or `<`, and be at most 64 characters.

## Knowledge map management

> These two endpoints are not part of the public API documentation and may change. They are
> supplied as-is; pin your SDK version if you depend on them.

```python
source = rb.get_map_source("KM_ID")     # GET /analysis/file/{kmID}
source.rblang                           # the RBLang XML
source.concepts, source.relationships   # summaries as returned by the API

rblang = """<?xml version="1.0" encoding="UTF-8"?>
<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">
  <concept name="Person" type="string"/>
  <concept name="Language" type="string"/>
  <rel name="speaks" subject="Person" object="Language"/>
  <concinst name="Tom" type="Person"/>
  <concinst name="German" type="Language"/>
  <relinst type="speaks" subject="Tom" object="German" cf="100"/>
</rbl:kb>"""

created = rb.create_map("Languages", rblang, description="Who speaks what")   # POST /maps
created.km_id                                   # GUID to pass to rb.start()
created.id                                      # Rainbird's numeric id
created.description                             # optional; sent only when you pass it
```

When Rainbird rejects the RBLang it still answers HTTP 200 with an `error` key; the SDK turns that
into `KnowledgeMapError` (`.errors`, `.body`). Only a truthy `err`, `errors` or `error` value
counts as a failure (`{"error": false}`, `null`, `""`, `0`, `[]` and `{}` are success); an
informational `message` alongside `kmID` on a success body does not.

## Command line

Installing the package adds a `rainbird` command. It reads the same environment variables as the
client, and the global options `--api-key`, `--base-url`, `--engine`, `--evidence-key`,
`--interaction-key`, `--timeout` and `--json` (print the raw API JSON) may be given before or after
the subcommand (`rainbird --json version` and `rainbird version --json` are equivalent).

```bash
export RAINBIRD_API_KEY=... RAINBIRD_KM_ID=...

rainbird version                                              # engine and SDK versions (no API key needed)
rainbird start --draft                                        # start a session, print its id
rainbird query -r speaks -s Tom                               # interactive question loop
rainbird query -r speaks -s Tom --fact Tom "lives in" Germany --answers German --json
rainbird query -r speaks -s Tom --answers "?" "theft,water damage" German@80 --no-input
rainbird decide -r speaks -s Tom --fact Tom "lives in" Germany --evidence root --json   # one-shot Decision
rainbird inject --session SID --fact Tom "lives in" Germany --fact Ana "lives in" France 80
rainbird evidence --session SID --fact-id WA:RF:... --tree --url   # nested tree + Studio link
rainbird interact "Tom lives in Germany. What does Tom speak?"
rainbird interactions --session SID --format csv
rainbird map get KM_ID --out map.rbl                          # undocumented endpoint
rainbird map create --name "My map" --file map.rbl [--description "..."]   # undocumented endpoint
```

Further commands: `respond`, `undo`, `explain`, `facts`, `session`; run `rainbird COMMAND --help`.
`decide` runs start / inject / query once and prints the `Decision` summary (`status`, `decision`,
`certainty`, `session`, `fact_id`, `evidence` URL; every result when there are several; the
evidence tree with `--evidence root|tree [--max-depth N]`), or `Decision.to_dict()` with `--json`.
It never prompts: a question is printed with status `question` and the command exits 3.
In `--answers` (and at the interactive prompt), `?` skips a question, `a,b` answers a plural
question and `value@80` sets the certainty; an `@` only counts as the certainty separator when a
number follows, so `tom@example.com` is a plain value, while a numeric suffix outside 0-100
(`German@150`, `German@-5`) is an error and an empty answer is rejected (`answer must not be
empty`). `?@50` still skips (the certainty is dropped), and certainty applies to the whole answer,
so write `a, b@70` rather than `a@70, b@80` (rejected). Scripted answers are checked only when
their question arrives, so a rejected one stops the run with exit code 1 after the session has
started; in interactive mode an answer the SDK or the engine rejects (`InvalidAnswerError`,
`InvalidFactError`, HTTP 400) is reported on stderr as `  ! ...` and the question is asked again,
while any other error (401, session not found, network) exits 1 at once. With `--json` the
interactive questions and prompts go to stderr, so stdout is always a single JSON document.
`--file` takes a JSON object, a list of fact objects, or a list of
`[subject, relationship, object, certainty?]` lists; anything else, a flat
`["Tom", "lives in", "Germany"]` included, is a usage error.
`evidence --tree` prints the tree nested: the root at column 0, each supporting fact indented two
spaces under the fact it supports, inline conditions as `- ...` lines, facts referenced only from
expression functions beneath their expression, and anything fetched but not reachable from the
root appended at the end.

Exit codes: `0` success; `1` API, SDK or file error (message on stderr); `2` usage error (also an
invalid `--fact`, such as a non-numeric or out-of-range certainty, or malformed or wrongly shaped
`--file` / `respond` JSON); `3` a question is still
pending and no input is available (`--no-input`, non-interactive stdin or Ctrl-D): the question is
printed to stdout (with `--json`, the question response plus `"session_id"`) and
`rainbird: session SID is waiting for an answer; resume with --session SID` to stderr; `130`
interrupted.

## Testing your own code

Inject an `httpx.Client` with an `httpx.MockTransport` to exercise your code without the network.
The SDK's own test-suite does this with a stateful in-memory fake (`tests/conftest.py`).

```python
import httpx
from rainbird_ai import Rainbird

def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/start/"):
        return httpx.Response(200, json={"id": "sid-1", "kmVersion": {"id": "v-1"}})
    if path.endswith("/query"):
        return httpx.Response(200, json={"sid": "sid-1", "result": [
            {"subject": "Tom", "relationship": "speaks", "object": "German",
             "certainty": 75, "factID": "WA:RF:abc"}]})
    return httpx.Response(404, json={"err": ["not found"]})

http = httpx.Client(transport=httpx.MockTransport(handler))
with Rainbird(api_key="test", base_url="https://fake-api.rainbird.ai", http_client=http) as rb:
    session = rb.start("km-1")
    response = session.query("speaks", subject="Tom")
    assert response.is_result and response.best.object == "German"
http.close()
```

## Requirements

- Python 3.11+
- `httpx>=0.27,<1.0`
- `anyio>=4`

## Links

- Developer documentation: https://docs.rainbird.ai/rainbird/developer-docs/overview
- API reference: https://rainbird.redoc.ly/
- OpenAPI specification: https://api.rainbird.ai/docs/openapi.yaml

## Contributing

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest                                   # unit tests against the in-memory fake API
RAINBIRD_API_KEY=... RAINBIRD_KM_ID=... .venv/bin/pytest -m integration   # real API
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/mypy                                     # strict, on src/
```

## License

MIT. See [LICENSE](LICENSE).
