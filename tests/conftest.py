"""Shared fixtures: an in-memory fake Rainbird API served through ``httpx.MockTransport``.

``FakeRainbird`` behaves like the documented API closely enough to exercise the
SDK end to end without network access. Tests script the decision flow with
``fake.queue(...)`` and can inject failures with ``fake.fail_next(...)``.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from rainbird_ai import Rainbird
from rainbird_ai.aio import AsyncRainbird

API_KEY = "test-api-key"
BASE_URL = "https://fake-api.rainbird.ai"
KM_ID = "11111111-2222-3333-4444-555555555555"
KM_VERSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

# --------------------------------------------------------------------------- canned payloads

QUESTION_SECOND_OBJECT = {
    "subject": "Germany",
    "dataType": "string",
    "relationship": "national language",
    "type": "Second Form Object",
    "plural": False,
    "allowCF": True,
    "allowUnknown": False,
    "canAdd": True,
    "prompt": "What is the national language of Germany?",
    "knownAnswers": [],
    "concepts": [
        {"conceptType": "language", "name": "English", "type": "string", "value": "English"},
        {"conceptType": "language", "name": "French", "type": "string", "value": "French"},
    ],
}

QUESTION_SECOND_SUBJECT = {
    "object": "French",
    "dataType": "string",
    "relationship": "speaks",
    "type": "Second Form Subject",
    "plural": True,
    "allowCF": False,
    "allowUnknown": True,
    "canAdd": False,
    "prompt": "Who speaks French?",
    "knownAnswers": [],
    "concepts": [
        {"conceptType": "person", "name": "Tom", "type": "string", "value": "Tom"},
        {"conceptType": "person", "name": "Ana", "type": "string", "value": "Ana"},
    ],
}

QUESTION_FIRST_FORM = {
    "subject": "Tom",
    "object": "French",
    "dataType": "string",
    "relationship": "speaks",
    "type": "First Form",
    "plural": False,
    "allowCF": True,
    "allowUnknown": True,
    "canAdd": False,
    "prompt": "Does Tom speak French?",
    "knownAnswers": [],
    "concepts": [],
}

QUESTION_PLURAL_LOSS = {
    "subject": "Ben",
    "dataType": "string",
    "relationship": "suffered",
    "type": "Second Form Object",
    "plural": True,
    "allowCF": True,
    "allowUnknown": False,
    "canAdd": False,
    "prompt": "What type of loss has Ben suffered?",
    "knownAnswers": [],
    "concepts": [
        {
            "conceptType": "loss",
            "name": "cosmetic damage",
            "type": "string",
            "value": "cosmetic damage",
        },
        {"conceptType": "loss", "name": "water damage", "type": "string", "value": "water damage"},
        {"conceptType": "loss", "name": "theft", "type": "string", "value": "theft"},
    ],
}

RESULT_TOM_SPEAKS_GERMAN = {
    "result": [
        {
            "subject": "Tom",
            "relationshipType": "speaks",
            "object": "German",
            "certainty": 75,
            "factID": "WA:RF:20ecfacf8825b6ace2645713fcbe0d0692b2fe49498156eba4b99776a95eaa51",
            "relationship": "speaks",
            "objectMetadata": {},
        }
    ],
    "queryProfile": [],
}

EVIDENCE_ROOT = {
    "factID": "WA:RF:20ecfacf8825b6ace2645713fcbe0d0692b2fe49498156eba4b99776a95eaa51",
    "source": "rule",
    "fact": {
        "subject": {"type": "person", "value": "Tom", "dataType": "string"},
        "relationship": {"type": "speaks"},
        "object": {"type": "language", "value": "German", "dataType": "string"},
        "certainty": 75,
    },
    "time": 1751644730474,
    "rule": {
        "bindings": {"COUNTRY": "Germany", "O": "German", "S": "Tom"},
        "conditions": [
            {
                "certainty": 100,
                "factID": "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e",
                "factKey": "414fcb69-0a58-4cde-a23a-be4b92a7bd54",
                "impact": 37.5,
                "object": "Germany",
                "objectType": "string",
                "relationship": "lives in",
                "salience": 100,
                "subject": "Tom",
            },
            {
                "certainty": 100,
                "factID": "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681",
                "factKey": "cb7b0d31-cb16-4f76-af7a-7ffe6450c025",
                "impact": 37.5,
                "object": "German",
                "objectType": "string",
                "relationship": "national language",
                "salience": 100,
                "subject": "Germany",
            },
            {
                "expression": {
                    "functions": {
                        "countRelationshipInstances( 'Tom', 'lives in', *)": {
                            "facts": [
                                {
                                    "certainty": 100,
                                    "factID": "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e",
                                    "object": "Germany",
                                    "objectType": "string",
                                    "relationship": "lives in",
                                    "subject": "Tom",
                                }
                            ],
                            "result": {"type": "number", "value": 1},
                        }
                    },
                    "text": "countRelationshipInstances( 'Tom', 'lives in', *)",
                    "value": "%N",
                },
                "impact": 25,
                "salience": 100,
                "wasMet": True,
            },
        ],
        "ruleMaxCertainty": 75,
    },
}


def leaf_evidence(fact_id: str, subject: str, rel: str, obj: str, source: str) -> dict[str, Any]:
    return {
        "factID": fact_id,
        "source": source,
        "fact": {
            "subject": {"type": "thing", "value": subject, "dataType": "string"},
            "relationship": {"type": rel},
            "object": {"type": "thing", "value": obj, "dataType": "string"},
            "certainty": 100,
        },
        "time": 1751644730000,
    }


EVIDENCE_LEAVES = {
    "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e": leaf_evidence(
        "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e",
        "Tom",
        "lives in",
        "Germany",
        "injection",
    ),
    "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681": leaf_evidence(
        "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681",
        "Germany",
        "national language",
        "German",
        "answer",
    ),
}

SESSION_VERSION = {
    "km": {
        "id": KM_ID,
        "name": "Languages",
        "versionID": KM_VERSION_ID,
        "versionCreated": "2025-07-09T14:27:22Z",
        "versionStatus": "Version3",
    }
}

SESSION_FACTS = {
    "global": [
        {
            "id": "WA:KF:global1",
            "subject": {"concept": "country", "value": "France", "dataType": "string"},
            "relationship": "national language",
            "object": {"concept": "language", "value": "French", "dataType": "string"},
            "certainty": 100,
            "source": "knowledgemap",
        }
    ],
    "local": [
        {
            "id": "WA:IF:local1",
            "subject": {"concept": "person", "value": "Tom", "dataType": "string"},
            "relationship": "lives in",
            "object": {"concept": "country", "value": "Germany", "dataType": "string"},
            "certainty": 100,
            "source": "injection",
        }
    ],
    "context": [],
}

INTERACTIONS = [
    {
        "values": {"start": {"useDraft": False, "kmVersionID": KM_VERSION_ID, "sessionID": "S"}},
        "event": "start",
        "created": "2025-07-10T10:06:27.780Z",
    },
    {
        "values": {"query": {"subject": "Tom", "relationship": "speaks"}},
        "event": "query",
        "created": "2025-07-10T10:06:44.532Z",
    },
    {
        "values": {
            "results": [
                {"subject": "Tom", "object": "German", "relationship": "speaks", "certainty": 75}
            ]
        },
        "event": "result",
        "created": "2025-07-10T10:07:34.276Z",
    },
]

INTERACT_QUESTION = {
    "responseType": "question",
    "facts": {},
    "metadata": {
        "querySubmittedAt": "2025-07-10T10:49:13.231973361Z",
        "languageEngineResponseTime": 2173,
        "llmTokens": {"completionTokens": 52, "promptTokens": 3117, "totalTokens": 3169},
        "apiVersion": "",
        "endpoint": "interact",
        "sessionID": "S",
    },
    "query": {"subject": "Ben", "relationship": "can claim", "object": None},
    "questions": [
        {
            "allowCF": True,
            "canAdd": True,
            "concepts": [
                {"conceptType": "policy", "name": "Gold", "type": "string", "value": "Gold"},
                {"conceptType": "policy", "name": "Silver", "type": "string", "value": "Silver"},
            ],
            "dataType": "string",
            "prompt": "Which policy does Ben hold?",
            "relationship": "holds",
            "subject": "Ben",
            "type": "Second Form Object",
        }
    ],
    "results": None,
}

INTERACT_RESULT = {
    "responseType": "result",
    "facts": {
        "injected": [
            {
                "subject": "Ben",
                "relationship": "holds",
                "object": "Gold",
                "cf": "",
                "certainty": 100,
                "metadata": {},
            }
        ],
        "invalid": [],
        "unmatched": ["and he likes tea"],
    },
    "metadata": {"endpoint": "interact", "sessionID": "S"},
    "query": {"subject": "Ben", "relationship": "can claim", "object": None},
    "results": [
        {
            "Subject": "Ben",
            "Object": "repair",
            "Certainty": 100,
            "FactID": "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea",
            "Relationship": "can claim",
        }
    ],
}

INTERACT_ERROR = {
    "responseType": "error",
    "error": {
        "code": 1001,
        "message": "Could not detect a query in the prompt",
        "suggestedChatResponse": "I couldn't work out what you want to know. Could you rephrase?",
        "statusCode": "400",
    },
    "metadata": {"endpoint": "interact", "sessionID": "S"},
}

EXPLAIN_OK = {
    "explanation": "Tom speaks German because he lives in Germany, whose national language is German.",
    "metadata": {
        "querySubmittedAt": "2025-07-10T11:16:12.597111859Z",
        "languageEngineResponseTime": 5538,
        "llmTokens": {"completionTokens": 174, "promptTokens": 726, "totalTokens": 900},
        "apiVersion": "",
        "endpoint": "explain",
        "sessionID": "S",
    },
}

RBLANG = """<?xml version="1.0" encoding="utf-8"?>
<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">
    <concept name="Person" type="string"/>
    <concept name="Language" type="string"/>
    <rel name="speaks" subject="Person" object="Language"/>
</rbl:kb>
"""

MAP_SOURCE = {
    "concepts": [{"name": "Person", "type": "string"}, {"name": "Language", "type": "string"}],
    "rels": [{"name": "speaks", "subject": "Person", "object": "Language"}],
    "rblang": RBLANG,
}

# Map-management endpoints (/maps/{kmID}/tests|runs|readme, /runs/{runID}, /validate).

TEST_SUITE = [
    {
        "name": "Fred speaks French",
        "steps": [
            {
                "expect": {"type": "none"},
                "then": {"type": "Query", "subject": "Fred", "relationship": "speaks"},
            },
            {
                "expect": {
                    "type": "Expect Result",
                    "results": [
                        {"subject": "Fred", "relationship": "speaks", "object": "French", "cf": 100}
                    ],
                },
                "then": {"type": "none"},
            },
        ],
    }
]

# Bodies of GET /runs/{runID}; the fake adds "runID". TEST_RUN_COMPLETE has the shape of a real
# api.rainbird.ai reply (engine 4.123.0): passing results carry "expected"/"actual" too.
TEST_RUN_COMPLETE = {
    "status": "complete",
    "summary": {"total": 2, "passed": 1, "failed": 1, "errored": 0, "notRun": 0},
    "results": [
        {
            "index": 0,
            "name": "Fred speaks French",
            "status": "pass",
            "expected": {
                "results": [
                    {"cf": 100, "object": "French", "relationship": "speaks", "subject": "Fred"}
                ],
                "type": "Expect Result",
            },
            "actual": {
                "result": [
                    {
                        "certainty": 100,
                        "factID": "WA:KF:6a4c3caa4d9e8f92a49dcc1434ce51d8bf368a9830ad42ac90324a90d539b417",
                        "object": "French",
                        "relationship": "speaks",
                        "subject": "Fred",
                    }
                ]
            },
        },
        {
            "index": 1,
            "name": "Fred lives in France",
            "status": "fail",
            "failedStepIndex": 1,
            "errors": ["expected response did not match actual response"],
            "expected": {
                "results": [
                    {"cf": 100, "object": "France", "relationship": "lives in", "subject": "Fred"}
                ],
                "type": "Expect Result",
            },
            "actual": {"result": []},
        },
    ],
}

TEST_RUN_PASSED = {
    "status": "complete",
    "summary": {"total": 1, "passed": 1, "failed": 0, "errored": 0, "notRun": 0},
    "results": [{"index": 0, "name": "Fred speaks French", "status": "pass"}],
}

TEST_RUN_RUNNING = {
    "status": "running",
    "summary": {"total": 2, "passed": 1, "failed": 0, "errored": 0, "notRun": 1},
    "results": [{"index": 0, "name": "Fred speaks French", "status": "pass"}],
}

TEST_RUN_FAILED = {
    "status": "failed",
    "summary": {"total": 0, "passed": 0, "failed": 0, "errored": 0, "notRun": 0},
    "results": [],
    "error": {"category": "suite_load", "message": "the map's tests couldn't be loaded"},
}

VALIDATE_OK = {
    "valid": True,
    "errors": [],
    "warnings": [
        {"message": "concept 'Language' has no instances", "line": 4},
        {"message": "map has no rules"},
    ],
}

VALIDATE_ERROR = {
    "valid": False,
    "errors": [{"message": "root element must be rbl:kb", "line": 1}],
    "warnings": [],
}


# --------------------------------------------------------------------------- the fake server


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    params: dict[str, str]
    json: Any


@dataclass
class FakeRainbird:
    """Minimal stateful stand-in for the Rainbird API."""

    requests: list[Recorded] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    decisions: deque[Any] = field(default_factory=deque)  # scripted bodies for query/response/undo
    failures: deque[tuple[int, Any, dict[str, str]]] = field(default_factory=deque)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    require_evidence_key: str | None = None
    require_interaction_key: str | None = None
    version_text: str = "4.89.0"
    interact_bodies: deque[Any] = field(default_factory=deque)
    map_create_body: Any = None
    map_source_body: Any = None
    test_suites: dict[str, Any] = field(default_factory=dict)  # kmID -> stored suite
    test_runs: dict[str, str] = field(default_factory=dict)  # runID -> kmID
    run_bodies: deque[Any] = field(default_factory=deque)  # scripted GET /runs/{runID} bodies
    readmes: dict[str, str] = field(default_factory=dict)  # kmID -> draft readme

    def __post_init__(self) -> None:
        self.evidence = {EVIDENCE_ROOT["factID"]: EVIDENCE_ROOT, **EVIDENCE_LEAVES}
        self.map_source_body = MAP_SOURCE
        self.map_create_body = {
            "kmID": "99999999-8888-7777-6666-555555555555",
            "id": 4242,
            "name": "New map",
        }

    # scripting helpers -------------------------------------------------------------

    def queue(self, *bodies: Any) -> None:
        """Queue bodies returned, in order, by the next query/response/undo calls."""
        self.decisions.extend(bodies)

    def queue_question(self, question: dict[str, Any], *extra: dict[str, Any]) -> None:
        body: dict[str, Any] = {"question": question}
        if extra:
            body["extraQuestions"] = list(extra)
        self.decisions.append(body)

    def queue_result(self, *items: dict[str, Any]) -> None:
        self.decisions.append({"result": list(items), "queryProfile": []})

    def queue_run(self, *bodies: Any) -> None:
        """Queue bodies returned, in order, by the next ``GET /runs/{runID}`` calls."""
        self.run_bodies.extend(bodies)

    def fail_next(
        self, status: int, body: Any = None, headers: dict[str, str] | None = None
    ) -> None:
        """Make the next request (any endpoint) fail with ``status``."""
        self.failures.append((status, body, headers or {}))

    @property
    def last(self) -> Recorded:
        return self.requests[-1]

    def calls(self, path_fragment: str) -> list[Recorded]:
        return [r for r in self.requests if path_fragment in r.path]

    # request handling ------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        body: Any = None
        if request.content:
            try:
                body = json.loads(request.content)
            except ValueError:
                body = request.content.decode()
        rec = Recorded(
            method=request.method,
            path=request.url.path,
            headers={k.lower(): v for k, v in request.headers.items()},
            params=dict(request.url.params),
            json=body,
        )
        self.requests.append(rec)

        if self.failures:
            status, fbody, fheaders = self.failures.popleft()
            if fbody is None:
                return httpx.Response(status, headers=fheaders, request=request)
            if isinstance(fbody, str):
                return httpx.Response(status, text=fbody, headers=fheaders, request=request)
            return httpx.Response(status, json=fbody, headers=fheaders, request=request)

        path = request.url.path
        parts = [p for p in path.split("/") if p]

        if path == "/version":
            return httpx.Response(
                200, text=self.version_text, headers={"content-type": "text/plain"}, request=request
            )

        if (
            (parts[0] == "maps" and len(parts) == 3 and parts[2] in {"tests", "runs", "readme"})
            or (parts[0] == "runs" and len(parts) == 2)
            or parts == ["validate"]
        ):
            return self._map_management(parts, rec, request)

        if (
            rec.headers.get("x-api-key") != API_KEY
            and parts[0] not in {"nl"}
            and parts[0] not in self.sessions
        ):
            # start/analysis/maps require the api key; session endpoints are session-authenticated
            if parts[0] in {"start", "analysis", "maps"}:
                return httpx.Response(
                    401, json={"err": ["Your API key is invalid"]}, request=request
                )

        if parts[0] == "start":
            sid = str(uuid.uuid4())
            self.sessions[sid] = {"km": parts[1], "params": rec.params, "facts": [], "answers": []}
            return httpx.Response(
                200, json={"id": sid, "kmVersion": {"id": KM_VERSION_ID}}, request=request
            )

        if parts[0] == "analysis":
            return self._analysis(parts, rec, request)

        if parts[0] == "nl":
            if parts[1] == "interact":
                out = self.interact_bodies.popleft() if self.interact_bodies else INTERACT_RESULT
                return httpx.Response(200, json=out, request=request)
            if parts[1] == "explain":
                if not body.get("factID", "").startswith("WA:"):
                    return httpx.Response(
                        200,
                        json={
                            "error": {
                                "code": 2001,
                                "message": "Fact not found",
                                "statusCode": "404",
                            }
                        },
                        request=request,
                    )
                return httpx.Response(200, json=EXPLAIN_OK, request=request)

        if parts[0] == "maps":
            if "<rbl:kb" not in (body or {}).get("rblang", ""):
                return httpx.Response(
                    200, json={"error": "RBLang parse error: missing rbl:kb root"}, request=request
                )
            return httpx.Response(200, json=self.map_create_body, request=request)

        # session endpoints: /{sid}/inject|query|response|undo
        if len(parts) == 2:
            sid, action = parts
            if sid not in self.sessions:
                return httpx.Response(404, json={"err": ["Session not found"]}, request=request)
            session = self.sessions[sid]
            if action == "inject":
                session["facts"].extend(body or [])
                return httpx.Response(200, json={"result": "OK"}, request=request)
            if action in {"query", "response", "undo"}:
                if action == "response":
                    session["answers"].append(body)
                if not self.decisions:
                    out: Any = RESULT_TOM_SPEAKS_GERMAN
                else:
                    out = self.decisions.popleft()
                if isinstance(out, dict):
                    out = {**out, "sid": sid}
                return httpx.Response(200, json=out, request=request)

        return httpx.Response(404, json={"err": [f"Unknown route {path}"]}, request=request)

    def _analysis(self, parts: list[str], rec: Recorded, request: httpx.Request) -> httpx.Response:
        kind = parts[1]
        if kind == "evidence":
            fact_id = parts[2]
            if (
                self.require_evidence_key
                and rec.headers.get("x-evidence-key") != self.require_evidence_key
            ):
                return httpx.Response(
                    403,
                    json={"err": ["Evidence is secured; provide x-evidence-key"]},
                    request=request,
                )
            node = self.evidence.get(
                httpx.URL(str(request.url)).path.split("/")[3]
            ) or self.evidence.get(fact_id)
            if node is None:
                return httpx.Response(404, json={"err": ["Fact not found"]}, request=request)
            return httpx.Response(200, json=node, request=request)
        if kind == "interactions":
            if (
                self.require_interaction_key
                and rec.headers.get("x-interaction-key") != self.require_interaction_key
            ):
                return httpx.Response(
                    403, json={"err": ["Interaction log is secured"]}, request=request
                )
            if rec.params.get("format", "json").lower() == "csv":
                return httpx.Response(
                    200,
                    text="event,created\nstart,2025-07-10T10:06:27.780Z\n",
                    headers={"content-type": "text/csv"},
                    request=request,
                )
            return httpx.Response(200, json=INTERACTIONS, request=request)
        if kind == "session":
            if rec.params.get("filter", "version") == "facts":
                facts = SESSION_FACTS
                rels = rec.params.get("relationships")
                if rels:
                    wanted = set(rels.split(","))
                    facts = {
                        k: [f for f in v if f["relationship"] in wanted]
                        for k, v in SESSION_FACTS.items()
                    }
                return httpx.Response(200, json=facts, request=request)
            return httpx.Response(200, json=SESSION_VERSION, request=request)
        if kind == "file":
            if parts[2] not in self._known_maps():
                return httpx.Response(
                    404, json={"err": ["Knowledge map not found"]}, request=request
                )
            return httpx.Response(200, json=self.map_source_body, request=request)
        return httpx.Response(404, json={"err": ["Unknown analysis route"]}, request=request)

    def _known_maps(self) -> set[str]:
        return {KM_ID, str((self.map_create_body or {}).get("kmID", ""))}

    def _map_management(
        self, parts: list[str], rec: Recorded, request: httpx.Request
    ) -> httpx.Response:
        """The engine-router routes: ``Version: v1`` first, then the key, then the route."""

        def reply(status: int, body: Any, **headers: str) -> httpx.Response:
            return httpx.Response(status, json=body, headers=headers, request=request)

        def error(status: int, message: str) -> httpx.Response:
            return reply(status, {"error": message})

        version = rec.headers.get("version", "")
        if version != "v1":
            return error(400, f"unsupported API version: {version}")
        key = rec.headers.get("x-api-key")
        if not key:
            return reply(403, {"error": {"type": "NO_PERMISSION"}})
        if key != API_KEY:
            return error(401, "invalid API key")
        body = rec.json if isinstance(rec.json, dict) else {}
        method = rec.method

        if parts == ["validate"] and method == "POST":
            rblang = body.get("rblang")
            if not rblang:
                return error(400, "no rblang specified in request")
            return reply(200, VALIDATE_OK if "<rbl:kb" in rblang else VALIDATE_ERROR)

        if parts[0] == "runs" and method == "GET":
            run_id = parts[1]
            if run_id not in self.test_runs:
                return error(404, "run not found")
            out = self.run_bodies.popleft() if self.run_bodies else TEST_RUN_COMPLETE
            return reply(200, {"runID": run_id, **out} if isinstance(out, dict) else out)

        km_id, action = parts[1], parts[2]
        if action == "runs" and method == "POST":  # accepted even for a map without tests
            run_id = str(uuid.uuid4())
            self.test_runs[run_id] = km_id
            return reply(202, {"runID": run_id}, location=f"/runs/{run_id}")
        if km_id not in self._known_maps():
            return error(404, "map not found")
        if action == "tests" and method == "GET":
            suite = self.test_suites.get(km_id)
            if not suite:
                return error(404, "no tests found")
            if not isinstance(suite, list):
                return error(409, "this map's tests are in a format the API can't read")
            return reply(200, {"kmID": km_id, "testCount": len(suite), "tests": suite})
        if action == "tests" and method == "POST":
            tests = body.get("tests")
            if tests is None:
                return error(400, "tests is required")
            if not isinstance(tests, list) or not tests:
                return error(400, "tests must be a non-empty array")
            if self.test_suites.get(km_id):
                return error(409, "tests already exist for this map; nothing was changed")
            self.test_suites[km_id] = tests
            return reply(201, {"kmID": km_id, "testCount": len(tests)})
        if action == "readme" and method == "POST":
            readme = body.get("readme")
            if not isinstance(readme, str) or not readme.strip():
                return error(400, "readme is required")
            if readme.lstrip("\ufeff \t\r\n").startswith("<"):
                return error(400, "readme must be markdown and can't start with <")
            if self.readmes.get(km_id, "").strip() not in {"", "<p><br></p>"}:
                return error(409, "this map already has a readme; nothing was changed")
            self.readmes[km_id] = readme
            return reply(201, {"kmID": km_id})
        return error(404, f"unknown route {rec.method} {rec.path}")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake() -> FakeRainbird:
    return FakeRainbird()


@pytest.fixture
def client(fake: FakeRainbird) -> Rainbird:
    http = httpx.Client(transport=httpx.MockTransport(fake.handle))
    rb = Rainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http)
    yield rb
    http.close()


@pytest.fixture
def aclient(fake: FakeRainbird) -> AsyncRainbird:
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    return AsyncRainbird(api_key=API_KEY, base_url=BASE_URL, http_client=http)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "RAINBIRD_API_KEY",
        "RAINBIRD_API_URL",
        "RAINBIRD_KM_ID",
        "RAINBIRD_EVIDENCE_KEY",
        "RAINBIRD_INTERACTION_KEY",
        "RAINBIRD_ENGINE",
    ):
        monkeypatch.delenv(var, raising=False)
