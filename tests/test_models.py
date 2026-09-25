"""Unit tests for ``rainbird_ai.models``.

Every model is exercised with ``from_dict`` on the exact ``example:`` blocks from the
OpenAPI specification (transcribed below as ``SPEC_*`` constants) and on the canned
payloads in ``tests/conftest.py``. The final class drives the models through the
sync/async clients against the in-memory fake to check the exact request payloads.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
from typing import Any

import pytest

from rainbird_ai import Rainbird
from rainbird_ai.aio import AsyncRainbird
from rainbird_ai.errors import NaturalLanguageError, RainbirdError, UnexpectedResponseError
from rainbird_ai.models import (
    Answer,
    Concept,
    Condition,
    CreateMapResult,
    Decision,
    DecisionStatus,
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
    Question,
    QuestionForm,
    QuestionResponse,
    ResultItem,
    ResultResponse,
    Rule,
    SessionFact,
    SessionFacts,
    StartResponse,
    _parse_datetime,
    parse_query_response,
    studio_url_for,
)
from tests.conftest import (
    BASE_URL,
    EVIDENCE_LEAVES,
    EVIDENCE_ROOT,
    EXPLAIN_OK,
    INTERACT_ERROR,
    INTERACT_QUESTION,
    INTERACT_RESULT,
    INTERACTIONS,
    KM_ID,
    KM_VERSION_ID,
    MAP_SOURCE,
    QUESTION_FIRST_FORM,
    QUESTION_PLURAL_LOSS,
    QUESTION_SECOND_OBJECT,
    QUESTION_SECOND_SUBJECT,
    RBLANG,
    RESULT_TOM_SPEAKS_GERMAN,
    SESSION_FACTS,
    SESSION_VERSION,
    FakeRainbird,
)

UTC = dt.timezone.utc

# --------------------------------------------------------------------------- spec examples
# Transcribed verbatim from the ``example:`` blocks in /tmp/rainbird_openapi.yaml.

SPEC_SID = "c1bf18d7-d1f0-44bc-a639-8cd4ded56dce"
SPEC_KM_VERSION_ID = "77135768-7634-49f0-9fbf-ed915ce08a66"
SPEC_NL_SID = "bdef015c-0ab4-417c-982f-b8d9ff48efc1"

SPEC_START = {"id": SPEC_SID, "kmVersion": {"id": SPEC_KM_VERSION_ID}}

SPEC_INJECT = {"result": "OK"}

SPEC_QUESTION_RESPONSE: dict[str, Any] = {
    "question": {
        "subject": "Ben",
        "dataType": "string",
        "relationship": "suffered",
        "type": "Second Form Object",
        "plural": True,
        "allowCF": True,
        "allowUnknown": False,
        "canAdd": True,
        "prompt": "What type of loss has Ben suffered?",
        "knownAnswers": [],
        "concepts": [
            {
                "conceptType": "loss",
                "name": "cosmetic damage",
                "type": "string",
                "value": "cosmetic damage",
            },
            {
                "conceptType": "loss",
                "name": "water damage",
                "type": "string",
                "value": "water damage",
            },
            {"conceptType": "loss", "name": "loss", "type": "string", "value": "loss"},
            {"conceptType": "loss", "name": "theft", "type": "string", "value": "theft"},
        ],
    },
    "sid": SPEC_SID,
}

SPEC_RESULT_REPAIR_ID = "WA:RF:82a7b990eb2fe491389f37de10b1cd752cbd692b8c2c0d3fb36b354491f68173"
SPEC_RESULT_REPLACEMENT_ID = (
    "WA:RF:63bc5b884a9256bebd30fa0cd35c3735e8de61f7e34b7859cc9289f1349058ed"
)

SPEC_RESULT_RESPONSE: dict[str, Any] = {
    "result": [
        {
            "subject": "Ben",
            "relationshipType": "can claim",
            "object": "repair",
            "certainty": 100,
            "factID": SPEC_RESULT_REPAIR_ID,
            "relationship": "can claim",
            "objectMetadata": {},
        },
        {
            "subject": "Ben",
            "relationshipType": "can claim",
            "object": "new replacement",
            "certainty": 97,
            "factID": SPEC_RESULT_REPLACEMENT_ID,
            "relationship": "can claim",
            "objectMetadata": {},
        },
    ],
    "queryProfile": [],
    "sid": SPEC_SID,
}

SPEC_COND_HOLDS = "WA:IF:af4d81cc4d1175c0c488d6632599f718520366541ce99e61fc83eff7f640749a"
SPEC_COND_PROVIDES = "WA:KF:c2a63a881f856798e2df8999d7b7956bc28dd50e8e99d6728ca46730955263c1"
SPEC_COND_SUFFERED = "WA:AF:82b2bc07382e0372685aa06e30b1c9db95ccc30051cfc2ea3de4c2e7fc69d0c6"
SPEC_COND_APPROPRIATE = "WA:KF:cb08149770bded892f06d37433c305f4f4667837323167d21580b24de3a9c5dd"
SPEC_COND_INSURES = "WA:KF:ec6e5652be4a48515d68c795b0497e87d6beeb83a384c4ae11ca562c6ba0e499"
SPEC_COND_HAPPENED = "WA:AF:c6a366105ea7691fffae2e75fe27293bd66b4155d352213cee51368272af07b6"
SPEC_COND_COVERS = "WA:KF:7467c2c1c00149ac477f0246173ef9cb19d08243624b52ddf8a7e05791cd5550"
SPEC_FUNC_FACT = "WA:AF:6ee11124ef7abd40707e1813a4d0b7851d86f1c36e42eaf3e559f02f441668e5"
SPEC_FUNC_TEXT = "countRelationshipInstances( 'Ben', 'has previously claimed', *)"


def _spec_func_fact(obj: str) -> dict[str, Any]:
    return {
        "certainty": 100,
        "factID": SPEC_FUNC_FACT,
        "factKey": "5a337648-3fa1-46ec-b67e-7f0aa5662863",
        "object": obj,
        "objectType": "string",
        "relationship": "has previously claimed",
        "subject": "Ben",
    }


SPEC_EVIDENCE: dict[str, Any] = {
    "factID": SPEC_RESULT_REPAIR_ID,
    "source": "rule",
    "fact": {
        "subject": {"type": "policy holder", "value": "Ben", "dataType": "string"},
        "relationship": {"type": "can claim"},
        "object": {"type": "claim", "value": "repair", "dataType": "string"},
        "certainty": 100,
    },
    "time": 1752142054143,
    "rule": {
        "bindings": {
            "EQUIPMENT": "mobile phone",
            "LOSS": "water damage",
            "O": "repair",
            "POLICY": "Gold",
            "S": "Ben",
        },
        "conditions": [
            {
                "certainty": 100,
                "factID": SPEC_COND_HOLDS,
                "factKey": "9c141933-344b-4d36-a3ef-ffdd96fbcb81",
                "impact": 14.29,
                "object": "Gold",
                "objectType": "string",
                "relationship": "holds",
                "salience": 100,
                "subject": "Ben",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_PROVIDES,
                "impact": 14.29,
                "object": "repair",
                "objectType": "string",
                "relationship": "provides",
                "salience": 100,
                "subject": "Gold",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_SUFFERED,
                "factKey": "04b02c60-73c1-436c-b0d3-1ee44df7c14b",
                "impact": 14.29,
                "object": "water damage",
                "objectType": "string",
                "relationship": "suffered",
                "salience": 100,
                "subject": "Ben",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_APPROPRIATE,
                "impact": 14.29,
                "object": "water damage",
                "objectType": "string",
                "relationship": "appropriate for",
                "salience": 100,
                "subject": "repair",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_INSURES,
                "impact": 14.29,
                "object": "mobile phone",
                "objectType": "string",
                "relationship": "insures",
                "salience": 100,
                "subject": "Gold",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_HAPPENED,
                "factKey": "22667649-4df6-4731-9498-e4eb9c25d22c",
                "impact": 14.29,
                "object": "mobile phone",
                "objectType": "string",
                "relationship": "happened to",
                "salience": 100,
                "subject": "water damage",
            },
            {
                "certainty": 100,
                "factID": SPEC_COND_COVERS,
                "impact": 14.29,
                "object": "water damage",
                "objectType": "string",
                "relationship": "covers",
                "salience": 100,
                "subject": "Gold",
            },
            {
                "expression": {
                    "functions": {
                        SPEC_FUNC_TEXT: {
                            "facts": [
                                _spec_func_fact("laptop"),
                                _spec_func_fact("headphones"),
                                _spec_func_fact("bicycle"),
                            ],
                            "result": {"type": "number", "value": 3},
                        }
                    },
                    "text": SPEC_FUNC_TEXT,
                    "value": "%RELS",
                },
                "impact": 14.29,
                "salience": 100,
                "wasMet": True,
            },
            {
                "expression": {"text": "%RELS is less than 10", "value": "%O"},
                "impact": 14.29,
                "salience": 100,
                "wasMet": True,
            },
        ],
        "ruleMaxCertainty": 100,
    },
}

SPEC_INTERACTIONS: list[dict[str, Any]] = [
    {
        "values": {
            "start": {"useDraft": False, "kmVersionID": SPEC_KM_VERSION_ID, "sessionID": SPEC_SID}
        },
        "event": "start",
        "created": "2025-07-10T10:06:27.780Z",
    },
    {
        "values": {
            "facts": [
                {"subject": "Ben", "object": "Gold", "relationship": "holds", "certainty": 100}
            ]
        },
        "event": "inject",
        "created": "2025-07-10T10:06:37.987Z",
    },
    {
        "values": {"query": {"subject": "Ben", "relationship": "can claim"}},
        "event": "query",
        "created": "2025-07-10T10:06:44.532Z",
    },
    {
        "values": {
            "questions": [
                {
                    "prompt": "What type of loss has Ben suffered?",
                    "relationship": "suffered",
                    "subject": "Ben",
                }
            ]
        },
        "event": "question",
        "created": "2025-07-10T10:06:44.713Z",
    },
    {
        "values": {
            "answers": [
                {
                    "subject": "Ben",
                    "object": "water damage",
                    "relationship": "suffered",
                    "certainty": 100,
                }
            ]
        },
        "event": "answer",
        "created": "2025-07-10T10:07:19.252Z",
    },
    {
        "values": {
            "questions": [
                {
                    "prompt": "What item of equipment suffered water damage?",
                    "relationship": "happened to",
                    "subject": "water damage",
                }
            ]
        },
        "event": "question",
        "created": "2025-07-10T10:07:19.523Z",
    },
    {
        "values": {
            "answers": [
                {
                    "subject": "water damage",
                    "object": "mobile phone",
                    "relationship": "happened to",
                    "certainty": 100,
                }
            ]
        },
        "event": "answer",
        "created": "2025-07-10T10:07:34.098Z",
    },
    {
        "values": {
            "results": [
                {
                    "subject": "Ben",
                    "object": "repair",
                    "relationship": "can claim",
                    "certainty": 100,
                },
                {
                    "subject": "Ben",
                    "object": "new replacement",
                    "relationship": "can claim",
                    "certainty": 97,
                },
            ]
        },
        "event": "result",
        "created": "2025-07-10T10:07:34.276Z",
    },
]

SPEC_SESSION_VERSION = {
    "km": {
        "id": SPEC_KM_VERSION_ID,
        "name": "Loss Assessment",
        "versionID": SPEC_KM_VERSION_ID,
        "versionCreated": "2025-07-09T14:27:22Z",
        "versionStatus": "Draft",
    }
}


def _spec_fact(
    fid: str, s_concept: str, s: str, rel: str, o_concept: str, o: str, cert: int, source: str
) -> dict[str, Any]:
    return {
        "id": fid,
        "subject": {"concept": s_concept, "value": s, "dataType": "string"},
        "relationship": rel,
        "object": {"concept": o_concept, "value": o, "dataType": "string"},
        "certainty": cert,
        "source": source,
    }


# The spec's ``facts`` example is very long; these are its first two global facts and the
# first three local facts, verbatim. It has no ``context`` key.
SPEC_SESSION_FACTS: dict[str, Any] = {
    "global": [
        _spec_fact(
            "WA:KF:2d8e855961c42735887a0df394fed28814630c9107d347f965f49703476441b1",
            "claim",
            "new replacement",
            "appropriate for",
            "loss",
            "theft",
            100,
            "knowledgemap",
        ),
        _spec_fact(
            "WA:KF:2f7c748e1665e91b677f28c8314b03e04e31690225106f973836566600e2e0d4",
            "claim",
            "refurbished replacement",
            "appropriate for",
            "loss",
            "water damage",
            80,
            "knowledgemap",
        ),
    ],
    "local": [
        _spec_fact(
            SPEC_RESULT_REPLACEMENT_ID,
            "policy holder",
            "Ben",
            "can claim",
            "claim",
            "new replacement",
            97,
            "rule",
        ),
        _spec_fact(
            SPEC_RESULT_REPAIR_ID,
            "policy holder",
            "Ben",
            "can claim",
            "claim",
            "repair",
            100,
            "rule",
        ),
        _spec_fact(
            SPEC_COND_HAPPENED,
            "loss",
            "water damage",
            "happened to",
            "equipment",
            "mobile phone",
            100,
            "answer",
        ),
    ],
}

SPEC_INTERACT_QUESTION: dict[str, Any] = {
    "responseType": "question",
    "facts": {},
    "metadata": {
        "querySubmittedAt": "2025-07-10T10:49:13.231973361Z",
        "languageEngineResponseTime": 2173,
        "llmTokens": {"completionTokens": 52, "promptTokens": 3117, "totalTokens": 3169},
        "apiVersion": "",
        "endpoint": "interact",
        "sessionID": SPEC_NL_SID,
    },
    "query": {"subject": "Ben", "relationship": "can claim", "object": None},
    "questions": [
        {
            "allowCF": True,
            "canAdd": True,
            "concepts": [
                {"conceptType": "policy", "name": "Gold", "type": "string", "value": "Gold"},
                {"conceptType": "policy", "name": "Silver", "type": "string", "value": "Silver"},
                {"conceptType": "policy", "name": "Bronze", "type": "string", "value": "Bronze"},
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

SPEC_INTERACT_RESULT: dict[str, Any] = {
    "responseType": "result",
    "facts": {
        "injected": [
            {
                "subject": "Ben",
                "relationship": "suffered",
                "object": "water damage",
                "cf": "",
                "certainty": 100,
                "metadata": {"source": "user"},
            },
            {
                "subject": "water damage",
                "relationship": "happened to",
                "object": "mobile phone",
                "cf": "",
                "certainty": 100,
                "metadata": {"source": "user"},
            },
            {
                "subject": "Ben",
                "relationship": "holds",
                "object": "Gold",
                "cf": "",
                "certainty": 100,
                "metadata": {},
            },
        ]
    },
    "metadata": {
        "querySubmittedAt": "2025-07-10T10:59:01.816771016Z",
        "languageEngineResponseTime": 4379,
        "llmTokens": {"completionTokens": 170, "promptTokens": 3229, "totalTokens": 3399},
        "apiVersion": "",
        "endpoint": "interact",
        "sessionID": SPEC_NL_SID,
    },
    "query": {"subject": "Ben", "relationship": "can claim", "object": None},
    "results": [
        {
            "subject": "Ben",
            "object": "repair",
            "certainty": 100,
            "factID": "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea",
            "relationship": "can claim",
        },
        {
            "subject": "Ben",
            "object": "new replacement",
            "certainty": 97,
            "factID": "WA:RF:91d518b8314f29515304e3a2a98bbad9d35ba0c50171a65c3b6c27a01cb645d5",
            "relationship": "can claim",
        },
    ],
}

SPEC_EXPLAIN: dict[str, Any] = {
    "explanation": (
        "Based on the information provided, Ben can claim for a repair. This decision is "
        "supported by several key facts:\n\nBen holds a Gold policy, which provides "
        "comprehensive coverage for his mobile phone. The Gold policy specifically covers "
        "water damage, which is the type of loss Ben has suffered. \n\nThe incident in "
        "question involves water damage to Ben's mobile phone. This aligns with the coverage "
        "provided by his Gold policy, as the policy explicitly insures mobile phones and "
        "covers water damage.\n\nIn response to this type of loss, a repair claim is deemed "
        "appropriate. The Gold policy includes provisions for repair claims, making this a "
        "suitable course of action for addressing the water damage to Ben's mobile "
        "phone.\n\nAll of these factors come together to support the conclusion that Ben is "
        "eligible to claim for a repair of his water-damaged mobile phone under his Gold "
        "policy."
    ),
    "metadata": {
        "querySubmittedAt": "2025-07-10T11:16:12.597111859Z",
        "languageEngineResponseTime": 5538,
        "llmTokens": {"completionTokens": 174, "promptTokens": 726, "totalTokens": 900},
        "apiVersion": "",
        "endpoint": "explain",
        "sessionID": SPEC_NL_SID,
    },
}


# --------------------------------------------------------------------------- helpers


def _variant(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Deep copy ``base`` and apply ``changes`` (a value of ``...`` deletes the key)."""
    data = copy.deepcopy(base)
    for key, value in changes.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    return data


# --------------------------------------------------------------------------- _parse_datetime


class TestParseDatetime:
    def test_z_suffix_is_utc(self) -> None:
        assert _parse_datetime("2025-07-09T14:27:22Z") == dt.datetime(
            2025, 7, 9, 14, 27, 22, tzinfo=UTC
        )

    def test_millisecond_fraction_with_z(self) -> None:
        assert _parse_datetime("2025-07-10T10:06:27.780Z") == dt.datetime(
            2025, 7, 10, 10, 6, 27, 780000, tzinfo=UTC
        )

    def test_nine_digit_fraction_is_truncated_to_microseconds(self) -> None:
        parsed = _parse_datetime("2025-07-10T10:49:13.231973361Z")
        assert parsed == dt.datetime(2025, 7, 10, 10, 49, 13, 231973, tzinfo=UTC)
        assert parsed is not None and parsed.tzinfo is not None

    def test_explicit_offset_is_preserved(self) -> None:
        parsed = _parse_datetime("2025-07-09T14:27:22+01:00")
        assert parsed == dt.datetime(
            2025, 7, 9, 14, 27, 22, tzinfo=dt.timezone(dt.timedelta(hours=1))
        )

    def test_epoch_milliseconds(self) -> None:
        assert _parse_datetime(1751644730474) == dt.datetime(
            2025, 7, 4, 15, 58, 50, 474000, tzinfo=UTC
        )

    def test_epoch_seconds(self) -> None:
        assert _parse_datetime(1751644730) == dt.datetime(2025, 7, 4, 15, 58, 50, tzinfo=UTC)

    def test_float_epoch_seconds(self) -> None:
        assert _parse_datetime(1751644730.474) == dt.datetime(
            2025, 7, 4, 15, 58, 50, 474000, tzinfo=UTC
        )

    def test_naive_string_is_assumed_utc(self) -> None:
        parsed = _parse_datetime("2025-07-09T14:27:22")
        assert parsed == dt.datetime(2025, 7, 9, 14, 27, 22, tzinfo=UTC)
        assert parsed is not None and parsed.tzinfo is UTC

    def test_space_separated_naive_string(self) -> None:
        assert _parse_datetime("2025-07-10 10:06:27") == dt.datetime(
            2025, 7, 10, 10, 6, 27, tzinfo=UTC
        )

    @pytest.mark.parametrize(
        "value", ["not a date", "garbage.value", "1751644730474", "", None, [], {}]
    )
    def test_unparseable_values_give_none(self, value: Any) -> None:
        assert _parse_datetime(value) is None


# --------------------------------------------------------------------------- Environment


class TestEnvironment:
    def test_values_are_the_documented_hosts(self) -> None:
        assert Environment.COMMUNITY.value == "https://api.rainbird.ai"
        assert Environment.ENTERPRISE.value == "https://enterprise-api.rainbird.ai"
        assert Environment.ENTERPRISE_SOUTH_AMERICA.value == "https://sa-enterprise-api.rainbird.ai"

    def test_str_and_format_give_the_url(self) -> None:
        assert str(Environment.COMMUNITY) == "https://api.rainbird.ai"
        assert f"{Environment.ENTERPRISE}" == "https://enterprise-api.rainbird.ai"
        assert Environment.COMMUNITY == "https://api.rainbird.ai"

    def test_lookup_by_value(self) -> None:
        assert Environment("https://api.rainbird.ai") is Environment.COMMUNITY

    @pytest.mark.parametrize(
        ("env", "studio"),
        [
            (Environment.COMMUNITY, "https://app.rainbird.ai"),
            (Environment.ENTERPRISE, "https://enterprise.rainbird.ai"),
            (Environment.ENTERPRISE_SOUTH_AMERICA, "https://sa-enterprise.rainbird.ai"),
        ],
    )
    def test_studio_url(self, env: Environment, studio: str) -> None:
        assert env.studio_url == studio

    @pytest.mark.parametrize(
        ("api", "studio"),
        [
            ("https://api.rainbird.ai", "https://app.rainbird.ai"),
            ("https://api.rainbird.ai/", "https://app.rainbird.ai"),
            ("https://enterprise-api.rainbird.ai", "https://enterprise.rainbird.ai"),
            ("https://sa-enterprise-api.rainbird.ai", "https://sa-enterprise.rainbird.ai"),
            ("https://acme-api.rainbird.ai", "https://acme.rainbird.ai"),
            ("https://acme-api.rainbird.ai/", "https://acme.rainbird.ai"),
            ("acme-api.rainbird.ai", "https://acme.rainbird.ai"),
            ("api.rainbird.ai", "https://app.rainbird.ai"),
            ("http://api.rainbird.ai", "http://app.rainbird.ai"),
            ("https://fake-api.rainbird.ai", "https://fake.rainbird.ai"),
        ],
    )
    def test_studio_url_for(self, api: str, studio: str) -> None:
        assert studio_url_for(api) == studio

    def test_studio_url_for_unknown_host_is_returned_unchanged(self) -> None:
        assert studio_url_for("https://localhost:8080") == "https://localhost:8080"
        assert studio_url_for("https://localhost:8080/") == "https://localhost:8080"

    def test_client_studio_url_and_evidence_link(self, client: Rainbird) -> None:
        assert client.studio_url == "https://fake.rainbird.ai"
        url = client.evidence_url("S1", "WA:RF:abc")
        assert url == f"https://fake.rainbird.ai/evidence?id=WA:RF:abc&api={BASE_URL}&sid=S1"


# --------------------------------------------------------------------------- QuestionForm / FactSource


class TestEnums:
    def test_question_form_values(self) -> None:
        assert QuestionForm.FIRST == "first"
        assert QuestionForm.SECOND_OBJECT == "second_object"
        assert QuestionForm.SECOND_SUBJECT == "second_subject"
        assert QuestionForm.UNKNOWN == "unknown"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("knowledgemap", FactSource.KNOWLEDGE_MAP),
            ("rule", FactSource.RULE),
            ("answer", FactSource.ANSWER),
            ("injection", FactSource.INJECTION),
            ("datasource", FactSource.DATASOURCE),
            ("synthesis", FactSource.SYNTHESIS),
        ],
    )
    def test_fact_source_parse_known(self, value: str, expected: FactSource) -> None:
        parsed = FactSource.parse(value)
        assert parsed is expected
        assert parsed == value

    def test_fact_source_parse_unknown_keeps_string(self) -> None:
        assert FactSource.parse("oracle") == "oracle"
        assert not isinstance(FactSource.parse("oracle"), FactSource)

    def test_fact_source_parse_none(self) -> None:
        assert FactSource.parse(None) is None


# --------------------------------------------------------------------------- Fact / Answer


class TestInputModels:
    def test_fact_payload_matches_inject_schema(self) -> None:
        fact = Fact("Ben", "holds", "Gold")
        assert fact.certainty == 100
        assert fact.to_payload() == {
            "subject": "Ben",
            "relationship": "holds",
            "object": "Gold",
            "certainty": 100,
        }

    def test_fact_payload_keeps_object_type_and_certainty(self) -> None:
        assert Fact("Ben", "age", 42, certainty=80).to_payload() == {
            "subject": "Ben",
            "relationship": "age",
            "object": 42,
            "certainty": 80,
        }

    def test_answer_payload_second_form(self) -> None:
        assert Answer("Ben", "suffered", "water damage").to_payload() == {
            "subject": "Ben",
            "relationship": "suffered",
            "object": "water damage",
            "certainty": 100,
        }

    def test_answer_payload_first_form_includes_yes_no(self) -> None:
        payload = Answer("Tom", "speaks", "French", answer="yes").to_payload()
        assert payload == {
            "subject": "Tom",
            "relationship": "speaks",
            "object": "French",
            "certainty": 100,
            "answer": "yes",
        }

    def test_answer_payload_skip_includes_unanswered_true(self) -> None:
        payload = Answer("Germany", "national language", "", unanswered=True).to_payload()
        assert payload == {
            "subject": "Germany",
            "relationship": "national language",
            "object": "",
            "certainty": 100,
            "unanswered": True,
        }

    def test_answer_payload_omits_unanswered_when_false(self) -> None:
        assert "unanswered" not in Answer("a", "b", "c", unanswered=False).to_payload()
        assert "answer" not in Answer("a", "b", "c").to_payload()

    def test_input_models_are_frozen(self) -> None:
        fact = Fact("a", "b", "c")
        with pytest.raises(dataclasses.FrozenInstanceError):
            fact.subject = "x"  # type: ignore[misc]
        answer = Answer("a", "b", "c")
        with pytest.raises(dataclasses.FrozenInstanceError):
            answer.answer = "yes"  # type: ignore[misc]


# --------------------------------------------------------------------------- StartResponse


class TestStartResponse:
    def test_spec_example(self) -> None:
        start = StartResponse.from_dict(SPEC_START)
        assert start.session_id == SPEC_SID
        assert start.km_version_id == SPEC_KM_VERSION_ID
        assert start.raw is SPEC_START

    def test_missing_km_version_gives_none(self) -> None:
        assert StartResponse.from_dict({"id": "abc"}).km_version_id is None

    def test_km_version_without_id_gives_none(self) -> None:
        assert StartResponse.from_dict({"id": "abc", "kmVersion": {}}).km_version_id is None

    def test_flat_km_version_id_fallback(self) -> None:
        start = StartResponse.from_dict({"id": "abc", "kmVersionID": "v1"})
        assert start.km_version_id == "v1"

    def test_alternate_session_id_keys(self) -> None:
        assert StartResponse.from_dict({"sessionID": "s"}).session_id == "s"
        assert StartResponse.from_dict({"sid": "s"}).session_id == "s"

    def test_non_string_ids_are_stringified(self) -> None:
        start = StartResponse.from_dict({"id": 12, "kmVersion": {"id": 7}})
        assert start.session_id == "12"
        assert start.km_version_id == "7"

    @pytest.mark.parametrize("body", [{}, {"id": ""}, {"id": None}, {"kmVersion": {"id": "v"}}])
    def test_missing_id_raises_value_error(self, body: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="session id"):
            StartResponse.from_dict(body)


# --------------------------------------------------------------------------- InjectResponse


class TestInjectResponse:
    def test_spec_example_is_ok(self) -> None:
        response = InjectResponse.from_dict(SPEC_INJECT)
        assert response.result == "OK"
        assert response.ok is True
        assert response.to_dict() == {"result": "OK"}

    @pytest.mark.parametrize("value", ["OK", "ok", " Ok ", "true", "SUCCESS"])
    def test_ok_variants(self, value: str) -> None:
        assert InjectResponse.from_dict({"result": value}).ok is True

    @pytest.mark.parametrize("value", ["FAIL", "error", "false", "rejected"])
    def test_not_ok_variants(self, value: str) -> None:
        response = InjectResponse.from_dict({"result": value})
        assert response.ok is False
        assert response.result == value

    def test_empty_body_counts_as_ok(self) -> None:
        response = InjectResponse.from_dict({})
        assert response.result is None
        assert response.ok is True

    def test_alternate_keys(self) -> None:
        assert InjectResponse.from_dict({"status": "OK"}).result == "OK"
        assert InjectResponse.from_dict({"ok": True}).result == "True"
        assert InjectResponse.from_dict({"ok": True}).ok is True


# --------------------------------------------------------------------------- Concept


class TestConcept:
    def test_spec_concept(self) -> None:
        concept = Concept.from_dict(SPEC_QUESTION_RESPONSE["question"]["concepts"][0])
        assert concept.name == "cosmetic damage"
        assert concept.value == "cosmetic damage"
        assert concept.concept_type == "loss"
        assert concept.data_type == "string"
        assert concept.invalid_response is False
        assert concept.fsid is None

    def test_name_falls_back_to_value(self) -> None:
        concept = Concept.from_dict({"value": 42, "type": "number"})
        assert concept.name == "42"
        assert concept.value == 42
        assert concept.data_type == "number"

    def test_value_falls_back_to_name(self) -> None:
        assert Concept.from_dict({"name": "Gold"}).value == "Gold"

    def test_invalid_response_and_fsid(self) -> None:
        concept = Concept.from_dict(
            {"name": "x", "value": "x", "invalidResponse": True, "fsid": 3, "dataType": "string"}
        )
        assert concept.invalid_response is True
        assert concept.fsid == 3
        assert concept.data_type == "string"

    def test_invalid_response_string_true(self) -> None:
        assert Concept.from_dict({"name": "x", "invalidResponse": "true"}).invalid_response is True


# --------------------------------------------------------------------------- Question


class TestQuestion:
    def test_spec_example_fields(self) -> None:
        q = Question.from_dict(SPEC_QUESTION_RESPONSE["question"])
        assert q.subject == "Ben"
        assert q.object is None
        assert q.relationship == "suffered"
        assert q.type == "Second Form Object"
        assert q.prompt == "What type of loss has Ben suffered?"
        assert q.data_type == "string"
        assert q.object_type is None
        assert q.plural is True
        assert q.allow_cf is True
        assert q.allow_unknown is False
        assert q.can_add is True
        assert q.known_answers == []
        assert [c.name for c in q.concepts] == ["cosmetic damage", "water damage", "loss", "theft"]
        assert q.options == ["cosmetic damage", "water damage", "loss", "theft"]
        assert q.object_metadata == {}
        assert q.raw is SPEC_QUESTION_RESPONSE["question"]

    def test_first_form_fixture(self) -> None:
        q = Question.from_dict(QUESTION_FIRST_FORM)
        assert q.form is QuestionForm.FIRST
        assert q.is_first_form is True
        assert q.is_second_form is False
        assert q.asks_for_subject is False
        assert q.asks_for_object is False
        assert q.subject == "Tom"
        assert q.object == "French"
        assert q.options == []
        assert q.allow_unknown is True

    def test_second_form_object_fixture(self) -> None:
        q = Question.from_dict(QUESTION_SECOND_OBJECT)
        assert q.form is QuestionForm.SECOND_OBJECT
        assert q.is_first_form is False
        assert q.is_second_form is True
        assert q.asks_for_object is True
        assert q.asks_for_subject is False
        assert q.options == ["English", "French"]
        assert q.plural is False

    def test_second_form_subject_fixture(self) -> None:
        q = Question.from_dict(QUESTION_SECOND_SUBJECT)
        assert q.form is QuestionForm.SECOND_SUBJECT
        assert q.asks_for_subject is True
        assert q.asks_for_object is False
        assert q.is_second_form is True
        assert q.subject is None
        assert q.object == "French"
        assert q.options == ["Tom", "Ana"]
        assert q.plural is True
        assert q.can_add is False

    @pytest.mark.parametrize(
        ("kind", "form"),
        [
            ("First Form", QuestionForm.FIRST),
            ("first form", QuestionForm.FIRST),
            ("Second Form Object", QuestionForm.SECOND_OBJECT),
            ("second form object", QuestionForm.SECOND_OBJECT),
            ("Second Form Subject", QuestionForm.SECOND_SUBJECT),
            ("", QuestionForm.UNKNOWN),
            ("Third Form", QuestionForm.UNKNOWN),
        ],
    )
    def test_form_detection_from_type(self, kind: str, form: QuestionForm) -> None:
        q = Question.from_dict(_variant(QUESTION_FIRST_FORM, type=kind))
        assert q.form is form

    def test_legacy_second_form_with_subject_asks_for_object(self) -> None:
        q = Question.from_dict(_variant(QUESTION_SECOND_OBJECT, type="Second Form"))
        assert q.form is QuestionForm.SECOND_OBJECT
        assert q.asks_for_object is True

    def test_legacy_second_form_with_object_asks_for_subject(self) -> None:
        q = Question.from_dict(_variant(QUESTION_SECOND_SUBJECT, type="Second Form"))
        assert q.form is QuestionForm.SECOND_SUBJECT
        assert q.asks_for_subject is True

    def test_legacy_second_form_with_empty_subject_asks_for_subject(self) -> None:
        q = Question.from_dict(_variant(QUESTION_SECOND_SUBJECT, type="Second Form", subject=""))
        assert q.form is QuestionForm.SECOND_SUBJECT

    def test_legacy_second_form_with_neither_side_defaults_to_object(self) -> None:
        q = Question.from_dict({"type": "Second Form", "relationship": "speaks", "prompt": "?"})
        assert q.form is QuestionForm.SECOND_OBJECT

    def test_missing_type_is_unknown(self) -> None:
        q = Question.from_dict({"relationship": "speaks", "prompt": "?"})
        assert q.type == ""
        assert q.form is QuestionForm.UNKNOWN
        assert q.is_first_form is False
        assert q.is_second_form is False

    def test_options_exclude_invalid_responses(self) -> None:
        data = _variant(QUESTION_SECOND_OBJECT)
        data["concepts"].append(
            {"conceptType": "language", "name": "???", "value": "???", "invalidResponse": True}
        )
        q = Question.from_dict(data)
        assert len(q.concepts) == 3
        assert q.options == ["English", "French"]

    def test_options_use_concept_values(self) -> None:
        q = Question.from_dict(
            {
                "type": "Second Form Object",
                "relationship": "age",
                "dataType": "number",
                "concepts": [{"name": "eighteen", "value": 18}, {"name": "twenty", "value": 20}],
            }
        )
        assert q.options == [18, 20]

    @pytest.mark.parametrize(
        ("data_type", "expected"),
        [("truth", True), ("Truth", True), ("TRUTH", True), ("string", False), (None, False)],
    )
    def test_is_truth(self, data_type: str | None, expected: bool) -> None:
        q = Question.from_dict(_variant(QUESTION_FIRST_FORM, dataType=data_type))
        assert q.is_truth is expected

    def test_str_is_the_prompt(self) -> None:
        assert str(Question.from_dict(QUESTION_FIRST_FORM)) == "Does Tom speak French?"

    def test_str_falls_back_to_triple_when_prompt_missing(self) -> None:
        q = Question.from_dict(_variant(QUESTION_FIRST_FORM, prompt=...))
        assert q.prompt == ""
        assert str(q) == "Tom speaks French"

    def test_defaults_when_flags_missing(self) -> None:
        q = Question.from_dict({"type": "First Form", "relationship": "r"})
        assert q.plural is False
        assert q.allow_cf is False
        assert q.allow_unknown is False
        assert q.can_add is True
        assert q.known_answers == []
        assert q.concepts == []
        assert q.object_metadata == {}
        assert q.data_type is None

    def test_string_booleans_are_parsed(self) -> None:
        q = Question.from_dict(
            {
                "type": "First Form",
                "relationship": "r",
                "plural": "true",
                "allowCF": "false",
                "allowUnknown": "1",
                "canAdd": "no",
            }
        )
        assert q.plural is True
        assert q.allow_cf is False
        assert q.allow_unknown is True
        assert q.can_add is False

    def test_object_type_and_metadata(self) -> None:
        q = Question.from_dict(
            _variant(
                QUESTION_SECOND_OBJECT,
                objectType="language",
                objectMetadata={"hint": "ISO name"},
                knownAnswers=["German"],
            )
        )
        assert q.object_type == "language"
        assert q.object_metadata == {"hint": "ISO name"}
        assert q.known_answers == ["German"]

    def test_interact_question_without_plural_key(self) -> None:
        q = Question.from_dict(SPEC_INTERACT_QUESTION["questions"][0])
        assert q.form is QuestionForm.SECOND_OBJECT
        assert q.plural is False
        assert q.options == ["Gold", "Silver", "Bronze"]
        assert q.prompt == "Which policy does Ben hold?"


# --------------------------------------------------------------------------- QuestionResponse


class TestQuestionResponse:
    def test_spec_example(self) -> None:
        response = QuestionResponse.from_dict(SPEC_QUESTION_RESPONSE)
        assert response.is_question is True
        assert response.is_result is False
        assert response.sid == SPEC_SID
        assert response.question.relationship == "suffered"
        assert response.extra_questions == []
        assert response.questions == [response.question]
        assert response.results == []
        assert response.raw is SPEC_QUESTION_RESPONSE

    def test_extra_questions_follow_the_main_question(self) -> None:
        body = {
            "question": QUESTION_SECOND_OBJECT,
            "extraQuestions": [QUESTION_FIRST_FORM, QUESTION_SECOND_SUBJECT],
            "sid": "S",
        }
        response = QuestionResponse.from_dict(body)
        assert len(response.extra_questions) == 2
        assert [q.type for q in response.questions] == [
            "Second Form Object",
            "First Form",
            "Second Form Subject",
        ]
        assert response.questions[0] is response.question
        assert response.extra_questions[1].asks_for_subject is True

    def test_session_id_key_variant(self) -> None:
        response = QuestionResponse.from_dict({"question": QUESTION_FIRST_FORM, "sessionID": "X"})
        assert response.sid == "X"

    def test_missing_sid_is_none(self) -> None:
        assert QuestionResponse.from_dict({"question": QUESTION_FIRST_FORM}).sid is None

    def test_is_flags_are_not_init_fields(self) -> None:
        names = {f.name for f in dataclasses.fields(QuestionResponse) if f.init}
        assert "is_question" not in names
        assert "is_result" not in names


# --------------------------------------------------------------------------- ResultItem


class TestResultItem:
    def test_spec_lowercase_keys(self) -> None:
        item = ResultItem.from_dict(SPEC_RESULT_RESPONSE["result"][0])
        assert item.subject == "Ben"
        assert item.relationship == "can claim"
        assert item.relationship_type == "can claim"
        assert item.object == "repair"
        assert item.certainty == 100
        assert item.fact_id == SPEC_RESULT_REPAIR_ID
        assert item.object_metadata == {}
        assert item.subject_metadata == {}

    def test_capitalised_keys_from_interact(self) -> None:
        item = ResultItem.from_dict(INTERACT_RESULT["results"][0])
        assert item.subject == "Ben"
        assert item.object == "repair"
        assert item.relationship == "can claim"
        assert item.certainty == 100
        assert item.fact_id == (
            "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea"
        )
        assert item.relationship_type is None

    def test_capitalised_metadata_keys(self) -> None:
        item = ResultItem.from_dict(
            {
                "Subject": "Ben",
                "Relationship": "can claim",
                "Object": "repair",
                "SubjectMetadata": {"a": 1},
                "ObjectMetadata": {"b": 2},
                "RelationshipType": "can claim",
            }
        )
        assert item.subject_metadata == {"a": 1}
        assert item.object_metadata == {"b": 2}
        assert item.relationship_type == "can claim"

    def test_subject_value_and_object_value_fallbacks(self) -> None:
        item = ResultItem.from_dict(
            {
                "SubjectValue": "Ben",
                "ObjectValue": "repair",
                "Relationship": "can claim",
                "Certainty": 90,
                "FactID": "WA:RF:x",
            }
        )
        assert item.subject == "Ben"
        assert item.object == "repair"
        assert item.certainty == 90
        assert item.fact_id == "WA:RF:x"

    def test_value_fallback_when_primary_key_is_null(self) -> None:
        item = ResultItem.from_dict(
            {"Subject": None, "SubjectValue": "Ben", "Object": None, "ObjectValue": 3}
        )
        assert item.subject == "Ben"
        assert item.object == 3

    def test_primary_key_wins_over_value_key(self) -> None:
        item = ResultItem.from_dict({"Subject": "Ben", "SubjectValue": "Benjamin"})
        assert item.subject == "Ben"

    def test_missing_fields(self) -> None:
        item = ResultItem.from_dict({})
        assert item.subject is None
        assert item.object is None
        assert item.relationship == ""
        assert item.certainty is None
        assert item.fact_id is None
        assert str(item) == "? ?"

    def test_certainty_string_and_float(self) -> None:
        assert ResultItem.from_dict({"certainty": "75"}).certainty == 75
        assert ResultItem.from_dict({"certainty": "75.5"}).certainty == 75.5
        assert ResultItem.from_dict({"certainty": 75.5}).certainty == 75.5

    def test_fact_id_camel_variant(self) -> None:
        assert ResultItem.from_dict({"factId": "WA:RF:y"}).fact_id == "WA:RF:y"

    def test_str_includes_certainty(self) -> None:
        item = ResultItem.from_dict(RESULT_TOM_SPEAKS_GERMAN["result"][0])
        assert str(item) == "Tom speaks German (75%)"

    def test_str_without_certainty(self) -> None:
        item = ResultItem.from_dict(
            {"subject": "Tom", "relationship": "speaks", "object": "German"}
        )
        assert str(item) == "Tom speaks German"


# --------------------------------------------------------------------------- ResultResponse


class TestResultResponse:
    def test_spec_example(self) -> None:
        response = ResultResponse.from_dict(SPEC_RESULT_RESPONSE)
        assert response.is_result is True
        assert response.is_question is False
        assert response.question is None
        assert response.sid == SPEC_SID
        assert len(response) == 2
        assert [r.object for r in response] == ["repair", "new replacement"]
        assert response.fact_ids == [SPEC_RESULT_REPAIR_ID, SPEC_RESULT_REPLACEMENT_ID]
        assert response.raw is SPEC_RESULT_RESPONSE

    def test_best_is_highest_certainty(self) -> None:
        response = ResultResponse.from_dict(SPEC_RESULT_RESPONSE)
        assert response.best is not None
        assert response.best.object == "repair"
        assert response.best.certainty == 100

    def test_best_picks_max_regardless_of_order(self) -> None:
        body = {"result": list(reversed(SPEC_RESULT_RESPONSE["result"]))}
        response = ResultResponse.from_dict(body)
        assert response.results[0].certainty == 97
        assert response.best is not None
        assert response.best.certainty == 100

    def test_best_tie_returns_first(self) -> None:
        body = {
            "result": [
                {"subject": "a", "relationship": "r", "object": "x", "certainty": 100},
                {"subject": "a", "relationship": "r", "object": "y", "certainty": 100},
            ]
        }
        best = ResultResponse.from_dict(body).best
        assert best is not None
        assert best.object == "x"

    def test_best_treats_missing_certainty_as_lowest(self) -> None:
        body = {
            "result": [
                {"subject": "a", "relationship": "r", "object": "unknown"},
                {"subject": "a", "relationship": "r", "object": "low", "certainty": 1},
            ]
        }
        best = ResultResponse.from_dict(body).best
        assert best is not None
        assert best.object == "low"

    def test_empty_result(self) -> None:
        response = ResultResponse.from_dict({"result": [], "sid": "S"})
        assert response.results == []
        assert response.best is None
        assert response.fact_ids == []
        assert len(response) == 0
        assert list(response) == []
        assert response.is_result is True

    def test_fact_ids_skip_missing(self) -> None:
        body = {
            "result": [
                {"subject": "a", "relationship": "r", "object": "x", "factID": "WA:RF:1"},
                {"subject": "a", "relationship": "r", "object": "y"},
                {"subject": "a", "relationship": "r", "object": "z", "factID": ""},
            ]
        }
        assert ResultResponse.from_dict(body).fact_ids == ["WA:RF:1"]

    def test_fixture_result(self) -> None:
        response = ResultResponse.from_dict(RESULT_TOM_SPEAKS_GERMAN)
        assert response.sid is None
        assert response.best is not None
        assert response.best.fact_id == EVIDENCE_ROOT["factID"]
        assert response.best.relationship_type == "speaks"

    def test_iteration_yields_result_items(self) -> None:
        response = ResultResponse.from_dict(SPEC_RESULT_RESPONSE)
        for item in response:
            assert isinstance(item, ResultItem)


# --------------------------------------------------------------------------- parse_query_response


class TestParseQueryResponse:
    def test_question_body(self) -> None:
        parsed = parse_query_response(SPEC_QUESTION_RESPONSE)
        assert isinstance(parsed, QuestionResponse)
        assert parsed.is_question

    def test_result_body(self) -> None:
        parsed = parse_query_response(SPEC_RESULT_RESPONSE)
        assert isinstance(parsed, ResultResponse)
        assert parsed.is_result

    def test_empty_result_list_is_a_result(self) -> None:
        parsed = parse_query_response({"result": [], "sid": "S"})
        assert isinstance(parsed, ResultResponse)
        assert len(parsed) == 0

    def test_question_takes_precedence_over_result(self) -> None:
        parsed = parse_query_response({"question": QUESTION_FIRST_FORM, "result": []})
        assert isinstance(parsed, QuestionResponse)

    def test_null_question_with_result_is_a_result(self) -> None:
        parsed = parse_query_response({"question": None, "result": []})
        assert isinstance(parsed, ResultResponse)

    def test_neither_raises_unexpected_response_listing_keys(self) -> None:
        body = {"foo": 1, "err": ["bad"]}
        with pytest.raises(UnexpectedResponseError, match=r"neither.*\['err', 'foo'\]") as ei:
            parse_query_response(body)
        assert ei.value.body is body
        assert ei.value.status_code is None  # no HTTP response involved at this level

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(UnexpectedResponseError):
            parse_query_response({})

    @pytest.mark.parametrize("body", [None, [], "text", 42, [{"result": []}]])
    def test_non_dict_raises(self, body: Any) -> None:
        with pytest.raises(UnexpectedResponseError, match="unexpected response body type") as ei:
            parse_query_response(body)
        assert ei.value.body == body

    def test_question_that_is_not_a_dict_and_no_result_raises(self) -> None:
        with pytest.raises(UnexpectedResponseError):
            parse_query_response({"question": "what?"})

    def test_does_not_raise_a_bare_value_error(self) -> None:
        # Public code paths raise RainbirdError subclasses only (UnexpectedResponseError is not
        # a ValueError), so callers can catch RainbirdError uniformly.
        with pytest.raises(RainbirdError):
            parse_query_response({"foo": 1})
        assert not issubclass(UnexpectedResponseError, ValueError)


# --------------------------------------------------------------------------- FactValue / SessionFact


class TestFactValue:
    def test_session_style_dict(self) -> None:
        value = FactValue.from_dict({"concept": "country", "value": "France", "dataType": "string"})
        assert value.value == "France"
        assert value.concept == "country"
        assert value.data_type == "string"
        assert str(value) == "France"

    def test_evidence_style_dict_uses_type_as_concept(self) -> None:
        value = FactValue.from_dict({"type": "person", "value": "Tom", "dataType": "string"})
        assert value.concept == "person"
        assert value.value == "Tom"

    def test_scalar_becomes_value(self) -> None:
        value = FactValue.from_dict("Tom")
        assert value.value == "Tom"
        assert value.concept is None
        assert value.data_type is None
        assert value.raw == {"value": "Tom"}
        assert value.to_dict() == {"value": "Tom"}

    def test_none(self) -> None:
        value = FactValue.from_dict(None)
        assert value.value is None
        assert str(value) == "None"

    def test_number_value(self) -> None:
        value = FactValue.from_dict({"concept": "age", "value": 42, "dataType": "number"})
        assert value.value == 42
        assert str(value) == "42"


class TestSessionFact:
    def test_spec_global_fact(self) -> None:
        fact = SessionFact.from_dict(SPEC_SESSION_FACTS["global"][1])
        assert fact.id == "WA:KF:2f7c748e1665e91b677f28c8314b03e04e31690225106f973836566600e2e0d4"
        assert fact.subject.concept == "claim"
        assert fact.subject.value == "refurbished replacement"
        assert fact.relationship == "appropriate for"
        assert fact.object.concept == "loss"
        assert fact.object.value == "water damage"
        assert fact.object.data_type == "string"
        assert fact.certainty == 80
        assert fact.source is FactSource.KNOWLEDGE_MAP
        assert str(fact) == "refurbished replacement appropriate for water damage (80%)"

    def test_fixture_local_fact(self) -> None:
        fact = SessionFact.from_dict(SESSION_FACTS["local"][0])
        assert fact.id == "WA:IF:local1"
        assert fact.source is FactSource.INJECTION
        assert str(fact) == "Tom lives in Germany (100%)"

    def test_evidence_style_relationship_dict(self) -> None:
        fact = SessionFact.from_dict(EVIDENCE_ROOT["fact"])
        assert fact.relationship == "speaks"
        assert fact.subject.concept == "person"
        assert fact.object.value == "German"
        assert fact.certainty == 75
        assert fact.id is None
        assert fact.source is None

    def test_fact_id_key_variant(self) -> None:
        fact = SessionFact.from_dict({"factID": "WA:RF:1", "relationship": "r"})
        assert fact.id == "WA:RF:1"

    def test_unknown_source_is_kept_as_string(self) -> None:
        fact = SessionFact.from_dict({"relationship": "r", "source": "oracle"})
        assert fact.source == "oracle"

    def test_missing_relationship_is_empty_string(self) -> None:
        fact = SessionFact.from_dict({})
        assert fact.relationship == ""
        assert fact.certainty is None
        assert str(fact) == "? ?"

    def test_str_without_certainty(self) -> None:
        fact = SessionFact.from_dict(
            {"subject": {"value": "Tom"}, "relationship": "speaks", "object": {"value": "German"}}
        )
        assert str(fact) == "Tom speaks German"


# --------------------------------------------------------------------------- SessionFacts


class TestSessionFacts:
    def test_fixture_groups(self) -> None:
        facts = SessionFacts.from_dict(SESSION_FACTS)
        assert len(facts.global_) == 1
        assert len(facts.local) == 1
        assert facts.context == []
        assert len(facts) == 2
        assert facts.raw is SESSION_FACTS

    def test_all_orders_global_context_local(self) -> None:
        data = _variant(
            SESSION_FACTS,
            context=[
                {
                    "id": "WA:CF:ctx1",
                    "subject": {"concept": "person", "value": "Ana", "dataType": "string"},
                    "relationship": "lives in",
                    "object": {"concept": "country", "value": "France", "dataType": "string"},
                    "certainty": 100,
                    "source": "injection",
                }
            ],
        )
        facts = SessionFacts.from_dict(data)
        assert [f.id for f in facts.all] == ["WA:KF:global1", "WA:CF:ctx1", "WA:IF:local1"]
        assert len(facts) == 3
        assert [f.id for f in facts] == [f.id for f in facts.all]

    def test_spec_example_without_context_key(self) -> None:
        facts = SessionFacts.from_dict(SPEC_SESSION_FACTS)
        assert len(facts.global_) == 2
        assert len(facts.local) == 3
        assert facts.context == []
        assert len(facts) == 5
        assert [f.source for f in facts.local] == [
            FactSource.RULE,
            FactSource.RULE,
            FactSource.ANSWER,
        ]
        assert facts.local[1].id == SPEC_RESULT_REPAIR_ID

    def test_facts_wrapper(self) -> None:
        wrapped = {"facts": SESSION_FACTS}
        facts = SessionFacts.from_dict(wrapped)
        assert len(facts) == 2
        assert facts.global_[0].id == "WA:KF:global1"
        assert facts.raw is wrapped

    def test_empty_body(self) -> None:
        facts = SessionFacts.from_dict({})
        assert facts.all == []
        assert len(facts) == 0
        assert list(facts) == []

    def test_non_list_scope_is_wrapped(self) -> None:
        facts = SessionFacts.from_dict({"local": SESSION_FACTS["local"][0]})
        assert len(facts.local) == 1


# --------------------------------------------------------------------------- KnowledgeMapVersion


class TestKnowledgeMapVersion:
    def test_spec_example_is_draft(self) -> None:
        km = KnowledgeMapVersion.from_dict(SPEC_SESSION_VERSION)
        assert km.id == SPEC_KM_VERSION_ID
        assert km.km_id == SPEC_KM_VERSION_ID
        assert km.name == "Loss Assessment"
        assert km.version_id == SPEC_KM_VERSION_ID
        assert km.version_status == "Draft"
        assert km.is_draft is True
        assert km.version_number is None
        assert km.version_created == dt.datetime(2025, 7, 9, 14, 27, 22, tzinfo=UTC)
        assert km.raw is SPEC_SESSION_VERSION

    def test_fixture_is_version_3(self) -> None:
        km = KnowledgeMapVersion.from_dict(SESSION_VERSION)
        assert km.id == KM_ID
        assert km.km_id == KM_ID
        assert km.name == "Languages"
        assert km.version_id == KM_VERSION_ID
        assert km.version_status == "Version3"
        assert km.is_draft is False
        assert km.version_number == 3

    @pytest.mark.parametrize(
        ("status", "number", "draft"),
        [
            ("Version1", 1, False),
            ("Version12", 12, False),
            ("version7", 7, False),
            ("Draft", None, True),
            ("draft", None, True),
            ("Version", None, False),
            ("Live", None, False),
            (None, None, False),
        ],
    )
    def test_version_number_and_is_draft(
        self, status: str | None, number: int | None, draft: bool
    ) -> None:
        body = {"km": {"id": "k", "versionStatus": status}}
        km = KnowledgeMapVersion.from_dict(body)
        assert km.version_number == number
        assert km.is_draft is draft

    def test_km_id_is_none_when_id_missing(self) -> None:
        km = KnowledgeMapVersion.from_dict({"km": {"versionStatus": "Draft"}})
        assert km.id is None
        assert km.km_id is None

    def test_unwrapped_km_body(self) -> None:
        km = KnowledgeMapVersion.from_dict(SESSION_VERSION["km"])
        assert km.id == KM_ID
        assert km.version_number == 3

    def test_version_id_camel_variant(self) -> None:
        km = KnowledgeMapVersion.from_dict({"km": {"versionId": "v"}})
        assert km.version_id == "v"

    def test_missing_fields(self) -> None:
        km = KnowledgeMapVersion.from_dict({})
        assert km.id is None
        assert km.name is None
        assert km.version_id is None
        assert km.version_status is None
        assert km.version_created is None


# --------------------------------------------------------------------------- Condition / ExpressionCondition / Rule


class TestConditions:
    def test_relationship_condition_fixture(self) -> None:
        cond = Condition.from_dict(EVIDENCE_ROOT["rule"]["conditions"][0])
        assert cond.subject == "Tom"
        assert cond.relationship == "lives in"
        assert cond.object == "Germany"
        assert cond.certainty == 100
        assert cond.salience == 100
        assert cond.impact == 37.5
        assert cond.fact_id == (
            "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e"
        )
        assert cond.fact_key == "414fcb69-0a58-4cde-a23a-be4b92a7bd54"
        assert cond.object_type == "string"
        assert cond.alt is None

    def test_relationship_condition_spec_without_fact_key(self) -> None:
        cond = Condition.from_dict(SPEC_EVIDENCE["rule"]["conditions"][1])
        assert cond.fact_id == SPEC_COND_PROVIDES
        assert cond.fact_key is None
        assert cond.impact == 14.29
        assert cond.subject == "Gold"
        assert cond.relationship == "provides"
        assert cond.object == "repair"

    def test_condition_alt_and_missing_fields(self) -> None:
        cond = Condition.from_dict({"alt": "or-branch", "factKey": 12})
        assert cond.alt == "or-branch"
        assert cond.fact_key == "12"
        assert cond.subject is None
        assert cond.relationship == ""
        assert cond.certainty is None
        assert cond.impact is None

    def test_expression_condition_fixture(self) -> None:
        cond = ExpressionCondition.from_dict(EVIDENCE_ROOT["rule"]["conditions"][2])
        assert cond.text == "countRelationshipInstances( 'Tom', 'lives in', *)"
        assert cond.value == "%N"
        assert cond.was_met is True
        assert cond.impact == 25
        assert cond.salience == 100
        assert cond.alt is None
        assert list(cond.functions) == ["countRelationshipInstances( 'Tom', 'lives in', *)"]
        assert cond.fact_ids == [
            "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e"
        ]

    def test_expression_condition_spec_function_facts(self) -> None:
        cond = ExpressionCondition.from_dict(SPEC_EVIDENCE["rule"]["conditions"][7])
        assert cond.text == SPEC_FUNC_TEXT
        assert cond.value == "%RELS"
        assert cond.was_met is True
        assert cond.functions[SPEC_FUNC_TEXT]["result"] == {"type": "number", "value": 3}
        # one id per fact, not de-duplicated at this level
        assert cond.fact_ids == [SPEC_FUNC_FACT, SPEC_FUNC_FACT, SPEC_FUNC_FACT]

    def test_expression_condition_without_functions(self) -> None:
        cond = ExpressionCondition.from_dict(SPEC_EVIDENCE["rule"]["conditions"][8])
        assert cond.text == "%RELS is less than 10"
        assert cond.value == "%O"
        assert cond.functions == {}
        assert cond.fact_ids == []
        assert cond.was_met is True

    def test_expression_condition_was_met_missing_is_none(self) -> None:
        cond = ExpressionCondition.from_dict({"expression": {"text": "x"}})
        assert cond.was_met is None
        assert cond.value is None

    def test_expression_condition_was_met_false_and_string(self) -> None:
        assert ExpressionCondition.from_dict({"expression": {}, "wasMet": False}).was_met is False
        assert ExpressionCondition.from_dict({"expression": {}, "wasMet": "true"}).was_met is True

    def test_rule_fixture_splits_condition_kinds(self) -> None:
        rule = Rule.from_dict(EVIDENCE_ROOT["rule"])
        assert rule.bindings == {"COUNTRY": "Germany", "O": "German", "S": "Tom"}
        assert rule.max_certainty == 75
        assert len(rule.conditions) == 3
        assert len(rule.relationship_conditions) == 2
        assert len(rule.expression_conditions) == 1
        assert all(isinstance(c, Condition) for c in rule.relationship_conditions)
        assert all(isinstance(c, ExpressionCondition) for c in rule.expression_conditions)
        assert [c.relationship for c in rule.relationship_conditions] == [
            "lives in",
            "national language",
        ]

    def test_rule_spec_example(self) -> None:
        rule = Rule.from_dict(SPEC_EVIDENCE["rule"])
        assert rule.bindings["POLICY"] == "Gold"
        assert rule.max_certainty == 100
        assert len(rule.conditions) == 9
        assert len(rule.relationship_conditions) == 7
        assert len(rule.expression_conditions) == 2
        assert isinstance(rule.conditions[7], ExpressionCondition)
        assert isinstance(rule.conditions[0], Condition)

    def test_rule_preserves_condition_order(self) -> None:
        rule = Rule.from_dict(SPEC_EVIDENCE["rule"])
        kinds = ["expr" if isinstance(c, ExpressionCondition) else "rel" for c in rule.conditions]
        assert kinds == ["rel"] * 7 + ["expr"] * 2

    def test_empty_rule(self) -> None:
        rule = Rule.from_dict({})
        assert rule.bindings == {}
        assert rule.conditions == []
        assert rule.max_certainty is None
        assert rule.relationship_conditions == []
        assert rule.expression_conditions == []


# --------------------------------------------------------------------------- Evidence


class TestEvidence:
    def test_spec_example(self) -> None:
        ev = Evidence.from_dict(SPEC_EVIDENCE)
        assert ev.fact_id == SPEC_RESULT_REPAIR_ID
        assert ev.source is FactSource.RULE
        assert ev.is_inferred is True
        assert ev.time == 1752142054143
        assert ev.timestamp == dt.datetime(2025, 7, 10, 10, 7, 34, 143000, tzinfo=UTC)
        assert ev.fact.id == SPEC_RESULT_REPAIR_ID
        assert ev.fact.subject.value == "Ben"
        assert ev.fact.subject.concept == "policy holder"
        assert ev.fact.relationship == "can claim"
        assert ev.fact.object.value == "repair"
        assert ev.fact.object.concept == "claim"
        assert ev.fact.certainty == 100
        assert ev.rule is not None
        assert ev.rule.max_certainty == 100
        assert ev.raw is SPEC_EVIDENCE

    def test_spec_child_fact_ids_include_function_facts_deduplicated(self) -> None:
        ev = Evidence.from_dict(SPEC_EVIDENCE)
        assert ev.child_fact_ids == [
            SPEC_COND_HOLDS,
            SPEC_COND_PROVIDES,
            SPEC_COND_SUFFERED,
            SPEC_COND_APPROPRIATE,
            SPEC_COND_INSURES,
            SPEC_COND_HAPPENED,
            SPEC_COND_COVERS,
            SPEC_FUNC_FACT,
        ]
        assert len(set(ev.child_fact_ids)) == len(ev.child_fact_ids)

    def test_fixture_root(self) -> None:
        ev = Evidence.from_dict(EVIDENCE_ROOT)
        assert ev.fact_id == EVIDENCE_ROOT["factID"]
        assert ev.source is FactSource.RULE
        assert ev.fact.subject.value == "Tom"
        assert ev.fact.relationship == "speaks"
        assert ev.fact.object.value == "German"
        assert ev.fact.certainty == 75
        assert ev.timestamp == dt.datetime(2025, 7, 4, 15, 58, 50, 474000, tzinfo=UTC)
        assert str(ev.fact) == "Tom speaks German (75%)"

    def test_fixture_child_fact_ids_deduplicate_expression_facts(self) -> None:
        ev = Evidence.from_dict(EVIDENCE_ROOT)
        # The expression's function fact repeats the first relationship condition's factID.
        assert ev.child_fact_ids == [
            "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e",
            "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681",
        ]
        assert set(ev.child_fact_ids) == set(EVIDENCE_LEAVES)

    def test_child_fact_ids_exclude_self(self) -> None:
        data = _variant(EVIDENCE_ROOT)
        data["rule"]["conditions"][0]["factID"] = EVIDENCE_ROOT["factID"]
        ev = Evidence.from_dict(data)
        assert EVIDENCE_ROOT["factID"] not in ev.child_fact_ids
        assert ev.child_fact_ids == [
            "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681",
            "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e",
        ]

    def test_child_fact_ids_skip_conditions_without_fact_id(self) -> None:
        data = _variant(EVIDENCE_ROOT)
        del data["rule"]["conditions"][1]["factID"]
        ev = Evidence.from_dict(data)
        assert ev.child_fact_ids == [
            "WA:IF:4b51e33a91a6d13203ef5f4a041b2bc079903d00e69e262829742403aff43f0e"
        ]

    def test_leaf_has_no_children(self) -> None:
        leaf_id = next(iter(EVIDENCE_LEAVES))
        ev = Evidence.from_dict(EVIDENCE_LEAVES[leaf_id])
        assert ev.rule is None
        assert ev.is_inferred is False
        assert ev.child_fact_ids == []
        assert ev.source is FactSource.INJECTION
        assert ev.fact.id == leaf_id
        assert ev.timestamp == dt.datetime(2025, 7, 4, 15, 58, 50, tzinfo=UTC)

    def test_answer_leaf_source(self) -> None:
        leaf = EVIDENCE_LEAVES[
            "WA:AF:5302d24b5e369eac902119aedce0c076a39096932ab97f9a78e0490b2cb02681"
        ]
        ev = Evidence.from_dict(leaf)
        assert ev.source is FactSource.ANSWER
        assert str(ev.fact) == "Germany national language German (100%)"

    def test_missing_time_gives_none_timestamp(self) -> None:
        ev = Evidence.from_dict(_variant(EVIDENCE_ROOT, time=...))
        assert ev.time is None
        assert ev.timestamp is None

    def test_iso_time_string_is_parsed(self) -> None:
        ev = Evidence.from_dict(_variant(EVIDENCE_ROOT, time="2025-07-04T15:58:50Z"))
        assert ev.time is None  # not numeric
        assert ev.timestamp is None

    def test_rule_that_is_not_a_dict_is_ignored(self) -> None:
        ev = Evidence.from_dict(_variant(EVIDENCE_ROOT, rule=None))
        assert ev.rule is None
        assert ev.child_fact_ids == []

    def test_fact_id_key_variants(self) -> None:
        assert Evidence.from_dict({"factId": "WA:RF:1"}).fact_id == "WA:RF:1"
        assert Evidence.from_dict({"id": "WA:RF:2"}).fact_id == "WA:RF:2"
        assert Evidence.from_dict({}).fact_id == ""


# --------------------------------------------------------------------------- InteractionEvent


class TestInteractionEvent:
    def test_spec_example_events(self) -> None:
        events = [InteractionEvent.from_dict(e) for e in SPEC_INTERACTIONS]
        assert [e.event for e in events] == [
            "start",
            "inject",
            "query",
            "question",
            "answer",
            "question",
            "answer",
            "result",
        ]
        assert events[0].values["start"]["kmVersionID"] == SPEC_KM_VERSION_ID
        assert events[1].values["facts"][0]["object"] == "Gold"
        assert events[7].values["results"][1]["certainty"] == 97
        assert events[0].created == dt.datetime(2025, 7, 10, 10, 6, 27, 780000, tzinfo=UTC)
        assert events[7].created == dt.datetime(2025, 7, 10, 10, 7, 34, 276000, tzinfo=UTC)

    def test_created_is_monotonic_in_spec_example(self) -> None:
        created = [InteractionEvent.from_dict(e).created for e in SPEC_INTERACTIONS]
        assert all(c is not None for c in created)
        assert created == sorted(created)  # type: ignore[type-var]

    def test_fixture_events(self) -> None:
        events = [InteractionEvent.from_dict(e) for e in INTERACTIONS]
        assert [e.event for e in events] == ["start", "query", "result"]
        assert events[1].values == {"query": {"subject": "Tom", "relationship": "speaks"}}
        assert events[0].raw is INTERACTIONS[0]

    def test_missing_fields(self) -> None:
        event = InteractionEvent.from_dict({})
        assert event.event == ""
        assert event.created is None
        assert event.values == {}

    def test_non_dict_values_become_empty(self) -> None:
        assert InteractionEvent.from_dict({"event": "x", "values": "oops"}).values == {}


# --------------------------------------------------------------------------- NLError / NLMetadata


class TestNLError:
    def test_fixture_error(self) -> None:
        err = NLError.from_dict(INTERACT_ERROR["error"])
        assert err.code == 1001
        assert err.message == "Could not detect a query in the prompt"
        assert err.suggested_chat_response == (
            "I couldn't work out what you want to know. Could you rephrase?"
        )
        assert err.status_code == "400"

    def test_string_code_is_converted(self) -> None:
        assert NLError.from_dict({"code": "2001", "message": "m"}).code == 2001

    def test_error_key_as_message_fallback(self) -> None:
        assert NLError.from_dict({"error": "boom"}).message == "boom"

    def test_missing_fields(self) -> None:
        err = NLError.from_dict({})
        assert err.code is None
        assert err.message == ""
        assert err.suggested_chat_response is None
        assert err.status_code is None


class TestNLMetadata:
    def test_spec_interact_metadata(self) -> None:
        meta = NLMetadata.from_dict(SPEC_INTERACT_QUESTION["metadata"])
        assert meta.endpoint == "interact"
        assert meta.session_id == SPEC_NL_SID
        assert meta.query_submitted_at == dt.datetime(2025, 7, 10, 10, 49, 13, 231973, tzinfo=UTC)
        assert meta.language_engine_response_ms == 2173
        assert meta.completion_tokens == 52
        assert meta.prompt_tokens == 3117
        assert meta.total_tokens == 3169
        assert meta.api_version == ""

    def test_spec_explain_metadata(self) -> None:
        meta = NLMetadata.from_dict(SPEC_EXPLAIN["metadata"])
        assert meta.endpoint == "explain"
        assert meta.language_engine_response_ms == 5538
        assert (meta.completion_tokens, meta.prompt_tokens, meta.total_tokens) == (174, 726, 900)
        assert meta.query_submitted_at == dt.datetime(2025, 7, 10, 11, 16, 12, 597111, tzinfo=UTC)

    def test_token_totals_are_consistent_in_spec(self) -> None:
        meta = NLMetadata.from_dict(SPEC_INTERACT_RESULT["metadata"])
        assert meta.prompt_tokens is not None and meta.completion_tokens is not None
        assert meta.prompt_tokens + meta.completion_tokens == meta.total_tokens == 3399

    def test_missing_tokens_give_none(self) -> None:
        meta = NLMetadata.from_dict(INTERACT_RESULT["metadata"])
        assert meta.endpoint == "interact"
        assert meta.session_id == "S"
        assert meta.prompt_tokens is None
        assert meta.completion_tokens is None
        assert meta.total_tokens is None
        assert meta.language_engine_response_ms is None
        assert meta.query_submitted_at is None
        assert meta.api_version is None

    def test_string_numbers_are_converted_to_int(self) -> None:
        meta = NLMetadata.from_dict(
            {
                "languageEngineResponseTime": "12.7",
                "llmTokens": {"promptTokens": "3", "completionTokens": 4.0, "totalTokens": "7"},
            }
        )
        assert meta.language_engine_response_ms == 12
        assert meta.prompt_tokens == 3
        assert meta.completion_tokens == 4
        assert meta.total_tokens == 7


# --------------------------------------------------------------------------- InteractResponse


class TestInteractResponse:
    def test_spec_question(self) -> None:
        r = InteractResponse.from_dict(SPEC_INTERACT_QUESTION)
        assert r.response_type == "question"
        assert r.is_question is True
        assert r.is_result is False
        assert r.is_error is False
        assert r.query == {"subject": "Ben", "relationship": "can claim", "object": None}
        assert len(r.questions) == 1
        assert r.question is r.questions[0]
        assert r.question is not None
        assert r.question.prompt == "Which policy does Ben hold?"
        assert r.question.options == ["Gold", "Silver", "Bronze"]
        assert r.prompt == "Which policy does Ben hold?"
        assert r.results == []
        assert r.error is None
        assert r.metadata is not None
        assert r.metadata.total_tokens == 3169
        assert r.injected_facts == []
        assert r.invalid_facts == []
        assert r.unmatched == []
        assert r.raw is SPEC_INTERACT_QUESTION

    def test_spec_result_lowercase_keys(self) -> None:
        r = InteractResponse.from_dict(SPEC_INTERACT_RESULT)
        assert r.is_result is True
        assert r.is_question is False
        assert r.is_error is False
        assert r.question is None
        assert r.prompt is None
        assert [x.object for x in r.results] == ["repair", "new replacement"]
        assert [x.certainty for x in r.results] == [100, 97]
        assert r.results[0].fact_id == (
            "WA:RF:e1d3a2380a814cea00d5a1adf69b2cdafba0e0228192cd8014400925bb4422ea"
        )
        assert len(r.injected_facts) == 3
        assert r.injected_facts[0]["metadata"] == {"source": "user"}
        assert r.invalid_facts == []
        assert r.unmatched == []
        assert r.metadata is not None
        assert r.metadata.language_engine_response_ms == 4379

    def test_fixture_result_capitalised_keys(self) -> None:
        r = InteractResponse.from_dict(INTERACT_RESULT)
        assert r.is_result is True
        assert len(r.results) == 1
        assert r.results[0].subject == "Ben"
        assert r.results[0].object == "repair"
        assert r.results[0].relationship == "can claim"
        assert r.results[0].certainty == 100
        assert r.injected_facts == INTERACT_RESULT["facts"]["injected"]
        assert r.invalid_facts == []
        assert r.unmatched == ["and he likes tea"]

    def test_fixture_question(self) -> None:
        r = InteractResponse.from_dict(INTERACT_QUESTION)
        assert r.is_question is True
        assert r.prompt == "Which policy does Ben hold?"
        assert r.question is not None
        assert r.question.asks_for_object is True
        assert r.metadata is not None
        assert r.metadata.prompt_tokens == 3117

    def test_fixture_error(self) -> None:
        r = InteractResponse.from_dict(INTERACT_ERROR)
        assert r.response_type == "error"
        assert r.is_error is True
        assert r.is_question is False
        assert r.is_result is False
        assert r.error is not None
        assert r.error.code == 1001
        assert r.error.status_code == "400"
        assert r.question is None
        assert r.results == []
        assert r.query is None
        assert r.prompt == "I couldn't work out what you want to know. Could you rephrase?"
        assert r.metadata is not None
        assert r.metadata.endpoint == "interact"

    def test_prompt_falls_back_to_error_message(self) -> None:
        data = _variant(INTERACT_ERROR)
        del data["error"]["suggestedChatResponse"]
        assert InteractResponse.from_dict(data).prompt == "Could not detect a query in the prompt"

    def test_raise_for_error_raises_natural_language_error(self) -> None:
        r = InteractResponse.from_dict(INTERACT_ERROR)
        with pytest.raises(NaturalLanguageError) as info:
            r.raise_for_error()
        exc = info.value
        assert exc.message == "Could not detect a query in the prompt"
        assert str(exc) == "Could not detect a query in the prompt"
        assert exc.code == 1001
        assert exc.suggested_chat_response == (
            "I couldn't work out what you want to know. Could you rephrase?"
        )
        assert exc.status_code == "400"
        assert exc.body is INTERACT_ERROR

    def test_raise_for_error_without_error_object(self) -> None:
        r = InteractResponse.from_dict({"responseType": "error"})
        assert r.is_error is True
        with pytest.raises(NaturalLanguageError, match="interact returned an error response"):
            r.raise_for_error()

    def test_raise_for_error_returns_self_on_question_and_result(self) -> None:
        q = InteractResponse.from_dict(INTERACT_QUESTION)
        assert q.raise_for_error() is q
        res = InteractResponse.from_dict(INTERACT_RESULT)
        assert res.raise_for_error() is res

    def test_response_type_inferred_error(self) -> None:
        r = InteractResponse.from_dict({"error": {"code": 1, "message": "m"}})
        assert r.response_type == "error"
        assert r.is_error is True

    def test_response_type_inferred_question(self) -> None:
        r = InteractResponse.from_dict({"questions": [QUESTION_FIRST_FORM]})
        assert r.response_type == "question"
        assert r.is_question is True

    def test_response_type_inferred_result(self) -> None:
        r = InteractResponse.from_dict({"results": []})
        assert r.response_type == "result"
        assert r.is_result is True
        r2 = InteractResponse.from_dict({"results": INTERACT_RESULT["results"]})
        assert r2.response_type == "result"

    def test_response_type_inferred_unknown(self) -> None:
        r = InteractResponse.from_dict({})
        assert r.response_type == "unknown"
        assert not r.is_question and not r.is_result and not r.is_error
        assert r.prompt is None
        assert r.raise_for_error() is r

    def test_inference_prefers_error_over_questions(self) -> None:
        r = InteractResponse.from_dict(
            {"error": {"message": "m"}, "questions": [QUESTION_FIRST_FORM]}
        )
        assert r.response_type == "error"

    def test_empty_response_type_string_triggers_inference(self) -> None:
        r = InteractResponse.from_dict({"responseType": "", "results": []})
        assert r.response_type == "result"

    def test_is_error_when_error_present_without_content(self) -> None:
        r = InteractResponse.from_dict({"responseType": "result", "error": {"message": "m"}})
        assert r.is_error is True

    def test_not_error_when_error_present_with_results(self) -> None:
        r = InteractResponse.from_dict(
            {
                "responseType": "result",
                "error": {"message": "warning"},
                "results": INTERACT_RESULT["results"],
            }
        )
        assert r.is_error is False
        assert r.raise_for_error() is r

    def test_empty_query_becomes_none(self) -> None:
        assert InteractResponse.from_dict({"responseType": "result", "query": {}}).query is None

    def test_unmatched_are_stringified(self) -> None:
        r = InteractResponse.from_dict({"responseType": "result", "facts": {"unmatched": [1, "a"]}})
        assert r.unmatched == ["1", "a"]

    def test_invalid_facts(self) -> None:
        invalid = [{"subject": "Ben", "relationship": "flies", "object": "Mars", "certainty": 100}]
        r = InteractResponse.from_dict({"responseType": "result", "facts": {"invalid": invalid}})
        assert r.invalid_facts == invalid


# --------------------------------------------------------------------------- ExplainResponse


class TestExplainResponse:
    def test_spec_example(self) -> None:
        r = ExplainResponse.from_dict(SPEC_EXPLAIN)
        assert r.explanation is not None
        assert r.explanation.startswith("Based on the information provided, Ben can claim")
        assert str(r) == r.explanation
        assert r.error is None
        assert r.metadata is not None
        assert r.metadata.endpoint == "explain"
        assert r.metadata.session_id == SPEC_NL_SID
        assert r.metadata.total_tokens == 900
        assert r.raw is SPEC_EXPLAIN

    def test_fixture(self) -> None:
        r = ExplainResponse.from_dict(EXPLAIN_OK)
        assert r.explanation == EXPLAIN_OK["explanation"]
        assert r.metadata is not None
        assert r.metadata.query_submitted_at == dt.datetime(
            2025, 7, 10, 11, 16, 12, 597111, tzinfo=UTC
        )
        assert (r.metadata.completion_tokens, r.metadata.prompt_tokens) == (174, 726)

    def test_error_body(self) -> None:
        body = {"error": {"code": 2001, "message": "Fact not found", "statusCode": "404"}}
        r = ExplainResponse.from_dict(body)
        assert r.explanation is None
        assert str(r) == ""
        assert r.error is not None
        assert r.error.code == 2001
        assert r.error.message == "Fact not found"
        assert r.error.status_code == "404"
        assert r.metadata is None

    def test_non_string_explanation_is_stringified(self) -> None:
        assert ExplainResponse.from_dict({"explanation": 42}).explanation == "42"

    def test_error_that_is_not_a_dict_is_ignored(self) -> None:
        r = ExplainResponse.from_dict({"explanation": "x", "error": "oops"})
        assert r.error is None
        assert r.explanation == "x"


# --------------------------------------------------------------------------- KnowledgeMapSource / CreateMapResult


class TestKnowledgeMapSource:
    def test_fixture_with_km_id_argument(self) -> None:
        src = KnowledgeMapSource.from_dict(MAP_SOURCE, km_id=KM_ID)
        assert src.km_id == KM_ID
        assert src.rblang == RBLANG
        assert src.rblang.startswith("<?xml")
        assert '<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">' in src.rblang
        assert src.concepts == MAP_SOURCE["concepts"]
        assert src.relationships == MAP_SOURCE["rels"]
        assert src.raw is MAP_SOURCE

    def test_body_km_id_wins_over_argument(self) -> None:
        src = KnowledgeMapSource.from_dict({**MAP_SOURCE, "kmID": "body-id"}, km_id="arg-id")
        assert src.km_id == "body-id"

    def test_km_id_preferred_over_id(self) -> None:
        src = KnowledgeMapSource.from_dict({"kmID": "guid", "id": 7, "rblang": "<x/>"})
        assert src.km_id == "guid"

    def test_id_used_when_km_id_missing(self) -> None:
        src = KnowledgeMapSource.from_dict({"id": 7, "rblang": "<x/>"})
        assert src.km_id == "7"

    def test_argument_used_when_body_has_no_id(self) -> None:
        assert KnowledgeMapSource.from_dict({"rblang": "<x/>"}, km_id="k").km_id == "k"

    def test_no_id_anywhere_is_empty_string(self) -> None:
        assert KnowledgeMapSource.from_dict({"rblang": "<x/>"}).km_id == ""

    def test_missing_rblang_and_lists(self) -> None:
        src = KnowledgeMapSource.from_dict({}, km_id="k")
        assert src.rblang == ""
        assert src.concepts == []
        assert src.relationships == []

    def test_relationships_key_variant(self) -> None:
        src = KnowledgeMapSource.from_dict({"relationships": [{"name": "r"}]}, km_id="k")
        assert src.relationships == [{"name": "r"}]


class TestCreateMapResult:
    def test_fixture_body(self) -> None:
        body = {"kmID": "99999999-8888-7777-6666-555555555555", "id": 4242, "name": "New map"}
        result = CreateMapResult.from_dict(body)
        assert result.km_id == "99999999-8888-7777-6666-555555555555"
        assert result.id == 4242
        assert result.name == "New map"
        assert result.raw is body

    def test_km_id_preferred_over_numeric_id(self) -> None:
        result = CreateMapResult.from_dict({"kmID": "guid", "id": 1})
        assert result.km_id == "guid"
        assert result.id == 1

    def test_numeric_id_used_when_km_id_missing(self) -> None:
        result = CreateMapResult.from_dict({"id": 1234})
        assert result.km_id == "1234"
        assert result.id == 1234

    @pytest.mark.parametrize("key", ["kmid", "kmId", "mapId", "knowledge_map_id"])
    def test_km_id_key_variants(self, key: str) -> None:
        assert CreateMapResult.from_dict({key: "guid"}).km_id == "guid"

    def test_name_argument_is_fallback_only(self) -> None:
        assert CreateMapResult.from_dict({"kmID": "g"}, name="from-arg").name == "from-arg"
        assert CreateMapResult.from_dict({"kmID": "g", "name": "body"}, name="arg").name == "body"
        assert CreateMapResult.from_dict({"kmID": "g"}).name is None

    @pytest.mark.parametrize(
        "body", [{}, {"name": "x"}, {"kmID": None}, {"kmID": None, "id": None}]
    )
    def test_missing_id_raises_value_error(self, body: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="knowledge map id"):
            CreateMapResult.from_dict(body, name="x")


# --------------------------------------------------------------------------- raw / to_dict / frozen


class TestRawAndImmutability:
    def test_to_dict_equals_raw_but_is_a_deep_copy(self) -> None:
        q = Question.from_dict(QUESTION_SECOND_OBJECT)
        d = q.to_dict()
        assert d == QUESTION_SECOND_OBJECT
        assert d is not q.raw
        assert d["concepts"] is not q.raw["concepts"]
        assert d["concepts"][0] is not q.raw["concepts"][0]

    def test_mutating_to_dict_does_not_touch_raw(self) -> None:
        response = ResultResponse.from_dict(copy.deepcopy(SPEC_RESULT_RESPONSE))
        d = response.to_dict()
        d["result"].clear()
        d["sid"] = "changed"
        assert response.raw == SPEC_RESULT_RESPONSE
        assert response.to_dict() == SPEC_RESULT_RESPONSE

    def test_to_dict_of_nested_evidence(self) -> None:
        ev = Evidence.from_dict(SPEC_EVIDENCE)
        d = ev.to_dict()
        assert d == SPEC_EVIDENCE
        d["rule"]["conditions"].pop()
        assert len(SPEC_EVIDENCE["rule"]["conditions"]) == 9
        assert ev.rule is not None and len(ev.rule.conditions) == 9

    def test_raw_is_the_original_object(self) -> None:
        assert Question.from_dict(QUESTION_FIRST_FORM).raw is QUESTION_FIRST_FORM
        assert Evidence.from_dict(EVIDENCE_ROOT).raw is EVIDENCE_ROOT

    def test_raw_is_excluded_from_repr(self) -> None:
        text = repr(Question.from_dict(QUESTION_FIRST_FORM))
        assert "raw=" not in text
        assert "prompt='Does Tom speak French?'" in text

    @pytest.mark.parametrize(
        ("model", "attr"),
        [
            (StartResponse.from_dict(SPEC_START), "session_id"),
            (InjectResponse.from_dict(SPEC_INJECT), "result"),
            (Question.from_dict(QUESTION_FIRST_FORM), "prompt"),
            (Concept.from_dict(QUESTION_SECOND_OBJECT["concepts"][0]), "name"),
            (QuestionResponse.from_dict(SPEC_QUESTION_RESPONSE), "sid"),
            (ResultItem.from_dict(SPEC_RESULT_RESPONSE["result"][0]), "certainty"),
            (ResultResponse.from_dict(SPEC_RESULT_RESPONSE), "results"),
            (FactValue.from_dict({"value": 1}), "value"),
            (SessionFact.from_dict(SESSION_FACTS["global"][0]), "relationship"),
            (SessionFacts.from_dict(SESSION_FACTS), "local"),
            (KnowledgeMapVersion.from_dict(SESSION_VERSION), "name"),
            (Condition.from_dict(EVIDENCE_ROOT["rule"]["conditions"][0]), "impact"),
            (ExpressionCondition.from_dict(EVIDENCE_ROOT["rule"]["conditions"][2]), "text"),
            (Rule.from_dict(EVIDENCE_ROOT["rule"]), "bindings"),
            (Evidence.from_dict(EVIDENCE_ROOT), "fact_id"),
            (InteractionEvent.from_dict(INTERACTIONS[0]), "event"),
            (NLError.from_dict(INTERACT_ERROR["error"]), "code"),
            (NLMetadata.from_dict(EXPLAIN_OK["metadata"]), "endpoint"),
            (InteractResponse.from_dict(INTERACT_RESULT), "response_type"),
            (ExplainResponse.from_dict(EXPLAIN_OK), "explanation"),
            (KnowledgeMapSource.from_dict(MAP_SOURCE, km_id=KM_ID), "km_id"),
            (CreateMapResult.from_dict({"kmID": "g"}), "km_id"),
            (
                Decision.from_response(
                    ResultResponse.from_dict(RESULT_TOM_SPEAKS_GERMAN),
                    session_id="s",
                    relationship="speaks",
                    subject="Tom",
                ),
                "decision",
            ),
        ],
        ids=lambda v: type(v).__name__ if not isinstance(v, str) else v,
    )
    def test_models_are_frozen(self, model: Any, attr: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(model, attr, "changed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.raw = {}
        with pytest.raises(dataclasses.FrozenInstanceError):
            del model.raw

    def test_frozen_flag_is_set_on_every_dataclass(self) -> None:
        for cls in (
            Fact,
            Answer,
            StartResponse,
            InjectResponse,
            Concept,
            Question,
            QuestionResponse,
            ResultItem,
            ResultResponse,
            FactValue,
            SessionFact,
            SessionFacts,
            KnowledgeMapVersion,
            Condition,
            ExpressionCondition,
            Rule,
            Evidence,
            InteractionEvent,
            NLError,
            NLMetadata,
            InteractResponse,
            ExplainResponse,
            KnowledgeMapSource,
            CreateMapResult,
            Decision,
        ):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen is True, cls  # type: ignore[union-attr]


# --------------------------------------------------------------------------- through the client


class TestModelsThroughClient:
    def test_start_response_is_kept_on_the_session(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        assert fake.last.method == "GET"
        assert fake.last.path == f"/start/{KM_ID}"
        assert fake.last.params == {}
        assert isinstance(session.start_response, StartResponse)
        assert session.start_response.session_id == session.id
        assert session.start_response.km_version_id == KM_VERSION_ID
        assert session.km_version_id == KM_VERSION_ID
        assert session.start_response.to_dict() == {
            "id": session.id,
            "kmVersion": {"id": KM_VERSION_ID},
        }

    def test_inject_fact_model_sends_exact_payload(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        fact = Fact("Ben", "holds", "Gold")
        response = session.inject(fact)
        assert fake.last.method == "POST"
        assert fake.last.path == f"/{session.id}/inject"
        assert fake.last.json == [fact.to_payload()]
        assert fake.last.json == [
            {"subject": "Ben", "relationship": "holds", "object": "Gold", "certainty": 100}
        ]
        assert isinstance(response, InjectResponse)
        assert response.ok is True
        assert response.result == "OK"

    def test_inject_many_facts_preserves_order_and_certainty(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        session.inject([Fact("Tom", "lives in", "Germany"), Fact("Tom", "age", 30, certainty=60)])
        assert fake.last.json == [
            {"subject": "Tom", "relationship": "lives in", "object": "Germany", "certainty": 100},
            {"subject": "Tom", "relationship": "age", "object": 30, "certainty": 60},
        ]

    def test_query_returns_result_response_with_sid(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        response = session.query("speaks", subject="Tom")
        assert fake.last.path == f"/{session.id}/query"
        assert fake.last.json == {"relationship": "speaks", "subject": "Tom"}
        assert isinstance(response, ResultResponse)
        assert response.sid == session.id
        assert response.question is None
        assert len(response) == 1
        assert response.best is not None
        assert response.best.fact_id == EVIDENCE_ROOT["factID"]
        assert str(response.best) == "Tom speaks German (75%)"
        assert response.to_dict() == {**RESULT_TOM_SPEAKS_GERMAN, "sid": session.id}

    def test_query_returns_question_response_with_extra_questions(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT, QUESTION_FIRST_FORM)
        session = client.start(KM_ID)
        response = session.query("speaks", subject="Tom")
        assert isinstance(response, QuestionResponse)
        assert response.sid == session.id
        assert response.question.asks_for_object is True
        assert [q.type for q in response.questions] == ["Second Form Object", "First Form"]
        assert response.extra_questions[0].is_first_form is True
        assert session.pending_question is response.question

    def test_respond_with_answer_models_sends_exact_payload(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_PLURAL_LOSS)
        session = client.start(KM_ID)
        session.query("can claim", subject="Ben")
        answers = [
            Answer("Ben", "suffered", "water damage"),
            Answer("Ben", "suffered", "theft", certainty=50),
        ]
        response = session.respond(answers)
        assert fake.last.path == f"/{session.id}/response"
        assert fake.last.json == {"answers": [a.to_payload() for a in answers]}
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Ben",
                    "relationship": "suffered",
                    "object": "water damage",
                    "certainty": 100,
                },
                {"subject": "Ben", "relationship": "suffered", "object": "theft", "certainty": 50},
            ]
        }
        assert isinstance(response, ResultResponse)

    def test_first_form_answer_payload_carries_yes(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_FIRST_FORM)
        session = client.start(KM_ID)
        question = session.query("speaks", subject="Tom").question
        assert question is not None and question.is_first_form
        session.answer(True)
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Tom",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 100,
                    "answer": "yes",
                }
            ]
        }

    def test_second_form_object_answer_goes_in_object(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        session = client.start(KM_ID)
        session.query("national language", subject="Germany")
        session.answer("English")
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "English",
                    "certainty": 100,
                }
            ]
        }

    def test_second_form_subject_answer_goes_in_subject(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        session = client.start(KM_ID)
        session.query("speaks", object="French")
        session.answer(["Tom", "Ana"])
        assert fake.last.json == {
            "answers": [
                {"subject": "Tom", "relationship": "speaks", "object": "French", "certainty": 100},
                {"subject": "Ana", "relationship": "speaks", "object": "French", "certainty": 100},
            ]
        }

    def test_skip_subject_question_sends_unanswered_with_empty_subject(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_SUBJECT)
        session = client.start(KM_ID)
        session.query("speaks", object="French")
        session.skip()
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "",
                    "relationship": "speaks",
                    "object": "French",
                    "certainty": 100,
                    "unanswered": True,
                }
            ]
        }

    def test_undo_sends_empty_object_and_parses_question(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        fake.queue(RESULT_TOM_SPEAKS_GERMAN, {"question": QUESTION_FIRST_FORM})
        session = client.start(KM_ID)
        session.query("speaks", subject="Tom")
        response = session.undo()
        assert fake.last.path == f"/{session.id}/undo"
        assert fake.last.json == {}
        assert isinstance(response, QuestionResponse)
        assert response.question.prompt == "Does Tom speak French?"

    def test_evidence_and_children(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        root = session.evidence(EVIDENCE_ROOT["factID"])
        assert fake.last.path == f"/analysis/evidence/{EVIDENCE_ROOT['factID']}/{session.id}"
        assert isinstance(root, Evidence)
        assert root.is_inferred
        assert root.child_fact_ids == list(EVIDENCE_LEAVES)
        leaves = [session.evidence(fid) for fid in root.child_fact_ids]
        assert [leaf.source for leaf in leaves] == [FactSource.INJECTION, FactSource.ANSWER]
        assert all(leaf.child_fact_ids == [] for leaf in leaves)
        assert len(fake.calls("/analysis/evidence/")) == 3

    def test_knowledge_map_version(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        km = session.knowledge_map()
        assert fake.last.path == f"/analysis/session/{session.id}"
        assert fake.last.params == {"filter": "version"}
        assert isinstance(km, KnowledgeMapVersion)
        assert km.version_number == 3
        assert km.is_draft is False
        assert km.version_created == dt.datetime(2025, 7, 9, 14, 27, 22, tzinfo=UTC)

    def test_session_facts_and_relationship_filter(
        self, client: Rainbird, fake: FakeRainbird
    ) -> None:
        session = client.start(KM_ID)
        facts = session.facts()
        assert fake.last.params == {"filter": "facts"}
        assert isinstance(facts, SessionFacts)
        assert len(facts) == 2
        filtered = session.facts(relationships=["national language"])
        assert fake.last.params == {"filter": "facts", "relationships": "national language"}
        assert [f.relationship for f in filtered] == ["national language"]
        assert filtered.local == []

    def test_interactions(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        events = session.interactions()
        assert fake.last.params == {"format": "json"}
        assert isinstance(events, list)
        assert all(isinstance(e, InteractionEvent) for e in events)
        assert [e.event for e in events] == ["start", "query", "result"]
        assert events[0].created == dt.datetime(2025, 7, 10, 10, 6, 27, 780000, tzinfo=UTC)

    def test_interact_result(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        response = session.interact("Ben holds a Gold policy and he likes tea")
        assert fake.last.path == "/nl/interact"
        assert fake.last.headers["version"] == "v1"
        assert fake.last.json == {
            "sessionID": session.id,
            "userPrompt": "Ben holds a Gold policy and he likes tea",
        }
        assert isinstance(response, InteractResponse)
        assert response.is_result
        assert response.results[0].object == "repair"
        assert response.unmatched == ["and he likes tea"]

    def test_interact_question_and_error(self, client: Rainbird, fake: FakeRainbird) -> None:
        fake.interact_bodies.extend([INTERACT_QUESTION, INTERACT_ERROR])
        session = client.start(KM_ID)
        question = session.interact("Can Ben claim?")
        assert question.is_question
        assert question.prompt == "Which policy does Ben hold?"
        error = session.interact("hello")
        assert error.is_error
        with pytest.raises(NaturalLanguageError):
            error.raise_for_error()

    def test_explain(self, client: Rainbird, fake: FakeRainbird) -> None:
        session = client.start(KM_ID)
        explanation = session.explain(EVIDENCE_ROOT["factID"], instructions="Be brief.")
        assert fake.last.path == "/nl/explain"
        assert fake.last.headers["version"] == "v1"
        assert fake.last.json == {
            "language": "en",
            "sessionID": session.id,
            "factID": EVIDENCE_ROOT["factID"],
            "instructions": "Be brief.",
        }
        assert isinstance(explanation, ExplainResponse)
        assert str(explanation) == EXPLAIN_OK["explanation"]
        assert explanation.metadata is not None
        assert explanation.metadata.total_tokens == 900

    def test_get_map_source(self, client: Rainbird, fake: FakeRainbird) -> None:
        source = client.get_map_source(KM_ID)
        assert fake.last.path == f"/analysis/file/{KM_ID}"
        assert isinstance(source, KnowledgeMapSource)
        assert source.km_id == KM_ID  # body has no id; falls back to the requested one
        assert source.rblang == RBLANG
        assert [c["name"] for c in source.concepts] == ["Person", "Language"]
        assert [r["name"] for r in source.relationships] == ["speaks"]

    def test_create_map(self, client: Rainbird, fake: FakeRainbird) -> None:
        result = client.create_map("New map", RBLANG)
        assert fake.last.path == "/maps"
        assert fake.last.headers["version"] == "v1"
        assert fake.last.json == {"rblang": RBLANG, "name": "New map"}
        assert isinstance(result, CreateMapResult)
        assert result.km_id == "99999999-8888-7777-6666-555555555555"
        assert result.id == 4242
        assert result.name == "New map"

    @pytest.mark.anyio
    async def test_async_client_parses_the_same_models(
        self, aclient: AsyncRainbird, fake: FakeRainbird
    ) -> None:
        fake.queue_question(QUESTION_SECOND_OBJECT)
        session = await aclient.start(KM_ID)
        assert isinstance(session.start_response, StartResponse)
        question = await session.query("national language", subject="Germany")
        assert isinstance(question, QuestionResponse)
        assert question.question.options == ["English", "French"]
        result = await session.answer("English")
        assert fake.last.json == {
            "answers": [
                {
                    "subject": "Germany",
                    "relationship": "national language",
                    "object": "English",
                    "certainty": 100,
                }
            ]
        }
        assert isinstance(result, ResultResponse)
        assert result.best is not None
        assert result.best.object == "German"
        await aclient.aclose()


# ------------------------------------------------------------ additions from the SDK parity round


class TestStringPlaceholders:
    """A missing subject or object renders as ``?`` (as the Go SDK does), never ``None``."""

    def test_question_fallback_marks_unknown_sides(self) -> None:
        assert str(Question.from_dict({"subject": "John", "relationship": "speaks"})) == (
            "John speaks ?"
        )
        assert str(Question.from_dict({"relationship": "lives in", "object": "England"})) == (
            "? lives in England"
        )
        assert str(Question.from_dict({"relationship": "lives in", "object": ""})) == (
            "? lives in ?"
        )
        assert str(Question.from_dict({})) == "? ?"

    def test_question_prompt_still_wins(self) -> None:
        q = Question.from_dict({"prompt": "Who lives in England?", "relationship": "lives in"})
        assert str(q) == "Who lives in England?"

    def test_result_item_marks_unknown_sides(self) -> None:
        assert str(ResultItem.from_dict({"subject": "John", "relationship": "speaks"})) == (
            "John speaks ?"
        )
        item = ResultItem.from_dict(
            {"subject": "", "relationship": "", "object": None, "certainty": 0}
        )
        assert str(item) == "? ? (0%)"
        assert "None" not in str(ResultItem.from_dict({}))

    def test_result_item_keeps_real_values_and_numbers(self) -> None:
        item = ResultItem.from_dict(
            {"subject": "John", "relationship": "favourite number", "object": 0, "certainty": 35}
        )
        assert str(item) == "John favourite number 0 (35%)"  # 0 is a value, not "unknown"

    def test_session_fact_marks_unknown_sides(self) -> None:
        fact = SessionFact.from_dict({"subject": {"value": "Tom"}, "relationship": "speaks"})
        assert str(fact) == "Tom speaks ?"
        assert "None" not in str(SessionFact.from_dict({}))


class TestFactMetadataPayload:
    def test_metadata_is_omitted_when_unset(self) -> None:
        assert "metadata" not in Fact("Tom", "lives in", "Germany").to_payload()

    def test_metadata_is_sent_and_copied(self) -> None:
        meta = {"source": "user", "extra": {"k": 1}}
        payload = Fact("Tom", "lives in", "Germany", metadata=meta).to_payload()
        assert payload["metadata"] == meta
        assert payload["metadata"] is not meta
        assert payload["metadata"]["extra"] is not meta["extra"]


class TestEvidenceFactSource:
    BODY = {
        "factID": "WA:RF:1",
        "source": "rule",
        "fact": {
            "subject": {"type": "Person", "value": "Dan", "dataType": "string"},
            "relationship": {"type": "speaks"},
            "object": {"type": "Language", "value": "German", "dataType": "string"},
            "certainty": 100,
        },
        "time": 1,
    }

    def test_envelope_source_is_copied_onto_the_fact(self) -> None:
        evidence = Evidence.from_dict(self.BODY)
        assert evidence.source is FactSource.RULE
        assert evidence.fact.id == "WA:RF:1"
        assert evidence.fact.source is FactSource.RULE

    def test_facts_own_source_wins(self) -> None:
        body = {**self.BODY, "fact": {**self.BODY["fact"], "source": "injection"}}
        evidence = Evidence.from_dict(body)
        assert evidence.source is FactSource.RULE
        assert evidence.fact.source is FactSource.INJECTION

    def test_no_source_anywhere_stays_none(self) -> None:
        body = {k: v for k, v in self.BODY.items() if k != "source"}
        evidence = Evidence.from_dict(body)
        assert evidence.source is None
        assert evidence.fact.source is None
        assert evidence.raw is body  # the envelope itself is untouched


class TestKnowledgeMapVersionExplicitNumber:
    @pytest.mark.parametrize(
        ("value", "status", "expected"),
        [
            (3, "Draft", 3),  # explicit number wins over the status
            (5, "Version3", 5),
            (7.0, "Draft", 7),
            ("12", "Draft", 12),
            (None, "Version3", 3),  # falls back to the status
            (None, "Draft", None),
            (True, "Version2", 2),  # bools are not version numbers
            ("three", "Version2", 2),
            (2.5, "Draft", None),
        ],
    )
    def test_explicit_version_number(self, value: Any, status: str, expected: int | None) -> None:
        km = {"id": "k", "versionStatus": status}
        if value is not None:
            km["versionNumber"] = value
        assert KnowledgeMapVersion.from_dict({"km": km}).version_number == expected
        assert KnowledgeMapVersion.from_dict(km).version_number == expected  # unwrapped body

    def test_is_draft_is_independent_of_the_number(self) -> None:
        km = KnowledgeMapVersion.from_dict({"km": {"versionStatus": "Draft", "versionNumber": 3}})
        assert km.is_draft is True
        assert km.version_number == 3


class TestCreateMapResultDescription:
    def test_description_from_response_or_argument(self) -> None:
        body = {"kmID": "k", "id": 1}
        assert CreateMapResult.from_dict(body).description is None
        assert CreateMapResult.from_dict(body, description="mine").description == "mine"
        echoed = {**body, "description": "theirs"}
        assert CreateMapResult.from_dict(echoed, description="mine").description == "theirs"
        assert CreateMapResult.from_dict(echoed).raw is echoed


# --------------------------------------------------------------------------- Decision


class TestDecision:
    RESULT = ResultResponse.from_dict(RESULT_TOM_SPEAKS_GERMAN)
    QUESTION = QuestionResponse.from_dict({"question": QUESTION_SECOND_OBJECT, "sid": "s"})
    EMPTY = ResultResponse.from_dict({"result": []})

    def build(self, response: Any, **kwargs: Any) -> Decision:
        kwargs.setdefault("session_id", "sid-1")
        kwargs.setdefault("relationship", "speaks")
        return Decision.from_response(response, **kwargs)

    def test_result_object_side(self) -> None:
        d = self.build(self.RESULT, subject="Tom")
        assert d.status is DecisionStatus.RESULT
        assert d.decision == "German"
        assert d.certainty == 75
        assert d.fact_id == RESULT_TOM_SPEAKS_GERMAN["result"][0]["factID"]
        assert d.question is None
        assert len(d.results) == 1 and isinstance(d.results[0], ResultItem)
        assert d.best is not None and d.best.object == "German"
        assert d.response is self.RESULT
        assert d.raw is RESULT_TOM_SPEAKS_GERMAN
        assert d.is_final and not d.is_question
        assert str(d) == "German (75%)"

    def test_result_subject_side(self) -> None:
        d = self.build(self.RESULT, object="German")
        assert d.decision == "Tom"

    def test_result_relationship_only_uses_object(self) -> None:
        d = self.build(self.RESULT)
        assert d.decision == "German"

    @pytest.mark.parametrize("blank", [None, ""])
    def test_blank_side_counts_as_not_given(self, blank: Any) -> None:
        assert self.build(self.RESULT, subject="Tom", object=blank).decision == "German"
        assert self.build(self.RESULT, subject=blank, object="German").decision == "Tom"

    def test_both_sides_given_is_boolean(self) -> None:
        assert self.build(self.RESULT, subject="Tom", object="German").decision is True
        d = self.build(self.EMPTY, subject="Tom", object="German")
        assert d.status is DecisionStatus.NO_RESULT
        assert d.decision is False

    def test_no_result(self) -> None:
        d = self.build(self.EMPTY, subject="Tom")
        assert d.status is DecisionStatus.NO_RESULT
        assert d.decision is None and d.certainty is None and d.fact_id is None
        assert d.results == [] and d.best is None
        assert d.is_final
        assert str(d) == "no result"

    def test_question(self) -> None:
        d = self.build(self.QUESTION, subject="Tom")
        assert d.status is DecisionStatus.QUESTION
        assert d.is_question and not d.is_final
        assert isinstance(d.question, Question)
        assert d.question.prompt == QUESTION_SECOND_OBJECT["prompt"]
        assert d.decision is None and d.certainty is None and d.fact_id is None
        assert d.results == [] and d.best is None
        assert d.raw is self.QUESTION.raw
        assert str(d) == f"question: {QUESTION_SECOND_OBJECT['prompt']}"

    def test_best_prefers_highest_certainty(self) -> None:
        low = dict(RESULT_TOM_SPEAKS_GERMAN["result"][0], object="English", certainty=10)
        response = ResultResponse.from_dict(
            {"result": [low, RESULT_TOM_SPEAKS_GERMAN["result"][0]]}
        )
        d = self.build(response, subject="Tom")
        assert d.decision == "German" and d.certainty == 75
        assert [r.object for r in d.results] == ["English", "German"]

    def test_evidence_and_metadata_are_kept(self) -> None:
        node = Evidence.from_dict(EVIDENCE_ROOT)
        d = self.build(
            self.RESULT,
            subject="Tom",
            evidence=[node],
            evidence_url="https://app.rainbird.ai/evidence?id=x",
            km_id=KM_ID,
            km_version_id="v1",
            session="session-object",
        )
        assert d.evidence == [node]
        assert d.evidence_url == "https://app.rainbird.ai/evidence?id=x"
        assert d.km_id == KM_ID and d.km_version_id == "v1"
        assert d.session == "session-object"

    def test_to_dict_flat_shape(self) -> None:
        node = Evidence.from_dict(EVIDENCE_ROOT)
        d = self.build(self.RESULT, subject="Tom", evidence=[node], km_id=KM_ID)
        data = d.to_dict()
        assert data == {
            "status": "result",
            "decision": "German",
            "certainty": 75,
            "session_id": "sid-1",
            "fact_id": RESULT_TOM_SPEAKS_GERMAN["result"][0]["factID"],
            "km_id": KM_ID,
            "km_version_id": None,
            "relationship": "speaks",
            "subject": "Tom",
            "object": None,
            "question": None,
            "results": [RESULT_TOM_SPEAKS_GERMAN["result"][0]],
            "evidence": [EVIDENCE_ROOT],
            "evidence_url": None,
        }
        data["results"][0]["object"] = "changed"
        assert d.results[0].object == "German"  # deep copies

    def test_to_dict_of_question(self) -> None:
        data = self.build(self.QUESTION, subject="Tom").to_dict()
        assert data["status"] == "question"
        assert data["question"] == QUESTION_SECOND_OBJECT
        assert data["results"] == [] and data["decision"] is None

    def test_session_and_response_are_not_in_repr(self) -> None:
        text = repr(self.build(self.RESULT, subject="Tom", session=object()))
        assert "session=" not in text and "response=" not in text and "raw=" not in text
        assert "decision='German'" in text

    def test_status_enum_is_a_str(self) -> None:
        assert DecisionStatus.RESULT == "result"
        assert str(DecisionStatus.NO_RESULT) == "no_result"
        assert DecisionStatus("question") is DecisionStatus.QUESTION
