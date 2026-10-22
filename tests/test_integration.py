"""Opt-in integration tests against the real Rainbird API.

Skipped unless ``RAINBIRD_API_KEY`` and ``RAINBIRD_KM_ID`` are set. Optional:
``RAINBIRD_API_URL`` (defaults to the Community host),
``RAINBIRD_TEST_RELATIONSHIP`` and ``RAINBIRD_TEST_SUBJECT`` to run a real query.

Run with::

    RAINBIRD_API_KEY=... RAINBIRD_KM_ID=... .venv/bin/python -m pytest -m integration -q
"""

from __future__ import annotations

import os

import pytest

from rainbird_ai import SKIP, Question, Rainbird, ResultResponse
from rainbird_ai.aio import AsyncRainbird

pytestmark = pytest.mark.integration

_API_KEY = os.environ.get("RAINBIRD_API_KEY")
_KM_ID = os.environ.get("RAINBIRD_KM_ID")
_needs_creds = pytest.mark.skipif(
    not (_API_KEY and _KM_ID), reason="set RAINBIRD_API_KEY and RAINBIRD_KM_ID to run"
)


@pytest.fixture(autouse=True)
def _keep_real_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest clears RAINBIRD_* for unit tests; restore the real values here.
    for name, value in (("RAINBIRD_API_KEY", _API_KEY), ("RAINBIRD_KM_ID", _KM_ID)):
        if value:
            monkeypatch.setenv(name, value)
    url = os.environ.get("RAINBIRD_API_URL")
    if url:
        monkeypatch.setenv("RAINBIRD_API_URL", url)


def _pick(question: Question) -> object:
    """Deterministic answerer: first option, ``yes`` for confirmations, skip if allowed."""
    if question.is_first_form:
        return True
    if question.options:
        return question.options[0] if not question.plural else question.options[:1]
    if question.allow_unknown:
        return SKIP
    if question.is_truth:
        return True
    if (question.data_type or "").lower() == "number":
        return 1
    if (question.data_type or "").lower() == "date":
        return "2024-01-01"
    return "unknown"


@_needs_creds
def test_version_and_session_lifecycle() -> None:
    with Rainbird() as rb:
        version = rb.version()
        assert version and version[0].isdigit()
        session = rb.start()
        assert session.id
        info = session.knowledge_map()
        assert info.id
        facts = session.facts()
        assert facts.all is not None


@_needs_creds
@pytest.mark.skipif(
    not os.environ.get("RAINBIRD_TEST_RELATIONSHIP"), reason="set RAINBIRD_TEST_RELATIONSHIP"
)
def test_query_to_result_and_evidence() -> None:
    relationship = os.environ["RAINBIRD_TEST_RELATIONSHIP"]
    subject = os.environ.get("RAINBIRD_TEST_SUBJECT")
    with Rainbird() as rb:
        session = rb.start()
        result = session.run(relationship, subject=subject, answerer=_pick, max_questions=25)
        assert isinstance(result, ResultResponse)
        if result.best and result.best.fact_id:
            tree = session.evidence_tree(result.best.fact_id, max_depth=2)
            assert tree and tree[0].fact_id == result.best.fact_id


@_needs_creds
@pytest.mark.anyio
async def test_async_start() -> None:
    async with AsyncRainbird() as rb:
        session = await rb.start()
        assert session.id
        info = await session.knowledge_map()
        assert info.id
