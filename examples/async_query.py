"""Query with ``AsyncRainbird`` under ``asyncio``, including several sessions in parallel.

Every network call is awaited. ``AsyncSession.run`` accepts a plain or ``async def``
answerer, so awaiting an external service (HTTP, database, LLM) inside it is natural.
Sessions are independent, so one session per subject can run concurrently with
``asyncio.gather``.

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id to query
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/async_query.py [RELATIONSHIP] [SUBJECT ...]
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rainbird_ai import SKIP, Question, QuestionResponse
from rainbird_ai.aio import AsyncRainbird


async def answer_for(question: Question) -> Any:
    """Async answerer: ``await`` a lookup here; this demo just picks something plausible."""
    await asyncio.sleep(0)  # stand-in for an awaited HTTP/DB/LLM call
    if question.is_first_form:
        return True  # yes/no confirmation
    if question.options:
        return question.options[0]
    if question.allow_unknown:
        return SKIP
    raise ValueError(f"no answer for {question.prompt!r}")


async def decide(rb: AsyncRainbird, relationship: str, subject: str) -> str:
    session = await rb.start()  # one session per subject
    result = await session.run(relationship, subject=subject, answerer=answer_for)
    best = result.best  # highest-certainty ResultItem, or None when nothing was inferred
    if best is None:
        return f"{subject}: no result (session {session.id})"
    return f"{subject}: {best}  factID={best.fact_id}"


async def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subjects = sys.argv[2:] or ["Tom", "Ana"]

    async with AsyncRainbird() as rb:  # reads RAINBIRD_API_KEY / RAINBIRD_API_URL
        # Without a key: await rainbird_ai.async_engine_version() hits the same GET /version.
        print("engine version", await rb.version())

        # 1. The manual loop, awaiting each step.
        session = await rb.start()  # reads RAINBIRD_KM_ID
        response = await session.query(relationship, subject=subjects[0])
        while isinstance(response, QuestionResponse):  # or: while response.is_question
            response = await session.answer(await answer_for(response.question))
        print("manual loop:", [str(item) for item in response.results])

        # 2. Many subjects at once, letting the SDK drive each question loop.
        lines = await asyncio.gather(*(decide(rb, relationship, s) for s in subjects))
        print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
