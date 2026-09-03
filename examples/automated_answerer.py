"""Answer questions automatically with ``Session.run`` and an answerer callback.

``session.run(relationship, subject=..., answerer=fn)`` sends the query and calls
``fn(question)`` for every question until a ``ResultResponse`` arrives. The callback may
return a plain value, a list of values (plural questions), ``Answer`` objects, or ``SKIP``.

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id to query
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/automated_answerer.py [RELATIONSHIP] [SUBJECT]
"""

from __future__ import annotations

import sys
from typing import Any

from rainbird_ai import SKIP, Question, Rainbird


def choose_answer(question: Question) -> Any:
    """Decide an answer without a human: the place to plug in a database lookup or an LLM."""
    print(f"Q: {question.prompt}  [{question.type}]")

    if question.is_first_form:
        # "Does Tom speak French?": True/False (or "yes"/"no") is sent as answer: "yes"|"no".
        return True

    if question.options:
        # Closed list of candidate values (question.concepts). Pick one of them; for a
        # plural question you may return a list and each value is sent as one answer.
        #
        # LLM hook: send question.prompt and question.options to your model and map its
        # reply back onto one option. Values outside the list are rejected client-side
        # (InvalidAnswerError) when question.can_add is False.
        return question.options[0]

    if question.allow_unknown:
        # Free text/number/date we cannot answer: SKIP sends {"unanswered": true}.
        # When allow_unknown is False the SDK rejects SKIP before sending (InvalidAnswerError).
        return SKIP

    # LLM hook: generate a value of type question.data_type here. Strings, numbers and bools
    # are sent as they are; a date/datetime is sent as "YYYY-MM-DD" (time of day dropped).
    raise ValueError(f"no automatic answer for {question.prompt!r} ({question.data_type})")


def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subject = sys.argv[2] if len(sys.argv) > 2 else "Tom"

    with Rainbird() as rb:  # reads RAINBIRD_API_KEY / RAINBIRD_API_URL
        session = rb.start()  # reads RAINBIRD_KM_ID
        # max_questions stops a map that keeps asking (RainbirdError is raised then).
        result = session.run(
            relationship, subject=subject, answerer=choose_answer, max_questions=25
        )
        if not result.results:
            print("No result: the engine could not infer anything.")
        for item in result.results:
            print(f"A: {item}  factID={item.fact_id}")


if __name__ == "__main__":
    main()
