"""Interactive query: print each question with its options and answer from the keyboard.

The core loop of the SDK: ``Rainbird.start`` -> ``Session.query`` -> ``Session.answer``
until the engine returns a ``ResultResponse``. ``session.answer`` builds the right payload
for the pending question (First Form yes/no, Second Form subject/object, plural, skip).
Before that, ``engine_version()`` shows the one call that needs no API key (``GET /version``).

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id to query
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/basic_query.py [RELATIONSHIP] [SUBJECT]
"""

from __future__ import annotations

import sys
from typing import Any

from rainbird_ai import (
    SKIP,
    InvalidAnswerError,
    Question,
    QuestionResponse,
    Rainbird,
    engine_version,
)


def ask(question: Question) -> Any:
    """Show the question and turn the typed text into a value ``session.answer`` accepts."""
    print(f"\n{question.prompt}")
    if question.options:  # candidate values from question.concepts (empty for free text)
        print("  options:", ", ".join(str(option) for option in question.options))
    if question.is_first_form:
        print("  answer yes or no")
    if question.plural:
        print("  several values allowed, separated by commas")
    if question.allow_unknown:
        print("  type ? to skip")
    text = input("> ").strip()
    if text == "?":
        return SKIP  # sends {"unanswered": true}; only valid when allow_unknown is set
    if question.plural and "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subject = sys.argv[2] if len(sys.argv) > 2 else "Tom"

    # GET /version is public: engine_version() needs no API key (host from RAINBIRD_API_URL).
    print(f"engine {engine_version()}")

    # Rainbird() reads RAINBIRD_API_KEY and RAINBIRD_API_URL; start() reads RAINBIRD_KM_ID.
    with Rainbird() as rb:
        session = rb.start()
        print(f"session {session.id}")
        response = session.query(relationship, subject=subject)
        while isinstance(response, QuestionResponse):  # or: while response.is_question
            try:
                response = session.answer(ask(response.question))
            except InvalidAnswerError as exc:
                print(f"  ! {exc}")  # rejected before sending; the question is still pending
        if not response.results:
            print("No result: the engine could not infer anything.")
        for item in response.results:
            print(f"{item}  factID={item.fact_id}")


if __name__ == "__main__":
    main()
