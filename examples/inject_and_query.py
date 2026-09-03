"""Inject known facts into a session, check they are visible, then query.

``Session.inject`` accepts ``Fact`` objects, dicts, ``(subject, relationship, object[,
certainty])`` tuples, or a list of any of those, and posts them to ``/{sid}/inject``.
Facts already in the session are used by the engine instead of asking a question.

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id (the facts below assume a "languages" style map)
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/inject_and_query.py [RELATIONSHIP] [SUBJECT]
"""

from __future__ import annotations

import sys
from typing import Any

from rainbird_ai import SKIP, Fact, Question, Rainbird

# Adapt subjects, relationships and objects to your knowledge map.
FACTS = [
    Fact("Tom", "lives in", "Germany"),
    Fact("Germany", "national language", "German"),
    Fact("Tom", "speaks", "English", certainty=80),  # certainty 1-100, default 100
]
# Objects may be str, int, float or bool, a date/datetime (sent as "YYYY-MM-DD", time of day
# dropped), a Decimal (int when integral) or an Enum member (its .value). None, dicts and
# lists are rejected before anything is sent.


def keyboard(question: Question) -> Any:
    """Answer anything the injected facts do not cover from the keyboard ("?" skips)."""
    print(f"\n{question.prompt}")
    if question.options:
        print("  options:", ", ".join(str(option) for option in question.options))
    text = input("> ").strip()
    return SKIP if text == "?" else text


def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subject = sys.argv[2] if len(sys.argv) > 2 else "Tom"

    with Rainbird() as rb:
        session = rb.start()

        # Batching: Rainbird recommends at most 250 facts per inject request. inject()
        # splits longer lists into batches of batch_size (default 250) and returns the
        # response of the last batch, so a list of 10,000 facts is still one call here.
        response = session.inject(FACTS)
        print(f"injected {len(FACTS)} facts: {response.result}")  # API replies {"result": "OK"}

        # Optional check: facts the session can see, grouped by scope. Injected facts are
        # "local" with source "injection"; the map's own facts are "global". relationships=
        # narrows the list to one name ("speaks") or several.
        visible = session.facts(relationships=[fact.relationship for fact in FACTS])
        for fact in visible.local:
            print(f"  local: {fact}")

        # Anything the facts do not cover is still asked; answer it from the keyboard.
        result = session.run(relationship, subject=subject, answerer=keyboard)
        if not result.results:
            print("No result: the engine could not infer anything.")
        for item in result.results:
            print(f"{item}  factID={item.fact_id}")


if __name__ == "__main__":
    main()
