"""Get a decision in one call with ``Rainbird.decide``.

``rb.decide(relationship, km_id=..., subject=..., facts=[...])`` starts a session, injects the
facts, sends the query and returns a flat ``Decision``: ``status`` (result / no_result /
question), ``decision``, ``certainty``, ``session_id``, ``fact_id``, ``results``, optional
``evidence`` and the Studio ``evidence_url``. When the engine needs more information the
returned ``decision.session`` already has the question pending, so you can keep answering.

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id to query
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/decide.py [RELATIONSHIP] [SUBJECT]
"""

from __future__ import annotations

import sys

from rainbird_ai import DecisionStatus, Rainbird


def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subject = sys.argv[2] if len(sys.argv) > 2 else "Tom"

    with Rainbird() as rb:  # RAINBIRD_API_KEY from the environment
        decision = rb.decide(
            relationship,
            subject=subject,  # km_id comes from RAINBIRD_KM_ID
            facts=[(subject, "lives in", "Germany", 90)],
            evidence="none",  # "root" or "tree" also fetches evidence (needs the evidence key)
        )
        print(f"status     {decision.status}")

        if decision.status is DecisionStatus.QUESTION:
            # The engine needs more information. ``decision.session`` is the Session used by
            # decide(), with the question already pending, so the normal loop continues it.
            response = decision.response
            while response is not None and response.is_question:
                print(response.question.prompt)
                if response.question.options:
                    print("options:", ", ".join(map(str, response.question.options)))
                response = decision.session.answer(input("> "))
            best = response.best if response is not None else None
            print(f"decision   {best.object if best else None}")
            print(f"certainty  {best.certainty if best else None}")
            print(f"session    {decision.session_id}")
            print(f"fact_id    {best.fact_id if best else None}")
            return

        # RESULT or NO_RESULT: the flat fields are filled in.
        print(f"decision   {decision.decision}")
        print(f"certainty  {decision.certainty}")
        print(f"session    {decision.session_id}")
        print(f"fact_id    {decision.fact_id}")
        if decision.evidence_url:
            print(f"evidence   {decision.evidence_url}")

        # For a fully automatic decision pass answerer= (see automated_answerer.py):
        # rb.decide(relationship, subject=subject, answerer=lambda q: q.options[0])


if __name__ == "__main__":
    main()
