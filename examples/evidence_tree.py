"""Query, then show why the best result holds using its evidence tree.

Every ``ResultItem`` carries a ``fact_id``. ``session.evidence_tree(fact_id)`` fetches its
evidence node and, breadth-first, every node its conditions (and expression functions) refer
to, each once and root first. This script re-nests them like ``rainbird evidence --tree``
(fetched conditions become child nodes, the rest ``- ...`` lines) and prints the Studio link.

Environment variables:
    RAINBIRD_API_KEY       API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID         knowledge map id to query
    RAINBIRD_API_URL       optional API host, default https://api.rainbird.ai
    RAINBIRD_EVIDENCE_KEY  optional x-evidence-key when evidence is secured for the map

Usage: python examples/evidence_tree.py [RELATIONSHIP] [SUBJECT]
"""

from __future__ import annotations

import sys
from typing import Any

from rainbird_ai import SKIP, Condition, Evidence, Question, Rainbird


def first_option(question: Question) -> Any:
    """Minimal answerer (see automated_answerer.py): yes, else first option, else skip/raise."""
    if question.is_first_form:
        return True
    if question.options:
        return question.options[0]
    if question.allow_unknown:  # SKIP raises InvalidAnswerError when allow_unknown is False
        return SKIP
    raise ValueError(f"no automatic answer for {question.prompt!r} ({question.data_type})")


def print_tree(nodes: dict[str, Evidence], fact_id: str, depth: int, seen: set[str]) -> None:
    """Print one node and, for an inferred fact, its conditions indented beneath it."""
    node = nodes.get(fact_id)
    if node is None or fact_id in seen:
        return
    seen.add(fact_id)
    pad = "  " * depth
    # node.fact prints as "Tom speaks German (75%)"; node.source as "rule", "answer", ...
    print(f"{pad}{node.fact}  source={node.source}  {node.fact_id}")
    if node.rule is None:  # leaves: answers, injections and map facts have no rule
        return
    for cond in node.rule.conditions:
        if isinstance(cond, Condition):
            if cond.fact_id and cond.fact_id in nodes and cond.fact_id not in seen:
                # The condition's own evidence was fetched: show it as a nested node.
                print_tree(nodes, cond.fact_id, depth + 1, seen)
            else:  # not fetched (max_depth) or already shown: one line under this node
                print(
                    f"{pad}  - {cond.subject} {cond.relationship} {cond.object} "
                    f"(certainty {cond.certainty}, impact {cond.impact})"
                )
        else:  # ExpressionCondition
            print(f"{pad}  - expr {cond.text!r} -> {cond.value!r} met={cond.was_met}")
            for fid in cond.fact_ids:  # facts the expression's functions used
                print_tree(nodes, fid, depth + 2, seen)


def main() -> None:
    relationship = sys.argv[1] if len(sys.argv) > 1 else "speaks"
    subject = sys.argv[2] if len(sys.argv) > 2 else "Tom"

    with Rainbird() as rb:
        session = rb.start()
        result = session.run(relationship, subject=subject, answerer=first_option)
        best = result.best  # the result with the highest certainty
        if best is None or not best.fact_id:
            print("No result to explain.")
            return
        print(f"best result: {best}\n")
        # One GET /analysis/evidence/{factID}/{sid} per node; max_depth=1 = direct conditions only
        nodes = {node.fact_id: node for node in session.evidence_tree(best.fact_id)}
        print_tree(nodes, best.fact_id, 0, set())
        print(f"\nopen in Studio: {session.evidence_url(best.fact_id)}")


if __name__ == "__main__":
    main()
