"""Natural-language interaction (beta): chat with a knowledge map, then ask why.

``Session.interact(prompt)`` posts to ``/nl/interact``. The engine extracts a query and any
facts from the text and replies with a question, a result or an error (``response_type``).
Errors such as "no query detected" are returned, not raised. While a question is pending
the session is locked to that query: send the answer as the next ``interact`` prompt.
``Session.explain(fact_id)`` posts to ``/nl/explain`` for a plain-language explanation.

Environment variables:
    RAINBIRD_API_KEY   API key (Rainbird Studio > knowledge map > Publish)
    RAINBIRD_KM_ID     knowledge map id
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/natural_language.py ["Which language does Tom speak? He lives in Germany"]
"""

from __future__ import annotations

import sys

from rainbird_ai import NaturalLanguageError, Rainbird


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Which language does Tom speak? He lives in Germany."

    with Rainbird() as rb:
        session = rb.start()
        reply = session.interact(prompt)
        while reply.is_question or reply.is_error:
            if reply.is_error:
                # reply.prompt is the chat-friendly suggested_chat_response (or the message);
                # reply.error.code / .message are for logs. Let the user rephrase.
                code = reply.error.code if reply.error else None
                print(f"engine: {reply.prompt}  (error code {code})")
            else:
                # The answer goes back as plain text in the next interact() call.
                for question in reply.questions:
                    print(f"engine: {question.prompt}")
                    if question.options:
                        print("  options:", ", ".join(str(o) for o in question.options))
            reply = session.interact(input("you> "))

        for fact in reply.injected_facts:  # facts the engine extracted from the prompts
            print(f"  fact: {fact.get('subject')} {fact.get('relationship')} {fact.get('object')}")
        if reply.unmatched:  # text the engine could not use
            print(f"  ignored: {reply.unmatched}")
        for item in reply.results:  # ResultItem (the NL endpoint's Capitalised keys are handled)
            print(f"result: {item}  factID={item.fact_id}")

        fact_ids = [item.fact_id for item in reply.results if item.fact_id]
        if not fact_ids:
            return
        try:
            explanation = session.explain(
                fact_ids[0], language="en", instructions="Keep it to two sentences."
            )
        except NaturalLanguageError as exc:  # /nl/explain returned an error object
            print(f"explain failed: {exc.message} (code {exc.code})")
            return
        print(f"\nwhy: {explanation.explanation}")
        if explanation.metadata:
            print(f"({explanation.metadata.total_tokens} LLM tokens)")


if __name__ == "__main__":
    main()
