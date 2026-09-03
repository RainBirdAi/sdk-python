"""Create a knowledge map from an RBLang string, then download its source again.

!!! UNDOCUMENTED, UNSTABLE ENDPOINTS !!!
``Rainbird.create_map`` (``POST /maps``) and ``Rainbird.get_map_source``
(``GET /analysis/file/{kmID}``) wrap endpoints that are NOT in Rainbird's public OpenAPI
specification. They are not covered by the API documentation, may change or disappear
without notice, and the SDK's handling is based on observed behaviour. Treat them as
unstable and check with Rainbird before relying on them.

One such behaviour: ``POST /maps`` answers HTTP 200 even when the RBLang does not parse,
signalling the failure with an ``error`` key in the body. The SDK detects that and raises
``KnowledgeMapError`` instead of returning a result.

Environment variables:
    RAINBIRD_API_KEY   API key allowed to create knowledge maps
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai
    (RAINBIRD_KM_ID is not needed: the script creates a map and prints its id.)

Usage: python examples/create_map_from_rblang.py ["Map name"]
"""

from __future__ import annotations

import sys

from rainbird_ai import KnowledgeMapError, Rainbird

# Minimal RBLang: two concepts and one relationship between them. Concept types are
# string | number | date | truth; facts are added with <concinst> and <relinst> elements.
RBLANG = """<?xml version="1.0" encoding="utf-8"?>
<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">
    <concept name="Person" type="string"/>
    <concept name="Language" type="string"/>
    <rel name="speaks" subject="Person" object="Language"/>
</rbl:kb>
"""


def main() -> None:
    name = " ".join(sys.argv[1:]) or "SDK example map"

    with Rainbird() as rb:  # reads RAINBIRD_API_KEY / RAINBIRD_API_URL
        try:
            created = rb.create_map(name, RBLANG)
        except KnowledgeMapError as exc:
            # HTTP 200 with {"error": "..."}: Rainbird's parser rejected the RBLang.
            print(f"Rainbird rejected the RBLang: {exc.message}")
            for detail in exc.errors:
                print(f"  - {detail}")
            raise SystemExit(1) from exc
        # km_id is the GUID to pass to rb.start(); id is Rainbird's numeric id.
        print(f"created {created.name!r}: km_id={created.km_id} id={created.id}")

        source = rb.get_map_source(created.km_id)  # {"concepts", "rels", "rblang"}
        print(f"concepts: {source.concepts}")
        print(f"relationships: {source.relationships}")
        print("rblang:")
        print(source.rblang)
        print(f"start a session with: rb.start({created.km_id!r})")


if __name__ == "__main__":
    main()
