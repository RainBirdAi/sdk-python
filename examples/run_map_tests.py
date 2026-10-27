"""Check a knowledge map the way a CI job would: validate its RBLang, make sure it has a test
suite, run the suite on the draft and exit non-zero when a test fails.

!!! UNDOCUMENTED, UNSTABLE ENDPOINTS !!!
The endpoints used here (``/analysis/file/{kmID}``, ``/maps/{kmID}/tests``,
``/maps/{kmID}/runs``, ``/runs/{runID}`` and ``/validate``) are NOT in Rainbird's public
OpenAPI specification and may change without notice. ``POST /validate`` was not yet deployed
on any Rainbird host in October 2026, so the script skips that step when the host answers 404.

When the map has no tests, one is created from the arguments: query SUBJECT RELATIONSHIP and
expect exactly one result, OBJECT. Tests created through the API appear in Studio's Unit
Tester, and a map's suite can be created only once (later changes are made in Studio).

Environment variables:
    RAINBIRD_API_KEY   API key with access to the map
    RAINBIRD_KM_ID     the knowledge map to test
    RAINBIRD_API_URL   optional API host, default https://api.rainbird.ai

Usage: python examples/run_map_tests.py [SUBJECT RELATIONSHIP OBJECT]   (default: Tom speaks German)
"""

from __future__ import annotations

import os
import sys
from typing import Any

from rainbird_ai import NotFoundError, Rainbird, TestRunFailedError, TestRunTimeoutError


def one_test(subject: str, relationship: str, obj: str) -> dict[str, Any]:
    # A step's "expect" is checked against the response to the PREVIOUS step, so the query is
    # the first step's "then" and the expected result is the second step's "expect".
    # Test suites write certainty as "cf" (create_tests() rejects "certainty").
    return {
        "name": f"{subject} {relationship} {obj}",
        "steps": [
            {
                "expect": {"type": "none"},
                "then": {"type": "Query", "subject": subject, "relationship": relationship},
            },
            {
                "expect": {
                    "type": "Expect Result",
                    "results": [
                        {"subject": subject, "relationship": relationship, "object": obj, "cf": 100}
                    ],
                },
                "then": {"type": "none"},
            },
        ],
    }


def main() -> None:
    km_id = os.environ["RAINBIRD_KM_ID"]
    args = sys.argv[1:]
    subject, relationship, obj = args if len(args) == 3 else ("Tom", "speaks", "German")

    with Rainbird() as rb:  # reads RAINBIRD_API_KEY / RAINBIRD_API_URL
        # 1. Does the map's RBLang compile? POST /validate stores nothing.
        try:
            checked = rb.validate_rblang(rb.get_map_source(km_id).rblang)
        except NotFoundError:
            print("validate: not available on this host yet, skipped")
        else:
            for message in [*checked.errors, *checked.warnings]:
                print(f"validate: {message}")  # "line 12: ..." when the compiler gives a line
            if not checked.valid:
                raise SystemExit(1)

        # 2. Give the map a suite if it has none.
        try:
            suite = rb.get_tests(km_id)
        except NotFoundError:
            rb.create_tests(km_id, [one_test(subject, relationship, obj)])
            suite = rb.get_tests(km_id)
        names = ", ".join(str(test.get("name", "?")) for test in suite.tests)
        print(f"{suite.test_count} test(s): {names}")

        # 3. Run the whole suite on the draft and wait for it.
        try:
            run = rb.run_tests(km_id, poll_interval=2, timeout=600)
        except TestRunFailedError as exc:  # the run itself could not finish
            print(f"{exc} (category {exc.category})")
            raise SystemExit(1) from exc
        except TestRunTimeoutError as exc:  # the run carries on server-side
            print(f"still running; keep waiting with rb.wait_for_test_run({exc.run_id!r})")
            raise SystemExit(1) from exc

    summary = run.summary
    print(
        f"{summary.passed} of {summary.total} passed ({summary.failed} failed, "
        f"{summary.errored} errored)"
    )
    for failure in run.failures:  # "expected" / "actual" hold the step's expectation and reply
        print(
            f"  {failure.status} [{failure.index}] {failure.name} at step "
            f"{failure.failed_step_index}: {'; '.join(failure.errors)}"
        )
    raise SystemExit(0 if run.all_passed else 1)


if __name__ == "__main__":
    main()
