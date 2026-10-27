# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Knowledge map test suites, test runs, readmes and RBLang validation on `Rainbird` and
  `AsyncRainbird`. **None of these endpoints is in the public OpenAPI specification; they may
  change.** All send `Version: v1`.
  - `create_tests(km_id, tests)` (`POST /maps/{kmID}/tests` -> `TestSuiteCreated`) and
    `get_tests(km_id)` (`GET /maps/{kmID}/tests` -> `TestSuite`) read and write the suite that
    Studio's Unit Tester uses. `create_tests()` also takes a `TestSuite` or its `{"tests": [...]}`
    body, so a suite can be copied between maps, and raises `InvalidFactError` before sending a
    fact that gives `certainty` instead of `cf` (the test runner would ignore it and use 100).
  - `start_test_run(km_id)` (`POST /maps/{kmID}/runs` -> `TestRunStarted`),
    `get_test_run(run_id)` (`GET /runs/{runID}` -> `TestRun`), `wait_for_test_run(run_id, *,
    poll_interval=1.0, timeout=300.0)` and `run_tests(km_id, ...)` (start, then wait). Waiting
    returns the `complete` run (check `TestRun.all_passed`), raises the new `TestRunFailedError`
    for a `failed` run and the new `TestRunTimeoutError` (a `RainbirdTimeoutError`) when the run
    has not finished in time.
  - `add_readme(km_id, readme)` (`POST /maps/{kmID}/readme` -> `ReadmeAdded`) sets the draft's
    Markdown readme when it has none.
  - `validate_rblang(rblang)` (`POST /validate` -> `ValidationResult`) compiles RBLang without
    storing it. **Not yet deployed on any Rainbird host as of 2026-10-27**: until the platform
    release that ships it, the call raises `NotFoundError`.
  - Models `TestSuite`, `TestSuiteCreated`, `TestRunStarted`, `TestRun`, `TestRunSummary`,
    `TestResult`, `TestRunError`, `ReadmeAdded`, `ValidationResult`, `ValidationMessage` and the
    enums `TestRunStatus` and `TestResultStatus`. Classes whose names start with `Test` set
    `__test__ = False`, so pytest does not try to collect them from your test modules.
- `ConflictError` (HTTP 409) and `RequestTooLargeError` (HTTP 413), both subclasses of
  `RainbirdAPIError`.
- CLI: `rainbird map validate`, `map add-readme`, `map get-tests`, `map create-tests`,
  `map run-tests` (`--no-wait`, `--poll-interval`, `--max-wait`) and `map test-run`. A new exit
  code 4 means a check failed (failing tests, invalid RBLang); exit 3 also covers a test run that
  has not finished.
- `examples/run_map_tests.py`: validate a map, give it a suite if it has none, run the suite.

### Changed

- HTTP 409 and 413 responses raise `ConflictError` / `RequestTooLargeError` instead of a plain
  `RainbirdAPIError`. Both subclass `RainbirdAPIError`, so existing `except` clauses still match.
- An error body of the form `{"error": {"type": "UNHANDLED", "trace": "<uuid>"}}` (or just
  `{"type": ...}`) now gives the message `UNHANDLED (trace <uuid>)` instead of the JSON dump of
  the inner object.

## [0.5.0] - 2026-10-22

First release published on PyPI.

[Unreleased]: https://github.com/RainBirdAi/sdk-python/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/RainBirdAi/sdk-python/releases/tag/v0.5.0
