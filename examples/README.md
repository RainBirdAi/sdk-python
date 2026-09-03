# Examples

Runnable scripts for the `rainbird_ai` SDK. Each one reads its configuration from the
environment. The query scripts (`basic_query.py`, `automated_answerer.py`, `evidence_tree.py`,
`inject_and_query.py`) take the relationship and subject to query as optional arguments
(defaults `speaks` / `Tom`; use values from your own knowledge map); `async_query.py` takes a
relationship followed by any number of subjects (defaults `speaks Tom Ana`);
`natural_language.py` takes a prompt; `create_map_from_rblang.py` takes a map name and needs no
`RAINBIRD_KM_ID`:

```bash
pip install rainbird-ai
export RAINBIRD_API_KEY=...   # Rainbird Studio > knowledge map > Publish
export RAINBIRD_KM_ID=...     # knowledge map id (not needed by create_map_from_rblang.py)
export RAINBIRD_API_URL=...   # optional; default https://api.rainbird.ai
python examples/basic_query.py speaks Tom
python examples/async_query.py speaks Tom Ana
python examples/natural_language.py "Which language does Tom speak? He lives in Germany"
python examples/create_map_from_rblang.py "My map"
```

| Script | Shows |
| --- | --- |
| `basic_query.py` | Key-less `engine_version()` (`GET /version` needs no API key), then the sync query loop: print each question and its options, answer from `input()`. |
| `automated_answerer.py` | `session.run()` with an answerer that picks from `question.options`, returns `SKIP` when allowed and `True` for yes/no questions; marks where an LLM call would go. |
| `async_query.py` | `AsyncRainbird` under `asyncio.run()`: a manual awaited loop, then several sessions in parallel with `asyncio.gather`. |
| `evidence_tree.py` | Query, then `session.evidence_tree()` for the best result printed as a nested tree in the layout of `rainbird evidence --tree` (fetched conditions become child nodes, the rest `- ...` lines) plus the Studio `evidence_url()`. |
| `inject_and_query.py` | Inject a list of `Fact` objects (batched automatically; note on accepted object types such as `date`), list the session's facts filtered by relationship, then query. |
| `natural_language.py` | `/nl/interact` loop handling question, result and error replies, then `/nl/explain` for the first result's fact id. |
| `create_map_from_rblang.py` | Undocumented endpoints: `create_map()` from an RBLang string, `get_map_source()`, and `KnowledgeMapError` handling. |

The scripts call the live Rainbird API and create real sessions (and, for the last one, a
real knowledge map).
