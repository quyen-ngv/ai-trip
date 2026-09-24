# GoRoute AI Trip Worker

Internal worker that turns an AI-trip job from the Java backend into a saved itinerary.
Java owns auth, quota, the job row and SSE; this service only calls back into
`/v1/api/internal/ai-trip-generations/{jobId}/{config|events|candidates|resolve-candidates|commit}`.

## Design: the AI owns the plan

**One LLM call per destination** returns the complete itinerary — which places, which day, what
time, in what order, rest blocks, and all wording. Code never re-plans. It only:

- prepares a compact, deduplicated candidate shortlist (breakfast-capable venues guaranteed),
- fetches the editable rules from the `config` table and assembles the prompt,
- guards the answer against what Java's commit rejects (unknown/duplicate ids, overlapping or
  invalid times, PLACE_VISIT without placeId, over-long strings),
- guarantees three meals a day (adds the nearest suitable venue when the model skips one),
- emits TRANSPORT rows only for hops ≥ 5 km and between destinations,
- falls back to a code-only itinerary (`scheduler.py`) if the model's JSON is unusable.

```
validate → retrieve → research → plan → verify → commit
```

| Step | Who | What |
|---|---|---|
| validate | code | Reject impossible date ranges; load prompt rules from `config` (label `AI_TRIP`). |
| retrieve | code | `/candidates` once per place group per destination (FOOD always requested), normalised. |
| web_research | **cheap AI query planner + browser Web Search** | For every destination with coordinates, a model writes one to three distinct Web queries per Food/requested spot group from the traveller's dietary, mobility, group, pace, activity, discovery, budget and free-text preferences. The browser tool must read at least three independent editorial/tourism/local-guide sources per query, then extracts only named places with cited URLs. Google Maps and map/directory results are excluded. A shared job budget bounds the work. |
| curate_web | **cheap LLM** | Chooses only IDs supplied by the cited Web evidence, keeping diverse candidates that match the traveller request. |
| resolve_places | **Java catalogue / Goong** | Java first accepts only a unique exact normalized catalogue-title match within 50 km. Otherwise it calls Goong server-side and accepts only an exact normalized result within 50 km. Raw Web addresses and coordinates are discarded; unresolved candidates have no address/coordinates. |
| plan | **quality LLM** | One call per destination → places, days, times, order, rest blocks, per-stop descriptions, trip description. |
| verify | code | Logs remaining violations (missing meal, thin day, duplicates). **Does not modify the plan.** |
| commit | code | `validators.finalize` enforces Java's contract, then posts. |

Activity names are always the `title` from `places` or from a cited Web extraction — the itinerary
model never names anything. Web results have no `placeId`, so they commit as `ACTIVITY` and retain
their source URLs in `notes`; no Web result is silently imported into the catalogue.

## Editable rules (no deploy needed)

Rows in `config` with label `AI_TRIP`, edited in goroute-admin, read at the start of every job:

| key | Purpose |
|---|---|
| `PLAN_RULES` | Core rules: meals, geography/travel time, time-of-day fit, visit lengths, rest blocks, descriptions. |
| `TRAVELLER_RULES` | Rules per profile: pace, children, elderly, couples, friends, solo, adventure, photography, food lovers, dietary, arrival/departure. |
| `DESCRIPTION_MAX_CHARS` | Character cap per activity description (default 200). |

`app/prompts.py` holds the same text as `DEFAULT_*` constants: the fallback when the table or the
endpoint is unavailable, and the seed for migration `V148__ai_trip_prompt_config.sql`. Keep them
in sync when editing defaults in code.

## Environment

| Var | Purpose |
|---|---|
| `AI_TRIP_INTERNAL_TOKEN` | Shared secret with Java. |
| `QUALITY_LLM_BASE_URL/_API_KEY/_MODEL` | Quality model for final itinerary planning. Falls back to `CHEAP_LLM_*` only for that planning call. |
| `CHEAP_LLM_BASE_URL/_API_KEY/_MODEL` | Required for Web research: it writes contextual queries and invokes the Responses `web_search` tool. Its endpoint/model must support Responses Web Search; this stage never falls back to the quality model. |
| `LLM_CACHE_PATH` | SQLite cache path (planning calls are not cached). |
| `AI_TRIP_WEB_RESEARCH_BUDGET_SECONDS` | Total browser-research budget per job, split across destinations (default 300). |
| `AI_TRIP_WEB_RESEARCH_MIN_CANDIDATES_PER_GROUP` | Minimum Web additions sought for Food and each requested spot group even when the catalogue is sufficient (default 3). |
| `AI_TRIP_WEB_RESEARCH_MIN_QUERIES_PER_GROUP` / `AI_TRIP_WEB_RESEARCH_MAX_QUERIES_PER_GROUP` | Bounds independently phrased AI queries per group (defaults 1 / 3; larger data gaps request more). |
| `AI_TRIP_WEB_RESEARCH_MIN_SOURCES_PER_QUERY` | Minimum independent cited Web sources the browser must read for each query (default 3; minimum 2). |
| `AI_TRIP_WEB_RESEARCH_MAX_TOOL_CALLS` | Upper bound on Web Search tool calls for one query (default 6). |
| `AI_TRIP_WEB_RESEARCH_TIMEOUT_SECONDS` | Timeout for one browser Web Search response (default 180). |
| `AI_TRIP_RESEARCH_CACHE_PATH` | SQLite path for Web-research and curation TTL cache (default `/tmp/ai_trip_research_cache.sqlite3`). |
| `AI_TRIP_WEB_RESEARCH_CACHE_TTL_SECONDS` / `AI_TRIP_WEB_CURATION_CACHE_TTL_SECONDS` | TTL for the two cheap-model node caches (defaults 21600 / 21600). Set `0` to disable either cache. |
| `AI_TRIP_WEB_CURATION_MAX_PER_GROUP` | Maximum Web candidates sent per group to Java identity resolution (default 12). |
| `AI_TRIP_CANDIDATES_PER_GROUP` | `/candidates` limit per group (default 120). |
| `AI_TRIP_LLM_TIMEOUT_SECONDS` | HTTP timeout for the planning call (default 180). |
| `LOG_LEVEL` | Default INFO. Prompts and payloads are not logged, only their sizes. |

## Multi-destination trips

Destinations are validated (chronological, no inverted ranges) and a shared handover day is
assigned to exactly one of them, so two provinces never plan the same day. Legs under 150 km
become a morning drive on the arrival day (that day starts later); longer legs become an
overnight block ending 08:00 on the arrival day.

## Tests

```
.venv/Scripts/python -m pytest -q
```

`tests/test_workflow.py` runs the whole graph against a fake Java and a fake LLM;
`tests/test_itinerary.py` locks in the safety net (AI times/order/wording respected, missing meals
added, overlaps fixed, bad ids dropped, unusable JSON raises so the caller can fall back);
the rest cover hours parsing, candidate normalisation (against the exact Java row shape),
the code-only scheduler, `finalize`, and the research client.
