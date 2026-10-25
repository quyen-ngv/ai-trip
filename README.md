# GoRoute AI Trip Worker

Internal worker that turns an AI-trip job from the Java backend into a saved itinerary.
Java owns auth, quota, the job row and SSE; this service only calls back into
`/v1/api/internal/ai-trip-generations/{jobId}/{config|events|candidates|resolve-candidates|commit}`.

It is also the AI service of the Python side: the Pengoo assistant (`app/assistant/`) and the model
steps of social video extraction run here. Every chat / Responses call goes through `app/providers.py`
(one transport, provider quirks, the `ai_api_calls` log); speech-to-text is `app/transcription.py`.
A Claude endpoint (any tier, or a social job on `ANTHROPIC`/`CLAUDE`) runs through the official Anthropic
SDK in `app/anthropic_adapter.py`, which translates OpenAI-shaped requests and answers both ways, keeps
Claude's thinking blocks across Pengoo's tool rounds, and runs web research with Claude's `web_search`
tool (`web_search_20250305` on Haiku, `web_search_20260209` elsewhere).
google-maps-bot keeps the media work and calls, synchronously:

| Endpoint | Body | Answer |
|---|---|---|
| `POST /v1/social/transcribe` | the MP3 (`X-File-Name`) | `{transcript, provider}` |
| `POST /v1/social/prefilter` | `{metadata, transcript, language, aiProvider?, aiModel?, aiBaseUrl?}` | `{rejection: null | {reason, provider, model, usage}}` |
| `POST /v1/social/extract` | the same plus `transcriptProvider`, `images` (data URLs in video order) | `{extraction, provider, model, usage}`; 422/502 with `detail` |

All take `X-Internal-Token` and, to log under the Java job, `X-Ai-Log-Url`, `X-Ai-Log-Token`, `X-Trace-Id`.

## Design: the AI owns the plan

**One LLM call per destination** returns the complete itinerary — which places, which day, what
time, in what order, rest blocks, and all wording. Code never re-plans. It only:

- prepares a compact, deduplicated candidate shortlist (breakfast-capable venues guaranteed, up to
  a quarter of food slots and a third of activity slots reserved for cited Web rows, which carry
  no rating and would otherwise always rank last), with short refs (`f3`, `a7`) instead of UUIDs
  that are mapped back before the answer is used,
- fetches the editable rules from the `config` table and assembles the prompt,
- guards the answer against what Java's commit rejects (unknown/duplicate ids, overlapping or
  invalid times, PLACE_VISIT without placeId, over-long strings),
- guarantees three meals a day (adds the nearest suitable venue when the model skips one),
- emits TRANSPORT rows only between destinations (the app draws the legs inside a destination),
- asks the planner once more (`PLAN_ITINERARY_RETRY`, 1.5× `max_tokens`) when its JSON is unusable,
  then falls back to a code-only itinerary (`scheduler.py`).

```
validate → retrieve → [select_candidates | web_research] → curate_web → resolve_places → plan → verify → commit
```

| Step | Who | What |
|---|---|---|
| validate | code | Reject impossible date ranges; load prompt rules from `config` (label `AI_TRIP`). |
| retrieve | code | `/candidates` once per place group per destination (FOOD always requested), normalised. |
| select_candidates | **cheap LLM** | Runs in parallel with web_research. Reads the WHOLE catalogue of a destination (compact rows: title, kinds, short description, price, tags, dishes; rating last) with the traveller profile including free text, and returns the shortlist ids by fit: user's words and hard constraints → interests/style/budget → meal coverage and kind diversity → rating. `shortlist` ranks those first and uses `quality` only as a tie-breaker; a model failure falls back to quality ranking. Skipped when the catalogue already fits the shortlist caps. |
| web_research | **cheap AI query planner + browser Web Search** | Only for the groups the catalogue cannot cover (`research_needs` = `shortage`, floor `AI_TRIP_WEB_RESEARCH_MIN_CANDIDATES_PER_GROUP`); a destination with enough catalogue rows makes no search at all. A model writes one to three distinct Web queries per short group from the traveller's dietary, mobility, group, pace, activity, discovery, budget and free-text preferences. The browser tool must read at least three independent editorial/tourism/local-guide sources per query, then extracts only named places with cited URLs. Google Maps and map/directory results are excluded. Queries run concurrently; a shared job budget bounds the work, and when it runs out the unfinished queries are cancelled while the rows already found are kept (a destination with nothing in the catalogue, e.g. abroad, depends on them entirely). |
| curate_web | **cheap LLM** | Chooses only IDs supplied by the cited Web evidence, keeping diverse candidates that match the traveller request. |
| resolve_places | **Java catalogue / Goong or Mapbox** | Java first accepts only a unique exact normalized catalogue-title match within 50 km. Otherwise it calls the map provider server-side (Mapbox outside Vietnam, where Goong has nothing; the requester's provider inside) and accepts only an exact normalized result within 50 km. Raw Web addresses and coordinates are discarded; unresolved candidates have no address/coordinates. |
| plan | **quality LLM** | One call per destination → the area of each half-day, places, days, times, order, rest blocks, per-stop descriptions, trip description. Each candidate is sent as name, address, opening hours and description (the catalogue's `aiDescription`, or the cited reason for a Web row), plus kind, coordinates, rating/review count, visit length, price, tags and up to 6 signature dishes; no distance or attributes. |
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
| `DESCRIPTION_MAX_CHARS` | Character cap per activity description (default 600, shrunk for long trips). |

`app/prompts.py` holds the same text as `DEFAULT_*` constants: the fallback when the table or the
endpoint is unavailable, and the seed for migrations `V148__ai_trip_prompt_config.sql`, `V229`
and `V235__ai_trip_prompt_rules_2026_10.sql`. Keep them in sync when editing defaults in code: a
new migration updates the rows only while they still hold the previous seed, so an operator's edit
in goroute-admin survives a deploy.

How the prompt is built (2026-10-13; research in `.docs/AI_TRIP_PLANNING_KNOWLEDGE.md` §E):

- static text first (identity, the two rule blocks, `STOP_WRITING_RULES`, the input legend, the
  output contract), then the traveller profile and the task, so OpenAI/DeepSeek prefix caching
  bills the shared part as cached input (OpenAI also gets `prompt_cache_key = ai-trip-plan`);
- rules say what to do, grouped under short headers, about 45 atomic constraints in total;
- every pace block carries a day skeleton (anchors + light stops), the heat window is
  10:30-15:30, and the evening block after dinner is the default Vietnamese evening;
- the model writes each day's `area` (morning → afternoon) before its stops, and the prompt ends
  with a five-line self-check that repeats the hard constraints.

## Environment

| Var | Purpose |
|---|---|
| `AI_TRIP_INTERNAL_TOKEN` | Shared secret with Java. |
| `AI_PROVIDER` | The one switch: `claude`, `openai` or `deepseek` (shared with goroute-app). Every tier and social job uses that provider's key and models below. |
| `CLAUDE_API_KEY` (or `ANTHROPIC_API_KEY`), `CLAUDE_MODEL`, `CLAUDE_CHEAP_MODEL` | Claude through the official SDK; both models default to `claude-haiku-5-5`. |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_CHEAP_MODEL` | Default `gpt-6-luna`; the cheap model defaults to the main one. Web research needs a model with Responses Web Search. |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_CHEAP_MODEL` | Default `deepseek-v4-flash`. DeepSeek has no web search, so web research finds nothing. |
| Main vs cheap | Main (`QUALITY_LLM`, `ASSISTANT_LLM`): itinerary planning, Pengoo, social extraction. Cheap (`CHEAP_LLM`): research queries, web search and curation, social pre-filter. |
| `CHEAP_LLM_*`, `QUALITY_LLM_*`, `ASSISTANT_LLM_*` (`_PROVIDER/_BASE_URL/_API_KEY/_MODEL`) | Optional pins: each value set wins over the switch for that tier, e.g. `QUALITY_LLM_MODEL=claude-opus-5-5`. |
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
| `ASSISTANT_WEB_SEARCH_PROVIDER` | `openai` (default): the cheap model searches, with OpenAI Responses or Claude's `web_search` tool, so it follows `AI_PROVIDER`. `claude` forces Claude. `tinyfish`, `tavily`, `brave` use that API (`ASSISTANT_WEB_SEARCH_API_KEY`). |
| `ASSISTANT_LLM_REASONING_EFFORT` | Effort for Pengoo's model rounds (default `low`; empty = provider default). Claude: `output_config.effort`; OpenAI: `reasoning_effort`, dropped to `none` when the model rejects it with tools. |
| `ASSISTANT_CACHE_TTL` | Claude only: TTL of the explicit cache breakpoint on Pengoo's static rules (`1h` default, `5m`, or empty for the default). |
| `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL` | Only for a social job pinned to `OPENAI_COMPATIBLE`. |
| `AI_REQUEST_TIMEOUT_SECONDS` / `SOCIAL_AI_REASONING_EFFORT` | Social extraction timeout (default 120; pre-filter at most 30) and the OpenAI reasoning effort override. |
| `GROQ_API_KEY` / `GROQ_WHISPER_MODEL`, `OPENAI_WHISPER_MODEL` | Speech-to-text: Groq first, then OpenAI, then local Whisper. |
| `LOCAL_WHISPER_*` | `ENABLED` (true), `MODEL` (small, baked in the image), `DEVICE` (cpu), `COMPUTE_TYPE` (int8), `LANGUAGE` (vi), `CACHE_DIR`, `BEAM_SIZE` (1), `CPU_THREADS` (2). |
| `LOG_LEVEL` | Default INFO. Prompts and payloads are not logged, only their sizes and the provider's token `usage` (`LLM usage ...`, `Web research usage ...`). |

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
