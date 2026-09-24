"""LangGraph pipeline for one AI trip job.

validate -> retrieve -> web_research -> curate_web -> resolve_places -> plan -> verify -> commit

The AI owns the itinerary: ONE call per destination returns places, days, times, order, rest
blocks and all wording. Code prepares a compact candidate shortlist, fetches the editable rules
from the `config` table, guards the result against what Java's commit rejects, and logs checks.
If the model fails, a code-only fallback (scheduler.py) still produces a valid trip.
"""
from __future__ import annotations
import asyncio, logging, time
from datetime import date
from typing import Any, TypedDict
import httpx
from langgraph.graph import StateGraph, START, END
from . import config
from .candidates import (ALL_GROUPS, FOOD, bucket, capacity, compact_for_plan, destination_dates,
                         is_food, merge, normalize, pace_plan, quality, research_needs, row_key)
from .geo import WEEKDAYS, haversine_km
from .itinerary import append_missing_social_items, materialize
from .llm import CachedModel
from .prompts import (CONFIG_LABEL, DEFAULT_DESCRIPTION_MAX_CHARS, KEY_DESCRIPTION_MAX_CHARS, build_itinerary_prompt)
from .research import WebResearcher
from .scheduler import intercity_transport, schedule_destination
from .validators import check, finalize, strip_internal

logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    job: dict[str, Any]
    cfg: dict[str, str]
    inventories: list[list[dict]]
    raw_web: list[list[dict]]
    curated_web: list[list[dict]]
    items: list[dict]
    description: str
    violations: list[str]


class JavaClient:
    def __init__(self, job: dict[str, Any], token: str):
        self.job, self.token = job, token

    @property
    def root(self) -> str:
        return self.job["callbackBaseUrl"].rstrip("/") + f"/v1/api/internal/ai-trip-generations/{self.job['jobId']}"

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Internal-Token": self.token, "X-Attempt-Id": self.job["attemptId"]}

    async def event(self, stage: str, progress: int, key: str, status: str = "RUNNING", error: str | None = None,
                    params: dict | None = None) -> None:
        body = {"attemptId": self.job["attemptId"], "stage": stage, "status": status, "progress": progress,
                "messageKey": key, "params": params or {}, "errorMessage": error}
        async with httpx.AsyncClient(timeout=30) as c:
            (await c.post(self.root + "/events", headers=self.headers, json=body)).raise_for_status()

    async def config(self) -> dict[str, str]:
        """Editable prompt rules from the `config` table (label AI_TRIP). Empty dict on any failure."""
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(self.root + "/config", headers=self.headers)
                r.raise_for_status()
                data = r.json().get("data") or {}
                return {str(k): str(v) for k, v in data.items() if v is not None}
        except Exception as e:
            logger.warning("config fetch failed, using built-in rules: %s", e)
            return {}

    async def candidates(self, d: dict, groups: list[str], limit: int) -> list[dict]:
        body = {"latitude": d["latitude"], "longitude": d["longitude"], "placeGroups": groups, "limit": limit}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(self.root + "/candidates", headers=self.headers, json=body)
            r.raise_for_status()
            return r.json().get("data") or []

    async def resolve_candidates(self, d: dict, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        body = {
            "latitude": d["latitude"], "longitude": d["longitude"],
            "candidates": [{
                "candidateId": row.get("candidateId"), "title": row.get("title"),
                "placeGroup": row.get("placeGroup"),
            } for row in candidates],
        }
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(self.root + "/resolve-candidates", headers=self.headers, json=body)
            r.raise_for_status()
            return r.json().get("data") or []

    async def commit(self, items: list[dict], description: str) -> Any:
        body = {"attemptId": self.job["attemptId"], "items": items, "tripDescription": description}
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(self.root + "/commit", headers=self.headers, json=body)
            if r.status_code != 200:
                logger.error("commit failed: %s %s", r.status_code, r.text[:500])
                r.raise_for_status()
            data = r.json()
            meta = data.get("meta") or {}
            if meta.get("code") and meta["code"] != 200000:
                raise RuntimeError(meta.get("message") or "commit rejected")
            return data.get("data")


# ------------------------------------------------------------------ helpers
def _day_number(first_start: date, d: date) -> int:
    return (d - first_start).days + 1


def _validate_destinations(dests: list[dict]) -> None:
    prev_start = None
    for d in dests:
        start, end = date.fromisoformat(d["startDate"]), date.fromisoformat(d["endDate"])
        if end < start:
            raise RuntimeError(f"Destination '{d.get('name') or ''}' has endDate before startDate")
        if prev_start is not None and start < prev_start:
            raise RuntimeError("Destinations are not in chronological order")
        prev_start = start


def _resolve_dest_dates(dests: list[dict]) -> list[list[date]]:
    """Each destination's dates, with shared handover days given to exactly one of them.

    When destination i starts on the day destination i-1 ends, both would otherwise plan a
    full day for the same dayNumber. The arriving destination keeps the day (the traveller
    spends its afternoon/evening there); a 1-day destination keeps its only day and the
    other side gives up instead."""
    all_dates = [destination_dates(d) for d in dests]
    for i in range(1, len(all_dates)):
        prev, cur = all_dates[i - 1], all_dates[i]
        while cur and prev and cur[0] <= prev[-1]:
            if len(prev) > 1 or len(cur) == 1:
                if cur[0] == prev[-1]:
                    prev.pop()
                else:
                    cur.pop(0)
            else:
                cur.pop(0)
    return all_dates


def _spread_by_group(rows: list[dict], limit: int, key=quality) -> list[dict]:
    """Round-robin over place groups so one dominant group cannot crowd out the others."""
    groups = {g: sorted(v, key=key, reverse=True) for g, v in bucket(rows).items() if v}
    out: list[dict] = []
    while len(out) < limit and any(groups.values()):
        for g in list(groups):
            if groups[g] and len(out) < limit:
                out.append(groups[g].pop(0))
    return out


def shortlist(rows: list[dict], pace: str | None, days: int) -> tuple[list[dict], list[dict]]:
    """Compact candidate set for the model: enough food for 3 meals/day with breakfast-capable
    venues guaranteed, and activities spread across groups. Pure code, no LLM."""
    cap = capacity(pace, days)
    social = [r for r in rows if r.get("source") == "social"]
    food_all = sorted((r for r in rows if is_food(r)), key=quality, reverse=True)
    social_food = [r for r in social if is_food(r)]
    breakfasty = [r for r in food_all if "breakfast" in (r.get("meals") or []) or _looks_breakfast(r)]
    food, seen = [], set()
    for r in [*breakfasty[: max(4, days * 2)], *food_all]:
        key = row_key(r)
        if key not in seen:
            seen.add(key)
            food.append(r)
    food_limit = max(16, cap["meals"] * 3)
    food = food[:food_limit]
    # A social row is pinned by the source video and must never be removed by the catalogue cap.
    food = social_food + [r for r in food if r not in social_food]
    acts_limit = max(14, int(cap["activities"] * 3))
    acts = [r for r in social if not is_food(r) and r["placeGroup"] != "ACCOMMODATION"]
    social_activity_keys = {row_key(r) for r in acts}
    acts += [r for r in _spread_by_group(
        [r for r in rows if r.get("source") != "social" and not is_food(r)
         and r["placeGroup"] != "ACCOMMODATION"],
        acts_limit,
    ) if row_key(r) not in social_activity_keys]
    return food, acts


def _destination_center(destination: dict, rows: list[dict]) -> dict:
    latitude, longitude = destination.get("latitude"), destination.get("longitude")
    if latitude is not None and longitude is not None:
        return {"latitude": float(latitude), "longitude": float(longitude)}
    for row in rows:
        if row.get("latitude") is not None and row.get("longitude") is not None:
            return {"latitude": row["latitude"], "longitude": row["longitude"]}
    return {"latitude": 0.0, "longitude": 0.0}


def _research_destination_input(destination: dict, fallback_name: str | None = None) -> dict:
    """Guarantee usable AI Web-research context without changing the persisted request.

    The V2 contract validates coordinates but makes ``destinations[].name`` optional.  A blank
    name used to make the researcher return before creating a query. Use the primary city name
    when available; for later unnamed destinations the planner receives coordinate context too.
    """
    out = dict(destination)
    name = str(out.get("name") or "").strip() or str(fallback_name or "").strip()
    if not name and out.get("latitude") is not None and out.get("longitude") is not None:
        name = f"{float(out['latitude']):.5f},{float(out['longitude']):.5f}"
    out["name"] = name
    return out


def _looks_breakfast(r: dict) -> bool:
    from .candidates import meal_hint
    return "breakfast" in meal_hint(r)


FALLBACK_DESC = {
    "vi": "Lịch trình được cân bằng theo sở thích, khoảng cách di chuyển và giờ mở cửa của từng địa điểm.",
    "en": "An itinerary balanced around your interests, travel distances and each place's opening hours.",
}
FALLBACK_NOTE = {"vi": "Phù hợp với nhịp độ và tuyến đường trong ngày.", "en": "Fits the day's pace and route."}


# ------------------------------------------------------------------ graph
async def build_workflow(job: dict[str, Any], token: str):
    java = JavaClient(job, token)
    req = job["request"]
    locale = "vi" if (job.get("locale") or "").lower().startswith("vi") else "en"
    dests = req["destinations"]
    pace = req.get("pace") or "BALANCED"
    selected = [str(g).upper() for g in (req.get("placeGroups") or []) if g]
    query_groups = list(dict.fromkeys([g for g in selected if g in ALL_GROUPS] + [FOOD])) if selected else ALL_GROUPS
    first_start = date.fromisoformat(dests[0]["startDate"])
    total_days = (date.fromisoformat(dests[-1]["endDate"]) - first_start).days + 1
    _validate_destinations(dests)                # fail fast, before any quota-costing LLM/candidate work
    dest_dates = _resolve_dest_dates(dests)      # handover days assigned to exactly one destination
    pace_start_minutes = pace_plan(pace)["start"]
    model = CachedModel("QUALITY_LLM", "CHEAP_LLM")
    research_model = CachedModel("CHEAP_LLM")

    async def validate(s: State):
        await java.event("VALIDATING", 5, "ai_trip.validating")
        return {"cfg": await java.config()}

    async def retrieve(s: State):
        await java.event("RETRIEVING", 15, "ai_trip.retrieving")

        async def one(d: dict) -> list[dict]:
            social_context = req.get("socialContext") if isinstance(req.get("socialContext"), dict) else {}
            social_rows = [normalize(r, source="social") for r in (social_context.get("candidates") or [])
                           if isinstance(r, dict)]
            if d.get("latitude") is not None and d.get("longitude") is not None:
                lists = await asyncio.gather(*(java.candidates(d, [g], config.CANDIDATES_PER_GROUP) for g in query_groups))
                catalogue_rows = merge(*([normalize(r) for r in lst] for lst in lists))
            else:
                # An unresolved social destination is still a valid input. The source rows are
                # enough for the planner; do not call Java's coordinate-required catalogue API.
                catalogue_rows = []
            rows = merge(social_rows, catalogue_rows)
            logger.info("retrieve %s: %s", d.get("name"), {g: len(v) for g, v in bucket(rows).items() if v})
            return rows

        return {"inventories": list(await asyncio.gather(*(one(d) for d in dests)))}

    async def web_research(s: State):
        inventories = list(s["inventories"])
        raw_web: list[list[dict]] = [[] for _ in dests]

        async def report(kind: str, params: dict):
            try:
                await java.event("RESEARCHING", 25, "ai_trip.researching", params=params)
            except Exception as e:  # progress is best effort
                logger.warning("event failed: %s", e)

        # One TOTAL research budget for the whole job, split across every routable destination.
        # AI-led Web research is mandatory even when the local catalogue is already sufficient.
        remaining = float(config.WEB_RESEARCH_BUDGET_SECONDS)
        researching = [
            (i, _research_destination_input(d, req.get("cityName") if i == 0 else None), research_needs(
                inventories[i], pace, len(dest_dates[i]), selected,
                config.WEB_RESEARCH_MIN_CANDIDATES_PER_GROUP,
            ))
            for i, d in enumerate(dests)
            if dest_dates[i] and d.get("latitude") is not None and d.get("longitude") is not None
        ]
        for pos, (i, d, needs) in enumerate(researching):
            if remaining <= 0:
                logger.warning("research budget exhausted; skipping %s (%s)", d.get("name"), needs)
                continue
            destinations_left = len(researching) - pos
            researcher = WebResearcher(
                locale=locale,
                request=req,
                query_model=research_model,
                reporter=report,
            )
            if not researcher.enabled:
                raise RuntimeError("AI Web research is unavailable: configure CHEAP_LLM_* with a Responses/Web Search-capable provider")
            await java.event("RESEARCHING", 25, "ai_trip.researching", params={"destination": d.get("name") or "", "needs": needs})
            started = time.monotonic()
            budget = max(1, int(remaining / destinations_left))
            try:
                # Search is independent of the local catalogue: an existing title still needs
                # citation evidence and Java will later resolve the identity in one controlled step.
                rows = await asyncio.wait_for(researcher.research_destination(d, needs, []), timeout=budget)
            except TimeoutError as exc:
                raise RuntimeError(f"AI Web research timed out for {d.get('name') or 'destination'}") from exc
            remaining -= time.monotonic() - started
            raw_web[i] = rows
        return {"inventories": inventories, "raw_web": raw_web}

    async def curate_web(s: State):
        await java.event("WEB_CURATING", 38, "ai_trip.web_curating")
        curated: list[list[dict]] = []
        for i, rows in enumerate(s.get("raw_web") or []):
            if not rows:
                curated.append([])
                continue
            destination = _research_destination_input(dests[i], req.get("cityName") if i == 0 else None)
            researcher = WebResearcher(locale=locale, request=req, query_model=research_model)
            selected_rows = await researcher.curate_candidates(destination, rows)
            curated.append(selected_rows)
            logger.info("curate %s: %d -> %d Web candidates", destination.get("name"), len(rows), len(selected_rows))
        return {"curated_web": curated}

    async def resolve_places(s: State):
        await java.event("RESOLVING_PLACES", 44, "ai_trip.resolving_places")
        inventories = list(s["inventories"])
        curated = s.get("curated_web") or [[] for _ in dests]
        for i, rows in enumerate(curated):
            if not rows or dests[i].get("latitude") is None or dests[i].get("longitude") is None:
                continue
            resolved = await java.resolve_candidates(dests[i], rows)
            by_id = {str(row.get("candidateId")): row for row in resolved if isinstance(row, dict) and row.get("candidateId")}
            enriched: list[dict] = []
            for raw in rows:
                result = by_id.get(str(raw.get("candidateId")))
                if result:
                    # Keep the Web rationale/citations from Python; trust only identity/location
                    # fields returned by Java's exact DB/Goong resolver.
                    row = {**raw, **result, "sourceUrls": raw.get("sourceUrls") or [],
                           "sourceTitles": raw.get("sourceTitles") or [], "why": raw.get("why") or ""}
                    enriched.append(normalize(row, source="web"))
                else:
                    enriched.append(raw)
            # Resolved candidates go first so their cited context is retained when they equal a
            # catalogue place already returned by `retrieve`.
            inventories[i] = merge(enriched, inventories[i])
            logger.info("resolve %s: %d candidates", dests[i].get("name"), len(enriched))
        return {"inventories": inventories}

    async def plan(s: State):
        await java.event("DEEP_SELECT", 50, "ai_trip.deep_select")
        cfg = s.get("cfg") or {}
        max_chars = int(cfg.get(KEY_DESCRIPTION_MAX_CHARS) or DEFAULT_DESCRIPTION_MAX_CHARS)

        async def ask(i: int) -> Any:
            d, rows, dates = dests[i], s["inventories"][i], dest_dates[i]
            if not dates or not rows:
                return None
            day_numbers = [_day_number(first_start, x) for x in dates]
            weekdays = [WEEKDAYS[x.weekday()] for x in dates]
            food, acts = shortlist(rows, pace, len(dates))
            first_note = last_note = ""
            if i > 0:
                km = haversine_km(dests[i - 1], d)
                first_note = (f"arrival day after a ~{int(km / 55 * 60) + 30}-minute drive from {dests[i - 1].get('name')}: start around "
                              f"{(pace_start_minutes + int(km / 55 * 60) + 45) // 60:02d}:00" if km < 150
                              else f"arrival morning after an overnight leg from {dests[i - 1].get('name')} (arrive ~08:00): keep it light")
            if i + 1 < len(dests) and dest_dates[i + 1]:
                last_note = f"the traveller moves on to {dests[i + 1].get('name')} afterwards: keep the evening free after dinner"
            system = build_itinerary_prompt(req, locale, d.get("name") or req.get("cityName") or "", len(dates),
                                            day_numbers, weekdays, cfg, first_note, last_note)
            payload = {
                "destination": d.get("name") or req.get("cityName") or "",
                "days": [{"dayNumber": n, "date": x.isoformat(), "weekday": w} for n, x, w in zip(day_numbers, dates, weekdays)],
                "food": [compact_for_plan(r, weekdays) for r in food],
                "activities": [compact_for_plan(r, weekdays) for r in acts],
            }
            out = await model.json(system, payload, max_tokens=12000, temperature=0.5, cache=False)
            logger.info("plan %s: food=%d acts=%d days=%d llm=%s", d.get("name"), len(food), len(acts), len(dates), out is not None)
            return out

        answers = await asyncio.gather(*(ask(i) for i in range(len(dests))))

        await java.event("BUILDING_ITINERARY", 78, "ai_trip.building_itinerary")
        items: list[dict] = []
        descriptions: list[str] = []
        used: set[str] = set()
        first_day_delay = 0
        for i, (d, rows, ai) in enumerate(zip(dests, s["inventories"], answers)):
            dates = dest_dates[i]
            if not dates:
                logger.warning("destination %s lost all its days to overlap resolution; skipping", d.get("name"))
                continue
            day_numbers = [_day_number(first_start, x) for x in dates]
            center = _destination_center(d, rows)
            first_start_min = pace_start_minutes + first_day_delay if first_day_delay else None
            try:
                dest_items, desc, notes = materialize(ai, rows, day_numbers, dates, pace, locale, used,
                                                      max_chars=max_chars, first_day_start=first_start_min)
                if desc:
                    descriptions.append(desc)
            except ValueError as e:
                logger.warning("AI itinerary unusable for %s (%s); using code-only fallback", d.get("name"), e)
                dest_items, _, notes = schedule_destination(None, rows, day_numbers, dates, pace, center, locale, used,
                                                            first_day_delay=first_day_delay)
                # No LLM wording in the fallback: use the place's own blurb, else a neutral line.
                blurbs = {row_key(r): (r.get("description") or "") for r in rows}
                for it in dest_items:
                    if it["type"] == "TRANSPORT":
                        continue
                    text = it.get("description") or blurbs.get(it.get("_placeKey")) or FALLBACK_NOTE[locale]
                    it["description"] = text[:max_chars] or None
            for n in notes:
                logger.info("itinerary %s: %s", d.get("name"), n)
            dest_items = append_missing_social_items(
                dest_items,
                rows,
                day_numbers,
                dates,
                pace,
                locale,
                max_chars=max_chars,
            )
            items += dest_items
            first_day_delay = 0
            nxt = next((j for j in range(i + 1, len(dests)) if dest_dates[j]), None)
            if nxt is not None:
                km = haversine_km(d, dests[nxt])
                arrive_day = _day_number(first_start, dest_dates[nxt][0])
                last_day = day_numbers[-1]
                last_end = max((int(it["endTime"][:2]) * 60 + int(it["endTime"][3:5]) for it in dest_items
                                if it["dayNumber"] == last_day and not it.get("endDayNumber")), default=20 * 60)
                if km < 150:
                    minutes = int(km / 55 * 60) + 30
                    items.append(intercity_transport(d, dests[nxt], arrive_day, pace_start_minutes, locale,
                                                     same_day=True, minutes=minutes))
                    first_day_delay = minutes + 15
                else:
                    items.append(intercity_transport(d, dests[nxt], last_day, last_end, locale, arrival_day=arrive_day))
        description = " ".join(dict.fromkeys(descriptions)).strip()[:500] or FALLBACK_DESC[locale]
        return {"items": items, "description": description}

    async def verify(s: State):
        await java.event("VERIFYING", 88, "ai_trip.verifying")
        violations = check(s["items"], pace, total_days)   # log only — the AI's plan is not rewritten here
        for v in violations:
            logger.warning("verify: %s", v)
        return {"violations": violations}

    async def commit(s: State):
        await java.event("SAVING", 94, "ai_trip.saving")
        items = strip_internal(finalize(s["items"], total_days))
        if not any(i["type"] != "TRANSPORT" for i in items):
            raise RuntimeError("No places available for this destination")
        await java.commit(items, s["description"])
        return s

    graph = StateGraph(State)
    nodes = [("validate", validate), ("retrieve", retrieve), ("web_research", web_research),
             ("curate_web", curate_web), ("resolve_places", resolve_places), ("plan", plan),
             ("verify", verify), ("commit", commit)]
    for name, fn in nodes:
        graph.add_node(name, fn)
    graph.add_edge(START, nodes[0][0])
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        graph.add_edge(a, b)
    graph.add_edge(nodes[-1][0], END)
    return graph.compile()


async def run_job(job: dict[str, Any], token: str) -> None:
    java = JavaClient(job, token)
    try:
        graph = await build_workflow(job, token)
        await graph.ainvoke({"job": job})
    except Exception as exc:
        logger.exception("job %s failed", job.get("jobId"))
        try:
            await java.event("FAILED", 0, "ai_trip.failed", "FAILED", str(exc)[:1000])
        except Exception as report_err:
            logger.error("could not report failure: %s", report_err)
        raise
