"""Pengoo's agent loop as a LangGraph graph: agent ⇄ tools → finish.

Each model round may call tools; each tool call is shown live in the chat as a step. The loop is
bounded (`maxToolCalls` per turn, then the model must answer), and the whole turn has a time
limit. Whatever happens, Java hears back exactly once: a reply, or a failure.

A round that neither calls a tool nor returns the final JSON is usually the model announcing
work ("Mình xem thời tiết nhé") and stopping. That text is never posted as the answer: the model
is sent back once to either call the tool or answer properly.

2026-10-25 (prod call-log audit): tool calls are bounded per tool and per answer, an exact repeat
of an earlier call is answered with a note instead of running again, an unknown tool name costs
nothing, and `get_place_details` / `travel_time` take a list so one model round fetches what
used to take six (the model asked for one place or one hop per round).

2026-10-28: search_places may return map rows (`map:` ids, `source: "map"`) for names the catalogue
lacks. The model only ever writes such an id; the reply rebuilds each map place (card, proposal,
plan stop) from the row this turn's tool returned, so a coordinate the model wrote never reaches
Java. A map: id this turn did not return is dropped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import Counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import ai_log
from . import images, read_url, web_search
from .chat_model import ChatModel
from .gateway import JavaGateway, TurnClosed
from .prompts import SYSTEM, fold, history_messages, question_message
from .tools import MAP_PREFIX, collect_names, map_search_step, step_for, tools_for

logger = logging.getLogger(__name__)

# Used only when Java does not send limits.timeoutSeconds (an older backend).
TURN_TIMEOUT_SECONDS = 120
MAX_TOOL_RESULT_CHARS = 6000
# The whole destination catalogue with its guide text is read in one go; cutting it would hide
# destinations from the model.
TOOL_RESULT_CHARS = {"explore_destinations": 40000, "read_url": 7000,
                     # A trip plan looks up the stay, the sights and food in one call (`searches`).
                     "search_places": 16000}
MAX_WEB_SEARCHES = 1
# Pages read per answer: a list article found by the search, or a link the user sent.
MAX_PAGE_READS = 2
BUDGET_USED = "Tool budget used up: write the final JSON answer now with what you have."
# Rounds the model may still ask for tools after the budget is spent before the turn ends anyway.
MAX_OVER_BUDGET_ROUNDS = 2
MAX_ACTIONS = 3
MAX_CARDS = 6
MAX_DESTINATIONS = 4
# Card type -> the id pools (see tools.collect_names) an id must come from.
CARD_POOLS = {"TRIP": ("public_trip", "trip"), "EVENT": ("event",), "MEETUP": ("meetup",),
              "CHECKIN": ("checkin",), "FOOD": ("dish",)}
MAX_MEMORY_CHARS = 1500
# A thread's title and what Pengoo keeps about the user across 1:1 chats (Java splits the profile
# into at most 12 lines of 80 characters).
MAX_TITLE_CHARS = 80
MAX_PROFILE_CHARS = 1000
# A map place (`mapPlace`, MapPlaceRef in Java) as Java accepts it.
MAP_PROVIDERS = ("GOONG", "MAPBOX")
MAP_PLACE_ID_CHARS = 255
MAP_NAME_CHARS = 200
MAP_ADDRESS_CHARS = 300
# The proposals that may take a map place instead of a catalogue one.
MAP_PLACE_ACTIONS = ("ADD", "REPLACE")
# Uses of one tool in one answer (items for the batch tools). Past the limit the call is refused
# with a short note and does not count toward the budget: the log showed get_weather asked for a
# price question and get_place_community for nine search rows one after another.
TOOL_LIMITS = {"get_weather": 1, "get_place_community": 2, "get_place_details": 8, "travel_time": 12,
               "search_places": 8, "explore_destinations": 1, "get_destination_guide": 2, "get_local_dishes": 1,
               "search_events": 2, "search_meetups": 1, "search_public_trips": 2, "get_budget": 1,
               "get_trip_checkins": 1, "rank_trip_places": 1, "list_my_trips": 1, "get_trip": 2}
TOOL_LIMIT_DEFAULT = 3
# Batch tools: (list argument, per-item argument built from it, cap). One model round, the Java
# calls run at once, one tool call against the budget.
BATCHES = {"get_place_details": ("place_ids", "place_id", 6), "travel_time": ("legs", None, 10),
           "search_places": ("searches", None, 6)}
# What one entry of search_places' `searches` may set; the call's own arguments are shared by all.
_SEARCH_KEYS = ("query", "types", "kind", "activity_id", "day", "city", "sort", "radius_km", "limit")
REPEATED_CALL = "Same call already answered above in this conversation: use that result."
NUDGE = ("Your last message was neither a tool call nor the final JSON, so the user would see nothing useful. "
         "If you meant to check something, call that tool now. Otherwise reply with the final JSON answer.")
# A short plain-text round, or one that says what Pengoo is about to do, is an announcement and
# gets the nudge. A longer plain-text round that reads as an answer is the answer: sending it back
# only made the model repeat it as JSON (one more model round, and the text showed twice). A
# trailing "nhé/nha" no longer counts: ordinary Vietnamese answers end with it.
ANNOUNCEMENT_MAX_CHARS = 160
_ANNOUNCEMENT = re.compile(
    r"^\W*(để mình|mình (sẽ|xem|tìm|kiểm tra|tra)|đợi mình|chờ mình|let me|i'll|i will|i am going to|i'm going to|one moment|hold on)\b",
    re.I)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
# The history annotation (prompts.history_messages) that a model sometimes copies into its answer.
_CARD_NOTE = re.compile(r"\s*\[(?:place cards shown|place): [^\]]*\]")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
PROPOSAL_ACTIONS = ("ADD", "MOVE", "UPDATE_TIME", "REMOVE", "REPLACE",
                    "ADD_DAYS", "PREPEND_DAYS", "REMOVE_DAYS", "SWAP_DAYS", "SHIFT_DATES",
                    "EDIT_TRIP", "EDIT_ACTIVITY", "NEW_ACTIVITY", "ADD_EXPENSE", "EDIT_EXPENSE",
                    "REMOVE_EXPENSE", "ADD_NOTE", "ADD_GUEST", "NEW_TRIP", "PLAN_DAYS")
MAX_PROPOSALS = 10
# A planned trip (NEW_TRIP) or the days of one (PLAN_DAYS); Java checks every value again.
PLAN_ACTIONS = ("NEW_TRIP", "PLAN_DAYS")
MAX_PLAN_DAYS = 30
MAX_PLAN_ITEMS = 14
_PLAN_TEXT = {"name": 255, "destination": 120, "description": 2000, "currency": 8,
              "startingPointName": 255, "startingPointAddress": 500}
_ITEM_TEXT = {"placeId": 64, "name": 255, "category": 40, "notes": 2000, "description": 2000, "address": 500,
              "endAddress": 500, "transportMode": 16, "costCurrency": 8}
# Questions asked before planning, each with tap answers.
MAX_QUESTIONS = 3
MAX_QUESTION_OPTIONS = 6
# The field names each trip-information action may set; Java checks every value again.
_CHANGE_FIELDS = {
    "EDIT_TRIP": ("name", "description", "budget", "currency", "visibility",
                  "startingPointName", "startingPointAddress", "startingPointTime"),
    "EDIT_ACTIVITY": ("name", "address", "endAddress", "notes", "description", "category",
                      "transportMode", "cost", "costCurrency", "stay"),
    "NEW_ACTIVITY": ("name", "address", "endAddress", "notes", "description", "category",
                     "transportMode", "cost", "costCurrency", "stay"),
    "ADD_EXPENSE": ("amount", "currency", "category", "description", "date", "paidBy", "splitWith"),
    "EDIT_EXPENSE": ("amount", "category", "description", "date", "paidBy"),
    "ADD_NOTE": ("content",),
    "ADD_GUEST": ("guestName",),
}
# Changes to the days themselves go first (Java keeps one and the adds it needs).
_DAY_CHANGES = ("PREPEND_DAYS", "REMOVE_DAYS", "SWAP_DAYS", "SHIFT_DATES", "ADD_DAYS")


class State(TypedDict, total=False):
    messages: list[dict]
    tool_calls: int
    web_searches: int
    pages_read: int
    budget_told: bool
    over_budget_rounds: int
    nudged: bool
    thin: bool
    web_pushed: bool
    final: dict
    usage: dict
    stop_reason: str


# Any of these makes a JSON object the reply's final format rather than data quoted in prose.
_OUTPUT_KEYS = {"answer", "places", "proposals", "destinations", "cards", "actions", "suggestions", "memory",
                "questions"}
_TRAILING_FENCE = re.compile(r"```(?:json)?\s*$")


def parse_json(content: str) -> dict | None:
    """The model's closing JSON object, or None when the content is not one.

    Also the common slip of writing the answer as prose and only the cards, buttons and
    follow-ups as JSON after it: the prose becomes the answer instead of the whole thing,
    JSON included, reaching the user as text."""
    text = _FENCE.sub("", content or "").strip()
    whole = _object(text)
    if whole is not None:
        return whole
    trailing = _prose_then_json(text)
    if trailing is not None:
        return trailing
    match = _JSON_OBJECT.search(text)
    return _object(match.group(0)) if match else None


def _object(candidate: str) -> dict | None:
    try:
        value = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) and _OUTPUT_KEYS & value.keys() else None


def _prose_then_json(text: str) -> dict | None:
    """Prose followed by one JSON object that runs to the end of the text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start > 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            value, end = None, -1
        if isinstance(value, dict) and _OUTPUT_KEYS & value.keys() and not _FENCE.sub("", text[end:]).strip():
            prose = _TRAILING_FENCE.sub("", text[:start]).strip()
            if prose and not str(value.get("answer") or "").strip():
                value["answer"] = prose
            return value
        start = text.find("{", start + 1)
    return None


def parse_final(content: str) -> dict:
    """The model's closing JSON; plain text (a model that ignored the format) becomes the answer.
    JSON cut off by the output limit keeps its answer text: prod 2026-10-28 posted a whole 3-day
    plan as raw JSON after thinking and the plan used up the 5000-token limit."""
    text = _FENCE.sub("", content or "").strip()
    return parse_json(text) or _salvage_answer(text) or {"answer": text}


_ANSWER_KEY = re.compile(r'^\{\s*"answer"\s*:\s*')


def _salvage_answer(text: str) -> dict | None:
    """The complete "answer" string at the head of a JSON object whose tail is missing."""
    match = _ANSWER_KEY.match(text)
    if not match:
        return None
    try:
        # strict=False: a raw line break inside the string is a common slip, not a reason to lose it.
        answer, _ = json.JSONDecoder(strict=False).raw_decode(text, match.end())
    except ValueError:
        return None
    return {"answer": answer} if isinstance(answer, str) and answer.strip() else None


# Words that say what a place is rather than which one it is. A name reduced to nothing but these
# ("Pho", "Cafe") is too generic to recognise in an answer.
_GENERIC_NAME_WORDS = {
    "lake", "ho", "quan", "nha", "hang", "restaurant", "cafe", "coffee", "ca", "phe", "the", "a", "an", "and", "of",
    "bar", "pub", "shop", "store", "market", "cho", "park", "cong", "vien", "temple", "chua", "den", "museum",
    "bao", "tang", "hotel", "homestay", "khach", "san", "resort", "beach", "bien", "bai", "street", "pho",
}


def _distinctive(name: str) -> str | None:
    """The part of a place name that identifies it, accent-free: "Trúc Bạch Lake" -> "truc bach".
    None when what is left is too short to tell one place from another."""
    tokens = fold(name).split()
    while tokens and tokens[0] in _GENERIC_NAME_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _GENERIC_NAME_WORDS:
        tokens.pop()
    if not tokens or (len(tokens) == 1 and len(tokens[0]) < 5):
        return None
    return " ".join(tokens)


def mentioned_places(answer: str, candidates: dict[str, str]) -> list[str]:
    """Ids of places (id -> name) that the answer names, in the order they appear."""
    text = f" {fold(answer)} "
    found: list[tuple[int, str]] = []
    for place_id, name in candidates.items():
        key = _distinctive(name)
        if key:
            at = text.find(f" {key} ")
            if at >= 0:
                found.append((at, place_id))
    return [place_id for _, place_id in sorted(found)]


def attach_mentioned_places(final: dict, candidates: dict[str, str], limit: int = 6) -> dict:
    """Cards for places the answer names but the model left out of "places".

    The model sometimes answers in plain text (or JSON without "places") right after a search, so
    a place it recommends by name shows no card. Only places returned by this turn's tools are
    candidates (catalogue rows and map rows, which clean_reply turns into map places): they are
    real, and Java checks them again."""
    places = [p for p in (final.get("places") or []) if isinstance(p, dict)]
    have = {str(p.get("placeId") or p.get("id") or "") for p in places}
    for place_id in mentioned_places(str(final.get("answer") or ""), candidates):
        if len(places) >= limit:
            break
        if place_id not in have:
            places.append({"placeId": place_id})
            have.add(place_id)
    return {**final, "places": places}


def looks_like_announcement(content: str, acted: bool = False) -> bool:
    """Plain text that is not yet an answer: phrased as "let me check", or short before any tool ran.
    After a tool call a short text is usually the answer itself, so only the phrasing counts."""
    text = (content or "").strip()
    return bool(_ANNOUNCEMENT.search(text)) or (not acted and len(text) < ANNOUNCEMENT_MAX_CHARS)


def clean_actions(raw: Any, ids: dict[str, set[str]]) -> list[dict]:
    """Buttons: open actions only for ids the tools or the chat produced; Java checks access again."""
    needs = {"OPEN_TRIP": ("tripId", "trip"), "OPEN_EVENT": ("eventId", "event"),
             "OPEN_MEETUP": ("meetupId", "meetup")}
    actions: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict) or len(actions) >= MAX_ACTIONS:
            continue
        action = str(item.get("action") or "").upper()
        label = str(item.get("label") or "").strip()[:80]
        if not label:
            continue
        if action == "CREATE_TRIP":
            out = {"action": action, "label": label}
            if item.get("city"):
                out["city"] = str(item["city"]).strip()[:80]
            start, end = str(item.get("startDate") or ""), str(item.get("endDate") or "")
            if _DATE.fullmatch(start) and _DATE.fullmatch(end):
                out["startDate"], out["endDate"] = start, end
            actions.append(out)
        elif action in needs:
            key, kind = needs[action]
            target = str(item.get(key) or "")
            if target in ids.get(kind, set()):
                actions.append({"action": action, "label": label, key: target})
    return actions


def clean_cards(raw: Any, ids: dict[str, set[str]]) -> list[dict]:
    """Cards of public items: only ids this turn's tools returned; Java checks each is shareable."""
    cards: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in raw or []:
        if not isinstance(item, dict) or len(cards) >= MAX_CARDS:
            continue
        kind = str(item.get("type") or "").upper()
        target = str(item.get("id") or "")
        pools = CARD_POOLS.get(kind)
        if not pools or (kind, target) in seen or not any(target in ids.get(pool, set()) for pool in pools):
            continue
        card = {"type": kind, "id": target}
        if kind == "FOOD":
            slug = str(item.get("citySlug") or "").strip()
            if not slug:
                continue
            card["citySlug"] = slug[:64]
        seen.add((kind, target))
        cards.append(card)
    return cards


def clean_destinations(raw: Any, ids: dict[str, set[str]]) -> list[dict]:
    """Destination cards: only areas a tool returned in this turn; dates only as a valid pair."""
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict) or len(out) >= MAX_DESTINATIONS:
            continue
        target = str(item.get("id") or item.get("destinationId") or "")
        if target not in ids.get("destination", set()) or any(d["id"] == target for d in out):
            continue
        card = {"id": target}
        note = str(item.get("note") or "").strip()[:300]
        if note:
            card["note"] = note
        start, end = str(item.get("startDate") or ""), str(item.get("endDate") or "")
        if _DATE.fullmatch(start) and _DATE.fullmatch(end):
            card["startDate"], card["endDate"] = start, end
        out.append(card)
    return out


def balance_bold(text: str) -> str:
    """Drops a stray closing "**": with an odd count the app shows the asterisks as text."""
    if text.count("**") % 2 == 0:
        return text
    cut = text.rfind("**")
    return (text[:cut] + text[cut + 2:]).rstrip()


def clean_changes(action: str, raw) -> dict:
    """The fields a trip-information proposal sets: only the names its action takes, scalars (or
    a list of ids for `splitWith`), strings cut to 2000 characters."""
    fields = _CHANGE_FIELDS.get(action)
    if not fields or not isinstance(raw, dict):
        return {}
    out = {}
    for key in fields:
        value = raw.get(key)
        if value is None or value == "":
            continue
        if key == "splitWith":
            if isinstance(value, list):
                ids = [str(v)[:64] for v in value if isinstance(v, (str, int)) and str(v).strip()][:30]
                if ids:
                    out[key] = ids
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value.strip()[:2000]
    return out


def _hhmm(value: Any) -> str | None:
    """'6:30' or '06:30' as '06:30'; None when it is not a time of day."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not match:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    return f"{hours:02d}:{minutes:02d}" if hours < 24 and minutes < 60 else None


def _bounded_int(value: Any, low: int, high: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def _money(value: Any) -> str | None:
    """A price as text for Java to parse: plain numbers stay plain, '3.605.000' is left as written."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()[:32]
    return None


def _texts(source: dict, limits: dict[str, int]) -> dict:
    return {key: str(source[key]).strip()[:limit] for key, limit in limits.items()
            if isinstance(source.get(key), (str, int, float)) and not isinstance(source.get(key), bool)
            and str(source[key]).strip()}


def _coordinate(value: Any, bound: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and -bound <= number <= bound else None


def map_place(row: Any) -> dict | None:
    """The `mapPlace` of a map row search_places returned (id "map:<PROVIDER>:<place id>"), inside
    the bounds Java accepts, or None when the row cannot make one. Java rejects the whole reply
    over one bad value, so a row that does not fit is left out rather than sent."""
    if not isinstance(row, dict):
        return None
    prefix, provider, place_id = (str(row.get("id") or "").split(":", 2) + ["", ""])[:3]
    name = str(row.get("name") or "").strip()[:MAP_NAME_CHARS]
    lat, lng = _coordinate(row.get("lat"), 90), _coordinate(row.get("lng"), 180)
    if (prefix + ":" != MAP_PREFIX or provider not in MAP_PROVIDERS or not place_id
            or len(place_id) > MAP_PLACE_ID_CHARS or not name or lat is None or lng is None):
        return None
    ref: dict[str, Any] = {"provider": provider, "providerPlaceId": place_id, "name": name}
    address = str(row.get("address") or "").strip()[:MAP_ADDRESS_CHARS]
    if address:
        ref["address"] = address
    ref["lat"], ref["lng"] = lat, lng
    return ref


def _map_ref(map_rows: dict[str, dict] | None, place_id: Any) -> dict | None:
    return map_place((map_rows or {}).get(str(place_id or "").strip()))


def clean_plan(raw: Any, map_rows: dict[str, dict] | None = None) -> dict | None:
    """The plan of a NEW_TRIP / PLAN_DAYS proposal, bounded to what Java accepts; None when no day
    has a stop with a name or a place. A stop on a map row of this turn (`map:` placeId) carries
    the row's position and provider id instead of a placeId."""
    if not isinstance(raw, dict):
        return None
    plan = _texts(raw, _PLAN_TEXT)
    start = str(raw.get("startDate") or "")
    if _DATE.fullmatch(start):
        plan["startDate"] = start
    count = _bounded_int(raw.get("dayCount"), 1, MAX_PLAN_DAYS)
    if count:
        plan["dayCount"] = count
    budget = _money(raw.get("budget"))
    if budget:
        plan["budget"] = budget
    departure = _hhmm(raw.get("departureTime"))
    if departure:
        plan["departureTime"] = departure
    days: list[dict] = []
    for day in raw.get("days") or []:
        if not isinstance(day, dict) or len(days) >= MAX_PLAN_DAYS:
            continue
        number = _bounded_int(day.get("dayNumber"), 1, 400)
        if number is None:
            continue
        items: list[dict] = []
        for item in day.get("items") or []:
            if not isinstance(item, dict) or len(items) >= MAX_PLAN_ITEMS:
                continue
            row = _texts(item, _ITEM_TEXT)
            # Read from the item, not the row: a provider id is longer than a placeId may be.
            place_ref = str(item.get("placeId") or "").strip()
            if place_ref.startswith(MAP_PREFIX):
                row.pop("placeId", None)
                ref = _map_ref(map_rows, place_ref)
                if ref:
                    row.setdefault("name", ref["name"])
                    if ref.get("address"):
                        row.setdefault("address", ref["address"])
                    row.update(lat=ref["lat"], lng=ref["lng"], mapProvider=ref["provider"],
                               mapPlaceId=ref["providerPlaceId"])
            if not row.get("name") and not row.get("placeId"):
                continue
            for key in ("startTime", "endTime"):
                at = _hhmm(item.get(key))
                if at:
                    row[key] = at
            cost = _money(item.get("cost"))
            if cost:
                row["cost"] = cost
            if "transportMode" in row:
                row["transportMode"] = row["transportMode"].upper()
            if item.get("stay") is True:
                row["stay"] = True
                until = _bounded_int(item.get("untilDay"), 1, 400)
                if until:
                    row["untilDay"] = until
            items.append(row)
        if items:
            entry: dict[str, Any] = {"dayNumber": number, "items": items}
            title = str(day.get("title") or "").strip()[:120]
            if title:
                entry["title"] = title
            days.append(entry)
    if not days:
        return None
    plan["days"] = days
    return plan


def clean_questions(raw: Any) -> list[dict]:
    """Questions with tap answers: a question needs two options to be worth tapping."""
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict) or len(out) >= MAX_QUESTIONS:
            continue
        text = str(item.get("text") or item.get("question") or "").strip()[:200]
        options = [str(option).strip()[:80] for option in item.get("options") or []
                   if isinstance(option, (str, int, float)) and not isinstance(option, bool) and str(option).strip()]
        options = list(dict.fromkeys(options))[:MAX_QUESTION_OPTIONS]
        if text and len(options) >= 2:
            question: dict[str, Any] = {"text": text, "options": options}
            if item.get("multiple") is True:
                question["multiple"] = True
            out.append(question)
    return out


def clean_reply(final: dict, allowed_places: set[str], usage: dict,
                ids: dict[str, set[str]] | None = None, map_rows: dict[str, dict] | None = None) -> dict:
    """What Java receives: bounded lists, places only from this turn's tool results, earlier
    cards or the trip; buttons only for ids seen in this turn. A `map:` id becomes a `mapPlace`
    rebuilt from this turn's map row (`map_rows`), in a card, an ADD/REPLACE or a plan stop."""
    places, seen = [], set()
    for item in final.get("places") or []:
        if not isinstance(item, dict):
            continue
        place_id = str(item.get("placeId") or item.get("id") or "").strip()
        if place_id in seen:
            continue
        note = str(item.get("note") or "").strip()[:300] or None
        if place_id.startswith(MAP_PREFIX):
            ref = _map_ref(map_rows, place_id)
            if ref:
                seen.add(place_id)
                places.append({"mapPlace": ref, "note": note})
        elif place_id in allowed_places:
            seen.add(place_id)
            places.append({"placeId": place_id, "note": note})
    proposals = []
    for item in final.get("proposals") or []:
        if isinstance(item, dict) and item.get("action") in PROPOSAL_ACTIONS:
            proposal = {key: item.get(key) for key in ("action", "tripId", "dayNumber", "toDayNumber", "days",
                                                       "startDate", "startTime", "endTime", "placeId", "activityId",
                                                       "expenseId", "category", "summary")
                        if item.get(key) not in (None, "")}
            if str(proposal.get("placeId") or "").strip().startswith(MAP_PREFIX):
                ref = _map_ref(map_rows, proposal.pop("placeId"))
                if ref and item["action"] in MAP_PLACE_ACTIONS:
                    proposal["mapPlace"] = ref
            changes = clean_changes(item.get("action"), item.get("changes"))
            if changes:
                proposal["changes"] = changes
            if item.get("action") in PLAN_ACTIONS:
                plan = clean_plan(item.get("plan"), map_rows)
                if plan is None:
                    continue
                proposal["plan"] = plan
            for key in ("startTime", "endTime"):
                if key in proposal and not _TIME.fullmatch(str(proposal[key])):
                    proposal.pop(key)
            if "startDate" in proposal and not _DATE.fullmatch(str(proposal["startDate"])):
                proposal.pop("startDate")
            for key in ("dayNumber", "toDayNumber", "days"):
                if key in proposal:
                    try:
                        proposal[key] = int(proposal[key])
                    except (TypeError, ValueError):
                        proposal.pop(key)
            if "summary" in proposal:
                proposal["summary"] = str(proposal["summary"])[:300]
            proposals.append(proposal)
    # A day change must survive the cap: the ADDs for new days are dropped without their extension.
    proposals.sort(key=lambda p: p["action"] not in _DAY_CHANGES)
    suggestions = [str(s).strip()[:120] for s in (final.get("suggestions") or []) if str(s).strip()][:4]
    answer = balance_bold(_CARD_NOTE.sub("", str(final.get("answer") or "")).strip()[:8000])
    reply = {"markdown": answer, "places": places[:8], "proposals": proposals[:MAX_PROPOSALS], "suggestions": suggestions,
             "actions": clean_actions(final.get("actions"), ids or {}),
             "cards": clean_cards(final.get("cards"), ids or {}),
             "destinations": clean_destinations(final.get("destinations"), ids or {}),
             "questions": clean_questions(final.get("questions")),
             "inputTokens": usage.get("input", 0), "outputTokens": usage.get("output", 0)}
    memory = final.get("memory")
    if isinstance(memory, str) and memory.strip():
        reply["memory"] = memory.strip()[:MAX_MEMORY_CHARS]
    # Java keeps a title only for a 1:1 thread that has none, and a profile only from a 1:1 chat.
    for key, limit in (("title", MAX_TITLE_CHARS), ("profile", MAX_PROFILE_CHARS)):
        value = final.get(key)
        if isinstance(value, str) and value.strip():
            reply[key] = value.strip()[:limit]
    return reply


# Fewer rows than this that really match the ask, and the catalogue is thin for it.
THIN_CATALOGUE_ROWS = 3
# "Near a stop of the trip" means walking distance or a very short ride.
NEAR_STOP_KM = 1.5
# PlaceSubType.Family.FOOD in Java; every other type is a sight (markets, beaches, viewpoints…).
FOOD_TYPES = {"RESTAURANT", "STREET_FOOD", "CAFE", "TEA", "BAR", "BAKERY", "DESSERT", "SEAFOOD", "VEGETARIAN",
              "VEGAN", "HALAL"}
# Not "the app's map": map rows (`source: "map"`) are a different source the model must tell apart.
THIN_CATALOGUE_NOTE = ("The app's catalogue has little for this ask: {reasons}. Call "
                       "web_search now for what the user asked (the kind, dish or fact plus the area and the city, "
                       "in the user's language), then answer with the best of both: rows here that truly fit as "
                       "place cards, web finds named in the text with their source link. Leave out rows that do "
                       "not fit; do not list them as a fallback.")
WEB_PUSH = ("Your search came back thin for this ask and web_search is still unused. Do not answer with places that "
            "do not fit or with 'ask the locals': call web_search now for exactly what the user asked, then write "
            "the final JSON with what fits.")


def flag_thin_catalogue(result: Any, web_left: bool, args: dict | None = None) -> Any:
    """Marks a search_places result the catalogue cannot really answer, so the model searches the web
    instead of stretching weak rows or naming venues from memory.

    Thin when any of these holds: the radius had to be widened; fewer than three rows match the
    asked name words (the tool pads a short list with same-kind places whose names do not contain
    them; a map row counts, the map found it by those words); fewer than three rows at all; an
    asked type has no row (asked SEAFOOD and MARKET, got only seafood restaurants: the market part
    is unanswered); the search was around a stop of the
    trip and fewer than three rows are within walking distance of it. Only while a web search is
    still available; otherwise the result is left as it came."""
    if not web_left or not isinstance(result, dict) or not isinstance(result.get("places"), list):
        return result
    rows = [row for row in result["places"] if isinstance(row, dict)]
    searched = result.get("searched") or {}
    words = fold(str(searched.get("nameWords") or "")).split()
    reasons: list[str] = []
    if any("radius" in str(step) for step in result.get("relaxed") or []):
        reasons.append("the search had to be widened")
    matching = [row for row in rows if row.get("source") == "map" or (
                row.get("nameMatch") is not False and set(words) <= set(fold(str(row.get("name") or "")).split()))]
    if len(matching) < THIN_CATALOGUE_ROWS:
        reasons.append(f"only {len(matching)} place(s) match '{searched.get('nameWords')}'" if words
                       else f"only {len(matching)} place(s) found")
    # Sibling types are often asked together as alternatives (CAFE or TEA); a type of the other
    # family (a MARKET next to SEAFOOD) is a second thing the user asked for.
    found_types = {str(t) for row in rows for t in row.get("types") or []}
    found_families = {t in FOOD_TYPES for t in found_types}
    missing = [str(t) for t in searched.get("types") or []
               if str(t) not in found_types and (str(t) in FOOD_TYPES) not in found_families]
    if missing:
        reasons.append("nothing of type " + ", ".join(missing))
    if (args or {}).get("activity_id"):
        close = [row for row in rows if isinstance(row.get("distanceKm"), (int, float))
                 and row["distanceKm"] <= NEAR_STOP_KM]
        if len(close) < THIN_CATALOGUE_ROWS:
            reasons.append(f"only {len(close)} place(s) within {NEAR_STOP_KM} km of {result.get('near') or 'that stop'}")
    if not reasons:
        return result
    return {**result, "coverage": "thin", "note": THIN_CATALOGUE_NOTE.format(reasons="; ".join(reasons))}


def web_offered(turn: dict) -> bool:
    """Whether this turn may reach the internet (web_search and read_url)."""
    return bool((turn.get("limits") or {}).get("webSearch")) and web_search.configured()


def build_graph(turn: dict, java: JavaGateway, model: ChatModel):
    limits = turn.get("limits") or {}
    max_tool_calls = int(limits.get("maxToolCalls") or 6)
    max_output = int(limits.get("maxOutputTokens") or 1800)
    web_allowed = web_offered(turn)
    tools = tools_for(str(turn.get("conversationType") or "TRIP"), web_search=web_allowed)
    # One provider cache per tool list: the trip chat and the 1:1 chat send different tools, and a
    # changed tool list is a different prefix.
    cache_key = f"pengoo-v4-{str(turn.get('conversationType') or 'TRIP').lower()}{'-web' if web_allowed else ''}"
    names: dict[str, str] = {}
    # Map rows by their `map:` id: the map cards of earlier answers, then what search_places returns now.
    map_rows: dict[str, dict] = {}
    ids: dict[str, set[str]] = {"place": set(), "event": set(), "meetup": set(), "trip": set(),
                                "public_trip": set(), "checkin": set(), "dish": set(), "destination": set()}
    allowed_places = ids["place"]
    for row in turn.get("history") or []:
        for place in row.get("places") or []:
            if isinstance(place, dict) and place.get("id"):
                allowed_places.add(str(place["id"]))
                if place.get("name"):
                    names[str(place["id"])] = str(place["name"])
        # Map cards of earlier answers come whole (Java has no row to look them up by), so "add the
        # second one to day 2" can still become a proposal.
        for place in row.get("mapPlaces") or []:
            if isinstance(place, dict) and map_place(place):
                map_rows[str(place["id"])] = place
    # Cards already shown are not attached again just because the answer names them.
    shown_map_ids = set(map_rows)
    trip = turn.get("trip") or {}
    for day in trip.get("days") or []:
        for activity in day.get("activities") or []:
            if activity.get("id") and activity.get("name"):
                names[str(activity["id"])] = str(activity["name"])
            if activity.get("placeId"):
                allowed_places.add(str(activity["placeId"]))
    if trip.get("id"):
        ids["trip"].add(str(trip["id"]))
        if trip.get("name"):
            names[str(trip["id"])] = str(trip["name"])
    # Places already in the chat or the trip are not offered again as cards by name; only what
    # this turn's tools return is.
    known_places = set(allowed_places)

    async def agent(state: State) -> State:
        if ai_log.credit_exhausted():
            # The asker ran out of AI credits on the last call: no further model call. Java posts
            # what was written so far with a "credits ran out" notice.
            return {"final": partial_answer(state["messages"]), "stop_reason": ai_log.CREDIT_EXHAUSTED}
        budget_left = max_tool_calls - state.get("tool_calls", 0)
        history = list(state["messages"])
        if budget_left <= 0 and not state.get("budget_told"):
            # Kept in the history, not added for one request and dropped: Claude binds each thinking
            # block to the exact conversation and tool list it came from, so a message that comes
            # and goes, or a tool list that shrinks to nothing, gets the next request refused (400
            # "the block is bound to a different conversation"). The tools stay offered; run_tools
            # answers any call past the budget with an error instead of running it.
            history.append({"role": "system", "content": BUDGET_USED})
        reply = await model.complete(history, tools, max_tokens=max_output,
                                     operation="ASSISTANT_TURN", cache_key=cache_key)
        if reply is None:
            raise RuntimeError("model call failed")
        usage = dict(state.get("usage") or {"input": 0, "output": 0})
        usage["input"] += reply["_usage"]["input"]
        usage["output"] += reply["_usage"]["output"]
        calls = reply.get("tool_calls") or []
        content = (reply.get("content") or "").strip()
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reply.get("_anthropic_content"):
            # Claude's blocks go back verbatim next round: its thinking must return with tool results.
            message["_anthropic_content"] = reply["_anthropic_content"]
        told = {"budget_told": True} if budget_left <= 0 else {}
        if calls and state.get("over_budget_rounds", 0) < MAX_OVER_BUDGET_ROUNDS:
            message["tool_calls"] = calls
            if content and not content.startswith("{"):
                await java.step("thought", {}, content[:300])
            over = state.get("over_budget_rounds", 0) + (1 if budget_left <= 0 else 0)
            return {"messages": history + [message], "usage": usage, "over_budget_rounds": over, **told}
        # Still calling tools after being told twice the budget is spent: stop here and answer with
        # what this reply and the earlier ones hold.
        final = parse_json(content)
        acted = state.get("tool_calls", 0) > 0
        if final is None and budget_left > 0 and not state.get("nudged") and looks_like_announcement(content, acted):
            # Announced work without doing it: one more round to act.
            if content:
                await java.step("thought", {}, content[:300])
            return {"messages": history + [message, {"role": "system", "content": NUDGE}],
                    "usage": usage, "nudged": True, **told}
        if (state.get("thin") and web_allowed and not state.get("web_searches") and not state.get("web_pushed")
                and budget_left > 0):
            # Prod 2026-10-25: asked about a morning market on the beach, the model got three seafood
            # restaurants 3-4 km away, ignored the thin flag, listed them and told the user to ask
            # the locals. One more round to search for what was actually asked.
            return {"messages": history + [message, {"role": "system", "content": WEB_PUSH}],
                    "usage": usage, "web_pushed": True, **told}
        final = final or parse_final(content)
        if not str(final.get("answer") or "").strip():
            # An empty JSON answer after a plain-text round: that text was the answer.
            earlier = [m.get("content") for m in state["messages"]
                       if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls")]
            if earlier:
                final["answer"] = _FENCE.sub("", earlier[-1]).strip()
        return {"messages": history + [message], "usage": usage, "final": final, **told}

    tool_names = {t["function"]["name"] for t in tools}
    # Results of this turn's Java calls by (tool, arguments): a repeated call is answered from here
    # without another request, and a batch reuses the items it already has.
    answered: dict[str, Any] = {}
    uses: Counter[str] = Counter()

    async def java_once(name: str, args: dict) -> Any:
        key = call_key(name, args)
        if key not in answered:
            result = await java.tool(name, args)
            collect_names(result, names, ids, map_rows)
            answered[key] = result
        return answered[key]

    async def post_map_step(searches: list[dict], results: list[Any]) -> None:
        """'Found on the map' in the thinking view, after a search the map added places to."""
        params = map_search_step(searches, results)
        if params:
            await java.step("map_search", params)

    async def run_tools(state: State) -> State:
        last = state["messages"][-1]
        results: list[dict] = []
        used = state.get("tool_calls", 0)
        searched = state.get("web_searches", 0)
        pages = state.get("pages_read", 0)
        thin = bool(state.get("thin"))
        seen_this_round: set[str] = set()
        for call in last.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or ""
            try:
                args = json.loads(function.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            def answer(payload: Any, call=call, name=name) -> None:
                results.append({"role": "tool", "tool_call_id": call.get("id"),
                                "content": json.dumps(payload, ensure_ascii=False, default=str)
                                [:TOOL_RESULT_CHARS.get(name, MAX_TOOL_RESULT_CHARS)]})

            if name not in tool_names:
                answer({"error": f"Unknown tool '{name}'. Use only the tools offered, or answer with what you have."})
                continue
            items = batch_items(name, args)
            if not items:
                answer({"error": "Missing arguments: give place_ids / legs (or place_id / from and to)."})
                continue
            key = call_key(name, args)
            if key in answered or key in seen_this_round:
                answer({"note": REPEATED_CALL})
                continue
            if used >= max_tool_calls:
                answer({"error": BUDGET_USED})
                continue
            limit = TOOL_LIMITS.get(name, TOOL_LIMIT_DEFAULT)
            if uses[name] + len(items) > limit:
                answer({"error": f"{name} was already used {uses[name]} time(s) in this answer (limit {limit}): "
                                 "answer with what you have."})
                continue
            seen_this_round.add(key)
            step_key, params = step_for(name, args, names)
            await java.step(step_key, params)
            if name == "web_search":
                if not web_allowed or searched >= MAX_WEB_SEARCHES:
                    result: Any = {"error": "Web search is not available for this answer; answer from what you have."}
                else:
                    searched += 1
                    result = await web_search.search(str(args.get("query") or ""))
            elif name == "read_url":
                if not web_allowed or pages >= MAX_PAGE_READS:
                    result = {"error": "No more pages can be read for this answer; answer from what you have."}
                else:
                    pages += 1
                    result = await read_url.read(str(args.get("url") or ""))
            elif name in BATCHES and BATCHES[name][0] in args:
                rows = await asyncio.gather(*(java_once(name, item) for item in items))
                if name == "search_places":
                    await post_map_step(items, rows)
                    # Each entry is judged on its own (prod 2026-10-28: a batch result was never
                    # flagged, so a thin "lẩu bò Ban Công" search got no web search at all).
                    web_left = web_allowed and searched < MAX_WEB_SEARCHES
                    rows = [flag_thin_catalogue(row, web_left, item) for item, row in zip(items, rows)]
                    thin = thin or any(isinstance(row, dict) and row.get("coverage") == "thin" for row in rows)
                    result = {"searches": [{"asked": {k: v for k, v in item.items() if k in _SEARCH_KEYS},
                                            **(row if isinstance(row, dict) else {"result": row})}
                                           for item, row in zip(items, rows)]}
                elif name == "travel_time":
                    result = {"legs": [{**item, **(row if isinstance(row, dict) else {"result": row})}
                                       for item, row in zip(items, rows)]}
                else:
                    result = {"places": list(rows)}
            else:
                result = await java_once(name, items[0])
                if name == "search_places":
                    await post_map_step(items, [result])
                    result = flag_thin_catalogue(result, web_allowed and searched < MAX_WEB_SEARCHES, args)
                    thin = thin or (isinstance(result, dict) and result.get("coverage") == "thin")
            uses[name] += len(items)
            used += 1
            answered.setdefault(key, result)
            answer(result)
        return {"messages": state["messages"] + results, "tool_calls": used, "web_searches": searched,
                "pages_read": pages, "thin": thin}

    async def finish(state: State) -> State:
        await java.step("writing_answer", {})
        found = {pid: names[pid] for pid in allowed_places - known_places if pid in names}
        found.update({map_id: str(row.get("name") or "") for map_id, row in map_rows.items()
                      if map_id not in shown_map_ids and map_place(row)})
        final = attach_mentioned_places(state.get("final") or {}, found)
        reply = clean_reply(final, allowed_places, state.get("usage") or {}, ids, map_rows)
        if state.get("stop_reason"):
            reply["stopReason"] = state["stop_reason"]
        await java.reply(reply)
        return {}

    def route(state: State) -> str:
        if state.get("final") is not None:
            return "finish"
        return "tools" if state["messages"][-1].get("tool_calls") else "agent"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finish": "finish", "agent": "agent"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finish", END)
    return graph.compile()


def partial_answer(messages: list[dict]) -> dict:
    """The answer when the turn stops for credit: the latest prose Pengoo wrote (a JSON reply or a
    bare tool round adds nothing readable), or an empty answer for Java's fixed notice."""
    for message in reversed(messages):
        content = (message.get("content") or "").strip() if message.get("role") == "assistant" else ""
        if content and not content.startswith("{"):
            return {"answer": _FENCE.sub("", content).strip()}
    return {"answer": ""}


def call_key(name: str, args: dict) -> str:
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def batch_items(name: str, args: dict) -> list[dict]:
    """The per-call argument sets one tool call stands for: several for a batch tool given its list
    argument, one otherwise. Empty when a batch tool got neither form."""
    if name not in BATCHES:
        return [args]
    list_key, item_key, cap = BATCHES[name]
    if name == "search_places":
        searches = [entry for entry in args.get(list_key) or [] if isinstance(entry, dict)]
        if not searches:
            return [args]
        shared = {key: value for key, value in args.items() if key in _SEARCH_KEYS}
        return [{**shared, **{key: value for key, value in entry.items() if key in _SEARCH_KEYS}}
                for entry in searches[:cap]]
    if name == "travel_time":
        legs = [leg for leg in args.get(list_key) or [] if isinstance(leg, dict) and leg.get("from") and leg.get("to")]
        if legs:
            return [{"from": str(leg["from"]), "to": str(leg["to"])} for leg in legs[:cap]]
        return [{"from": str(args["from"]), "to": str(args["to"])}] if args.get("from") and args.get("to") else []
    ids = [str(v) for v in args.get(list_key) or [] if v]
    if not ids and args.get(item_key):
        ids = [str(args[item_key])]
    return [{item_key: value} for value in dict.fromkeys(ids[:cap])]


def initial_messages(turn: dict, photos: list[str] | None = None, web: bool = False) -> list[dict]:
    """The static rules, the chat so far, then one user message with the per-turn context and the
    question: everything before the last message is a stable prefix for the provider cache.
    Photos sent with the question go in that last message, before its text. `web`: web search is
    offered in this turn (the context then says when the user asked for it)."""
    question: Any = question_message(turn, len(photos or []), web)
    if photos:
        question = [{"type": "image_url", "image_url": {"url": url}} for url in photos]             + [{"type": "text", "text": question}]
    return ([{"role": "system", "content": SYSTEM}] + history_messages(turn)
            + [{"role": "user", "content": question}])


async def run_turn(turn: dict, token: str, model: ChatModel | None = None) -> None:
    """Answers one turn; always ends with exactly one reply or failure sent to Java."""
    java = JavaGateway(turn["callbackBaseUrl"], turn["turnId"], token)
    log_payloads = bool(turn.get("logPayloads", True))

    async def sink(entry: dict) -> Any:
        if not log_payloads:
            entry = {**entry, "request": None, "response": None, "error": None}
        return await java.ai_call(entry)

    binding = ai_log.bind(sink, feature="ASSISTANT")
    try:
        chosen = model or ChatModel()
        if not chosen.configured:
            raise RuntimeError("assistant model is not configured")
        if turn.get("trip"):
            await java.step("reading_trip", {})
        photos = await images.load(turn.get("images") or [])
        if photos:
            await java.step("reading_photo", {"count": len(photos)})
        graph = build_graph(turn, java, chosen)
        limits = turn.get("limits") or {}
        # Each agent round plus its tool round is two graph steps; on top of the budget, the rounds
        # refused past it, one nudge round, start and finish.
        limit = 2 * (int(limits.get("maxToolCalls") or 6) + 2 + MAX_OVER_BUDGET_ROUNDS) + 6
        # Java sends how long it waits before failing the turn itself, minus a margin (a PRO budget
        # of 20 tool calls needs far longer than the old fixed 120 s).
        timeout = int(limits.get("timeoutSeconds") or TURN_TIMEOUT_SECONDS)
        messages = initial_messages(turn, photos, web_offered(turn))
        await asyncio.wait_for(graph.ainvoke({"messages": messages, "tool_calls": 0},
                                             {"recursion_limit": limit}), timeout=timeout)
    except TurnClosed:
        logger.info("assistant turn %s closed by Java before the answer", turn.get("turnId"))
    except Exception as exc:
        logger.exception("assistant turn %s failed", turn.get("turnId"))
        await java.fail(f"{type(exc).__name__}: {exc}")
    finally:
        ai_log.unbind(binding)
