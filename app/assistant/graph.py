"""Pengoo's agent loop as a LangGraph graph: agent ⇄ tools → finish.

Each model round may call tools; each tool call is shown live in the chat as a step. The loop is
bounded (`maxToolCalls` per turn, then the model must answer), and the whole turn has a time
limit. Whatever happens, Java hears back exactly once: a reply, or a failure.

A round that neither calls a tool nor returns the final JSON is usually the model announcing
work ("Mình xem thời tiết nhé") and stopping. That text is never posted as the answer: the model
is sent back once to either call the tool or answer properly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import ai_log
from . import web_search
from .chat_model import ChatModel
from .gateway import JavaGateway, TurnClosed
from .prompts import SYSTEM, context_message, history_messages
from .tools import collect_names, step_for, tools_for

logger = logging.getLogger(__name__)

TURN_TIMEOUT_SECONDS = 120
MAX_TOOL_RESULT_CHARS = 6000
# The whole destination catalogue with its guide text is read in one go; cutting it would hide
# destinations from the model.
TOOL_RESULT_CHARS = {"explore_destinations": 40000}
MAX_WEB_SEARCHES = 1
MAX_ACTIONS = 3
MAX_CARDS = 6
MAX_DESTINATIONS = 4
# Card type -> the id pools (see tools.collect_names) an id must come from.
CARD_POOLS = {"TRIP": ("public_trip", "trip"), "EVENT": ("event",), "MEETUP": ("meetup",),
              "CHECKIN": ("checkin",), "FOOD": ("dish",)}
MAX_MEMORY_CHARS = 1500
NUDGE = ("Your last message was neither a tool call nor the final JSON, so the user would see nothing useful. "
         "If you meant to check something, call that tool now. Otherwise reply with the final JSON answer.")
# A short plain-text round, or one that says what Pengoo is about to do, is an announcement and
# gets the nudge. A longer plain-text round that reads as an answer is the answer: sending it back
# only made the model repeat it as JSON (one more model round, and the text showed twice).
ANNOUNCEMENT_MAX_CHARS = 160
_ANNOUNCEMENT = re.compile(
    r"^\W*(để mình|mình (sẽ|xem|tìm|kiểm tra|tra)|đợi mình|chờ mình|let me|i'll|i will|i am going to|i'm going to|one moment|hold on)\b"
    r"|\b(nhé|nha|nhá)\W*$", re.I)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
# The history annotation (prompts.history_messages) that a model sometimes copies into its answer.
_CARD_NOTE = re.compile(r"\s*\[(?:place cards shown|place): [^\]]*\]")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
PROPOSAL_ACTIONS = ("ADD", "MOVE", "UPDATE_TIME", "REMOVE", "REPLACE",
                    "ADD_DAYS", "PREPEND_DAYS", "REMOVE_DAYS", "SWAP_DAYS", "SHIFT_DATES")
# Changes to the days themselves go first (Java keeps one and the adds it needs).
_DAY_CHANGES = ("PREPEND_DAYS", "REMOVE_DAYS", "SWAP_DAYS", "SHIFT_DATES", "ADD_DAYS")


class State(TypedDict, total=False):
    messages: list[dict]
    tool_calls: int
    web_searches: int
    nudged: bool
    final: dict
    usage: dict


# Any of these makes a JSON object the reply's final format rather than data quoted in prose.
_OUTPUT_KEYS = {"answer", "places", "proposals", "destinations", "cards", "actions", "suggestions", "memory"}
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
    """The model's closing JSON; plain text (a model that ignored the format) becomes the answer."""
    return parse_json(content) or {"answer": _FENCE.sub("", content or "").strip()}


# Words that say what a place is rather than which one it is. A name reduced to nothing but these
# ("Pho", "Cafe") is too generic to recognise in an answer.
_GENERIC_NAME_WORDS = {
    "lake", "ho", "quan", "nha", "hang", "restaurant", "cafe", "coffee", "ca", "phe", "the", "a", "an", "and", "of",
    "bar", "pub", "shop", "store", "market", "cho", "park", "cong", "vien", "temple", "chua", "den", "museum",
    "bao", "tang", "hotel", "homestay", "khach", "san", "resort", "beach", "bien", "bai", "street", "pho",
}
_NAME_TOKEN = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower().replace("đ", "d"))
    return " ".join(_NAME_TOKEN.findall("".join(c for c in text if unicodedata.category(c) != "Mn")))


def _distinctive(name: str) -> str | None:
    """The part of a place name that identifies it, accent-free: "Trúc Bạch Lake" -> "truc bach".
    None when what is left is too short to tell one place from another."""
    tokens = _fold(name).split()
    while tokens and tokens[0] in _GENERIC_NAME_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _GENERIC_NAME_WORDS:
        tokens.pop()
    if not tokens or (len(tokens) == 1 and len(tokens[0]) < 5):
        return None
    return " ".join(tokens)


def mentioned_places(answer: str, candidates: dict[str, str]) -> list[str]:
    """Ids of catalogue places (id -> name) that the answer names, in the order they appear."""
    text = f" {_fold(answer)} "
    found: list[tuple[int, str]] = []
    for place_id, name in candidates.items():
        key = _distinctive(name)
        if key:
            at = text.find(f" {key} ")
            if at >= 0:
                found.append((at, place_id))
    return [place_id for _, place_id in sorted(found)]


def attach_mentioned_places(final: dict, candidates: dict[str, str], limit: int = 6) -> dict:
    """Cards for catalogue places the answer names but the model left out of "places".

    The model sometimes answers in plain text (or JSON without "places") right after a search, so
    a catalogue place it recommends by name shows no card. Only places returned by this turn's
    tools are candidates: they are real and public, and Java checks them again."""
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


def clean_reply(final: dict, allowed_places: set[str], usage: dict,
                ids: dict[str, set[str]] | None = None) -> dict:
    """What Java receives: bounded lists, places only from this turn's tool results, earlier
    cards or the trip; buttons only for ids seen in this turn."""
    places, seen = [], set()
    for item in final.get("places") or []:
        if not isinstance(item, dict):
            continue
        place_id = str(item.get("placeId") or item.get("id") or "")
        if place_id in allowed_places and place_id not in seen:
            seen.add(place_id)
            note = str(item.get("note") or "").strip()[:300] or None
            places.append({"placeId": place_id, "note": note})
    proposals = []
    for item in final.get("proposals") or []:
        if isinstance(item, dict) and item.get("action") in PROPOSAL_ACTIONS:
            proposal = {key: item.get(key) for key in ("action", "tripId", "dayNumber", "toDayNumber", "days",
                                                       "startDate", "startTime", "endTime", "placeId", "activityId",
                                                       "category", "summary")
                        if item.get(key) not in (None, "")}
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
    reply = {"markdown": answer, "places": places[:8], "proposals": proposals[:6], "suggestions": suggestions,
             "actions": clean_actions(final.get("actions"), ids or {}),
             "cards": clean_cards(final.get("cards"), ids or {}),
             "destinations": clean_destinations(final.get("destinations"), ids or {}),
             "inputTokens": usage.get("input", 0), "outputTokens": usage.get("output", 0)}
    memory = final.get("memory")
    if isinstance(memory, str) and memory.strip():
        reply["memory"] = memory.strip()[:MAX_MEMORY_CHARS]
    return reply


def build_graph(turn: dict, java: JavaGateway, model: ChatModel):
    limits = turn.get("limits") or {}
    max_tool_calls = int(limits.get("maxToolCalls") or 6)
    max_output = int(limits.get("maxOutputTokens") or 1800)
    web_allowed = bool(limits.get("webSearch")) and web_search.configured()
    tools = tools_for(str(turn.get("conversationType") or "TRIP"), web_search=web_allowed)
    # One provider cache per tool list: the trip chat and the 1:1 chat send different tools, and a
    # changed tool list is a different prefix.
    cache_key = f"pengoo-v3-{str(turn.get('conversationType') or 'TRIP').lower()}{'-web' if web_allowed else ''}"
    names: dict[str, str] = {}
    ids: dict[str, set[str]] = {"place": set(), "event": set(), "meetup": set(), "trip": set(),
                                "public_trip": set(), "checkin": set(), "dish": set(), "destination": set()}
    allowed_places = ids["place"]
    for row in turn.get("history") or []:
        for place in row.get("places") or []:
            if isinstance(place, dict) and place.get("id"):
                allowed_places.add(str(place["id"]))
                if place.get("name"):
                    names[str(place["id"])] = str(place["name"])
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
        budget_left = max_tool_calls - state.get("tool_calls", 0)
        messages = list(state["messages"])
        if budget_left <= 0:
            messages.append({"role": "system", "content": "Tool budget used up: write the final JSON answer now."})
        reply = await model.complete(messages, tools if budget_left > 0 else None, max_tokens=max_output,
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
        if calls:
            message["tool_calls"] = calls
            if content and not content.startswith("{"):
                await java.step("thought", {}, content[:300])
            return {"messages": state["messages"] + [message], "usage": usage}
        final = parse_json(content)
        acted = state.get("tool_calls", 0) > 0
        if final is None and budget_left > 0 and not state.get("nudged") and looks_like_announcement(content, acted):
            # Announced work without doing it: one more round to act.
            if content:
                await java.step("thought", {}, content[:300])
            return {"messages": state["messages"] + [message, {"role": "system", "content": NUDGE}],
                    "usage": usage, "nudged": True}
        final = final or parse_final(content)
        if not str(final.get("answer") or "").strip():
            # An empty JSON answer after a plain-text round: that text was the answer.
            earlier = [m.get("content") for m in state["messages"]
                       if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls")]
            if earlier:
                final["answer"] = _FENCE.sub("", earlier[-1]).strip()
        return {"messages": state["messages"] + [message], "usage": usage, "final": final}

    async def run_tools(state: State) -> State:
        last = state["messages"][-1]
        results: list[dict] = []
        used = state.get("tool_calls", 0)
        searched = state.get("web_searches", 0)
        for call in last.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or ""
            try:
                args = json.loads(function.get("arguments") or "{}")
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            key, params = step_for(name, args, names)
            await java.step(key, params)
            if name == "web_search":
                if not web_allowed or searched >= MAX_WEB_SEARCHES:
                    result: dict = {"error": "Web search is not available for this answer; answer from what you have."}
                else:
                    searched += 1
                    result = await web_search.search(str(args.get("query") or ""))
            else:
                result = await java.tool(name, args)
                collect_names(result, names, ids)
            used += 1
            results.append({"role": "tool", "tool_call_id": call.get("id"),
                            "content": json.dumps(result, ensure_ascii=False, default=str)[:TOOL_RESULT_CHARS.get(name, MAX_TOOL_RESULT_CHARS)]})
        return {"messages": state["messages"] + results, "tool_calls": used, "web_searches": searched}

    async def finish(state: State) -> State:
        await java.step("writing_answer", {})
        found = {pid: names[pid] for pid in allowed_places - known_places if pid in names}
        final = attach_mentioned_places(state.get("final") or {}, found)
        await java.reply(clean_reply(final, allowed_places, state.get("usage") or {}, ids))
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


def initial_messages(turn: dict) -> list[dict]:
    question = str(turn.get("question") or "").strip()
    user = (turn.get("user") or {}).get("name") or ""
    return ([{"role": "system", "content": SYSTEM}, {"role": "system", "content": context_message(turn)}]
            + history_messages(turn)
            + [{"role": "user", "content": f"{user}: {question}" if user else question}])


async def run_turn(turn: dict, token: str, model: ChatModel | None = None) -> None:
    """Answers one turn; always ends with exactly one reply or failure sent to Java."""
    java = JavaGateway(turn["callbackBaseUrl"], turn["turnId"], token)
    log_payloads = bool(turn.get("logPayloads", True))

    async def sink(entry: dict) -> None:
        if not log_payloads:
            entry = {**entry, "request": None, "response": None, "error": None}
        await java.ai_call(entry)

    binding = ai_log.bind(sink, feature="ASSISTANT")
    try:
        chosen = model or ChatModel()
        if not chosen.configured:
            raise RuntimeError("assistant model is not configured")
        if turn.get("trip"):
            await java.step("reading_trip", {})
        graph = build_graph(turn, java, chosen)
        # Each agent round plus its tool round is two graph steps, plus one nudge round, start and finish.
        limit = 2 * (int((turn.get("limits") or {}).get("maxToolCalls") or 6) + 2) + 6
        await asyncio.wait_for(graph.ainvoke({"messages": initial_messages(turn), "tool_calls": 0},
                                             {"recursion_limit": limit}), timeout=TURN_TIMEOUT_SECONDS)
    except TurnClosed:
        logger.info("assistant turn %s closed by Java before the answer", turn.get("turnId"))
    except Exception as exc:
        logger.exception("assistant turn %s failed", turn.get("turnId"))
        await java.fail(f"{type(exc).__name__}: {exc}")
    finally:
        ai_log.unbind(binding)
