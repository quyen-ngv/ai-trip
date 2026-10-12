"""Pengoo's agent loop as a LangGraph graph: agent ⇄ tools → finish.

Each model round may call tools; each tool call is shown live in the chat as a step. The loop is
bounded (`maxToolCalls` per turn, then the model must answer), and the whole turn has a time
limit. Whatever happens, Java hears back exactly once: a reply, or a failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import ai_log
from .chat_model import ChatModel
from .gateway import JavaGateway, TurnClosed
from .prompts import SYSTEM, context_message, history_messages
from .tools import collect_names, step_for, tools_for

logger = logging.getLogger(__name__)

TURN_TIMEOUT_SECONDS = 120
MAX_TOOL_RESULT_CHARS = 6000
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class State(TypedDict, total=False):
    messages: list[dict]
    tool_calls: int
    final: dict
    usage: dict


def parse_final(content: str) -> dict:
    """The model's closing JSON; plain text (a model that ignored the format) becomes the answer."""
    text = _FENCE.sub("", content or "").strip()
    match = _JSON_OBJECT.search(text)
    for candidate in (text, match.group(0) if match else None):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and ("answer" in value or "places" in value):
            return value
    return {"answer": text}


def clean_reply(final: dict, allowed_places: set[str], usage: dict) -> dict:
    """What Java receives: bounded lists, places only from this turn's tool results or the trip."""
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
        if isinstance(item, dict) and item.get("action") in ("ADD", "MOVE", "UPDATE_TIME", "REMOVE"):
            proposal = {key: item.get(key) for key in ("action", "tripId", "dayNumber", "toDayNumber", "startTime",
                                                       "endTime", "placeId", "activityId", "category", "summary")
                        if item.get(key) not in (None, "")}
            for key in ("startTime", "endTime"):
                if key in proposal and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(proposal[key])):
                    proposal.pop(key)
            for key in ("dayNumber", "toDayNumber"):
                if key in proposal:
                    try:
                        proposal[key] = int(proposal[key])
                    except (TypeError, ValueError):
                        proposal.pop(key)
            if "summary" in proposal:
                proposal["summary"] = str(proposal["summary"])[:300]
            proposals.append(proposal)
    suggestions = [str(s).strip()[:120] for s in (final.get("suggestions") or []) if str(s).strip()][:4]
    answer = str(final.get("answer") or "").strip()[:8000]
    return {"markdown": answer, "places": places[:8], "proposals": proposals[:6], "suggestions": suggestions,
            "inputTokens": usage.get("input", 0), "outputTokens": usage.get("output", 0)}


def build_graph(turn: dict, java: JavaGateway, model: ChatModel):
    limits = turn.get("limits") or {}
    max_tool_calls = int(limits.get("maxToolCalls") or 6)
    max_output = int(limits.get("maxOutputTokens") or 1800)
    tools = tools_for(str(turn.get("conversationType") or "TRIP"))
    names: dict[str, str] = {}
    allowed_places: set[str] = set()
    trip = turn.get("trip") or {}
    for day in trip.get("days") or []:
        for activity in day.get("activities") or []:
            if activity.get("id") and activity.get("name"):
                names[str(activity["id"])] = str(activity["name"])
            if activity.get("placeId"):
                allowed_places.add(str(activity["placeId"]))
    if trip.get("id") and trip.get("name"):
        names[str(trip["id"])] = str(trip["name"])

    async def agent(state: State) -> State:
        budget_left = max_tool_calls - state.get("tool_calls", 0)
        messages = list(state["messages"])
        if budget_left <= 0:
            messages.append({"role": "system", "content": "Tool budget used up: write the final JSON answer now."})
        reply = await model.complete(messages, tools if budget_left > 0 else None, max_tokens=max_output,
                                     operation="ASSISTANT_TURN", cache_key="pengoo-v1")
        if reply is None:
            raise RuntimeError("model call failed")
        usage = dict(state.get("usage") or {"input": 0, "output": 0})
        usage["input"] += reply["_usage"]["input"]
        usage["output"] += reply["_usage"]["output"]
        calls = reply.get("tool_calls") or []
        content = (reply.get("content") or "").strip()
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if calls:
            message["tool_calls"] = calls
            if content and not content.startswith("{"):
                await java.step("thought", {}, content[:300])
            return {"messages": state["messages"] + [message], "usage": usage}
        return {"messages": state["messages"] + [message], "usage": usage, "final": parse_final(content)}

    async def run_tools(state: State) -> State:
        last = state["messages"][-1]
        results: list[dict] = []
        used = state.get("tool_calls", 0)
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
            result = await java.tool(name, args)
            collect_names(result, names, allowed_places)
            used += 1
            results.append({"role": "tool", "tool_call_id": call.get("id"),
                            "content": json.dumps(result, ensure_ascii=False, default=str)[:MAX_TOOL_RESULT_CHARS]})
        return {"messages": state["messages"] + results, "tool_calls": used}

    async def finish(state: State) -> State:
        await java.step("writing_answer", {})
        await java.reply(clean_reply(state.get("final") or {}, allowed_places, state.get("usage") or {}))
        return {}

    def route(state: State) -> str:
        return "finish" if state.get("final") is not None else "tools"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finish": "finish"})
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
        # Each agent round plus its tool round is two graph steps, plus start and finish.
        limit = 2 * (int((turn.get("limits") or {}).get("maxToolCalls") or 6) + 2) + 4
        await asyncio.wait_for(graph.ainvoke({"messages": initial_messages(turn), "tool_calls": 0},
                                             {"recursion_limit": limit}), timeout=TURN_TIMEOUT_SECONDS)
    except TurnClosed:
        logger.info("assistant turn %s closed by Java before the answer", turn.get("turnId"))
    except Exception as exc:
        logger.exception("assistant turn %s failed", turn.get("turnId"))
        await java.fail(f"{type(exc).__name__}: {exc}")
    finally:
        ai_log.unbind(binding)
