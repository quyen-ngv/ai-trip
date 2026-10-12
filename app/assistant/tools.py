"""Tool definitions the model sees, and the visible "thinking" step each call produces.

Tools run in Java with the asker's permissions; this module only describes them and turns a
call into the step shown live in the chat (the app translates the step key).
"""
from __future__ import annotations

from typing import Any


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [],
                       "additionalProperties": False},
    }}


_DAY = {"type": "integer", "minimum": 1, "description": "Trip day number (1 = first day)."}
_CITY = {"type": "string", "description": "A destination name such as 'Đà Nẵng' or 'Hội An' when there is no trip day to anchor on."}

TRIP_TOOLS = [
    _fn("search_places",
        "Find real catalogue places (food or sights) near a trip activity, a trip day, or a city. "
        "Results stay inside that destination's own area. Use the returned ids in your answer.",
        {"query": {"type": "string", "description": "What to look for, e.g. 'bún chả cá', 'rooftop bar', 'museum'. Omit for the best-rated places."},
         "kind": {"type": "string", "enum": ["food", "sight", "any"]},
         "activity_id": {"type": "string", "description": "Search around this activity of the trip (best for 'near X')."},
         "day": _DAY, "city": _CITY,
         "radius_km": {"type": "number", "minimum": 0.5, "maximum": 30},
         "limit": {"type": "integer", "minimum": 1, "maximum": 8}}),
    _fn("get_place_details",
        "Opening hours for every day, price band, listing tags, menu dishes and description of one place.",
        {"place_id": {"type": "string"}}, ["place_id"]),
    _fn("get_weather", "Weather now and the forecast for about a week, for a trip day or a city.",
        {"day": _DAY, "city": _CITY}),
    _fn("search_events", "Published events (concerts, festivals, exhibitions…) near a trip day or city.",
        {"day": _DAY, "city": _CITY}),
    _fn("get_local_dishes", "Signature local dishes of the destination (for 'what should we eat here').",
        {"day": _DAY, "city": _CITY}),
    _fn("travel_time", "Straight-line distance and rough walking/ride time between two activities or places.",
        {"from": {"type": "string", "description": "Activity id or place id."},
         "to": {"type": "string", "description": "Activity id or place id."}}, ["from", "to"]),
]

DIRECT_TOOLS = TRIP_TOOLS + [
    _fn("list_my_trips", "The user's own trips (upcoming first) with ids, dates and whether they can edit them.", {}),
    _fn("get_trip", "Days and activities of one of the user's trips.", {"trip_id": {"type": "string"}}, ["trip_id"]),
]


def tools_for(conversation_type: str) -> list[dict]:
    return TRIP_TOOLS if conversation_type == "TRIP" else DIRECT_TOOLS


STEP_KEYS = {
    "search_places": "search_places", "get_place_details": "place_details", "get_weather": "weather",
    "search_events": "events", "get_local_dishes": "local_dishes", "travel_time": "travel_time",
    "list_my_trips": "my_trips", "get_trip": "trip_details",
}


def step_for(name: str, args: dict, names: dict[str, str]) -> tuple[str, dict]:
    """The visible step of one tool call: a translatable key and short display params.

    `names` maps ids already seen in this turn (activities, places, trips) to their names, so
    the step says "Hotel Mường Thanh" rather than an id."""
    params: dict[str, Any] = {}
    if args.get("query"):
        params["query"] = str(args["query"])[:60]
    if args.get("day") is not None:
        params["day"] = args["day"]
    if args.get("city"):
        params["city"] = str(args["city"])[:40]
    if args.get("activity_id") and names.get(str(args["activity_id"])):
        params["near"] = names[str(args["activity_id"])][:60]
    if name == "get_place_details" and names.get(str(args.get("place_id"))):
        params["name"] = names[str(args.get("place_id"))][:60]
    if name == "get_trip" and names.get(str(args.get("trip_id"))):
        params["name"] = names[str(args.get("trip_id"))][:60]
    if name == "travel_time":
        if names.get(str(args.get("from"))):
            params["from"] = names[str(args.get("from"))][:40]
        if names.get(str(args.get("to"))):
            params["to"] = names[str(args.get("to"))][:40]
    return STEP_KEYS.get(name, "thought"), params


def collect_names(result: Any, names: dict[str, str], place_ids: set[str]) -> None:
    """Remember ids → names from a tool result; place ids become eligible for cards."""
    if isinstance(result, list):
        for item in result:
            collect_names(item, names, place_ids)
        return
    if not isinstance(result, dict):
        return
    ident, label = result.get("id"), result.get("name") or result.get("title")
    if ident and label:
        names[str(ident)] = str(label)
        if result.get("kind") in ("food", "sight"):
            place_ids.add(str(ident))
    for value in result.values():
        if isinstance(value, (dict, list)):
            collect_names(value, names, place_ids)
