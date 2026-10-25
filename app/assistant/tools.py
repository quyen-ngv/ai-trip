"""Tool definitions the model sees, and the visible "thinking" step each call produces.

Catalogue tools run in Java with the asker's permissions; `web_search` runs here (see
`web_search.py`) and is only offered when it is configured and allowed for the turn. This module
describes the tools and turns a call into the step shown live in the chat (the app translates
the step key).
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [],
                       "additionalProperties": False},
    }}


_DAY = {"type": "integer", "minimum": 1, "description": "Trip day number (1 = first day)."}
_CITY = {"type": "string", "description": "A destination name such as 'Đà Nẵng' or 'Hội An' when there is no trip day to anchor on."}
_DATE = {"type": "string", "description": "yyyy-MM-dd (Vietnam time)."}

PLACE_TYPES = ["RESTAURANT", "STREET_FOOD", "CAFE", "TEA", "BAR", "BAKERY", "DESSERT", "SEAFOOD", "VEGETARIAN",
               "VEGAN", "HALAL", "NATURE", "BEACH", "PARK", "VIEWPOINT", "SPIRITUAL", "MUSEUM", "HERITAGE",
               "LANDMARK", "SHOPPING", "MARKET", "ENTERTAINMENT", "THEME_PARK"]
EVENT_CATEGORIES = ["MUSIC", "FESTIVAL", "TRADITIONAL_FESTIVAL", "PERFORMING_ARTS", "EXHIBITION", "FILM",
                    "FOOD_DRINK", "MARKET_FAIR", "SPORTS", "WORKSHOP", "CONFERENCE", "NIGHTLIFE", "FAMILY_KIDS",
                    "HOLIDAY_CELEBRATION", "OTHER"]
MEETUP_CATEGORIES = ["FOOD_DRINK", "COFFEE", "NIGHTLIFE", "OUTDOOR_ACTIVE", "SIGHTSEEING", "ENTERTAINMENT",
                     "SHOPPING", "WELLNESS", "SOCIAL", "OTHER"]

TRIP_TOOLS = [
    _fn("explore_destinations",
        "Every destination in the app's catalogue for 'where should we go', each with its slogan and its guide's "
        "own words on what it is, its vibe and its best season (free text such as 'tháng 11-3' = November to "
        "March, 'tránh mưa bão tháng 7-9' = avoid those months). Read them and pick by region, month and vibe "
        "yourself; show your picks as destination cards.",
        {}),
    _fn("get_destination_guide",
        "The full guide of one destination as the app's editors wrote it: about, vibe, where to stay, food, best "
        "season, things to watch out for. Use it to give concrete advice and tips (cite it, do not invent).",
        {"destination_id": {"type": "string", "description": "id from explore_destinations."}, "city": _CITY}),
    _fn("search_places",
        "Find real catalogue places near a trip activity, a trip day, or a city; results stay inside that "
        "destination's own area. The catalogue matches place NAMES word by word, so: put the KIND in `types` "
        "(e.g. CAFE, BAR, ENTERTAINMENT for board games/karaoke, MUSEUM), and use `query` only for 1-3 words likely "
        "in a name or menu (a dish like 'bún chả', a brand, a place name). Leave out words like 'quán', 'ngon', "
        "'gần', 'để gặp bạn mới'. Rows carry rating, reviews, trips and checkins by app travellers, price, tags, "
        "menu, `hours` for the day asked (`hoursOn`) and distance: enough to recommend without further calls. "
        "Rows the tool added to fill the list, whose names do not contain the asked words, are marked "
        "`nameMatch: false`. The tool also relaxes the search itself and reports what it relaxed.",
        {"query": {"type": "string", "description": "Name words only, e.g. 'bún chả cá', 'Cộng', 'rooftop'. Omit when the kind is enough."},
         "types": {"type": "array", "items": {"type": "string", "enum": PLACE_TYPES}, "maxItems": 4},
         "kind": {"type": "string", "enum": ["food", "sight", "stay", "any"],
                  "description": "stay = hotels, homestays, resorts (they have no types)."},
         "activity_id": {"type": "string", "description": "Search around this activity of the trip (best for 'near X')."},
         "day": _DAY, "city": _CITY,
         "sort": {"type": "string", "enum": ["popular", "rating"],
                  "description": "Order when there are no name words. popular = most used by app travellers (trips that "
                                 "plan it + check-ins there; rows carry 'trips' and 'checkins'); rating = review score."},
         "radius_km": {"type": "number", "minimum": 0.5, "maximum": 30},
         "limit": {"type": "integer", "minimum": 1, "maximum": 8}}),
    _fn("get_place_details",
        "Address, phone, weekly opening hours, price band, tags, menu dishes and description of up to 6 places in "
        "one call (`place_ids`). Only for what the search row does not already show.",
        {"place_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6,
                       "description": "Place ids; one call for all the places you need."},
         "place_id": {"type": "string", "description": "A single place id (same as place_ids with one entry)."}}),
    _fn("get_place_community",
        "What app travellers did at one place: how many trips plan it and check-ins it has, the latest public "
        "check-ins (with captions), users' reviews, and public trips that include it. Only when the user asks "
        "'is it worth it / what do people say / is it crowded with travellers', for at most 2 places per answer.",
        {"place_id": {"type": "string"}}, ["place_id"]),
    _fn("search_public_trips",
        "Itineraries other travellers shared publicly, around a trip day or a city (best-rated first: many stops and "
        "photos, recent), by words in the title, or the ones that include one place. Use for 'lịch trình mẫu', "
        "'người khác đi Đà Lạt thế nào', 'chuyến nào có ghé chỗ này'. Show them as TRIP cards.",
        {"day": _DAY, "city": _CITY,
         "query": {"type": "string", "description": "Words in the itinerary title, e.g. 'phượt', '3 ngày'."},
         "place_id": {"type": "string", "description": "Only trips that include this place."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 6}}),
    _fn("get_weather",
        "Weather now and the forecast for about a week, for a trip day or a city. Only when the user asks about "
        "the weather or the answer is an outdoor plan within the next 7 days; once per answer.",
        {"day": _DAY, "city": _CITY}),
    _fn("search_events",
        "Published events (concerts, festivals, exhibitions, workshops, language-exchange nights…) near a trip day "
        "or city, between dates. If nothing matches, the next upcoming events nearby come back in 'upcomingInstead'.",
        {"day": _DAY, "city": _CITY, "from": _DATE, "to": _DATE,
         "categories": {"type": "array", "items": {"type": "string", "enum": EVENT_CATEGORIES}, "maxItems": 4},
         "query": {"type": "string", "description": "Words that should appear in the event title."}}),
    _fn("search_meetups",
        "Meetups other app users host in the next few days (coffee, walks, game nights, drinks…) that anyone can "
        "join: the best answer to 'meet new people / make friends / find company'.",
        {"day": _DAY, "city": _CITY,
         "categories": {"type": "array", "items": {"type": "string", "enum": MEETUP_CATEGORIES}, "maxItems": 4}}),
    _fn("get_local_dishes", "Signature local dishes of the destination (for 'what should we eat here').",
        {"day": _DAY, "city": _CITY}),
    _fn("travel_time",
        "Straight-line distance and rough walking/ride time between activities or places. Check every hop you "
        "need in ONE call with `legs`. The trip data already carries `rideMinToNext` between consecutive stops.",
        {"legs": {"type": "array", "minItems": 1, "maxItems": 10,
                  "items": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                            "required": ["from", "to"], "additionalProperties": False},
                  "description": "Pairs of activity or place ids."},
         "from": {"type": "string", "description": "Activity id or place id (single hop)."},
         "to": {"type": "string", "description": "Activity id or place id (single hop)."}}),
    _fn("get_trip_checkins",
        "Public check-ins the members posted during this trip, grouped by place with who checked in: 'where did we "
        "check in most', 'who has been where'. Private check-ins are not visible.",
        {}),
    _fn("rank_trip_places",
        "The places in this trip ranked by how popular they are with app travellers (trips that plan them, "
        "check-ins there): 'which of our stops are the famous ones / most visited'.",
        {}),
    _fn("get_budget",
        "The trip's money so far in its currency: budget, spent, remaining, spent by category, estimates of planned "
        "activities, and the asker's own share. For 'how much have we spent / are we over budget'.",
        {}),
]

DIRECT_TOOLS = TRIP_TOOLS + [
    _fn("list_my_trips", "The user's own trips (upcoming first) with ids, dates and whether they can edit them.", {}),
    _fn("get_trip", "Days and activities of one of the user's trips.", {"trip_id": {"type": "string"}}, ["trip_id"]),
]
# In the 1:1 chat these read a trip the user names.
_NEEDS_TRIP = {"get_budget", "get_trip_checkins", "rank_trip_places"}
DIRECT_TOOLS = [t for t in DIRECT_TOOLS if t["function"]["name"] not in _NEEDS_TRIP] + [
    _fn("get_budget", "Budget, spending by category and the user's own share for one of their trips.",
        {"trip_id": {"type": "string"}}, ["trip_id"]),
    _fn("get_trip_checkins", "Public check-ins posted during one of the user's trips, grouped by place.",
        {"trip_id": {"type": "string"}}, ["trip_id"]),
    _fn("rank_trip_places", "Places of one of the user's trips ranked by popularity with app travellers.",
        {"trip_id": {"type": "string"}}, ["trip_id"]),
]

WEB_SEARCH = _fn(
    "web_search",
    "Search the internet, at most once per answer. Use it (1) to find real places when search_places came back "
    "thin for the ask (few rows, rows that do not fit, a part of the ask no row covers, or coverage thin): query "
    "exactly what is missing plus the area and the city; (2) for local facts you are not sure of (is there a "
    "morning market on this beach, does the ferry still run) and facts that must be current (an event date, "
    "whether a venue still exists, news). Not for what you already know well. Results are unverified web snippets: name the places in the answer with the source link; never "
    "turn them into place cards or proposals.",
    {"query": {"type": "string", "description": "A short web query in the user's language, with the city."}},
    ["query"])


READ_URL = _fn(
    "read_url",
    "Open one web page and read its text (title, headings, paragraphs, lists), at most twice per answer. Use it on "
    "a link the user sent, and on the best web_search result when its snippet does not hold what you need: a list "
    "article ('Top 10 quán …') names its shops only inside the page, so read it before answering. Prefer local food "
    "or news sites. Pages built by JavaScript (Facebook, TikTok, Google Maps) come back nearly empty: try another. "
    "Unverified like web_search: name what it says with the link as the source; never place cards or proposals.",
    {"url": {"type": "string", "description": "The full http(s) link."}},
    ["url"])


def tools_for(conversation_type: str, web_search: bool = False) -> list[dict]:
    """Reading pages comes with web search: one switch for everything that reaches the internet."""
    tools = TRIP_TOOLS if conversation_type == "TRIP" else DIRECT_TOOLS
    return tools + [WEB_SEARCH, READ_URL] if web_search else tools


# The app translates these keys; a key it does not know reads as "working" plus the step detail.
STEP_KEYS = {
    "search_places": "search_places", "get_place_details": "place_details", "get_weather": "weather",
    "search_events": "events", "search_meetups": "meetups", "get_local_dishes": "local_dishes",
    "travel_time": "travel_time", "list_my_trips": "my_trips", "get_trip": "trip_details",
    "web_search": "web_search", "read_url": "read_page", "get_budget": "budget", "search_public_trips": "public_trips",
    "get_place_community": "place_community", "get_trip_checkins": "trip_checkins",
    "rank_trip_places": "trip_place_ranking", "explore_destinations": "destinations",
    "get_destination_guide": "destination_guide",
}


def step_for(name: str, args: dict, names: dict[str, str]) -> tuple[str, dict]:
    """The visible step of one tool call: a translatable key and short display params.

    `names` maps ids already seen in this turn (activities, places, trips) to their names, so
    the step says "Hotel Mường Thanh" rather than an id."""
    params: dict[str, Any] = {}
    if args.get("query"):
        params["query"] = str(args["query"])[:60]
    elif args.get("types"):
        params["query"] = ", ".join(str(t).lower() for t in args["types"])[:60]
    if args.get("day") is not None:
        params["day"] = args["day"]
    if args.get("city"):
        params["city"] = str(args["city"])[:40]
    if args.get("activity_id") and names.get(str(args["activity_id"])):
        params["near"] = names[str(args["activity_id"])][:60]
    place_ids = [str(p) for p in args.get("place_ids") or [] if p]
    if not place_ids and args.get("place_id"):
        place_ids = [str(args["place_id"])]
    if name in ("get_place_details", "get_place_community", "search_public_trips") and place_ids and names.get(place_ids[0]):
        params["name"] = names[place_ids[0]][:60] + (f" +{len(place_ids) - 1}" if len(place_ids) > 1 else "")
    if name == "get_destination_guide" and names.get(str(args.get("destination_id"))):
        params["name"] = names[str(args.get("destination_id"))][:60]
    if name == "read_url":
        params["name"] = (urlsplit(str(args.get("url") or "")).hostname or "").removeprefix("www.")[:60]
    if name == "get_trip" and names.get(str(args.get("trip_id"))):
        params["name"] = names[str(args.get("trip_id"))][:60]
    if name == "travel_time":
        legs = [leg for leg in args.get("legs") or [] if isinstance(leg, dict)] or [args]
        if names.get(str(legs[0].get("from"))):
            params["from"] = names[str(legs[0].get("from"))][:40]
        if names.get(str(legs[0].get("to"))):
            params["to"] = names[str(legs[0].get("to"))][:40] + (f" +{len(legs) - 1}" if len(legs) > 1 else "")
    return STEP_KEYS.get(name, "thought"), params


def collect_names(result: Any, names: dict[str, str], ids: dict[str, set[str]]) -> None:
    """Remember ids → names from a tool result; ids of places, events, meetups, trips, check-ins
    and dishes become eligible for cards and buttons (`ids` is keyed by kind). A `placeId` on any
    row (an activity, a check-in) is a real catalogue place too."""
    if isinstance(result, list):
        for item in result:
            collect_names(item, names, ids)
        return
    if not isinstance(result, dict):
        return
    ident, label = result.get("id"), result.get("name") or result.get("title")
    if ident and label:
        names[str(ident)] = str(label)
        kind = result.get("kind")
        if kind in ("food", "sight", "stay"):
            ids["place"].add(str(ident))
        elif kind == "trip":
            # Someone else's public itinerary: a card, never an OPEN_TRIP button.
            ids.setdefault("public_trip", set()).add(str(ident))
        elif kind in ("event", "meetup", "checkin", "dish", "destination"):
            ids.setdefault(kind, set()).add(str(ident))
        elif "startDate" in result or "canEdit" in result:
            ids["trip"].add(str(ident))
    if result.get("placeId"):
        ids["place"].add(str(result["placeId"]))
    for value in result.values():
        if isinstance(value, (dict, list)):
            collect_names(value, names, ids)
