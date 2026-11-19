"""Checks on a plan Pengoo wrote (NEW_TRIP / PLAN_DAYS), after `graph.clean_plan`.

The prompt alone did not hold these. Prod 2026-11-14, 88 chat plans: 11.6% of meal slots empty,
117 meals with no venue ("Ăn trưa gần đồi chè"), 47 lunches or dinners at a café or a snack stall,
9 of 70 multi-day plans with a night nobody sleeps anywhere, 18 trips "starting" at their own
destination, and stops carrying the id of another place ("Chợ Đà Lạt, ăn trưa" with the id of
Quảng trường Lâm Viên: Java shows the catalogue name, so the lunch became a square).

Two things happen here, both pure functions:
- `relink_places` repairs stops whose placeId points at another place, in code;
- `plan_gaps` lists what a day still lacks; `graph` sends the model back once with
  `repair_message` (one round only: more rounds stop helping and cost a model call each).

A plan copied from the user's own photo or text is theirs: the caller skips these checks for it.
"""
from __future__ import annotations

import re
from typing import Any

from ..itinerary import MEAL_DEFAULT_START, MEAL_WINDOWS
from .prompts import fold

# A stop counts as a main meal when it is one of these and starts inside the meal's window.
_FOOD_CATEGORIES = {"restaurant", "food", "street_food"}
# Search-row types that are a break, not a meal (PlaceSubType names of the FOOD family).
_BREAK_TYPES = {"CAFE", "TEA", "DESSERT", "BAKERY", "BAR"}
_MEAL_TYPES = {"RESTAURANT", "STREET_FOOD", "SEAFOOD", "VEGETARIAN", "VEGAN", "HALAL"}
# Names that are a snack or a drink whatever the listing says. A bakery is a fine breakfast.
# Not "che" or "tra": folded, "Chế Trang" and "trả phòng" would match.
_SNACK = re.compile(r"banh trang nuong|\bkem\b|sua dau|tra sua|tao pho|\bcafe\b|ca phe|coffee|\btea\b")
_BREAKFAST_OK = re.compile(r"banh mi|bakery|banh")
# A stop that is a meal by its name alone ("Ăn trưa gần đồi chè", "Lunch near the lake").
_MEAL_NAME = re.compile(r"^(an (sang|trua|toi|dem)|bua (sang|trua|toi)|breakfast|lunch|dinner)\b")
# Meals that honestly have no venue of their own: at the stay, on the road, self-catered.
# "trên": inside the park, on the island tour or the boat ("Ăn trưa trên Bà Nà").
_NO_VENUE_OK = re.compile(r"homestay|khach san|hotel|resort|bungalow|villa|cho o|tai phong|gan nha|o nha|\btren\b|"
                          r"doc duong|tram dung|buffet|tu tuc|tu nau|picnic|da ngoai|on the way|on the road|"
                          r"at the stay|on board|truoc khi (di|xuat phat|len xe)")
# Words that say what is done at a place, not which place it is.
# Not "dem": "Chợ Đêm" is a different place from "Chợ".
_ACTION_WORDS = {"an", "sang", "trua", "toi", "bua", "lunch", "dinner", "breakfast", "di", "dao", "bo", "ngam",
                 "tham", "quan", "ghe", "check", "in", "out", "nhan", "tra", "phong", "nghi", "ngoi", "o", "tai", "va",
                 "gan", "quanh", "the", "at", "and", "near", "around", "visit", "walk", "khu", "du", "lich", "nha",
                 "hang", "restaurant", "mua", "sam", "chup", "anh", "hoang", "hon", "binh", "minh", "san", "may"}
_TIME = re.compile(r"^(\d{1,2}):(\d{2})")
# Owed when the day's stops run this many minutes past the meal's usual hour (see `_owed`).
# Breakfast is owed only by a day that starts early in town: a later start means it was eaten at
# the stay or before setting out.
_MEAL_MARGIN = {"BREAKFAST": 15, "LUNCH": 60, "DINNER": 60}
MAX_GAPS = 8
# Gaps worth a model round on their own; the others are only added to a message sent anyway.
_REPAIR_KINDS = {"NO_MEAL", "SNACK_AS_MEAL", "MEAL_UNLINKED", "NO_STAY", "OVERLAP"}


def _minutes(value: Any) -> int | None:
    match = _TIME.match(str(value or ""))
    return int(match.group(1)) * 60 + int(match.group(2)) if match else None


def _words(name: Any, drop: set[str] | None = None) -> set[str]:
    """The words that identify a place in a stop's name, accent-free."""
    words = {w for w in fold(str(name or "")).split() if len(w) > 1 and w not in _ACTION_WORDS}
    kept = words - (drop or set())
    return kept or words


def collect_place_rows(result: Any, rows: dict[str, dict]) -> None:
    """Remember every catalogue place a tool result shows: id -> {name, kind, types}."""
    if isinstance(result, list):
        for item in result:
            collect_place_rows(item, rows)
        return
    if not isinstance(result, dict):
        return
    ident, name = result.get("id"), result.get("name") or result.get("title")
    if isinstance(ident, str) and name and result.get("kind") in ("food", "sight", "stay"):
        rows[ident] = {"name": str(name), "kind": result["kind"],
                       "types": [str(t) for t in result.get("types") or []]}
    for value in result.values():
        if isinstance(value, (dict, list)):
            collect_place_rows(value, rows)


def _item_family(item: dict) -> str | None:
    category = str(item.get("category") or "").lower()
    if item.get("stay") or category == "hotel":
        return "stay"
    if category in _FOOD_CATEGORIES or category == "cafe":
        return "food"
    return "sight" if category else None


def relink_places(plan: dict, rows: dict[str, dict]) -> list[str]:
    """Fixes stops whose placeId is another place's. Returns one line per fix, for the log.

    A stop is suspect when its name and the row of its placeId cannot be one place (`_same_place`). It
    then gets the id of the row it does name, when this turn showed one ("Phở Thi" carrying the
    id of Bánh căn Lệ). With no such row, it loses the id only when the kinds conflict (a lunch on
    a square's id): a stop without a place is better than a stop at the wrong place. Same kind and
    no better row is left alone: "Hồ Xuân Hương" on the row "Spring Fragrant Lake Park" is the
    same lake in another language, "Cà phê chill" on a café's row is a description of that café.
    """
    fixes: list[str] = []
    place_words = _words(plan.get("destination"))
    for day in plan.get("days") or []:
        for item in day.get("items") or []:
            place_id, name = item.get("placeId"), item.get("name")
            row = rows.get(str(place_id)) if place_id else None
            if not row or not name:
                continue
            named, listed = _words(name, place_words), _words(row.get("name"), place_words)
            if not named or not listed or _same_place(named, listed):
                continue
            best = _best_row(named, rows, place_words, exclude=str(place_id))
            family, kind = _item_family(item), row.get("kind")
            if best:
                item["placeId"] = best
                fixes.append(f"'{name}': {row.get('name')} -> {rows[best]['name']}")
            elif family and kind and family != kind:
                item.pop("placeId")
                fixes.append(f"'{name}': {row.get('name')} -> no place")
    return fixes


def _same_place(named: set[str], listed: set[str]) -> bool:
    """Whether a stop's name and a row's name can be one place: they share a word, and more than
    one when both names are long ("Mì Quảng Thuận" and "Quảng trường Lâm Viên" share only "quảng")."""
    shared = named & listed
    return bool(shared) and not (len(shared) == 1 and len(named) >= 3 and len(listed) >= 3)


def _best_row(named: set[str], rows: dict[str, dict], drop: set[str], exclude: str) -> str | None:
    best, best_score = None, 0.0
    for place_id, row in rows.items():
        if place_id == exclude:
            continue
        listed = _words(row.get("name"), drop)
        shared = named & listed
        if not shared:
            continue
        score = len(shared) / len(named | listed)
        if score > best_score:
            best, best_score = place_id, score
    # High bar: "Chợ Đà Lạt" must not move to "Chợ Đêm Đà Lạt".
    return best if best_score >= 0.75 else None


def drop_start_at_destination(plan: dict) -> bool:
    """A trip does not set out from its own destination ("Trung tâm Đà Lạt" for a Đà Lạt trip):
    the starting point is where the travellers leave from. An airport or station there stays."""
    start, destination = plan.get("startingPointName"), plan.get("destination")
    if not start or not destination:
        return False
    generic = {"trung", "tam", "tp", "thanh", "pho", "thi", "tran", "xa", "city", "center", "centre", "downtown",
               "town", "khu", "vuc"}
    start_words = set(fold(str(start)).split()) - generic
    destination_words = set(fold(str(destination).split(",")[0]).split()) - generic
    if not destination_words or not start_words <= destination_words:
        return False
    for key in ("startingPointName", "startingPointAddress", "departureTime"):
        plan.pop(key, None)
    return True


def _is_snack(item: dict, row: dict | None) -> bool:
    types = set((row or {}).get("types") or [])
    if types and types <= _BREAK_TYPES:
        return True
    if types & _MEAL_TYPES:
        return False
    return str(item.get("category") or "").lower() == "cafe" or bool(_SNACK.search(fold(str(item.get("name") or ""))))


def _is_food(item: dict, row: dict | None) -> bool:
    if (row or {}).get("kind") == "food":
        return True
    category = str(item.get("category") or "").lower()
    return category in _FOOD_CATEGORIES or category == "cafe" or bool(_MEAL_NAME.match(fold(str(item.get("name") or ""))))


def _owed(meal: str, first: int, last: int) -> bool:
    """A day owes a meal when its stops (rides left out) run across that meal's usual hour: a day
    that starts at 14:00 owes no lunch, a day that ends with the 12:30 airport taxi owes none either."""
    hour = MEAL_DEFAULT_START[meal] + _MEAL_MARGIN[meal]
    return first <= hour <= last


def plan_gaps(plan: dict, rows: dict[str, dict], new_trip: bool) -> list[dict]:
    """What the plan lacks, most important first, at most MAX_GAPS. `rows`: this turn's places."""
    gaps: list[dict] = []
    days = sorted((d for d in plan.get("days") or [] if isinstance(d, dict)), key=lambda d: d.get("dayNumber") or 0)
    seen_places: dict[str, int] = {}
    covered_nights: set[int] = set()
    linked_meals, unlinked = 0, []
    for day in days:
        number = day.get("dayNumber") or 0
        items = [i for i in day.get("items") or [] if isinstance(i, dict)]
        stops = [i for i in items if not i.get("transportMode")]
        times = [(_minutes(i.get("startTime")), _minutes(i.get("endTime"))) for i in stops]
        starts = [s for s, _ in times if s is not None]
        if not starts:
            continue
        first, last = min(starts), max([e for _, e in times if e is not None] + starts)
        # The travellers ate before or on the ride that opens the day.
        rode_in = any(i.get("transportMode") and (_minutes(i.get("startTime")) or 0) < first for i in items)
        anchor = next((i.get("name") for i in stops if not _is_food(i, rows.get(str(i.get("placeId"))))), None)
        for meal, (low, high) in MEAL_WINDOWS.items():
            if not _owed(meal, first, last) or (meal == "BREAKFAST" and rode_in):
                continue
            in_window = [i for i in stops if (_minutes(i.get("startTime")) or -1) >= low
                         and (_minutes(i.get("startTime")) or -1) < high
                         and _is_food(i, rows.get(str(i.get("placeId"))))]
            meals = [i for i in in_window if not _is_snack(i, rows.get(str(i.get("placeId"))))
                     or (meal == "BREAKFAST" and _BREAKFAST_OK.search(fold(str(i.get("name") or ""))))]
            if not meals:
                snack = in_window[0].get("name") if in_window else None
                gaps.append({"kind": "SNACK_AS_MEAL" if snack else "NO_MEAL", "day": number, "meal": meal,
                             "name": snack, "near": anchor})
                continue
            for item in meals:
                folded = fold(str(item.get("name") or ""))
                if item.get("placeId") or item.get("mapPlaceId"):
                    linked_meals += 1
                elif _MEAL_NAME.match(folded) and not _NO_VENUE_OK.search(folded):
                    unlinked.append({"kind": "MEAL_UNLINKED", "day": number, "meal": meal,
                                     "name": item.get("name"), "near": anchor})
        ordered = sorted((t for t in zip(times, stops) if t[0][0] is not None), key=lambda t: t[0][0])
        for (before, a), (after, b) in zip(ordered, ordered[1:]):
            if before[1] is not None and after[0] < before[1] - 5 and not a.get("stay") and not b.get("stay"):
                gaps.append({"kind": "OVERLAP", "day": number, "name": f"{a.get('name')} / {b.get('name')}"})
                break
        for item in stops:
            if item.get("stay"):
                until = item.get("untilDay") or number + 1
                covered_nights.update(range(number, max(until, number + 1)))
            elif item.get("placeId") and str(item.get("category") or "").lower() != "hotel":
                if str(item["placeId"]) in seen_places and seen_places[str(item["placeId"])] != number:
                    gaps.append({"kind": "REPEATED_PLACE", "day": number, "name": item.get("name")})
                seen_places.setdefault(str(item["placeId"]), number)
    # Meals without a venue are worth another search only where the plan shows the app knows food
    # venues there. A Hà Giang loop whose every meal is "ăn trưa ở Yên Minh" has none to find.
    if linked_meals:
        gaps += unlinked
    if new_trip and len(days) >= 2:
        numbers = [d.get("dayNumber") or 0 for d in days]
        missing = [n for n in range(min(numbers), max(numbers)) if n not in covered_nights]
        if missing:
            gaps.append({"kind": "NO_STAY", "day": missing[0]})
        if plan.get("startingPointName"):
            for kind, day in (("NO_ARRIVAL_LEG", days[0]), ("NO_RETURN_LEG", days[-1])):
                if not any(i.get("transportMode") for i in day.get("items") or [] if isinstance(i, dict)):
                    gaps.append({"kind": kind, "day": day.get("dayNumber") or 0})
    order = {"NO_MEAL": 0, "SNACK_AS_MEAL": 1, "NO_STAY": 2, "MEAL_UNLINKED": 3, "OVERLAP": 4}
    gaps.sort(key=lambda g: (order.get(g["kind"], 9), g["day"]))
    return gaps[:MAX_GAPS]


def worth_repair(gaps: list[dict]) -> bool:
    return any(gap["kind"] in _REPAIR_KINDS for gap in gaps)


_MEAL_WORDS = {"BREAKFAST": "breakfast", "LUNCH": "lunch", "DINNER": "dinner"}


def _window(meal: str) -> str:
    low, high = MEAL_WINDOWS[meal]
    return f"{low // 60:02d}:{low % 60:02d}-{high // 60:02d}:{high % 60:02d}"


def repair_message(gaps: list[dict]) -> str:
    """The system message that sends the model back once to finish the plan."""
    lines = []
    for gap in gaps:
        day, meal = gap["day"], _MEAL_WORDS.get(gap.get("meal") or "", "")
        near = f" near '{gap['near']}'" if gap.get("near") else ""
        if gap["kind"] == "NO_MEAL":
            lines.append(f"- Day {day}: no {meal} ({_window(gap['meal'])}). Add a food venue serving {meal}{near}.")
        elif gap["kind"] == "SNACK_AS_MEAL":
            lines.append(f"- Day {day}: '{gap['name']}' is a snack or a café, not {meal}. Add a real {meal} venue"
                         f"{near} and keep the snack as a break after it.")
        elif gap["kind"] == "MEAL_UNLINKED":
            lines.append(f"- Day {day}: '{gap['name']}' names no venue. Pick a real {meal} venue{near}.")
        elif gap["kind"] == "NO_STAY":
            lines.append(f"- Night of day {day}: nowhere to sleep. Add the stay (stay: true, untilDay = the "
                         "check-out day), or extend the existing stay's untilDay.")
        elif gap["kind"] == "OVERLAP":
            lines.append(f"- Day {day}: {gap['name']} overlap in time. Give each its own slot with the ride between.")
        elif gap["kind"] == "REPEATED_PLACE":
            lines.append(f"- Day {day}: '{gap['name']}' is already on another day. Use a different place.")
        elif gap["kind"] == "NO_ARRIVAL_LEG":
            lines.append(f"- Day {day}: the trip has a starting point but no ride to the destination. Add the "
                         "transport leg (transportMode, address = from, endAddress = to) before the first stop.")
        elif gap["kind"] == "NO_RETURN_LEG":
            lines.append(f"- Day {day}: no ride back to the starting point. Add the return leg after the last stop.")
    return ("The plan you just wrote is not complete yet:\n" + "\n".join(lines) + "\n"
            "Fix exactly these now. For each missing or unnamed meal, call search_places ONCE with `searches` "
            "(one entry per meal: types RESTAURANT or STREET_FOOD, the area or the stop it is near, the city) "
            "unless a row you already have fits; a meal on the road or at the stay may stay a name-only stop "
            "that says so. Then reply with the final JSON again: the SAME plan, whole, every other stop and "
            "time unchanged except what must shift to make room, each new stop with the id of ITS OWN row. "
            "Keep the answer text, places and suggestions. The user has not seen the plan yet, so the answer "
            "describes the finished plan, never what you fixed.")
