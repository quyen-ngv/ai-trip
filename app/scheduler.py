"""Deterministic scheduling: turn an LLM slot plan into timed PLACE_VISIT / TRANSPORT items.

The model decides *which* places and *which slot*; this module decides everything that must be
consistent: meals present, stop counts per pace, geographic ordering, travel time, opening hours,
day windows, and the transport rows between stops.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from .candidates import activity_category, is_food, meal_hint, pace_plan, quality, row_key
from .geo import WEEKDAYS, fmt_time, has_coords, haversine_km, intercity_mode, is_open, next_opening, travel_estimate

logger = logging.getLogger(__name__)

SLOTS = ["BREAKFAST", "MORNING", "LUNCH", "AFTERNOON", "DINNER", "EVENING"]
MEAL_OF = {"BREAKFAST": "breakfast", "LUNCH": "lunch", "DINNER": "dinner"}
MEAL_TYPICAL = {"breakfast": 8 * 60 + 15, "lunch": 12 * 60 + 15, "dinner": 19 * 60}
LUNCH_EARLIEST, LUNCH_LATEST = 11 * 60 + 30, 14 * 60
DINNER_EARLIEST, DINNER_LATEST = 18 * 60, 20 * 60 + 30
MORNING_CUTOFF, AFTERNOON_CUTOFF = 13 * 60 + 30, 19 * 60 + 30
# In-city hops don't get TRANSPORT rows (travel time is still budgeted between stops);
# a row appears only for genuinely far hops and between destinations.
MIN_TRANSPORT_KM = 5.0

REST_KINDS = {
    "hotel": {"vi": "Nghỉ ngơi tại khách sạn", "en": "Rest at the hotel", "category": "hotel", "minutes": 90},
    "walk": {"vi": "Đi dạo, tự do khám phá", "en": "Leisurely walk & explore", "category": "attraction", "minutes": 60},
    "free": {"vi": "Thời gian tự do", "en": "Free time", "category": "attraction", "minutes": 60},
}


def _rest_row(kind: str, locale: str, seq: int) -> dict:
    spec = REST_KINDS[kind]
    return {"id": None, "googlePlaceId": f"rest-{seq}", "title": spec["vi" if locale.startswith("vi") else "en"],
            "address": None, "latitude": None, "longitude": None, "placeGroup": "REST", "category": spec["category"],
            "reviewRating": 0.0, "reviewCount": 0, "score": None, "distanceKm": None,
            "visitDurationMinutes": spec["minutes"], "description": "", "attributes": "", "menuHighlights": [],
            "openHours": None, "source": "generated"}


@dataclass
class Stop:
    row: dict
    slot: str
    minutes: int
    start: int = 0
    end: int = 0
    travel_km: float = 0.0
    travel_min: int = 0
    travel_mode: str = ""


@dataclass
class DayPlan:
    day_number: int
    day_date: date
    theme: str = ""
    stops: list[Stop] = field(default_factory=list)

    @property
    def weekday(self) -> str:
        return WEEKDAYS[self.day_date.weekday()]


def combined_score(r: dict) -> float:
    fit = r.get("fit")
    if fit is None:
        fit = quality(r) * 10
    return 0.65 * (fit / 10.0) + 0.35 * quality(r)


def _meals_of(r: dict) -> list[str]:
    meals = r.get("meals")
    if isinstance(meals, list) and meals:
        return [str(m).lower() for m in meals]
    return meal_hint(r)


def _default_minutes(r: dict, slot: str, pace: dict) -> int:
    if slot == "BREAKFAST":
        return min(60, pace["meal"])
    if slot in MEAL_OF:
        return pace["meal"]
    return int(r.get("visitDurationMinutes") or pace["visit"])


# --------------------------------------------------------------------- plan parsing
def parse_plan(plan: Any, rows_by_id: dict[str, dict], day_numbers: list[int], dates: list[date],
               pace: dict, used: set[str], locale: str = "vi") -> list[DayPlan]:
    """Tolerant conversion of the model's JSON into DayPlans. Unknown ids, duplicates and
    food/non-food slot mismatches are dropped here; gaps are filled later."""
    by_day = {n: DayPlan(n, d) for n, d in zip(day_numbers, dates)}
    days = (plan or {}).get("days") if isinstance(plan, dict) else None
    rest_seq = 0
    for d in days or []:
        if not isinstance(d, dict):
            continue
        try:
            n = int(d.get("dayNumber"))
        except (TypeError, ValueError):
            continue
        if n not in by_day:
            continue
        by_day[n].theme = str(d.get("theme") or "")[:80]
        snacks = night_food = 0
        for s in d.get("stops") or []:
            if not isinstance(s, dict):
                continue
            slot = str(s.get("slot") or "").upper()
            if slot not in SLOTS:
                continue
            try:
                minutes = int(s.get("minutes") or 0)
            except (TypeError, ValueError):
                minutes = 0
            # rest / stroll / free-time blocks proposed by the model (no place attached)
            rest = str(s.get("rest") or "").lower()
            if rest in REST_KINDS:
                rest_seq += 1
                row = _rest_row(rest, locale, rest_seq)
                if slot in MEAL_OF:
                    slot = "AFTERNOON"
                by_day[n].stops.append(Stop(row, slot, minutes if 30 <= minutes <= 180 else REST_KINDS[rest]["minutes"]))
                continue
            pid = str(s.get("candidateId") or s.get("placeId") or s.get("id") or "")
            row = rows_by_id.get(pid)
            if not row or row_key(row) in used:
                continue
            food = is_food(row)
            if slot in MEAL_OF and not food:
                slot = "MORNING" if slot == "BREAKFAST" else "AFTERNOON" if slot == "LUNCH" else "EVENING"
            elif slot in MEAL_OF and food and slot == "BREAKFAST" and "snack" in _meals_of(row):
                continue  # a cafe/dessert place is never breakfast; the real breakfast is filled later
            elif slot not in MEAL_OF and food:
                snackish = "snack" in _meals_of(row)
                if slot == "EVENING" and night_food < 1:
                    night_food += 1          # ăn đêm / chè / night-market food after dinner: allowed, one per day
                elif slot in ("MORNING", "AFTERNOON") and snackish and snacks < 1:
                    snacks += 1              # one cafe/dessert break per day, always after a meal by construction
                else:
                    continue
            minutes = minutes if 20 <= minutes <= 360 else _default_minutes(row, slot, pace)
            if slot in MEAL_OF:
                minutes = min(minutes, 120)
            by_day[n].stops.append(Stop(row, slot, minutes))
            used.add(row_key(row))
    return [by_day[n] for n in day_numbers]


# --------------------------------------------------------------------- filling gaps
def _centroid(day: DayPlan, center: dict) -> dict:
    pts = [s.row for s in day.stops if has_coords(s.row)]
    if not pts:
        return center
    return {"latitude": sum(p["latitude"] for p in pts) / len(pts), "longitude": sum(p["longitude"] for p in pts) / len(pts)}


def _pick(pool: list[dict], used: set[str], anchor: dict, weekday: str, at: int, minutes: int,
          predicate=None, ignore_hours: bool = False) -> dict | None:
    best, best_score = None, -1e9
    for r in pool:
        key = row_key(r)
        if key in used or (predicate and not predicate(r)):
            continue
        if not ignore_hours and is_open(r.get("openHours"), weekday, at, at + minutes) is False:
            continue
        km = haversine_km(anchor, r) if has_coords(r) and has_coords(anchor) else 3.0
        score = combined_score(r) - 0.04 * km
        if score > best_score:
            best, best_score = r, score
    return best


def fill_day(day: DayPlan, food_pool: list[dict], act_pool: list[dict], pace: dict, center: dict, used: set[str]) -> None:
    anchor = _centroid(day, center)
    # 1. one food stop per meal
    for slot, meal in MEAL_OF.items():
        if any(s.slot == slot for s in day.stops):
            continue
        at = MEAL_TYPICAL[meal]
        no_snack = lambda r: "snack" not in _meals_of(r)  # a cafe/dessert place never becomes a main meal
        row = _pick(food_pool, used, anchor, day.weekday, at, pace["meal"], lambda r, m=meal: m in _meals_of(r)) \
            or _pick(food_pool, used, anchor, day.weekday, at, pace["meal"], no_snack) \
            or _pick(food_pool, used, anchor, day.weekday, at, pace["meal"], no_snack, ignore_hours=True) \
            or _pick(food_pool, used, anchor, day.weekday, at, pace["meal"], ignore_hours=True)
        if row:
            day.stops.append(Stop(row, slot, _default_minutes(row, slot, pace)))
            used.add(row_key(row))
    # 2. activity count within pace ± 1
    target = pace["activities"]
    acts = [s for s in day.stops if s.slot not in MEAL_OF]
    while len(acts) > target + 1:
        removable = [s for s in acts if s.row.get("source") != "social"]
        if not removable:
            break
        worst = min(removable, key=lambda s: combined_score(s.row))
        day.stops.remove(worst)
        acts.remove(worst)
    # Auto-fill uses daytime slots; EVENING only appears when the model planned it or the pace is
    # EAGER, so a museum is never pushed to 20:00 just to hit a count.
    slots_cycle = (["MORNING", "MORNING", "AFTERNOON", "AFTERNOON", "EVENING"] if pace["activities"] >= 4
                   else ["MORNING", "AFTERNOON", "AFTERNOON"] if pace["activities"] >= 3 else ["MORNING", "AFTERNOON"])
    have = [s.slot for s in acts]
    i = 0
    attempts = 0
    while len(acts) < target and attempts < target * 3:
        attempts += 1
        slot = slots_cycle[i % len(slots_cycle)]
        i += 1
        if have.count(slot) >= slots_cycle.count(slot):
            continue
        at = {"MORNING": 9 * 60 + 30, "AFTERNOON": 14 * 60 + 30, "EVENING": 20 * 60}[slot]
        # prefer groups not yet represented today
        present = {s.row["placeGroup"] for s in acts}
        row = _pick(act_pool, used, anchor, day.weekday, at, pace["visit"], lambda r: r["placeGroup"] not in present) \
            or _pick(act_pool, used, anchor, day.weekday, at, pace["visit"])
        if not row:
            break
        stop = Stop(row, slot, _default_minutes(row, slot, pace))
        day.stops.append(stop)
        acts.append(stop)
        have.append(slot)
        used.add(row_key(row))


# --------------------------------------------------------------------- timing
def _nearest_order(stops: list[Stop], prev: dict | None) -> list[Stop]:
    if any(s.row.get("source") == "social" for s in stops):
        # A social video is an ordered source, not a catalogue to optimise by distance.
        return sorted(stops, key=lambda s: (
            int(s.row.get("socialSequence") or 10**9),
            str(s.row.get("candidateId") or s.row.get("candidateRef") or row_key(s.row)),
        ))
    remaining = list(stops)
    ordered: list[Stop] = []
    cur = prev
    while remaining:
        if cur and has_coords(cur):
            nxt = min(remaining, key=lambda s: haversine_km(cur, s.row) if has_coords(s.row) else 99)
        else:
            nxt = remaining[0]
        remaining.remove(nxt)
        ordered.append(nxt)
        cur = nxt.row
    return ordered


def time_day(day: DayPlan, pace: dict, start_minutes: int | None = None) -> list[str]:
    """Assign start/end to every stop in slot order; drop stops that cannot fit. Returns notes."""
    notes: list[str] = []
    by_slot: dict[str, list[Stop]] = {s: [] for s in SLOTS}
    for s in day.stops:
        by_slot[s.slot].append(s)
    t = start_minutes if start_minutes is not None else pace["start"]
    prev: dict | None = None
    placed: list[Stop] = []
    option_windows: dict[str, tuple[int, int]] = {}
    social_stops = sorted(
        [stop for stop in day.stops if stop.row.get("source") == "social"],
        key=lambda stop: (
            int(stop.row.get("socialSequence") or 10**9),
            str(stop.row.get("candidateId") or stop.row.get("candidateRef") or row_key(stop.row)),
        ),
    )
    social_stop_ids = {id(stop) for stop in social_stops}

    def selected_for_slot(slot: str) -> list[Stop]:
        """Keep one ordinary stop, but keep every stop in each explicit option group."""
        selected: list[Stop] = []
        ordinary_added = False
        groups: set[str] = set()
        for stop in by_slot[slot]:
            if id(stop) in social_stop_ids:
                continue
            group = str(stop.row.get("optionGroupId") or "").strip()
            if group:
                if group in groups:
                    continue
                groups.add(group)
                selected.extend(candidate for candidate in by_slot[slot]
                                if id(candidate) not in social_stop_ids
                                if str(candidate.row.get("optionGroupId") or "").strip() == group)
            elif not ordinary_added:
                selected.append(stop)
                ordinary_added = True
        return selected

    def put(stop: Stop, earliest: int | None, latest_start: int | None, mandatory: bool) -> bool:
        nonlocal t, prev
        option_group = str(stop.row.get("optionGroupId") or "").strip()
        if option_group and option_group in option_windows:
            stop.start, stop.end = option_windows[option_group]
            stop.travel_km = stop.travel_min = 0
            stop.travel_mode = ""
            placed.append(stop)
            return True
        km = mins = 0
        mode = ""
        if prev is not None and has_coords(prev) and has_coords(stop.row):
            km = haversine_km(prev, stop.row)
            mode, mins = travel_estimate(km)
        arrive = t + mins
        if earliest is not None:
            arrive = max(arrive, earliest)
        dur = stop.minutes
        open_now = is_open(stop.row.get("openHours"), day.weekday, arrive, arrive + dur)
        if open_now is False:
            nxt = next_opening(stop.row.get("openHours"), day.weekday, arrive)
            if nxt is not None and nxt - arrive <= 90 and (latest_start is None or nxt <= latest_start):
                arrive = nxt
            elif not mandatory:
                notes.append(f"day {day.day_number}: dropped {stop.row['title']} (closed at {fmt_time(arrive)[:5]})")
                return False
        if latest_start is not None and arrive > latest_start and not mandatory:
            notes.append(f"day {day.day_number}: dropped {stop.row['title']} (too late)")
            return False
        if arrive + dur > pace["end"]:
            if mandatory and arrive < pace["end"]:
                dur = max(30, pace["end"] - arrive)
            elif not mandatory:
                notes.append(f"day {day.day_number}: dropped {stop.row['title']} (past day end)")
                return False
        stop.start, stop.end = arrive, arrive + dur
        stop.travel_km, stop.travel_min, stop.travel_mode = km, mins, mode
        placed.append(stop)
        if option_group:
            option_windows[option_group] = (stop.start, stop.end)
        t, prev = stop.end, stop.row
        return True

    # Social-video points are an ordered source list. Schedule them first in video order even
    # when a malformed/model-produced slot would otherwise move a later point ahead of an
    # earlier one. Explicit option groups still share the first option's time window.
    for s in social_stops:
        put(s, None, None, True)

    # breakfast
    for s in selected_for_slot("BREAKFAST"):
        put(s, None, None, True)
    # morning; overflow moves to afternoon
    carry: list[Stop] = []
    for s in _nearest_order([stop for stop in by_slot["MORNING"] if id(stop) not in social_stop_ids], prev):
        if t + s.minutes > MORNING_CUTOFF:
            carry.append(s)
            continue
        put(s, None, MORNING_CUTOFF - 30, False)
    for s in selected_for_slot("LUNCH"):
        put(s, LUNCH_EARLIEST, LUNCH_LATEST, True)
    carry2: list[Stop] = []
    for s in _nearest_order([stop for stop in by_slot["AFTERNOON"] if id(stop) not in social_stop_ids] + carry, prev):
        if t + s.minutes > AFTERNOON_CUTOFF:
            carry2.append(s)
            continue
        put(s, None, AFTERNOON_CUTOFF - 30, False)
    for s in selected_for_slot("DINNER"):
        put(s, DINNER_EARLIEST, DINNER_LATEST, True)
    for s in _nearest_order([stop for stop in by_slot["EVENING"] if id(stop) not in social_stop_ids] + carry2, prev):
        put(s, None, pace["end"] - 45, False)

    day.stops = placed
    return notes


# --------------------------------------------------------------------- items
def _visit_item(stop: Stop, day: int, locale: str) -> dict:
    r = stop.row
    item = {
        "type": "PLACE_VISIT" if r.get("id") else "ACTIVITY",
        "name": r["title"],
        "dayNumber": day,
        "sortOrder": 0,
        "startTime": fmt_time(stop.start),
        "endTime": fmt_time(stop.end),
        "category": activity_category(r),
        "address": (r.get("address") or None) if (r.get("id") or r.get("externalResolutionVerified") or r.get("source") != "web") else None,
        "latitude": r.get("latitude") if (r.get("id") or r.get("externalResolutionVerified") or r.get("source") != "web") else None,
        "longitude": r.get("longitude") if (r.get("id") or r.get("externalResolutionVerified") or r.get("source") != "web") else None,
        "_slot": stop.slot,
        "_placeKey": row_key(r),
    }
    if r.get("id"):
        item["placeId"] = r["id"]
    elif r.get("externalResolutionVerified"):
        item["externalPlaceResolution"] = "GOONG"
    if r.get("candidateId"):
        item["candidateId"] = r["candidateId"]
    if r.get("source") == "web" and not r.get("id") and not r.get("externalResolutionVerified"):
        urls = [str(url) for url in (r.get("sourceUrls") or []) if str(url).startswith(("https://", "http://"))]
        prefix = ("Gợi ý từ nguồn Web đã trích dẫn, chưa có trong dữ liệu của chúng tôi." if locale.startswith("vi")
                  else "Suggested from cited Web sources; not yet in our database.")
        item["notes"] = (prefix + (" Sources: " + ", ".join(urls[:3]) if urls else ""))[:900]
    if r.get("source") == "social":
        item["_socialSequence"] = r.get("socialSequence")
        item["sourceSocialJobId"] = r.get("socialJobId")
        item["sourceSocialCandidateRef"] = r.get("candidateId") or r.get("candidateRef")
        item["optionGroupId"] = r.get("optionGroupId")
        item["optionIndex"] = r.get("optionIndex")
        item["relation"] = r.get("relation")
        if r.get("resolutionStatus") and r.get("resolutionStatus") != "RESOLVED":
            unresolved = "Địa điểm chưa xác định được trên bản đồ." if locale.startswith("vi") else "The place could not be resolved on the map."
            item["notes"] = ((item.get("notes") + " " if item.get("notes") else "") + unresolved)
    return {key: value for key, value in item.items() if value is not None}


def _transport_item(prev: Stop, cur: Stop, day: int, locale: str) -> dict:
    km, mins, mode = cur.travel_km, cur.travel_min, cur.travel_mode
    # leave so as to arrive on time; free time (if any) stays before the transport, not after
    start = max(prev.end, cur.start - mins)
    short_title = cur.row["title"][:80]  # activities.name is VARCHAR(255); titles can reach 500
    label = f"Di chuyển đến {short_title}" if locale.startswith("vi") else f"Travel to {short_title}"
    unit = "phút" if locale.startswith("vi") else "min"
    return {
        "type": "TRANSPORT", "name": label, "dayNumber": day, "sortOrder": 0,
        "startTime": fmt_time(start), "endTime": fmt_time(max(cur.start, start + 1)),
        "address": prev.row.get("address") or None, "latitude": prev.row.get("latitude"), "longitude": prev.row.get("longitude"),
        "endAddress": cur.row.get("address") or None, "endLatitude": cur.row.get("latitude"), "endLongitude": cur.row.get("longitude"),
        "transportMode": mode, "durationValueToNext": mins * 60, "durationToNext": f"{mins} {unit}",
        "distanceValueToNext": int(km * 1000), "distanceToNext": f"{km:.1f} km",
    }


def items_for_day(day: DayPlan, locale: str) -> list[dict]:
    out: list[dict] = []
    prev: Stop | None = None
    for s in sorted(day.stops, key=lambda x: x.start):
        if prev is not None and s.travel_km >= MIN_TRANSPORT_KM and s.start > prev.end:
            out.append(_transport_item(prev, s, day.day_number, locale))
        out.append(_visit_item(s, day.day_number, locale))
        prev = s
    return out


def intercity_transport(from_dest: dict, to_dest: dict, day_number: int, start_minutes: int, locale: str, *,
                        same_day: bool = False, minutes: int | None = None, arrival_day: int | None = None) -> dict:
    """Inter-destination leg. Short hops (same_day) run on the morning of the arrival day;
    long hops are an overnight block ending 08:00 on the arrival day."""
    km = haversine_km(from_dest, to_dest)
    label = ("Di chuyển sang " if locale.startswith("vi") else "Travel to ") + (to_dest.get("name") or "")
    base = {
        "type": "TRANSPORT", "name": label.strip()[:255], "sortOrder": 0,
        "address": from_dest.get("name"), "latitude": from_dest.get("latitude"), "longitude": from_dest.get("longitude"),
        "endAddress": to_dest.get("name"), "endLatitude": to_dest.get("latitude"), "endLongitude": to_dest.get("longitude"),
        "distanceValueToNext": int(km * 1000), "distanceToNext": f"{km:.0f} km",
    }
    if same_day:
        dur = minutes or max(45, int(km / 55 * 60) + 30)
        return {**base, "dayNumber": day_number, "transportMode": "CAR",
                "startTime": fmt_time(start_minutes), "endTime": fmt_time(start_minutes + dur),
                "durationValueToNext": dur * 60}
    start = min(max(start_minutes + 30, 20 * 60), 23 * 60)
    return {**base, "dayNumber": day_number, "endDayNumber": arrival_day if arrival_day is not None else day_number + 1,
            "transportMode": intercity_mode(km), "startTime": fmt_time(start), "endTime": "08:00:00"}


def schedule_destination(plan: Any, rows: list[dict], day_numbers: list[int], dates: list[date], pace_name: str | None,
                         center: dict, locale: str, used: set[str],
                         first_day_delay: int = 0) -> tuple[list[dict], list[DayPlan], list[str]]:
    """Full pipeline for one destination: parse -> fill -> time -> items.
    first_day_delay pushes the first day's start (arrival after an inter-city drive)."""
    pace = pace_plan(pace_name)
    rows_by_id = {row_key(r): r for r in rows if row_key(r)}
    rows_by_id.update({str(r["id"]): r for r in rows if r.get("id")})
    rows_by_id.update({str(r["candidateId"]): r for r in rows if r.get("candidateId")})
    food_pool = sorted((r for r in rows if is_food(r)), key=combined_score, reverse=True)
    act_pool = sorted((r for r in rows if not is_food(r) and r["placeGroup"] != "ACCOMMODATION"), key=combined_score, reverse=True)
    days = parse_plan(plan, rows_by_id, day_numbers, dates, pace, used, locale)
    notes: list[str] = []
    for idx, day in enumerate(days):
        start = pace["start"] + first_day_delay if idx == 0 and first_day_delay else None
        fill_day(day, food_pool, act_pool, pace, center, used)
        notes += time_day(day, pace, start)
        # Stops dropped by time_day stay reserved in `used`, so a second fill pass picks
        # different places instead of retrying the ones that did not fit.
        if len([s for s in day.stops if s.slot not in MEAL_OF]) < pace["activities"] \
                or any(not any(s.slot == m for s in day.stops) for m in MEAL_OF):
            fill_day(day, food_pool, act_pool, pace, center, used)
            notes += time_day(day, pace, start)
    items: list[dict] = []
    for day in days:
        items += items_for_day(day, locale)
    return items, days, notes
