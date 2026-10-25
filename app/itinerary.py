"""Turn the AI's itinerary JSON into commit items. The AI owns the plan; this module is only
the safety net for what Java's commit endpoint would reject (unknown ids, reused places,
bad/overlapping times) plus the two things product insists on: three meals a day and
short descriptions. Nothing here re-plans."""
from __future__ import annotations
import logging
import re
from datetime import date
from typing import Any
from .candidates import activity_category, is_food, meal_hint, pace_plan, row_key
from .geo import WEEKDAYS, fmt_time, has_coords, haversine_km
from .scheduler import REST_KINDS, _rest_row, combined_score

logger = logging.getLogger(__name__)

MEAL_WINDOWS = {"BREAKFAST": (5 * 60 + 30, 10 * 60 + 30), "LUNCH": (11 * 60, 14 * 60 + 30), "DINNER": (17 * 60 + 30, 21 * 60 + 30)}
MEAL_DEFAULT_START = {"BREAKFAST": 7 * 60 + 30, "LUNCH": 12 * 60, "DINNER": 18 * 60 + 30}
# The trip overview: a few practical lines (season, getting around, what to eat, a tip).
TRIP_DESCRIPTION_CHARS = 900
_TIME = re.compile(r"^\s*(\d{1,2})[:h](\d{2})")


def _parse_time(v: Any) -> int | None:
    if v is None:
        return None
    m = _TIME.match(str(v))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 60 + mi if 0 <= h <= 24 and 0 <= mi <= 59 else None


def _default_minutes(row: dict, pace: dict) -> int:
    if row.get("placeGroup") == "REST":
        return int(row.get("visitDurationMinutes") or 60)
    if is_food(row):
        return pace["meal"]
    return int(row.get("visitDurationMinutes") or pace["visit"])


def _slot_of(row: dict, start: int) -> str:
    if is_food(row):
        for slot, (lo, hi) in MEAL_WINDOWS.items():
            if lo <= start < hi and "snack" not in (row.get("meals") or meal_hint(row)):
                return slot
        return "SNACK"
    return "MORNING" if start < 12 * 60 else "AFTERNOON" if start < 18 * 60 else "EVENING"


class _Stop:
    __slots__ = ("row", "start", "end", "description", "slot", "stay")

    def __init__(self, row: dict, start: int, end: int, description: str,
                 stay: tuple[int, int] | None = None):
        self.row, self.start, self.end, self.description = row, start, end, description
        self.slot = _slot_of(row, start)
        self.stay = stay


def parse_stay(value: Any) -> tuple[int, int] | None:
    """The planner's time-to-spend estimate: [min, max] minutes, a single number, or "60-90".
    Nonsense (non-numbers, zero, more than ten hours) is ignored rather than guessed at."""
    numbers: list[float] = []
    if isinstance(value, (int, float)):
        numbers = [float(value)]
    elif isinstance(value, (list, tuple)):
        for part in value[:2]:
            try:
                numbers.append(float(part))
            except (TypeError, ValueError):
                return None
    elif isinstance(value, str):
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value)[:2]]
    if not numbers:
        return None
    lo, hi = int(round(min(numbers))), int(round(max(numbers)))
    if lo < 5 or hi > 600:
        return None
    return lo, hi


def fit_text(text: str, max_chars: int) -> str:
    """Shortens to `max_chars` at the last sentence or line end when one is close enough, else at
    a word, with an ellipsis; never mid-word when it can be helped."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    ends = [cut.rfind(mark) for mark in (". ", "! ", "? ", "\n")]
    best = max(ends)
    if best >= int(max_chars * 0.6):
        return cut[: best + 1].rstrip()
    space = cut.rfind(" ")
    return (cut[:space] if space >= int(max_chars * 0.6) else cut).rstrip(" ,;:-") + "…"


def _round_minutes(minutes: int) -> int:
    return max(5, int(round(minutes / 5.0)) * 5) if minutes < 60 else int(round(minutes / 30.0)) * 30


def _hours(minutes: int, locale: str) -> str:
    value = minutes / 60.0
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if locale.startswith("vi") else text


def stay_label(lo: int, hi: int, locale: str) -> str:
    """"Nên dành 45–60 phút" / "Plan for 1.5–2 h": what the traveller should budget there."""
    vi = locale.startswith("vi")
    lo, hi = _round_minutes(lo), _round_minutes(hi)
    if hi <= 60:
        span = f"{lo}–{hi}" if lo != hi else f"{lo}"
        amount = f"{span} {'phút' if vi else 'min'}"
    elif lo >= 60:
        span = f"{_hours(lo, locale)}–{_hours(hi, locale)}" if lo != hi else _hours(lo, locale)
        amount = f"{span} {'giờ' if vi else 'h'}"
    else:
        amount = f"{lo} {'phút' if vi else 'min'}–{_hours(hi, locale)} {'giờ' if vi else 'h'}"
    return f"⏱ {'Nên dành' if vi else 'Plan for'} {amount}"


def add_time_estimates(items: list[dict], locale: str) -> None:
    """Puts the time-to-spend estimate first in every place stop's description: the planner's own
    range when it gave one, else the planned slot. Transport items and rest blocks have none
    (their time is the slot itself)."""
    for item in items:
        if item.get("type") == "TRANSPORT" or item.get("_rest"):
            continue
        stay = item.get("_stay")
        if not stay:
            minutes = _slot_minutes(item)
            if minutes is None:
                continue
            stay = (minutes, minutes)
        label = stay_label(stay[0], stay[1], locale)
        body = (item.get("description") or "").strip()
        item["description"] = f"{label}\n{body}" if body else label


def _slot_minutes(item: dict) -> int | None:
    start, end = _parse_time(item.get("startTime")), _parse_time(item.get("endTime"))
    if start is None or end is None or item.get("endDayNumber"):
        return None
    minutes = end - start
    return minutes if 5 <= minutes <= 600 else None


def _fix_overlaps(stops: list[_Stop], pace: dict) -> list[str]:
    """Keep the AI's order while preserving same-window option alternatives."""
    notes: list[str] = []
    if any(s.row.get("source") == "social" for s in stops):
        stops.sort(key=lambda s: (
            int(s.row.get("socialSequence") or 10**9),
            s.start,
            row_key(s.row),
        ))
    else:
        stops.sort(key=lambda s: s.start)
    prev_end = None
    option_windows: dict[str, tuple[int, int]] = {}
    kept: list[_Stop] = []
    for s in stops:
        dur = max(20, s.end - s.start) if s.end > s.start else _default_minutes(s.row, pace)
        option_group = str(s.row.get("optionGroupId") or "").strip()
        option_window = option_windows.get(option_group) if option_group else None
        if option_window is not None:
            s.start, s.end = option_window
        else:
            if prev_end is not None and s.start < prev_end + 5:
                s.start = prev_end + 10
            s.end = s.start + dur
        if s.start >= 23 * 60 + 30:
            notes.append(f"dropped {s.row['title']} (past midnight)")
            continue
        s.end = min(s.end, 23 * 60 + 59)
        s.slot = _slot_of(s.row, s.start)
        kept.append(s)
        prev_end = max(prev_end or s.end, s.end)
        if option_group:
            option_windows.setdefault(option_group, (s.start, s.end))
    stops[:] = kept
    return notes


def _ensure_meals(stops: list[_Stop], food_pool: list[dict], used: set[str], pace: dict, weekday: str,
                  locale: str) -> list[str]:
    notes: list[str] = []
    present = {s.slot for s in stops}
    for meal, (lo, hi) in MEAL_WINDOWS.items():
        if meal in present:
            continue
        tag = meal.lower()
        anchor = min(stops, key=lambda s: abs(s.start - MEAL_DEFAULT_START[meal]), default=None)
        anchor_row = anchor.row if anchor and has_coords(anchor.row) else None

        def score(r: dict) -> float:
            km = haversine_km(anchor_row, r) if anchor_row and has_coords(r) else 3.0
            bonus = 0.3 if tag in (r.get("meals") or meal_hint(r)) else 0.0
            return combined_score(r) + bonus - 0.04 * km

        pick = None
        for r in sorted(food_pool, key=score, reverse=True):
            key = row_key(r)
            if key in used or "snack" in (r.get("meals") or meal_hint(r)):
                continue
            pick = r
            break
        if not pick:
            notes.append(f"no food candidate left for {tag}")
            continue
        # place it in the window, after whatever ends before the window opens
        start = MEAL_DEFAULT_START[meal]
        for s in stops:
            if s.start < hi and s.end > start:
                start = max(start, s.end + 10)
        if start >= hi:
            start = lo + 15
        text = ("Bữa " + {"breakfast": "sáng", "lunch": "trưa", "dinner": "tối"}[tag] + " tại " + pick["title"]) if locale.startswith("vi") \
            else f"{tag.capitalize()} at {pick['title']}"
        stops.append(_Stop(pick, start, start + pace["meal"], text))
        used.add(row_key(pick))
        notes.append(f"added missing {tag}: {pick['title']}")
    return notes


def _item(stop: _Stop, day: int, locale: str, max_chars: int) -> dict:
    r = stop.row
    # Only raw cited-Web discoveries lack a trusted location. Existing/social data retains its
    # established contract; Web rows get location only after Java marks them verified.
    location_is_verified = bool(r.get("id") or r.get("externalResolutionVerified") or r.get("source") != "web")
    item = {
        "type": "PLACE_VISIT" if r.get("id") else "ACTIVITY",
        "name": r["title"], "dayNumber": day, "sortOrder": 0,
        "startTime": fmt_time(stop.start), "endTime": fmt_time(stop.end),
        "category": activity_category(r) if r.get("placeGroup") != "REST" else r.get("category"),
        "address": (r.get("address") or None) if location_is_verified else None,
        "latitude": r.get("latitude") if location_is_verified else None,
        "longitude": r.get("longitude") if location_is_verified else None,
        "description": fit_text(stop.description or "", max_chars) or None,
        "_slot": stop.slot, "_placeKey": row_key(r),
        "_stay": stop.stay,
        "_rest": r.get("placeGroup") == "REST" or None,
    }
    if r.get("id"):
        item["placeId"] = r["id"]
    elif r.get("externalResolutionVerified"):
        item["externalPlaceResolution"] = r.get("resolvedBy") or "GOONG"
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


def materialize(ai: Any, rows: list[dict], day_numbers: list[int], dates: list[date], pace_name: str | None,
                locale: str, used: set[str], *, max_chars: int = 200, fill_missing_meals: bool = True,
                first_day_start: int | None = None) -> tuple[list[dict], str, list[str]]:
    """AI JSON -> (items, tripDescription, notes). Raises ValueError when the JSON is unusable."""
    if not isinstance(ai, dict) or not isinstance(ai.get("days"), list):
        raise ValueError("itinerary JSON has no days")
    pace = pace_plan(pace_name)
    rows_by_id = {row_key(r): r for r in rows if row_key(r)}
    rows_by_id.update({str(r["id"]): r for r in rows if r.get("id")})
    rows_by_id.update({str(r["candidateId"]): r for r in rows if r.get("candidateId")})
    food_pool = sorted((r for r in rows if is_food(r)), key=combined_score, reverse=True)
    by_number = {}
    for d in ai["days"]:
        if isinstance(d, dict):
            try:
                by_number[int(d.get("dayNumber"))] = d
            except (TypeError, ValueError):
                pass
    items: list[dict] = []
    notes: list[str] = []
    rest_seq = 0
    for idx, (n, dt) in enumerate(zip(day_numbers, dates)):
        weekday = WEEKDAYS[dt.weekday()]
        raw = by_number.get(n) or (ai["days"][idx] if idx < len(ai["days"]) and isinstance(ai["days"][idx], dict) else {})
        stops: list[_Stop] = []
        cursor = first_day_start if (idx == 0 and first_day_start) else pace["start"]
        for s in raw.get("stops") or []:
            if not isinstance(s, dict):
                continue
            rest = str(s.get("rest") or "").lower()
            if rest in REST_KINDS:
                rest_seq += 1
                row = _rest_row(rest, locale, rest_seq)
            else:
                pid = str(s.get("candidateId") or s.get("placeId") or s.get("id") or "")
                row = rows_by_id.get(pid)
                if not row:
                    notes.append(f"day {n}: unknown id {pid!r} dropped")
                    continue
                key = row_key(row)
                if key in used:
                    notes.append(f"day {n}: {row['title']} already used, dropped")
                    continue
                used.add(key)
            start = _parse_time(s.get("start") or s.get("startTime"))
            end = _parse_time(s.get("end") or s.get("endTime"))
            if start is None:
                start = cursor
            if end is None or end <= start:
                end = start + _default_minutes(row, pace)
            stops.append(_Stop(row, start, end, str(s.get("description") or ""),
                               None if rest in REST_KINDS else parse_stay(s.get("stay"))))
            cursor = end + 15
        if first_day_start and idx == 0:
            for s in stops:
                if s.start < first_day_start:  # arrival morning: nothing before the drive ends
                    shift = first_day_start - s.start
                    s.start += shift
                    s.end += shift
        notes += [f"day {n}: {x}" for x in _fix_overlaps(stops, pace)]
        if fill_missing_meals:
            notes += [f"day {n}: {x}" for x in _ensure_meals(stops, food_pool, used, pace, weekday, locale)]
            notes += [f"day {n}: {x}" for x in _fix_overlaps(stops, pace)]
        # No "Travel to …" rows inside a destination: the app draws every leg between two stops
        # itself (distance and time from the map), so a row per hop only doubled the list.
        for s in stops:
            items.append(_item(s, n, locale, max_chars))
        if not stops:
            notes.append(f"day {n}: AI returned no stops")
    desc = str(ai.get("tripDescription") or "").strip()
    return items, desc[:TRIP_DESCRIPTION_CHARS], notes


def append_missing_social_items(
    items: list[dict],
    rows: list[dict],
    day_numbers: list[int],
    dates: list[date],
    pace_name: str | None,
    locale: str,
    *,
    max_chars: int = 200,
) -> list[dict]:
    """Make every extracted social mention visible in the generated itinerary.

    The planner may choose not to repeat a row that looks redundant, but social extraction is
    an ordered source list and the product contract is to retain every mention.  Missing rows
    are therefore appended deterministically using their video order/day hint.  Options keep a
    shared window and remain separate activities; unresolved rows become ACTIVITY rows because
    they have no GoRoute place id.
    """
    social_rows = [r for r in rows if r.get("source") == "social"]
    if not social_rows or not day_numbers:
        return items

    def sequence(row: dict) -> int:
        try:
            return int(row.get("socialSequence") or 0)
        except (TypeError, ValueError):
            return 0

    def candidate_ref(row: dict) -> str:
        return str(row.get("candidateId") or row.get("candidateRef") or row_key(row))

    existing_refs = {
        str(item.get("sourceSocialCandidateRef"))
        for item in items
        if item.get("sourceSocialCandidateRef")
    }
    option_windows: dict[str, tuple[int, int, int]] = {}
    day_cursor: dict[int, int] = {}
    for item in items:
        day = item.get("dayNumber")
        if not isinstance(day, int) or item.get("type") == "TRANSPORT":
            continue
        start = _parse_time(item.get("startTime")) or pace_plan(pace_name)["start"]
        end = _parse_time(item.get("endTime")) or start + pace_plan(pace_name)["visit"]
        day_cursor[day] = max(day_cursor.get(day, 0), end + 10)
        group = str(item.get("optionGroupId") or "").strip()
        if group and item.get("sourceSocialCandidateRef"):
            option_windows.setdefault(group, (day, start, end))

    ordered = sorted(social_rows, key=lambda row: (sequence(row) or 10**9, candidate_ref(row)))
    pace = pace_plan(pace_name)
    for position, row in enumerate(ordered):
        ref = candidate_ref(row)
        if not ref or ref in existing_refs:
            continue
        try:
            hint = int(row.get("dayHint") or 0)
        except (TypeError, ValueError):
            hint = 0
        if hint in day_numbers:
            day = hint
        elif 1 <= hint <= len(day_numbers):
            day = day_numbers[hint - 1]
        else:
            day = day_numbers[min(len(day_numbers) - 1, position * len(day_numbers) // max(1, len(ordered)))]

        group = str(row.get("optionGroupId") or "").strip()
        window = option_windows.get(group) if group else None
        if window is not None:
            day, start, end = window
        else:
            start = _parse_time(row.get("timeHint")) or day_cursor.get(day, pace["start"])
            start = max(pace["start"], min(start, pace["end"] - 45))
            end = min(pace["end"], start + _default_minutes(row, pace))
            if group:
                option_windows[group] = (day, start, end)
        description = social_stop_text(row) or (
            "Kiểm tra lại địa chỉ và giờ mở cửa trước khi đi." if locale.startswith("vi")
            else "Check the address and opening hours before you go."
        )
        item = _item(_Stop(row, start, end, description), day, locale, max_chars)
        item["_socialSequence"] = sequence(row)
        _insert_social_item(items, item)
        existing_refs.add(ref)
        if not group:
            day_cursor[day] = max(day_cursor.get(day, 0), end + 10)

    return items


SOCIAL_TEXT_MAX_CHARS = 1500
SOCIAL_TIPS_MAX = 6


def social_stop_text(row: dict) -> str:
    """A video stop's guidance written by the extractor, then its tips one per line."""
    text = str(row.get("description") or "").strip()
    tips = [tip for tip in (row.get("tips") or []) if tip and tip not in text][:SOCIAL_TIPS_MAX]
    lines = ([text] if text else []) + [f"• {tip}" for tip in tips]
    return "\n".join(lines)[:SOCIAL_TEXT_MAX_CHARS]


def social_trip_text(social_context: dict) -> str:
    """The trip overview the extractor wrote from the video, then its general advice."""
    text = str(social_context.get("tripDescription") or "").strip()
    advice = [str(tip).strip() for tip in (social_context.get("generalGuidance") or [])
              if isinstance(tip, str) and tip.strip() and tip.strip() not in text][:SOCIAL_TIPS_MAX]
    lines = ([text] if text else []) + [f"• {tip}" for tip in advice]
    return "\n".join(lines)[:SOCIAL_TEXT_MAX_CHARS]


def apply_social_texts(items: list[dict], rows: list[dict]) -> None:
    """Give every video stop the text drawn from the video instead of the planner's one-liner.

    The planner sees a 120-character excerpt and writes a single sentence; the extractor saw the
    whole video. Stops without extracted text keep whatever the planner wrote.
    """
    by_ref = {str(row.get("candidateId")): row for row in rows
              if row.get("source") == "social" and row.get("candidateId")}
    for item in items:
        row = by_ref.get(str(item.get("sourceSocialCandidateRef") or ""))
        text = social_stop_text(row) if row else ""
        if text:
            item["description"] = text


def _insert_social_item(items: list[dict], item: dict) -> None:
    """Insert a missing source row without changing the source order of existing rows.

    The regular itinerary output may contain meals and transport rows around the social
    mentions. Sorting the whole list by time here can silently put video point 2 before point
    1 when the model used a different slot. Insert only relative to social rows so the source
    order remains authoritative while the surrounding itinerary structure stays intact.
    """
    day = int(item.get("dayNumber") or 0)
    sequence = int(item.get("_socialSequence") or 10**9)
    same_day = [
        (index, existing)
        for index, existing in enumerate(items)
        if int(existing.get("dayNumber") or 0) == day
        and (existing.get("sourceSocialCandidateRef") or existing.get("_socialSequence"))
    ]
    for index, existing in same_day:
        existing_sequence = int(existing.get("_socialSequence") or 10**9)
        if existing_sequence > sequence:
            items.insert(index, item)
            return
    if same_day:
        items.insert(same_day[-1][0] + 1, item)
        return
    for index, existing in enumerate(items):
        if int(existing.get("dayNumber") or 0) > day:
            items.insert(index, item)
            return
    items.append(item)
