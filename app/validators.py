"""Deterministic itinerary checks and the final shape Java's commit endpoint accepts."""
from __future__ import annotations
from typing import Any
from .candidates import pace_plan

MEAL_SLOTS = ("BREAKFAST", "LUNCH", "DINNER")


def _minutes(t: str | None) -> int:
    if not t:
        return 0
    h, m = t.split(":")[:2]
    return int(h) * 60 + int(m)


def _is_social_item(item: dict) -> bool:
    return bool(item.get("sourceSocialCandidateRef") or item.get("_socialSequence"))


def _social_sequence(item: dict, fallback: int) -> tuple[int, int]:
    try:
        sequence = int(item.get("_socialSequence") or 0)
    except (TypeError, ValueError):
        sequence = 0
    return (sequence if sequence > 0 else 10**9, fallback)


def _enforce_social_order(items: list[dict]) -> list[dict]:
    """Keep source-video activities monotonic even if the model assigns bad days.

    The model normally returns ordered days, but source order is a stronger product invariant
    than a malformed day/slot choice. Move an out-of-order social row forward (and keep an
    option group on the first option's day), then replace only social positions within each day
    so meals and other planner rows keep their surrounding structure.
    """
    social_indices = [index for index, item in enumerate(items) if _is_social_item(item)]
    if not social_indices:
        return items

    ordered_indices = sorted(
        social_indices,
        key=lambda index: _social_sequence(items[index], index),
    )
    previous_day = 0
    option_days: dict[str, int] = {}
    for index in ordered_indices:
        item = dict(items[index])
        day = int(item["dayNumber"])
        option_group = str(item.get("optionGroupId") or "").strip()
        if option_group and option_group in option_days:
            day = option_days[option_group]
        else:
            day = max(day, previous_day)
            if option_group:
                option_days[option_group] = day
        if day != item["dayNumber"]:
            item["dayNumber"] = day
        items[index] = item
        previous_day = max(previous_day, day)

    days = {int(items[index]["dayNumber"]) for index in social_indices}
    for day in days:
        positions = [
            index for index, item in enumerate(items)
            if int(item["dayNumber"]) == day and _is_social_item(item)
        ]
        social_rows = sorted(
            (items[index] for index in positions),
            key=lambda item: _social_sequence(item, 0),
        )
        for index, item in zip(positions, social_rows):
            items[index] = item
    return items


def check(items: list[dict], pace_name: str | None, total_days: int) -> list[str]:
    """Human-readable violations. Used for logging and for the retry decision."""
    pace = pace_plan(pace_name)
    out: list[str] = []
    seen: set[str] = set()
    for n in range(1, total_days + 1):
        day = [i for i in items if i.get("dayNumber") == n and i.get("type") != "TRANSPORT"]
        if not day:
            out.append(f"day {n}: empty")
            continue
        slots = [i.get("_slot") for i in day]
        for meal in MEAL_SLOTS:
            if meal not in slots:
                out.append(f"day {n}: missing {meal.lower()}")
        acts = sum(1 for s in slots if s not in MEAL_SLOTS)
        if acts < max(1, pace["activities"] - 1):
            out.append(f"day {n}: only {acts} non-meal stops for pace")
        first, last = _minutes(day[0].get("startTime")), _minutes(day[-1].get("endTime"))
        if last < 18 * 60 and last - first < 8 * 60:
            out.append(f"day {n}: ends at {day[-1].get('endTime')}, too short")
        for i in day:
            key = i.get("_placeKey") or i.get("placeId")
            if key and key in seen:
                out.append(f"day {n}: {i.get('name')} appears twice")
            if key:
                seen.add(key)
    return out


def finalize(items: list[dict], total_days: int) -> list[dict]:
    """Enforce what AiTripGenerationService.validateItems checks, in the order Java sees rows:
    day in range, start<end (unless overnight), no overlap within a day, PLACE_VISIT has placeId."""
    clean: list[dict] = []
    for i in items:
        if not (1 <= int(i.get("dayNumber", 0)) <= total_days):
            continue
        if i.get("type") == "PLACE_VISIT" and not i.get("placeId"):
            continue
        if i.get("endDayNumber") and int(i["endDayNumber"]) > total_days:
            i = {**i, "endDayNumber": None, "endTime": "23:59:00"}
        clean.append(i)
    if any(_is_social_item(i) for i in clean):
        # The source-video order is a product requirement. Materialize/scheduler already place
        # rows in that order; repair malformed day assignments before grouping instead of
        # re-sorting by map time.
        clean = _enforce_social_order(clean)
        clean = [item for _, item in sorted(enumerate(clean), key=lambda pair: (pair[1]["dayNumber"], pair[0]))]
    else:
        clean.sort(key=lambda x: (x["dayNumber"], _minutes(x.get("startTime")), 0 if x.get("type") == "TRANSPORT" else 1))
    out: list[dict] = []
    last_end: dict[int, int] = {}
    option_windows: dict[tuple[int, str], tuple[int, int]] = {}
    for i in clean:
        d = i["dayNumber"]
        s, e = _minutes(i.get("startTime")), _minutes(i.get("endTime"))
        overnight = bool(i.get("endDayNumber"))
        social = bool(i.get("sourceSocialCandidateRef") or i.get("_socialSequence"))
        option_group = str(i.get("optionGroupId") or "").strip()
        option_window = option_windows.get((d, option_group)) if option_group else None
        if option_window is not None:
            # Alternatives are separate activities but intentionally occupy one choice window.
            s, e = option_window
        else:
            prev = last_end.get(d, 0)
            if s < prev:  # overlap: push forward, keep the duration
                dur = (e - s) if e > s else 30
                s = prev
                e = s + dur if not overnight else e
            if option_group:
                option_windows[(d, option_group)] = (s, e)
        if not overnight and e <= s:
            e = s + 30
        if s >= 24 * 60 or (not overnight and e > 24 * 60):
            if social:
                # A video may contain more named points than fit into the AI-selected duration.
                # Keep the point as an unscheduled ACTIVITY instead of losing source data. Java
                # accepts nullable times for activities; the user can place it later in the trip.
                unscheduled = {**i}
                unscheduled.pop("startTime", None)
                unscheduled.pop("endTime", None)
                unscheduled.pop("endDayNumber", None)
                out.append(unscheduled)
            continue
        i = {**i, "startTime": f"{s // 60:02d}:{s % 60:02d}:00"}
        if not overnight:
            i["endTime"] = f"{e // 60:02d}:{e % 60:02d}:00"
        last_end[d] = max(last_end.get(d, 0), e if not overnight else 24 * 60)
        out.append(i)
    for n, i in enumerate(out):
        i["sortOrder"] = n
    return out


def strip_internal(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in i.items() if not k.startswith("_") and v is not None} for i in items]
