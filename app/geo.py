"""Geometry, travel-time estimates and Google-Maps opening-hours parsing."""
from __future__ import annotations
import math, re
from typing import Any

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def haversine_km(a: dict, b: dict) -> float:
    try:
        la1, lo1 = math.radians(float(a["latitude"])), math.radians(float(a["longitude"]))
        la2, lo2 = math.radians(float(b["latitude"])), math.radians(float(b["longitude"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    dp, dl = la2 - la1, lo2 - lo1
    q = math.sin(dp / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(q), math.sqrt(1 - q))


def has_coords(p: dict | None) -> bool:
    return bool(p) and p.get("latitude") is not None and p.get("longitude") is not None


def travel_estimate(km: float) -> tuple[str, int]:
    """(transportMode, minutes) for an in-city hop. Walk short hops, taxi otherwise."""
    if km < 1.2:
        return "WALKING", max(5, int(round(km / 4.5 * 60)) + 3)
    return "TAXI", max(8, int(round(km / 22 * 60)) + 6)


def intercity_mode(km: float) -> str:
    if km > 600:
        return "PLANE"
    if km > 150:
        return "TRAIN"
    return "CAR"


# ---------------------------------------------------------------- opening hours
# Rules (see memory "open hours format rules"): Google's typographic spaces, elided
# meridiem, overnight windows, "Open 24 hours"/"Closed" literals, trailing "" padding,
# and unparseable == unknown (None), never closed.
_TIME = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)
_SEP = re.compile(r"\s*(?:-|–|—|\bto\b)\s*", re.I)


def _minutes(hour: int, minute: int, meridiem: str | None) -> int | None:
    if hour > 24 or minute > 59:
        return None
    if meridiem:
        m = meridiem.lower().replace(".", "")
        if hour == 12:
            hour = 0
        if m == "pm":
            hour += 12
    return hour * 60 + minute


def parse_hours_entry(entry: str) -> list[tuple[int, int]] | None:
    """One display string -> list of (start,end) minute windows. end>1440 means overnight.
    [] = closed. None = cannot parse."""
    text = re.sub(r"\s+", " ", (entry or "")).strip()
    if not text:
        return None
    low = text.lower()
    if "24 hours" in low or "24 giờ" in low or low.startswith("open 24"):
        return [(0, 1440)]
    if low in {"closed", "đóng cửa"} or low.startswith("closed"):
        return []
    windows: list[tuple[int, int]] = []
    for chunk in re.split(r",\s*", text):
        parts = _SEP.split(chunk)
        if len(parts) != 2:
            return None
        m1, m2 = _TIME.fullmatch(parts[0].strip()), _TIME.fullmatch(parts[1].strip())
        if not m1 or not m2:
            return None
        mer1, mer2 = m1.group(3), m2.group(3)
        # elided meridiem borrows the other side's
        mer1 = mer1 or mer2
        mer2 = mer2 or mer1
        s = _minutes(int(m1.group(1)), int(m1.group(2) or 0), mer1)
        e = _minutes(int(m2.group(1)), int(m2.group(2) or 0), mer2)
        if s is None or e is None:
            return None
        # "6 - 9:30 PM" with borrowed PM gives 18:00-21:30; "11 AM - 2 AM" -> overnight
        if e <= s:
            e += 1440
        windows.append((s, e))
    return windows or None


def _entries(raw: Any) -> list[str]:
    """A day's display strings, cleaned. Google Maps pads some days with an icon glyph from the
    Private Use Area (seen: U+E14D) as an extra entry; it is not whitespace, so it used to make
    the whole day unparseable (196 of ~8,600 weekday lists in a prod sample, 2026-10-02)."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    cleaned = (re.sub(r"\s+", " ", re.sub(r"[-]", "", e)).strip() for e in raw if isinstance(e, str))
    return [e for e in cleaned if e]


def open_windows(open_hours: Any, weekday: str) -> list[tuple[int, int]] | None:
    """Windows for a weekday name ('Monday'). None when hours are unknown/unparseable."""
    if not isinstance(open_hours, dict):
        return None
    entries = open_hours.get(weekday)
    if entries is None:
        entries = open_hours.get(weekday.lower()) or open_hours.get(weekday[:3])
    entries = _entries(entries)
    if not entries:
        return None
    windows: list[tuple[int, int]] = []
    for e in entries:
        w = parse_hours_entry(e)
        if w is None:
            return None
        windows.extend(w)
    return windows


def is_open(open_hours: Any, weekday: str, start: int, end: int) -> bool | None:
    """True/False when hours are known for the weekday, None when unknown."""
    windows = open_windows(open_hours, weekday)
    if windows is None:
        return None
    for s, e in windows:
        if s <= start and end <= e:
            return True
    return False


def next_opening(open_hours: Any, weekday: str, after: int) -> int | None:
    windows = open_windows(open_hours, weekday)
    if not windows:
        return None
    starts = sorted(s for s, _ in windows if s >= after)
    return starts[0] if starts else None


def _hhmm(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _day_hours(entries: Any) -> str:
    """One weekday's entries as 24h windows ("07:00-22:00,17:00-02:00"); raw text if unparseable."""
    entries = _entries(entries)
    if not entries:
        return ""
    windows: list[tuple[int, int]] = []
    for entry in entries:
        parsed = parse_hours_entry(entry)
        if parsed is None:
            return "; ".join(entries)
        windows += parsed
    if not windows:
        return "closed"
    if windows == [(0, 1440)]:
        return "24h"
    return ",".join(f"{_hhmm(s)}-{_hhmm(e)}" for s, e in windows)


def hours_summary(open_hours: Any, weekdays: list[str]) -> str:
    """Compact, model-readable summary limited to the trip's weekdays.

    Each weekday appears once even on long trips, and days sharing the same hours are merged,
    so a venue open the same hours all week costs a few tokens: "daily 07:00-22:00"."""
    if not isinstance(open_hours, dict):
        return ""
    by_value: dict[str, list[str]] = {}
    for wd in dict.fromkeys(weekdays):
        value = _day_hours(open_hours.get(wd))
        if value:
            by_value.setdefault(value, []).append(wd[:3])
    if len(by_value) == 1 and len(next(iter(by_value.values()))) == len(set(weekdays)):
        return "daily " + next(iter(by_value))
    return " | ".join(f"{','.join(days)} {value}" for value, days in by_value.items())


def fmt_time(minutes: int) -> str:
    minutes = max(0, minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"
