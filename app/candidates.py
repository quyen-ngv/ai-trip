"""Candidate normalisation, compaction for prompts, buckets and capacity planning."""
from __future__ import annotations
import math, re
from datetime import date, timedelta
from typing import Any
from .geo import hours_summary

FOOD = "FOOD_AND_DRINK"
ATTRACTIONS = "ATTRACTIONS"
ACTIVITY_GROUPS = [ATTRACTIONS]
ALL_GROUPS = [FOOD] + ACTIVITY_GROUPS

# Activity groups before backend V219 folded them into ATTRACTIONS. Old app requests and stored
# jobs still carry them; their meaning now lives in the place sub-types.
LEGACY_ACTIVITY_GROUPS = frozenset({"CULTURE_AND_HERITAGE", "NATURE_AND_OUTDOORS", "SHOPPING_AND_MARKET", "OTHER"})

# ATTRACTIONS does the work the four old activity groups each did, so its per-group budgets
# (catalogue rows, research queries, curated web rows) are this many times the base.
ATTRACTIONS_SHARE = 3

# Sub-type families that keep a day varied now that every sight shares one group.
_KIND_FAMILIES = (
    ("OUTDOOR", {"NATURE", "BEACH", "PARK", "VIEWPOINT"}),
    ("CULTURE", {"SPIRITUAL", "MUSEUM", "HERITAGE", "LANDMARK"}),
    ("SHOPPING", {"SHOPPING", "MARKET"}),
    ("FUN", {"ENTERTAINMENT", "THEME_PARK"}),
)


def normalize_group(value: Any) -> str:
    """Current group code for any code the worker may receive, legacy ones included."""
    group = str(value or "").strip().upper()
    if not group or group in LEGACY_ACTIVITY_GROUPS:
        return ATTRACTIONS
    if group == "FOOD":
        return FOOD
    return group


def group_budget(group: str, base: int) -> int:
    return base * ATTRACTIONS_SHARE if group == ATTRACTIONS else base


def diversity_key(row: dict) -> str:
    """What a stop counts as when spreading a day: its group, or for sights the sub-type family."""
    group = row.get("placeGroup") or ATTRACTIONS
    if group != ATTRACTIONS:
        return group
    kinds = set(row.get("subTypes") or [])
    for family, members in _KIND_FAMILIES:
        if kinds & members:
            return family
    return ATTRACTIONS

# Stops per full day, excluding the three meals. Windows are minutes from midnight.
PACE_PLAN = {
    "RELAXED": {"activities": 2, "start": 8 * 60 + 30, "end": 21 * 60 + 30, "meal": 90, "visit": 105},
    "BALANCED": {"activities": 3, "start": 8 * 60, "end": 22 * 60, "meal": 75, "visit": 90},
    "EAGER": {"activities": 4, "start": 7 * 60, "end": 23 * 60, "meal": 60, "visit": 75},
}

_CAFE = re.compile(r"caf[eé]|coffee|tea|bakery|dessert|ice ?cream|kem|trà|bánh ngọt|chè", re.I)
_BAR = re.compile(r"\bbar\b|pub|club|beer|bia|lounge|nightlife", re.I)


def pace_plan(pace: str | None) -> dict:
    return PACE_PLAN.get((pace or "BALANCED").upper(), PACE_PLAN["BALANCED"])


def destination_days(d: dict) -> int:
    return (date.fromisoformat(d["endDate"]) - date.fromisoformat(d["startDate"])).days + 1


def destination_dates(d: dict) -> list[date]:
    start = date.fromisoformat(d["startDate"])
    return [start + timedelta(days=i) for i in range(destination_days(d))]


def row_key(row: dict) -> str:
    """Return the stable planner key without confusing a social mention with a POI id.

    A social candidate may have no GoRoute place id and no Google place id yet.  Its
    candidateRef is still a real, stable identity for planning and must not collapse all
    unresolved mentions into one empty key.
    """
    # A social option can intentionally resolve to the same GoRoute/Google place as another
    # option. Its candidateRef is the source identity and must win over the resolved POI id.
    if row.get("source") == "social" or row.get("socialJobId"):
        return str(row.get("candidateId") or row.get("candidateRef") or row.get("id")
                   or row.get("googlePlaceId") or "")
    return str(row.get("id") or row.get("candidateId") or row.get("candidateRef")
               or row.get("googlePlaceId") or "")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def flatten_attributes(attrs: Any, limit: int = 160) -> str:
    if not isinstance(attrs, dict):
        return ""
    parts = []
    for k, v in attrs.items():
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        if isinstance(v, bool):
            if v:
                parts.append(str(k))
        elif isinstance(v, (str, int, float)) and str(v).strip():
            parts.append(f"{k}={v}")
        elif isinstance(v, list) and v:
            parts.append(f"{k}={','.join(str(x) for x in v[:4])}")
    return ", ".join(parts)[:limit]


def normalize(row: dict, source: str = "db") -> dict:
    """Map a Java `/candidates` row (or a research row) to the worker's internal shape."""
    group = normalize_group(row.get("placeGroup"))
    menu = row.get("menuHighlights") or []
    if not isinstance(menu, list):
        menu = []
    desc = (row.get("description") or "").strip()
    # A social row's description and tips are the only record of what the video says about the
    # stop, and they become the activity's text, so they are kept whole.
    desc_limit = 1200 if source == "social" else 300
    tips = [re.sub(r"\s+", " ", tip).strip()[:300] for tip in (row.get("tips") or [])
            if isinstance(tip, str) and tip.strip()][:8] if source == "social" else []
    return {
        "id": str(row["id"]) if row.get("id") else None,
        "candidateId": str(row.get("candidateId") or row.get("candidateRef") or "") or None,
        "googlePlaceId": row.get("googlePlaceId") or "",
        "title": (row.get("title") or "").strip(),
        "address": row.get("address") or "",
        "latitude": _num(row.get("latitude")),
        "longitude": _num(row.get("longitude")),
        "placeGroup": group,
        "category": row.get("category") or "",
        "subTypes": [str(kind).upper() for kind in (row.get("subTypes") or []) if kind],
        "reviewRating": _num(row.get("reviewRating")) or 0.0,
        "reviewCount": int(_num(row.get("reviewCount")) or 0),
        "score": _num(row.get("score")),
        "distanceKm": _num(row.get("distanceKm")),
        "visitDurationMinutes": int(_num(row.get("visitDurationMinutes")) or 0) or None,
        "description": re.sub(r"\s+", " ", desc)[:desc_limit],
        "tips": tips,
        "why": re.sub(r"\s+", " ", str(row.get("why") or "")).strip()[:220],
        "sourceUrls": [str(url)[:500] for url in (row.get("sourceUrls") or [])
                       if isinstance(url, str) and url.strip()][:3],
        "sourceTitles": [str(title)[:180] for title in (row.get("sourceTitles") or [])
                         if isinstance(title, str) and title.strip()][:3],
        "attributes": flatten_attributes(row.get("attributes")),
        "menuHighlights": [str(m)[:40] for m in menu[:8]],
        "openHours": row.get("openHours") if isinstance(row.get("openHours"), dict) else None,
        "source": source,
        "socialJobId": str(row.get("socialJobId")) if row.get("socialJobId") else None,
        "socialSequence": int(_num(row.get("socialSequence")) or 0) or None,
        "dayHint": int(_num(row.get("dayHint")) or 0) or None,
        "timeHint": row.get("timeHint") or None,
        "optionGroupId": row.get("optionGroupId") or None,
        "optionIndex": int(_num(row.get("optionIndex")) or 0) or None,
        "relation": row.get("relation") or None,
        "resolutionStatus": row.get("resolutionStatus") or None,
        "resolvedBy": row.get("resolvedBy") or None,
        "externalResolutionVerified": bool(row.get("externalResolutionVerified")),
        "isPinned": bool(row.get("isPinned")) or source == "social",
    }


def is_food(row: dict) -> bool:
    return row.get("placeGroup") == FOOD


def quality(row: dict) -> float:
    """0..1 from rating, review volume and the backend's overall score."""
    rating = (row.get("reviewRating") or 0) / 5.0
    reviews = min(1.0, math.log1p(row.get("reviewCount") or 0) / math.log1p(3000))
    q = 0.6 * rating + 0.4 * reviews
    score = row.get("score")
    if score is not None:
        # place_overall_score is on a 0..10 scale in the DB; clamp defensively
        q = 0.6 * q + 0.4 * max(0.0, min(1.0, score / 10.0))
    return q


def merge(*lists: list[dict]) -> list[dict]:
    seen_ids: set[str] = set()
    seen_google: set[str] = set()
    seen_candidates: set[str] = set()
    out: list[dict] = []
    for rows in lists:
        for r in rows:
            candidate_id = r.get("candidateId") or r.get("candidateRef")
            if candidate_id and candidate_id in seen_candidates:
                continue
            if r.get("source") == "social":
                # Source mentions remain distinct even when two options resolve to one POI.
                if candidate_id:
                    seen_candidates.add(candidate_id)
                if r.get("id"):
                    seen_ids.add(r["id"])
                out.append(r)
                continue
            if r.get("id") and r["id"] in seen_ids:
                continue
            if r.get("googlePlaceId") and r["googlePlaceId"] in seen_google:
                continue
            if r.get("id"):
                seen_ids.add(r["id"])
            if r.get("googlePlaceId"):
                seen_google.add(r["googlePlaceId"])
            if candidate_id:
                seen_candidates.add(candidate_id)
            out.append(r)
    return out


def bucket(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {g: [] for g in ALL_GROUPS}
    for r in rows:
        out.setdefault(r["placeGroup"], []).append(r)
    return out


def capacity(pace: str | None, days: int) -> dict:
    p = pace_plan(pace)
    return {"days": days, "meals": 3 * days, "activitiesPerDay": p["activities"], "activities": p["activities"] * days}


def shortage(rows: list[dict], pace: str | None, days: int, selected_groups: list[str]) -> dict[str, int]:
    """How many more candidates each group needs before planning is comfortable.

    Food: 2 candidates per meal. Activities: 2.5 per planned stop, spread over the groups the
    user asked for (all activity groups when they asked for none)."""
    cap = capacity(pace, days)
    groups = [g for g in selected_groups if g in ACTIVITY_GROUPS] or ACTIVITY_GROUPS
    b = bucket(rows)
    need: dict[str, int] = {}
    food_need = cap["meals"] * 2 - len(b[FOOD])
    if food_need > 0:
        need[FOOD] = food_need
    per_group = math.ceil(cap["activities"] * 2.5 / len(groups))
    for g in groups:
        missing = per_group - len(b.get(g, []))
        if missing > 0:
            need[g] = missing
    return need


def research_needs(rows: list[dict], pace: str | None, days: int, selected_groups: list[str],
                   minimum_per_group: int) -> dict[str, int]:
    """Return AI Web-research targets for one destination.

    Research is an enrichment step, not merely an empty-catalogue fallback: every routable
    destination searches food plus its requested activity groups. The capacity shortage still
    increases the target when the local catalogue is thin.
    """
    minimum = max(1, int(minimum_per_group))
    pace_name = (pace or "BALANCED").upper()
    if pace_name == "RELAXED":
        minimum = max(2, minimum - 1)
    elif pace_name == "EAGER":
        minimum += max(1, minimum // 2)
    selected = [g for g in selected_groups if g in ACTIVITY_GROUPS]
    groups = [FOOD, *(selected or ACTIVITY_GROUPS)]
    missing = shortage(rows, pace, days, selected_groups)
    return {group: max(group_budget(group, minimum), missing.get(group, 0)) for group in dict.fromkeys(groups)}


def compact_for_plan(r: dict, weekdays: list[str], ref: str) -> dict:
    """One candidate as the planner sees it. `ref` is a short alias ("f12", "a3") that the
    caller maps back to `row_key`: a UUID costs ~20 tokens on the way in and again in every
    output stop, and short refs are copied back more reliably."""
    out: dict[str, Any] = {"id": ref, "title": r["title"]}
    if not is_food(r):
        out["group"] = r["placeGroup"]
    if r.get("subTypes"):
        out["kinds"] = r["subTypes"]
    if r.get("category"):
        out["category"] = r["category"]
    if r.get("fit") is not None:
        out["fit"] = r["fit"]
    if r.get("why"):
        out["why"] = r["why"][:120]
    if r.get("meals"):
        out["meals"] = r["meals"]
    if r.get("reviewRating"):
        out["rating"] = round(r["reviewRating"], 1)
    if r.get("reviewCount"):
        out["reviews"] = r["reviewCount"]
    if r.get("latitude") is not None:
        out["lat"] = round(r["latitude"], 4)
        out["lng"] = round(r["longitude"], 4)
    if r.get("distanceKm") is not None:
        out["km"] = round(r["distanceKm"], 1)
    if r.get("visitDurationMinutes"):
        out["visitMin"] = r["visitDurationMinutes"]
    if r.get("menuHighlights"):
        out["menu"] = r["menuHighlights"][:5]
    if r.get("description"):
        out["desc"] = r["description"][:120]
    hours = hours_summary(r.get("openHours"), weekdays)
    if hours:
        out["hours"] = hours[:200]
    if r.get("source") == "web":
        # Citations stay on the row and reach the activity notes in code; the planner only
        # needs to know the stop is editorially recommended and whether it has a location.
        out["source"] = "web" if r.get("latitude") is not None else "web_unlocated"
    if r.get("source") == "social":
        out["source"] = "social_video"
        out["pinned"] = True
        if r.get("socialSequence") is not None:
            out["videoSequence"] = r["socialSequence"]
        if r.get("dayHint") is not None:
            out["videoDay"] = r["dayHint"]
        if r.get("timeHint"):
            out["videoTime"] = r["timeHint"]
        if r.get("optionGroupId"):
            out["optionGroupId"] = r["optionGroupId"]
        if r.get("optionIndex") is not None:
            out["optionIndex"] = r["optionIndex"]
        if r.get("relation"):
            out["relation"] = r["relation"]
        if r.get("resolutionStatus"):
            out["resolutionStatus"] = r["resolutionStatus"]
    return out


# First match wins: a beach park is a beach, a temple museum is spiritual.
_SUB_TYPE_CATEGORIES = (
    ("BEACH", "beach"),
    ("SPIRITUAL", "spiritual"),
    ("THEME_PARK", "entertainment"),
    ("ENTERTAINMENT", "entertainment"),
    ("NATURE", "nature"),
    ("PARK", "nature"),
    ("VIEWPOINT", "nature"),
    ("SHOPPING", "shopping"),
    ("MARKET", "shopping"),
)


def activity_category(r: dict) -> str:
    """Map a place to one of the categories the Flutter app is willing to store."""
    g = r.get("placeGroup") or ""
    text = f"{r.get('category', '')} {r.get('title', '')}".lower()
    if g == FOOD:
        return "restaurant"
    if g == "ACCOMMODATION":
        return "hotel"
    # Catalogue sub-types are curated, so they win over the group and the keyword guesses below.
    kinds = set(r.get("subTypes") or [])
    for kind, category in _SUB_TYPE_CATEGORIES:
        if kind in kinds:
            return category
    if g == "SHOPPING_AND_MARKET":
        return "shopping"
    if g == "NATURE_AND_OUTDOORS":
        return "beach" if re.search(r"beach|bãi biển|biển", text) else "nature"
    if g == "CULTURE_AND_HERITAGE":
        return "spiritual" if re.search(r"temple|pagoda|church|shrine|chùa|đền|nhà thờ|miếu", text) else "attraction"
    if _BAR.search(text) or re.search(r"amusement|theme park|cinema|theater|show|karaoke", text):
        return "entertainment"
    return "attraction"


def meal_hint(r: dict) -> list[str]:
    """Which meals a food venue plausibly serves, from its category when the LLM gave none."""
    text = f"{r.get('category', '')} {r.get('title', '')} {' '.join(r.get('menuHighlights') or [])}".lower()
    if _BAR.search(text):
        return ["dinner"]
    if _CAFE.search(text):
        return ["snack"]  # coffee/dessert is a break AFTER a meal, never a breakfast substitute
    if re.search(r"phở|pho|bánh mì|banh mi|xôi|bún|cháo|breakfast|bánh cuốn", text):
        return ["breakfast", "lunch"]
    return ["lunch", "dinner"]
