"""Prompt assembly for the single itinerary call.

The AI owns the plan: which places, which day, what time, in what order, rest blocks, and the
wording. Code only supplies the traveller profile, the candidate list, the rules text and the
output contract, then validates the result against what the Java commit endpoint accepts.

Rules live in the `config` table (label AI_TRIP) so product can tune them without a deploy;
the DEFAULT_* constants below are the fallback AND the seed for that table
(see goroute migration V148__ai_trip_prompt_config.sql — keep them in sync).

Field values must stay in sync with goroute_fe `create_ai_trip_screen.dart`:
placeGroups, pace (RELAXED|BALANCED|EAGER), discoveryStyle (FAMOUS|BALANCED|HIDDEN_GEMS),
groupComposition (free string: "Solo traveler", "Couple (2 adults)", "Family with children",
"Family with elderly members", "Friends group"), dietaryRestrictions, mobilityConsiderations
("Kid-friendly", "Elderly-friendly"), activityTypes (Adventure, Photography).
"""
from __future__ import annotations
import re
from typing import Any

CONFIG_LABEL = "AI_TRIP"
KEY_PLAN_RULES = "PLAN_RULES"
KEY_TRAVELLER_RULES = "TRAVELLER_RULES"
KEY_DESCRIPTION_MAX_CHARS = "DESCRIPTION_MAX_CHARS"

PLACE_GROUPS = {
    "FOOD_AND_DRINK": "restaurants, street food, cafes, local specialties",
    "CULTURE_AND_HERITAGE": "museums, temples, historical sites, galleries, monuments",
    "NATURE_AND_OUTDOORS": "parks, beaches, mountains, gardens, viewpoints, waterfalls",
    "SHOPPING_AND_MARKET": "markets, night markets, shopping streets, craft shops",
    "ATTRACTIONS": "landmarks, theme parks, entertainment venues, famous spots",
}
DIETARY = {
    "vegetarian": "no meat, poultry or seafood", "vegan": "no animal products", "halal": "halal only, no pork, no alcohol",
    "no_pork": "no pork", "pescatarian": "no meat or poultry, seafood ok", "beef_free": "no beef", "jain": "strict vegetarian, no root vegetables",
}
LANG = {"vi": "Vietnamese", "en": "English"}

# ----------------------------------------------------------------------------- default rules
DEFAULT_PLAN_RULES = """\
## Core planning rules
1. Use ONLY candidate ids from the input. Never invent a place. Never use a place twice in the whole trip.
2. Every day: BREAKFAST (06:30-09:00), LUNCH (11:30-13:30), DINNER (18:00-20:30) — each at a real food venue.
   - Breakfast = real breakfast food (phở, bún, bánh mì, xôi, cháo, bánh cuốn, cơm tấm...). NEVER a cafe/coffee/dessert place — nobody drinks coffee on an empty stomach.
   - A cafe/dessert stop is a break AFTER a meal, at most one per day. Late-night food (ăn đêm, chè, night-market snacks) after dinner is welcome, especially for food lovers.
   - Keep >= 3 hours between main meals. Vietnamese breakfast places sell out by ~09:30; many kitchens close 14:00-17:00.
3. Geography: plan each half-day inside ONE area; consecutive stops must be close (use lat/lng). Never zig-zag across the city, never cross the city more than once a day. Budget realistic travel time between stops: walking ~4 km/h for < 1.5 km, otherwise taxi/Grab ~20 km/h in the city (~14 km/h in rush hours 07:00-09:00 and 16:00-18:00), plus 5-10 minutes to find a ride.
4. Time-of-day fit (respect opening hours first when given):
   - Temples/pagodas 06:30-09:30 or 16:00-17:30 · Museums/galleries daytime, ideal 11:00-15:00 as a heat refuge · Wet/morning markets 06:00-09:00
   - Night markets, bars, rooftops, walking streets, shows: after 17:30 only · Beaches 06:00-09:00 or 16:00-18:30 · Viewpoints at sunrise or 60 min before sunset (sunset ~17:15 in Dec, ~18:30 in Jun)
   - Waterfalls/nature/boat trips in the morning · Cable cars/mountain parks at opening or after 16:00 (clouds/rain after 10:00) · Theme parks: full day, arrive at opening
   - Heat (Mar-Sep north/central, all year south): no long outdoor stop 11:00-15:00; put indoor/AC, lunch, cafe or a hotel rest there.
5. Visit lengths: meal 45-90 min; temple 30-60; museum 90-150; market 45-90; beach 90-180; big park/cable car/theme park 240-420. Add a 10-15 min buffer after every stop.
6. Rest blocks (no place) are allowed and encouraged when the traveller profile calls for them: {"rest":"hotel"} midday rest, {"rest":"walk"} a stroll in the neighbourhood, {"rest":"free"} free time.
7. Never output transport items — travel is handled outside the plan.
8. Discovery style: FAMOUS = mostly high-review-count landmarks; BALANCED = famous anchors + local favourites; HIDDEN_GEMS = local favourites with high ratings, one famous landmark per day for orientation.
9. Cover the traveller's chosen interests across the trip (not all on one day); vary categories within a day (no three temples in a row); the user's own words override every default.
10. Descriptions: ONE concise sentence per stop (why go + what to do/eat there), grounded in the given data (a menu dish, the view, the vibe). Trip description: at most 2 short sentences. No marketing fluff, no invented facts.
11. When social-video candidates are present, preserve their videoSequence order above every optimisation. Keep every social candidate in the itinerary, including unresolved candidates as ACTIVITY rows. If several rows share optionGroupId, output every row separately with the same slot/time; never merge or discard an option.
"""

DEFAULT_TRAVELLER_RULES = """\
## Rules by traveller profile (apply every matching block)
### Pace
- RELAXED: 2 non-meal stops/day, start 08:30, leisurely meals (90 min), a rest or free block midday, end by 21:30. Evenings still welcome (sunset, night market).
- BALANCED: 3 non-meal stops/day, start 08:00, standard visits, end ~22:00. An evening activity is welcome.
- EAGER: 4-5 non-meal stops/day, can start 06:00-07:00 (sunrise, morning markets) and end 23:00 (nightlife). Efficient transitions, quick meals.
### Family with children (mobilityConsiderations contains Kid-friendly, or the group mentions children/kids/baby)
- Max 3 stops/day, each <= 2 h; hotel rest 12:00-15:00 in hot months; dinner by 18:30; nothing after 20:00; walking <= 3 km/day.
- NO adventure, extreme, trekking, cliffs, long boat rides, bars, nightlife or steep-stairs sites (e.g. Hang Múa, Fansipan summit steps, long hikes). Prefer parks, beaches at cool hours, interactive museums, water puppets, cable cars, gentle boats.
- Babies/toddlers (0-3): 2 outdoor stops/day max, a nap block at the hotel is mandatory, stroller-friendly areas only.
### Elderly / limited mobility (Elderly-friendly, or the group mentions elderly/parents/grandparents)
- 2 stops/day with seating; <= 1 km walking per stop; midday rest 12:00-15:00; end by 19:30; door-to-door transport assumed.
- Avoid steps-only sites, steep climbs, long standing, rough terrain, motorbike loops, sleeper buses. Seated sightseeing is ideal: cruises, rowing boats in the morning, cyclo, cable cars (stay at the station).
### Couple / honeymoon
- 2 stops + 1 "moment" per day: golden-hour viewpoint, riverside or beach sunset, a good dinner around 19:30, a spa/free block 13:00-16:00. Quieter alternatives over crowded spots; photo stops early morning (empty streets 06:30-08:00).
### Friends group
- Social dining, nightlife and photo spots welcome; next-day first stop >= 10:30 after a bar night; no dawn activities the morning after.
### Solo
- Walkable areas, social cafes, safe neighbourhoods, easy public transport; bars/nightlife fine.
### Adventure / trekking (activityTypes contains Adventure, or the user asks for trekking/hiking)
- Start 06:00-07:30, finish all trekking/riding by 16:30; 20-30% time buffer; rest day after a big trek; never combine a trek with an intercity move on the same day. Skip entirely when children or elderly travel.
### Photography
- Put viewpoints/landmarks at sunrise or golden hour; markets and old streets before 08:00.
### Food lovers (FOOD_AND_DRINK selected or the user talks about eating)
- Named breakfast venue before 08:30, regional must-eat dishes, one evening food crawl (17:30-20:30) or late-night snack; use the menu highlights given.
### Dietary restrictions
- Hard constraint on every food venue. Vegetarian/vegan: prefer quán chay; halal: only venues that read as halal/seafood-safe; warn via description when a venue is uncertain.
### Arrival / departure
- First day of a destination after a move: start later (nothing before ~10:30 after a morning drive; a light day after an overnight train/bus). Last day: morning only if a move follows.
"""

DEFAULT_DESCRIPTION_MAX_CHARS = 200


# ----------------------------------------------------------------------------- helpers
def language(locale: str | None) -> str:
    return LANG["vi"] if (locale or "").lower().startswith("vi") else LANG["en"]


def preferences_block(req: dict[str, Any], locale: str | None) -> str:
    lines = ["## Traveller profile"]
    groups = [g for g in (req.get("placeGroups") or []) if g in PLACE_GROUPS]
    lines.append("Interests: " + ("; ".join(f"{g} ({PLACE_GROUPS[g]})" for g in groups) if groups
                                  else "not specified — cover food, culture, nature and attractions"))
    lines.append(f"Pace: {(req.get('pace') or 'BALANCED').upper()}")
    if req.get("groupComposition"):
        lines.append(f"Group: {req['groupComposition']}")
    lines.append(f"Discovery style: {(req.get('discoveryStyle') or 'BALANCED').upper()}")
    diet = req.get("dietaryRestrictions") or []
    if diet:
        lines.append("DIETARY (hard constraint): " + "; ".join(f"{d}: {DIETARY.get(d, d)}" for d in diet))
    mob = req.get("mobilityConsiderations") or []
    if mob:
        lines.append("Mobility (hard constraint): " + ", ".join(mob))
    acts = req.get("activityTypes") or []
    if acts:
        lines.append("Activity types wanted: " + ", ".join(acts))
    if req.get("budgetMax"):
        lines.append(f"Budget for the whole trip: {req.get('budgetMin') or 0:,.0f}–{req['budgetMax']:,.0f} {req.get('budgetCurrency') or ''}".strip())
    text = (req.get("preferenceText") or "").strip()
    if text:
        lines.append("USER'S OWN WORDS (highest priority, override defaults when they conflict):\n> " + re.sub(r"\s+", " ", text))
    return "\n".join(lines)


def build_itinerary_prompt(req: dict[str, Any], locale: str | None, destination: str, days: int,
                           day_numbers: list[int], weekdays: list[str], cfg: dict[str, str],
                           first_day_note: str = "", last_day_note: str = "") -> str:
    """System prompt for the single planning call of one destination."""
    plan_rules = cfg.get(KEY_PLAN_RULES) or DEFAULT_PLAN_RULES
    traveller_rules = cfg.get(KEY_TRAVELLER_RULES) or DEFAULT_TRAVELLER_RULES
    max_chars = int(cfg.get(KEY_DESCRIPTION_MAX_CHARS) or DEFAULT_DESCRIPTION_MAX_CHARS)
    day_list = ", ".join(f"day {n} ({wd})" for n, wd in zip(day_numbers, weekdays))
    extra = ""
    if first_day_note:
        extra += f"\n- Day {day_numbers[0]}: {first_day_note}"
    if last_day_note:
        extra += f"\n- Day {day_numbers[-1]}: {last_day_note}"
    social = req.get("socialContext") if isinstance(req.get("socialContext"), dict) else None
    if social:
        extra += "\n- SOCIAL VIDEO PRIORITY: candidate order follows the source video. Preserve every candidate, and output options as separate stops sharing one time window. A candidateId is stable even when placeId is absent."
        if social.get("durationBasis"):
            extra += f"\n- Social duration decision: {social['durationBasis']}"
    return f"""You are TripMind, an expert Vietnam travel planner. You produce the COMPLETE itinerary for {destination}: which places, which day, what time, in what order, plus the wording. Names may be Vietnamese.

{preferences_block(req, locale)}

{plan_rules}
{traveller_rules}

## Input
JSON {{"destination": "...", "days": [...], "food": [...], "activities": [...]}}. Each candidate: id, candidateId, title, group, category, rating, reviews, km (from centre), lat, lng, visitMin, menu, hours (for the trip's weekdays; may be missing), desc, note. Social candidates may additionally include videoSequence, videoDay, videoTime, optionGroupId, optionIndex, relation, resolutionStatus.

## Task
Plan {days} day(s): {day_list}.{extra}
If this is a social-video itinerary, videoSequence is the strongest ordering signal. Do not reorder solely for rating, geography, meals, or opening hours. Keep alternative options as distinct stops (same optionGroupId and same time); the user can choose later.
Fill every day from morning to evening according to the traveller profile. Times are local "HH:MM", chronological within a day, no overlaps, with realistic travel gaps between stops.
Write all free text in {language(locale)}. Each stop description <= {max_chars} characters, one sentence. tripDescription <= 2 short sentences.

## Output (JSON only) — exactly {days} entries in "days", in the listed order
{{"tripDescription":"...","days":[{{"dayNumber":{day_numbers[0]},"theme":"...","stops":[
  {{"candidateId":"<candidateId>","placeId":"<id-or-omit-if-unresolved>","start":"07:30","end":"08:15","description":"..."}},
  {{"placeId":"<id>","start":"08:30","end":"10:00","description":"..."}},
  {{"rest":"hotel","start":"12:30","end":"14:30","description":"..."}}
]}}]}}"""
