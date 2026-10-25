"""Prompt assembly for the single itinerary call.

The AI owns the plan: which places, which day, what time, in what order, rest blocks, and the
wording. Code only supplies the traveller profile, the candidate list, the rules text and the
output contract, then validates the result against what the Java commit endpoint accepts.

Rules live in the `config` table (label AI_TRIP) so product can tune them without a deploy;
the DEFAULT_* constants below are the fallback AND the seed for that table
(see goroute migrations V148__ai_trip_prompt_config.sql, V229, V235 — keep them in sync).

2026-10-23, after a Mộc Châu plan read like a list of addresses: rules 12 (season first) and 13 (a
restaurant or club is never a sight), each day one loop out of the base with the farthest sight first,
signature dishes across the meals, stop guides written like Vietnamese bloggers' trip reports (why this
hour, what to order, famous entry fees as approximate figures, no filler), and a trip overview of 3-5
practical lines instead of 2 sentences. Research in .docs/AI_TRIP_PLANNING_KNOWLEDGE.md §F.

Prompt layout (2026-10-13 rewrite, see .docs/AI_TRIP_PLANNING_KNOWLEDGE.md §E):
- everything static first (identity, rules, input legend, output contract) so providers with
  prefix caching bill it as cached input; the traveller profile and the task come last;
- rules are phrased as what to do, grouped under short headers, about 45 atomic constraints
  in total (instruction following degrades past ~50-100 for frontier models);
- the planner names the area of each half-day before its stops (geographic cohesion is the
  dimension LLM planners fail most), and the prompt ends with a six-line self-check that
  repeats the hard constraints (recency).

Field values must stay in sync with goroute_fe `create_ai_trip_screen.dart`:
placeGroups, pace (RELAXED|BALANCED|EAGER), discoveryStyle (FAMOUS|BALANCED|HIDDEN_GEMS),
groupComposition (free string: "Solo traveler", "Couple (2 adults)", "Family with children",
"Family with elderly members", "Friends group"), dietaryRestrictions, mobilityConsiderations
("Kid-friendly", "Elderly-friendly"), activityTypes (Adventure, Photography).
"""
from __future__ import annotations
import re
from typing import Any

KEY_PLAN_RULES = "PLAN_RULES"
KEY_TRAVELLER_RULES = "TRAVELLER_RULES"
KEY_DESCRIPTION_MAX_CHARS = "DESCRIPTION_MAX_CHARS"

PLACE_GROUPS = {
    "FOOD_AND_DRINK": "restaurants, street food, cafes, local specialties",
    "ATTRACTIONS": "sights of every kind: museums, temples, historic sites, parks, beaches, viewpoints, markets, shopping, landmarks, entertainment",
}
DIETARY = {
    "vegetarian": "no meat, poultry or seafood", "vegan": "no animal products", "halal": "halal only, no pork, no alcohol",
    "no_pork": "no pork", "pescatarian": "no meat or poultry, seafood ok", "beef_free": "no beef", "jain": "strict vegetarian, no root vegetables",
}
LANG = {"vi": "Vietnamese", "en": "English"}

# ----------------------------------------------------------------------------- default rules
DEFAULT_PLAN_RULES = """\
## Core planning rules
1. Places: use only candidate ids from the input, each at most once in the whole trip. Every stop is a real candidate or a rest block.
2. Meals, every day, each at a food candidate: BREAKFAST 06:30-09:00, LUNCH 11:30-13:30, DINNER 18:00-20:30, at least 3 hours apart.
   - Breakfast is breakfast food (phở, bún, bánh mì, xôi, cháo, bánh cuốn, cơm tấm...) at a venue tagged breakfast or clearly serving it. A cafe or dessert place is a break after a meal (at most one per day), not a meal.
   - Street breakfast sells out by ~09:30 and many kitchens pause 14:00-17:00; after 13:30 choose a restaurant, not a stall.
   - Evenings in Vietnam continue after dinner: a night market, walking street, riverside or late-night snack (ăn đêm, chè) is the default evening block when the profile allows it.
   - Across the trip the meals carry the destination's signature dishes, a different one at each main meal when the candidates allow (Mộc Châu: bê chao, cá suối nướng, xôi nương, lẩu cá hồi; Hà Nội: phở, bún chả, bánh cuốn). Lunch is near the morning's last stop; dinner is near the base.
3. Geography: give each half-day ONE area and write it in the day's "area" field before the stops; consecutive stops are neighbours (use lat/lng), and the day crosses the city at most once. Run each day as one loop out of the base (the town centre where travellers sleep): the farthest sight goes first, in the morning, and the day works its way back towards the base. In a spread-out mountain or countryside destination (Mộc Châu, Sa Pa, Hà Giang, Đà Lạt's outskirts) that means one direction per day, never out to one side and across to the other in the same afternoon. Travel time between stops: walking ~4 km/h up to 1.5 km; in a city taxi/Grab ~20 km/h (~14 km/h in rush hours 07:00-09:00 and 16:00-18:00) plus 5-10 minutes to find a ride; on mountain and country roads ~35 km/h by car or motorbike. Leave that gap between a stop's end and the next start.
4. Opening hours first: arrive after opening, leave at least 30 minutes before closing; missing hours mean unknown, so keep such stops in mainstream daytime hours. Then time-of-day fit:
   - Temples/pagodas at opening (06:30-09:00) or after 16:00 · Museums/galleries 09:00-17:00, ideal 11:00-15:00 as a heat refuge · Wet/morning markets 06:00-09:00
   - Night markets, bars, rooftops, walking streets, shows after 17:30 · Beaches 06:00-09:00 or 16:00-18:30 · Viewpoints at sunrise or 60 min before sunset (~17:15 in Dec, ~18:30 in Jun)
   - Waterfalls, nature, boat trips in the morning · Cable cars and mountain parks at opening or after 16:00 (cloud and rain after 10:00) · Theme parks: a full day from opening
   - Heat (Mar-Sep north and centre, all year south): 10:30-15:30 is for indoor, shaded or air-conditioned stops, lunch, a cafe or a hotel rest; outdoor sights go before or after.
5. Visit lengths at the place (travel excluded): meal 45-90 min; temple 30-60; museum 90-150; market 45-90; beach 90-180; big park, cable car or theme park 240-420. Add 10-15 minutes of slack after every stop.
6. Rest blocks are stops without a place: {"rest":"hotel"} a midday rest, {"rest":"walk"} a stroll nearby, {"rest":"free"} free time. Use them when the profile asks for a slower day or the heat rule leaves a gap.
7. Travel between cities is handled outside the plan: output places and rest blocks only.
8. Discovery style: FAMOUS = mostly high-review-count landmarks; BALANCED = famous anchors plus local favourites; HIDDEN_GEMS = well-rated places with few reviews, plus one famous landmark per day for orientation.
9. Variety: spread the traveller's interests across the days, vary kinds within a day (two temples in a row is enough), and let the traveller's own words win over every default here.
10. Wording: each stop gets a practical guide for this traveller (see "Writing each stop") and the trip gets an overview (see "Writing the trip overview"); everything is grounded in the given data or stable, widely known facts, with no invented specifics.
11. Social-video itineraries: keep every social candidate, follow videoSequence above every other ordering rule, and output rows that share an optionGroupId as separate stops with the same time window (the user picks later).
12. Season first: plan around what makes these dates special here. The season's highlight (a flower or harvest season, cloud hunting, swimming weather, a festival) gets a stop in its best light and is named in the trip overview; a sight whose season is over is left out or described honestly (no blooming fields in the wrong month).
13. What a row is: its title and kinds decide. A restaurant, food shop, store, bar or club is a meal or a break, never a sight stop, whatever group it came in; a sight stop is somewhere people go to see or do something.
"""

DEFAULT_TRAVELLER_RULES = """\
## Rules by traveller profile (apply every matching block)
### Pace (non-meal stops per day; anchors are big sights, light stops are cafes, viewpoints, markets, walks)
- RELAXED: 1 anchor + 1 light stop, start 08:30, long meals (90 min), a rest or free block midday, last stop ends by 21:30. Skeleton: breakfast → anchor → lunch → rest → light stop or sunset → dinner → optional evening stroll.
- BALANCED: 2 anchors + 1 light stop, start 08:00, standard visits, last stop ends ~22:00. Skeleton: breakfast → anchor → light stop → lunch → indoor stop or cafe in the heat → anchor → sunset or evening block → dinner.
- EAGER: 3 anchors + 1-2 light stops, start 06:00-07:00 for sunrise or a morning market, quick meals, last stop ends by 23:00 (nightlife allowed).
### Family with children (mobilityConsiderations contains Kid-friendly, or the group mentions children/kids/baby)
- At most 3 stops/day, each 2 h or less; hotel rest 12:00-15:00 in hot months; dinner by 18:30; the day ends by 20:00; walking 3 km/day or less.
- Choose parks, beaches at cool hours, interactive museums, water puppets, cable cars and gentle boats. Leave out adventure, trekking, cliffs, long boat rides, bars, nightlife and steep-stairs sites (Hang Múa, Fansipan summit steps, long hikes).
- Babies and toddlers (0-3): 2 outdoor stops/day at most, a nap block at the hotel every day, stroller-friendly areas only.
### Elderly or limited mobility (Elderly-friendly, or the group mentions elderly/parents/grandparents)
- 2 stops/day with seating; 1 km of walking per stop at most; midday rest 12:00-15:00; the day ends by 19:30; door-to-door transport assumed.
- Seated sightseeing is ideal: cruises, rowing boats in the morning, cyclo, cable cars (stay at the station). Leave out steps-only sites, steep climbs, long standing, rough terrain, motorbike loops and sleeper buses.
### Couple or honeymoon
- 2 stops + 1 "moment" per day: a golden-hour viewpoint, a riverside or beach sunset, dinner around 19:30, a spa or free block 13:00-16:00. Prefer quieter alternatives to crowded spots; photo stops early (empty streets 06:30-08:00).
### Friends group
- Social dining, nightlife and photo spots are welcome; after a bar night the next day starts at 10:30 or later with no dawn activity.
### Solo
- Walkable areas, social cafes, safe neighbourhoods, easy public transport; bars and nightlife are fine.
### Adventure or trekking (activityTypes contains Adventure, or the user asks for trekking/hiking)
- Start 06:00-07:30 and finish all trekking or riding by 16:30; keep a 20-30% time buffer; a lighter day follows a big trek; a trek and an intercity move never share a day. Skip this block when children or elderly travel.
### Photography
- Viewpoints and landmarks at sunrise or golden hour; markets and old streets before 08:00.
### Food lovers (FOOD_AND_DRINK selected or the user talks about eating)
- A named breakfast venue before 08:30, the region's must-eat dishes across the days, one evening food crawl (17:30-20:30) or late-night snack; name real dishes from each venue's menu.
### Dietary restrictions
- A hard constraint on every food venue: vegetarian/vegan need veg or vegan tags, kinds or a fitting menu (prefer quán chay); halal needs halal tags or a seafood-safe menu; say in the description when a venue needs checking.
### Arrival and departure
- First day of a destination after a move: start later (nothing before ~10:30 after a morning drive; a light day after an overnight train or bus). Last day before a move: morning only, free after lunch. Otherwise the traveller is already in town at the pace's start time.
"""

DEFAULT_DESCRIPTION_MAX_CHARS = 600

# Code-owned (not in the config table): how each stop is written. It follows the editable rules,
# and it is static text, so it stays inside the cached prompt prefix.
STOP_WRITING_RULES = """\
## Writing each stop
- "stay": minutes a traveller like this one spends AT the place (travel excluded), as [min, max], e.g. [45, 60] for a noodle shop, [90, 150] for a big museum; make start and end fit it when the day allows.
- "description": a mini-guide from a friend who has been there, in the traveller's language, written the way Vietnamese travel bloggers write their own trip reports. Open with the most useful thing: why this stop at this hour (the light, the season, the crowd, the heat), what to do or see, or the dish to order. Then only the practical notes that matter for THIS stop and THIS traveller: what to try or skip, where to stand for the view or the photo, how to get in and what a ticket costs when the data says so, what to bring or wear (temples: shoulders and knees covered; beaches: swim at flagged zones), cash or card, queues and quiet hours, booking, fit for kids, elderly, wheelchairs or the diet, weather or safety cautions.
- Length and shape: 1-3 sentences for an ordinary stop, up to 5 for a day's anchor; a sentence plus up to 3 short "• " lines when a list reads better. Vary the shape from stop to stop; a market gets "what to look for", a viewpoint "when the light is best", a restaurant "what to order + one tip".
- A meal stop names what to order: dishes from its menu, else the regional specialty the venue is known for. A sight stop says what to do there and, for a famous paid sight, the entry or photo fee when it is widely known, as an approximate figure ("vé khoảng 150.000đ", "about 150,000₫").
- Every sentence tells this traveller something they can act on. Filler that would fit any stop ("a great place to visit", "fits the day's route", "enjoy the beautiful scenery") is left out.
- Accuracy over colour: use the candidate's own data (desc, menu, tags, price, hours, kinds) and stable, widely known facts about famous places. When unsure, give general advice ("weekends get crowded", "check today's hours") rather than a specific claim; a venue's hours, phone number, own prices and dishes come only from the data.
- Leave out the stop's name, address, times and the stay estimate: the app shows them already.
- Rest blocks: one short line on how to use the break.
- Vietnamese text addresses the reader as "bạn" in a warm, friendly voice (never "quý khách"); English text addresses "you". Place names and dish names stay in their original spelling.

## Writing the trip overview ("tripDescription")
What a friend tells you before you leave, in the traveller's language: 3-5 lines, each starting with "• ", each concrete for THESE dates and THIS traveller.
• The dates: what makes them special here (the season's highlight), the weather and temperature range, what to wear or bring.
• Getting around: where to base, how to move (motorbike, taxi, car with driver), how far the far sights are and which day they are on.
• Food: 2-4 signature dishes to try and which meals in the plan serve them.
• One tip that saves time, money or trouble: book ahead, carry cash, a fee, fog on a pass, a closing day, weekend crowds.
No marketing phrases and no repeat of the stop list."""

# Legend for the candidate rows; static, so it belongs to the cached prefix.
INPUT_LEGEND = """\
## Input
JSON {"destination": "...", "days": [{"dayNumber", "date", "weekday"}], "food": [...], "activities": [...]}.
Every food row is a food venue; every activity row is a sight or activity. Row fields (any may be missing):
- id: short ref such as "f3" or "a7"; copy it exactly into the output
- title, address (street and area)
- hours: 24h local time for the trip's weekdays; "daily" = same every day; missing = unknown, not closed
- desc: what the place is (for Web rows, why cited articles recommend it)
- kinds: curated sub-types (BEACH, SPIRITUAL, MUSEUM, CAFE, HALAL, VEGETARIAN...) or category when there are none
- rating, reviews, lat, lng, visitMin (typical visit minutes)
- price: band per person such as "1-100k₫" or a level such as "affordable"
- tags: facts from the venue's listing: meals served (breakfast, brunch, lunch, dinner, dessert, late_night), kids, wheelchair, veg, vegan, halal, alcohol, romantic, family, groups, solo, view, rooftop, live_music, outdoor, reservation_required, touristy
- menu (food only): signature dishes
- source: "web" = recommended by cited travel articles; "web_unlocated" = same but with no verified location, so place it where it fits the day without relying on geography; "social_video" rows also carry videoSequence, videoDay, videoTime, optionGroupId, optionIndex, relation, resolutionStatus
Treat price, tags and menu as facts: breakfast goes to breakfast-tagged venues or clear breakfast food; children prefer "kids" venues and skip alcohol-led bars; wheelchair users need "wheelchair"; diets need matching tags, kinds or menu; the budget is read from price."""

OUTPUT_CONTRACT = """\
## Output (JSON only) — exactly one entry in "days" per planned day, in the listed order
{"tripDescription":"• ...\\n• ...\\n• ...","days":[{"dayNumber":1,"area":"morning area → afternoon area","stops":[
  {"id":"f3","start":"07:30","end":"08:15","stay":[30,45],"description":"..."},
  {"id":"a7","start":"08:30","end":"10:00","stay":[60,90],"description":"..."},
  {"rest":"hotel","start":"12:30","end":"14:30","description":"..."}
]}]}
Times are local "HH:MM", chronological within a day, no overlaps, with the travel gap of rule 3 between stops."""

SELF_CHECK = """\
## Before answering, check every day
- breakfast, lunch and dinner at food ids in their windows; breakfast is not a cafe
- no id used twice in the trip; every id exists in the input; "area" named
- stops chronological, inside opening hours, with travel gaps; no outdoor sight 10:30-15:30 in hot months
- the hard constraints (dietary, mobility, children, elderly) and the traveller's own words are respected
- the stop count matches the pace and the profile
- each day is one loop out of the base, farthest first, with no back-and-forth across the destination
- no restaurant, shop, bar or club used as a sight; every meal names a dish; no stock filler sentences
- tripDescription has the dates, getting around, food and tip lines"""


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
    if req.get("travelStyle"):
        lines.append(f"Travel style: {req['travelStyle']}")
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


def build_select_prompt(req: dict[str, Any], locale: str | None, destination: str, days: int,
                        food_count: int, activity_count: int) -> str:
    """System prompt for the cheap shortlist call: pick the catalogue rows that fit THIS
    traveller, from the whole catalogue, before the planning call sees them. Rating is only a
    tie-breaker — a 4.8-star steakhouse is a wrong answer for a vegan request."""
    return f"""You shortlist places for a trip planner. Return JSON only.

## Input
The full catalogue for {destination} ({days} days) as {{"food":[...],"activities":[...]}}. Row fields (any may be missing): id (short ref), title, kinds or category, desc, price, tags (facts from the listing: meals served = breakfast/lunch/dinner, kids, wheelchair, veg, vegan, halal, alcohol, romantic, family, view, outdoor...), menu (food only: signature dishes), rating, reviews.

## Task
Pick exactly {food_count} food ids and exactly {activity_count} activity ids, in order of fit (all ids of a group when it has fewer). Decide in this order:
1. The traveller's own words and hard constraints: a diet needs matching tags, kinds or menu; wheelchair needs the wheelchair tag or an obviously flat venue; children skip alcohol-led venues. Leave out anything that conflicts.
2. Stated interests, travel style, discovery style and budget: a budget request takes cheap price bands; FAMOUS takes many reviews; HIDDEN_GEMS takes well-rated places with few reviews.
3. Coverage of the whole stay: enough breakfast-capable venues for every morning (breakfast tag, or phở/bánh mì/xôi/cháo/bún-type dishes), lunch and dinner venues, and activities spread across kinds (nature, culture, markets, entertainment) so no kind dominates.
4. Only then higher rating and review count.
Use only ids from the input. Output: {{"food":["f3","f12",...],"activities":["a7",...]}}

{preferences_block(req, locale)}"""


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
        extra += ("\n- SOCIAL VIDEO ITINERARY: videoSequence is the strongest ordering signal; keep every candidate "
                  "(unresolved ones included), and output options sharing an optionGroupId as separate stops with one time window.")
        extra += ("\n- Write tripDescription and every description as the planner talking to the traveller (tips, what to do or eat, timing); "
                  "the words video, clip, creator and source stay out, and nothing says the order follows anything.")
        if social.get("durationBasis"):
            extra += f"\n- Social duration decision: {social['durationBasis']}"
    # Static text first (identity, rules, legend, contract) and per-request text last, so providers
    # with automatic prefix caching (DeepSeek, OpenAI) bill the shared prefix as cached input.
    return f"""You are Pengoo, an expert Vietnam travel planner. You produce the COMPLETE itinerary for one destination: which places, which day, what time, in what order, plus the wording. Names may be Vietnamese.

{plan_rules}
{traveller_rules}
{STOP_WRITING_RULES}

{INPUT_LEGEND}

{OUTPUT_CONTRACT}

{preferences_block(req, locale)}

## Task
Destination: {destination}. Plan {days} day(s): {day_list}.{extra}
Fill every day from morning to evening according to the traveller profile. Write all free text in {language(locale)}. Each stop description is at most {max_chars} characters; tripDescription follows "Writing the trip overview" in at most 700 characters.

{SELF_CHECK}"""
