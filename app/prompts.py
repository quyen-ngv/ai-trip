"""System prompts for AI trip generation with detailed field explanations.

IMPORTANT: This file MUST stay in sync with Flutter frontend constants.
Frontend file: goroute_fe/lib/features/home/presentation/screens/create_ai_trip_screen.dart

When frontend values change, update the corresponding mappings here.
"""

# ============================================================================
# FIELD VALUE MAPPING (MUST MATCH FRONTEND)
# ============================================================================

PLACE_GROUPS = {
    'FOOD_AND_DRINK': {
        'label': '🍽️ Food & Drink',
        'description': 'Restaurants, cafes, street food, local cuisine, food markets, cooking classes',
        'keywords': ['restaurant', 'cafe', 'food', 'dining', 'cuisine', 'eatery', 'bistro'],
        'priority': 'High-rated eateries, local specialties, diverse meal options across breakfast/lunch/dinner'
    },
    'CULTURE_AND_HERITAGE': {
        'label': '🏛️ Culture & Heritage',
        'description': 'Museums, temples, historical sites, art galleries, cultural performances, monuments',
        'keywords': ['museum', 'temple', 'historical', 'heritage', 'cultural', 'art', 'monument', 'shrine'],
        'priority': 'Significant cultural landmarks, UNESCO sites, authentic local experiences'
    },
    'NATURE_AND_OUTDOORS': {
        'label': '🌿 Nature & Outdoors',
        'description': 'Parks, beaches, mountains, hiking trails, gardens, scenic viewpoints, waterfalls',
        'keywords': ['park', 'beach', 'mountain', 'nature', 'garden', 'outdoor', 'trail', 'scenic', 'waterfall'],
        'priority': 'Natural beauty, outdoor activities, photo spots, weather-appropriate locations'
    },
    'SHOPPING_AND_MARKET': {
        'label': '🛍️ Shopping & Market',
        'description': 'Markets, shopping streets, local crafts, night markets, malls, souvenir shops',
        'keywords': ['market', 'shopping', 'mall', 'boutique', 'craft', 'souvenir', 'bazaar'],
        'priority': 'Authentic markets, local products, souvenir shopping, budget-friendly options'
    },
    'ATTRACTIONS': {
        'label': '🏖️ Attractions',
        'description': 'Theme parks, entertainment venues, popular tourist spots, landmarks, viewpoints',
        'keywords': ['attraction', 'landmark', 'tourist', 'entertainment', 'theme park', 'famous'],
        'priority': 'Must-see attractions, family-friendly venues, well-reviewed experiences'
    },
}

PACE_OPTIONS = {
    'RELAXED': {
        'label': '🍃 Relaxed',
        'activities_per_day': '2-3',
        'start_time': '8:00-9:00',
        'end_time': '20:00-22:00',  # CAN include evening/nightlife
        'meal_duration': '90-120min',
        'buffer_time': '90min',
        'characteristics': [
            'Longer visit durations at each place',
            'Extended meal breaks with time to relax',
            'Generous buffer time between activities',
            'Flexible timing, no rushing',
            'CAN include evening activities: bars, night markets, sunset spots',
            'CAN start early for meaningful sunrise/beach experiences',
        ],
        'guidelines': [
            'Avoid back-to-back tight schedules',
            'Include leisurely meals (90-120min)',
            'Allow rest periods between major activities',
            'Evening activities WELCOME (bars, night markets, sunset)',
            'Sunrise spots WELCOME if meaningful (beach, mountain viewpoint)',
        ]
    },
    'BALANCED': {
        'label': '🚶 Balanced',
        'activities_per_day': '3-4',
        'start_time': '8:00-9:00',
        'end_time': '20:00-22:00',
        'meal_duration': '60-90min',
        'buffer_time': '60min',
        'characteristics': [
            'Standard visit durations',
            'Reasonable meal breaks',
            'Moderate walking and transport',
            'Mix of major attractions and relaxed time',
            'Evening activities if fits schedule',
        ],
        'guidelines': [
            'Standard scheduling, not too rushed',
            'Mix major attractions with relaxed meals',
            'Evening activities WELCOME',
        ]
    },
    'EAGER': {
        'label': '⚡ Eager',
        'activities_per_day': '4-5+',
        'start_time': '5:00-8:00',  # CAN start much earlier
        'end_time': '21:00-23:00',  # CAN end later
        'meal_duration': '45-60min',
        'buffer_time': '45min',
        'characteristics': [
            'Efficient scheduling, maximize experiences',
            'Quick meals, optimize travel time',
            'CAN start very early (5-6am) for sunrise, beaches, early markets',
            'CAN end late (10-11pm) with nightlife, night markets, bars',
            'More activities packed per day',
        ],
        'guidelines': [
            'Maximize experiences, efficient transitions',
            'Early starts ENCOURAGED (sunrise at beach, early market visits)',
            'Late endings ENCOURAGED (nightlife, bars, evening entertainment)',
            'Shorter breaks, more activities',
        ]
    },
}

GROUP_COMPOSITIONS = {
    'SOLO': {
        'label': '🎒 Solo',
        'description': 'Solo traveler',
        'characteristics': [
            'Social venues (cafes with communal seating)',
            'Walkable areas, safe neighborhoods',
            'Solo-friendly activities',
            'Easy public transport access',
            'CAN enjoy bars, nightlife independently',
        ]
    },
    'COUPLE': {
        'label': '💑 Couple',
        'description': 'Couple (2 adults)',
        'characteristics': [
            'Romantic settings, scenic spots',
            'Fine dining options',
            'Photography locations',
            'Intimate venues',
            'Sunset viewpoints, beach walks',
            'Couple-friendly bars, wine bars',
        ]
    },
    'FAMILY': {
        'label': '👨\u200d👩\u200d👧 Family',
        'description': 'Family (may include children/elderly)',
        'characteristics': [
            'Kid-friendly attractions if hasChildren',
            'Accessible venues if hasElderly',
            'Family restaurants',
            'Interactive museums, playgrounds',
            'Rest areas, accessible facilities',
            'Avoid very late-night activities if young children',
        ]
    },
    'FRIENDS': {
        'label': '👥 Friends',
        'description': 'Friends group',
        'characteristics': [
            'Social activities, group dining',
            'Nightlife, bars, entertainment',
            'Adventure options',
            'Photo-worthy spots',
            'Group-friendly restaurants',
            'Evening/night activities WELCOME',
        ]
    },
}

DISCOVERY_STYLES = {
    'FAMOUS': {
        'label': '🌟 Famous',
        'popular_ratio': 0.9,  # 90% famous, 10% hidden
        'description': 'Focus on well-known attractions and top-rated places',
        'guidelines': [
            'START with famous attractions (must-see landmarks)',
            'Prioritize top-rated tourist spots',
            'Well-known restaurants with proven quality',
            'Classic travel experience, iconic photos',
            'High review counts (500+) preferred',
            'Can add 1-2 hidden gems as bonus',
        ]
    },
    'BALANCED': {
        'label': '⚖️ Balanced',
        'popular_ratio': 0.6,  # 60% famous, 40% hidden
        'description': 'Mix popular attractions with local favorites',
        'guidelines': [
            'START with 1-2 famous attractions per day',
            'Fill remaining slots with local favorites',
            'Mix tourist areas with residential neighborhoods',
            'Balance proven quality with authentic discoveries',
            'Medium-to-high review counts (100+)',
        ]
    },
    'HIDDEN_GEMS': {
        'label': '💎 Hidden Gems',
        'popular_ratio': 0.3,  # 30% famous, 70% hidden
        'description': 'Prefer local favorites and off-beaten-path spots',
        'guidelines': [
            'Include 1 famous landmark per day for orientation',
            'Focus on local favorites, neighborhood spots',
            'Avoid over-touristed locations',
            'Prefer authentic local restaurants',
            'Lower review counts OK if rating high (4.5+)',
            'Residential areas, local hangouts preferred',
        ]
    },
}

DIETARY_RESTRICTIONS = {
    'vegetarian': 'No meat, poultry, seafood (eggs and dairy OK)',
    'vegan': 'No animal products (no meat, dairy, eggs, honey)',
    'halal': 'Islamic dietary laws (no pork, no alcohol, proper slaughter required)',
    'no_pork': 'Exclude pork and pork products',
    'pescatarian': 'No meat/poultry (seafood, eggs, dairy OK)',
    'beef_free': 'No beef products',
    'jain': 'No root vegetables, no meat, strict vegetarian principles',
}

MOBILITY_CONSIDERATIONS = {
    'Kid-friendly': [
        'Stroller access required',
        'Changing facilities available',
        'Kid menus at restaurants',
        'Short visit durations (under 2 hours per activity)',
        'Avoid long walks (>30min), stairs-only venues',
        'Avoid adult-only spaces',
        'Playgrounds, interactive elements welcome',
    ],
    'Elderly-friendly': [
        'Elevator/ramp access required',
        'Seating areas available',
        'Restrooms accessible',
        'Climate-controlled spaces preferred',
        'Avoid steep hills, long standing',
        'Avoid extreme temperatures',
        'Avoid rough terrain',
    ],
}

ACTIVITY_TYPES = {
    'Adventure': {
        'description': 'Zip-lining, water sports, hiking, extreme activities, adventure tours',
        'keywords': ['adventure', 'zip-line', 'water sport', 'hiking', 'extreme', 'climbing'],
    },
    'Photography': {
        'description': 'Scenic viewpoints, golden hour spots, Instagrammable locations',
        'keywords': ['viewpoint', 'scenic', 'photo spot', 'panorama', 'sunset', 'sunrise'],
    },
}

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

TRIP_CONTEXT_GUIDE = """
# TripMind AI Trip Generation Guidelines

## CRITICAL RULES

### 1. All Paces Include Evening Activities
**IMPORTANT**: RELAXED, BALANCED, and EAGER can ALL include evening/night activities:
- Bars, pubs, wine bars
- Night markets, street food at night
- Sunset viewpoints, beach sunset walks
- Evening cultural performances
- Late dinners (8-9pm or later)

RELAXED = fewer activities with more time, NOT avoiding evenings.
EAGER = can start earlier (5-6am) AND end later (10-11pm).

### 2. Meal Requirements
**MANDATORY**: Every day MUST have 3 meals:
- Breakfast: 7:00-9:30am (can be earlier for EAGER)
- Lunch: 11:30am-2:00pm (REQUIRED)
- Dinner: 6:00pm-9:00pm (REQUIRED, can be later)

### 3. Special Timing Flexibility
Don't rigidly enforce hours if meaningful:
- Sunrise spots (beach, mountain): 5:00-7:00am
- Night markets: 7:00pm-11:00pm
- Bars/nightlife: 8:00pm-midnight
- Early markets: 6:00-8:00am

### 4. Discovery Style Priority
**Always prioritize famous/popular places FIRST**, then add hidden gems:
- FAMOUS: Start with top attractions, add 1-2 hidden gems
- BALANCED: Start with famous, fill with local favorites
- HIDDEN_GEMS: Include 1 famous landmark/day, focus on locals

## Place Groups
"""

for group_id, data in PLACE_GROUPS.items():
    TRIP_CONTEXT_GUIDE += f"\n**{group_id}** - {data['label']}\n"
    TRIP_CONTEXT_GUIDE += f"{data['description']}\n"
    TRIP_CONTEXT_GUIDE += f"Priority: {data['priority']}\n"

TRIP_CONTEXT_GUIDE += "\n## Pace Settings\n"
for pace_id, data in PACE_OPTIONS.items():
    TRIP_CONTEXT_GUIDE += f"\n**{pace_id}** - {data['label']}\n"
    TRIP_CONTEXT_GUIDE += f"Activities: {data['activities_per_day']}/day\n"
    TRIP_CONTEXT_GUIDE += f"Hours: {data['start_time']} - {data['end_time']}\n"
    TRIP_CONTEXT_GUIDE += "Key points:\n"
    for char in data['characteristics']:
        TRIP_CONTEXT_GUIDE += f"  • {char}\n"

TRIP_CONTEXT_GUIDE += "\n## Group Composition\n"
for comp_id, data in GROUP_COMPOSITIONS.items():
    TRIP_CONTEXT_GUIDE += f"\n**{comp_id}** - {data['label']}\n"
    for char in data['characteristics']:
        TRIP_CONTEXT_GUIDE += f"  • {char}\n"

TRIP_CONTEXT_GUIDE += "\n## Discovery Style\n"
for style_id, data in DISCOVERY_STYLES.items():
    ratio = int(data['popular_ratio'] * 100)
    TRIP_CONTEXT_GUIDE += f"\n**{style_id}** - {data['label']}\n"
    TRIP_CONTEXT_GUIDE += f"Ratio: {ratio}% famous, {100-ratio}% hidden\n"
    for guide in data['guidelines']:
        TRIP_CONTEXT_GUIDE += f"  • {guide}\n"

TRIP_CONTEXT_GUIDE += "\n## Dietary Restrictions (CRITICAL)\n"
for diet_id, desc in DIETARY_RESTRICTIONS.items():
    TRIP_CONTEXT_GUIDE += f"- **{diet_id}**: {desc}\n"

TRIP_CONTEXT_GUIDE += "\n## Scoring Formula (timeScore 0-100)\n"
TRIP_CONTEXT_GUIDE += """
Base: 50 points

+30: Interest match (place category matches placeGroups)
+25: Discovery fit (popularity aligns with style)
+25: Dietary compliance (food venue meets restrictions) OR -50 if violates
+10: Mobility suitable (accessible per requirements) OR -30 if not
+10: User preference keywords match

Special bonuses:
+10: Special timing value (sunrise spot, night market, etc.)

NEVER select if violates:
- Dietary restrictions (food venue)
- Mobility requirements
- Explicit "avoid" in preferenceText
"""


def build_filter_prompt(preferences: dict) -> str:
    """Build context-aware prompt for fast_filter stage - MINIMAL version with HARD CONSTRAINTS."""
    
    place_groups = preferences.get('placeGroups', [])
    pace = preferences.get('pace', 'BALANCED')
    group = preferences.get('groupComposition', '')
    discovery = preferences.get('discoveryStyle', 'BALANCED')
    dietary = preferences.get('dietaryRestrictions', [])
    mobility = preferences.get('mobilityConsiderations', [])
    pref_text = preferences.get('preferenceText', '').strip()
    budget_min = preferences.get('budgetMin')
    budget_max = preferences.get('budgetMax')
    
    parts = ["# TripMind AI Trip Selection\n"]
    
    # === HARD CONSTRAINTS (NON-NEGOTIABLE) ===
    parts.append("## ⚠️ HARD CONSTRAINTS (MUST FOLLOW - NO EXCEPTIONS)")
    parts.append("")
    parts.append("### 1. THREE MEALS PER DAY (MANDATORY)")
    parts.append("Every day MUST have EXACTLY 3 main meals:")
    parts.append("- **Breakfast**: 7:00-10:00am (main meal, 60-90min)")
    parts.append("- **Lunch**: 11:30am-2:30pm (main meal, 60-90min)")
    parts.append("- **Dinner**: 6:00-10:00pm (main meal, 60-90min)")
    parts.append("- Cafes/snacks (30-45min) do NOT count as main meals")
    parts.append("")
    
    parts.append("### 2. MEAL SPACING")
    parts.append("- MINIMUM 3 hours between main meals")
    parts.append("- Example: Breakfast 8am → Lunch 12pm (4h gap ✓)")
    parts.append("- Example: Breakfast 8am → Lunch 10am (2h gap ✗ INVALID)")
    parts.append("")
    
    parts.append("### 3. INTEREST DIVERSITY")
    parts.append("Each day MUST have variety:")
    parts.append("- MAX 40% of activities from same category")
    parts.append("- If user selected Food + Culture + Nature:")
    parts.append("  - ✓ Food 40%, Culture 30%, Nature 30%")
    parts.append("  - ✗ Food 70%, Culture 20%, Nature 10% (INVALID)")
    parts.append("")
    
    parts.append("### 4. ACTIVITY WINDOW")
    pace_data = PACE_OPTIONS.get(pace, PACE_OPTIONS['BALANCED'])
    parts.append(f"**{pace_data['label']}** requirements:")
    parts.append(f"- {pace_data['activities_per_day']} activities/day")
    parts.append(f"- Start: {pace_data['start_time']}")
    parts.append(f"- End: {pace_data['end_time']}")
    
    if pace == 'BALANCED':
        parts.append("- Last substantial activity MUST be >= 6:00pm OR total window >= 8 hours")
        parts.append("- ✗ Ending at 1:30pm is NOT Balanced (INVALID)")
    elif pace == 'RELAXED':
        parts.append("- CAN include evening activities (bars, night markets, sunset)")
        parts.append("- Fewer activities with longer duration, NOT early ending")
    elif pace == 'EAGER':
        parts.append("- CAN start as early as 5-6am (sunrise, markets)")
        parts.append("- CAN end as late as 10-11pm (nightlife, bars)")
        parts.append("- Maximize experiences throughout the day")
    parts.append("")
    
    parts.append("### 5. TRANSPORT TIMING")
    parts.append("Transport slots MUST match actual duration:")
    parts.append("- Walking (<1.5km): time = (distance_km / 4.5 * 60) + 5min buffer")
    parts.append("- Taxi/car: time = (distance_km / 20 * 60) + 5min buffer")
    parts.append("- ✗ 0.7km walk = 10min actual → do NOT allocate 60min slot")
    parts.append("- ✓ 0.7km walk = 10min + 5min buffer = 15min slot")
    parts.append("")
    
    parts.append("---\n")
    
    # === USER PREFERENCES ===
    parts.append("## 📋 Trip Preferences\n")
    
    # 1. Place Groups (only selected ones)
    if place_groups:
        parts.append("### Interests (MUST have diversity - see constraint #3)")
        for g in place_groups:
            data = PLACE_GROUPS.get(g, {})
            parts.append(f"- **{data.get('label', g)}**: {data.get('description', '')}")
        parts.append("")
    
    # 2. Group (only selected one)
    if group:
        comp_data = GROUP_COMPOSITIONS.get(group)
        if comp_data:
            parts.append(f"### Group: {comp_data.get('label', group)}")
            chars = comp_data.get('characteristics', [])
            for char in chars[:3]:  # Only first 3
                parts.append(f"- {char}")
            if mobility:
                parts.append(f"- Mobility: {', '.join(mobility)}")
            parts.append("")
    
    # 3. Discovery (only selected one)
    style_data = DISCOVERY_STYLES.get(discovery, DISCOVERY_STYLES['BALANCED'])
    ratio = int(style_data['popular_ratio'] * 100)
    parts.append(f"### Discovery: {style_data['label']} ({ratio}% famous, {100-ratio}% hidden)")
    for guide in style_data['guidelines'][:3]:  # Only first 3
        parts.append(f"- {guide}")
    parts.append("")
    
    # 4. Dietary (CRITICAL)
    if dietary:
        parts.append("### ⚠️ CRITICAL Dietary Restrictions")
        for d in dietary:
            parts.append(f"- **{d}**: {DIETARY_RESTRICTIONS.get(d, d)}")
        parts.append("**NEVER select food venues violating these**\n")
    
    # 5. Budget
    if budget_max:
        parts.append(f"### Budget: ${budget_min or 0:,.0f} - ${budget_max:,.0f}\n")
    
    # 6. User text (HIGHEST PRIORITY)
    if pref_text:
        parts.append(f"### 🔥 User Preferences (OVERRIDE defaults when conflict)")
        parts.append(f"> {pref_text}\n")
    
    return "\n".join(parts)


def build_enrichment_prompt(preferences: dict) -> str:
    """Build detailed prompt for deep_select enrichment stage - with HARD CONSTRAINTS."""
    
    # Reuse base prompt with hard constraints
    base = build_filter_prompt(preferences)
    
    # Add enrichment task with explicit rules
    enrichment_task = """
---

## 🎯 Task: Enrich Places

For EACH place, you MUST calculate and return:

### 1. timeScore (0-100)
Formula:
- Base: 50 points
- +30 if category matches user interests (placeGroups)
- +25 if popularity matches discovery style
- +25 if food venue meets dietary restrictions (-50 if violates)
- +10 if accessible per mobility requirements (-30 if not)
- +10 if matches user preference keywords
- +10 bonus for special timing value (sunrise spot, night market, etc.)

### 2. isFoodVenue (boolean)
- `true` for: restaurants, cafes, bistros, food markets
- `false` for: attractions, museums, parks, etc.

### 3. mealType (string)
**CRITICAL for constraint validation:**
- **"breakfast"**: Opens 7-10am, suitable for breakfast (pho, banh mi, etc.)
- **"lunch"**: Opens 11:30am-2:30pm, main restaurant
- **"dinner"**: Opens 6-10pm, main restaurant
- **"snack"**: Cafes, dessert shops, light bites (does NOT count as main meal)
- **"anytime"**: Food venue open all day

### 4. visitDurationMinutes (integer)
- Main meal (breakfast/lunch/dinner): 60-90 minutes
- Snack/cafe: 30-45 minutes
- Small attraction/temple: 60-90 minutes
- Museum/major attraction: 90-180 minutes
- Theme park/large venue: 180-360 minutes

### 5. dietaryTags (array of strings)
If food venue, list applicable tags:
- "vegetarian", "vegan", "halal", "pescatarian", "beef_free", "no_pork", "jain"

### 6. openingHoursValid (boolean)
- `true` if opening hours data exists and is parseable
- `false` if no hours data or invalid format

---

## 📤 Output Format

Return JSON array with ALL input places enriched:
```json
[
  {
    "id": "place-id",
    "timeScore": 85,
    "isFoodVenue": true,
    "mealType": "breakfast",
    "visitDurationMinutes": 75,
    "dietaryTags": ["vegetarian"],
    "openingHoursValid": true
  },
  ...
]
```

**Order by timeScore DESC (highest first).**

**REMEMBER**: You will be judged on constraint compliance later. Ensure:
- Diverse categories (not all food)
- Proper meal type classification
- Realistic visit durations
"""
    
    return base + "\n" + enrichment_task


def build_verify_prompt(preferences: dict) -> str:
    """Build prompt for verify stage - STRICT validation with hard constraints."""
    
    # Reuse base trip context
    base = build_filter_prompt(preferences)
    
    # Add verify task with explicit validation rules
    verify_task = """
---

## 🔍 Task: STRICT Itinerary Validation

You are a STRICT validator. Your job is to CHECK constraints, NOT justify violations.

### Validation Checklist (MUST all pass):

#### ✅ 1. Three Meals Per Day
For EACH day, verify:
- [ ] Breakfast exists (7-10am, 60-90min)
- [ ] Lunch exists (11:30am-2:30pm, 60-90min)
- [ ] Dinner exists (6-10pm, 60-90min)
- [ ] Cafes/snacks do NOT count as main meals

**If ANY day missing ANY meal → FAIL**

#### ✅ 2. Meal Spacing
- [ ] Minimum 3 hours between main meals
- [ ] Example: 8am breakfast → 12pm lunch (4h ✓)
- [ ] Example: 8am breakfast → 10:30am lunch (2.5h ✗)

**If ANY meals <3h apart → FAIL**

#### ✅ 3. Interest Diversity
- [ ] No single category >40% of activities per day
- [ ] All user-selected interests represented

**If >40% same category → FAIL**

#### ✅ 4. Activity Window
For BALANCED pace:
- [ ] Last activity >= 6:00pm OR total window >= 8 hours
- [ ] Ending at 1:30pm is NOT balanced

**If window too short → FAIL**

#### ✅ 5. Transport Timing
- [ ] Transport slots match actual duration + buffer
- [ ] Walking 0.7km = ~15min slot (not 60min)
- [ ] No excessive buffers

**If transport slots unrealistic → FAIL**

#### ✅ 6. Dietary Compliance
- [ ] All food venues meet dietary restrictions
- [ ] No violations of halal/vegan/etc.

**If ANY violation → FAIL**

---

## 📤 Output Format

```json
{
  "valid": true/false,
  "violations": [
    "Day 1 missing dinner",
    "Day 2: breakfast-lunch gap only 2 hours",
    "Day 1: 60% food venues (>40% limit)",
    "Day 1 ends at 1:30pm (too early for Balanced)",
    "Transport slot 60min for 10min walk"
  ],
  "description": "2-3 sentence trip summary (only if valid=true)",
  "reasons": {
    "Place Name": "Why it fits preferences (only if valid=true)"
  }
}
```

**CRITICAL**: 
- If `valid=false`, list ALL violations clearly
- Do NOT try to justify violations ("buffer", "flexibility", etc.)
- Be STRICT - these are HARD constraints, not guidelines
"""
    
    return base + "\n" + verify_task
