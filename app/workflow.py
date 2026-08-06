from __future__ import annotations
import asyncio, math, re, logging
from datetime import date, datetime, timedelta
from typing import Any, TypedDict
import httpx
from langgraph.graph import StateGraph, START, END
from ortools.sat.python import cp_model
from .llm import CachedModel

logger = logging.getLogger(__name__)

class State(TypedDict, total=False):
    job: dict[str,Any]
    inventories: list[list[dict[str,Any]]]
    filtered: list[list[dict[str,Any]]]
    enriched: list[list[dict[str,Any]]]
    selected: list[list[dict[str,Any]]]
    items: list[dict[str,Any]]
    description: str

class JavaClient:
    def __init__(self, job: dict[str,Any], token: str): self.job=job; self.token=token
    @property
    def root(self): return self.job["callbackBaseUrl"].rstrip("/")+f"/v1/api/internal/ai-trip-generations/{self.job['jobId']}"
    @property
    def headers(self): return {"X-Internal-Token":self.token,"X-Attempt-Id":self.job["attemptId"]}
    async def event(self,stage,progress,key,status="RUNNING",error=None):
        body={"attemptId":self.job["attemptId"],"stage":stage,"status":status,"progress":progress,"messageKey":key,"params":{},"errorMessage":error}
        async with httpx.AsyncClient(timeout=30) as c: (await c.post(self.root+"/events",headers=self.headers,json=body)).raise_for_status()
    async def candidates(self,d):
        body={"latitude":d["latitude"],"longitude":d["longitude"],"placeGroups":[x.upper() for x in self.job["request"].get("placeGroups",[])],"limit":250}
        async with httpx.AsyncClient(timeout=60) as c:
            r=await c.post(self.root+"/candidates",headers=self.headers,json=body);r.raise_for_status();return r.json()["data"]
    async def commit(self,items,description):
        body={"attemptId":self.job["attemptId"],"items":items,"tripDescription":description}
        async with httpx.AsyncClient(timeout=90) as c:
            r=await c.post(self.root+"/commit",headers=self.headers,json=body)
            
            if r.status_code != 200:
                logger.error(f"Commit failed: {r.status_code} - {r.text}")
                r.raise_for_status()
            
            response_json = r.json()
            
            # Check if backend returned business error (meta.code exists and != 200000)
            # Backend uses 200000 for SUCCESS, not 200
            if "meta" in response_json:
                code = response_json["meta"].get("code")
                if code and code != 200000:
                    error_msg = response_json["meta"].get("message", "Business error")
                    logger.error(f"Business error from backend: code={code}, message={error_msg}")
                    raise Exception(error_msg)
            
            # Success case: return data
            return response_json.get("data")

def tokens(text): return set(re.findall(r"[\wÀ-ỹ]+",(text or "").lower()))
def haversine(a,b):
    p1,p2=math.radians(float(a["latitude"])),math.radians(float(b["latitude"]));dp=p2-p1;dl=math.radians(float(b["longitude"])-float(a["longitude"]));q=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 6371*2*math.atan2(math.sqrt(q),math.sqrt(1-q))

def rank(place,prefs):
    corpus=" ".join(str(place.get(k) or "") for k in ("title","address","category","menuHighlights","attributes","description"))
    overlap=len(tokens(prefs)&tokens(corpus)); quality=float(place.get("score") or place.get("reviewRating") or 0); reviews=math.log1p(place.get("reviewCount") or 0)
    return overlap*20+quality*2+reviews-float(place.get("distanceKm") or 0)*.05

def destination_days(d): return (date.fromisoformat(d["endDate"])-date.fromisoformat(d["startDate"])).days+1

def solve_schedule(places_with_time, days, day_offset, prefs):
    """OR-Tools scheduling with duration, travel time, and pace constraints."""
    if not places_with_time:
        return []
    
    # Configuration
    DAY_START_MINUTES = 8 * 60  # 8:00
    DAY_END_MINUTES = 20 * 60    # 20:00
    MAX_ACTIVITIES_PER_DAY = 5   # RELAXED pace
    WALKING_SPEED_KM_PER_HOUR = 4.5
    TAXI_SPEED_KM_PER_HOUR = 20
    
    model = cp_model.CpModel()
    
    # Decision variables
    x = {}  # x[i,d] = place i assigned to day d
    start_time = {}  # start_time[i] = start time in minutes from midnight
    
    for i in range(len(places_with_time)):
        for d in range(days):
            x[i,d] = model.new_bool_var(f'x_{i}_{d}')
        # Start time domain: 8:00 to 19:00 (allow 1h activity until 20:00)
        start_time[i] = model.new_int_var(DAY_START_MINUTES, DAY_END_MINUTES - 60, f'start_{i}')
    
    # Constraint 1: Each place used at most once
    for i in range(len(places_with_time)):
        model.add(sum(x[i,d] for d in range(days)) <= 1)
    
    # Constraint 2: Max activities per day (pace control)
    for d in range(days):
        model.add(sum(x[i,d] for i in range(len(places_with_time))) <= MAX_ACTIVITIES_PER_DAY)
    
    # Constraint 3: Non-overlapping intervals with travel time
    for d in range(days):
        intervals = []
        
        for i in range(len(places_with_time)):
            p = places_with_time[i]
            duration = p.get('visitDurationMinutes') or 90  # Default 90 if None
            
            # Create interval only if place assigned to this day
            interval = model.new_optional_interval_var(
                start_time[i],
                duration,
                start_time[i] + duration,
                x[i,d],
                f'interval_{i}_{d}'
            )
            intervals.append(interval)
        
        # No overlap within day
        model.add_no_overlap(intervals)
        
        # Add minimum gap between activities for travel time
        for i1 in range(len(places_with_time)):
            for i2 in range(len(places_with_time)):
                if i1 >= i2:
                    continue
                
                p1 = places_with_time[i1]
                p2 = places_with_time[i2]
                
                # Calculate travel time between places
                km = haversine(p1, p2) if p1.get('latitude') and p2.get('latitude') else 0
                if km < 1.5:
                    travel_minutes = max(10, int(km / WALKING_SPEED_KM_PER_HOUR * 60))
                else:
                    travel_minutes = max(5, int(km / TAXI_SPEED_KM_PER_HOUR * 60))
                
                # If both selected in same day, enforce minimum gap
                both_selected = model.new_bool_var(f'both_{i1}_{i2}_{d}')
                model.add(both_selected == 1).only_enforce_if([x[i1,d], x[i2,d]])
                
                # If i2 starts after i1 ends, add travel time
                duration1 = p1.get('visitDurationMinutes') or 90
                i2_after_i1 = model.new_bool_var(f'i2_after_i1_{i1}_{i2}_{d}')
                model.add(start_time[i2] >= start_time[i1] + duration1).only_enforce_if([both_selected, i2_after_i1])
                model.add(start_time[i2] >= start_time[i1] + duration1 + travel_minutes).only_enforce_if([both_selected, i2_after_i1])
                
                # If i1 starts after i2 ends, add travel time
                duration2 = p2.get('visitDurationMinutes') or 90
                i1_after_i2 = model.new_bool_var(f'i1_after_i2_{i1}_{i2}_{d}')
                model.add(start_time[i1] >= start_time[i2] + duration2).only_enforce_if([both_selected, i1_after_i2])
                model.add(start_time[i1] >= start_time[i2] + duration2 + travel_minutes).only_enforce_if([both_selected, i1_after_i2])
                
                # One must be after the other if both selected
                model.add(i2_after_i1 + i1_after_i2 >= 1).only_enforce_if(both_selected)
    
    # Constraint 4: Soft meal constraints (penalties instead of hard constraints)
    meal_penalties = []
    
    for d in range(days):
        breakfast_places = [i for i,p in enumerate(places_with_time) 
                           if p.get('isFoodVenue') and p.get('mealType') in ['breakfast', 'snack']]
        lunch_places = [i for i,p in enumerate(places_with_time) 
                       if p.get('isFoodVenue') and p.get('mealType') in ['lunch', 'snack']]
        dinner_places = [i for i,p in enumerate(places_with_time) 
                        if p.get('isFoodVenue') and p.get('mealType') in ['dinner', 'snack']]
        
        # Breakfast missing penalty (lower priority)
        if breakfast_places:
            has_breakfast = model.new_bool_var(f'has_breakfast_{d}')
            model.add(sum(x[i,d] for i in breakfast_places) >= 1).only_enforce_if(has_breakfast)
            model.add(sum(x[i,d] for i in breakfast_places) == 0).only_enforce_if(has_breakfast.Not())
            breakfast_penalty = model.new_int_var(0, 1000, f'breakfast_penalty_{d}')
            model.add(breakfast_penalty == 0).only_enforce_if(has_breakfast)
            model.add(breakfast_penalty == 10).only_enforce_if(has_breakfast.Not())
            meal_penalties.append(breakfast_penalty)
        
        # Lunch missing penalty
        if lunch_places:
            has_lunch = model.new_bool_var(f'has_lunch_{d}')
            model.add(sum(x[i,d] for i in lunch_places) >= 1).only_enforce_if(has_lunch)
            model.add(sum(x[i,d] for i in lunch_places) == 0).only_enforce_if(has_lunch.Not())
            lunch_penalty = model.new_int_var(0, 3000, f'lunch_penalty_{d}')
            model.add(lunch_penalty == 0).only_enforce_if(has_lunch)
            model.add(lunch_penalty == 30).only_enforce_if(has_lunch.Not())
            meal_penalties.append(lunch_penalty)
        
        # Dinner missing penalty
        if dinner_places:
            has_dinner = model.new_bool_var(f'has_dinner_{d}')
            model.add(sum(x[i,d] for i in dinner_places) >= 1).only_enforce_if(has_dinner)
            model.add(sum(x[i,d] for i in dinner_places) == 0).only_enforce_if(has_dinner.Not())
            dinner_penalty = model.new_int_var(0, 3000, f'dinner_penalty_{d}')
            model.add(dinner_penalty == 0).only_enforce_if(has_dinner)
            model.add(dinner_penalty == 30).only_enforce_if(has_dinner.Not())
            meal_penalties.append(dinner_penalty)
    
    # Objective: maximize preference scores - meal penalties
    score_sum = sum(
        int(p.get('timeScore', 50) * 100) * x[i,d]
        for i,p in enumerate(places_with_time)
        for d in range(days)
    )
    total_penalty = sum(meal_penalties) if meal_penalties else 0
    model.maximize(score_sum - total_penalty)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8
    status = solver.solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        logger.warning(f"Solver failed with status {status}, using fallback scheduler")
        # Fallback: simple round-robin scheduler
        return fallback_schedule(places_with_time, days, day_offset, prefs)
    
    # Extract solution
    schedule = []
    for d in range(days):
        day_items = []
        for i, p in enumerate(places_with_time):
            if solver.value(x[i,d]):
                place = p.copy()
                place['dayNumber'] = day_offset + d + 1
                place['startMinutes'] = solver.value(start_time[i])
                
                # Convert minutes to HH:MM
                hour = place['startMinutes'] // 60
                minute = place['startMinutes'] % 60
                place['startTime'] = f"{hour:02d}:{minute:02d}:00"
                
                # Calculate end time
                duration = p.get('visitDurationMinutes') or 90  # Default 90 if None
                end_minutes = place['startMinutes'] + duration
                end_hour = end_minutes // 60
                end_minute = end_minutes % 60
                place['endTime'] = f"{end_hour:02d}:{end_minute:02d}:00"
                
                day_items.append(place)
        
        # Sort by start time within day
        day_items.sort(key=lambda x: x['startMinutes'])
        schedule.extend(day_items)
    
    return schedule


def fallback_schedule(places, days, day_offset, prefs):
    """Simple fallback scheduler when OR-Tools fails."""
    logger.info("Using fallback scheduler")
    
    # Take top places by score
    sorted_places = sorted(places, key=lambda p: p.get('timeScore', 0), reverse=True)
    target = min(days * 3, len(sorted_places))
    selected = sorted_places[:target]
    
    # Distribute across days
    per_day = max(1, target // days)
    schedule = []
    
    for d in range(days):
        day_places = selected[d * per_day:(d + 1) * per_day]
        current_time = 8 * 60  # Start at 8:00
        
        for p in day_places:
            place = p.copy()
            place['dayNumber'] = day_offset + d + 1
            place['startMinutes'] = current_time
            
            hour = current_time // 60
            minute = current_time % 60
            place['startTime'] = f"{hour:02d}:{minute:02d}:00"
            
            duration = p.get('visitDurationMinutes') or 90  # Default 90 if None
            end_time = current_time + duration
            end_hour = end_time // 60
            end_minute = end_time % 60
            place['endTime'] = f"{end_hour:02d}:{end_minute:02d}:00"
            
            schedule.append(place)
            
            # Add gap for next activity (including travel)
            current_time = end_time + 60  # 1 hour gap
    
    return schedule

async def build_workflow(job,token):
    java=JavaClient(job,token);cheap=CachedModel("CHEAP_LLM");quality=CachedModel("QUALITY_LLM")
    async def validate(s): await java.event("VALIDATING",5,"ai_trip.validating");return s
    async def retrieve(s):
        await java.event("RETRIEVING",15,"ai_trip.retrieving")
        inv=await asyncio.gather(*(java.candidates(d) for d in job["request"]["destinations"]))
        return {"inventories":inv}
    
    async def fast_filter(s):
        """Fast filter: reduce pool from ~120-250 to top ~40-60 by relevance."""
        await java.event("FAST_FILTER",30,"ai_trip.fast_filter")
        prefs=job["request"].get("preferenceText",""); filtered=[]
        
        for d,inv in zip(job["request"]["destinations"],s["inventories"]):
            ordered=sorted(inv,key=lambda p:rank(p,prefs),reverse=True)
            # Send compact fields for 120 places
            compact_places = [{
                "id": p["id"], 
                "name": p.get("name",""), 
                "address": p.get("address",""),
                "rating": p.get("rating",0), 
                "types": p.get("types",[]),
                "attributes": p.get("attributes", {})
            } for p in ordered[:120]]
            
            target_count = max(40, destination_days(d) * 8)  # More generous pool
            
            cheap_result=await cheap.json(
                f"Choose top {target_count} most relevant place ids IN ORDER from the input based on preferences. Never invent ids. Return JSON with format: {{\"ids\":[\"id1\",\"id2\",...]}}",
                {"preferences":prefs,"places":compact_places,"count":target_count},
                max_tokens=2048
            )
            
            # Preserve LLM ordering if available
            if cheap_result and cheap_result.get("ids"):
                id_set = set(str(x) for x in cheap_result["ids"])
                selected = [p for p in ordered if str(p["id"]) in id_set]
                filtered.append(selected[:target_count])
            else:
                filtered.append(ordered[:target_count])
        
        return {"filtered":filtered}
    
    async def deep_select(s):
        """Deep select: LLM enriches with time options, meal types, scores."""
        await java.event("DEEP_SELECT",50,"ai_trip.deep_select")
        prefs=job["request"].get("preferenceText","")
        enriched=[]
        
        for d,places in zip(job["request"]["destinations"],s["filtered"]):
            # Send full details including opening hours for deep analysis
            full_places = [{
                "id": p["id"],
                "name": p.get("name",""),
                "address": p.get("address",""),
                "rating": p.get("rating",0),
                "types": p.get("types",[]),
                "attributes": p.get("attributes",{}),
                "description": p.get("description",""),
                "about": p.get("about",""),
                "category": p.get("category",""),
                "priceLevel": p.get("priceLevel"),
                "visitDurationMinutes": p.get("visitDurationMinutes"),
                "openingHours": p.get("openingHours",""),
                "weeklyOpeningHours": p.get("weeklyOpeningHours",[]),
                "openPeriods": p.get("openPeriods",[])
            } for p in places]
            
            system_prompt = """Analyze each place and return JSON array with this EXACT format for each:
{
  "id": "original_id",
  "isFoodVenue": true/false,
  "mealType": "breakfast|lunch|dinner|snack|null",
  "timeScore": 0-100 (higher=better fit for user preferences),
  "visitDurationMinutes": estimated_minutes,
  "dietaryTags": ["halal", "vegan", etc],
  "openingHoursValid": true/false (false if no opening hours data)
}

Rules:
- Never invent IDs, use exact input IDs
- isFoodVenue=true for restaurants, cafes, food stalls
- mealType based on ACTUAL opening hours from openingHours/weeklyOpeningHours/openPeriods data
- If no opening hours data available, set openingHoursValid=false and guess conservatively
- timeScore reflects how well place matches user preferences (NOT time-of-day preference)
- visitDurationMinutes: food venues 60-90min, attractions 90-180min, theme parks 180-360min
- Output JSON array with ALL input places"""

            quality_result=await quality.json(
                system_prompt,
                {"preferences":prefs,"places":full_places},
                max_tokens=8192
            )
            
            if quality_result and isinstance(quality_result, list):
                # Merge enrichment back to original places
                enrich_map = {str(e["id"]): e for e in quality_result if "id" in e}
                for p in places:
                    if str(p["id"]) in enrich_map:
                        p.update(enrich_map[str(p["id"])])
                enriched.append(places)
            else:
                # Fallback: use original places with default values
                logger.warning("Deep select LLM failed, using defaults")
                for p in places:
                    types_str = str(p.get("types",[])).lower()
                    category_str = str(p.get("category","")).lower()
                    p.setdefault("isFoodVenue", "restaurant" in types_str or "food" in category_str or "cafe" in types_str)
                    p.setdefault("mealType", None)
                    p.setdefault("timeScore", 50)
                    p.setdefault("openingHoursValid", False)
                enriched.append(places)
        
        return {"enriched":enriched}
    
    async def select(s):
        """Select: OR-Tools scheduling with meal & time constraints."""
        await java.event("SCHEDULING",65,"ai_trip.scheduling")
        selected=[]
        day_offset = 0
        
        for d,places in zip(job["request"]["destinations"],s["enriched"]):
            days = destination_days(d)
            scheduled = solve_schedule(places, days, day_offset, job["request"].get("preferenceText",""))
            selected.append(scheduled)
            day_offset += days  # Increment for next destination
        
        return {"selected":selected}
    async def schedule(s):
        """Schedule: build transport items between places that already have times from solver."""
        await java.event("BUILDING_ITINERARY",75,"ai_trip.building_itinerary")
        req=job["request"];items=[];last_in_day={}
        
        # Flatten all places from all destinations
        all_places = []
        for places_list in s["selected"]:
            all_places.extend(places_list)
        
        # Sort by day and start time
        all_places.sort(key=lambda p: (p.get('dayNumber', 1), p.get('startMinutes', 0)))
        
        for p in all_places:
            day = p.get('dayNumber', 1)
            last = last_in_day.get(day)
            
            # Add transport if there's a previous place in same day
            if last and last.get("latitude") is not None and p.get("latitude") is not None:
                km = haversine(last, p)
                
                # Realistic travel time
                if km < 1.5:
                    mins = max(10, int(km / 4.5 * 60))  # Walking speed 4.5 km/h
                    mode = "WALKING"
                else:
                    mins = max(5, int(km / 20 * 60))    # Taxi speed 20 km/h
                    mode = "TAXI"
                
                # Transport timing: end of last activity -> start of current activity
                transport_start_minutes = last.get('startMinutes', 0) + (last.get('visitDurationMinutes') or 90)
                transport_end_minutes = p.get('startMinutes', 0)
                
                # Validate: transport end must be after start
                if transport_end_minutes > transport_start_minutes:
                    transport_start_hour = transport_start_minutes // 60
                    transport_start_min = transport_start_minutes % 60
                    transport_end_hour = transport_end_minutes // 60
                    transport_end_min = transport_end_minutes % 60
                    
                    items.append({
                        "type":"TRANSPORT",
                        "name":f"Di chuyển đến {p.get('title', p.get('name',''))}",
                        "dayNumber":day,
                        "sortOrder":len(items),
                        "startTime":f"{transport_start_hour:02d}:{transport_start_min:02d}:00",
                        "endTime":f"{transport_end_hour:02d}:{transport_end_min:02d}:00",
                        "address":last.get("address"),
                        "latitude":last.get("latitude"),
                        "longitude":last.get("longitude"),
                        "endAddress":p.get("address"),
                        "endLatitude":p.get("latitude"),
                        "endLongitude":p.get("longitude"),
                        "transportMode":mode,
                        "durationValueToNext":mins*60,
                        "durationToNext":f"{mins} phút",
                        "distanceValueToNext":int(km*1000),
                        "distanceToNext":f"{km:.1f} km"
                    })
                else:
                    logger.warning(f"Skipping transport: end ({transport_end_minutes}) <= start ({transport_start_minutes})")
            
            # Add place visit (times already set by solver)
            items.append({
                "type":"PLACE_VISIT",
                "placeId":p["id"],
                "name":p.get("title", p.get("name","")),
                "dayNumber":day,
                "sortOrder":len(items),
                "startTime":p.get('startTime', '09:00:00'),
                "endTime":p.get('endTime', '10:30:00'),
                "category":str(p.get("placeGroup") or p.get("category") or "activity").lower()
            })
            
            # Update last place for this day
            last_in_day[day] = p
        
        # Inter-destination transport (simplified markers)
        destinations = req["destinations"]
        if len(destinations) > 1:
            current_offset = 0
            for dest_idx in range(len(destinations) - 1):
                dest = destinations[dest_idx]
                next_dest = destinations[dest_idx + 1]
                
                # Find last day of current destination
                current_dest_days = sum(destination_days(destinations[i]) for i in range(dest_idx + 1))
                last_day = current_dest_days
                
                km = haversine(dest, next_dest)
                mode = "PLANE" if km > 500 else "CAR"
                
                items.append({
                    "type":"TRANSPORT",
                    "name":"Di chuyển sang điểm đến tiếp theo",
                    "dayNumber":last_day,
                    "endDayNumber":last_day + 1,
                    "sortOrder":len(items),
                    "startTime":"20:00:00",
                    "endTime":"08:00:00",
                    "address":dest.get("address", ""),
                    "latitude":dest.get("latitude"),
                    "longitude":dest.get("longitude"),
                    "endAddress":next_dest.get("address", ""),
                    "endLatitude":next_dest.get("latitude"),
                    "endLongitude":next_dest.get("longitude"),
                    "transportMode":mode,
                    "distanceValueToNext":int(km*1000),
                    "distanceToNext":f"{km:.1f} km"
                })
        
        # Final sort and reindex
        items.sort(key=lambda x:(x["dayNumber"], x.get("startTime",""), x["type"]))
        for i,x in enumerate(items):
            x["sortOrder"]=i
        
        return {"items":items}
    async def verify(s):
        await java.event("VERIFYING",88,"ai_trip.verifying")
        # Build enriched items with full place details for quality LLM
        enriched_items = []
        place_map = {}
        for places_list in s.get("enriched", s.get("selected", [])):
            for p in places_list:
                place_map[p["id"]] = p
        
        for item in s["items"]:
            enriched = item.copy()
            if item["type"] == "PLACE_VISIT" and item.get("placeId"):
                place = place_map.get(item["placeId"])
                if place:
                    enriched["placeDetails"] = {
                        "description": place.get("description", ""),
                        "about": place.get("about", ""),
                        "attributes": place.get("attributes", {}),
                        "category": place.get("category", ""),
                        "rating": place.get("rating", 0)
                    }
            enriched_items.append(enriched)
        
        result=await quality.json(
            "Verify this itinerary using only supplied facts. Provide a trip description and reasons for each item. Return JSON with format: {\"description\":\"...\",\"reasons\":{\"itemName\":\"reason\",...}}",
            {"preferences":job["request"].get("preferenceText",""),"items":enriched_items},
            max_tokens=8192
        )
        desc=(result or {}).get("description") or "Lịch trình được cân bằng theo thời gian, khoảng cách và sở thích đã cung cấp."
        reasons=(result or {}).get("reasons",{})
        for x in s["items"]:
            if x["type"]!="TRANSPORT":x["description"]=reasons.get(x["name"],x.get("description") or "Phù hợp với nhịp độ và tuyến đường trong ngày.")
        return {"items":s["items"],"description":desc}
    async def commit(s): await java.event("SAVING",94,"ai_trip.saving");await java.commit(s["items"],s["description"]);return s
    graph=StateGraph(State)
    for name,node in [("validate",validate),("retrieve",retrieve),("fast_filter",fast_filter),("deep_select",deep_select),("select",select),("schedule",schedule),("verify",verify),("commit",commit)]:graph.add_node(name,node)
    graph.add_edge(START,"validate");graph.add_edge("validate","retrieve");graph.add_edge("retrieve","fast_filter");graph.add_edge("fast_filter","deep_select");graph.add_edge("deep_select","select");graph.add_edge("select","schedule");graph.add_edge("schedule","verify");graph.add_edge("verify","commit");graph.add_edge("commit",END)
    return graph.compile()

async def run_job(job,token):
    java=JavaClient(job,token)
    try:
        graph=await build_workflow(job,token);await graph.ainvoke({"job":job})
    except Exception as exc:
        try: await java.event("FAILED",0,"ai_trip.failed","FAILED",str(exc)[:1000])
        except Exception: pass
        raise
