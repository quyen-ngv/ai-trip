from __future__ import annotations
import asyncio, math, logging
from datetime import date
from typing import Any, TypedDict
import httpx
from langgraph.graph import StateGraph, START, END
from .llm import CachedModel
from .prompts import build_filter_prompt, build_enrichment_prompt, build_verify_prompt

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

def haversine(a,b):
    p1,p2=math.radians(float(a["latitude"])),math.radians(float(b["latitude"]));dp=p2-p1;dl=math.radians(float(b["longitude"])-float(a["longitude"]));q=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 6371*2*math.atan2(math.sqrt(q),math.sqrt(1-q))

def destination_days(d): return (date.fromisoformat(d["endDate"])-date.fromisoformat(d["startDate"])).days+1

def simple_schedule(places, days, day_offset):
    """Simple scheduler: top places by score, distribute across days."""
    if not places:
        return []
    
    sorted_places = sorted(places, key=lambda p: p.get('timeScore', 0), reverse=True)
    target = min(days * 3, len(sorted_places))
    selected = sorted_places[:target]
    
    per_day = max(1, target // days)
    schedule = []
    
    for d in range(days):
        day_places = selected[d * per_day:(d + 1) * per_day]
        current_time = 8 * 60
        
        for p in day_places:
            place = p.copy()
            place['dayNumber'] = day_offset + d + 1
            place['startMinutes'] = current_time
            
            hour = current_time // 60
            minute = current_time % 60
            place['startTime'] = f"{hour:02d}:{minute:02d}:00"
            
            duration = p.get('visitDurationMinutes') or 90
            end_time = current_time + duration
            end_hour = end_time // 60
            end_minute = end_time % 60
            place['endTime'] = f"{end_hour:02d}:{end_minute:02d}:00"
            
            schedule.append(place)
            current_time = end_time + 60
    
    return schedule

async def build_workflow(job,token):
    java=JavaClient(job,token);cheap=CachedModel("CHEAP_LLM");quality=CachedModel("CHEAP_LLM")  # Use CHEAP_LLM (flash) for both
    async def validate(s): await java.event("VALIDATING",5,"ai_trip.validating");return s
    async def retrieve(s):
        await java.event("RETRIEVING",15,"ai_trip.retrieving")
        inv=await asyncio.gather(*(java.candidates(d) for d in job["request"]["destinations"]))
        return {"inventories":inv}
    
    async def fast_filter(s):
        """Fast filter: reduce pool from ~120-250 to top ~40-60 by relevance."""
        await java.event("FAST_FILTER",30,"ai_trip.fast_filter")
        prefs=job["request"].get("preferenceText",""); filtered=[]
        
        # Build rich context prompt
        preferences_dict = {
            'placeGroups': job["request"].get("placeGroups", []),
            'pace': job["request"].get("pace", "BALANCED"),
            'groupComposition': job["request"].get("groupComposition", ""),
            'discoveryStyle': job["request"].get("discoveryStyle", "BALANCED"),
            'dietaryRestrictions': job["request"].get("dietaryRestrictions", []),
            'mobilityConsiderations': job["request"].get("mobilityConsiderations", []),
            'activityTypes': job["request"].get("activityTypes", []),
            'preferenceText': prefs,
            'budgetMin': job["request"].get("budgetMin"),
            'budgetMax': job["request"].get("budgetMax"),
        }
        
        system_prompt = build_filter_prompt(preferences_dict)
        
        for d,inv in zip(job["request"]["destinations"],s["inventories"]):
            # Send ALL places with minimal fields (just title + address)
            compact_places = [{
                "id": p["id"], 
                "name": p.get("name",""), 
                "address": p.get("address",""),
            } for p in inv]
            
            target_count = max(40, destination_days(d) * 8)
            
            cheap_result=await cheap.json(
                system_prompt + f"\n\nReturn JSON: {{\"ids\":[\"id1\",\"id2\",...]}}\nSelect top {target_count} most relevant place IDs in priority order.",
                {"places":compact_places,"targetCount":target_count},
                max_tokens=2048
            )
            
            # Preserve LLM ordering
            if cheap_result and cheap_result.get("ids"):
                id_set = set(str(x) for x in cheap_result["ids"])
                id_to_place = {str(p["id"]): p for p in inv}
                selected = [id_to_place[pid] for pid in cheap_result["ids"] if pid in id_to_place]
                filtered.append(selected[:target_count])
            else:
                # Fallback: top by rating
                filtered.append(sorted(inv, key=lambda p: p.get("rating", 0), reverse=True)[:target_count])
        
        return {"filtered":filtered}
    
    async def deep_select(s):
        """Deep select: LLM enriches with time options, meal types, scores."""
        await java.event("DEEP_SELECT",50,"ai_trip.deep_select")
        prefs=job["request"].get("preferenceText","")
        enriched=[]
        
        # Build rich context prompt
        preferences_dict = {
            'placeGroups': job["request"].get("placeGroups", []),
            'pace': job["request"].get("pace", "BALANCED"),
            'groupComposition': job["request"].get("groupComposition", ""),
            'discoveryStyle': job["request"].get("discoveryStyle", "BALANCED"),
            'dietaryRestrictions': job["request"].get("dietaryRestrictions", []),
            'mobilityConsiderations': job["request"].get("mobilityConsiderations", []),
            'activityTypes': job["request"].get("activityTypes", []),
            'preferenceText': prefs,
            'budgetMin': job["request"].get("budgetMin"),
            'budgetMax': job["request"].get("budgetMax"),
        }
        
        system_prompt = build_enrichment_prompt(preferences_dict)
        
        for d,places in zip(job["request"]["destinations"],s["filtered"]):
            days = destination_days(d)
            # Dynamic limit based on days: max 8 places per day
            max_places = min(len(places), days * 8)
            places_to_enrich = places[:max_places]
            
            # Send only non-empty fields to reduce token usage
            full_places = []
            for p in places_to_enrich:
                place = {"id": p["id"]}
                
                # Always include basic fields
                if p.get("name"):
                    place["name"] = p["name"]
                if p.get("address"):
                    place["address"] = p["address"]
                if p.get("category"):
                    place["category"] = p["category"]
                
                # Include numeric fields if meaningful
                rating = p.get("rating", 0)
                if rating and rating > 0:
                    place["rating"] = rating
                
                review_count = p.get("reviewCount", 0)
                if review_count and review_count > 0:
                    place["reviewCount"] = review_count
                
                # Include arrays if not empty
                types = p.get("types", [])
                if types:
                    place["types"] = types
                
                # Include attributes only if not empty, strip description & source_found
                attrs = p.get("attributes", {})
                if attrs:
                    # Keep only 'value' field from each attribute to save tokens
                    cleaned_attrs = {}
                    for key, val in attrs.items():
                        if isinstance(val, dict) and 'value' in val:
                            cleaned_attrs[key] = val['value']
                        elif not isinstance(val, dict):
                            # Keep primitive values as-is
                            cleaned_attrs[key] = val
                    
                    if cleaned_attrs:
                        place["attributes"] = cleaned_attrs
                
                # Include description/openingHours only if present
                desc = p.get("description", "")
                if desc:
                    place["description"] = desc
                
                opening = p.get("openingHours", "")
                if opening:
                    place["openingHours"] = opening
                
                # Include optional fields only if present
                if p.get("priceLevel"):
                    place["priceLevel"] = p["priceLevel"]
                
                if p.get("visitDurationMinutes"):
                    place["visitDurationMinutes"] = p["visitDurationMinutes"]
                
                full_places.append(place)
            
            # Retry logic for LLM call
            quality_result = None
            for attempt in range(3):
                quality_result = await quality.json(
                    system_prompt + "\n\nReturn JSON array with EXACT format specified in guidelines. Include ALL input places.",
                    {"places":full_places},
                    max_tokens=32768  # Max tokens for DeepSeek v4
                )
                
                if quality_result and isinstance(quality_result, list):
                    break
                    
                logger.warning(f"Deep select attempt {attempt+1}/3 failed: got {type(quality_result)}")
                if attempt < 2:
                    await asyncio.sleep(2)  # Wait before retry
            
            logger.info(f"Deep select for destination {d.get('name', 'unknown')}: sent {len(full_places)} places")
            logger.info(f"Result type: {type(quality_result)}")
            logger.info(f"Full result: {quality_result}")
            
            if quality_result and isinstance(quality_result, list):
                logger.info(f"Successfully enriched {len(quality_result)} places")
                # Merge enrichment back to original places
                enrich_map = {str(e["id"]): e for e in quality_result if "id" in e}
                for p in places_to_enrich:
                    if str(p["id"]) in enrich_map:
                        p.update(enrich_map[str(p["id"])])
                enriched.append(places_to_enrich)
            else:
                # LLM failed after 3 retries - ABORT
                error_msg = f"AI enrichment failed after 3 retries - got {type(quality_result).__name__} instead of list"
                logger.error(error_msg)
                logger.error(f"Full quality_result: {quality_result}")
                raise Exception(error_msg)
        
        return {"enriched":enriched}
    
    async def select(s):
        """Select: simple scheduling by timeScore."""
        await java.event("SCHEDULING",65,"ai_trip.scheduling")
        selected=[]
        day_offset = 0
        
        for d,places in zip(job["request"]["destinations"],s["enriched"]):
            days = destination_days(d)
            scheduled = simple_schedule(places, days, day_offset)
            selected.append(scheduled)
            day_offset += days
        
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
                    # Only include non-empty placeDetails fields
                    details = {}
                    
                    if place.get("description"):
                        details["description"] = place["description"]
                    
                    if place.get("about"):
                        details["about"] = place["about"]
                    
                    attrs = place.get("attributes", {})
                    if attrs:
                        details["attributes"] = attrs
                    
                    if place.get("category"):
                        details["category"] = place["category"]
                    
                    rating = place.get("rating", 0)
                    if rating and rating > 0:
                        details["rating"] = rating
                    
                    # Only add placeDetails if there's actual data
                    if details:
                        enriched["placeDetails"] = details
            
            enriched_items.append(enriched)
        
        # Build full preferences context
        preferences_dict = {
            'placeGroups': job["request"].get("placeGroups", []),
            'pace': job["request"].get("pace", "BALANCED"),
            'groupComposition': job["request"].get("groupComposition", ""),
            'discoveryStyle': job["request"].get("discoveryStyle", "BALANCED"),
            'dietaryRestrictions': job["request"].get("dietaryRestrictions", []),
            'mobilityConsiderations': job["request"].get("mobilityConsiderations", []),
            'activityTypes': job["request"].get("activityTypes", []),
            'preferenceText': job["request"].get("preferenceText", ""),
            'budgetMin': job["request"].get("budgetMin"),
            'budgetMax': job["request"].get("budgetMax"),
        }
        
        system_prompt = build_verify_prompt(preferences_dict)
        
        result=await quality.json(
            system_prompt,
            {"items":enriched_items},
            max_tokens=8192
        )
        
        # Check validation result
        if result:
            is_valid = result.get("valid", True)  # Default true for backward compatibility
            violations = result.get("violations", [])
            
            if not is_valid and violations:
                # Itinerary failed validation
                error_msg = "Itinerary validation failed:\n- " + "\n- ".join(violations)
                logger.error(error_msg)
                raise Exception(f"Constraint violations detected: {'; '.join(violations[:3])}")  # First 3 violations
            
            desc = result.get("description") or "Lịch trình được cân bằng theo thời gian, khoảng cách và sở thích đã cung cấp."
            reasons = result.get("reasons", {})
        else:
            desc = "Lịch trình được cân bằng theo thời gian, khoảng cách và sở thích đã cung cấp."
            reasons = {}
        
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
        logger.error(f"Job failed: {type(exc).__name__}: {str(exc)}")
        try: 
            await java.event("FAILED",0,"ai_trip.failed","FAILED",str(exc)[:1000])
        except Exception as report_err:
            logger.error(f"Failed to report error: {report_err}")
        raise
