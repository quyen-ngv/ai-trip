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

async def build_workflow(job,token):
    java=JavaClient(job,token);cheap=CachedModel("CHEAP_LLM");quality=CachedModel("CHEAP_LLM")  # Use CHEAP_LLM (flash) for both
    async def validate(s): await java.event("VALIDATING",5,"ai_trip.validating");return s
    async def retrieve(s):
        await java.event("RETRIEVING",15,"ai_trip.retrieving")
        inv=await asyncio.gather(*(java.candidates(d) for d in job["request"]["destinations"]))
        return {"inventories":inv}
    
    async def fast_filter(s):
        """Fast filter: rank ALL places by relevance with compact format."""
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
            # Send ALL places with compact format (like deep_select)
            compact_places = []
            for p in inv:
                # Build compact info string
                info_parts = []
                if p.get("name"):
                    info_parts.append(p["name"])
                if p.get("address"):
                    info_parts.append(p["address"])
                if p.get("category"):
                    info_parts.append(p["category"])
                
                place = {
                    "id": p["id"],
                    "info": ", ".join(info_parts) if info_parts else ""
                }
                
                # Add calculatedScore
                score = p.get("calculatedScore") or p.get("rating", 0)
                if score and score > 0:
                    place["score"] = score
                
                # Flatten attributes
                attrs = p.get("attributes", {})
                if attrs:
                    attr_parts = []
                    for key, val in attrs.items():
                        if isinstance(val, dict) and 'value' in val:
                            val = val['value']
                        if isinstance(val, bool):
                            attr_parts.append(f"{key}={str(val).lower()}")
                        elif val:
                            attr_parts.append(f"{key}={val}")
                    if attr_parts:
                        place["attributes"] = ", ".join(attr_parts)
                
                compact_places.append(place)
            
            cheap_result=await cheap.json(
                system_prompt + f"\n\nReturn JSON: {{\"ids\":[\"id1\",\"id2\",...]}}\nSelect ALL relevant place IDs in priority order (most relevant first).",
                {"places":compact_places},
                max_tokens=4096
            )
            
            # Preserve LLM ordering
            if cheap_result and cheap_result.get("ids"):
                id_to_place = {str(p["id"]): p for p in inv}
                selected = [id_to_place[pid] for pid in cheap_result["ids"] if pid in id_to_place]
                filtered.append(selected)
            else:
                # Fallback: ALL places sorted by calculatedScore or rating
                filtered.append(sorted(inv, key=lambda p: p.get("calculatedScore") or p.get("rating", 0), reverse=True))
        
        return {"filtered":filtered}
    
    async def deep_select(s):
        """Deep select: LLM generates full itinerary with scheduling for each day."""
        await java.event("DEEP_SELECT",50,"ai_trip.deep_select")
        prefs=job["request"].get("preferenceText","")
        all_items=[]
        
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
        
        day_offset = 0
        for d,places in zip(job["request"]["destinations"],s["filtered"]):
            days = destination_days(d)
            # Dynamic limit based on days
            max_places = min(len(places), days * 12)  # More places for LLM to choose
            places_to_send = places[:max_places]
            
            # Send compact format with description
            compact_places = []
            for p in places_to_send:
                # Build compact info string
                info_parts = []
                if p.get("name"):
                    info_parts.append(p["name"])
                if p.get("address"):
                    info_parts.append(p["address"])
                if p.get("category"):
                    info_parts.append(p["category"])
                
                place = {
                    "id": p["id"],
                    "info": ", ".join(info_parts) if info_parts else ""
                }
                
                # Add score
                score = p.get("calculatedScore") or p.get("rating", 0)
                if score and score > 0:
                    place["score"] = score
                
                # Flatten attributes
                attrs = p.get("attributes", {})
                if attrs:
                    attr_parts = []
                    for key, val in attrs.items():
                        if isinstance(val, dict) and 'value' in val:
                            val = val['value']
                        if isinstance(val, bool):
                            attr_parts.append(f"{key}={str(val).lower()}")
                        elif val:
                            attr_parts.append(f"{key}={val}")
                    if attr_parts:
                        place["attributes"] = ", ".join(attr_parts)
                
                # Add description if present
                if p.get("description"):
                    place["description"] = p["description"]
                
                # Add types as comma-separated string
                types = p.get("types", [])
                if types:
                    place["types"] = ", ".join(types)
                
                # Add openingHours if present
                if p.get("openingHours"):
                    place["openingHours"] = p["openingHours"]
                
                # Add visitDurationMinutes if present
                if p.get("visitDurationMinutes"):
                    place["visitDurationMinutes"] = p["visitDurationMinutes"]
                
                # Add lat/long for transport calculation
                if p.get("latitude"):
                    place["latitude"] = p["latitude"]
                if p.get("longitude"):
                    place["longitude"] = p["longitude"]
                
                compact_places.append(place)
            
            # Ask LLM to generate full itinerary (PLACE_VISIT items only)
            prompt = system_prompt + f"""

Generate a complete {days}-day itinerary with place visits only (NO transport items).

Return JSON array of items with this EXACT format:
[
  {{
    "type": "PLACE_VISIT",
    "placeId": "id from input",
    "name": "place name",
    "dayNumber": {day_offset + 1} to {day_offset + days},
    "sortOrder": 0,
    "startTime": "HH:MM:SS",
    "endTime": "HH:MM:SS",
    "category": "food_and_drink|attraction|activity|shopping|accommodation|nightlife"
  }},
  ...
]

CRITICAL RULES:
- Start each day around 08:00-09:00
- Include breakfast (08:00-09:30), lunch (12:00-13:30), dinner (18:00-21:00)
- Balance categories: max 40% food_and_drink, include attractions/activities
- End day around 21:00-22:00 for Balanced pace
- Leave realistic gaps between activities for travel time
- sortOrder: sequential (0, 1, 2, ...)
- NO TRANSPORT items, only PLACE_VISIT
"""

            quality_result = await quality.json(
                prompt,
                {"places": compact_places, "days": days, "dayOffset": day_offset},
                max_tokens=32768
            )
            
            logger.info(f"Deep select for destination {d.get('name', 'unknown')}: sent {len(compact_places)} places")
            
            if quality_result and isinstance(quality_result, list):
                logger.info(f"Successfully generated {len(quality_result)} itinerary items")
                all_items.extend(quality_result)
            else:
                # LLM failed - throw error (no fallback)
                logger.error(f"AI scheduling failed for destination {d.get('name', 'unknown')}")
                raise Exception("AI trip generation failed: LLM could not generate itinerary")
            
            day_offset += days
        
        # Sort and reindex
        all_items.sort(key=lambda x: (x.get("dayNumber", 1), x.get("sortOrder", 0)))
        for i, item in enumerate(all_items):
            item["sortOrder"] = i
        
        return {"items": all_items, "filtered": s["filtered"]}
    
    async def verify(s):
        await java.event("VERIFYING",88,"ai_trip.verifying")
        # Build enriched items with full place details for quality LLM
        enriched_items = []
        place_map = {}
        
        # Build place map from filtered list
        for places_list in s.get("filtered", []):
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
        
        # Check validation result - if violations found, trigger ONE retry
        if result:
            is_valid = result.get("valid", True)
            violations = result.get("violations", [])
            
            if not is_valid and violations and not s.get("retry_attempt"):
                # First validation failure - trigger retry of deep_select
                logger.warning(f"Itinerary has {len(violations)} violations, triggering retry:")
                for v in violations[:5]:
                    logger.warning(f"  - {v}")
                
                return {
                    **s,  # Preserve all state
                    "needs_retry": True,
                    "violations": violations,
                    "retry_attempt": True
                }
            elif not is_valid and violations and s.get("retry_attempt"):
                # Already retried - accept with violations
                logger.warning(f"Retry still has violations, accepting itinerary:")
                for v in violations[:5]:
                    logger.warning(f"  - {v}")
            
            desc = result.get("description") or "Lịch trình được cân bằng theo thời gian, khoảng cách và sở thích đã cung cấp."
            reasons = result.get("reasons", {})
        else:
            desc = "Lịch trình được cân bằng theo thời gian, khoảng cách và sở thích đã cung cấp."
            reasons = {}
        
        for x in s["items"]:
            if x["type"]!="TRANSPORT":x["description"]=reasons.get(x["name"],x.get("description") or "Phù hợp với nhịp độ và tuyến đường trong ngày.")
        
        return {"items":s["items"],"description":desc,"needs_retry":False}
    def should_retry(s):
        """Check if needs retry based on validation violations."""
        return "retry" if s.get("needs_retry") else "commit"
    
    async def commit(s): await java.event("SAVING",94,"ai_trip.saving");await java.commit(s["items"],s["description"]);return s
    
    graph=StateGraph(State)
    # Register nodes (removed select and schedule)
    for name,node in [("validate",validate),("retrieve",retrieve),("fast_filter",fast_filter),("deep_select",deep_select),("verify",verify),("commit",commit)]:graph.add_node(name,node)
    
    # Build graph edges
    graph.add_edge(START,"validate")
    graph.add_edge("validate","retrieve")
    graph.add_edge("retrieve","fast_filter")
    graph.add_edge("fast_filter","deep_select")
    graph.add_edge("deep_select","verify")
    
    # Conditional edge: retry goes back to deep_select, otherwise commit
    graph.add_conditional_edges("verify",should_retry,{"retry":"deep_select","commit":"commit"})
    graph.add_edge("commit",END)
    
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
