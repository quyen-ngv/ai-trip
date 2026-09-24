"""AI-led Web research for itinerary candidates.

The chat model first writes narrow, request-aware search queries. A separate OpenAI Responses
model must then use the built-in Web Search tool, read independent editorial sources, and return
only named places backed by the URLs it used. This intentionally does *not* call Google Maps or
the place-scraping worker: Web discoveries remain external ``ACTIVITY`` candidates until a later
catalogue/import flow resolves them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from . import config
from .candidates import ACTIVITY_GROUPS, FOOD, normalize
from .llm import CachedModel
from .research_cache import ResearchCache

logger = logging.getLogger(__name__)

Reporter = Callable[[str, dict[str, Any]], Awaitable[None]]

_QUERY_SYSTEM = """You write Web-search queries for a trip planner. Return JSON only.

Create the requested number of genuinely different, natural-language Web queries for every
requested place group. Queries must name the destination and faithfully incorporate the traveller
request: dietary restrictions (for example Halal), pace, travel style, group composition,
children/elderly/accessibility, activity interests, discovery style, budget and free text.

The goal is evidence from editorial/local-guide articles, not a map or business directory. For
example: \"top halal restaurants in Da Nang local guide\", \"hidden gems in Da Nang for a
relaxed family itinerary\", or \"best accessible cultural sights in Hue for seniors\". Never ask
for Google Maps or map results.

Return {\"queries\":[{\"group\": one requested group, \"query\": string}]} with exactly the
requested count for each group. Allowed groups are FOOD_AND_DRINK, CULTURE_AND_HERITAGE,
NATURE_AND_OUTDOORS, SHOPPING_AND_MARKET, ATTRACTIONS."""

_RESEARCH_INSTRUCTIONS = """You are a travel web-research agent. You MUST use Web Search before
answering. Search the Web for the supplied query, then read and cross-check at least the requested
number of independent editorial, tourism-board, publication, or credible local-guide sources.

Do not use Google Maps, maps.google.com, a map listing, a generic business directory, or search
result snippets as evidence. Extract only concrete places actually named by the sources. Do not
invent a venue, address, coordinates, opening hours, rating, or dietary/accessibility claim. A
place can be returned only if it has at least one matching source URL from the Web Search sources.

Return JSON only in this exact shape:
{
  \"sources\": [{\"url\": \"https://...\", \"title\": \"...\"}],
  \"places\": [{
    \"name\": \"...\", \"address\": \"... or empty\", \"category\": \"... or empty\",
    \"reason\": \"short reason grounded in the sources and the traveller request\",
    \"sourceUrls\": [\"https://...\"]
  }]
}
"""

_CURATE_SYSTEM = """You curate cited Web discoveries for a trip planner. Return JSON only.

Choose only candidateIds supplied in the evidence; never invent or rename a place. Keep diverse
choices that fit the traveller request, including dietary rules, pace, children, older travellers,
accessibility, budget and discovery style. Remove weak, duplicate, irrelevant, or poorly evidenced
items. Return {\"candidateIds\":[\"...\"]}."""


def _clean_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _source_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if url.startswith(("https://", "http://")) else ""


def _normalise_group(value: Any) -> str | None:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().upper())
    aliases = {
        "FOOD": FOOD,
        "RESTAURANTS": FOOD,
        "RESTAURANT": FOOD,
        "CULTURE": "CULTURE_AND_HERITAGE",
        "NATURE": "NATURE_AND_OUTDOORS",
        "SHOPPING": "SHOPPING_AND_MARKET",
        "ATTRACTION": "ATTRACTIONS",
        "SPOTS": "ATTRACTIONS",
    }
    text = aliases.get(text, text)
    return text if text in {FOOD, *ACTIVITY_GROUPS} else None


def _output_text(response: dict[str, Any]) -> str:
    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            value = content.get("text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return "\n".join(parts)


def _parse_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text or "", flags=re.S).strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _response_sources(response: dict[str, Any]) -> dict[str, str]:
    """Extract URLs that the Responses API says the web-search call/citations used.

    Output layouts vary slightly between Responses API versions. Walk only Web Search action
    sources and message annotations, rather than trusting arbitrary URLs written in model text.
    """
    found: dict[str, str] = {}

    def collect(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                collect(child)
            return
        if not isinstance(value, dict):
            return
        url = _source_url(value.get("url"))
        if url:
            found.setdefault(url, _clean_text(value.get("title") or value.get("text"), 180) or url)
        for child in value.values():
            if isinstance(child, (dict, list)):
                collect(child)

    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            collect(((item.get("action") or {}).get("sources")))
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if isinstance(content, dict):
                    collect(content.get("annotations"))
    return found


class WebResearcher:
    """Create AI queries and turn cited generic-Web results into external candidates."""

    def __init__(
        self,
        *,
        locale: str = "vi",
        request: dict[str, Any] | None = None,
        query_model: CachedModel | Any | None = None,
        reporter: Reporter | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        min_queries_per_group: int | None = None,
        max_queries_per_group: int | None = None,
        min_sources_per_query: int | None = None,
        max_tool_calls: int | None = None,
    ):
        self.locale = "vi" if (locale or "").lower().startswith("vi") else "en"
        self.request = request if isinstance(request, dict) else {}
        # Research is intentionally cheap: one CHEAP_LLM model writes the query and invokes
        # Responses ``web_search``. There is no quality-model fallback for this stage.
        self.query_model = query_model or CachedModel("CHEAP_LLM")
        self.reporter = reporter
        self.base_url = (base_url if base_url is not None else getattr(self.query_model, "url", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else getattr(self.query_model, "key", "")
        self.model = model if model is not None else getattr(self.query_model, "model", "")
        self.min_queries = max(1, min_queries_per_group if min_queries_per_group is not None
                               else config.WEB_RESEARCH_MIN_QUERIES_PER_GROUP)
        self.max_queries = max(self.min_queries, max_queries_per_group if max_queries_per_group is not None
                               else config.WEB_RESEARCH_MAX_QUERIES_PER_GROUP)
        self.min_sources = max(2, min_sources_per_query if min_sources_per_query is not None
                               else config.WEB_RESEARCH_MIN_SOURCES_PER_QUERY)
        self.max_tool_calls = max(self.min_sources, max_tool_calls if max_tool_calls is not None
                                  else config.WEB_RESEARCH_MAX_TOOL_CALLS)
        self.cache = ResearchCache(config.RESEARCH_CACHE_PATH)
        # Test/local adapters must execute their supplied fake requests deterministically. Real
        # worker models are CachedModel instances and receive the persistent node cache.
        self.cache_enabled = isinstance(self.query_model, CachedModel)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model and getattr(self.query_model, "configured", True))

    @property
    def endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/responses") else f"{self.base_url}/responses"

    def _requested_query_count(self, target: int) -> int:
        # Bigger gaps get more independently phrased searches, bounded for predictable spend.
        return min(self.max_queries, max(self.min_queries, (max(1, int(target)) + 2) // 3))

    def _request_context(self) -> dict[str, Any]:
        fields = (
            "pace", "placeGroups", "travelStyle", "groupComposition", "dietaryRestrictions",
            "mobilityConsiderations", "activityTypes", "discoveryStyle", "budgetMin", "budgetMax",
            "budgetCurrency", "preferenceText",
        )
        return {field: self.request[field] for field in fields if self.request.get(field) not in (None, "", [])}

    def _cache_input(self, destination: dict[str, Any], needs: dict[str, int]) -> dict[str, Any]:
        return {
            "model": self.model,
            "destination": {key: destination.get(key) for key in ("name", "latitude", "longitude")},
            "needs": needs,
            "request": self._request_context(),
            "locale": self.locale,
            "minSources": self.min_sources,
        }

    async def plan_queries(self, destination: dict[str, Any], needs: dict[str, int]) -> dict[str, list[str]]:
        targets = {_normalise_group(group): int(target) for group, target in needs.items() if _normalise_group(group)}
        if not targets:
            return {}
        per_group = {group: self._requested_query_count(target) for group, target in targets.items()}
        payload = {
            "destination": {
                "name": _clean_text(destination.get("name"), 160),
                "latitude": destination.get("latitude"),
                "longitude": destination.get("longitude"),
            },
            "travellerRequest": self._request_context(),
            "researchTargets": targets,
            "queriesRequiredPerGroup": per_group,
            "locale": self.locale,
        }
        planned = await self.query_model.json(_QUERY_SYSTEM, payload, max_tokens=2500, temperature=0.2, cache=False)
        if not isinstance(planned, dict) or not isinstance(planned.get("queries"), list):
            raise RuntimeError("AI could not create Web research queries")
        grouped: dict[str, list[str]] = {group: [] for group in targets}
        for item in planned["queries"]:
            if not isinstance(item, dict):
                continue
            group = _normalise_group(item.get("group"))
            query = _clean_text(item.get("query"), 360)
            if group in grouped and query and query not in grouped[group]:
                grouped[group].append(query)
        missing = {group: count for group, count in per_group.items() if len(grouped[group]) < count}
        if missing:
            raise RuntimeError(f"AI returned too few Web research queries: {missing}")
        return {group: queries[:per_group[group]] for group, queries in grouped.items()}

    async def _request(self, query: str, destination: dict[str, Any], group: str) -> dict[str, Any] | None:
        body = {
            "model": self.model,
            "instructions": _RESEARCH_INSTRUCTIONS,
            "input": json.dumps({
                "query": query,
                "destination": _clean_text(destination.get("name"), 160),
                "placeGroup": group,
                "travellerRequest": self._request_context(),
                "minimumIndependentSources": self.min_sources,
            }, ensure_ascii=False),
            "tools": [{"type": "web_search", "search_context_size": "high"}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "max_tool_calls": self.max_tool_calls,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "trip_web_research",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["sources", "places"],
                        "properties": {
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object", "additionalProperties": False,
                                    "required": ["url", "title"],
                                    "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
                                },
                            },
                            "places": {
                                "type": "array",
                                "items": {
                                    "type": "object", "additionalProperties": False,
                                    "required": ["name", "address", "category", "reason", "sourceUrls"],
                                    "properties": {
                                        "name": {"type": "string"}, "address": {"type": "string"},
                                        "category": {"type": "string"}, "reason": {"type": "string"},
                                        "sourceUrls": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=config.WEB_RESEARCH_TIMEOUT_SECONDS) as client:
                response = await client.post(self.endpoint, headers=headers, json=body)
            if response.status_code != 200:
                logger.warning("Web research failed (%s): %s", response.status_code, response.text[:400])
                return None
            value = response.json()
            return value if isinstance(value, dict) else None
        except Exception as exc:
            logger.warning("Web research request failed for %r: %s", query, exc)
            return None

    def _rows_from_response(self, response: dict[str, Any], group: str, known_titles: set[str]) -> list[dict]:
        result = _parse_json(_output_text(response))
        cited_sources = _response_sources(response)
        if not result or len(cited_sources) < self.min_sources:
            logger.warning("Web research response had %d cited sources; need %d", len(cited_sources), self.min_sources)
            return []
        rows: list[dict] = []
        for item in result.get("places") or []:
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("name"), 255)
            title_key = title.casefold()
            urls = list(dict.fromkeys(url for url in (item.get("sourceUrls") or []) if _source_url(url) in cited_sources))
            if not title or title_key in known_titles or not urls:
                continue
            candidate_id = "web:" + hashlib.sha256((title_key + "|" + "|".join(sorted(urls))).encode()).hexdigest()[:24]
            row = normalize({
                "candidateId": candidate_id,
                "title": title,
                # Web evidence is not an authoritative location record. Only Java's later
                # catalogue/Goong resolver may attach an address or coordinates.
                "address": "",
                "placeGroup": group,
                "category": _clean_text(item.get("category"), 120),
                "description": _clean_text(item.get("reason"), 300),
                "why": _clean_text(item.get("reason"), 220),
                "sourceUrls": urls[:3],
                "sourceTitles": [cited_sources[url] for url in urls[:3]],
                "resolutionStatus": "RAW_WEB",
            }, source="web")
            if row["title"]:
                rows.append(row)
                known_titles.add(title_key)
        return rows

    async def research_destination(self, destination: dict[str, Any], needs: dict[str, int], known: list[dict]) -> list[dict]:
        if not needs:
            return []
        if not self.enabled:
            raise RuntimeError("AI Web research requires a Responses/Web Search-capable CHEAP_LLM configuration")
        cache_key = ResearchCache.key("web-research-v2", self._cache_input(destination, needs))
        cached = self.cache.get(cache_key) if self.cache_enabled else None
        if isinstance(cached, list):
            logger.info("Web research cache hit for %s", destination.get("name"))
            return [normalize(row, source="web") for row in cached if isinstance(row, dict)]
        query_plan = await self.plan_queries(destination, needs)
        known_titles = {_clean_text(row.get("title"), 255).casefold() for row in known if row.get("title")}
        rows: list[dict] = []
        for group, queries in query_plan.items():
            for index, query in enumerate(queries, start=1):
                response = await self._request(query, destination, group)
                if response:
                    rows.extend(self._rows_from_response(response, group, known_titles))
                if self.reporter:
                    await self.reporter("web_search", {
                        "destination": destination.get("name") or "",
                        "group": group,
                        "query": index,
                        "queries": len(queries),
                        "found": len([row for row in rows if row["placeGroup"] == group]),
                    })
        if self.cache_enabled:
            self.cache.put(cache_key, rows, config.WEB_RESEARCH_CACHE_TTL_SECONDS)
        return rows

    async def curate_candidates(self, destination: dict[str, Any], rows: list[dict]) -> list[dict]:
        """Cheap LLM shortlist between Web evidence and Java identity resolution."""
        if not rows:
            return []
        evidence = [{
            "candidateId": row.get("candidateId"), "title": row.get("title"),
            "group": row.get("placeGroup"), "category": row.get("category"),
            "reason": row.get("why") or row.get("description"), "sources": row.get("sourceUrls") or [],
        } for row in rows]
        cache_value = {"destination": self._cache_input(destination, {}), "candidates": evidence,
                       "limitPerGroup": config.WEB_CURATION_MAX_PER_GROUP}
        cache_key = ResearchCache.key("web-curation-v1", cache_value)
        cached = self.cache.get(cache_key) if self.cache_enabled else None
        if isinstance(cached, list):
            ids = {str(value) for value in cached}
        else:
            result = await self.query_model.json(_CURATE_SYSTEM, {
                "destination": _clean_text(destination.get("name"), 160),
                "travellerRequest": self._request_context(),
                "maxPerGroup": config.WEB_CURATION_MAX_PER_GROUP,
                "candidates": evidence,
            }, max_tokens=3000, temperature=0.1, cache=False)
            if not isinstance(result, dict) or not isinstance(result.get("candidateIds"), list):
                raise RuntimeError("AI could not curate Web research candidates")
            ids = {str(value) for value in result["candidateIds"]}
            if not ids:
                raise RuntimeError("AI Web curation returned no supported candidates")
            if self.cache_enabled:
                self.cache.put(cache_key, sorted(ids), config.WEB_CURATION_CACHE_TTL_SECONDS)
        per_group: dict[str, int] = {}
        curated: list[dict] = []
        for row in rows:
            group = str(row.get("placeGroup") or "")
            if row.get("candidateId") not in ids or per_group.get(group, 0) >= config.WEB_CURATION_MAX_PER_GROUP:
                continue
            curated.append(row)
            per_group[group] = per_group.get(group, 0) + 1
        return curated
