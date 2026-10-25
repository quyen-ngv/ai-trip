"""Pengoo's last-resort internet search, used when the catalogue has nothing.

Offered only when Java allows it for the turn (`limits.webSearch`, config ASSISTANT.WEB_SEARCH_ENABLED)
and a provider is configured here; the graph allows one call per turn. Results are short snippets
with their links. They never become place cards or proposals, because nothing in them is checked
against the catalogue.

Providers (ASSISTANT_WEB_SEARCH_PROVIDER):
  openai (default)  The model's own web search, the same way the trip planner researches
                    (`research.py`): OpenAI Responses web_search, or Claude's web_search tool when
                    the model is Claude, so it follows AI_PROVIDER. Uses ASSISTANT_WEB_SEARCH_BASE_URL /
                    _API_KEY / _MODEL, falling back to the CHEAP_LLM tier. Logged to ai_api_calls.
  claude            Claude's web_search tool through `providers.responses`. ASSISTANT_WEB_SEARCH_*
                    as above, falling back to the CHEAP_LLM tier when it is Claude, else to
                    CLAUDE_API_KEY with CLAUDE_CHEAP_MODEL.
  tinyfish          TinyFish Search (free up to 12,000 a day, 30 a minute); title, url and snippet
                    per result, no model call. Needs ASSISTANT_WEB_SEARCH_API_KEY.
  tavily | brave    A search API; needs ASSISTANT_WEB_SEARCH_API_KEY.

A search API that fails (rate limit, outage) hands the query to the CHEAP_LLM model (OpenAI or
Claude) when that tier is set, so the traveller still gets an answer; the ASSISTANT_WEB_SEARCH_*
values then belong to that API.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

from .. import ai_log, config, providers
from ..research import _output_text, _response_sources
from ..research_cache import ResearchCache

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
SNIPPET_CHARS = 300
TIMEOUT_SECONDS = 30
_TAG = re.compile(r"<[^>]+>")
# One search call per question: the second call used to open a page the first had already
# summarised, and OpenAI bills every call. Results are kept for a while: ticket prices, opening
# hours and event dates do not change between two travellers asking the same thing.
MAX_TOOL_CALLS = max(1, int(os.getenv("ASSISTANT_WEB_SEARCH_MAX_TOOL_CALLS", "1") or 1))
CACHE_TTL_SECONDS = max(0, int(os.getenv("ASSISTANT_WEB_SEARCH_CACHE_TTL_SECONDS", str(3 * 24 * 3600)) or 0))
_cache: ResearchCache | None = None
_cache_broken = False


def _cache_store() -> ResearchCache | None:
    global _cache, _cache_broken
    if CACHE_TTL_SECONDS <= 0 or _cache_broken:
        return None
    if _cache is None:
        try:
            _cache = ResearchCache(config.RESEARCH_CACHE_PATH)
        except Exception as exc:  # unwritable path: search still works, uncached
            logger.warning("assistant web search cache unavailable: %s", exc)
            _cache_broken = True
    return _cache

_INSTRUCTIONS = """Search the web for the query and answer for a traveller in Vietnam. Return JSON only:
{"results": [{"title": "...", "url": "https://...", "snippet": "the relevant fact in one or two sentences"}]}
Use at most 5 results, only from pages you actually read, prefer recent and local sources, and keep
each snippet factual (names, dates, address, price). When the query asks for places (a dish, a kind
of shop), give one result per place: its name, street or area, and what it is known for. No results →
{"results": []}."""


_MODEL_PROVIDERS = ("openai", "claude", "anthropic")


def _provider() -> str:
    return os.getenv("ASSISTANT_WEB_SEARCH_PROVIDER", "openai").strip().lower() or "openai"


def _model_endpoint() -> providers.Endpoint:
    """The model that searches. Behind a search API it is only the fallback, and
    ASSISTANT_WEB_SEARCH_* hold that API's key, so it runs on the planner's CHEAP_LLM tier."""
    cheap = providers.from_tiers("CHEAP_LLM")
    provider = _provider()
    if provider not in _MODEL_PROVIDERS:
        return cheap
    base = cheap
    if provider != "openai" and not cheap.is_claude:
        base = providers.provider_defaults("CLAUDE", cheap=True)

    def pick(key: str, fallback: str) -> str:
        return os.getenv(f"ASSISTANT_WEB_SEARCH_{key}", "").strip() or fallback
    return providers.Endpoint(pick("BASE_URL", base.base_url).rstrip("/"), pick("API_KEY", base.api_key),
                              pick("MODEL", base.model), "ASSISTANT_WEB_SEARCH")


def configured() -> bool:
    if _provider() in _MODEL_PROVIDERS:
        return _model_endpoint().configured
    return bool(os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip())


async def search(query: str) -> dict[str, Any]:
    """{"results": [{title, url, snippet}], "note": ...} or {"error": ...}; never raises."""
    query = (query or "").strip()[:200]
    if not query:
        return {"error": "query is required"}
    if not configured():
        return {"error": "Web search is not available."}
    provider = _provider()
    store = _cache_store()
    cache_key = ResearchCache.key("assistant-web-search-v1", {"provider": provider, "query": query.casefold()})
    cached = store.get(cache_key) if store else None
    if isinstance(cached, list) and cached:
        logger.info("assistant web search via %s: %d cached results", provider, len(cached))
        return _reply(cached)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            try:
                results = await _SEARCHERS.get(provider, _model_search)(client, query)
            except Exception as exc:
                if provider in _MODEL_PROVIDERS or not _model_endpoint().configured:
                    raise
                logger.warning("assistant web search failed (%s), asking the CHEAP_LLM model instead: %s", provider, exc)
                results = await _model_search(client, query)
    except Exception as exc:  # network, quota, bad JSON
        logger.warning("assistant web search failed (%s): %s", provider, exc)
        return {"error": "Web search is unavailable right now."}
    logger.info("assistant web search via %s: %d results", provider, len(results))
    if store and results:
        store.put(cache_key, results, CACHE_TTL_SECONDS)
    return _reply(results)


def _reply(results: list[dict]) -> dict[str, Any]:
    return {"results": results,
            "note": "Unverified web snippets. Mention them as 'theo <source>' with the link; no cards, no proposals."}


async def _model_search(client: httpx.AsyncClient, query: str) -> list[dict]:
    """OpenAI Responses web_search, or Claude's web_search tool when the endpoint is Claude."""
    body = {"instructions": _INSTRUCTIONS, "input": query,
            "tools": [{"type": "web_search", "search_context_size": "low"}], "tool_choice": "required",
            "include": ["web_search_call.action.sources"], "max_tool_calls": MAX_TOOL_CALLS, "store": False,
            "reasoning": {"effort": "low"}}
    result = await providers.responses(_model_endpoint(), body, operation="WEB_SEARCH",
                                       timeout=TIMEOUT_SECONDS, client=client)
    if not result.ok:
        raise RuntimeError(f"web search failed: {result.error}")
    value = result.value
    cited = _response_sources(value)
    try:
        parsed = providers.parse_json(_output_text(value) or "")
    except ValueError:
        parsed = {}
    rows = [_row(item.get("title"), item.get("url"), item.get("snippet"))
            for item in (parsed.get("results") if isinstance(parsed, dict) else None) or [] if isinstance(item, dict)]
    # Only links the search really returned; a model may otherwise make one up.
    rows = [row for row in rows if not cited or row["url"] in cited]
    if not rows and cited:
        rows = [_row(title, url, "") for url, title in list(cited.items())[:MAX_RESULTS]]
    return rows[:MAX_RESULTS]


async def _report_paid_search(provider: str, request: dict, response: httpx.Response, started: float) -> None:
    """Logs one paid search as a web search call of model "<provider>-search", so it is charged to
    the asker's AI credits once AI_PRICING has that model's webSearchPer1k."""
    ok = response.status_code == 200
    await ai_log.report(operation="WEB_SEARCH", provider=provider, model=f"{provider}-search", ok=ok,
                        request=request, response=response.json() if ok else None,
                        error=None if ok else {"httpStatus": response.status_code, "body": response.text[:4000]},
                        web_search_calls=1 if ok else 0, latency_ms=int((time.monotonic() - started) * 1000))


async def _tavily(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    body = {"query": query, "max_results": MAX_RESULTS, "search_depth": "basic", "include_answer": False}
    started = time.monotonic()
    response = await client.post("https://api.tavily.com/search",
                                 headers={"Authorization": f"Bearer {key}"}, json=body)
    await _report_paid_search("tavily", body, response, started)
    response.raise_for_status()
    return [_row(item.get("title"), item.get("url"), item.get("content"))
            for item in (response.json().get("results") or [])][:MAX_RESULTS]


async def _brave(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    params = {"q": query, "count": MAX_RESULTS}
    started = time.monotonic()
    response = await client.get("https://api.search.brave.com/res/v1/web/search",
                                headers={"X-Subscription-Token": key, "Accept": "application/json"},
                                params=params)
    await _report_paid_search("brave", params, response, started)
    response.raise_for_status()
    items = ((response.json().get("web") or {}).get("results")) or []
    return [_row(item.get("title"), item.get("url"), item.get("description")) for item in items][:MAX_RESULTS]


async def _tinyfish(client: httpx.AsyncClient, query: str) -> list[dict]:
    key = os.getenv("ASSISTANT_WEB_SEARCH_API_KEY", "").strip()
    params = {"query": query}
    started = time.monotonic()
    response = await client.get("https://api.search.tinyfish.ai", headers={"X-API-Key": key}, params=params)
    ok = response.status_code == 200
    value = response.json() if ok else None
    # Logged so the admin AI-calls page shows these searches too; no tokens, and free, so no
    # web_search_calls count that the price table would flag as unpriced.
    await ai_log.report(operation="WEB_SEARCH", provider="tinyfish", model="search", ok=ok, request=params,
                        response=value, error=None if ok else {"httpStatus": response.status_code,
                                                               "body": response.text[:4000]},
                        latency_ms=int((time.monotonic() - started) * 1000))
    response.raise_for_status()
    items = (value or {}).get("results") or []
    return [_row(item.get("title"), item.get("url"), item.get("snippet"))
            for item in items if isinstance(item, dict) and item.get("url")][:MAX_RESULTS]


_SEARCHERS = {"tinyfish": _tinyfish, "tavily": _tavily, "brave": _brave}


def _row(title: Any, url: Any, snippet: Any) -> dict:
    text = " ".join(_TAG.sub("", str(snippet or "")).split())
    return {"title": str(title or "")[:120], "url": str(url or ""),
            "snippet": text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")}
