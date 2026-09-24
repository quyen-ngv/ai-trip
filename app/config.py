"""Environment-driven settings for the AI trip worker."""
from __future__ import annotations
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# Java callback auth
INTERNAL_TOKEN = os.getenv("AI_TRIP_INTERNAL_TOKEN", "")

# Candidate retrieval
CANDIDATES_PER_GROUP = _int("AI_TRIP_CANDIDATES_PER_GROUP", 120)
CANDIDATE_RADIUS_KM = 50.0  # mirrors PlaceMapper.findActiveForAiWithinRadius

# AI browser research. The same cheap model writes queries and invokes its Responses API
# ``web_search`` tool; no separate browser provider credentials are required.
WEB_RESEARCH_BUDGET_SECONDS = _int("AI_TRIP_WEB_RESEARCH_BUDGET_SECONDS", 300)
WEB_RESEARCH_MIN_CANDIDATES_PER_GROUP = max(1, _int("AI_TRIP_WEB_RESEARCH_MIN_CANDIDATES_PER_GROUP", 3))
# One narrow request can be enough for a simple group, while richer requests can use up to three
# independently phrased AI queries.  Each query must cite at least three Web sources by default.
WEB_RESEARCH_MIN_QUERIES_PER_GROUP = max(1, _int("AI_TRIP_WEB_RESEARCH_MIN_QUERIES_PER_GROUP", 1))
WEB_RESEARCH_MAX_QUERIES_PER_GROUP = max(
    WEB_RESEARCH_MIN_QUERIES_PER_GROUP,
    _int("AI_TRIP_WEB_RESEARCH_MAX_QUERIES_PER_GROUP", 3),
)
WEB_RESEARCH_MIN_SOURCES_PER_QUERY = max(2, _int("AI_TRIP_WEB_RESEARCH_MIN_SOURCES_PER_QUERY", 3))
WEB_RESEARCH_MAX_TOOL_CALLS = max(
    WEB_RESEARCH_MIN_SOURCES_PER_QUERY,
    _int("AI_TRIP_WEB_RESEARCH_MAX_TOOL_CALLS", 6),
)
WEB_RESEARCH_TIMEOUT_SECONDS = _int("AI_TRIP_WEB_RESEARCH_TIMEOUT_SECONDS", 180)
WEB_RESEARCH_CACHE_TTL_SECONDS = max(0, _int("AI_TRIP_WEB_RESEARCH_CACHE_TTL_SECONDS", 21600))
WEB_CURATION_CACHE_TTL_SECONDS = max(0, _int("AI_TRIP_WEB_CURATION_CACHE_TTL_SECONDS", 21600))
WEB_CURATION_MAX_PER_GROUP = max(1, _int("AI_TRIP_WEB_CURATION_MAX_PER_GROUP", 12))
RESEARCH_CACHE_PATH = os.getenv("AI_TRIP_RESEARCH_CACHE_PATH", "/tmp/ai_trip_research_cache.sqlite3")

# LLM
LLM_TIMEOUT_SECONDS = _int("AI_TRIP_LLM_TIMEOUT_SECONDS", 180)
