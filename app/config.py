"""Environment-driven settings for the AI trip worker."""
from __future__ import annotations
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# Candidate retrieval
CANDIDATES_PER_GROUP = _int("AI_TRIP_CANDIDATES_PER_GROUP", 120)

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

# Social video extraction (the media pipeline runs in google-maps-bot, which calls /v1/social/*).
# Provider, keys and models come from the AI_PROVIDER switch, read by `providers`.
AI_REQUEST_TIMEOUT_SECONDS = _int("AI_REQUEST_TIMEOUT_SECONDS", 120)
# Only sent to OpenAI reasoning-capable models (o-series, gpt-5+). Empty means "low" for the
# extraction and, for the topic pre-filter, "none" where the model accepts it (gpt-5.1+, gpt-6).
# The API default ("medium") made the extraction call take ~110s against a 120s timeout.
SOCIAL_AI_REASONING_EFFORT = os.getenv("SOCIAL_AI_REASONING_EFFORT", "").strip().lower()

# Speech-to-text: Groq, else OpenAI, else faster-whisper on this machine.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
LOCAL_WHISPER_ENABLED = os.getenv("LOCAL_WHISPER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "small")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")
LOCAL_WHISPER_COMPUTE_TYPE = os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8")
LOCAL_WHISPER_LANGUAGE = os.getenv("LOCAL_WHISPER_LANGUAGE", "vi")
LOCAL_WHISPER_CACHE_DIR = os.getenv("LOCAL_WHISPER_CACHE_DIR", "")
# Greedy decoding (1) is ~2-3x faster on CPU than faster-whisper's default beam of 5.
LOCAL_WHISPER_BEAM_SIZE = max(1, _int("LOCAL_WHISPER_BEAM_SIZE", 1))
# 0 lets CTranslate2 pick, which counts host cores, not the container's CPU quota.
LOCAL_WHISPER_CPU_THREADS = max(0, _int("LOCAL_WHISPER_CPU_THREADS", 2))
