"""The model steps of social video place extraction, called by google-maps-bot over /v1/social/*.

google-maps-bot downloads the video, cuts frames and audio, verifies candidates on Google Maps and
applies the evidence guardrail; everything that needs a model is here: the topic pre-filter, the
place extraction (transcript + frames + caption) and, in `transcription.py`, speech-to-text.

Provider and model come from the job (admin config SOCIAL_LOCATION.AI_PROVIDER / AI_MODEL), resolved
by `providers.from_provider`. Calls are logged under feature SOCIAL_LOCATION with the Java job id
as trace id, to the log URL the bot passes along.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from . import ai_log, config, providers

logger = logging.getLogger(__name__)

_OPENAI_REASONING_MODEL = re.compile(r"^(o\d|gpt-([5-9]|\d{2,})(?!-chat))", re.IGNORECASE)
# gpt-5.1 and later (gpt-6-luna included) accept reasoning_effort="none", which
# turns reasoning off; gpt-5/gpt-5-mini and the o-series do not.
_OPENAI_SUPPORTS_NO_REASONING = re.compile(r"^gpt-(5\.\d|[6-9]|\d{2,})", re.IGNORECASE)
# max_completion_tokens covers hidden reasoning AND the answer. A budget sized
# for the answer alone comes back as empty content with finish_reason=length.
_REASONING_EXTRACTION_MAX_TOKENS = 32000
_REASONING_PREFILTER_MAX_TOKENS = 2000
# Room for many places, each with its description, tips and evidence.
_EXTRACTION_MAX_TOKENS = 16000
_PREFILTER_MAX_TOKENS = 120
_PREFILTER_MAX_TRANSCRIPT_CHARS = 4000


def bind_log(url: str | None, token: str | None, trace_id: str | None):
    """Send this request's AI call log entries to Java at `url` (the bot derives it from the job's
    callback URL), with the job's internal token."""
    async def sink(entry: dict) -> None:
        headers = {"X-Internal-Token": token} if token else {}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, headers=headers, json=entry)
        if response.status_code >= 300:
            logger.warning("AI call log rejected (%s): %s", response.status_code, response.text[:200])

    return ai_log.bind(sink if url else None, feature="SOCIAL_LOCATION", trace_id=trace_id or None)


def _is_openai_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models reject any temperature but the default and spend
    max_completion_tokens on reasoning before writing the answer."""
    return bool(_OPENAI_REASONING_MODEL.match(model.strip()))


def _openai_reasoning_effort(model: str, *, thorough: bool = False) -> str:
    """`thorough` is the place extraction: a little reasoning makes the model walk the whole
    video and enumerate every stop. The topic pre-filter only needs a quick yes/no."""
    if config.SOCIAL_AI_REASONING_EFFORT:
        return config.SOCIAL_AI_REASONING_EFFORT
    if thorough:
        return "low"
    return "none" if _OPENAI_SUPPORTS_NO_REASONING.match(model.strip()) else "low"


def _limits(endpoint: providers.Endpoint, max_tokens: int, reasoning_max_tokens: int, *, thorough: bool) -> dict:
    """Token budget and sampling per provider: OpenAI reasoning models get an effort and room to
    think, other OpenAI models the new token field, DeepSeek and the rest the classic payload.
    Claude always thinks (adaptive) and spends max_tokens on it, so it gets the reasoning budget
    and a low effort for both calls."""
    if endpoint.is_claude:
        return {"reasoning_effort": config.SOCIAL_AI_REASONING_EFFORT or "low", "max_tokens": reasoning_max_tokens}
    if endpoint.name == "OPENAI" and _is_openai_reasoning_model(endpoint.model):
        return {"reasoning_effort": _openai_reasoning_effort(endpoint.model, thorough=thorough),
                "max_completion_tokens": reasoning_max_tokens}
    if endpoint.name == "OPENAI":
        # Current OpenAI models reject the legacy max_tokens field.
        return {"max_completion_tokens": max_tokens} if thorough else {"temperature": 0, "max_completion_tokens": max_tokens}
    return {"temperature": 0, "max_tokens": max_tokens}


def _unusable(endpoint: providers.Endpoint, result: providers.Result) -> str:
    error = result.error or {}
    if "httpStatus" in error:
        return f"{endpoint.name} chat completion failed with HTTP {error['httpStatus']}: {str(error.get('body'))[:1000]}"
    return f"{endpoint.name} returned no usable answer: {error.get('reason')} {error.get('message') or ''}".strip()


async def prefilter(*, metadata: dict[str, Any], transcript: str, ai_provider: str | None = None,
                    ai_model: str | None = None, ai_base_url: str | None = None) -> dict[str, Any] | None:
    """Cheap text-only triage, called before frame extraction/the full vision
    call. It only ever answers "clearly unrelated" or "not sure" -- never
    "relevant" -- so a false positive here can only cost an unnecessary full
    pipeline run, never a wrongly dropped real place video. Any failure
    (network, parsing, missing signal) returns None and the caller falls
    through to the full pipeline unchanged.
    """
    caption = str(metadata.get("caption") or "").strip()
    title = str(metadata.get("title") or "").strip()
    tags = metadata.get("tags") or []
    transcript_excerpt = (transcript or "").strip()[:_PREFILTER_MAX_TRANSCRIPT_CHARS]

    # Nothing to triage on -- e.g. a silent video whose only evidence lives in
    # on-screen text/signage. Only the full vision call can judge that.
    if not (caption or title or tags or transcript_excerpt):
        return None

    try:
        endpoint = providers.from_provider(ai_provider, ai_model, ai_base_url)
    except Exception:
        logger.warning("Topic pre-filter skipped: could not resolve AI credentials", exc_info=True)
        return None

    prompt = f"""
You are a fast triage filter in front of a travel/food place-extraction pipeline.
Decide only whether this social video is CLEARLY UNRELATED to any real-world travel destination, place, or food/restaurant topic.
Default to "not unrelated" whenever there is any doubt, any hint of a location/itinerary/food/venue, or when the text below is too sparse to judge confidently -- a later step that also looks at the video frames will make the careful call.
Only answer clearly_unrelated=true for unambiguous cases: pure dance/lipsync/comedy trends, product ads or unboxings unrelated to any venue, tutorials, personal vlogs with no place/food subject, gaming, or news/politics with no travel or food content.

Text (caption/title/tags/transcript excerpt, not necessarily reliable):
{json.dumps({"caption": caption, "title": title, "tags": tags, "transcript_excerpt": transcript_excerpt}, ensure_ascii=False)}

Return only this JSON object, no markdown, no commentary:
{{"clearly_unrelated": boolean, "reason": string}}
""".strip()

    body = {"messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"},
            **_limits(endpoint, _PREFILTER_MAX_TOKENS, _REASONING_PREFILTER_MAX_TOKENS, thorough=False)}
    result = await providers.chat(endpoint, body, operation="PREFILTER_TOPIC",
                                  timeout=min(30, config.AI_REQUEST_TIMEOUT_SECONDS), parse=providers.json_answer)
    if not result.ok:
        logger.warning("Topic pre-filter call failed; falling through to full pipeline: %s", result.error)
        return None
    parsed = result.value
    if not isinstance(parsed, dict) or str(parsed.get("clearly_unrelated")).strip().casefold() not in {"1", "true", "yes"}:
        return None

    reason = str(parsed.get("reason") or "Content has no travel, food, or place subject").strip()
    logger.info("Topic pre-filter rejected video before frame extraction: reason=%s", reason)
    return {"reason": reason, "provider": endpoint.name, "model": endpoint.model, "usage": result.usage}


async def extract(*, metadata: dict[str, Any], transcript: str, transcript_provider: str, images: list[str],
                  language: str, ai_provider: str | None = None, ai_model: str | None = None,
                  ai_base_url: str | None = None) -> dict[str, Any]:
    """{extraction, provider, model, usage}: the model's raw answer; the bot normalises it.

    `images` are data URLs, in video order. Raises ValueError/RuntimeError with the reason when the
    provider is unknown or unconfigured or the answer is unusable."""
    endpoint = providers.from_provider(ai_provider, ai_model, ai_base_url)
    logger.info("Extracting candidates with AI: provider=%s model=%s transcript_chars=%s images=%s language=%s",
                endpoint.name, endpoint.model, len(transcript), len(images), language)
    context = {
        "creator_metadata": metadata,
        "transcript_primary_evidence": transcript,
        "transcript_provider": transcript_provider,
        "image_count": len(images),
        "response_language": language,
    }

    system_prompt = f"""
You are a meticulous social-video place extraction engine for a travel application.
Analyze the complete input (every transcript sentence, every supplied frame, the caption and the tagged location) before answering, and return one valid JSON object only.
You are the primary identity judge: the decision about which real place a mention refers to is yours, not a downstream string match.
Reason silently in two passes: first inventory every possible place mention, then challenge each identity with all audio, visual, route and surrounding context.
Return every distinct named place or concrete visit point the video presents as a useful travel spot; a named mention whose identity stays uncertain is kept as a candidate with an explicit unresolved verification state rather than dropped.
Accept normal transcription errors, missing accents, abbreviations, aliases and paraphrases when the combined evidence still establishes the identity, and write the correct, commonly used name.
Only a directly evidenced proper name becomes a candidate: an unnamed activity, venue category, route instruction or nearby landmark stays what it is (a named transport leg or named area may remain an activity when it is not a resolvable place).
For every candidate, write what a traveller needs to know about that place from what this video says or shows, and nothing from outside knowledge.
Each candidate query is Google Maps-ready: the exact name plus the most specific supported branch, address, district, city, region and country context.
Write user-facing descriptions and guidance in {language}; proper names and quoted evidence keep their original language.
""".strip()

    prompt = f"""
Extract real-world place candidates from this TikTok/Instagram travel or food video.

## Topic gate
- is_relevant=true when the video is materially about travel, a destination, a real-world place, a food/restaurant/cafe review or practical visit guidance.
- is_relevant=false for unrelated entertainment, personal content, sales, politics, gaming, generic memes or content with no travel/food/place subject: explain briefly in relevance_reason, set found=false and candidates=[].

## Evidence and trust
- The audio transcript is high-trust evidence with possible recognition errors: judge semantic support, not literal equality.
- Frames establish identity through readable storefront signs, labels, maps and on-screen titles. A title the creator puts over the video to name the current stop (often numbered, e.g. "3. Cây cô đơn") is direct identity evidence: EXACT_VISIBLE_SIGN with that frame index. Platform UI, usernames, hashtags, watermarks, phone numbers and ads carry no identity.
- Subtitle lines burned into the frames are the creator's own words, usually spelled correctly: use them to fix speech-recognition errors.
- Speech recognition garbles Vietnamese proper names ("tà sồa" for "Tà Xùa", "mỏng cá heo" for "Mỏm cá heo", "amê hyu" for "Army Hill"): write the correct name when subtitles, on-screen text or the well-known landmarks of the destination make it clear. That is correcting a transcription, not inferring a place.
- A specific audio mention wins over generic scenery, food photos, interiors or weak visual guesses; ambiguous imagery adds no extra candidates.
- creator_metadata.caption is the creator's own description and creator_metadata.taggedLocation is the place the creator pinned to the post. A place the caption names explicitly (typically after "📍", "địa chỉ", "tại", "ở", or with its ticket price or opening hours) and the taggedLocation name are direct identity evidence: CREATOR_CAPTION, with a caption_context evidence entry quoting the caption text or the tagged name.
- When the video is only music and scenery, without speech or readable on-screen text, the caption and taggedLocation are the identity source: list the places they name, never return nothing because audio and frames are silent.
- The taggedLocation address and city fill address_hint, city_hint and query for the place the post is about. When the caption names a more specific spot inside the tagged area (a farm or cafe inside a town or plantation area), the caption spot is the candidate and the tag is its area context; the tagged area is a separate candidate only when it is presented as a distinct stop.
- Hashtags, a bare city or province name, uploader names and sales lines (combos, room offers, "inbox"/"inb", booking contacts) carry no place identity.
- Evidence for one place may be spread across nearby slides, frames and audio: use the full-video context to disambiguate it while quoting the evidence for that individual place.

## What is a candidate
- A directly evidenced proper name of a place, business or visit point. Generic references ("quán cafe này", "quán bánh căn", "đi ăn", "đi ngắm biển") and broad cities, neighbourhoods or beach categories without a spoken or visible proper name are not candidates: record them in discarded_mentions with a reason.
- A route, access or support venue near a named destination is a separate candidate only with its own directly evidenced proper name; a geographically plausible Google Maps result does not create one.
- A titled or numbered stop that is an experience rather than a business (e.g. "Milky way" stargazing) is a candidate, named as the video titles it, with place_type_hint "activity".
- When visual and audio identities conflict, or a branch stays ambiguous, keep the named mention with uncertainty_reason so downstream planning shows it as an ACTIVITY.
- Every candidate carries at least one identity-supporting evidence entry quoting the actual transcript phrase or visible text as it appears.
- CONFIRMED means the evidence establishes one specific identity and no similarly plausible alternative remains; confidence is calibrated judgment, not a substitute for evidence.
- EXPLICIT_AUDIO_NAME tolerates minor transcript/OCR spelling differences when the spoken phrase semantically names the place. EXACT_VISIBLE_SIGN requires a readable identity-bearing sign or title and its 1-based supplied frame index. MULTIMODAL_CONFIRMED requires mutually supporting audio and visual identity evidence. CREATOR_CAPTION requires a caption_context evidence entry quoting the caption or taggedLocation text that names the place.
- CONTEXTUAL_INFERENCE and GENERIC_REFERENCE belong in discarded_mentions, whatever the confidence. inference_used=true marks a name guessed from activity, geography, a nearby POI, search knowledge or outside knowledge; such an item is discarded.

## Completeness and order
- Inspect the entire transcript and every supplied frame/slide before deciding the candidate list, then walk the transcript sentence by sentence, the frames in order and the caption line by line: every named stop, numbered item, viewpoint, eatery, cafe, homestay or named experience ends up as a candidate or in discarded_mentions with a reason. If the video numbers its stops 1..N, all N are present.
- Never merge multiple explicitly named list items, branches, restaurants, attractions or stops into one candidate; listicle language, numbered slides, route stops and phrases such as "N places" mean several candidates.
- The order in which points appear in the video is the highest-priority signal; explicit day/time is additional metadata, and popularity or map score changes no order.
- "A or B", "choose one", "option 1/2" or alternatives presented at the same moment become one candidate per option with the same optionGroupId, increasing optionIndex and a relation such as "OPTION" or "SAME_TIME".
- There is no candidate-count quota: return all distinct named spots the video supports, including unresolved named spots, however many.
- contentType is ITINERARY for a route, multi-day plan or ordered visit sequence, PLACE_LIST for a list of spots without a meaningful schedule (then dayHint/timeHint are null and no schedule is invented). ITINERARY always has durationDays: when the video does not state it, decide a reasonable duration from its route, sequence and spot density, with durationBasis and durationConfidence; this decision is autonomous.

## Writing for the traveller (in {language})
- Write every user-facing field (trip_title, trip_description, useful_summary, general_guidance, candidate description, useful_info, visit_guidance) the way a seasoned local planner writes a real trip plan: concrete, practical and warm, speaking directly to the traveller. Share what to do, see, eat or order there, the best time to go, prices, how to get there, what to bring or avoid, and the first-hand experience and recommendations voiced in the clip.
- Every fact comes from the audio, visible text, frames or the caption; prices, opening hours, transport details, reservation needs and tips appear only when the input supports them. When the video gives little about a place, one short plain sentence is enough.
- Never refer to the source in those fields: the words "video", "clip", "TikTok", "creator", "theo video", "trong video", "bạn ấy", "người quay" stay out, as does any remark about following or preserving an order. The caption, title and hashtags are not copied into user-facing fields.
- trip_title: a short natural name for the trip (at most 60 characters) built from the destination, length and character of the plan, e.g. "Đà Lạt 3 ngày săn mây và cà phê đồi". Not the video's title.
- trip_description: 2-4 sentences with the overall picture and the most useful practical advice from the video (when to go, how to get around, budget, pace, what to prepare).
- Candidate description: 2-4 sentences about that stop for someone following the plan: what makes it worth the stop, what to do or order, and any tip, timing or price mentioned. Extra stand-alone tips go to useful_info/visit_guidance, one tip per entry; information not tied to one place goes to useful_summary/general_guidance.
- query: a Google Maps-ready search phrase (exact place or business name plus the most specific supported address, district, city/region and country); search_context holds concise disambiguating context from the video (branch, neighbourhood, landmark, route, cuisine, nearby place), never invented. A specific POI is never replaced by the city alone.
- Place names, addresses and quoted sign text keep their original language where appropriate.

Return only one valid JSON object matching the structure below, without markdown or commentary.

Return this schema:
{{
  "is_relevant": boolean,
  "relevance_reason": string,
  "found": boolean,
  "needs_confirmation": boolean,
  "summary": string,                 // in {language}, internal one-line summary of the video
  "trip_title": string,              // in {language}, <= 60 characters, never the video's title
  "trip_description": string,        // in {language}, 2-4 sentences of overview and practical advice
  "useful_summary": string,          // in {language}, helpful non-place-specific information from the video
  "general_guidance": [string],      // in {language}, practical tips/instructions not tied to a single place
  "discarded_mentions": [
    {{
      "mention": string,
      "identity_status": "LIKELY"|"AMBIGUOUS",
      "reason": string
    }}
  ],
  "contentType": "ITINERARY"|"PLACE_LIST",
  "destination": {{"name": string|null, "city": string|null, "region": string|null, "country": string|null, "address": string|null}},
  "durationDays": integer|null,       // required for ITINERARY; null for PLACE_LIST
  "durationBasis": string|null,       // explain the automatic duration decision
  "durationConfidence": number|null,
  "candidates": [
    {{
      "candidateRef": string,         // unique within this video, e.g. social-0001
      "sequence": integer,             // 1-based order in the video
      "dayHint": integer|null,
      "timeHint": string|null,
      "optionGroupId": string|null,
      "optionIndex": integer|null,
      "relation": string|null,
      "name": string,
      "query": string,
      "description": string,          // in {language}, 2-4 sentences of practical guidance for this stop
      "useful_info": [string],        // in {language}, useful facts/details specifically about this place
      "visit_guidance": [string],     // in {language}, practical guidance specifically for this place
      "city_hint": string|null,
      "region_hint": string|null,
      "country_hint": string|null,
      "address_hint": string|null,
      "search_context": string|null,   // concise video context used to disambiguate Google Maps search
      "aliases": [string],
      "place_type_hint": string|null,
      "identity_status": "CONFIRMED"|"LIKELY"|"AMBIGUOUS",
      "mention_type": "EXPLICIT_AUDIO_NAME"|"EXACT_VISIBLE_SIGN"|"MULTIMODAL_CONFIRMED"|"CREATOR_CAPTION"|"CONTEXTUAL_INFERENCE"|"GENERIC_REFERENCE",
      "inference_used": boolean,
      "uncertainty_reason": string|null,
      "evidence": [
        {{
          "source": "audio"|"frame"|"slide"|"caption_context",
          "quote": string,
          "frame_index": integer|null,
          "supports_identity": boolean,
          "explanation": string|null
        }}
      ],
      "confidence": number
    }}
  ]
}}

Context:
{json.dumps(context, ensure_ascii=False)}

Return all named candidates in source order. found is true when at least one distinct named point exists. If the media contains no useful point, return an empty candidates list and let the downstream mode decide whether a place list can be shown.
"""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame_index, image in enumerate(images, start=1):
        user_content.append({"type": "text", "text": f"SUPPLIED_FRAME_INDEX: {frame_index}"})
        user_content.append({"type": "image_url", "image_url": {"url": image, "detail": "high"}})
    body = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        "response_format": {"type": "json_object"},
        **_limits(endpoint, _EXTRACTION_MAX_TOKENS, _REASONING_EXTRACTION_MAX_TOKENS, thorough=True),
    }
    result = await providers.chat(endpoint, body, operation="EXTRACT_CANDIDATES",
                                  timeout=config.AI_REQUEST_TIMEOUT_SECONDS, parse=providers.json_answer)
    if not result.ok:
        raise RuntimeError(_unusable(endpoint, result))
    if not isinstance(result.value, dict):
        raise ValueError(f"AI extraction returned unexpected payload type: {type(result.value).__name__}")
    logger.info("AI extraction complete: provider=%s model=%s found=%s candidates=%s", endpoint.name,
                endpoint.model, result.value.get("found"), len(result.value.get("candidates") or []))
    return {"extraction": result.value, "provider": endpoint.name, "model": endpoint.model,
            "usage": result.usage if (result.data or {}).get("usage") else None}
