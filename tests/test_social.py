import json

import httpx
import pytest

from app import providers, social

ANSWER = ('{"is_relevant":true,"relevance_reason":"travel review","found":true,"needs_confirmation":false,'
          '"summary":"summary","useful_summary":"useful","general_guidance":[],"candidates":[]}')
FRAME = "data:image/jpeg;base64," + "A" * 80
_REAL_CLIENT = httpx.AsyncClient


def _provider(monkeypatch, body=None, status=200):
    """Answer every model call with `body`; returns the list of (url, request json) sent."""
    sent = []
    body = body or {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
                    "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}]}

    def handler(request):
        sent.append((str(request.url), json.loads(request.content)))
        return httpx.Response(status, json=body) if status == 200 else httpx.Response(status, text="boom")

    monkeypatch.setattr(providers.httpx, "AsyncClient",
                        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(key, "test-secret")
    return sent


async def _extract(model="gpt-5-mini", provider="openai", images=(), transcript="Cafe Giang in Hanoi", **kw):
    return await social.extract(metadata={"title": "Travel"}, transcript=transcript, transcript_provider="test",
                                images=list(images), language="vi", ai_provider=provider, ai_model=model, **kw)


@pytest.mark.asyncio
async def test_deepseek_uses_openai_compatible_chat_completion(monkeypatch):
    sent = _provider(monkeypatch)
    result = await _extract("deepseek-v4-flash", "deepseek", [FRAME], ai_base_url="https://api.deepseek.com")

    assert (result["provider"], result["model"]) == ("DEEPSEEK", "deepseek-v4-flash")
    [(url, body)] = sent
    assert url == "https://api.deepseek.com/chat/completions"
    assert body["response_format"]["type"] == "json_object"
    assert "max_tokens" in body and "max_completion_tokens" not in body
    assert body["messages"][0]["role"] == "system"
    content = body["messages"][1]["content"]
    assert content[1]["text"] == "SUPPLIED_FRAME_INDEX: 1"
    assert content[2]["image_url"] == {"url": FRAME, "detail": "high"}


@pytest.mark.asyncio
async def test_openai_reasoning_model_gets_effort_room_and_no_temperature(monkeypatch):
    sent = _provider(monkeypatch)
    result = await _extract()

    body = sent[0][1]
    assert result["usage"] == {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150}
    assert result["extraction"]["is_relevant"] is True
    assert "max_tokens" not in body and "temperature" not in body
    assert body["reasoning_effort"] == "low" and body["max_completion_tokens"] > 8000
    assert "valid json object" in str(body["messages"]).lower()


@pytest.mark.asyncio
async def test_extraction_reasons_a_little_and_the_prefilter_not_at_all(monkeypatch):
    for model in ("gpt-6-luna", "gpt-5.4-mini"):
        sent = _provider(monkeypatch)
        await _extract(model)
        assert sent[0][1]["reasoning_effort"] == "low", model
        assert social._openai_reasoning_effort(model) == "none", model


@pytest.mark.asyncio
async def test_openai_non_reasoning_model_gets_no_reasoning_effort(monkeypatch):
    sent = _provider(monkeypatch)
    await _extract("gpt-4.1-mini")
    body = sent[0][1]
    assert "reasoning_effort" not in body and body["max_completion_tokens"] == 16000


@pytest.mark.asyncio
async def test_empty_ai_content_reports_why_instead_of_a_json_error(monkeypatch):
    _provider(monkeypatch, {"usage": {"prompt_tokens": 9000, "completion_tokens": 8000,
                                      "completion_tokens_details": {"reasoning_tokens": 8000}},
                            "choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    with pytest.raises(RuntimeError, match="finish_reason=length.*reasoning_tokens=8000"):
        await _extract()


@pytest.mark.asyncio
async def test_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported AI provider"):
        await _extract(provider="unknown")


@pytest.mark.asyncio
async def test_prompt_assigns_identity_judgement_to_ai_in_the_planner_voice(monkeypatch):
    sent = _provider(monkeypatch)
    await _extract(transcript="A video covering several destinations")
    messages = sent[0][1]["messages"]
    prompt = messages[1]["content"][0]["text"]
    assert "complete input" in messages[0]["content"] and "primary identity judge" in messages[0]["content"]
    for phrase in ("CONFIRMED means", "discarded_mentions", "quán cafe này", "There is no candidate-count quota",
                   "entire transcript and every supplied frame/slide", "Never merge multiple explicitly named list items",
                   '"trip_title"', '"trip_description"', "Never refer to the source", "Not the video's title"):
        assert phrase in prompt
    assert "Return at most" not in prompt


@pytest.mark.asyncio
async def test_topic_prefilter_sends_no_temperature_to_reasoning_model(monkeypatch):
    sent = _provider(monkeypatch, {"choices": [{"message": {"content": '{"clearly_unrelated": false, "reason": "cafe"}'}}]})
    await social.prefilter(metadata={"title": "Cafe hopping"}, transcript="This cafe in Hanoi is great",
                           ai_provider="openai", ai_model="gpt-5-mini")
    body = sent[0][1]
    assert "temperature" not in body and body["reasoning_effort"] == "low" and body["max_completion_tokens"] > 120


@pytest.mark.asyncio
async def test_topic_prefilter_skips_call_with_no_signal(monkeypatch):
    sent = _provider(monkeypatch)
    assert await social.prefilter(metadata={}, transcript="") is None
    assert sent == []


@pytest.mark.asyncio
async def test_topic_prefilter_returns_none_when_ai_finds_it_relevant(monkeypatch):
    _provider(monkeypatch, {"choices": [{"message": {"content": '{"clearly_unrelated": false, "reason": "a cafe"}'}}]})
    assert await social.prefilter(metadata={"title": "Cafe hopping"}, transcript="This cafe in Hanoi is great",
                                  ai_provider="deepseek") is None


@pytest.mark.asyncio
async def test_topic_prefilter_rejects_clearly_unrelated_content(monkeypatch):
    sent = _provider(monkeypatch, {"usage": {"prompt_tokens": 40, "completion_tokens": 8},
                                   "choices": [{"message": {"content": '{"clearly_unrelated": true, "reason": "pure dance trend"}'}}]})
    result = await social.prefilter(metadata={"title": "Dance challenge"}, transcript="", ai_provider="deepseek")

    assert result == {"reason": "pure dance trend", "provider": "DEEPSEEK", "model": "deepseek-v4-flash",
                      "usage": {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48}}
    body = sent[0][1]
    assert body["max_tokens"] == 120 and isinstance(body["messages"][0]["content"], str)


@pytest.mark.asyncio
async def test_topic_prefilter_fails_open_on_http_error(monkeypatch):
    _provider(monkeypatch, status=500)
    assert await social.prefilter(metadata={"title": "Dance challenge"}, transcript="", ai_provider="deepseek") is None


@pytest.mark.asyncio
async def test_social_calls_are_logged_with_the_job_and_no_inline_images(monkeypatch):
    _provider(monkeypatch)
    entries = []

    async def sink(entry):
        entries.append(entry)

    binding = social.ai_log.bind(sink, feature="SOCIAL_LOCATION", trace_id="job-1")
    try:
        await _extract(images=[FRAME])
    finally:
        social.ai_log.unbind(binding)

    [entry] = entries
    assert (entry["feature"], entry["traceId"], entry["operation"]) == ("SOCIAL_LOCATION", "job-1", "EXTRACT_CANDIDATES")
    assert "base64" not in json.dumps(entry["request"]).replace("<data-url", "")
