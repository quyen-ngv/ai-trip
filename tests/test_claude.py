import pytest

from app import ai_log, anthropic_adapter, providers, social
from app.assistant.chat_model import ChatModel
from app.llm import CachedModel


class _Final:
    def __init__(self, body):
        self.body = body

    def to_dict(self):
        return self.body


class _Stream:
    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get_final_message(self):
        return _Final(self.body)


class FakeClaude:
    """Stands in for AsyncAnthropic: records each Messages call and answers from a script."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []
        self.messages = self

    def with_options(self, **kwargs):
        return self

    def stream(self, **params):
        self.calls.append(params)
        return _Stream(self.answers.pop(0))


def _answer(*content, stop="end_turn", usage=None):
    return {"model": "claude-haiku-5-5", "stop_reason": stop, "content": list(content),
            "usage": usage or {"input_tokens": 100, "output_tokens": 20}}


def _text(value):
    return {"type": "text", "text": value}


@pytest.fixture
def claude(monkeypatch):
    def install(*answers):
        fake = FakeClaude(*answers)
        monkeypatch.setattr(anthropic_adapter, "client", lambda api_key, base_url: fake)
        return fake
    return install


@pytest.mark.asyncio
async def test_a_claude_tier_needs_only_provider_and_key(monkeypatch, claude):
    monkeypatch.setenv("LLM_CACHE_PATH", ":memory:")
    monkeypatch.setenv("CHEAPC_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("CHEAPC_LLM_API_KEY", "sk-ant")
    fake = claude(_answer({"type": "thinking", "thinking": "", "signature": "s"}, _text('{"ok": true}'),
                          usage={"input_tokens": 50, "cache_read_input_tokens": 900, "output_tokens": 30}))
    entries = []

    async def sink(entry):
        entries.append(entry)

    model = CachedModel("CHEAPC_LLM")
    assert (model.url, model.model) == ("https://api.anthropic.com", "claude-haiku-5-5")
    binding = ai_log.bind(sink)
    try:
        assert await model.json("rules", {"x": 1}, cache=False, temperature=0.3, max_tokens=500,
                                operation="PLAN_ITINERARY") == {"ok": True}
    finally:
        ai_log.unbind(binding)

    [params] = fake.calls
    assert params["system"] == "rules" and params["messages"] == [{"role": "user", "content": '{"x":1}'}]
    assert "temperature" not in params and "response_format" not in params
    assert params["max_tokens"] == 4096 and params["cache_control"] == {"type": "ephemeral"}
    [entry] = entries
    assert (entry["provider"], entry["status"]) == ("anthropic", "SUCCEEDED")
    assert (entry["inputTokens"], entry["cachedInputTokens"], entry["outputTokens"]) == (950, 900, 30)


@pytest.mark.asyncio
async def test_pengoo_tool_round_sends_claude_blocks_back_verbatim(monkeypatch, claude):
    for key, value in {"PROVIDER": "claude", "API_KEY": "sk-ant", "MODEL": "claude-haiku-5-5"}.items():
        monkeypatch.setenv(f"ASSISTANT_LLM_{key}", value)
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    call = {"type": "tool_use", "id": "toolu_1", "name": "search_places", "input": {"query": "phở"}}
    fake = claude(_answer(thinking, call, stop="tool_use"), _answer(_text("Phở Ngọc Béo nhé.")))
    tools = [{"type": "function", "function": {"name": "search_places", "description": "Search",
                                                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
    model = ChatModel()
    history = [{"role": "system", "content": "You are Pengoo."}, {"role": "user", "content": "Ăn phở ở đâu?"}]

    first = await model.complete(history, tools)
    assert first["tool_calls"][0]["function"]["name"] == "search_places"
    assert first["tool_calls"][0]["id"] == "toolu_1"
    assistant = {"role": "assistant", "content": None, "tool_calls": first["tool_calls"],
                 "_anthropic_content": first["_anthropic_content"]}
    results = [{"role": "tool", "tool_call_id": "toolu_1", "content": '{"places": []}'}]
    second = await model.complete(history + [assistant] + results + [{"role": "system", "content": "Budget used."}],
                                  tools)

    assert second["content"] == "Phở Ngọc Béo nhé."
    sent = fake.calls[1]
    assert sent["system"] == "You are Pengoo."
    assert sent["tools"][0]["input_schema"]["properties"]["query"] == {"type": "string"}
    assert sent["messages"][1] == {"role": "assistant", "content": [thinking, call]}
    assert sent["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"places": []}'}]}
    assert sent["messages"][3] == {"role": "system", "content": "Budget used."}


@pytest.mark.asyncio
async def test_a_refusal_is_an_unusable_answer_not_text(monkeypatch, claude):
    monkeypatch.setenv("ASSISTANT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ASSISTANT_LLM_API_KEY", "sk-ant")
    claude({**_answer(), "stop_reason": "refusal", "stop_details": {"category": "cyber", "explanation": "no"}})
    endpoint = providers.from_tiers("ASSISTANT_LLM")
    result = await providers.chat(endpoint, {"messages": [{"role": "user", "content": "x"}]}, operation="T", timeout=5)
    assert not result.ok and not result.retryable and result.error["reason"] == "refusal"


@pytest.mark.asyncio
async def test_social_extraction_on_claude_sends_frames_as_image_blocks(monkeypatch, claude):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    fake = claude(_answer(_text('Here it is: {"is_relevant": true, "found": false, "candidates": []}')))
    frame = "data:image/jpeg;base64," + "A" * 80

    result = await social.extract(metadata={"title": "x"}, transcript="t", transcript_provider="p", images=[frame],
                                  language="vi", ai_provider="CLAUDE")

    assert (result["provider"], result["model"]) == ("ANTHROPIC", "claude-haiku-5-5")
    assert result["extraction"]["is_relevant"] is True
    [params] = fake.calls
    assert params["output_config"] == {"effort": "low"} and params["max_tokens"] == 32000
    assert "temperature" not in params
    content = params["messages"][0]["content"]
    assert content[1] == {"type": "text", "text": "SUPPLIED_FRAME_INDEX: 1"}
    assert content[2] == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "A" * 80}}


@pytest.mark.asyncio
async def test_claude_web_search_resumes_a_paused_turn_and_reports_its_sources(claude):
    search = {"type": "web_search_tool_result", "tool_use_id": "srv_1",
              "content": [{"type": "web_search_result", "url": "https://a.vn/x", "title": "A"},
                          {"type": "web_search_result", "url": "https://b.vn/y", "title": "B"}]}
    fake = claude(_answer({"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "q"}},
                          search, stop="pause_turn"),
                  _answer(_text('{"sources": [{"url": "https://a.vn/x"'), _text(', "title": "A"}], "places": []}')))
    endpoint = providers.Endpoint("https://api.anthropic.com", "sk-ant", "claude-haiku-5-5")

    result = await providers.responses(endpoint, {"instructions": "Research.", "input": "halal food Da Nang",
                                                  "tools": [{"type": "web_search"}], "max_tool_calls": 6},
                                       operation="WEB_SEARCH", timeout=30)

    assert result.ok
    assert fake.calls[0]["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}]
    assert fake.calls[1]["messages"][-1]["role"] == "assistant"          # the paused turn is resumed
    from app.research import _output_text, _parse_json, _response_sources
    assert set(_response_sources(result.value)) == {"https://a.vn/x", "https://b.vn/y"}
    assert _parse_json(_output_text(result.value)) == {"sources": [{"url": "https://a.vn/x", "title": "A"}], "places": []}
    assert ai_log.responses_usage(result.value)[3] == 1


def test_web_search_uses_dynamic_filtering_off_haiku():
    params = anthropic_adapter.web_search_params("claude-sonnet-5-5", {"input": "q"})
    assert params["tools"][0]["type"] == "web_search_20260209"
