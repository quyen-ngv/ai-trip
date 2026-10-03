import httpx
import pytest

from app.llm import CachedModel


@pytest.mark.asyncio
async def test_call_replaces_unsupported_max_tokens(monkeypatch):
    responses = [
        httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unsupported parameter: 'max_tokens' is not supported with this model."
                }
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
                ]
            },
        ),
    ]
    requests = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            requests.append(json.copy())
            return responses.pop(0)

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeAsyncClient)

    model = CachedModel.__new__(CachedModel)
    model.url = "https://example.test"
    model.key = "test-key"
    model.prefix, model.model = "TEST", "gpt-5.6-terra"

    result = await model._call({
        "model": "gpt-5.6-terra",
        "max_tokens": 123,
        "messages": [],
    })

    assert result == ({"ok": True}, False)
    assert requests[0]["max_tokens"] == 123
    assert "max_completion_tokens" not in requests[0]
    assert "max_tokens" not in requests[1]
    assert requests[1]["max_completion_tokens"] == 123


@pytest.mark.asyncio
async def test_truncated_output_is_not_retried(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            calls.append(json)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"days": ['},
                                                          "finish_reason": "length"}]})

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("LLM_CACHE_PATH", ":memory:")
    for k, v in {"BASE_URL": "https://example.test", "API_KEY": "k", "MODEL": "m"}.items():
        monkeypatch.setenv(f"TRUNC_{k}", v)
    assert await CachedModel("TRUNC").json("s", {"x": 1}, cache=False, retries=1) is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_each_call_is_reported_to_the_bound_sink(monkeypatch):
    from app import ai_log

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "prompt_cache_hit_tokens": 1000}})

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("LLM_CACHE_PATH", ":memory:")
    for k, v in {"BASE_URL": "https://api.deepseek.com", "API_KEY": "k", "MODEL": "deepseek-chat"}.items():
        monkeypatch.setenv(f"SINK_{k}", v)
    entries = []

    async def sink(entry):
        entries.append(entry)

    binding = ai_log.bind(sink)
    try:
        assert await CachedModel("SINK").json("rules", {"x": 1}, cache=False, operation="PLAN_ITINERARY") == {"ok": True}
    finally:
        ai_log.unbind(binding)

    [entry] = entries
    assert entry["feature"] == "AI_TRIP" and entry["operation"] == "PLAN_ITINERARY"
    assert entry["provider"] == "deepseek" and entry["model"] == "deepseek-chat" and entry["status"] == "SUCCEEDED"
    assert entry["request"]["messages"][0]["content"] == "rules"          # the prompt itself is kept
    assert (entry["inputTokens"], entry["cachedInputTokens"], entry["outputTokens"]) == (1200, 1000, 80)


@pytest.mark.asyncio
async def test_a_failing_sink_never_breaks_the_call(monkeypatch):
    from app import ai_log

    async def broken(entry):
        raise RuntimeError("java down")

    binding = ai_log.bind(broken)
    try:
        await ai_log.report(operation="X", provider="p", model="m", ok=True, request={})
    finally:
        ai_log.unbind(binding)
