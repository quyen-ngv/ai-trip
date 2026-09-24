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

    result = await model._call({
        "model": "gpt-5.6-terra",
        "max_tokens": 123,
        "messages": [],
    })

    assert result == {"ok": True}
    assert requests[0]["max_tokens"] == 123
    assert "max_completion_tokens" not in requests[0]
    assert "max_tokens" not in requests[1]
    assert requests[1]["max_completion_tokens"] == 123
