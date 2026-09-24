import json
from datetime import date

import pytest

from app import research
from app.itinerary import materialize
from app.research import WebResearcher


class FakeQueryModel:
    configured = True

    def __init__(self):
        self.calls = []

    async def json(self, system, payload, **kwargs):
        self.calls.append((system, payload, kwargs))
        return {
            "queries": [
                {"group": "FOOD_AND_DRINK", "query": "top halal restaurants in Da Nang local guide"},
                {"group": "ATTRACTIONS", "query": "hidden gems in Da Nang for families with seniors"},
            ],
        }


def web_response():
    sources = [
        {"url": "https://travel.example/halal", "title": "Halal guide"},
        {"url": "https://tourism.example/food", "title": "Tourism board"},
        {"url": "https://magazine.example/danang", "title": "Local magazine"},
    ]
    return {
        "output_text": json.dumps({
            "sources": sources,
            "places": [
                {
                    "name": "Halal Kitchen Da Nang", "address": "Hai Chau, Da Nang",
                    "category": "Halal restaurant", "reason": "Named by two guides as a halal option.",
                    "sourceUrls": ["https://travel.example/halal", "https://tourism.example/food"],
                },
                {
                    "name": "Uncited Restaurant", "address": "", "category": "Restaurant",
                    "reason": "Must not get through.", "sourceUrls": ["https://invented.example/nope"],
                },
            ],
        }),
        "output": [{"type": "web_search_call", "action": {"sources": sources}}],
    }


@pytest.mark.asyncio
async def test_research_uses_ai_query_and_keeps_only_cited_web_candidates():
    model = FakeQueryModel()
    researcher = WebResearcher(
        locale="vi",
        request={
            "pace": "RELAXED", "dietaryRestrictions": ["halal"],
            "groupComposition": "Gia đình có trẻ em và người lớn tuổi",
            "mobilityConsiderations": ["Wheelchair-accessible"],
            "discoveryStyle": "HIDDEN_GEMS", "preferenceText": "muốn ăn Halal và đi ít",
        },
        query_model=model,
        base_url="https://api.example/v1",
        api_key="key",
        model="web-model",
        min_sources_per_query=3,
    )
    seen_queries = []

    async def fake_request(query, destination, group):
        seen_queries.append((query, destination["name"], group))
        return web_response()

    researcher._request = fake_request
    rows = await researcher.research_destination(
        {"name": "Đà Nẵng", "latitude": 16.05, "longitude": 108.2},
        {"FOOD_AND_DRINK": 3, "ATTRACTIONS": 3},
        [{"title": "Known place"}],
    )

    assert len(model.calls) == 1
    request_context = model.calls[0][1]["travellerRequest"]
    assert request_context["dietaryRestrictions"] == ["halal"]
    assert "người lớn tuổi" in request_context["groupComposition"]
    assert {group for _, _, group in seen_queries} == {"FOOD_AND_DRINK", "ATTRACTIONS"}
    assert [row["title"] for row in rows] == ["Halal Kitchen Da Nang"]
    assert rows[0]["source"] == "web" and rows[0]["id"] is None
    assert rows[0]["sourceUrls"] == ["https://travel.example/halal", "https://tourism.example/food"]
    assert rows[0]["candidateId"].startswith("web:")
    assert rows[0]["address"] == ""
    assert rows[0]["latitude"] is None and rows[0]["longitude"] is None

    items, _, _ = materialize(
        {"days": [{"dayNumber": 1, "stops": [{
            "candidateId": rows[0]["candidateId"], "start": "12:00", "end": "13:00",
            "description": "Ăn trưa phù hợp yêu cầu Halal.",
        }]}]},
        rows, [1], [date(2026, 9, 10)], "BALANCED", "vi", set(), fill_missing_meals=False,
    )
    assert items[0]["type"] == "ACTIVITY"
    assert "https://travel.example/halal" in items[0]["notes"]


@pytest.mark.asyncio
async def test_research_rejects_results_without_the_required_number_of_web_sources():
    model = FakeQueryModel()
    researcher = WebResearcher(
        query_model=model, base_url="https://api.example/v1", api_key="key", model="web-model",
        min_sources_per_query=3,
    )
    result = web_response()
    result["output"][0]["action"]["sources"] = result["output"][0]["action"]["sources"][:2]

    async def fake_request(*_):
        return result

    researcher._request = fake_request
    rows = await researcher.research_destination({"name": "Da Nang"}, {"FOOD_AND_DRINK": 3}, [])
    assert rows == []


def test_web_research_reuses_the_cheap_model_responses_configuration():
    class CheapModel(FakeQueryModel):
        url = "https://api.openai.com/v1"
        key = "cheap-key"
        model = "cheap-web-model"

    researcher = WebResearcher(query_model=FakeQueryModel(), base_url="", api_key="", model="")
    assert not researcher.enabled
    configured = WebResearcher(query_model=CheapModel())
    assert configured.enabled
    assert configured.endpoint == "https://api.openai.com/v1/responses"
    assert configured.api_key == "cheap-key" and configured.model == "cheap-web-model"


@pytest.mark.asyncio
async def test_browser_request_requires_openai_web_search_and_returns_its_sources(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return web_response()

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, url, *, headers, json):
            captured.update(url=url, headers=headers, body=json)
            return Response()

    monkeypatch.setattr(research.httpx, "AsyncClient", lambda **_: Client())
    researcher = WebResearcher(
        query_model=FakeQueryModel(), base_url="https://api.openai.com/v1", api_key="key", model="gpt-5",
    )
    response = await researcher._request("top halal restaurants in Da Nang", {"name": "Da Nang"}, "FOOD_AND_DRINK")

    assert response == web_response()
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer key"
    assert captured["body"]["tools"] == [{"type": "web_search", "search_context_size": "high"}]
    assert captured["body"]["tool_choice"] == "required"
    assert captured["body"]["include"] == ["web_search_call.action.sources"]
    assert captured["body"]["text"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_query_planner_refuses_to_silently_drop_a_requested_group():
    class IncompleteModel(FakeQueryModel):
        async def json(self, *args, **kwargs):
            return {"queries": [{"group": "FOOD_AND_DRINK", "query": "halal restaurants Da Nang"}]}

    researcher = WebResearcher(
        query_model=IncompleteModel(), base_url="https://api.example/v1", api_key="key", model="web-model",
    )
    with pytest.raises(RuntimeError, match="too few Web research queries"):
        await researcher.plan_queries({"name": "Da Nang"}, {"FOOD_AND_DRINK": 3, "ATTRACTIONS": 3})


@pytest.mark.asyncio
async def test_cheap_curation_keeps_only_ids_returned_from_cited_evidence():
    class CuratingModel(FakeQueryModel):
        async def json(self, system, payload, **kwargs):
            if "curate cited Web discoveries" in system:
                return {"candidateIds": ["web:two", "invented"]}
            return await super().json(system, payload, **kwargs)

    researcher = WebResearcher(query_model=CuratingModel(), base_url="https://api.example/v1", api_key="key", model="web-model")
    rows = [
        {"candidateId": "web:one", "title": "One", "placeGroup": "FOOD_AND_DRINK", "sourceUrls": ["https://one"]},
        {"candidateId": "web:two", "title": "Two", "placeGroup": "FOOD_AND_DRINK", "sourceUrls": ["https://two"]},
    ]
    curated = await researcher.curate_candidates({"name": "Da Nang"}, rows)
    assert [row["candidateId"] for row in curated] == ["web:two"]
