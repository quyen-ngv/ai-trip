"""Repairs on Pengoo's final answer, from the prod call log of 2026-11-12..15."""
import json

from app.assistant import graph, prompts

PLAN = {"action": "PLAN_DAYS", "plan": {"days": [
    {"dayNumber": 1, "items": [{"name": "Phở Thìn", "startTime": "07:00"}, {"name": "Hồ Gươm", "startTime": "08:00"}]},
    {"dayNumber": 2, "items": [{"name": "Văn Miếu", "startTime": "08:00"}]}], "name": "Hà Nội 2N1Đ"}}


def answer_with(plan_json: str) -> str:
    return '{"answer": "Lịch 2 ngày Hà Nội đây.", "proposals": [' + plan_json + '], "suggestions": ["Thêm 1 ngày"]}'


class TestJsonRepair:
    def test_one_closer_too_many(self):
        broken = json.dumps(PLAN, ensure_ascii=False).replace(']}]}', ']}]}}', 1)
        out = graph.parse_final(answer_with(broken))
        assert out["proposals"][0]["action"] == "PLAN_DAYS"
        assert len(out["proposals"][0]["plan"]["days"]) == 2

    def test_one_closer_too_few(self):
        broken = json.dumps(PLAN, ensure_ascii=False).replace('}]}, {"dayNumber": 2', '}], {"dayNumber": 2', 1)
        out = graph.parse_final(answer_with(broken))
        assert [d["dayNumber"] for d in out["proposals"][0]["plan"]["days"]] == [1, 2]

    def test_the_plan_closed_too_early(self):
        """Prod 2026-11-12: "…}]}]}, "budget": …" ended the plan before its last keys."""
        text = ('{"answer": "Lịch Sa Pa.", "proposals": [{"action": "PLAN_DAYS", "plan": {"days": [{"dayNumber": 1, '
                '"items": [{"name": "Cát Cát"}]}]}, "budget": 3250000, "name": "Sapa 3N2Đ"}, "summary": "Lịch 3 ngày"}], '
                '"suggestions": []}')
        out = graph.parse_final(text)
        assert out["answer"] == "Lịch Sa Pa."
        assert out["proposals"][0]["plan"]["days"][0]["items"][0]["name"] == "Cát Cát"

    def test_brackets_inside_strings_are_left_alone(self):
        text = '{"answer": "Giá {khoảng} 50k ]", "suggestions": ["a}"]'
        assert graph.parse_final(text) == {"answer": "Giá {khoảng} 50k ]", "suggestions": ["a}"]}

    def test_valid_json_is_untouched(self):
        text = answer_with(json.dumps(PLAN, ensure_ascii=False))
        assert graph.parse_final(text) == json.loads(text)

    def test_prose_stays_the_answer(self):
        assert graph.parse_final("Tổng chuyến khoảng 6 triệu.") == {"answer": "Tổng chuyến khoảng 6 triệu."}


class TestDestinationCards:
    CANDIDATES = {"d1": "Đà Lạt", "d2": "Phú Quốc", "d3": "Hà Giang", "d4": "Hà Nội", "d5": "Mộc Châu"}

    def test_bold_picks_become_cards(self):
        final = {"answer": "Tháng 11 hợp nhất: - **Đà Lạt**: dã quỳ. - **Phú Quốc**: mùa khô. - **Hà Giang**: tam giác mạch."}
        out = graph.attach_mentioned_destinations(final, self.CANDIDATES)
        assert [d["id"] for d in out["destinations"]] == ["d1", "d2", "d3"]

    def test_a_place_named_in_passing_is_not_a_pick(self):
        final = {"answer": "Tháng 12 chọn **Mộc Châu**, cách Hà Nội khoảng 200 km, hoa cải nở."}
        out = graph.attach_mentioned_destinations(final, self.CANDIDATES)
        assert [d["id"] for d in out["destinations"]] == ["d5"]

    def test_cards_the_model_gave_are_kept(self):
        final = {"answer": "**Đà Lạt**", "destinations": [{"id": "d2"}]}
        assert graph.attach_mentioned_destinations(final, self.CANDIDATES)["destinations"] == [{"id": "d2"}]


def test_a_bare_mention_asks_to_answer_the_message_above():
    turn = {"question": "", "user": {"name": "Fun"}, "conversationType": "TRIP", "locale": "vi"}
    original = prompts.context_message
    prompts.context_message = lambda turn, web=False: "ctx"     # the clock and trip are not under test
    try:
        text = prompts.question_message(turn)
    finally:
        prompts.context_message = original
    assert text.endswith("]") and "Fun only called you" in text and "latest message" in text
