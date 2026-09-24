"""The AI owns the plan; these lock in what the safety net must still guarantee."""
from datetime import date
from app.candidates import normalize
from app.itinerary import append_missing_social_items, materialize
from app.validators import finalize, strip_internal

C = (16.0544, 108.2022)


def row(i, group, **kw):
    base = {"id": f"00000000-0000-4000-8000-{i:012d}", "googlePlaceId": f"g{i}",
            "title": f"Quán {i}" if group == "FOOD_AND_DRINK" else f"Điểm {i}",
            "address": f"{i} Trần Phú", "latitude": C[0] + (i % 5) * 0.003, "longitude": C[1] + ((i // 5) % 5) * 0.003,
            "placeGroup": group, "category": "Restaurant" if group == "FOOD_AND_DRINK" else "Museum",
            "reviewRating": 4.3, "reviewCount": 800}
    base.update(kw)
    return normalize(base)


ROWS = [row(i, "FOOD_AND_DRINK") for i in range(8)] + [row(20 + i, "CULTURE_AND_HERITAGE") for i in range(6)]
IDS = {r["title"]: r["id"] for r in ROWS}


def run(stops, **kw):
    ai = {"tripDescription": "Hai câu ngắn gọn.", "days": [{"dayNumber": 1, "theme": "Trung tâm", "stops": stops}]}
    return materialize(ai, ROWS, [1], [date(2026, 9, 8)], "BALANCED", "vi", set(), **kw)


def test_ai_plan_is_respected_verbatim():
    items, desc, _ = run([
        {"placeId": IDS["Quán 0"], "start": "07:15", "end": "08:00", "description": "Ăn sáng bún bò nóng."},
        {"placeId": IDS["Điểm 20"], "start": "08:30", "end": "10:00", "description": "Tham quan bảo tàng."},
        {"placeId": IDS["Quán 1"], "start": "12:00", "end": "13:00", "description": "Ăn trưa hải sản."},
        {"placeId": IDS["Quán 2"], "start": "18:30", "end": "19:30", "description": "Ăn tối đặc sản."},
    ])
    visits = [i for i in items if i["type"] == "PLACE_VISIT"]
    assert [i["startTime"] for i in visits] == ["07:15:00", "08:30:00", "12:00:00", "18:30:00"]  # AI times kept
    assert [i["name"] for i in visits] == ["Quán 0", "Điểm 20", "Quán 1", "Quán 2"]              # AI order kept
    assert visits[0]["description"] == "Ăn sáng bún bò nóng."                                    # AI wording kept
    assert desc == "Hai câu ngắn gọn."


def test_missing_breakfast_is_added():
    items, _, notes = run([
        {"placeId": IDS["Quán 1"], "start": "12:00", "end": "13:00", "description": "Trưa."},
        {"placeId": IDS["Quán 2"], "start": "18:30", "end": "19:30", "description": "Tối."},
    ])
    slots = [i["_slot"] for i in items]
    assert "BREAKFAST" in slots and any("added missing breakfast" in n for n in notes)
    breakfast = next(i for i in items if i["_slot"] == "BREAKFAST")
    assert breakfast["startTime"] < "10:30:00"


def test_overlaps_pushed_and_unknown_or_duplicate_ids_dropped():
    items, _, notes = run([
        {"placeId": IDS["Quán 0"], "start": "07:30", "end": "08:30", "description": "Sáng."},
        {"placeId": IDS["Điểm 20"], "start": "08:00", "end": "09:30", "description": "Chồng giờ."},
        {"placeId": "does-not-exist", "start": "10:00", "end": "11:00", "description": "Ma."},
        {"placeId": IDS["Quán 0"], "start": "11:00", "end": "12:00", "description": "Trùng."},
        {"placeId": IDS["Quán 1"], "start": "12:00", "end": "13:00", "description": "Trưa."},
        {"placeId": IDS["Quán 2"], "start": "18:30", "end": "19:30", "description": "Tối."},
    ])
    day = [i for i in items if i["dayNumber"] == 1]
    for a, b in zip(day, day[1:]):
        assert b["startTime"] >= a["endTime"]                       # no overlap survives
    assert any("unknown id" in n for n in notes) and any("already used" in n for n in notes)
    names = [i["name"] for i in items if i["type"] == "PLACE_VISIT"]
    assert len(names) == len(set(names))


def test_rest_block_and_description_cap():
    long_desc = "x" * 400
    items, _, _ = run([
        {"placeId": IDS["Quán 0"], "start": "07:30", "end": "08:30", "description": long_desc},
        {"rest": "walk", "start": "09:00", "end": "10:00", "description": "Đi dạo bờ sông."},
        {"placeId": IDS["Quán 1"], "start": "12:00", "end": "13:00", "description": "Trưa."},
        {"placeId": IDS["Quán 2"], "start": "18:30", "end": "19:30", "description": "Tối."},
    ], max_chars=200)
    assert len(items[0]["description"]) == 200
    rest = next(i for i in items if i["name"].startswith("Đi dạo"))
    assert rest["type"] == "ACTIVITY" and "placeId" not in rest and "notes" not in rest


def test_bad_json_raises_so_caller_can_fall_back():
    import pytest
    with pytest.raises(ValueError):
        materialize({"nope": 1}, ROWS, [1], [date(2026, 9, 8)], "BALANCED", "vi", set())
    with pytest.raises(ValueError):
        materialize(None, ROWS, [1], [date(2026, 9, 8)], "BALANCED", "vi", set())


def test_social_options_with_one_resolved_place_stay_as_two_activities():
    shared_id = "00000000-0000-4000-8000-000000009999"
    options = [
        normalize({
            "id": shared_id, "googlePlaceId": "same", "candidateRef": "social-0001",
            "socialJobId": "social-job", "socialSequence": 1, "title": "Option A",
            "address": "A", "latitude": C[0], "longitude": C[1],
            "placeGroup": "FOOD_AND_DRINK", "category": "Restaurant",
            "optionGroupId": "lunch", "optionIndex": 1, "relation": "OPTION",
            "resolutionStatus": "RESOLVED",
        }, source="social"),
        normalize({
            "id": shared_id, "googlePlaceId": "same", "candidateRef": "social-0002",
            "socialJobId": "social-job", "socialSequence": 2, "title": "Option B",
            "address": "B", "latitude": C[0], "longitude": C[1],
            "placeGroup": "FOOD_AND_DRINK", "category": "Restaurant",
            "optionGroupId": "lunch", "optionIndex": 2, "relation": "OPTION",
            "resolutionStatus": "RESOLVED",
        }, source="social"),
    ]
    ai = {"days": [{"dayNumber": 1, "stops": [
        {"candidateId": "social-0001", "start": "12:00", "end": "13:00"},
        {"candidateId": "social-0002", "start": "12:00", "end": "13:00"},
    ]}]}

    items, _, _ = materialize(ai, options, [1], [date(2026, 9, 8)], "BALANCED", "en", set(), fill_missing_meals=False)

    assert [item["sourceSocialCandidateRef"] for item in items] == ["social-0001", "social-0002"]
    assert [item["startTime"] for item in items] == ["12:00:00", "12:00:00"]
    assert [item["optionIndex"] for item in items] == [1, 2]


def test_missing_social_rows_are_appended_and_unresolved_rows_are_activities():
    rows = [
        normalize({
            "id": None, "candidateRef": "social-0001", "socialJobId": "job", "socialSequence": 1,
            "title": "Named but unresolved", "address": "Video address", "placeGroup": "OTHER",
            "optionGroupId": "meal", "optionIndex": 1, "relation": "OPTION",
            "resolutionStatus": "UNRESOLVED",
        }, source="social"),
        normalize({
            "id": None, "candidateRef": "social-0002", "socialJobId": "job", "socialSequence": 2,
            "title": "Second option", "address": "Video address", "placeGroup": "OTHER",
            "optionGroupId": "meal", "optionIndex": 2, "relation": "OPTION",
            "resolutionStatus": "UNRESOLVED",
        }, source="social"),
    ]

    items = append_missing_social_items([], rows, [1], [date(2026, 9, 8)], "BALANCED", "en")

    assert [item["sourceSocialCandidateRef"] for item in items] == ["social-0001", "social-0002"]
    assert all(item["type"] == "ACTIVITY" for item in items)
    assert items[0]["startTime"] == items[1]["startTime"]
    assert items[0]["endTime"] == items[1]["endTime"]


def test_missing_social_row_is_inserted_between_existing_video_points():
    rows = [
        normalize({
            "id": None, "candidateRef": "social-0001", "socialJobId": "job", "socialSequence": 1,
            "title": "Point one", "placeGroup": "OTHER", "resolutionStatus": "UNRESOLVED",
        }, source="social"),
        normalize({
            "id": None, "candidateRef": "social-0002", "socialJobId": "job", "socialSequence": 2,
            "title": "Point two", "placeGroup": "OTHER", "resolutionStatus": "UNRESOLVED",
        }, source="social"),
        normalize({
            "id": None, "candidateRef": "social-0003", "socialJobId": "job", "socialSequence": 3,
            "title": "Point three", "placeGroup": "OTHER", "resolutionStatus": "UNRESOLVED",
        }, source="social"),
    ]
    existing = []
    for row_value, start in ((rows[0], "08:00"), (rows[2], "10:00")):
        existing.append({
            "type": "ACTIVITY", "name": row_value["title"], "dayNumber": 1,
            "startTime": f"{start}:00", "endTime": f"{int(start[:2]) + 1:02d}:00:00",
            "sourceSocialCandidateRef": row_value["candidateId"],
            "_socialSequence": row_value["socialSequence"],
        })

    result = append_missing_social_items(
        existing, rows, [1], [date(2026, 9, 8)], "BALANCED", "en"
    )

    assert [item["sourceSocialCandidateRef"] for item in result] == [
        "social-0001", "social-0002", "social-0003"
    ]


def test_finalize_retains_social_points_that_do_not_fit_the_selected_duration():
    items = [
        {
            "type": "ACTIVITY", "name": f"Point {index}", "dayNumber": 1,
            "startTime": "23:00:00", "endTime": "23:59:00",
            "sourceSocialCandidateRef": f"social-{index:04d}", "_socialSequence": index,
        }
        for index in range(1, 8)
    ]

    out = strip_internal(finalize(items, 1))

    assert len(out) == len(items)
    assert [item["name"] for item in out] == [item["name"] for item in items]
    assert out[0]["startTime"] == "23:00:00"
    assert "startTime" not in out[-1] and "endTime" not in out[-1]


def test_finalize_repairs_social_day_inversion_and_keeps_options_together():
    items = [
        {
            "type": "ACTIVITY", "name": "Point two", "dayNumber": 1,
            "startTime": "12:00:00", "endTime": "13:00:00",
            "sourceSocialCandidateRef": "social-2", "_socialSequence": 2,
            "optionGroupId": "lunch", "optionIndex": 2,
        },
        {
            "type": "ACTIVITY", "name": "Point one", "dayNumber": 2,
            "startTime": "12:00:00", "endTime": "13:00:00",
            "sourceSocialCandidateRef": "social-1", "_socialSequence": 1,
            "optionGroupId": "lunch", "optionIndex": 1,
        },
    ]

    out = strip_internal(finalize(items, 2))

    assert [item["name"] for item in out] == ["Point one", "Point two"]
    assert [item["dayNumber"] for item in out] == [2, 2]
    assert [item["startTime"] for item in out] == ["12:00:00", "12:00:00"]
