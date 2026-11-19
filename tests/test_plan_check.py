"""The code checks on a chat plan (app/assistant/plan_check.py). Cases come from prod plans of
2026-11: a lunch carrying a square's id, a trip starting at its own destination, days without a
lunch or dinner, meals with no venue, a night without a stay."""
from app.assistant import plan_check
from app.assistant.graph import clean_plan

ROWS = {
    "sq": {"name": "Quảng trường Lâm Viên", "kind": "sight", "types": ["LANDMARK"]},
    "market": {"name": "Chợ đà lạt", "kind": "sight", "types": ["MARKET"]},
    "night": {"name": "Chợ Đêm Đà Lạt", "kind": "sight", "types": ["MARKET"]},
    "lake": {"name": "Spring Fragrant Lake Park", "kind": "sight", "types": ["PARK"]},
    "pho": {"name": "Phở Thi", "kind": "food", "types": ["RESTAURANT"]},
    "banhcan": {"name": "Bánh căn Lệ", "kind": "food", "types": ["STREET_FOOD"]},
    "cafe": {"name": "Blackbird Coffee", "kind": "food", "types": ["CAFE"]},
    "snack": {"name": "Bánh Tráng nướng Dì Đinh", "kind": "food", "types": ["DESSERT"]},
    "hotel": {"name": "M.C House Dalat", "kind": "stay", "types": []},
}


def stop(name, start, end, category="attraction", **extra):
    return {"name": name, "startTime": start, "endTime": end, "category": category, **extra}


def plan(*days, **fields):
    return {"destination": "Đà Lạt", "days": [{"dayNumber": n + 1, "items": list(items)} for n, items in enumerate(days)],
            **fields}


def kinds(gaps):
    return [(g["kind"], g["day"], g.get("meal")) for g in gaps]


class TestRelink:
    def test_a_stop_gets_the_id_of_the_row_it_names(self):
        p = plan([stop("Phở Thi", "07:30", "08:15", "restaurant", placeId="banhcan")])
        plan_check.relink_places(p, ROWS)
        assert p["days"][0]["items"][0]["placeId"] == "pho"

    def test_lunch_at_the_market_leaves_the_square(self):
        p = plan([stop("Chợ Đà Lạt, ăn trưa", "11:30", "12:30", "restaurant", placeId="sq")])
        plan_check.relink_places(p, ROWS)
        assert p["days"][0]["items"][0]["placeId"] == "market"      # not the night market

    def test_a_kind_conflict_with_no_better_row_drops_the_id(self):
        p = plan([stop("Mì Quảng Thuận", "07:45", "08:30", "restaurant", placeId="sq")])
        plan_check.relink_places(p, ROWS)
        assert "placeId" not in p["days"][0]["items"][0]
        assert p["days"][0]["items"][0]["name"] == "Mì Quảng Thuận"

    def test_a_translated_sight_name_keeps_its_place(self):
        p = plan([stop("Hồ Xuân Hương", "15:00", "16:00", "nature", placeId="lake")])
        assert plan_check.relink_places(p, ROWS) == []
        assert p["days"][0]["items"][0]["placeId"] == "lake"

    def test_a_description_on_a_cafe_row_keeps_its_place(self):
        p = plan([stop("Cà phê chill và chụp ảnh", "16:00", "17:00", "cafe", placeId="cafe")])
        assert plan_check.relink_places(p, ROWS) == []

    def test_clean_plan_relinks_and_drops_a_start_at_the_destination(self):
        raw = {"name": "Đà Lạt săn mây 3N2Đ", "destination": "Đà Lạt", "startingPointName": "Trung tâm Đà Lạt",
               "departureTime": "07:00",
               "days": [{"dayNumber": 1, "items": [{"placeId": "banhcan", "name": "Phở Thi", "startTime": "07:30",
                                                    "endTime": "08:15", "category": "restaurant"}]}]}
        cleaned = clean_plan(raw, {}, ROWS)
        assert "startingPointName" not in cleaned and "departureTime" not in cleaned
        assert cleaned["days"][0]["items"][0]["placeId"] == "pho"


class TestStartingPoint:
    def test_another_town_or_an_airport_stays(self):
        for start in ("Hà Nội", "Sân bay Liên Khương", "Bến xe Miền Đông, TP.HCM"):
            p = {"destination": "Đà Lạt", "startingPointName": start}
            assert plan_check.drop_start_at_destination(p) is False
            assert p["startingPointName"] == start

    def test_the_destination_itself_goes(self):
        for start in ("Đà Lạt", "Trung tâm Đà Lạt", "TP Đà Lạt"):
            p = {"destination": "Đà Lạt, Lâm Đồng", "startingPointName": start, "departureTime": "07:00"}
            assert plan_check.drop_start_at_destination(p) is True
            assert "startingPointName" not in p and "departureTime" not in p


class TestGaps:
    def test_the_prod_da_lat_plan(self):
        """Day 1 starts at 11:30 with no lunch venue, day 2 has an unnamed lunch and a night market
        for dinner, day 3 ends after lunch."""
        p = plan(
            [stop("Quảng trường Lâm Viên", "11:30", "12:30", placeId="sq"),
             stop("M.C House Dalat", "14:00", "14:30", "hotel", placeId="hotel", stay=True, untilDay=3),
             stop("Lẩu gà lá é Tao Ngộ", "18:30", "20:00", "restaurant", placeId="pho")],
            [stop("Bún bò Huế cô Thuý", "07:00", "07:45", "restaurant", placeId="banhcan"),
             stop("Đồi chè Cầu Đất", "08:45", "10:30", "nature"),
             stop("Ăn trưa gần đồi chè", "11:15", "12:15", "restaurant"),
             stop("Thác Datanla", "13:00", "15:00", "nature"),
             stop("Chợ Đêm Đà Lạt", "18:30", "20:30", "shopping", placeId="night")],
            [stop("Săn mây Đồi Đa Phú", "06:15", "07:15", "nature"),
             stop("Bún Bò Cô Bê", "07:45", "08:30", "restaurant"),
             stop("Bún riêu Chế Trang", "11:30", "12:30", "restaurant"),
             stop("Tháp Vinaphone", "13:00", "13:30")])
        found = kinds(plan_check.plan_gaps(p, ROWS, new_trip=True))
        assert ("NO_MEAL", 1, "LUNCH") in found
        assert ("NO_MEAL", 2, "DINNER") in found
        assert ("MEAL_UNLINKED", 2, "LUNCH") in found
        assert not [g for g in found if g[1] == 3]            # "Chế Trang" is not chè; no dinner owed
        assert plan_check.worth_repair(plan_check.plan_gaps(p, ROWS, new_trip=True))

    def test_a_snack_or_a_cafe_is_not_dinner(self):
        p = plan([stop("Phở Thi", "12:00", "13:00", "restaurant", placeId="pho"),
                  stop("Bánh Tráng nướng Dì Đinh", "18:00", "19:00", "restaurant", placeId="snack"),
                  stop("Chợ Đêm Đà Lạt", "19:30", "20:30", "shopping", placeId="night")])
        assert ("SNACK_AS_MEAL", 1, "DINNER") in kinds(plan_check.plan_gaps(p, ROWS, new_trip=False))

    def test_a_day_owes_only_the_meals_it_spans(self):
        arrival = plan([stop("Xe Sài Gòn → Đà Lạt", "12:30", "18:30", "transport", transportMode="BUS"),
                        stop("Phở Thi", "19:00", "20:00", "restaurant", placeId="pho"),
                        stop("M.C House Dalat", "20:15", "21:00", "hotel", placeId="hotel", stay=True, untilDay=2)],
                       [stop("Bánh căn Lệ", "07:00", "07:45", "restaurant", placeId="banhcan"),
                        stop("Thác Datanla", "08:00", "10:30", "nature"),
                        stop("Trả phòng", "11:00", "12:00", "hotel"),
                        stop("Xe về Sài Gòn", "12:15", "18:30", "transport", transportMode="BUS")])
        assert plan_check.plan_gaps(arrival, ROWS, new_trip=True) == []

    def test_meals_without_a_venue_are_left_alone_where_the_app_knows_no_food(self):
        loop = plan([stop("Ăn sáng phở ở Đồng Văn", "07:00", "07:45", "restaurant"),
                     stop("Đèo Mã Pí Lèng", "08:30", "11:00", "nature"),
                     stop("Ăn trưa ở Mèo Vạc", "12:00", "13:00", "restaurant"),
                     stop("Ăn tối ở Đồng Văn", "18:30", "19:30", "restaurant")])
        assert plan_check.plan_gaps(loop, ROWS, new_trip=False) == []

    def test_a_meal_at_the_stay_or_inside_a_park_needs_no_venue(self):
        p = plan([stop("Phở Thi", "07:00", "07:45", "restaurant", placeId="pho"),
                  stop("Bà Nà Hills", "08:30", "15:00", "entertainment"),
                  stop("Ăn trưa trên Bà Nà", "12:00", "13:00", "restaurant"),
                  stop("Ăn tối tại homestay", "18:30", "19:30", "restaurant")])
        assert [g for g in kinds(plan_check.plan_gaps(p, ROWS, new_trip=False)) if g[0] == "MEAL_UNLINKED"] == []

    def test_a_night_without_a_stay(self):
        p = plan([stop("Phở Thi", "12:00", "13:00", "restaurant", placeId="pho")],
                 [stop("Bánh căn Lệ", "07:00", "07:45", "restaurant", placeId="banhcan")])
        assert ("NO_STAY", 1, None) in kinds(plan_check.plan_gaps(p, ROWS, new_trip=True))
        assert ("NO_STAY", 1, None) not in kinds(plan_check.plan_gaps(p, ROWS, new_trip=False))

    def test_legs_and_repeats_alone_do_not_cost_a_round(self):
        assert plan_check.worth_repair([{"kind": "REPEATED_PLACE", "day": 2}, {"kind": "NO_RETURN_LEG", "day": 3}]) is False


def test_repair_message_names_every_gap():
    gaps = [{"kind": "NO_MEAL", "day": 1, "meal": "LUNCH", "near": "Quảng trường Lâm Viên"},
            {"kind": "MEAL_UNLINKED", "day": 2, "meal": "LUNCH", "name": "Ăn trưa gần đồi chè", "near": "Đồi chè Cầu Đất"},
            {"kind": "NO_STAY", "day": 1}]
    text = plan_check.repair_message(gaps)
    assert "Day 1: no lunch (11:00-14:30)" in text and "Quảng trường Lâm Viên" in text
    assert "Ăn trưa gần đồi chè" in text and "Night of day 1" in text
    assert "SAME plan" in text
