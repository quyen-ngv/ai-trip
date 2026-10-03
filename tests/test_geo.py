from app.geo import haversine_km, is_open, next_opening, open_windows, parse_hours_entry, travel_estimate

NNBSP, THIN = " ", " "


def test_parse_google_typographic_spaces_and_elided_meridiem():
    assert parse_hours_entry(f"9:30{NNBSP}AM{THIN}–{THIN}9{NNBSP}PM") == [(9 * 60 + 30, 21 * 60)]
    assert parse_hours_entry("6 - 9:30 PM") == [(18 * 60, 21 * 60 + 30)]
    assert parse_hours_entry("11 AM–2 AM") == [(11 * 60, 26 * 60)]  # overnight
    assert parse_hours_entry("Open 24 hours") == [(0, 1440)]
    assert parse_hours_entry("Closed") == []
    assert parse_hours_entry("07:00 to 17:00") == [(7 * 60, 17 * 60)]
    assert parse_hours_entry("garbage") is None


def test_open_windows_ignores_padding_and_reports_unknown():
    hours = {"Monday": ["10:30 AM - 1:30 PM", "6 - 9:30 PM", ""], "Tuesday": ["???"]}
    assert open_windows(hours, "Monday") == [(10 * 60 + 30, 13 * 60 + 30), (18 * 60, 21 * 60 + 30)]
    assert open_windows(hours, "Tuesday") is None          # unreadable != closed
    assert open_windows(hours, "Wednesday") is None
    assert is_open(hours, "Monday", 12 * 60, 13 * 60) is True
    assert is_open(hours, "Monday", 15 * 60, 16 * 60) is False
    assert is_open(None, "Monday", 15 * 60, 16 * 60) is None
    assert next_opening(hours, "Monday", 15 * 60) == 18 * 60


def test_travel_and_distance():
    hanoi = {"latitude": 21.0285, "longitude": 105.8542}
    hoan_kiem = {"latitude": 21.0287, "longitude": 105.8524}
    assert 0.1 < haversine_km(hanoi, hoan_kiem) < 0.3
    assert travel_estimate(0.5)[0] == "WALKING"
    assert travel_estimate(5.0) == ("TAXI", 20)


def test_hours_summary_merges_days_and_lists_each_weekday_once():
    from app.geo import hours_summary
    week = {d: ["7:00 AM – 10:00 PM"] for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")}
    ten_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
    assert hours_summary(week, ten_days) == "daily 07:00-22:00"
    mixed = {**week, "Sunday": ["Closed"], "Saturday": ["11 AM – 2 AM"], "Friday": ["by appointment"]}
    assert hours_summary(mixed, ["Thursday", "Friday", "Saturday", "Sunday"]) == \
        "Thu 07:00-22:00 | Fri by appointment | Sat 11:00-02:00 | Sun closed"


def test_icon_glyph_entry_is_ignored():
    from app.geo import hours_summary
    hours = {"Monday": ["6:30 AM–12 AM", ""]}          # real Google Maps shape (prod, 2026-10)
    assert open_windows(hours, "Monday") == [(6 * 60 + 30, 1440)]
    assert hours_summary(hours, ["Monday"]) == "daily 06:30-00:00"
