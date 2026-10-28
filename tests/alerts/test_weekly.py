"""Weekly delivery scheduling respects each user's local clock."""

from datetime import UTC, datetime

from wealth.alerts.weekly import _is_local_hour


def test_copenhagen_profiles_match_the_08_hour_during_winter() -> None:
    utc_time = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)

    assert _is_local_hour(utc_time, "Europe/Copenhagen", 8)
    assert not _is_local_hour(utc_time, "Europe/Copenhagen", 9)


def test_india_profiles_match_the_08_hour_without_daylight_saving() -> None:
    utc_time = datetime(2026, 1, 5, 2, 30, tzinfo=UTC)

    assert _is_local_hour(utc_time, "Asia/Kolkata", 8)
    assert not _is_local_hour(utc_time, "Asia/Kolkata", 7)


def test_unknown_profile_timezones_are_skipped() -> None:
    utc_time = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)

    assert not _is_local_hour(utc_time, "Unknown/Example", 8)
