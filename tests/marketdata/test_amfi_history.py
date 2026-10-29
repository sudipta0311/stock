"""Offline checks for AMFI's historical NAV download format."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from wealth.marketdata.sources import amfi_history

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"


def test_small_history_fixture_keeps_subcategory_and_plan_details() -> None:
    rows = amfi_history.parse_nav_history(
        (FIXTURES / "amfi_nav_history_small.txt").read_text(encoding="utf-8")
    )

    assert len(rows) == 3
    assert rows[0].scheme_code == "710001"
    assert rows[0].sub_category == "Large Cap"
    assert (rows[0].plan_type, rows[0].option_type) == ("direct", "growth")
    assert (rows[1].plan_type, rows[1].option_type) == ("regular", "idcw")
    assert rows[2].sub_category == "Liquid"
    assert rows[0].as_of == date(2026, 9, 30)


def test_history_fetch_uses_bounded_official_download_and_filters_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    class Response:
        text = (FIXTURES / "amfi_nav_history_small.txt").read_text(encoding="utf-8")

    def fake_get(url: str, *, timeout: int) -> Response:
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(amfi_history, "get", fake_get)

    rows = amfi_history.fetch_nav_history(date(2026, 9, 1), date(2026, 9, 30))

    assert len(rows) == 3
    assert calls[0][0].startswith(amfi_history.AMFI_NAV_HISTORY_URL + "?")
    assert "frmdt=01-Sep-2026" in calls[0][0]
    assert "todt=30-Sep-2026" in calls[0][0]
    assert calls[0][1] == 120


def test_history_fetch_rejects_ranges_over_amfi_limit() -> None:
    with pytest.raises(ValueError, match="cannot exceed 90 days"):
        amfi_history.fetch_nav_history(date(2026, 1, 1), date(2026, 4, 1))
