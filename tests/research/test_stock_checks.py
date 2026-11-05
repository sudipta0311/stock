from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

import wealth.research.stock_checks as stock_checks
from wealth.db.models import FundamentalPIT
from wealth.research.stock_checks import (
    _stock_check_payloads,
    dma200_from_sessions,
    revenue_yoy_percentages,
)

_SECURITY_ID = uuid4()


def _revenue(period_end: date, value: str, known_at: datetime) -> FundamentalPIT:
    return FundamentalPIT(
        security_id=_SECURITY_ID,
        metric="revenue",
        period_end=period_end,
        period_type="Q",
        source="SYNTHETIC",
        known_at=known_at,
        value=Decimal(value),
        unit="INR",
        filed_at=known_at,
        source_url="https://example.test/synthetic-filing",
    )


def test_revenue_yoy_uses_consecutive_quarters_and_stops_at_a_missing_quarter() -> None:
    rows = [
        _revenue(date(2024, 3, 31), "100", datetime(2024, 5, 1, tzinfo=UTC)),
        _revenue(date(2025, 3, 31), "120", datetime(2025, 5, 1, tzinfo=UTC)),
        _revenue(date(2024, 9, 30), "100", datetime(2024, 11, 1, tzinfo=UTC)),
        _revenue(date(2025, 9, 30), "80", datetime(2025, 11, 1, tzinfo=UTC)),
    ]

    result = revenue_yoy_percentages(rows, date(2026, 1, 1))

    assert result == (Decimal("-20.0"),)


def test_revenue_yoy_ignores_facts_not_known_at_the_check_date() -> None:
    rows = [
        _revenue(date(2024, 12, 31), "100", datetime(2025, 2, 1, tzinfo=UTC)),
        _revenue(date(2024, 12, 31), "1000", datetime(2026, 11, 10, tzinfo=UTC)),
        _revenue(date(2025, 12, 31), "200", datetime(2026, 2, 1, tzinfo=UTC)),
    ]

    result = revenue_yoy_percentages(rows, date(2026, 11, 7))

    assert result == (Decimal("100.0"),)


def test_dma200_requires_two_hundred_sessions() -> None:
    assert dma200_from_sessions([Decimal("10")] * 199) is None
    assert dma200_from_sessions([Decimal(value) for value in range(1, 201)]) == Decimal("100.5")


def _install_payload_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    security_id: UUID,
    held_ids: set[UUID],
    latest_price_date: date,
) -> None:
    session = SimpleNamespace(scalars=lambda _statement: [SimpleNamespace(id=security_id)])

    @contextmanager
    def fake_system_session(_engine: object) -> Iterator[Any]:
        yield session

    prices = [
        SimpleNamespace(
            close=Decimal(index),
            as_of=latest_price_date - timedelta(days=200 - index),
            source="SYNTHETIC",
        )
        for index in range(1, 201)
    ]
    monkeypatch.setattr(stock_checks, "system_session", fake_system_session)
    monkeypatch.setattr(stock_checks, "_held_security_ids", lambda _session: held_ids)
    monkeypatch.setattr(
        stock_checks,
        "_screened_rows",
        lambda _session, _date, _rules: {str(security_id): {"decile": 8, "coverage": "synthetic"}},
    )
    monkeypatch.setattr(
        stock_checks,
        "_price_sessions",
        lambda _session, _ids, _date: {security_id: prices},
    )
    monkeypatch.setattr(stock_checks, "_revenue_rows", lambda _session, _ids, _date: {})


def test_ranked_but_not_held_share_skips_sector_valuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = date(2026, 10, 30)
    security_id = uuid4()
    _install_payload_fakes(
        monkeypatch,
        security_id=security_id,
        held_ids=set(),
        latest_price_date=as_of,
    )
    valuation_calls: list[str] = []
    monkeypatch.setattr(
        stock_checks,
        "valuation_stats_for_security",
        lambda *_args: valuation_calls.append("called"),
    )

    rows, held_ids, screened_ids, valuation_checked = _stock_check_payloads(
        object(), "2026-W44", as_of, screener_rules=stock_checks.ScreenerRules.load()
    )

    assert held_ids == set()
    assert screened_ids == {security_id}
    assert valuation_checked == 0
    assert valuation_calls == []
    row = rows[0]
    assert row["quant_decile"] == 8
    assert row["factor_coverage"] == "synthetic"
    assert row["pe_percentile"] is None
    assert row["pe_vs_sector"] is None
    assert row["price"] == Decimal("200")
    assert row["dma200"] == Decimal("100.5")
    assert row["sources"]["valuation"] is None


def test_stale_price_fields_are_null_but_source_date_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = date(2026, 10, 30)
    latest_price_date = as_of - timedelta(days=21)
    security_id = uuid4()
    _install_payload_fakes(
        monkeypatch,
        security_id=security_id,
        held_ids=set(),
        latest_price_date=latest_price_date,
    )

    rows, _held_ids, _screened_ids, _valuation_checked = _stock_check_payloads(
        object(), "2026-W44", as_of, screener_rules=stock_checks.ScreenerRules.load()
    )

    row = rows[0]
    assert row["price"] is None
    assert row["dma200"] is None
    assert row["below_dma200"] is None
    assert row["sources"]["price"]["as_of"] == latest_price_date.isoformat()


def test_stock_check_run_reports_valuation_checked_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security_id = uuid4()
    session = SimpleNamespace(execute=lambda _statement: None)

    @contextmanager
    def fake_system_session(_engine: object) -> Iterator[Any]:
        yield session

    monkeypatch.setattr(stock_checks, "system_session", fake_system_session)
    monkeypatch.setattr(
        stock_checks,
        "_stock_check_payloads",
        lambda *_args, **_kwargs: (
            [
                {
                    "security_id": security_id,
                    "quant_decile": 8,
                    "factor_coverage": "synthetic",
                    "pe_percentile": None,
                    "pe_vs_sector": None,
                    "revenue_yoy_pct": [],
                    "price": None,
                    "dma200": None,
                }
            ],
            {security_id},
            {security_id},
            1,
        ),
    )

    result = stock_checks.run_stock_checks(object(), "2026-W44")

    assert result["valuation_checked"] == 1
