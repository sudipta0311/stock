"""Target-only reference-data loading tests for fund look-through."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from wealth.db.models import FundBasket, FundBasketWeight
from wealth.policy.actions import SatelliteIdea
from wealth.policy.data import _load_fund_lookthrough
from wealth.policy.lookthrough import same_company
from wealth.policy.personalise import (
    HoldingInput,
    MarketContext,
    PolicyProfile,
    UserData,
    _digest_inputs,
)
from wealth.schemas.profile import FinancialContext, InvestingPrefs


class _ScalarRows:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def all(self) -> list[Any]:
        return self.rows

    def __iter__(self):
        return iter(self.rows)


class _ReferenceSession:
    def __init__(self, baskets: list[FundBasket], weights: list[FundBasketWeight]) -> None:
        self.baskets = baskets
        self.weights = weights
        self.statements: list[Any] = []

    def scalars(self, statement: Any) -> _ScalarRows:
        self.statements.append(statement)
        entity = statement.column_descriptions[0]["entity"]
        return _ScalarRows(self.baskets if entity is FundBasket else self.weights)


def _holding(
    security_id: str,
    *,
    security_type: str,
    name: str,
    symbol: str | None,
    isin: str | None,
    price: Decimal | None,
    sub_category: str | None = None,
    market: str = "IN",
    currency: str = "INR",
    sleeve: str = "equity_india",
    category: str = "equity:india",
) -> HoldingInput:
    return HoldingInput(
        security_id=security_id,
        symbol=symbol,
        market=market,
        currency=currency,
        security_type=security_type,
        sleeve=sleeve,
        category=category,
        quantity=Decimal(100),
        price=price,
        name=name,
        isin=isin,
        sub_category=sub_category,
        member_id="member-one",
    )


def _basket(as_of: date, basket_id: str = "nifty100") -> FundBasket:
    return FundBasket(
        id=basket_id,
        label=basket_id,
        as_of=as_of,
        source="NSE_ARCHIVE+MSCI",
        source_url="https://example.test/nifty100.csv",
        weights_estimated=True,
        weight_source="msci_em+imi",
        members_total=100,
        members_found=94,
    )


def test_loader_queries_only_target_isins_and_symbols() -> None:
    company_isin = "IN0000000010"
    fund = _holding(
        "fund-id",
        security_type="mf",
        name="Synthetic active fund",
        symbol="FUND",
        isin=None,
        price=Decimal("10000"),
        sub_category="Large Cap",
    )
    direct = _holding(
        "direct-id",
        security_type="equity",
        name="Synthetic Alpha",
        symbol="ALPHA",
        isin=company_isin,
        price=Decimal("100"),
    )
    idea = SatelliteIdea(
        security_id="idea-id",
        symbol="BETA",
        market="IN",
        sector="Banks",
        price=Decimal("100"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("35"),
        confidence="HIGH",
        isin="IN0000000020",
    )
    weight_rows = [
        FundBasketWeight(
            basket_id="nifty100",
            market="IN",
            isin=company_isin,
            symbol="ALPHA",
            name="Synthetic Alpha",
            weight_pct=Decimal("12.5"),
        ),
        FundBasketWeight(
            basket_id="nifty100",
            market="IN",
            isin="IN0000000020",
            symbol="BETA",
            name="Synthetic Beta",
            weight_pct=Decimal("7"),
        ),
    ]
    session = _ReferenceSession([_basket(date(2026, 11, 3))], weight_rows)

    through, _members, _parts, as_of, notes, _keys = _load_fund_lookthrough(
        session,
        (fund, direct),
        type("Research", (), {"ideas": (idea,)})(),
        {"INR/INR": Decimal(1)},
        date(2026, 11, 4),
    )

    compiled = str(session.statements[2].compile(compile_kwargs={"literal_binds": True}))
    assert "fund_basket_weights.isin IN" in compiled
    assert "fund_basket_weights.symbol IN" in compiled
    assert through["direct-id"] == Decimal("125000.00")
    assert through["idea-id"] == Decimal("70000.00")
    assert as_of == {"nifty100": date(2026, 11, 3)}
    assert notes == []


def test_loader_skips_unvalued_fund_and_notes_stale_basket() -> None:
    fund = _holding(
        "fund-id",
        security_type="mf",
        name="Synthetic active fund",
        symbol="FUND",
        isin=None,
        price=None,
        sub_category="Large Cap",
    )
    session = _ReferenceSession([], [])

    through, _members, _parts, as_of, notes, _keys = _load_fund_lookthrough(
        session,
        (fund,),
        None,
        {},
        date(2026, 11, 4),
    )

    assert through == {}
    assert as_of == {}
    assert "lookthrough_fund_unvalued" in notes
    assert "fund_lookthrough_missing:nifty100" in notes
    assert len(session.statements) == 1


def test_loader_resolves_symbol_only_idea_from_fresh_reference_row() -> None:
    apple_isin = "US0378331005"
    as_of = date(2026, 11, 3)
    fund = _holding(
        "acwi-fund",
        security_type="etf",
        name="Synthetic ACWI fund",
        symbol="ACWI",
        isin=None,
        price=Decimal("1000"),
        market="US",
        currency="USD",
        sleeve="equity_global",
        category="etf",
    )
    direct = _holding(
        "direct-share",
        security_type="equity",
        name="Synthetic Apple share",
        symbol=None,
        isin=apple_isin,
        price=Decimal("100"),
        market="US",
        currency="USD",
        sleeve="equity_global",
        category="stock",
    )
    idea = SatelliteIdea(
        security_id="apple-idea",
        symbol="AAPL",
        market="US",
        sector="Technology",
        price=Decimal("100"),
        quant_decile=9,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("30"),
        confidence="HIGH",
        isin=None,
    )
    weights = [
        FundBasketWeight(
            basket_id="msci_acwi",
            market="US",
            isin=apple_isin,
            symbol=None,
            name="Synthetic global company",
            weight_pct=Decimal("8"),
        ),
        FundBasketWeight(
            basket_id="sp500",
            market="US",
            isin=apple_isin,
            symbol="AAPL",
            name="Synthetic symbol reference",
            weight_pct=Decimal("7"),
        ),
    ]
    session = _ReferenceSession(
        [_basket(as_of, "msci_acwi"), _basket(as_of, "sp500")],
        weights,
    )

    through, _members, _parts, _dates, notes, company_keys = _load_fund_lookthrough(
        session,
        (fund, direct),
        type("Research", (), {"ideas": (idea,)})(),
        {"USD/INR": Decimal(1)},
        date(2026, 11, 4),
    )

    assert through["apple-idea"] == Decimal("8000")
    assert same_company(
        "US",
        apple_isin,
        None,
        "US",
        None,
        "AAPL",
        company_keys,
    )
    assert notes == []


def test_plan_digest_changes_when_a_used_basket_date_changes() -> None:
    user_data = UserData(
        profile=PolicyProfile(),
        financial_context=FinancialContext(monthly_expenses=Decimal("10000")),
        goals=(),
        valued_assets=(),
        holdings=(),
        investing=InvestingPrefs(),
        lookthrough_as_of={"nifty100": date(2026, 11, 3)},
    )
    market = MarketContext(fx_rates={}, debt_yields={}, index_valuation_percentiles={})
    updated = UserData(
        **{
            **user_data.__dict__,
            "lookthrough_as_of": {"nifty100": date(2026, 11, 4)},
        }
    )

    old_digest = _digest_inputs(user_data, market, None, "2026-W45", date(2026, 11, 4))
    new_digest = _digest_inputs(updated, market, None, "2026-W45", date(2026, 11, 4))

    assert old_digest != new_digest
