"""Playwright database lifecycle and synthetic state helpers."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session
from tests.db.migrated_database import MigratedDatabase, create_migrated_database
from tests.fixtures.cas.make_pdf import write_encrypted_pdf

from wealth.api.fund_check import add_weekly_fund_action
from wealth.db.models import (
    Account,
    Action,
    FundBasket,
    FundBasketWeight,
    FxRate,
    GoldPrice,
    HoldingLot,
    HouseholdMember,
    NavDaily,
    PriceEod,
    ResearchPackRecord,
    Security,
    StockCheck,
    User,
)
from wealth.db.session import normalize_database_url
from wealth.policy.data import load_policy_inputs
from wealth.policy.personalise import personalise

ROOT = Path(__file__).resolve().parents[1]


def seed_market_data(database: MigratedDatabase) -> None:
    """Add local synthetic price, NAV, FX, and gold rows to the isolated schema."""
    today = date.today()
    equity = Security(
        market="IN",
        symbol="E2E-EQUITY",
        name="Synthetic e2e equity",
        security_type="equity",
        currency="INR",
        sleeve="equity_india",
        category="equity:india",
    )
    fund = Security(
        market="IN",
        symbol="E2E-FUND",
        name="Synthetic e2e fund",
        security_type="mf",
        currency="INR",
        sleeve="debt",
        category="mf:debt",
        amfi_scheme_code="E2E0001",
        plan_type="direct",
        option="growth",
    )
    with Session(database.owner_engine) as session, session.begin():
        session.add_all((equity, fund))
        session.flush()
        session.add_all(
            (
                PriceEod(
                    security_id=equity.id,
                    as_of=today,
                    close=Decimal("101.25"),
                    source="e2e",
                    source_url="https://example.test/market/price",
                ),
                NavDaily(
                    security_id=fund.id,
                    as_of=today,
                    nav=Decimal("123.45670000"),
                    source="e2e",
                ),
                FxRate(
                    pair="DKK/INR",
                    as_of=today,
                    rate=Decimal("12.34560000"),
                    source="e2e",
                    source_url="https://example.test/market/fx",
                ),
                FxRate(
                    pair="USD/INR",
                    as_of=today,
                    rate=Decimal("83.00000000"),
                    source="e2e",
                    source_url="https://example.test/market/fx",
                ),
                GoldPrice(
                    as_of=today,
                    inr_per_gram_24k=Decimal("6500.000000"),
                    source="e2e",
                    source_url="https://example.test/market/gold",
                ),
            )
        )


def _assert_no_e2e_resources(database_url: str) -> None:
    engine = create_engine(normalize_database_url(database_url), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            schema_count = connection.execute(
                text("SELECT count(*) FROM pg_namespace WHERE left(nspname, 4) = 'e2e_'")
            ).scalar_one()
            role_count = connection.execute(
                text("SELECT count(*) FROM pg_roles WHERE left(rolname, 4) = 'e2e_'")
            ).scalar_one()
        if schema_count or role_count:
            raise RuntimeError("Temporary e2e database resources remain after teardown")
    finally:
        engine.dispose()


def _setup_stack() -> int:
    owner_url = os.getenv("TEST_DATABASE_URL")
    if not owner_url:
        raise RuntimeError("TEST_DATABASE_URL is required for the e2e stack")
    temporary_dir = Path(tempfile.mkdtemp(prefix="wealth-c24-e2e-"))
    try:
        cas_password = secrets.token_urlsafe(14)
        api_secret = secrets.token_urlsafe(48)
        cas_path = temporary_dir / "synthetic-cas.pdf"
        fixture_path = ROOT / "tests" / "fixtures" / "cas" / "nsdl.txt"
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
        write_encrypted_pdf(lines, cas_password, cas_path)

        with create_migrated_database(schema_prefix="e2e", login_prefix="e2e_login") as database:
            seed_market_data(database)
            setup = {
                "api_database_url": database.api_database_url,
                "api_secret": api_secret,
                "cas_path": str(cas_path),
                "cas_password": cas_password,
                "schema_name": database.schema_name,
                "temporary_dir": str(temporary_dir),
            }
            output_path = os.getenv("E2E_SETUP_OUTPUT_FILE")
            release_path = os.getenv("E2E_SETUP_RELEASE_FILE")
            if bool(output_path) != bool(release_path):
                raise RuntimeError("Both e2e setup handoff paths are required")
            if output_path and release_path:
                handoff = Path(output_path)
                pending = handoff.with_suffix(handoff.suffix + ".pending")
                pending.write_text(
                    json.dumps(setup, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                pending.replace(handoff)
                deadline = time.monotonic() + 20 * 60
                release = Path(release_path)
                while not release.exists() and time.monotonic() < deadline:
                    time.sleep(0.1)
                if not release.exists():
                    raise RuntimeError("The local e2e stack did not release its database")
            else:
                sys.stdout.write(json.dumps(setup, separators=(",", ":")) + "\n")
                sys.stdout.flush()
                sys.stdin.buffer.read()
        _assert_no_e2e_resources(owner_url)
        return 0
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _owner_engine() -> tuple[Engine, str]:
    owner_url = os.getenv("TEST_DATABASE_URL")
    schema_name = os.getenv("E2E_SCHEMA_NAME")
    if not owner_url or not schema_name or not schema_name.startswith("e2e_"):
        raise RuntimeError("The isolated e2e database is not configured")
    scoped_url = make_url(normalize_database_url(owner_url)).update_query_dict(
        {"options": f"-csearch_path={schema_name}"}
    )
    return create_engine(scoped_url, pool_pre_ping=True), schema_name


def _seed_action(email: str) -> int:
    engine, _schema_name = _owner_engine()
    try:
        with Session(engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic e2e user was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic e2e household has no owner member")
            iso = date.today().isocalendar()
            session.add(
                Action(
                    user_id=user.id,
                    week=f"{iso.year}-W{iso.week:02}",
                    rank=1,
                    kind="BUILD_EMERGENCY_FUND",
                    payload={
                        "kind": "BUILD_EMERGENCY_FUND",
                        "amount": "5000.00",
                        "member_id": str(member.id),
                    },
                )
            )
        return 0
    finally:
        engine.dispose()


def _seed_broker_import_account(email: str) -> int:
    engine, _schema_name = _owner_engine()
    try:
        with Session(engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic e2e user was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic e2e household has no owner member")
            member.residency = {
                **dict(member.residency or {}),
                "status": "NRI",
                "country": "DK",
            }
            session.add(
                Account(
                    user_id=user.id,
                    member_id=member.id,
                    institution="Synthetic Nordnet account",
                    account_type="investment",
                    tax_wrapper=None,
                )
            )
        return 0
    finally:
        engine.dispose()


def _seed_icici_portfolio_securities(email: str) -> int:
    engine, _schema_name = _owner_engine()
    securities = (
        ("SYNTH-ICICI-PAINT", "Example Paints Limited", "equity"),
        ("SYNTH-ICICI-SONS", "Sample and Sons Limited", "equity"),
        ("SYNTH-ICICI-NANDI", "Nandi Wall's (India) Limited", "equity"),
        ("SYNTH-ICICI-NIFTY", "Alpha Nifty 50 ETF", "etf"),
        ("SYNTH-ICICI-PICK", "Listed Synthetic Index Fund", "etf"),
    )
    try:
        with Session(engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic e2e user was not found")
            for symbol, name, security_type in securities:
                security = session.scalar(
                    select(Security).where(Security.market == "IN", Security.symbol == symbol)
                )
                if security is None:
                    security = Security(
                        market="IN",
                        symbol=symbol,
                        name=name,
                        security_type=security_type,
                        currency="INR",
                        sleeve="equity_india",
                        category="equity:india",
                    )
                    session.add(security)
                else:
                    security.name = name
                    security.security_type = security_type
        return 0
    finally:
        engine.dispose()


def _seed_money_list_holdings(email: str) -> int:
    engine, _schema_name = _owner_engine()
    try:
        with Session(engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic Money list user was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic Money list household has no owner member")

            groups = (
                ("fund", "mf", "mf:equity"),
                ("share", "equity", "equity:india"),
            )
            for label, security_type, category in groups:
                for index in range(1, 10):
                    symbol = f"C63-E2E-{label.upper()}-{index}"
                    security = session.scalar(
                        select(Security).where(
                            Security.market == "IN", Security.symbol == symbol
                        )
                    )
                    if security is None:
                        security = Security(
                            market="IN",
                            symbol=symbol,
                            name=f"Synthetic {label} {index}",
                            security_type=security_type,
                            currency="INR",
                            sleeve="equity_india",
                            category=category,
                        )
                        session.add(security)
                        session.flush()

                    lot = session.scalar(
                        select(HoldingLot).where(
                            HoldingLot.user_id == user.id,
                            HoldingLot.security_id == security.id,
                        )
                    )
                    if lot is None:
                        session.add(
                            HoldingLot(
                                user_id=user.id,
                                member_id=member.id,
                                security_id=security.id,
                                quantity=Decimal("1"),
                                cost_per_unit=Decimal(index * 10),
                                currency="INR",
                                source="manual",
                                statement_value=Decimal(1000 + index),
                                statement_date=date.today(),
                            )
                        )
        return 0
    finally:
        engine.dispose()


def _seed_holding_reviews(email: str) -> int:
    engine, _schema_name = _owner_engine()
    today = date.today()
    iso = today.isocalendar()
    week = f"{iso.year}-W{iso.week:02}"
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
                {"lock_key": "c49-e2e-holding-reviews"},
            )
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic holding-review user was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic holding-review household has no owner member")
            member.investing = {
                **dict(member.investing or {}),
                "new_money_this_week": "2500000",
                "satellite_pct": "20",
            }

            scenarios = (
                ("C49-E2E-ADD", "Synthetic share to add", "100", "9", "60", "1.0", [], "99"),
                (
                    "C49-E2E-SELL",
                    "Synthetic share to sell",
                    "50",
                    "4",
                    "45",
                    "1.0",
                    ["-5", "-4"],
                    "60",
                ),
            )
            for symbol, name, raw_price, decile, percentile, sector_ratio, revenue, dma in scenarios:
                price = Decimal(raw_price)
                security = session.scalar(
                    select(Security).where(Security.market == "IN", Security.symbol == symbol)
                )
                if security is None:
                    security = Security(
                        market="IN",
                        symbol=symbol,
                        name=name,
                        security_type="equity",
                        currency="INR",
                        sleeve="equity_india",
                        category="equity:india",
                        sector="Synthetic sector",
                        industry="Synthetic industry",
                    )
                    session.add(security)
                    session.flush()
                price_row = session.scalar(
                    select(PriceEod).where(
                        PriceEod.security_id == security.id,
                        PriceEod.as_of == today,
                        PriceEod.source == "synthetic_c49",
                    )
                )
                if price_row is None:
                    session.add(
                        PriceEod(
                            security_id=security.id,
                            as_of=today,
                            close=price,
                            source="synthetic_c49",
                            source_url="https://example.test/synthetic-c49-price",
                        )
                    )
                else:
                    price_row.close = price
                check = session.get(StockCheck, (security.id, week))
                values = {
                    "as_of": today,
                    "quant_decile": int(decile),
                    "factor_coverage": "medium",
                    "pe_percentile": Decimal(percentile),
                    "pe_vs_sector": Decimal(sector_ratio),
                    "revenue_yoy_pct": revenue,
                    "price": price,
                    "dma200": Decimal(dma),
                    "below_dma200": price < Decimal(dma),
                    "sources": {"fixture": "synthetic"},
                }
                if check is None:
                    session.add(StockCheck(security_id=security.id, week=week, **values))
                else:
                    for key, value in values.items():
                        setattr(check, key, value)
                quantity = Decimal("10") if symbol.endswith("ADD") else Decimal("2")
                existing_lot = session.scalar(
                    select(HoldingLot).where(
                        HoldingLot.user_id == user.id,
                        HoldingLot.security_id == security.id,
                    )
                )
                if existing_lot is None:
                    session.add(
                        HoldingLot(
                            user_id=user.id,
                            member_id=member.id,
                            security_id=security.id,
                            quantity=quantity,
                            cost_per_unit=None if symbol.endswith("ADD") else Decimal("40"),
                            acquired_on=None
                            if symbol.endswith("ADD")
                            else today - timedelta(days=400),
                            source="manual",
                            currency="INR",
                        )
                    )

            anchor_symbol = "C49-E2E-ANCHOR"
            anchor = session.scalar(
                select(Security).where(
                    Security.market == "IN",
                    Security.symbol == anchor_symbol,
                )
            )
            if anchor is None:
                anchor = Security(
                    market="IN",
                    symbol=anchor_symbol,
                    name="Synthetic broad fund",
                    security_type="mf",
                    currency="INR",
                    sleeve="equity_global",
                    category="equity:global",
                )
                session.add(anchor)
                session.flush()
            anchor_lot = session.scalar(
                select(HoldingLot).where(
                    HoldingLot.user_id == user.id,
                    HoldingLot.security_id == anchor.id,
                )
            )
            if anchor_lot is None:
                session.add(
                    HoldingLot(
                        user_id=user.id,
                        member_id=member.id,
                        security_id=anchor.id,
                        quantity=Decimal("1"),
                        statement_value=Decimal("1000000"),
                        statement_date=today,
                        source="manual",
                        currency="INR",
                    )
                )
        return 0
    finally:
        engine.dispose()


def _seed_denmark_buy_action(email: str) -> int:
    engine, _schema_name = _owner_engine()
    try:
        with Session(engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic e2e user was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic e2e household has no owner member")
            member.residency = {
                **dict(member.residency or {}),
                "status": "NRI",
                "country": "DK",
            }
            member.investing = {"new_money_this_week": "12345.60"}
            account = Account(
                user_id=user.id,
                member_id=member.id,
                institution="Synthetic broker",
                account_type="investment",
                tax_wrapper="ask",
            )
            portfolio_account = Account(
                user_id=user.id,
                member_id=member.id,
                institution="Synthetic portfolio account",
                account_type="investment",
                tax_wrapper="ordinary",
            )
            session.add_all((account, portfolio_account))
            iso = date.today().isocalendar()
            week = f"{iso.year}-W{iso.week:02}"
            session.flush()
            security = session.scalar(
                select(Security).where(Security.market == "IN", Security.symbol == "E2E-EQUITY")
            )
            if security is None:
                raise RuntimeError("The Denmark e2e scenario has no synthetic portfolio holding")
            session.add(
                HoldingLot(
                    user_id=user.id,
                    account_id=portfolio_account.id,
                    member_id=member.id,
                    security_id=security.id,
                    quantity=Decimal("10000"),
                    cost_per_unit=Decimal("100"),
                    acquired_on=date.today(),
                    source="manual",
                    currency="INR",
                )
            )
            session.flush()
            user_data, market, research = load_policy_inputs(
                session,
                user.id,
                week,
                date.today(),
                deployment_mode="personal",
            )
            computed = personalise(user_data, market, research, week, date.today())
            computed = computed.model_copy(
                update={
                    "actions": add_weekly_fund_action(
                        session,
                        user.id,
                        date.today(),
                        computed.actions,
                        can_write=True,
                    )
                }
            )
            buy_action = next(
                (action for action in computed.actions if action.get("kind") == "BUY_ABROAD"),
                None,
            )
            if buy_action is None:
                raise RuntimeError("The Denmark e2e scenario did not produce a fund purchase")
            session.add(
                Action(
                    user_id=user.id,
                    week=week,
                    rank=int(buy_action["rank"]),
                    kind=str(buy_action["kind"]),
                    payload=buy_action,
                )
            )
        return 0
    finally:
        engine.dispose()


def _seed_research_pack(email: str) -> int:
    engine, _schema_name = _owner_engine()
    iso = date.today().isocalendar()
    week = f"{iso.year}-W{iso.week:02}"
    scenarios = [
        ("IN", "C43-IN-ONE", "Synthetic Long Company Name for Phone Layouts", "INR", "123.45"),
        ("IN", "C43-IN-TWO", "Synthetic India Test Company", "INR", "88.10"),
        ("US", "C43-US-ONE", "Synthetic US Test Company One", "USD", "45.67"),
        ("US", "C43-US-TWO", "Synthetic US Test Company Two", "USD", "72.30"),
    ]
    try:
        with Session(engine) as session, session.begin():
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
                {"lock_key": "c43-e2e-research-pack"},
            )
            user = session.scalar(select(User).where(User.email == email.casefold()))
            if user is None:
                raise RuntimeError("The synthetic Ideas household was not found")
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user.id,
                    HouseholdMember.is_self.is_(True),
                )
            )
            if member is None:
                raise RuntimeError("The synthetic Ideas household has no owner member")
            member.investing = {**dict(member.investing or {}), "us_stock_picks_opt_in": True}
            ideas: list[dict[str, object]] = []
            for market, symbol, name, currency, price in scenarios:
                security = session.scalar(
                    select(Security).where(
                        Security.market == market,
                        Security.symbol == symbol,
                    )
                )
                if security is None:
                    security = Security(
                        market=market,
                        symbol=symbol,
                        name=name,
                        security_type="equity",
                        currency=currency,
                        sleeve="equity_india" if market == "IN" else "equity_global",
                        sector="Technology",
                        industry="Software",
                    )
                    session.add(security)
                    session.flush()
                else:
                    security.name = name
                ideas.append(
                    {
                        "security_id": str(security.id),
                        "symbol": symbol,
                        "market": market,
                        "sector": "Technology",
                        "price": price,
                        "quant_decile": 7,
                        "dossier_status": "APPROVED",
                        "valuation_percentile": "60",
                        "confidence": "HIGH",
                        "why": {
                            "reasons": ["Synthetic research note for layout coverage."],
                            "risk": "Synthetic risk note.",
                        },
                        "entry_hint": {
                            "code": "buy_in_parts",
                            "parts": 3,
                            "reasons": ["regular_entry_schedule"],
                        },
                    }
                )
            fund = session.scalar(
                select(Security).where(
                    Security.market == "IN",
                    Security.symbol == "C47-E2E-SP500-FUND",
                )
            )
            if fund is None:
                fund = Security(
                    market="IN",
                    symbol="C47-E2E-SP500-FUND",
                    name="Synthetic S&P 500 fund",
                    security_type="mf",
                    currency="INR",
                    sleeve="equity_global",
                    category="equity:global",
                    sub_category="International / FoF Overseas",
                )
                session.add(fund)
                session.flush()
            existing_lot = session.scalar(
                select(HoldingLot).where(
                    HoldingLot.user_id == user.id,
                    HoldingLot.security_id == fund.id,
                )
            )
            if existing_lot is None:
                session.add(
                    HoldingLot(
                        user_id=user.id,
                        member_id=member.id,
                        security_id=fund.id,
                        quantity=Decimal(1),
                        source="manual",
                        currency="INR",
                        statement_value=Decimal("2000000"),
                        statement_date=date.today(),
                    )
                )
            basket = session.get(FundBasket, "sp500")
            if basket is None:
                basket = FundBasket(
                    id="sp500",
                    label="S&P 500",
                    as_of=date.today(),
                    source="synthetic_e2e",
                    source_url="https://example.test/synthetic-sp500",
                    weights_estimated=True,
                    weight_source="synthetic_e2e",
                    members_total=3,
                    members_found=3,
                )
                session.add(basket)
            else:
                basket.as_of = date.today()
                basket.members_total = 3
                basket.members_found = 3
            session.flush()
            # A bulk delete runs now; ORM deletes would flush after the inserts below
            # and trip the per-basket unique index when a second project reseeds.
            session.execute(delete(FundBasketWeight).where(FundBasketWeight.basket_id == "sp500"))
            session.add_all(
                (
                    FundBasketWeight(
                        basket_id="sp500",
                        market="US",
                        symbol="C43-US-ONE",
                        name="Synthetic US Test Company One",
                        weight_pct=Decimal("1.5"),
                    ),
                    FundBasketWeight(
                        basket_id="sp500",
                        market="US",
                        symbol="C43-US-TWO",
                        name="Synthetic US Test Company Two",
                        weight_pct=Decimal("50"),
                    ),
                    FundBasketWeight(
                        basket_id="sp500",
                        market="US",
                        symbol="C47-E2E-OTHER",
                        name="Synthetic S&P 500 company",
                        weight_pct=Decimal("48.5"),
                    ),
                )
            )
            pack = session.get(ResearchPackRecord, week)
            if pack is None:
                session.add(ResearchPackRecord(week=week, pack={"ideas": ideas}))
            else:
                pack.pack = {"ideas": ideas}
        return 0
    finally:
        engine.dispose()


def _verify_deleted(email: str) -> int:
    engine, _schema_name = _owner_engine()
    try:
        with Session(engine) as session:
            user = session.scalar(select(User).where(User.email == email.casefold()))
        if user is not None:
            raise RuntimeError("Synthetic account rows remain after deletion")
        return 0
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["setup"]:
        return _setup_stack()
    if len(args) == 2 and args[0] == "seed-action":
        return _seed_action(args[1])
    if len(args) == 2 and args[0] == "seed-broker-import-account":
        return _seed_broker_import_account(args[1])
    if len(args) == 2 and args[0] == "seed-icici-portfolio-securities":
        return _seed_icici_portfolio_securities(args[1])
    if len(args) == 2 and args[0] == "seed-money-list-holdings":
        return _seed_money_list_holdings(args[1])
    if len(args) == 2 and args[0] == "seed-holding-reviews":
        return _seed_holding_reviews(args[1])
    if len(args) == 2 and args[0] == "seed-denmark-buy":
        return _seed_denmark_buy_action(args[1])
    if len(args) == 2 and args[0] == "seed-research-pack":
        return _seed_research_pack(args[1])
    if len(args) == 2 and args[0] == "verify-deleted":
        return _verify_deleted(args[1])
    print(
        "Use setup, seed-action <synthetic-email>, seed-holding-reviews, "
        "seed-money-list-holdings <synthetic-email>, seed-research-pack, "
        "or verify-deleted <synthetic-email>."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
