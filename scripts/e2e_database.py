"""Playwright database lifecycle and synthetic state helpers."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session
from tests.db.migrated_database import MigratedDatabase, create_migrated_database
from tests.fixtures.cas.make_pdf import write_encrypted_pdf

from wealth.db.models import (
    Action,
    FxRate,
    GoldPrice,
    HouseholdMember,
    NavDaily,
    PriceEod,
    Security,
    User,
)
from wealth.db.session import normalize_database_url

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
    if len(args) == 2 and args[0] == "verify-deleted":
        return _verify_deleted(args[1])
    print("Use setup, seed-action <synthetic-email>, or verify-deleted <synthetic-email>.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
