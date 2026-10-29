"""PostgreSQL ORM models for user data and shared securities."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative metadata for the wealth database."""


class UserOwned:
    """Common identity and tenant key for user-owned tables."""

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )


def _tenant_identity(table_name: str) -> UniqueConstraint:
    return UniqueConstraint("user_id", "id", name=f"uq_{table_name}_user_id_id")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),
        Index("uq_users_email_lower", func.lower(text("email")), unique=True),
        UniqueConstraint("auth_subject", name="uq_users_auth_subject"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(120))
    auth_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Profile(UserOwned, Base):
    __tablename__ = "profiles"
    __table_args__ = (
        _tenant_identity("profiles"),
        UniqueConstraint("user_id", name="uq_profiles_user_id"),
        CheckConstraint(
            "city_tier IS NULL OR city_tier BETWEEN 1 AND 3", name="ck_profiles_city_tier"
        ),
    )

    tax_regime: Mapped[str | None] = mapped_column(String(16))
    tax_slab_pct: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
    city_tier: Mapped[int | None] = mapped_column(SmallInteger)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, server_default="Asia/Kolkata")
    language: Mapped[str] = mapped_column(String(16), nullable=False, server_default="en")
    risk_answers: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    risk_profile: Mapped[str | None] = mapped_column(String(32))
    financial_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    investing: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    residency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {
            "status": "RI",
            "country": None,
            "us_person": False,
            "planned_return_year": None,
            "settle_plan": "undecided",
        },
        server_default=text(
            "jsonb_build_object('status', 'RI', 'country', NULL, 'us_person', false, "
            "'planned_return_year', NULL, 'settle_plan', 'undecided')"
        ),
    )
    checklist: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class HouseholdMember(UserOwned, Base):
    __tablename__ = "household_members"
    __table_args__ = (
        _tenant_identity("household_members"),
        Index(
            "uq_household_members_one_self_per_user",
            "user_id",
            unique=True,
            postgresql_where=text("is_self"),
        ),
        CheckConstraint(
            "relationship IN ('self', 'spouse', 'child', 'parent', 'other')",
            name="ck_household_members_relationship",
        ),
        CheckConstraint(
            "is_self = (relationship = 'self')", name="ck_household_members_self_relationship"
        ),
    )

    display_name: Mapped[str] = mapped_column(String(40), nullable=False)
    relationship: Mapped[str] = mapped_column(String(40), nullable=False)
    is_self: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    residency: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {
            "status": "RI",
            "country": "IN",
            "us_person": False,
            "planned_return_year": None,
            "settle_plan": "undecided",
        },
        server_default=text(
            '\'{"status":"RI","country":"IN","us_person":false,'
            '"planned_return_year":null,"settle_plan":"undecided"}\'::jsonb'
        ),
    )
    tax: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    investing: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class HouseholdAccess(Base):
    __tablename__ = "household_access"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_household_access_member_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint("role IN ('partner', 'viewer')", name="ck_household_access_role"),
        CheckConstraint("owner_user_id <> grantee_user_id", name="ck_household_access_distinct"),
        Index("ix_household_access_grantee", "grantee_user_id", "revoked_at"),
        Index(
            "uq_household_access_owner_member_active",
            "owner_user_id",
            "member_id",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )

    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    grantee_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class HouseholdInvite(Base):
    __tablename__ = "household_invites"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_household_invites_member_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint("role IN ('partner', 'viewer')", name="ck_household_invites_role"),
        CheckConstraint(
            "token_sha256 ~ '^[0-9a-f]{64}$'", name="ck_household_invites_token_sha256"
        ),
        Index("ix_household_invites_owner_created", "owner_user_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    email_lower: Mapped[str] = mapped_column(String(320), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    member_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    token_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Goal(UserOwned, Base):
    __tablename__ = "goals"
    __table_args__ = (
        _tenant_identity("goals"),
        CheckConstraint("priority BETWEEN 1 AND 5", name="ck_goals_priority"),
        CheckConstraint(
            "currency IN ('INR', 'DKK', 'EUR', 'USD')",
            name="ck_goals_currency",
        ),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_goals_member_tenant",
            ondelete="RESTRICT",
        ),
    )

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    goal_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    target_year: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    priority: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="3")
    member_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))


class Account(UserOwned, Base):
    __tablename__ = "accounts"
    __table_args__ = (
        _tenant_identity("accounts"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_accounts_member_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "masked_number IS NULL OR masked_number ~ '^X*[0-9A-Za-z]{0,4}$'",
            name="ck_accounts_masked_number",
        ),
        CheckConstraint(
            "tax_wrapper IS NULL OR tax_wrapper IN ('ask', 'ordinary', 'pension')",
            name="ck_accounts_tax_wrapper",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    joint: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    institution: Mapped[str] = mapped_column(String(120), nullable=False)
    account_type: Mapped[str] = mapped_column(String(32), nullable=False)
    masked_number: Mapped[str | None] = mapped_column(String(32))
    tax_wrapper: Mapped[str | None] = mapped_column(String(16))


class ManualAsset(UserOwned, Base):
    __tablename__ = "manual_assets"
    __table_args__ = (
        _tenant_identity("manual_assets"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_manual_assets_member_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_manual_assets_account_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "asset_type IN ('savings', 'fd', 'rd', 'ppf', 'epf', 'nps', 'po_scheme', 'bond', "
            "'gold', 'term_insurance', 'health_insurance', 'traditional_policy', 'ulip', "
            "'property', 'loan')",
            name="ck_manual_assets_asset_type",
        ),
        CheckConstraint("status IN ('active', 'closed')", name="ck_manual_assets_status"),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    joint: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    account_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    valuation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    value_cached: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    value_as_of: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AssetHistory(UserOwned, Base):
    __tablename__ = "asset_history"
    __table_args__ = (
        _tenant_identity("asset_history"),
        ForeignKeyConstraint(
            ["user_id", "asset_id"],
            ["manual_assets.user_id", "manual_assets.id"],
            name="fk_asset_history_asset_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "changed_by IN ('user', 'import', 'system')", name="ck_asset_history_changed_by"
        ),
    )

    asset_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    changed_fields: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    changed_by: Mapped[str] = mapped_column(String(16), nullable=False)


class LoanEventRecord(UserOwned, Base):
    __tablename__ = "loan_events"
    __table_args__ = (
        _tenant_identity("loan_events"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_loan_events_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "loan_id"],
            ["manual_assets.user_id", "manual_assets.id"],
            name="fk_loan_events_loan_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        CheckConstraint(
            "kind IN ('prepayment', 'rate_change', 'emi_change', 'balance_update', "
            "'missed_emi', 'closed')",
            name="ck_loan_events_kind",
        ),
        CheckConstraint("amount IS NULL OR amount >= 0", name="ck_loan_events_amount_nonnegative"),
        CheckConstraint(
            "rate_pct IS NULL OR rate_pct BETWEEN 0 AND 60",
            name="ck_loan_events_rate_range",
        ),
        CheckConstraint("emi IS NULL OR emi >= 0", name="ck_loan_events_emi_nonnegative"),
        Index("ix_loan_events_tenant_loan_date", "user_id", "loan_id", "on_date"),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    loan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    on_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    rate_pct: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
    emi: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    note: Mapped[str | None] = mapped_column(String(140))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AssetCashflow(UserOwned, Base):
    __tablename__ = "asset_cashflows"
    __table_args__ = (
        _tenant_identity("asset_cashflows"),
        ForeignKeyConstraint(
            ["user_id", "asset_id"],
            ["manual_assets.user_id", "manual_assets.id"],
            name="fk_asset_cashflows_asset_tenant",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "kind IN ('deposit', 'premium', 'interest', 'maturity', 'withdrawal')",
            name="ck_asset_cashflows_kind",
        ),
    )

    asset_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    on_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    planned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class Security(Base):
    __tablename__ = "securities"
    __table_args__ = (
        Index(
            "uq_securities_market_symbol",
            "market",
            "symbol",
            unique=True,
            postgresql_where=text("symbol IS NOT NULL"),
        ),
        UniqueConstraint("isin", name="uq_securities_isin"),
        Index("ix_securities_amfi_scheme_code", "amfi_scheme_code"),
        Index("ix_securities_nse_symbol", "nse_symbol"),
        CheckConstraint(
            "plan_type IS NULL OR plan_type IN ('direct', 'regular')",
            name="ck_securities_plan_type",
        ),
        CheckConstraint(
            "option IS NULL OR option IN ('growth', 'idcw')",
            name="ck_securities_option",
        ),
        CheckConstraint("ter IS NULL OR ter >= 0", name="ck_securities_ter_non_negative"),
        CheckConstraint("market IN ('IN', 'US', 'DK', 'EU', 'GB')", name="ck_securities_market"),
        CheckConstraint(
            "currency IN ('INR', 'USD', 'EUR', 'DKK', 'GBP', 'SEK')",
            name="ck_securities_currency",
        ),
        CheckConstraint(
            "domicile IS NULL OR domicile ~ '^[A-Z]{2}$'",
            name="ck_securities_domicile",
        ),
        CheckConstraint(
            "security_type IN ('equity', 'etf', 'mf', 'bond', 'sgb')",
            name="ck_securities_security_type",
        ),
        CheckConstraint(
            "sleeve IS NULL OR sleeve IN "
            "('equity_india', 'equity_global', 'debt', 'gold', 'cash', 'real_estate', "
            "'protection')",
            name="ck_securities_sleeve",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    market: Mapped[str] = mapped_column(String(2), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32))
    isin: Mapped[str | None] = mapped_column(String(12))
    exchange: Mapped[str | None] = mapped_column(String(32))
    domicile: Mapped[str | None] = mapped_column(String(2))
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    security_type: Mapped[str] = mapped_column(String(16), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    sleeve: Mapped[str | None] = mapped_column(String(32))
    category: Mapped[str | None] = mapped_column(String(80))
    amfi_scheme_code: Mapped[str | None] = mapped_column(String(16))
    sub_category: Mapped[str | None] = mapped_column(String(80))
    plan_type: Mapped[str | None] = mapped_column(String(12))
    option: Mapped[str | None] = mapped_column(String(12))
    ter: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    ter_as_of: Mapped[date | None] = mapped_column(Date)
    ter_source: Mapped[str | None] = mapped_column(String(80))
    nse_symbol: Mapped[str | None] = mapped_column(String(32))
    series: Mapped[str | None] = mapped_column(String(8))


class PriceEod(Base):
    __tablename__ = "prices_eod"
    __table_args__ = (
        Index("ix_prices_eod_as_of", "as_of"),
        Index("ix_prices_eod_security_as_of", "security_id", "as_of"),
    )

    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class NavDaily(Base):
    __tablename__ = "nav_daily"
    __table_args__ = (
        Index("ix_nav_daily_as_of", "as_of"),
        Index("ix_nav_daily_security_as_of", "security_id", "as_of"),
    )

    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    nav: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class NavMonthly(Base):
    __tablename__ = "nav_monthly"
    __table_args__ = (
        Index("ix_nav_monthly_month_end", "month_end"),
        CheckConstraint("nav > 0", name="ck_nav_monthly_nav_positive"),
    )

    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    month_end: Mapped[date] = mapped_column(Date, primary_key=True)
    nav: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)


class FxRate(Base):
    __tablename__ = "fx_rates"
    __table_args__ = (Index("ix_fx_rates_pair_as_of", "pair", "as_of"),)

    pair: Mapped[str] = mapped_column(String(7), primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    rate: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    publication_lag_days: Mapped[int | None] = mapped_column(SmallInteger)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class GoldPrice(Base):
    __tablename__ = "gold_prices"
    __table_args__ = (Index("ix_gold_prices_as_of", "as_of"),)

    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    inr_per_gram_24k: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MarketdataRun(Base):
    __tablename__ = "marketdata_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('succeeded', 'partial', 'failed')", name="ck_marketdata_runs_status"
        ),
        Index("ix_marketdata_runs_started_at", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    errors: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class FundamentalPIT(Base):
    __tablename__ = "fundamentals_pit"
    __table_args__ = (
        CheckConstraint("period_type IN ('Q', 'FY')", name="ck_fundamentals_pit_period_type"),
        CheckConstraint(
            "value > '-Infinity'::numeric AND value < 'Infinity'::numeric",
            name="ck_fundamentals_pit_value_finite",
        ),
        Index("ix_fundamentals_pit_asof", "security_id", "metric", "known_at"),
        Index("ix_fundamentals_pit_period", "security_id", "metric", "period_end", "period_type"),
    )

    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    metric: Mapped[str] = mapped_column(String(48), primary_key=True)
    period_end: Mapped[date] = mapped_column(Date, primary_key=True)
    period_type: Mapped[str] = mapped_column(String(2), primary_key=True)
    source: Mapped[str] = mapped_column(String(24), primary_key=True)
    known_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    unit: Mapped[str] = mapped_column(String(24), nullable=False)
    filed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    is_latest_for_period: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )


class UniverseMember(Base):
    __tablename__ = "universe_members"
    __table_args__ = (
        CheckConstraint(
            "removed_on IS NULL OR removed_on >= added_on",
            name="ck_universe_members_dates",
        ),
        Index(
            "uq_universe_members_active",
            "universe",
            "security_id",
            unique=True,
            postgresql_where=text("removed_on IS NULL"),
        ),
        Index("ix_universe_members_added_on", "universe", "added_on"),
    )

    universe: Mapped[str] = mapped_column(String(24), primary_key=True)
    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    added_on: Mapped[date] = mapped_column(Date, primary_key=True)
    removed_on: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    survivorship_label: Mapped[str] = mapped_column(String(80), nullable=False)


class HoldingLot(UserOwned, Base):
    __tablename__ = "holdings_lots"
    __table_args__ = (
        _tenant_identity("holdings_lots"),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_holdings_lots_account_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_holdings_lots_member_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_holdings_lots_security",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_holdings_lots_import_job_tenant",
            ondelete="CASCADE",
        ),
    )

    account_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    joint: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    security_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    cost_per_unit: Mapped[Decimal | None] = mapped_column(Numeric(16, 4))
    acquired_on: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    account_ref: Mapped[str | None] = mapped_column(String(16))
    statement_value: Mapped[Decimal | None] = mapped_column(Numeric(16, 2))
    statement_date: Mapped[date | None] = mapped_column(Date)
    import_job_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    trade_ids: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    fx_at_acquisition: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    cost_per_unit_inr: Mapped[Decimal | None] = mapped_column(Numeric(16, 4))
    fx_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    __table_args__ += (
        CheckConstraint("source IN ('cas', 'broker', 'manual')", name="ck_holdings_lots_source"),
        CheckConstraint(
            "account_ref IS NULL OR account_ref ~ '^XXXX[A-Z0-9]{4}$'",
            name="ck_holdings_lots_account_ref_masked",
        ),
    )


class FundLaggingCheck(UserOwned, Base):
    __tablename__ = "fund_lagging_checks"
    __table_args__ = (
        _tenant_identity("fund_lagging_checks"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_fund_lagging_checks_member_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_fund_lagging_checks_security",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "month_end = (date_trunc('month', month_end) + interval '1 month - 1 day')::date",
            name="ck_fund_lagging_checks_month_end",
        ),
        Index(
            "ix_fund_lagging_checks_household_member_security_month",
            "user_id",
            "member_id",
            "security_id",
            "month_end",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    security_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    month_end: Mapped[date] = mapped_column(Date, nullable=False)
    bottom_quartile: Mapped[bool] = mapped_column(Boolean, nullable=False)


class FundTransaction(UserOwned, Base):
    __tablename__ = "fund_transactions"
    __table_args__ = (
        _tenant_identity("fund_transactions"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_fund_transactions_member_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_fund_transactions_security",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "kind IN ('buy', 'sell', 'switch_in', 'switch_out', 'dividend_reinvest', 'other')",
            name="ck_fund_transactions_kind",
        ),
        CheckConstraint("amount >= 0 AND units >= 0", name="ck_fund_transactions_non_negative"),
        CheckConstraint(
            "account_ref IS NULL OR account_ref ~ '^XXXX[A-Z0-9]{4}$'",
            name="ck_fund_transactions_account_ref_masked",
        ),
        Index(
            "ix_fund_transactions_household_member_security_date",
            "user_id",
            "member_id",
            "security_id",
            "transaction_date",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    security_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    units: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    account_ref: Mapped[str | None] = mapped_column(String(16))


class ImportJob(UserOwned, Base):
    __tablename__ = "import_jobs"
    __table_args__ = (
        _tenant_identity("import_jobs"),
        CheckConstraint(
            "status IN ('ready', 'needs_mapping', 'confirmed', 'failed', 'discarded')",
            name="ck_import_jobs_status",
        ),
        CheckConstraint(
            "source IS NULL OR source IN ('nsdl', 'cdsl', 'cams_kfin', 'broker')",
            name="ck_import_jobs_source",
        ),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_import_jobs_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_import_jobs_account_tenant",
            ondelete="SET NULL (account_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
    )

    doc_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(40))
    statement_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str | None] = mapped_column(String(16))
    counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parse_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    account_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    member_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    file_deleted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ImportItem(UserOwned, Base):
    __tablename__ = "import_items"
    __table_args__ = (
        _tenant_identity("import_items"),
        ForeignKeyConstraint(
            ["user_id", "job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_import_items_job_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["user_id", "match_lot_id"],
            ["holdings_lots.user_id", "holdings_lots.id"],
            name="fk_import_items_match_lot_tenant",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["match_security_id"],
            ["securities.id"],
            name="fk_import_items_match_security",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('new', 'changed', 'duplicate', 'unknown', 'accepted', 'rejected', "
            "'enrich', 'replace_with_lots', 'quantity_mismatch')",
            name="ck_import_items_status",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    item_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="new")
    match_lot_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    match_security_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    diff: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    user_override: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ImportTemplate(UserOwned, Base):
    __tablename__ = "import_templates"
    __table_args__ = (
        _tenant_identity("import_templates"),
        UniqueConstraint(
            "user_id", "kind", "header_fingerprint", name="uq_import_templates_household_header"
        ),
        CheckConstraint("kind IN ('holdings', 'tradebook')", name="ck_import_templates_kind"),
        CheckConstraint(
            "header_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_import_templates_fingerprint",
        ),
    )

    name: Mapped[str] = mapped_column(String(80), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    header_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    header_names: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    columns: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RealisedTrade(UserOwned, Base):
    __tablename__ = "realised_trades"
    __table_args__ = (
        _tenant_identity("realised_trades"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_realised_trades_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_realised_trades_account_tenant",
            ondelete="SET NULL (account_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_realised_trades_import_job_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_realised_trades_security",
            ondelete="RESTRICT",
        ),
        CheckConstraint("quantity > 0", name="ck_realised_trades_quantity_positive"),
        CheckConstraint("holding_period IN ('LT', 'ST')", name="ck_realised_trades_holding_period"),
        Index(
            "ix_realised_trades_household_member_date_period",
            "user_id",
            "member_id",
            "trade_date",
            "holding_period",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    account_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    import_job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    security_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    acquired_on: Mapped[date] = mapped_column(Date, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    cost_per_unit: Mapped[Decimal] = mapped_column(Numeric(16, 4), nullable=False)
    sale_price: Mapped[Decimal] = mapped_column(Numeric(16, 4), nullable=False)
    gain: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    holding_period: Mapped[str] = mapped_column(String(2), nullable=False)
    trade_id: Mapped[str | None] = mapped_column(String(80))
    source_broker: Mapped[str] = mapped_column(String(32), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    fx_at_acquisition: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    fx_at_sale: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    cost_per_unit_inr: Mapped[Decimal | None] = mapped_column(Numeric(16, 4))
    sale_price_inr: Mapped[Decimal | None] = mapped_column(Numeric(16, 4))
    gain_inr: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    fx_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class BrokerCashFlowRecord(UserOwned, Base):
    __tablename__ = "broker_cash_flows"
    __table_args__ = (
        _tenant_identity("broker_cash_flows"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_broker_cash_flows_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_broker_cash_flows_account_tenant",
            ondelete="SET NULL (account_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_broker_cash_flows_import_job_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        CheckConstraint("kind IN ('deposit', 'withdrawal')", name="ck_broker_cash_flows_kind"),
        CheckConstraint("amount > 0", name="ck_broker_cash_flows_amount_positive"),
        CheckConstraint(
            "currency IN ('INR', 'USD', 'EUR', 'DKK', 'GBP', 'SEK')",
            name="ck_broker_cash_flows_currency",
        ),
        Index(
            "ix_broker_cash_flows_household_member_date",
            "user_id",
            "member_id",
            "transaction_date",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    account_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    import_job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    source_broker: Mapped[str] = mapped_column(String(32), nullable=False)
    transaction_id: Mapped[str | None] = mapped_column(String(80))


class NetworthSnapshot(UserOwned, Base):
    __tablename__ = "networth_snapshots"
    __table_args__ = (
        _tenant_identity("networth_snapshots"),
        UniqueConstraint("user_id", "on_date", name="uq_networth_snapshots_user_date"),
    )

    on_date: Mapped[date] = mapped_column(Date, nullable=False)
    totals: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class Action(UserOwned, Base):
    __tablename__ = "actions"
    __table_args__ = (
        _tenant_identity("actions"),
        CheckConstraint("week ~ '^[0-9]{4}-W[0-9]{2}$'", name="ck_actions_week"),
        CheckConstraint("status IN ('proposed', 'done', 'dismissed')", name="ck_actions_status"),
    )

    week: Mapped[str] = mapped_column(String(8), nullable=False)
    rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="proposed")
    done_by: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class WeeklyPlan(Base):
    __tablename__ = "weekly_plans"
    __table_args__ = (
        CheckConstraint(
            "week ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'",
            name="ck_weekly_plans_week",
        ),
        Index("ix_weekly_plans_week", "week"),
    )

    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    week: Mapped[str] = mapped_column(String(8), primary_key=True)
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ResearchPackRecord(Base):
    """One frozen weekly research pack, shared read-only policy input."""

    __tablename__ = "research_packs"
    __table_args__ = (
        CheckConstraint(
            "week ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'",
            name="ck_research_packs_week",
        ),
    )

    week: Mapped[str] = mapped_column(String(8), primary_key=True)
    pack: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ExitCard(UserOwned, Base):
    __tablename__ = "exit_cards"
    __table_args__ = (
        _tenant_identity("exit_cards"),
        ForeignKeyConstraint(
            ["user_id", "lot_id"],
            ["holdings_lots.user_id", "holdings_lots.id"],
            name="fk_exit_cards_lot_tenant",
            ondelete="CASCADE",
        ),
    )

    lot_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    rules: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Reminder(UserOwned, Base):
    __tablename__ = "reminders"
    __table_args__ = (
        _tenant_identity("reminders"),
        ForeignKeyConstraint(
            ["user_id", "asset_id"],
            ["manual_assets.user_id", "manual_assets.id"],
            name="fk_reminders_asset_tenant",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "user_id", "asset_id", "due_on", "kind", name="uq_reminders_delivery_dedupe"
        ),
        Index("ix_reminders_notify_on", "user_id", "notify_on", "sent_at"),
    )

    asset_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    due_on: Mapped[date] = mapped_column(Date, nullable=False)
    notify_on: Mapped[date | None] = mapped_column(Date)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PushSubscription(UserOwned, Base):
    __tablename__ = "push_subscriptions"
    __table_args__ = (_tenant_identity("push_subscriptions"),)

    endpoint: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    keys: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AuthSession(UserOwned, Base):
    """One login's rotating refresh-token session on a device."""

    __tablename__ = "auth_sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "id", name="uq_auth_sessions_user_id_id"),
        UniqueConstraint("token_hash", name="uq_auth_sessions_token_hash"),
        CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="ck_auth_sessions_token_hash"),
        CheckConstraint(
            "previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{64}$'",
            name="ck_auth_sessions_previous_hash",
        ),
        Index("ix_auth_sessions_user_expiry", "user_id", "expires_at"),
        Index("ix_auth_sessions_previous_hash", "previous_hash"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device_label: Mapped[str] = mapped_column(String(60), nullable=False)


class StatementInboxKey(UserOwned, Base):
    """A one-way upload credential for a household's Gmail helper."""

    __tablename__ = "statement_inbox_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "id", name="uq_statement_inbox_keys_user_id_id"),
        UniqueConstraint("key_hash", name="uq_statement_inbox_keys_key_hash"),
        CheckConstraint("key_hash ~ '^[0-9a-f]{64}$'", name="ck_statement_inbox_keys_key_hash"),
        Index("ix_statement_inbox_keys_user_created", "user_id", "created_at"),
    )

    created_by: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StatementInboxItem(UserOwned, Base):
    """An unparsed email attachment waiting for household review."""

    __tablename__ = "statement_inbox_items"
    __table_args__ = (
        UniqueConstraint("user_id", "id", name="uq_statement_inbox_items_user_id_id"),
        UniqueConstraint(
            "user_id", "message_id", "file_hash", name="uq_statement_inbox_items_message_file"
        ),
        ForeignKeyConstraint(
            ["user_id", "key_id"],
            ["statement_inbox_keys.user_id", "statement_inbox_keys.id"],
            name="fk_statement_inbox_items_key_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_statement_inbox_items_job_tenant",
            ondelete="SET NULL (import_job_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        CheckConstraint(
            "sender_type IN ('cams', 'kfintech', 'mfcentral', 'nsdl', 'cdsl', 'other')",
            name="ck_statement_inbox_items_sender_type",
        ),
        CheckConstraint(
            "status IN ('needs_password', 'review', 'unreadable')",
            name="ck_statement_inbox_items_status",
        ),
        CheckConstraint("file_hash ~ '^[0-9a-f]{64}$'", name="ck_statement_inbox_items_file_hash"),
        Index("ix_statement_inbox_items_user_received", "user_id", "received_at"),
        Index("ix_statement_inbox_items_key_received", "key_id", "received_at"),
    )

    key_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    sender_type: Mapped[str] = mapped_column(String(16), nullable=False)
    sender: Mapped[str] = mapped_column(String(320), nullable=False)
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message_id: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    pdf_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    status: Mapped[str] = mapped_column(String(24), nullable=False, server_default="needs_password")
    error_code: Mapped[str | None] = mapped_column(String(40))
    import_job_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class StatementSavedPassword(UserOwned, Base):
    """An AES-GCM protected password remembered for one statement sender."""

    __tablename__ = "statement_saved_passwords"
    __table_args__ = (
        UniqueConstraint("user_id", "id", name="uq_statement_saved_passwords_user_id_id"),
        UniqueConstraint("user_id", "sender_type", name="uq_statement_saved_passwords_sender"),
        CheckConstraint(
            "sender_type IN ('cams', 'kfintech', 'mfcentral', 'nsdl', 'cdsl', 'other')",
            name="ck_statement_saved_passwords_sender_type",
        ),
        CheckConstraint("octet_length(nonce) = 12", name="ck_statement_saved_passwords_nonce"),
    )

    sender_type: Mapped[str] = mapped_column(String(16), nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class HighlightDismissal(Base):
    """A login's dismissal state for one highlight in one shared household."""

    __tablename__ = "highlight_dismissals"
    __table_args__ = (
        CheckConstraint("highlight_key ~ '^[a-z0-9_]{1,100}$'", name="ck_highlight_dismissals_key"),
        CheckConstraint(
            "fingerprint ~ '^[0-9a-f]{24}$'", name="ck_highlight_dismissals_fingerprint"
        ),
        Index(
            "ix_highlight_dismissals_household_login",
            "household_owner_id",
            "login_user_id",
        ),
    )

    login_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    household_owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    highlight_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(24), nullable=False)
    dismissed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Consent(UserOwned, Base):
    __tablename__ = "consents"
    __table_args__ = (_tenant_identity("consents"),)

    purpose: Mapped[str] = mapped_column(String(80), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AlertLog(UserOwned, Base):
    """Delivery metadata only; message bodies are represented by a content hash."""

    __tablename__ = "alerts_log"
    __table_args__ = (
        _tenant_identity("alerts_log"),
        UniqueConstraint("user_id", "dedupe_key", name="uq_alerts_log_user_dedupe"),
        Index("ix_alerts_log_user_sent", "user_id", "sent_at"),
    )

    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    week: Mapped[str | None] = mapped_column(String(8))
    dedupe_key: Mapped[str] = mapped_column(String(160), nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(Text)
    body_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SentinelEvent(Base):
    """Shared, deterministic Sentinel event facts consumed by the urgent alert job."""

    __tablename__ = "sentinel_events"
    __table_args__ = (
        UniqueConstraint("security_id", "event_code", "source", name="uq_sentinel_event_source"),
        Index("ix_sentinel_events_urgent_detected", "urgent", "detected_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    security_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="CASCADE"), nullable=False
    )
    event_code: Mapped[str] = mapped_column(String(48), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_tool_call_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    urgent: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class AuditLog(UserOwned, Base):
    __tablename__ = "audit_log"
    __table_args__ = (_tenant_identity("audit_log"),)

    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    actor: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
