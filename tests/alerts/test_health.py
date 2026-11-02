"""Deterministic health evaluation, output, and read-only database coverage."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session
from tests.db.migrated_database import MigratedDatabase, create_migrated_database

from wealth.alerts import health
from wealth.alerts.health import (
    DayRowCount,
    HealthFacts,
    MarketdataRunFact,
    evaluate_health,
    read_only_transaction,
    weekday_sessions_oldest_first,
)
from wealth.db.models import (
    AlertLog,
    Consent,
    FxRate,
    HouseholdAccess,
    HouseholdMember,
    MarketdataRun,
    NavDaily,
    PriceEod,
    Profile,
    ResearchPackRecord,
    Security,
    User,
    WeeklyPlan,
)

_NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
_SATURDAY = datetime(2026, 10, 31, 19, 0, tzinfo=UTC)
_SUNDAY_BEFORE = datetime(2026, 10, 31, 23, 59, tzinfo=UTC)
_SUNDAY_START = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _outside_github_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI sets these, and the CLI then prints annotations before its JSON line.
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def _healthy_facts() -> HealthFacts:
    weekly_sources = {
        "universes": {"universes_loaded": 1},
        "nse_history": {"prices_eod": 1, "sessions_requested": 1},
        "edgar": {"fundamentals_pit": 0, "symbols": 1},
        "us_index_pe": {"index_valuations": 1},
        "nse_fundamentals": {"fundamentals_pit": 0, "symbols": 1},
    }
    run = MarketdataRunFact(
        started_at=datetime(2026, 10, 30, 19, 0, tzinfo=UTC),
        status="succeeded",
        sources=tuple(weekly_sources),
        source_counts=weekly_sources,
    )
    price_day = DayRowCount(date(2026, 10, 30), 10)
    return HealthFacts(
        india_prices=(price_day,),
        india_fund_prices=(price_day,),
        abroad_prices=(price_day,),
        fx_latest={pair: date(2026, 10, 30) for pair in ("DKK/INR", "USD/INR", "EUR/INR")},
        marketdata_runs=(run,),
        research_pack_week="2026-W44",
        research_pack_idea_counts={"IN": 1, "US": 1},
        weekly_plan_week="2026-W44",
        eligible_plan_count=0,
        missing_plan_count=0,
        eligible_email_count=0,
        missing_email_count=0,
        email_sending_configured=False,
    )


def _checks(result: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [item for item in result["checks"] if item["key"] == key]


def test_fresh_market_data_and_empty_history_baseline_are_ok() -> None:
    result = evaluate_health(_healthy_facts(), _NOW)

    assert result["ok"] is True
    assert all(item["level"] == "ok" for item in result["checks"])
    assert _checks(result, "india_prices")[0]["facts"]["expected_date"] == "2026-10-30"


def test_price_data_older_than_two_weekday_sessions_is_a_problem() -> None:
    facts = replace(_healthy_facts(), india_prices=(DayRowCount(date(2026, 10, 27), 10),))

    check = _checks(evaluate_health(facts, _NOW), "india_prices")[0]

    assert check["level"] == "problem"
    assert check["facts"]["trading_days_old"] == 3
    assert "Rerun Daily market data." in check["message"]


def test_missing_fund_and_overseas_prices_name_the_workflow() -> None:
    facts = replace(_healthy_facts(), india_fund_prices=(), abroad_prices=())

    result = evaluate_health(facts, _NOW)
    fund_check = _checks(result, "india_fund_prices")[0]
    abroad_check = _checks(result, "abroad_prices")[0]

    assert fund_check["level"] == "problem"
    assert "No Indian fund prices" in fund_check["message"]
    assert abroad_check["level"] == "problem"
    assert "Rerun Overseas market data." in abroad_check["message"]


def test_price_messages_use_natural_sentence_case_for_overseas_labels() -> None:
    missing_facts = replace(_healthy_facts(), india_prices=(), abroad_prices=())
    missing = evaluate_health(missing_facts, _NOW)
    assert _checks(missing, "india_prices")[0]["message"] == (
        "No India share prices are stored. Rerun Daily market data."
    )
    assert _checks(missing, "abroad_prices")[0]["message"] == (
        "No overseas share prices are stored. Rerun Overseas market data."
    )

    fresh = evaluate_health(_healthy_facts(), _NOW)
    assert _checks(fresh, "abroad_prices")[0]["message"].startswith(
        "Overseas share prices are up to date."
    )

    stale_facts = replace(_healthy_facts(), abroad_prices=(DayRowCount(date(2026, 10, 27), 10),))
    stale = evaluate_health(stale_facts, _NOW)
    assert _checks(stale, "abroad_prices")[0]["message"].startswith(
        "Overseas share prices are 3 trading days old."
    )

    low_volume_facts = replace(
        _healthy_facts(),
        abroad_prices=(
            DayRowCount(date(2026, 10, 30), 3),
            *(DayRowCount(date(2026, 10, 29) - timedelta(days=offset), 10) for offset in range(10)),
        ),
    )
    low_volume = evaluate_health(low_volume_facts, _NOW)
    assert _checks(low_volume, "abroad_prices")[0]["message"].startswith(
        "Overseas share prices stored fewer than half the usual rows."
    )


def test_newest_price_date_with_less_than_half_median_rows_is_a_problem() -> None:
    newest = DayRowCount(date(2026, 10, 30), 4)
    earlier = tuple(
        DayRowCount(date(2026, 10, 29) - timedelta(days=offset), 10)
        for offset in range(PRICE_HISTORY_LENGTH)
    )
    facts = replace(_healthy_facts(), india_prices=(newest, *earlier))

    check = _checks(evaluate_health(facts, _NOW), "india_prices")[0]

    assert check["level"] == "problem"
    assert check["facts"]["previous_dates"] == PRICE_HISTORY_LENGTH
    assert "fewer than half" in check["message"]


PRICE_HISTORY_LENGTH = 10


@pytest.mark.parametrize(
    ("latest", "expected_level"),
    [(date(2026, 10, 29), "ok"), (date(2026, 10, 27), "problem"), (None, "problem")],
)
def test_fx_pair_freshness_uses_four_calendar_days(
    latest: date | None, expected_level: str
) -> None:
    facts = _healthy_facts()
    rates = dict(facts.fx_latest)
    rates["USD/INR"] = latest
    check = next(
        item
        for item in _checks(evaluate_health(replace(facts, fx_latest=rates), _NOW), "fx")
        if "USD/INR" in item["message"]
    )

    assert check["level"] == expected_level
    if expected_level == "problem":
        assert "USD/INR" in check["message"]


def test_weekly_data_accepts_zero_new_filings_when_symbols_were_processed() -> None:
    result = evaluate_health(_healthy_facts(), _NOW)

    assert _checks(result, "weekly_data")[0]["level"] == "ok"


def test_weekly_data_fails_when_a_source_has_no_recent_run() -> None:
    facts = replace(_healthy_facts(), marketdata_runs=())

    check = _checks(evaluate_health(facts, _NOW), "weekly_data")[0]

    assert check["level"] == "problem"
    assert "nse_history" in check["facts"]["missing_sources"]


def test_weekly_data_fails_when_a_source_stored_zero_rows() -> None:
    facts = _healthy_facts()
    run = replace(
        facts.marketdata_runs[0],
        source_counts={
            **facts.marketdata_runs[0].source_counts,
            "universes": {"universes_loaded": 0},
        },
    )

    check = _checks(evaluate_health(replace(facts, marketdata_runs=(run,)), _NOW), "weekly_data")[0]

    assert check["level"] == "problem"
    assert check["facts"]["empty_sources"] == ["universes"]


def test_nse_history_zero_rows_is_ok_when_no_sessions_needed() -> None:
    facts = _healthy_facts()
    run = replace(
        facts.marketdata_runs[0],
        source_counts={
            **facts.marketdata_runs[0].source_counts,
            "nse_history": {"prices_eod": 0, "sessions_requested": 0},
        },
    )

    check = _checks(evaluate_health(replace(facts, marketdata_runs=(run,)), _NOW), "weekly_data")[0]

    assert check["level"] == "ok"


def test_recent_failed_runs_are_problems_and_partial_runs_are_warnings() -> None:
    facts = _healthy_facts()
    recent = replace(facts.marketdata_runs[0], started_at=datetime(2026, 11, 1, 10, tzinfo=UTC))
    partial = replace(recent, status="partial", error_sources=("edgar",))
    failed = replace(recent, status="failed", error_sources=("us_index_pe",))

    partial_check = _checks(
        evaluate_health(replace(facts, marketdata_runs=(partial,)), _NOW), "failed_runs"
    )[0]
    failed_check = _checks(
        evaluate_health(replace(facts, marketdata_runs=(failed,)), _NOW), "failed_runs"
    )[0]

    assert partial_check["level"] == "warning"
    assert "edgar" in partial_check["facts"]["partial_sources"]
    assert failed_check["level"] == "problem"
    assert "us_index_pe" in failed_check["facts"]["failed_sources"]


def test_research_pack_without_one_market_is_a_warning_naming_that_market() -> None:
    facts = replace(_healthy_facts(), research_pack_idea_counts={"IN": 2, "US": 0})

    check = _checks(evaluate_health(facts, _NOW), "research_pack")[0]

    assert check["level"] == "warning"
    assert "US" in check["message"]


def test_research_pack_missing_after_sunday_midnight_is_a_problem() -> None:
    facts = replace(_healthy_facts(), research_pack_idea_counts=None)

    check = _checks(evaluate_health(facts, _SUNDAY_START), "research_pack")[0]

    assert check["level"] == "problem"


def test_not_due_windows_cover_research_data_and_weekly_user_jobs() -> None:
    facts = replace(
        _healthy_facts(),
        research_pack_idea_counts=None,
        missing_plan_count=3,
        eligible_plan_count=3,
        missing_email_count=2,
        eligible_email_count=2,
        email_sending_configured=True,
    )
    saturday = evaluate_health(facts, _SATURDAY)
    sunday_before = evaluate_health(facts, _SUNDAY_BEFORE)
    sunday_after = evaluate_health(facts, _SUNDAY_START)

    assert _checks(saturday, "research_pack")[0]["level"] == "ok"
    assert "not_due_until" in _checks(saturday, "research_pack")[0]["facts"]
    assert _checks(saturday, "weekly_data")[0]["level"] == "ok"
    assert _checks(sunday_before, "research_pack")[0]["level"] == "ok"
    assert _checks(sunday_before, "weekly_plans")[0]["level"] == "ok"
    assert _checks(sunday_before, "weekly_email")[0]["level"] == "ok"
    assert _checks(sunday_after, "research_pack")[0]["level"] == "problem"
    assert (
        _checks(sunday_after, "weekly_data")[0]["level"] != "ok"
        or "not_due_until" not in _checks(sunday_after, "weekly_data")[0]["facts"]
    )
    assert _checks(sunday_after, "weekly_plans")[0]["facts"]["not_due_until"].startswith(
        "2026-11-02T00:00:00"
    )


def test_weekly_plan_and_email_checks_run_on_monday() -> None:
    facts = replace(
        _healthy_facts(),
        eligible_plan_count=4,
        missing_plan_count=2,
        eligible_email_count=3,
        missing_email_count=1,
        email_sending_configured=True,
    )
    result = evaluate_health(facts, _NOW)

    plans = _checks(result, "weekly_plans")[0]
    email = _checks(result, "weekly_email")[0]
    assert plans["level"] == "problem" and "2 households" in plans["message"]
    assert email["level"] == "problem" and "1 opted-in user" in email["message"]


def test_weekend_uses_friday_as_the_last_completed_session() -> None:
    result = evaluate_health(_healthy_facts(), _SATURDAY)

    assert _checks(result, "india_prices")[0]["facts"]["expected_date"] == "2026-10-30"


def test_single_weekday_holiday_stays_inside_the_two_session_tolerance() -> None:
    # The calendar helper treats a holiday as a weekday; the two-session allowance covers one.
    assert weekday_sessions_oldest_first(date(2026, 1, 15), date(2026, 1, 19)) == 2


def test_email_check_is_skipped_when_sending_is_not_configured() -> None:
    facts = replace(
        _healthy_facts(),
        email_sending_configured=False,
        eligible_email_count=2,
        missing_email_count=2,
    )

    check = _checks(evaluate_health(facts, _NOW), "weekly_email")[0]

    assert check["level"] == "ok"
    assert "check is skipped" in check["message"]


def test_github_annotations_and_step_summary_use_check_results(tmp_path: Path) -> None:
    summary_path = tmp_path / "step-summary.md"
    output = StringIO()
    result = {
        "checks": [
            {"key": "india_prices", "level": "problem", "message": "Rerun Daily market data."},
            {"key": "research_pack", "level": "warning", "message": "Rerun Weekly research."},
        ]
    }

    health.emit_github_outputs(
        result,
        environ={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary_path)},
        stdout=output,
    )

    assert "::error title=india_prices::Rerun Daily market data." in output.getvalue()
    assert "::warning title=research_pack::Rerun Weekly research." in output.getvalue()
    summary = summary_path.read_text(encoding="utf-8")
    assert "| india_prices | problem | Rerun Daily market data. |" in summary
    assert "| research_pack | warning | Rerun Weekly research. |" in summary


class _FakeEngine:
    def dispose(self) -> None:
        return None


def _stub_cli(monkeypatch: pytest.MonkeyPatch, facts: HealthFacts, database_url: str) -> None:
    monkeypatch.setenv("OWNER_DATABASE_URL", database_url)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("EMAIL_PROVIDER", raising=False)
    monkeypatch.setattr(health, "normalize_database_url", lambda value: value)
    monkeypatch.setattr(health, "create_engine", lambda *_args, **_kwargs: _FakeEngine())
    monkeypatch.setattr(
        health,
        "load_health_facts",
        lambda _engine, _now, *, email_sending_configured: facts,
    )


def test_cli_exit_codes_and_output_never_echo_database_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_url = "postgresql://health_user:health_password@127.0.0.1/health_test"
    _stub_cli(monkeypatch, _healthy_facts(), fake_url)

    success = health.main(["--now", "2026-11-02T09:00:00+00:00"])
    success_output = capsys.readouterr()
    assert success == 0
    assert json.loads(success_output.out)["ok"] is True

    stale = replace(_healthy_facts(), india_prices=())
    _stub_cli(monkeypatch, stale, fake_url)
    failure = health.main(["--now", "2026-11-02T09:00:00Z"])
    failure_output = capsys.readouterr()
    assert failure == 1
    assert json.loads(failure_output.out)["ok"] is False

    monkeypatch.delenv("OWNER_DATABASE_URL", raising=False)
    missing = health.main(["--now", "2026-11-02T09:00:00Z"])
    missing_output = capsys.readouterr()
    assert missing == 2
    assert "OWNER_DATABASE_URL or TEST_DATABASE_URL" in missing_output.err
    all_output = success_output.out + success_output.err + failure_output.out + failure_output.err
    assert fake_url not in all_output
    assert "health_password" not in all_output


@pytest.fixture
def health_database() -> Any:
    if not os.getenv("TEST_DATABASE_URL"):
        pytest.skip("TEST_DATABASE_URL is not set")
    with create_migrated_database(schema_prefix="health", login_prefix="health_test") as database:
        yield database


def _seed_health_database(database: MigratedDatabase) -> None:
    run_time = datetime(2026, 10, 30, 19, 0, tzinfo=UTC)
    with Session(database.owner_engine) as session, session.begin():
        india = Security(
            market="IN",
            symbol="HEALTH-IN",
            name="Synthetic India security",
            security_type="equity",
            currency="INR",
            sleeve="equity_india",
        )
        fund = Security(
            market="IN",
            symbol="HEALTH-FUND",
            name="Synthetic India fund",
            security_type="mf",
            currency="INR",
            sleeve="equity_india",
        )
        abroad = Security(
            market="US",
            symbol="HEALTH-US",
            name="Synthetic overseas security",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
        )
        session.add_all([india, fund, abroad])
        session.flush()
        for security, source in ((india, "NSE"), (abroad, "AlphaVantage")):
            session.add(
                PriceEod(
                    security_id=security.id,
                    as_of=date(2026, 10, 30),
                    close=Decimal("1.00"),
                    source=source,
                    source_url="https://example.invalid/health",
                    fetched_at=run_time,
                )
            )
        session.add(
            NavDaily(
                security_id=fund.id,
                as_of=date(2026, 10, 30),
                nav=Decimal("1.00000000"),
                source="AMFI",
                fetched_at=run_time,
            )
        )
        for pair in ("DKK/INR", "USD/INR", "EUR/INR"):
            session.add(
                FxRate(
                    pair=pair,
                    as_of=date(2026, 10, 30),
                    rate=Decimal("1.00000000"),
                    source="Synthetic",
                    source_url="https://example.invalid/health",
                    publication_lag_days=0,
                    fetched_at=run_time,
                )
            )
        session.add(
            MarketdataRun(
                started_at=run_time,
                finished_at=run_time,
                status="succeeded",
                counts={
                    "universes": {"universes_loaded": 1},
                    "nse_history": {"prices_eod": 1, "sessions_requested": 1},
                    "edgar": {"fundamentals_pit": 0, "symbols": 1},
                    "us_index_pe": {"index_valuations": 1},
                    "nse_fundamentals": {"fundamentals_pit": 0, "symbols": 1},
                },
                errors={},
            )
        )
        session.add(
            ResearchPackRecord(
                week="2026-W44",
                pack={"markets": {"IN": {"idea_count": 1}, "US": {"idea_count": 1}}},
            )
        )


@pytest.mark.db
def test_database_loader_emits_json_and_rejects_writes(
    health_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_health_database(health_database)
    monkeypatch.delenv("OWNER_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "TEST_DATABASE_URL",
        health_database.owner_engine.url.render_as_string(hide_password=False),
    )
    monkeypatch.delenv("EMAIL_PROVIDER", raising=False)

    exit_code = health.main(["--now", "2026-11-02T09:00:00+00:00"])
    output = capsys.readouterr()

    payload = json.loads(output.out)
    assert exit_code == 0, payload
    assert payload["ok"] is True
    assert payload["checked_at"] == "2026-11-02T09:00:00Z"
    assert "postgresql://" not in output.out + output.err
    assert "password" not in output.out.casefold() + output.err.casefold()
    with (
        pytest.raises(DBAPIError),
        read_only_transaction(health_database.owner_engine) as connection,
    ):
        connection.execute(
            text("INSERT INTO research_packs (week, pack) VALUES ('2026-W45', '{}'::jsonb)")
        )


@pytest.mark.db
def test_weekly_email_counts_only_eligible_plan_recipients(
    health_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_health_database(health_database)
    week = "2026-W44"
    granted_at = datetime(2026, 10, 20, tzinfo=UTC)
    users = {
        key: User(email=f"{key}@example.test", auth_subject=f"health-{key}")
        for key in (
            "owner_sent",
            "owner_missing",
            "partner_owner",
            "partner_login",
            "no_plan_owner",
            "withdrawn_owner",
            "deleted_owner",
        )
    }
    users["deleted_owner"].deleted_at = granted_at + timedelta(days=1)
    with Session(health_database.owner_engine) as session, session.begin():
        session.add_all(users.values())
        session.flush()
        profile_owners = (
            "owner_sent",
            "owner_missing",
            "partner_owner",
            "no_plan_owner",
            "withdrawn_owner",
            "deleted_owner",
        )
        session.add_all(Profile(user_id=users[key].id) for key in profile_owners)
        session.add_all(
            Consent(
                user_id=user.id,
                purpose="weekly_email",
                granted_at=granted_at,
                withdrawn_at=granted_at + timedelta(days=1) if key == "withdrawn_owner" else None,
            )
            for key, user in users.items()
            if key != "partner_owner"
        )
        session.add_all(
            WeeklyPlan(user_id=users[key].id, week=week, plan={})
            for key in (
                "owner_sent",
                "owner_missing",
                "partner_owner",
                "withdrawn_owner",
                "deleted_owner",
            )
        )
        partner_member = HouseholdMember(
            user_id=users["partner_owner"].id,
            display_name="Synthetic member",
            relationship="spouse",
            is_self=False,
        )
        session.add(partner_member)
        session.flush()
        session.add(
            HouseholdAccess(
                owner_user_id=users["partner_owner"].id,
                grantee_user_id=users["partner_login"].id,
                member_id=partner_member.id,
                role="partner",
                created_at=granted_at,
            )
        )
        session.add(
            AlertLog(
                user_id=users["owner_sent"].id,
                kind="weekly",
                week=week,
                dedupe_key=f"weekly:{week}:{users['owner_sent'].id}",
                subject="Weekly brief",
                body_sha256="a" * 64,
            )
        )

    monkeypatch.delenv("OWNER_DATABASE_URL", raising=False)
    monkeypatch.setenv(
        "TEST_DATABASE_URL",
        health_database.owner_engine.url.render_as_string(hide_password=False),
    )
    monkeypatch.setenv("EMAIL_PROVIDER", "brevo")

    exit_code = health.main(["--now", "2026-11-02T09:00:00+00:00"])
    output = capsys.readouterr()
    payload = json.loads(output.out)
    email_check = _checks(payload, "weekly_email")[0]

    assert exit_code == 1
    assert email_check["level"] == "problem"
    assert email_check["facts"] == {
        "week": week,
        "eligible_count": 2,
        "missing_count": 1,
    }
