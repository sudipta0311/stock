from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from wealth.db.models import FeedItem, Security
from wealth.research.agenda import build_cio_agenda
from wealth.research.etf_selector import (
    CandidateMetrics,
    CategoryRule,
    SelectorRules,
    _amc_international_suspension,
    _score,
    matches_category,
)
from wealth.research.router import FakeRouter, RunBudget, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.rules_admin import apply_proposal_to_rules
from wealth.research.watchers import (
    FeedFinding,
    RateDocument,
    _exact_rate_quotes,
    _period_from_text,
    build_rate_proposals,
    build_rulebook_proposals,
)

_ROOT = Path(__file__).resolve().parents[2]


def _category() -> CategoryRule:
    return CategoryRule(
        "in_nifty50",
        "India large companies",
        "IN",
        ("etf", "mf"),
        "NIFTY 50",
        ("nifty 50",),
        ("next 50",),
    )


def _selector_rules() -> SelectorRules:
    return SelectorRules(
        (_category(),),
        90,
        20,
        Decimal("5"),
        Decimal("500"),
        Decimal("1.0"),
        {
            "liquidity": Decimal("30"),
            "premium": Decimal("25"),
            "tracking_difference": Decimal("25"),
            "expense_ratio": Decimal("15"),
            "assets": Decimal("5"),
        },
    )


def test_selector_matches_by_category_and_rejects_other_markets() -> None:
    category = _category()
    matching = Security(
        market="IN",
        symbol="SYN1",
        name="Synthetic Nifty 50 Fund",
        security_type="etf",
        currency="INR",
    )
    excluded = Security(
        market="IN",
        symbol="SYN2",
        name="Synthetic Nifty Next 50 Fund",
        security_type="etf",
        currency="INR",
    )
    foreign = Security(
        market="US",
        symbol="SYN3",
        name="Synthetic Nifty 50 Fund",
        security_type="etf",
        currency="USD",
    )

    assert matches_category(matching, category)
    assert not matches_category(excluded, category)
    assert not matches_category(foreign, category)


def test_selector_flags_only_a_recent_explicit_international_subscription_pause() -> None:
    item = FeedItem(
        source="pib",
        guid="synthetic-amc-notice",
        title="Synthetic AMC pauses international fund subscriptions",
        url="https://example.test/amc-notice",
        published_at=datetime(2026, 10, 30, tzinfo=UTC),
        summary="The synthetic notice confirms a pause in new subscriptions.",
    )
    old_item = FeedItem(
        source="pib",
        guid="synthetic-old-notice",
        title="Synthetic overseas fund subscriptions paused",
        url="https://example.test/old-notice",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        summary="An old notice.",
    )

    assert _amc_international_suspension([item], as_of=date(2026, 10, 30)) is True
    assert _amc_international_suspension([old_item], as_of=date(2026, 10, 30)) is None


def test_missing_ter_has_no_score_contribution() -> None:
    rules = _selector_rules()
    missing = CandidateMetrics(Decimal("100"), Decimal("0"), Decimal("0"), None, None)
    available = CandidateMetrics(Decimal("100"), Decimal("0"), Decimal("0"), Decimal("0.2"), None)

    assert _score(missing, rules, is_etf=True) < _score(available, rules, is_etf=True)


def test_rate_watcher_uses_exact_quote_and_detects_changes() -> None:
    changed_quote = "Public Provident Fund pays 7.2% for Q1 FY 2027-28."
    unchanged_quote = "Public Provident Fund pays 7.1% for Q3 FY 2026-27."
    documents = [
        RateDocument(
            "india_post", "Savings rates", "https://www.indiapost.gov.in/savings", changed_quote
        ),
        RateDocument(
            "dea_notice", "Small-savings period", "https://dea.gov.in/notice", "Q1 FY 2027-28"
        ),
    ]

    changed = build_rate_proposals(documents, [changed_quote])
    unchanged_documents = [
        RateDocument(
            "india_post", "Savings rates", "https://www.indiapost.gov.in/savings", unchanged_quote
        ),
        RateDocument(
            "dea_notice", "Small-savings period", "https://dea.gov.in/notice", "Q3 FY 2026-27"
        ),
    ]
    unchanged = build_rate_proposals(unchanged_documents, [unchanged_quote])

    assert len(changed) == 1
    assert changed[0]["proposed_value"] == {
        "from": "2027-04-01",
        "to": "2027-06-30",
        "rate_pct": "7.2",
        "source_url": "https://www.indiapost.gov.in/savings",
    }
    assert unchanged == []


def test_nano_rate_quotes_are_checked_as_exact_source_substrings() -> None:
    quote = "Public Provident Fund pays 7.2% for Q1 FY 2027-28."
    docs = [
        RateDocument("india_post", "Savings rates", "https://www.indiapost.gov.in/savings", quote)
    ]
    router = FakeRouter()
    router.register("nano", {"quotes": [quote, "Invented rate: 7.3%."]})
    context = RunContext(budget=RunBudget(max_usd=Decimal("1"), max_calls=5))

    assert _exact_rate_quotes(router, docs, context) == [quote]
    assert router.calls == 1


def test_rate_period_parser_accepts_official_unicode_dash_punctuation() -> None:
    assert _period_from_text("Q1 FY 2027–28") == (
        date(2027, 4, 1),
        date(2027, 6, 30),
    )
    assert _period_from_text("FY 2027—28", epf=True) == (
        date(2027, 4, 1),
        date(2028, 3, 31),
    )
    assert _period_from_text("2027-04-01 – 2027-06-30") == (
        date(2027, 4, 1),
        date(2027, 6, 30),
    )


def test_rulebook_watcher_rejects_non_substring_quotes() -> None:
    item = FeedItem(
        source="rbi",
        guid="synthetic-feed-item",
        title="Official overseas transfer notice",
        url="https://www.rbi.org.in/synthetic-notice",
        published_at=datetime(2026, 10, 30, tzinfo=UTC),
        summary="The notice discusses transfers without changing a rate.",
    )

    proposals = build_rulebook_proposals(
        [item],
        [
            FeedFinding(
                topic="lrs_tcs",
                quote="New overseas transfer collection is 20%.",
            )
        ],
    )

    assert proposals == []


def test_rule_admin_updates_only_a_temporary_rules_copy(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    shutil.copytree(_ROOT / "rules", rules_dir)
    proposal = {
        "kind": "small_savings_rate",
        "path": "rules/schemes/small_savings.yaml#/schemes/ppf/periods",
        "proposed_value": {
            "from": "2027-04-01",
            "to": "2027-06-30",
            "rate_pct": "7.2",
            "source_url": "https://www.indiapost.gov.in/savings",
        },
        "evidence": {"quote": "Public Provident Fund pays 7.2%."},
    }

    target = apply_proposal_to_rules(proposal, rules_dir)
    updated = __import__("yaml").safe_load(target.read_text(encoding="utf-8"))

    assert updated["schemes"]["ppf"]["periods"][-1]["verified"] is True
    assert updated["schemes"]["ppf"]["periods"][-1]["rate_pct"] == "7.2"


def test_cio_agenda_counts_and_spend_stay_within_research_rules() -> None:
    rules = ResearchRules.load()
    context = RunContext(budget=RunBudget(max_usd=Decimal("5"), max_calls=20))
    router = FakeRouter()
    router.register(
        "mini", {"focus": ["Review the failed assumptions."], "note": "Use saved evidence."}
    )

    agenda = build_cio_agenda(
        week="2026-W44",
        rules=rules,
        run_ctx=context,
        router=router,
        scorekeeper_checklist=["Review weak assumptions."],
        regime_notes=["Recent official data was mixed."],
    )

    limits = agenda["market_limits"]
    assert sum(item["max_dossiers"] for item in limits.values()) <= rules.max_dossiers_per_run
    assert all(item["max_ideas"] <= rules.max_ideas_per_market for item in limits.values())
    assert all(Decimal(item["budget_usd"]) <= Decimal("5") for item in limits.values())
    assert agenda["focus"] == ["Review the failed assumptions."]


def test_core_picks_are_plain_direct_growth_trackers_with_a_measured_metric() -> None:
    from wealth.research.etf_selector import (
        CandidateMetrics,
        eligible_core_fund,
        has_measured_metric,
    )

    def fund(
        name: str, *, kind: str = "mf", plan: str | None = "direct", option: str | None = "growth"
    ) -> Security:
        return Security(
            market="IN",
            symbol="SYN",
            name=name,
            security_type=kind,
            currency="INR",
            plan_type=plan,
            option=option,
        )

    assert eligible_core_fund(fund("Synthetic Nifty 50 Index Fund"))
    assert eligible_core_fund(fund("Synthetic Nifty 50 ETF", kind="etf", plan=None, option=None))
    for name in (
        "Synthetic ELSS Tax Saver Nifty 50 Index Fund",
        "Synthetic Nifty 50 Equal Weight Index Fund",
        "Synthetic S&P 500 Top 50 ETF Fund of Fund",
        "Synthetic Global Equity Alpha Fund of Fund",
    ):
        assert not eligible_core_fund(fund(name)), name
    assert not eligible_core_fund(fund("Synthetic Nifty 50 Index Fund", plan="regular"))
    assert not eligible_core_fund(fund("Synthetic Nifty 50 Index Fund", option="idcw"))

    nothing = CandidateMetrics(None, None, None, None, None)
    assert not has_measured_metric(nothing)
    assert has_measured_metric(CandidateMetrics(Decimal("50000000"), None, None, None, None))
