"""Copy coverage, deterministic content, and email snapshot tests."""

from __future__ import annotations

import ast
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

import yaml

from wealth.alerts.brief import Brief, build_brief, render_html, render_text
from wealth.alerts.consent import create_unsubscribe_token
from wealth.alerts.copy import load_copy
from wealth.alerts.email import rendered_body_hash
from wealth.core.plain_language import find_banned_terms

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_PLAN = _ROOT / "tests" / "golden" / "weekly_plan_user1.json"
_SNAPSHOTS = Path(__file__).resolve().parent / "golden"
_SYNTHETIC_USER_ID = "00000000-0000-0000-0000-000000000017"


def _brief(actions: bool) -> Brief:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    if not actions:
        plan["actions"] = []
        plan["exit_cards"] = []
    return _brief_from_plan(plan)


def _brief_from_plan(
    plan: dict[str, Any], *, deployment_mode: Literal["personal", "public"] | None = None
) -> Brief:
    return build_brief(
        plan,
        {"net_worth": "1942400.00"},
        [
            {
                "date": "2026-11-14",
                "asset_id": "00000000-0000-0000-0000-000000000111",
                "kind": "maturity",
                "amount": "250000.00",
                "label": "Synthetic FD maturity",
            }
        ],
        None,
        {
            "app_base_url": "https://app.example",
            "manage_alerts_url": "https://app.example/settings/alerts",
            "unsubscribe_url": (
                "https://app.example/v1/email/unsubscribe?token="
                + create_unsubscribe_token(
                    _SYNTHETIC_USER_ID,
                    "weekly_email",
                    "synthetic-test-secret",
                    now=datetime(2026, 10, 28, tzinfo=UTC),
                )
            ),
            "disclaimer": "Research and education only, not personalised investment advice.",
            "why_received": "You received this because you opted in to weekly email.",
            "postal_address": "Synthetic postal address",
        },
        deployment_mode=deployment_mode,
    )


def _brief_with_stress(current: str, target: str) -> Brief:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["actions"] = []
    plan["exit_cards"] = []
    difference = Decimal(target) - Decimal(current)
    plan["stress"]["india_equity_minus_25"]["difference"].update(
        {"current_total": current, "target_total": target, "difference": format(difference, "f")}
    )
    return _brief_from_plan(plan)


def _reason_codes_from_source() -> set[str]:
    codes: set[str] = set()

    def add_string_constants(node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", node.value):
                codes.add(node.value)
        elif isinstance(node, ast.IfExp):
            add_string_constants(node.body)
            add_string_constants(node.orelse)
        elif not isinstance(node, ast.Compare):
            for child in ast.iter_child_nodes(node):
                add_string_constants(child)

    policy_root = _ROOT / "src" / "wealth" / "policy"
    for path in policy_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg
                in {
                    "reason_code",
                    "reason_codes",
                    "message_key",
                }
                or isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"reasons", "reason_codes"}
                    for target in node.targets
                )
            ):
                add_string_constants(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"reasons", "reason_codes"}
                and node.args
            ):
                add_string_constants(node.args[0])

    def visit_rules(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in {"reason_code", "reason_codes", "message_key"}:
                    visit_rules(child, str(key))
                else:
                    visit_rules(child, str(key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit_rules(child, parent_key)
        elif parent_key in {"reason_code", "reason_codes", "message_key"} and isinstance(
            value, str
        ):
            codes.add(value)

    rules_root = _ROOT / "rules"
    for path in rules_root.rglob("*.yaml"):
        visit_rules(yaml.safe_load(path.read_text(encoding="utf-8")))
    return codes


def test_every_policy_reason_code_and_message_has_shared_copy() -> None:
    copy = load_copy()
    covered = set(copy["reasons"]) | set(copy["messages"])
    missing = sorted(_reason_codes_from_source() - covered)
    assert missing == []


def test_actions_and_none_html_and_text_match_golden_snapshots(request) -> None:
    for status, brief in (("actions", _brief(True)), ("none", _brief(False))):
        html = render_html(brief)
        text = render_text(brief)
        html_path = _SNAPSHOTS / f"brief_{status}.html"
        text_path = _SNAPSHOTS / f"brief_{status}.txt"
        if request.config.getoption("--update-golden"):
            html_path.write_text(html, encoding="utf-8")
            text_path.write_text(text, encoding="utf-8")
        assert html == html_path.read_text("utf-8")
        assert text == text_path.read_text("utf-8")


def test_brief_builder_loads_legacy_ri_plan_with_null_country() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["residency"]["country"] = None

    brief = _brief_from_plan(plan)

    assert brief.status == "ACTIONS"
    assert brief == _brief(True)


def test_sell_action_alone_makes_brief_status_actions() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    sell_action = next(action for action in plan["actions"] if action["kind"] == "SELL_TRIM")
    plan["actions"] = [sell_action]

    assert _brief_from_plan(plan).status == "ACTIONS"
    assert _brief(False).status == "NONE"


def test_nri_brief_shows_two_localized_residency_notes_and_local_sleeve_name() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["residency"] = {"status": "NRI", "country": "DK", "us_person": False}
    plan["residency_notes"] = ["dk_share_income", "dk_abis_check", "dk_worldwide_income"]
    plan["disclaimer"] = (
        "Information and education only. Not investment advice under MiFID II; "
        "we are not authorised by Finanstilsynet."
    )
    plan["deployment_mode"] = "public"

    brief = _brief_from_plan(plan)

    assert brief.sections["for_nris"]["heading"] == "For NRIs"
    assert len(brief.sections["for_nris"]["notes"]) == 2
    assert "27% up to DKK 79,400" in brief.sections["for_nris"]["notes"][0]
    global_row = next(
        row
        for row in brief.sections["plan_at_a_glance"]
        if row["sleeve"] == "Shares where you live / global"
    )
    assert global_row["target"].endswith("%")
    assert brief.disclaimer == plan["disclaimer"]


def test_brief_uses_the_deployment_mode_supplied_by_alert_settings() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["disclaimer"] = "Jurisdiction-specific public copy."

    brief = _brief_from_plan(plan, deployment_mode="public")

    assert brief.disclaimer == "Jurisdiction-specific public copy."


def test_household_brief_attributes_actions_and_net_worth_to_members() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["members"] = [
        {
            "id": "synthetic-self",
            "name": "Self",
            "relationship": "self",
            "net_worth": "1000000.00",
            "notes": [],
        },
        {
            "id": "synthetic-spouse",
            "name": "Spouse",
            "relationship": "spouse",
            "net_worth": "500000.00",
            "notes": [],
        },
    ]
    plan["actions"][0].update(
        {
            "member_name": "Spouse",
            "member_relationship": "spouse",
            "clubbing_check": "Check the source of funds.",
        }
    )

    brief = _brief_from_plan(plan)

    assert brief.actions[0].title.startswith("Spouse:")
    assert "Check the source of funds." in brief.actions[0].why
    assert [member["text"] for member in brief.sections["household"]["members"]] == [
        "Self: net worth ₹10 L",
        "Spouse: net worth ₹5 L",
    ]
    assert (
        brief.subject == f"{len(brief.actions)} actions this week · Plan health {brief.plan_health}"
    )


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def test_no_llm_numbers_every_rendered_number_is_in_brief_data() -> None:
    brief = _brief(True)
    assert brief.no_llm_numbers is True
    brief_json = json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    html_parser = _VisibleText()
    html_parser.feed(render_html(brief))
    visible_outputs = (" ".join(html_parser.parts), render_text(brief))
    for visible in visible_outputs:
        for number in re.findall(r"\d+(?:[,.]\d+)*", visible):
            assert number in brief_json


def test_rendered_briefs_show_no_raw_policy_codes() -> None:
    copy = load_copy()
    codes = set(copy["reasons"]) | set(copy["messages"])
    for group in copy["labels"].values():
        if isinstance(group, dict):
            codes.update(group)
    snake_case_codes = sorted(code for code in codes if "_" in code)
    code_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(code) for code in snake_case_codes)
        + r")(?![A-Za-z0-9_])"
    )
    uppercase_pattern = re.compile(r"\b[A-Z]{2,}_[A-Z_]+\b")

    for brief in (_brief(True), _brief(False)):
        for body in (render_html(brief), render_text(brief)):
            without_urls = re.sub(r"https?://[^\s\"'<>]+", "", body)
            assert uppercase_pattern.search(without_urls) is None
            assert code_pattern.search(without_urls) is None


def test_india_stress_sentence_explains_when_target_loses_more() -> None:
    headline = _brief_with_stress("-2250.00", "-157745.50").sections["world_and_money"]["headline"]

    assert "If Indian shares fall 25%" in headline
    assert "today's money would drop about ₹2,250" in headline
    assert "Your planned mix would drop ₹1.58 L, because it has more Indian shares." in headline
    assert "difference" not in headline
    assert "-₹" not in headline


def test_india_stress_sentence_says_when_target_loses_less() -> None:
    headline = _brief_with_stress("-2500.00", "-1000.00").sections["world_and_money"]["headline"]

    assert "Your planned mix would lose ₹1,500 less." in headline
    assert "difference" not in headline


def test_rendered_html_body_hash_is_stable() -> None:
    body = render_html(_brief(True))
    assert rendered_body_hash(body) == rendered_body_hash(body)


def test_brief_omits_track_record_until_scorekeeper_exists() -> None:
    brief = _brief(True)
    assert "track_record" not in brief.sections
    assert brief.status == "ACTIONS"


def test_danish_abroad_buy_list_renders_dkk_account_and_broker_steps() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["residency"] = {"status": "NRI", "country": "DK", "us_person": False}
    plan["abroad_buy_list"] = [
        {
            "member_id": _SYNTHETIC_USER_ID,
            "account": "ask",
            "broker": "nordnet",
            "instrument": {
                "name": "European world fund",
                "ticker": "WLDX",
                "exchange": "Xetra",
                "isin": "IE00ABC00001",
                "currency": "EUR",
            },
            "amount_dkk": "10000",
            "amount_inr": "125000.00",
            "reason_codes": ["ask_first"],
            "how_to_key": "how_to.nordnet",
            "details": {"tax_verified": False},
        }
    ]
    plan["actions"] = [
        {
            "rank": 1,
            "kind": "BUY_ABROAD",
            "sleeve": "equity_global",
            "instrument_category": "abroad_etf",
            "amount": "125000.00",
            "reason_code": "buy_abroad",
            "params": {
                "broker": "nordnet",
                "account_type": "ASK",
                "abroad_buy_list": plan["abroad_buy_list"],
            },
            "member_id": _SYNTHETIC_USER_ID,
            "member_name": "You",
            "member_relationship": "self",
        }
    ]

    brief = _brief_from_plan(plan)
    html = render_html(brief)
    text = render_text(brief)

    assert "DKK 10,000" in html
    assert "Share savings account" in html
    assert "Nordnet" in html
    assert "Search the fund code." in html
    assert "IE00ABC00001" in text


def test_nri_household_email_has_no_banned_terms() -> None:
    plan = json.loads(_GOLDEN_PLAN.read_text(encoding="utf-8"))
    plan["residency"] = {"status": "NRI", "country": "DK", "us_person": False}
    plan["residency_notes"] = ["dk_share_income", "dk_worldwide_income"]
    plan["members"] = [
        {
            "id": "synthetic-self",
            "name": "Self",
            "relationship": "self",
            "net_worth": "1000000.00",
            "residency": {"status": "NRI", "country": "DK", "us_person": False},
        },
        {
            "id": "synthetic-partner",
            "name": "Partner",
            "relationship": "spouse",
            "net_worth": "500000.00",
            "residency": {"status": "RI", "country": "IN", "us_person": False},
        },
    ]
    brief = _brief_from_plan(plan)
    html = render_html(brief)
    text = render_text(brief)
    parser = _VisibleText()
    parser.feed(html)

    assert find_banned_terms(" ".join(parser.parts)) == []
    assert find_banned_terms(text) == []
