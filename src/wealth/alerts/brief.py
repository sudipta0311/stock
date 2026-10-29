"""Deterministic weekly brief assembly and email rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict, Field

from wealth.alerts.copy import format_reason, load_copy
from wealth.core.money import format_inr
from wealth.policy.personalise import WeeklyPlan


class BriefAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    kind: str
    title: str
    amount: str | None
    why: str
    deeplink: str
    member_name: str = "You"
    member_relationship: str = "self"


class Brief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    week: str
    as_of: date
    status: Literal["ACTIONS", "NONE"]
    plan_health: int = Field(ge=0, le=100)
    subject: str
    preheader: str
    household_header: str | None = None
    actions: list[BriefAction] = Field(max_length=3)
    sections: dict[str, Any]
    no_llm_numbers: Literal[True] = True
    disclaimer: str
    why_received: str
    manage_alerts_url: str
    unsubscribe_url: str
    postal_address: str


def _document(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value or {})


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return Decimal(value)
    return None


def _compact_amount(value: Any) -> str | None:
    amount = _decimal(value)
    return None if amount is None else format_inr(amount, style="compact")


def _dkk_amount(value: Any) -> str | None:
    amount = _decimal(value)
    if amount is None:
        return None
    whole = amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"DKK {whole:,.0f}"


def _date_label(value: Any) -> str:
    if isinstance(value, date):
        selected = value
    elif isinstance(value, str):
        try:
            selected = date.fromisoformat(value)
        except ValueError:
            return value
    else:
        return ""
    return f"{selected.day} {selected.strftime('%b')}"


def _plan_document(plan: WeeklyPlan | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, WeeklyPlan):
        return plan.model_dump(mode="json")
    return WeeklyPlan.model_validate(plan).model_dump(mode="json")


def _user_value(user: Any, key: str, default: str) -> str:
    value = user.get(key) if isinstance(user, Mapping) else getattr(user, key, None)
    return str(value) if value else default


def personalize_plan_for_viewer(
    plan: Mapping[str, Any],
    members: Mapping[str, tuple[str, str]],
    *,
    me_member_id: str | None,
    owner_self_member_id: str | None,
    owner_name: str,
) -> dict[str, Any]:
    """Render each member reference from the logged-in person's point of view."""
    result = dict(plan)
    actions: list[dict[str, Any]] = []
    for raw_action in plan.get("actions", []):
        action = dict(raw_action)
        member_id = str(action.get("member_id") or "")
        member = members.get(member_id)
        if member is not None:
            if member_id == me_member_id:
                action["member_name"] = "You"
                action["member_relationship"] = "self"
            elif member_id == owner_self_member_id:
                action["member_name"] = owner_name
                action["member_relationship"] = "owner"
            else:
                action["member_name"], action["member_relationship"] = member
        actions.append(action)
    result["actions"] = actions
    return result


def _action_records(
    plan: Mapping[str, Any], user: Any, copy: Mapping[str, Any]
) -> list[BriefAction]:
    base_url = _user_value(user, "app_base_url", "https://app.example")
    labels = copy["labels"]
    action_labels = labels["action_kinds"]
    instrument_labels = labels["instrument_categories"]
    records: list[BriefAction] = []
    for rank, raw in enumerate(plan.get("actions", [])[:3], start=1):
        action = dict(raw)
        kind = str(action.get("kind", "ACTION"))
        title = str(action_labels.get(kind, "Review this part of your plan"))
        category = action.get("instrument_category")
        params = action.get("params") if isinstance(action.get("params"), Mapping) else {}
        if kind == "BUY_ABROAD":
            lines = params.get("abroad_buy_list", [])
            first_line = lines[0] if isinstance(lines, list) and lines else {}
            title = (
                f"{title} · {_dkk_amount(first_line.get('amount_dkk')) or 'DKK amount unavailable'}"
            )
        elif kind in {"SELL_EXIT", "SELL_TRIM", "REVIEW_EXIT", "SWAP"}:
            symbol = action.get("symbol")
            if symbol:
                title = f"{title} · {symbol}"
            quantity = _decimal(action.get("quantity"))
            if quantity is not None:
                title = f"{title} · {format(quantity.normalize(), 'f')} shares"
        elif category:
            title = f"{title} · {instrument_labels.get(category, 'Investment option')}"
        amount = None if kind == "BUY_ABROAD" else _compact_amount(action.get("amount"))
        if amount:
            title = f"{title} · {amount}"
        code = str(action.get("reason_code", ""))
        why = (
            format_reason(code, action.get("params", {}))
            if code
            else "This follows the rules in your plan."
        )
        clubbing_check = action.get("clubbing_check")
        if isinstance(clubbing_check, str) and clubbing_check:
            why = f"{why} {clubbing_check}"
        member_name = str(action.get("member_name") or "You")
        title = f"{member_name}: {title}"
        records.append(
            BriefAction(
                rank=int(action.get("rank", rank)),
                kind=kind,
                title=title,
                amount=amount,
                why=why,
                deeplink=str(action.get("deeplink") or f"{base_url.rstrip('/')}/plan"),
                member_name=member_name,
                member_relationship=str(action.get("member_relationship") or "self"),
            )
        )
    return records


def _allocation_rows(plan: Mapping[str, Any], copy: Mapping[str, Any]) -> list[dict[str, str]]:
    labels = copy["email"]["allocation_labels"]
    rows = []
    for sleeve, values in plan.get("allocation", {}).items():
        residency_status = plan.get("residency", {}).get("status", "RI")
        if sleeve == "equity_global" and residency_status != "RI":
            label = labels.get("equity_global_nri", "Shares where you live / global")
        else:
            label = labels.get(sleeve, sleeve.replace("_", " ").title())
        rows.append(
            {
                "sleeve": str(label),
                "target": f"{values.get('target_pct', '0')}%",
                "actual": f"{values.get('current_pct', '0')}%",
            }
        )
    return rows


def _calendar_events(calendar: Any, as_of: date) -> list[dict[str, Any]]:
    if isinstance(calendar, Mapping):
        events = calendar.get("maturity_calendar", [])
    else:
        events = calendar or []
    end = as_of + timedelta(days=30)
    selected: list[dict[str, Any]] = []
    for item in events:
        event = _document(item)
        try:
            event_date = date.fromisoformat(str(event.get("date")))
        except ValueError:
            continue
        if as_of <= event_date <= end:
            event["date_label"] = _date_label(event_date)
            event["amount_display"] = _compact_amount(event.get("amount"))
            selected.append(event)
    return sorted(selected, key=lambda item: (item["date"], str(item.get("label", ""))))


def _exit_watch(plan: Mapping[str, Any], labels: Mapping[str, Any]) -> list[dict[str, Any]]:
    watch: list[dict[str, Any]] = []
    for card in plan.get("exit_cards", []):
        triggers = card.get("triggers", {})
        active = [
            {
                "label": labels["trigger_keys"].get(key, "Holding check"),
                "status_label": labels["trigger_states"].get(
                    item.get("status"), labels["trigger_states"]["unknown"]
                ),
                "value": item.get("value"),
            }
            for key, item in triggers.items()
            if item.get("status") in {"watch", "triggered"}
        ]
        if active:
            watch.append(
                {
                    "symbol": card.get("symbol", "Holding"),
                    "action_label": labels["exit_actions"].get(
                        card.get("action"), "Review holding"
                    ),
                    "triggers": active,
                }
            )
    return watch


def _stress_section(
    plan: Mapping[str, Any], email_copy: Mapping[str, Any]
) -> dict[str, Any] | None:
    scenario = plan.get("stress", {}).get("india_equity_minus_25")
    if not scenario:
        return None
    difference = scenario.get("difference", {})
    current_impact = _decimal(difference.get("current_total"))
    target_impact = _decimal(difference.get("target_total"))
    if current_impact is None or target_impact is None:
        return None
    current = _compact_amount(abs(current_impact)) or "₹0"
    target = _compact_amount(abs(target_impact)) or "₹0"
    if target_impact < current_impact:
        headline = email_copy["india_stress_more_risk"].format(current=current, target=target)
    elif target_impact > current_impact:
        headline = email_copy["india_stress_less_risk"].format(
            current=current,
            less=_compact_amount(target_impact - current_impact) or "₹0",
        )
    else:
        headline = email_copy["india_stress_equal_risk"].format(current=current)
    return {
        "headline": headline,
        "tilt_notes": [
            note
            for note in plan.get("notes", [])
            if str(note).startswith(("geo_tilt:", "geopolitical_tilt:"))
        ],
    }


def build_brief(
    plan: WeeklyPlan | Mapping[str, Any],
    networth: Mapping[str, Any] | BaseModel | None,
    calendar: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    track_record: Any | None,
    user: Any,
    *,
    deployment_mode: Literal["personal", "public"] | None = None,
) -> Brief:
    """Build a calm, fully deterministic email brief from policy and Money Map data."""
    del track_record  # Scorekeeper is a later card; do not show an empty score section.
    data = _plan_document(plan)
    copy = load_copy()
    email_copy = copy["email"]
    as_of = (
        data["as_of"] if isinstance(data["as_of"], date) else date.fromisoformat(str(data["as_of"]))
    )
    actions = _action_records(data, user, copy)
    status: Literal["ACTIONS", "NONE"] = "ACTIONS" if actions else "NONE"
    score = int(data.get("plan_health", 100))
    subject = (
        email_copy["subject_actions"].format(count=len(actions), score=score)
        if actions
        else email_copy["subject_none"]
    )
    preheader = (
        email_copy["preheader_actions"].format(count=len(actions))
        if actions
        else email_copy["preheader_none"]
    )

    totals = _document(networth)
    net_worth = totals.get("net_worth")
    if net_worth is None and totals.get("assets_total") is not None:
        assets = _decimal(totals.get("assets_total")) or Decimal(0)
        liabilities = _decimal(totals.get("liabilities_total")) or Decimal(0)
        net_worth = assets - liabilities
    number_section = {
        "net_worth": _compact_amount(net_worth),
        "text": (
            email_copy["net_worth"].format(amount=_compact_amount(net_worth))
            if net_worth is not None
            else "Net worth is not available yet."
        ),
    }
    events = _calendar_events(calendar, as_of)
    exit_watch = _exit_watch(data, copy["labels"])
    stress = _stress_section(data, email_copy)
    sections: dict[str, Any] = {
        "number": number_section,
        "actions": [action.model_dump(mode="json") for action in actions],
        "plan_at_a_glance": _allocation_rows(data, copy),
        "coming_up": events,
        "exit_watch": exit_watch,
        "world_and_money": stress,
    }
    denmark_copy = copy["denmark"]
    abroad_lines = data.get("abroad_buy_list", [])
    if isinstance(abroad_lines, list) and abroad_lines:
        broker_copy = denmark_copy.get("brokers", {})
        how_to_copy = denmark_copy.get("how_to", {})
        account_copy = {
            "ask": denmark_copy["account_ask"],
            "ordinary": denmark_copy["account_ordinary"],
            "pension": denmark_copy["account_pension"],
        }
        sections["abroad_buy_list"] = [
            {
                "name": line.get("instrument", {}).get("name", "European fund"),
                "fund_code_label": denmark_copy["fund_code_label"],
                "fund_code": line.get("instrument", {}).get("isin", ""),
                "amount": _dkk_amount(line.get("amount_dkk")) or "",
                "account": account_copy.get(line.get("account"), ""),
                "broker": broker_copy.get(line.get("broker"), ""),
                "how_to": how_to_copy.get(line.get("broker"), []),
            }
            for line in abroad_lines[:3]
            if isinstance(line, Mapping) and isinstance(line.get("instrument"), Mapping)
        ]
    members = data.get("members", [])
    if members:
        joint_note = copy.get("household_notes", {}).get("joint_asset_tax_owner")
        household_notes = (
            [joint_note]
            if isinstance(joint_note, str)
            and any("joint_asset_tax_owner" in member.get("notes", []) for member in members)
            else []
        )
        sections["household"] = {
            "heading": email_copy["household_heading"],
            "notes": household_notes,
            "members": [
                {
                    "name": str(member.get("name") or "Household member"),
                    "net_worth": _compact_amount(member.get("net_worth")) or "₹0",
                    "text": email_copy["member_net_worth"].format(
                        name=str(member.get("name") or "Household member"),
                        amount=_compact_amount(member.get("net_worth")) or "₹0",
                    ),
                }
                for member in members
            ],
        }
    if data.get("residency", {}).get("status", "RI") != "RI":
        note_copy = copy.get("residency_notes", {})
        codes = data.get("residency_notes", [])[:2]
        sections["for_nris"] = {
            "heading": email_copy["for_nris_heading"],
            "notes": [note_copy[code] for code in codes if code in note_copy],
        }
    if not actions:
        sections["actions_empty_state"] = email_copy["no_actions"]
    selected_deployment_mode = deployment_mode or data.get("deployment_mode", "personal")
    if selected_deployment_mode not in {"personal", "public"}:
        raise ValueError("deployment_mode must be personal or public")
    plan_disclaimer = data.get("disclaimer") if selected_deployment_mode == "public" else None
    if selected_deployment_mode == "personal":
        personal_copy = copy.get("disclaimers", {}).get("personal")
        if isinstance(personal_copy, str):
            plan_disclaimer = personal_copy

    return Brief(
        week=data["week"],
        as_of=as_of,
        status=status,
        plan_health=score,
        subject=subject,
        preheader=preheader,
        household_header=_user_value(user, "household_header", "") or None,
        actions=actions,
        sections=sections,
        disclaimer=plan_disclaimer or _user_value(user, "disclaimer", email_copy["disclaimer"]),
        why_received=_user_value(user, "why_received", email_copy["why_received"]),
        manage_alerts_url=_user_value(
            user, "manage_alerts_url", "https://app.example/settings/alerts"
        ),
        unsubscribe_url=_user_value(user, "unsubscribe_url", "https://app.example/unsubscribe"),
        postal_address=_user_value(user, "postal_address", "Postal address provided in settings"),
    )


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_ENV = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=True)
_TEXT_ENV = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)


def render_html(brief: Brief) -> str:
    """Render a table-based HTML email with autoescaping enabled."""
    copy = load_copy()
    return _HTML_ENV.get_template("brief.html.j2").render(
        brief=brief,
        email=copy["email"],
        denmark=copy["denmark"],
        date_label=f"{brief.as_of.day} {brief.as_of.strftime('%b %Y')}",
    )


def render_text(brief: Brief) -> str:
    """Render the plain-text alternative with the same brief facts."""
    copy = load_copy()
    return (
        _TEXT_ENV.get_template("brief.txt.j2")
        .render(
            brief=brief,
            email=copy["email"],
            denmark=copy["denmark"],
            date_label=f"{brief.as_of.day} {brief.as_of.strftime('%b %Y')}",
        )
        .strip()
        + "\n"
    )
