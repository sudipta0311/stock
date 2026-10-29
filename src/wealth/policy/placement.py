"""Pure, deterministic placement of household actions by member and tax fit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from wealth.core.money import round_paise
from wealth.policy.actions import Action
from wealth.policy.debt_picker import DebtNeed, DebtPick, pick_debt_with_unranked
from wealth.policy.household import MemberContext
from wealth.policy.residency import (
    eligibility_rule,
    india_core_vehicle,
    load_residency_rules,
)
from wealth.policy.routes import RouteContext, RouteDecision, select_route
from wealth.schemas.common import LiquidityClass


@dataclass(frozen=True)
class PlacementResult:
    """Actions assigned to their primary acting member plus explanatory notes."""

    actions: tuple[Action, ...]
    notes: tuple[str, ...]
    carry_forward: Decimal = Decimal("0.00")


def place_actions(
    actions: Sequence[Action],
    members: Sequence[MemberContext],
    *,
    debt_yields: Mapping[str, Decimal],
    fx_rates: Mapping[str, Decimal],
    amc_international_suspended: bool = False,
    etf_inav_premium_pct: Decimal | None = None,
    us_situs_holdings_usd: Decimal = Decimal(0),
    as_of_year: int | None = None,
) -> PlacementResult:
    """Assign pooled household recommendations using each owner's rules and room."""
    if not members:
        return PlacementResult(tuple(actions), ())
    available = {member.member_id: member.new_money for member in members}
    notes: list[str] = []
    placed: list[Action] = []
    carry_forward = Decimal("0.00")
    selected_year = as_of_year or date.today().year

    for action in actions:
        if action.kind.startswith("SELL_") or action.kind in {"REVIEW_EXIT", "SWAP"}:
            placed.append(action)
            continue
        if action.kind == "PAY_EXTRA_ON_LOAN":
            member = next((item for item in members if item.member_id == action.member_id), None)
            if member is None:
                member = _preferred(members, available)[0]
            _spend_household(available, member.member_id, action.amount, members)
            placed.append(_assign(action, member))
            continue

        if action.kind == "PAY_DOWN_DEBT":
            candidates = sorted(
                members,
                key=lambda member: (
                    -member.snapshot.liabilities_total,
                    -available.get(member.member_id, Decimal(0)),
                    not member.member.is_self,
                    member.member_id,
                ),
            )
            member = candidates[0]
            _spend_household(available, member.member_id, action.amount, members)
            placed.append(_assign(action, member))
            continue

        if action.kind == "REBALANCE_SIPS":
            chunks, chunk_notes = _place_sip_redirection(
                action,
                members,
                debt_yields=debt_yields,
                fx_rates=fx_rates,
                amc_international_suspended=amc_international_suspended,
                etf_inav_premium_pct=etf_inav_premium_pct,
                us_situs_holdings_usd=us_situs_holdings_usd,
                as_of_year=selected_year,
            )
            placed.extend(chunks)
            notes.extend(chunk_notes)
            continue

        if action.sleeve == "equity_india":
            chunks, chunk_notes, unplaced = _place_india_equity(action, members, available)
        elif action.sleeve == "equity_global":
            chunks, chunk_notes, unplaced = _place_global_equity(
                action,
                members,
                available,
                fx_rates=fx_rates,
                amc_international_suspended=amc_international_suspended,
                etf_inav_premium_pct=etf_inav_premium_pct,
                us_situs_holdings_usd=us_situs_holdings_usd,
            )
        elif action.sleeve == "debt" and action.kind in {
            "BUY_LUMPSUM",
            "BUY_STP",
            "BUILD_EMERGENCY_FUND",
        }:
            chunks, chunk_notes, unplaced = _place_debt(
                action,
                members,
                available,
                debt_yields=debt_yields,
                as_of_year=selected_year,
            )
        else:
            source = next(
                (member for member in members if member.member_id == action.member_id), None
            )
            dk_gold = (
                action.instrument_category == "gold_etf"
                and source is not None
                and source.residency.country == "DK"
            )
            eligible_members = [
                member
                for member in members
                if (not dk_gold or member.residency.status == "RI")
                and _instrument_eligibility(member, action.instrument_category)
                in {"allowed", "conditional"}
            ]
            if not eligible_members:
                if dk_gold:
                    notes.extend(("dk_fund_tax_review", "dk_gold_carried_forward"))
                else:
                    notes.append(f"no_eligible_member:{action.instrument_category}")
                carry_forward = round_paise(carry_forward + action.amount)
                continue
            member = _preferred(eligible_members, available)[0]
            _spend_household(available, member.member_id, action.amount, members)
            chunks, chunk_notes, unplaced = [_assign(action, member)], [], Decimal(0)

        placed.extend(chunks)
        notes.extend(chunk_notes)
        carry_forward = round_paise(carry_forward + unplaced)

    return PlacementResult(
        tuple(_merge_member_sip_actions(placed)), tuple(dict.fromkeys(notes)), carry_forward
    )


def _merge_member_sip_actions(actions: Sequence[Action]) -> list[Action]:
    """Keep at most one SIP redirection action per member, combining same destinations."""
    merged: list[Action] = []
    positions: dict[str, int] = {}
    for action in actions:
        if action.kind != "REBALANCE_SIPS":
            merged.append(action)
            continue
        position = positions.get(action.member_id)
        if position is None:
            positions[action.member_id] = len(merged)
            merged.append(action)
            continue
        previous = merged[position]
        changes_by_destination: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}
        for change in [
            *previous.params.get("changes", []),
            *action.params.get("changes", []),
        ]:
            if not isinstance(change, dict):
                continue
            destination = tuple(
                sorted(
                    (key, str(value)) for key, value in change.items() if key not in {"from", "to"}
                )
            )
            current = changes_by_destination.get(destination)
            if current is None:
                current = {key: value for key, value in change.items() if key not in {"from", "to"}}
                current["from"] = Decimal(0)
                current["to"] = Decimal(0)
                changes_by_destination[destination] = current
            current["from"] = Decimal(str(current["from"])) + Decimal(str(change.get("from", "0")))
            current["to"] = Decimal(str(current["to"])) + Decimal(str(change.get("to", "0")))
        merged_changes = [
            {
                **change,
                "from": format(round_paise(Decimal(str(change["from"]))), ".2f"),
                "to": format(round_paise(Decimal(str(change["to"]))), ".2f"),
            }
            for change in changes_by_destination.values()
        ]
        merged[position] = previous.model_copy(
            update={
                "amount": round_paise(previous.amount + action.amount),
                "clubbing_check": previous.clubbing_check or action.clubbing_check,
                "params": {**previous.params, "changes": merged_changes},
            }
        )
    return merged


def _preferred(
    members: Sequence[MemberContext], available: Mapping[str, Decimal]
) -> list[MemberContext]:
    return sorted(
        members,
        key=lambda member: (
            available.get(member.member_id, Decimal(0)) <= 0,
            -available.get(member.member_id, Decimal(0)),
            not member.member.is_self,
            member.member_id,
        ),
    )


def _assign(
    action: Action,
    member: MemberContext,
    *,
    instrument: str | None = None,
    route: str | None = None,
    params: Mapping[str, object] | None = None,
    clubbing_check: str | None = None,
    amount: Decimal | None = None,
) -> Action:
    updated_params = dict(action.params)
    if params:
        updated_params.update(params)
    changes: dict[str, object] = {
        "member_id": member.member_id,
        "member_name": member.name,
        "member_relationship": member.relationship,
        "clubbing_check": clubbing_check,
        "params": updated_params,
    }
    if instrument is not None:
        changes["instrument_category"] = instrument
    if route is not None:
        changes["route"] = route
    if amount is not None:
        changes["amount"] = round_paise(amount)
    return action.model_copy(update=changes)


def _spend_household(
    available: dict[str, Decimal],
    member_id: str,
    amount: Decimal,
    members: Sequence[MemberContext],
) -> bool:
    """Consume this member's money first, then other members' pooled cash."""
    remaining = round_paise(amount)
    own = min(remaining, available.get(member_id, Decimal(0)))
    available[member_id] = round_paise(available.get(member_id, Decimal(0)) - own)
    remaining = round_paise(remaining - own)
    used_other = remaining > 0
    donors = sorted(
        (item for item in members if item.member_id != member_id),
        key=lambda item: (-available.get(item.member_id, Decimal(0)), item.member_id),
    )
    for donor in donors:
        if remaining <= 0:
            break
        donated = min(remaining, available.get(donor.member_id, Decimal(0)))
        available[donor.member_id] = round_paise(
            available.get(donor.member_id, Decimal(0)) - donated
        )
        remaining = round_paise(remaining - donated)
    return used_other


def _clubbing_note_if_needed(
    member: MemberContext,
    used_other_money: bool,
) -> str | None:
    if member.relationship != "spouse" or not used_other_money:
        return None
    return str(load_residency_rules()["tax_notes"]["spouse_clubbing"]["copy"])


def _can_buy_india_core(member: MemberContext, vehicle: str) -> bool:
    instrument = "in_mf" if vehicle == "in_core_index" else "in_equity_direct"
    rule = eligibility_rule(instrument, member.residency)
    if rule is None:
        return False
    if member.residency.us_person and instrument == "in_mf":
        return False
    # Direct Indian equity may require a conditional PIS route for an NRI. It remains
    # available as an informational recommendation, with the rulebook note attached.
    if instrument == "in_equity_direct" and rule["new"] == "conditional":
        return True
    return rule["new"] == "allowed"


def _instrument_eligibility(member: MemberContext, instrument: str) -> str:
    aliases = {
        "in_core_index": "in_mf",
        "in_international_index_fund": "in_mf",
        "in_listed_international_etf": "in_etf",
        "in_equity_direct_core": "in_equity_direct",
        "ucits_etf": "ucits_global_etf",
        "us_etf": "ucits_global_etf",
        "us_broad_market_etf": "ucits_global_etf",
        "sweep_fd": "bank_fd",
        "po_td_1y": "po_td",
        "po_td_2y": "po_td",
        "po_td_3y": "po_td",
        "po_td_5y": "po_td",
    }
    key = aliases.get(instrument, instrument)
    return member.eligibility_status.get(key, "not_allowed")


def _place_india_equity(
    action: Action,
    members: Sequence[MemberContext],
    available: dict[str, Decimal],
    *,
    origin_member_id: str | None = None,
    consume_new_money: bool = True,
) -> tuple[list[Action], list[str], Decimal]:
    eligible = [
        (member, india_core_vehicle(member.residency))
        for member in members
        if _can_buy_india_core(member, india_core_vehicle(member.residency))
    ]
    if not eligible:
        return [], ["india_equity_no_eligible_member"], action.amount

    eligible.sort(
        key=lambda pair: (
            0 if pair[1] == "in_core_index" else 1,
            pair[0].member_id != origin_member_id if origin_member_id is not None else False,
            available.get(pair[0].member_id, Decimal(0)) <= 0,
            -available.get(pair[0].member_id, Decimal(0)),
            not pair[0].member.is_self,
            pair[0].member_id,
        )
    )
    member, vehicle = eligible[0]
    used_other = (
        _spend_household(available, member.member_id, action.amount, members)
        if consume_new_money
        else member.member_id != origin_member_id
    )
    clubbing = _clubbing_note_if_needed(member, used_other)
    notes = (
        [f"{member.name}: direct_equity_nri_pis"]
        if vehicle == "in_equity_direct_core" and (member.residency.status != "RI")
        else []
    )
    return [_assign(action, member, instrument=vehicle, clubbing_check=clubbing)], notes, Decimal(0)


def _route_for_member(
    member: MemberContext,
    fx_rates: Mapping[str, Decimal],
    *,
    amc_international_suspended: bool,
    etf_inav_premium_pct: Decimal | None,
    us_situs_holdings_usd: Decimal,
) -> RouteDecision:
    prefs = member.member.investing
    return select_route(
        RouteContext(
            monthly_global_new_money=prefs.monthly_sip_capacity,
            annual_global_new_money=round_paise(
                prefs.monthly_sip_capacity * 12 + prefs.new_money_this_week
            ),
            has_lrs_broker=prefs.has_lrs_broker,
            prefers_no_paperwork=prefs.prefers_no_paperwork,
            us_situs_holdings_usd=us_situs_holdings_usd,
            lrs_used_this_fy=prefs.lrs_used_this_fy,
            amc_international_suspended=amc_international_suspended,
            etf_inav_premium_pct=etf_inav_premium_pct,
            residency=member.residency,
        )
    )


def _lrs_room_inr(member: MemberContext, fx_rates: Mapping[str, Decimal]) -> Decimal:
    usd_inr = fx_rates.get("USD/INR")
    if usd_inr is None or usd_inr <= 0:
        return Decimal(0)
    return round_paise(member.limits_room.get("lrs_annual_usd", Decimal(0)) * usd_inr)


def _tcs_params(member: MemberContext, amount: Decimal) -> dict[str, object]:
    rules = load_residency_rules()["limits"]
    threshold = member.limits_room.get("lrs_tcs_threshold_inr", Decimal(0))
    rate = member.limits_room.get("lrs_tcs_pct", Decimal(0))
    used = member.member.investing.lrs_used_this_fy
    before = max(Decimal(0), used - threshold)
    after = max(Decimal(0), used + amount - threshold)
    tcs_base = round_paise(max(Decimal(0), after - before))
    return {
        "lrs_annual_limit_usd": str(rules["lrs_annual_usd"]),
        "lrs_room_usd": str(member.limits_room.get("lrs_annual_usd", Decimal(0))),
        "tcs_threshold_inr": str(threshold),
        "tcs_pct": str(rate),
        "estimated_tcs_inr": str(round_paise(tcs_base * rate / Decimal(100))),
    }


def _place_global_equity(
    action: Action,
    members: Sequence[MemberContext],
    available: dict[str, Decimal],
    *,
    fx_rates: Mapping[str, Decimal],
    amc_international_suspended: bool,
    etf_inav_premium_pct: Decimal | None,
    us_situs_holdings_usd: Decimal,
) -> tuple[list[Action], list[str], Decimal]:
    ordered = _preferred(members, available)
    chunks: list[Action] = []
    notes: list[str] = []
    remaining = action.amount
    for member in ordered:
        decision = _route_for_member(
            member,
            fx_rates,
            amc_international_suspended=amc_international_suspended,
            etf_inav_premium_pct=etf_inav_premium_pct,
            us_situs_holdings_usd=us_situs_holdings_usd,
        )
        if decision.vehicle == "no_open_route":
            continue
        if _instrument_eligibility(member, decision.vehicle) == "not_allowed":
            continue
        if member.residency.country == "DK":
            dkk_inr = fx_rates.get("DKK/INR")
            ask_room = member.limits_room.get("ask_deposit_dkk", Decimal(0))
            ask_room_unchecked = dkk_inr is None or dkk_inr <= 0
            ask_room_inr = round_paise(ask_room * dkk_inr) if not ask_room_unchecked else remaining
            ask_amount = min(remaining, ask_room_inr)
            ask_room_exceeded = not ask_room_unchecked and remaining > ask_room_inr
            if ask_room_unchecked:
                notes.append("ask_room_unchecked_fx_missing")
            if ask_amount > 0:
                used_other = _spend_household(available, member.member_id, ask_amount, members)
                clubbing = _clubbing_note_if_needed(member, used_other)
                chunks.append(
                    _assign(
                        action,
                        member,
                        instrument="ucits_global_etf",
                        route="LOCAL",
                        amount=ask_amount,
                        params={
                            "account_type": "ASK",
                            "broker": member.member.investing.broker,
                            "ask_tax_rate_pct": str(
                                load_residency_rules()["denmark"]["ask"]["tax_rate_pct_2026"]
                            ),
                        },
                        clubbing_check=clubbing,
                    )
                )
                remaining = round_paise(remaining - ask_amount)
            if remaining > 0:
                used_other = _spend_household(available, member.member_id, remaining, members)
                clubbing = _clubbing_note_if_needed(member, used_other)
                chunks.append(
                    _assign(
                        action,
                        member,
                        instrument="ucits_global_etf",
                        route="LOCAL",
                        amount=remaining,
                        params={
                            "account_type": "ORDINARY",
                            "broker": member.member.investing.broker,
                        },
                        clubbing_check=clubbing,
                    )
                )
                remaining = Decimal(0)
            if ask_room_exceeded:
                notes.append(f"{member.name}: global_equity_ask_room_spent_then_local_account")
            return chunks, notes, remaining

        chunk_amount = remaining
        params: dict[str, object] = {}
        if decision.route == "C" and member.residency.status == "RI":
            chunk_amount = min(remaining, _lrs_room_inr(member, fx_rates))
            if chunk_amount <= 0:
                continue
            params.update(_tcs_params(member, chunk_amount))
        used_other = _spend_household(available, member.member_id, chunk_amount, members)
        clubbing = _clubbing_note_if_needed(member, used_other)
        chunks.append(
            _assign(
                action,
                member,
                instrument=decision.vehicle,
                route=decision.route,
                amount=chunk_amount,
                params=params,
                clubbing_check=clubbing,
            )
        )
        remaining = round_paise(remaining - chunk_amount)
        notes.extend(f"{member.name}: {warning}" for warning in decision.warnings)
        if remaining <= 0:
            break

    if remaining > 0:
        if chunks:
            notes.append("global_equity_route_or_lrs_limit_room_exhausted")
        else:
            notes.append("global_equity_no_open_route")
    return chunks, notes, remaining


def _place_sip_redirection(
    action: Action,
    members: Sequence[MemberContext],
    *,
    debt_yields: Mapping[str, Decimal],
    fx_rates: Mapping[str, Decimal],
    amc_international_suspended: bool,
    etf_inav_premium_pct: Decimal | None,
    us_situs_holdings_usd: Decimal,
    as_of_year: int,
) -> tuple[list[Action], list[str]]:
    """Apply residency and account rules to each member's proposed SIP increases."""
    source = next((member for member in members if member.member_id == action.member_id), None)
    if source is None:
        source = next((member for member in members if member.member.is_self), members[0])
    raw_changes = action.params.get("changes")
    if not isinstance(raw_changes, list):
        return [], ["sip_redirection_changes_missing"]

    grouped: dict[str, dict[str, object]] = {}
    notes: list[str] = []

    def add_change(
        member: MemberContext,
        change: dict[str, object],
        amount: Decimal,
        *,
        funded_by_other: bool = False,
    ) -> None:
        group = grouped.setdefault(
            member.member_id,
            {"member": member, "changes": [], "amount": Decimal(0), "clubbing": False},
        )
        changes = group["changes"]
        assert isinstance(changes, list)
        changes.append(change)
        group["amount"] = round_paise(group["amount"] + amount)  # type: ignore[operator]
        group["clubbing"] = bool(group["clubbing"]) or (
            funded_by_other and member.relationship == "spouse"
        )

    for raw_change in raw_changes:
        if not isinstance(raw_change, dict):
            continue
        change = dict(raw_change)
        sleeve = str(change.get("sleeve", ""))
        try:
            before = Decimal(str(change.get("from", "0")))
            after = Decimal(str(change.get("to", "0")))
        except InvalidOperation:
            notes.append("sip_redirection_amount_invalid")
            continue
        increase = round_paise(after - before)
        if increase <= 0:
            add_change(source, change, Decimal(0))
            continue

        placements: list[tuple[MemberContext, Decimal, dict[str, object]]] = []
        if sleeve == "equity_india":
            proposal = action.model_copy(update={"sleeve": sleeve, "amount": increase})
            chunks, placement_notes, unplaced = _place_india_equity(
                proposal,
                members,
                {},
                origin_member_id=source.member_id,
                consume_new_money=False,
            )
            notes.extend(placement_notes)
            if chunks and unplaced == 0:
                chunk = chunks[0]
                member = next(item for item in members if item.member_id == chunk.member_id)
                placements.append(
                    (
                        member,
                        increase,
                        {"instrument_category": chunk.instrument_category},
                    )
                )
        elif sleeve == "equity_global":
            placements, placement_notes = _sip_global_assignments(
                source,
                increase,
                members,
                fx_rates=fx_rates,
                amc_international_suspended=amc_international_suspended,
                etf_inav_premium_pct=etf_inav_premium_pct,
                us_situs_holdings_usd=us_situs_holdings_usd,
            )
            notes.extend(placement_notes)
        elif sleeve == "gold":
            if source.residency.country == "DK":
                ri_members = [
                    member
                    for member in members
                    if member.residency.status == "RI"
                    and _instrument_eligibility(member, "gold_etf") in {"allowed", "conditional"}
                ]
                if ri_members:
                    recipient = _preferred(ri_members, {})[0]
                    placements.append((recipient, increase, {"instrument_category": "gold_etf"}))
                else:
                    notes.extend(("dk_fund_tax_review", "dk_gold_carried_forward"))
            elif _instrument_eligibility(source, "gold_etf") in {
                "allowed",
                "conditional",
            }:
                placements.append((source, increase, {"instrument_category": "gold_etf"}))
        elif sleeve in {"debt", "cash"}:
            debt_action = action.model_copy(
                update={
                    "sleeve": "debt",
                    "amount": increase,
                    "kind": "BUY_LUMPSUM",
                    "instrument_category": "bank_fd",
                }
            )
            for member, pick in _ranked_debt_candidates(
                debt_action,
                members,
                debt_yields,
                as_of_year,
            ):
                if _instrument_eligibility(member, pick.instrument) in {
                    "allowed",
                    "conditional",
                }:
                    placements.append(
                        (
                            member,
                            increase,
                            {"instrument_category": pick.instrument},
                        )
                    )
                    break

        if not placements:
            unchanged = dict(change)
            unchanged["to"] = format(before, ".2f")
            add_change(source, unchanged, Decimal(0))
            if sleeve != "gold" or source.residency.country != "DK":
                notes.append(f"sip_increase_unplaced:{sleeve or 'unknown'}")
            continue

        transfers_to_other_member = any(
            member.member_id != source.member_id for member, _amount, _metadata in placements
        )
        if transfers_to_other_member and before > 0:
            unchanged = dict(change)
            unchanged["to"] = format(before, ".2f")
            add_change(source, unchanged, Decimal(0))

        remaining_before = before
        for position, (member, amount, metadata) in enumerate(placements):
            placed_change: dict[str, object] = {
                "sleeve": sleeve,
                "from": format(
                    remaining_before
                    if position == 0 and member.member_id == source.member_id
                    else Decimal(0),
                    ".2f",
                ),
                "to": format(
                    remaining_before + amount
                    if position == 0 and member.member_id == source.member_id
                    else amount,
                    ".2f",
                ),
                **metadata,
            }
            add_change(
                member,
                placed_change,
                amount,
                funded_by_other=member.member_id != source.member_id,
            )
            if member.member_id == source.member_id and position == 0:
                remaining_before = Decimal(0)

    output: list[Action] = []
    for member in members:
        group = grouped.get(member.member_id)
        if group is None:
            continue
        member_changes = group["changes"]
        assert isinstance(member_changes, list)
        if not member_changes:
            continue
        amount = group["amount"]
        assert isinstance(amount, Decimal)
        clubbing_check = (
            str(load_residency_rules()["tax_notes"]["spouse_clubbing"]["copy"])
            if group["clubbing"]
            else None
        )
        output.append(
            action.model_copy(
                update={
                    "member_id": member.member_id,
                    "member_name": member.name,
                    "member_relationship": member.relationship,
                    "amount": amount,
                    "clubbing_check": clubbing_check,
                    "params": {**action.params, "changes": member_changes},
                }
            )
        )
    return output, notes


def _sip_global_assignments(
    source: MemberContext,
    amount: Decimal,
    members: Sequence[MemberContext],
    *,
    fx_rates: Mapping[str, Decimal],
    amc_international_suspended: bool,
    etf_inav_premium_pct: Decimal | None,
    us_situs_holdings_usd: Decimal,
) -> tuple[list[tuple[MemberContext, Decimal, dict[str, object]]], list[str]]:
    ordered = [source, *(member for member in _preferred(members, {}) if member != source)]
    placements: list[tuple[MemberContext, Decimal, dict[str, object]]] = []
    notes: list[str] = []
    remaining = amount
    for member in ordered:
        decision = _route_for_member(
            member,
            fx_rates,
            amc_international_suspended=amc_international_suspended,
            etf_inav_premium_pct=etf_inav_premium_pct,
            us_situs_holdings_usd=us_situs_holdings_usd,
        )
        instrument = decision.vehicle
        if (
            instrument == "no_open_route"
            or _instrument_eligibility(member, instrument) == "not_allowed"
        ):
            continue
        if member.residency.country == "DK":
            dkk_inr = fx_rates.get("DKK/INR")
            if dkk_inr is None or dkk_inr <= 0:
                placements.append(
                    (
                        member,
                        remaining,
                        {
                            "instrument_category": "ucits_global_etf",
                            "route": "LOCAL",
                            "account_type": "ASK",
                            "broker": member.member.investing.broker,
                        },
                    )
                )
                notes.append("ask_room_unchecked_fx_missing")
                return placements, notes
            ask_room = member.limits_room.get("ask_deposit_dkk", Decimal(0))
            ask_room_inr = round_paise(ask_room * dkk_inr)
            ask_amount = min(remaining, ask_room_inr)
            if ask_amount > 0:
                placements.append(
                    (
                        member,
                        ask_amount,
                        {
                            "instrument_category": "ucits_global_etf",
                            "route": "LOCAL",
                            "account_type": "ASK",
                            "broker": member.member.investing.broker,
                        },
                    )
                )
                remaining = round_paise(remaining - ask_amount)
            if remaining > 0:
                placements.append(
                    (
                        member,
                        remaining,
                        {
                            "instrument_category": "ucits_global_etf",
                            "route": "LOCAL",
                            "account_type": "ORDINARY",
                            "broker": member.member.investing.broker,
                        },
                    )
                )
                notes.append(f"{member.name}: global_equity_ask_room_spent_then_local_account")
                return placements, notes
            return placements, notes

        chunk_amount = remaining
        if decision.route == "C" and member.residency.status == "RI":
            chunk_amount = min(remaining, _lrs_room_inr(member, fx_rates))
        if chunk_amount <= 0:
            continue
        placements.append(
            (
                member,
                chunk_amount,
                {
                    "instrument_category": instrument,
                    "route": decision.route,
                },
            )
        )
        remaining = round_paise(remaining - chunk_amount)
        if remaining <= 0:
            return placements, notes
    if remaining > 0:
        notes.append("global_equity_no_open_route")
        return [], notes
    return placements, notes


def _debt_need(action: Action) -> DebtNeed:
    if action.kind == "BUILD_EMERGENCY_FUND":
        return DebtNeed(
            amount=action.amount,
            horizon_years=Decimal(0),
            liquidity_need=LiquidityClass.INSTANT,
        )
    horizon = Decimal(15)
    return DebtNeed(
        amount=action.amount,
        horizon_years=horizon,
        liquidity_need=LiquidityClass.LOCKED,
    )


def _ranked_debt_candidates(
    action: Action,
    members: Sequence[MemberContext],
    debt_yields: Mapping[str, Decimal],
    as_of_year: int,
) -> list[tuple[MemberContext, DebtPick]]:
    need = _debt_need(action)
    ranked: list[tuple[MemberContext, DebtPick]] = []
    for member in members:
        birth_year = member.member.tax.birth_year or member.member.investing.birth_year
        age = max(0, as_of_year - birth_year) if birth_year is not None else 0
        result = pick_debt_with_unranked(
            need,
            member.financial_context,
            member.tax_context.slab_pct,
            age,
            debt_yields,
            member.snapshot,
            residency=member.residency,
        )
        ranked.extend(
            (member, pick)
            for pick in result.ranked_picks
            if _instrument_eligibility(member, pick.instrument) in {"allowed", "conditional"}
        )
    ranked.sort(
        key=lambda item: (
            -item[1].after_tax_yield_pct,
            item[0].new_money <= 0,
            not item[0].member.is_self,
            item[0].member_id,
        )
    )
    return ranked


def _debt_room(member: MemberContext, instrument: str) -> Decimal | None:
    room_keys = {
        "ppf": "ppf_annual_inr",
        "scss": "scss_inr",
        "ssy": "ssy_annual_inr",
        "pomis": "pomis_inr",
    }
    key = room_keys.get(instrument)
    return member.limits_room.get(key, Decimal(0)) if key else None


def _place_debt(
    action: Action,
    members: Sequence[MemberContext],
    available: dict[str, Decimal],
    *,
    debt_yields: Mapping[str, Decimal],
    as_of_year: int,
) -> tuple[list[Action], list[str], Decimal]:
    candidates = _ranked_debt_candidates(action, members, debt_yields, as_of_year)
    if not candidates:
        return [], ["debt_no_priced_eligible_candidate"], action.amount

    remaining = action.amount
    placed: list[Action] = []
    notes: list[str] = []
    used: set[tuple[str, str]] = set()
    for member, pick in candidates:
        key = (member.member_id, pick.instrument)
        if key in used:
            continue
        used.add(key)
        room = _debt_room(member, pick.instrument)
        amount = min(remaining, room) if room is not None else remaining
        if amount <= 0:
            continue
        used_other = _spend_household(available, member.member_id, amount, members)
        clubbing = _clubbing_note_if_needed(member, used_other)
        placed.append(
            _assign(
                action,
                member,
                instrument=pick.instrument,
                amount=amount,
                params={"after_tax_yield_pct": format(pick.after_tax_yield_pct, "f")},
                clubbing_check=clubbing,
            )
        )
        remaining = round_paise(remaining - amount)
        notes.append(
            f"{member.name}: {pick.instrument} after-tax yield "
            f"{format(pick.after_tax_yield_pct, 'f')}%"
        )
        if remaining <= 0:
            break
    if remaining > 0:
        notes.append("debt_limit_room_exhausted")
    return placed, notes, remaining
