"""Household fund-check API contract tests using offline fakes."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wealth.api import fund_check as fund_check_api
from wealth.db.models import HouseholdMember
from wealth.policy.fund_check import FundCheckResult, FundMember


def _result() -> FundCheckResult:
    return FundCheckResult(
        status="fine",
        summary_key="fund_check.good",
        checks=(),
        actions=(),
        early_warning={
            "bucket": "small",
            "months_until": 5,
            "direction": "over",
            "verified": False,
        },
        details={"funds": [], "mix": {}, "notes": [], "sips": []},
        lagging_candidates=(),
        sip_amounts=(),
    )


class _Session:
    def __init__(self, member_id: object = None) -> None:
        self.member_id = member_id

    def scalar(self, _statement: object) -> object:
        return self.member_id


def _client(member_id: object = None) -> TestClient:
    user = SimpleNamespace(
        session=_Session(member_id),
        household_owner_id=uuid4(),
        role="owner",
    )
    app = FastAPI()
    app.include_router(fund_check_api.build_router(lambda: user))
    return TestClient(app)


def test_get_fund_check_returns_capped_contract_and_member_warnings(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fund_check_api, "evaluate_fund_check", lambda *_args, **_kwargs: _result())
    monkeypatch.setattr(
        fund_check_api,
        "_build_inputs",
        lambda *_args: ([], [], {}, [], [FundMember("member-one")], None, None, set()),
    )

    response = _client().get("/v1/fund-check")

    assert response.status_code == 200
    assert response.json()["status"] == "fine"
    assert response.json()["summary_key"] == "fund_check.good"
    assert response.json()["checks"] == []
    assert response.json()["early_warning"]["family"]["months_until"] == 5
    assert response.json()["early_warning"]["members"]["member-one"]["months_until"] == 5
    assert response.json()["details"]["funds"] == []


def test_get_fund_check_validates_member_ids(monkeypatch) -> None:
    monkeypatch.setattr(fund_check_api, "evaluate_fund_check", lambda *_args, **_kwargs: _result())
    client = _client()

    response = client.get("/v1/fund-check?member=not-a-member")

    assert response.status_code == 422


def test_member_lookup_is_limited_to_the_selected_household(monkeypatch) -> None:
    monkeypatch.setattr(fund_check_api, "evaluate_fund_check", lambda *_args, **_kwargs: _result())
    client = _client()

    response = client.get(f"/v1/fund-check?member={uuid4()}")

    assert response.status_code == 404


def test_member_residency_uses_current_household_plan_before_stored_profile() -> None:
    owner_id = uuid4()
    member_id = uuid4()
    row = HouseholdMember(
        id=member_id,
        user_id=owner_id,
        display_name="Synthetic member",
        relationship="self",
        is_self=True,
        residency={"country": "IN", "status": "RI"},
        tax={},
        investing={},
    )

    class Session:
        def scalars(self, _statement: object) -> list[HouseholdMember]:
            return [row]

        def execute(self, _statement: object) -> list[tuple[str, None]]:
            return []

    members, _rows = fund_check_api._member_rows(
        Session(),
        owner_id,
        None,
        {
            "members": [
                {
                    "id": str(member_id),
                    "residency": {"country": "DK", "status": "NRI"},
                }
            ]
        },
    )

    assert members[0].residency_country == "DK"
    assert members[0].residency_status == "NRI"


def test_weekly_plan_adds_one_member_action_within_the_shared_three_action_cap(
    monkeypatch,
) -> None:
    result = FundCheckResult(
        status="attention",
        summary_key="fund_check.attention",
        checks=(
            {
                "reason_code": "regular_plan",
                "member_id": "member-spouse",
                "money_at_stake": "5000.00",
                "details": {},
            },
        ),
        actions=(
            {
                "kind": "SWITCH_TO_DIRECT",
                "member_id": "member-spouse",
                "security_id": "synthetic-security",
                "amount": "500000.00",
                "reason_code": "regular_plan",
                "target_security_id": "synthetic-direct",
            },
            {
                "kind": "REDIRECT_SIP",
                "member_id": "member-spouse",
                "security_id": "synthetic-other",
                "amount": "1000.00",
                "reason_code": "projected_band_breach",
            },
        ),
        early_warning=None,
        details={},
        lagging_candidates=(),
        sip_amounts=(),
    )
    captured: dict[str, object] = {}

    def evaluate(*_args, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(fund_check_api, "evaluate_fund_check", evaluate)
    actions = fund_check_api.add_weekly_fund_action(
        None,
        uuid4(),
        date(2026, 10, 31),
        [
            {"kind": "EXISTING_A", "rank": 1, "amount": "100.00", "money_at_stake": "10000"},
            {"kind": "EXISTING_B", "rank": 2, "amount": "100.00", "money_at_stake": "9000"},
            {"kind": "EXISTING_C", "rank": 3, "amount": "100.00", "money_at_stake": "3000"},
        ],
        can_write=False,
    )

    assert len(actions) == 3
    assert sum(action["kind"] == "SWITCH_TO_DIRECT" for action in actions) == 1
    selected = next(action for action in actions if action["kind"] == "SWITCH_TO_DIRECT")
    assert selected["member_id"] == "member-spouse"
    assert selected["rank"] == 3
    assert [action["rank"] for action in actions] == [1, 2, 3]
    assert captured["can_write"] is False
