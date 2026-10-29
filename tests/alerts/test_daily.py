"""Frozen-clock checks for maturity-calendar reminder offsets."""

from datetime import date

from wealth.alerts.daily import reminders_due


def test_reminders_are_scheduled_thirty_seven_and_zero_days_before_event() -> None:
    today = date(2026, 10, 28)
    events = [
        {
            "date": "2026-11-27",
            "asset_id": "asset-thirty",
            "kind": "maturity",
            "label": "Synthetic deposit",
        },
        {
            "date": "2026-11-04",
            "asset_id": "asset-seven",
            "kind": "renewal",
            "label": "Synthetic renewal",
        },
        {
            "date": "2026-10-28",
            "asset_id": "asset-today",
            "kind": "maturity",
            "label": "Synthetic maturity today",
        },
        {
            "date": "2026-11-26",
            "asset_id": "asset-twenty-nine",
            "kind": "maturity",
            "label": "Not due for a reminder yet",
        },
    ]

    due = reminders_due(events, today)

    assert [item["offset_days"] for item in due] == [0, 7, 30]
    assert {item["asset_id"] for item in due} == {"asset-today", "asset-seven", "asset-thirty"}
    assert {item["notify_on"] for item in due} == {today.isoformat()}


def test_loan_payment_and_rate_review_use_opt_in_reminder_dates() -> None:
    today = date(2026, 10, 28)
    events = [
        {
            "date": "2026-10-31",
            "asset_id": "loan-payment",
            "kind": "loan_payment",
            "label": "Loan monthly payment",
        },
        {
            "date": "2026-10-28",
            "asset_id": "loan-rate-review",
            "kind": "loan_rate_review",
            "label": "Check your loan rate",
        },
        {
            "date": "2026-10-31",
            "asset_id": "other-event",
            "kind": "maturity",
            "label": "Other dated item",
        },
    ]

    due = reminders_due(events, today)

    assert {(item["asset_id"], item["offset_days"]) for item in due} == {
        ("loan-payment", 3),
        ("loan-rate-review", 0),
    }
