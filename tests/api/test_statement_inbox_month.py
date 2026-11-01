from datetime import UTC, datetime
from uuid import uuid4

import pytest

from wealth.api.statement_inbox import StatementInboxItemResponse, _statement_month


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Your NSDL CAS - September 2026 (monthly statement)", "2026-09"),
        ("Consolidated account statement: FEB. 2027", "2027-02"),
        ("Investment statement for 2026 Mar", "2026-03"),
    ],
)
def test_statement_month_reads_english_month_and_year(
    subject: str,
    expected: str,
) -> None:
    assert _statement_month(subject) == expected


@pytest.mark.parametrize(
    "subject",
    [
        "Statement received",
        "Provider update for September",
        "Statement for Sep 26",
    ],
)
def test_statement_month_is_null_without_a_full_month_and_year(subject: str) -> None:
    assert _statement_month(subject) is None


def test_statement_month_is_in_the_inbox_item_response() -> None:
    item = StatementInboxItemResponse(
        id=uuid4(),
        sender_type="nsdl",
        received_at=datetime(2026, 10, 16, tzinfo=UTC),
        status="needs_password",
        error_code=None,
        import_job_id=None,
        subject="Synthetic provider statement",
        statement_month="2026-09",
    )

    assert item.model_dump()["statement_month"] == "2026-09"
