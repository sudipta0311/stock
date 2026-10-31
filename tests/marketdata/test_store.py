from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from wealth.db.models import IndexValuation
from wealth.marketdata.store import latest_index_valuation_percentile


@pytest.mark.parametrize(
    ("history_days", "expected_label"),
    [(1058, "short_history"), (1059, "long_history")],
)
def test_index_valuation_history_label_uses_monthly_series_boundary(
    history_days: int, expected_label: str
) -> None:
    first_as_of = date(2024, 1, 1)
    latest_as_of = first_as_of + timedelta(days=history_days)
    rows = [
        IndexValuation(
            index_name="S&P 500",
            as_of=first_as_of,
            pe=Decimal("20"),
            source="synthetic",
            source_url="synthetic",
        ),
        IndexValuation(
            index_name="S&P 500",
            as_of=latest_as_of,
            pe=Decimal("25"),
            source="synthetic",
            source_url="synthetic",
        ),
    ]
    session = Mock(spec=Session)
    session.scalars.return_value = rows

    result = latest_index_valuation_percentile(session, "S&P 500", latest_as_of)

    assert result is not None
    assert result.history_label == expected_label
