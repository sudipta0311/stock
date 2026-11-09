from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError

from wealth.marketdata.refresh import _error_summary


def test_database_error_summary_includes_only_the_constraint_name() -> None:
    driver_error = RuntimeError("synthetic driver detail")
    driver_error.diag = SimpleNamespace(constraint_name="uq_synthetic_listing")
    error = IntegrityError("synthetic statement", {}, driver_error)

    assert _error_summary(error) == "IntegrityError (uq_synthetic_listing)"
