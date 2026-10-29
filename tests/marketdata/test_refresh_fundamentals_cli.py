from __future__ import annotations

import pytest

from wealth.marketdata.refresh import _parse_args


def test_refresh_cli_accepts_fundamentals_universes_and_symbol_filter() -> None:
    args = _parse_args(
        [
            "--only",
            "edgar,nse_fundamentals,universes",
            "--symbols",
            "MSFT,BRK.B,MSFT",
        ]
    )

    assert args.sources == ("edgar", "nse_fundamentals", "universes")
    assert args.symbols == ("MSFT", "BRK.B")


def test_refresh_cli_rejects_malformed_symbol_filters() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--only", "edgar", "--symbols", "MSFT,not valid"])
