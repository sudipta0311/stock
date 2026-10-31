"""Versioned limits and deterministic thresholds for weekly research."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "research.yaml"


@dataclass(frozen=True)
class ResearchRules:
    max_dossiers_per_run: int
    max_usd_per_dossier: Decimal
    max_ideas_per_market: int
    discount_rate_pct: dict[str, Decimal]
    cheap_pe_percentile_max: Decimal
    expensive_pe_percentile_min: Decimal
    cheap_earnings_yield_vs_sector: Decimal
    expensive_earnings_yield_vs_sector: Decimal
    macro_max_usd: Decimal = Decimal("0.05")
    cash_flow_optional_markets: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path | None = None) -> ResearchRules:
        rules_path = path or _RULES_PATH
        raw: dict[str, Any] = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        valuation = raw["valuation"]
        result = cls(
            max_dossiers_per_run=int(raw.get("max_dossiers_per_run", 20)),
            max_usd_per_dossier=Decimal(str(raw.get("max_usd_per_dossier", "0.05"))),
            max_ideas_per_market=int(raw.get("max_ideas_per_market", 5)),
            discount_rate_pct={
                str(market): Decimal(str(value))
                for market, value in raw.get("discount_rate_pct", {}).items()
            },
            cheap_pe_percentile_max=Decimal(str(valuation["cheap_pe_percentile_max"])),
            expensive_pe_percentile_min=Decimal(str(valuation["expensive_pe_percentile_min"])),
            cheap_earnings_yield_vs_sector=Decimal(
                str(valuation["cheap_earnings_yield_vs_sector"])
            ),
            expensive_earnings_yield_vs_sector=Decimal(
                str(valuation["expensive_earnings_yield_vs_sector"])
            ),
            macro_max_usd=Decimal(str(raw.get("macro_max_usd", "0.05"))),
            cash_flow_optional_markets=tuple(
                str(market).upper() for market in valuation.get("cash_flow_optional_markets", [])
            ),
        )
        if result.max_dossiers_per_run < 0 or result.max_ideas_per_market < 0:
            raise ValueError("Research shortlist limits must not be negative")
        if result.max_usd_per_dossier < 0:
            raise ValueError("Research dossier budget must not be negative")
        if not result.macro_max_usd.is_finite() or result.macro_max_usd < 0:
            raise ValueError("Macro research budget must be finite and non-negative")
        if not {"IN", "US"} <= set(result.discount_rate_pct):
            raise ValueError("Research discount rates are required for India and US")
        return result

    def discount_rate(self, market: str) -> Decimal:
        return self.discount_rate_pct.get(market.upper(), Decimal("9"))
