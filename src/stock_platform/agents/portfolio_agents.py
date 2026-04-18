from __future__ import annotations

from collections import defaultdict
from typing import Any

from stock_platform.data.repository import build_overlap_score_rows
from stock_platform.utils.rules import clamp


class PortfolioAgents:
    def __init__(self, repo: Any, provider: Any) -> None:
        self.repo = repo
        self.provider = provider

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Safe float conversion — returns default for None, empty string, or invalid values."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _portfolio_total_assets(self, payload: dict[str, Any]) -> float:
        return (
            sum(float(row.get("market_value", 0)) for row in payload.get("mutual_funds", []))
            + sum(float(row.get("market_value", 0)) for row in payload.get("etfs", []))
            + sum(float(row.get("market_value", 0)) for row in payload.get("direct_equities", []))
        ) or 1.0

    def capture_user_portfolio(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        prefs = {
            "macro_thesis": payload.get("macro_thesis", ""),
            "investable_surplus": self._to_float(payload.get("investable_surplus")),
            "direct_equity_corpus": self._to_float(payload.get("direct_equity_corpus")),
        }
        self.repo.set_state("user_preferences", prefs)

        mf_rows = [
            {
                "instrument_name": str(row.get("instrument_name") or "").strip(),
                "market_value": self._to_float(row.get("market_value")),
                "source": "user_input",
            }
            for row in payload.get("mutual_funds", [])
            if str(row.get("instrument_name") or "").strip()
        ]
        self.repo.replace_raw_holdings("mutual_fund", mf_rows)

        etf_rows = [
            {
                "instrument_name": str(row.get("instrument_name") or "").strip(),
                "market_value": self._to_float(row.get("market_value")),
                "source": "user_input",
            }
            for row in payload.get("etfs", [])
            if str(row.get("instrument_name") or "").strip()
        ]
        self.repo.replace_raw_holdings("etf", etf_rows)

        # Only replace direct equity rows when the payload explicitly provides them.
        # An empty list (e.g., from a PDF upload that has no direct equity section)
        # must NOT wipe manually entered holdings saved from a previous ingestion.
        raw_direct = payload.get("direct_equities", [])
        direct_rows = [
            {
                "instrument_name": str(row.get("instrument_name") or row.get("symbol") or "").strip(),
                "symbol": str(row.get("symbol") or "").strip().upper() or None,
                "quantity": self._to_float(row.get("quantity")),
                "market_value": self._to_float(row.get("market_value")),
                "source": "user_input",
            }
            for row in raw_direct
            if str(row.get("instrument_name") or row.get("symbol") or "").strip()
        ]
        if direct_rows:
            self.repo.replace_raw_holdings("direct_equity", direct_rows)

        return {"payload": payload}

    def parse_mutual_funds(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        statement_month = payload.get("statement_month") or None
        positions: list[dict[str, Any]] = []
        total_assets = self._portfolio_total_assets(payload)
        for fund in payload.get("mutual_funds", []):
            fund_value = float(fund.get("market_value", 0))
            holdings, source = self.provider.get_fund_holdings(fund["instrument_name"], month=statement_month)
            if not holdings:
                positions.append(
                    self.provider.build_proxy_holding(
                        fund["instrument_name"],
                        "mutual_fund",
                        (fund_value / total_assets) * 100,
                        holdings_source=source,
                    )
                )
                continue
            for symbol, weight in holdings.items():
                normalized_symbol = self.provider.normalize_symbol(symbol)
                snapshot = self.provider.get_stock_snapshot(normalized_symbol)
                positions.append(
                    {
                        "instrument_name": fund["instrument_name"],
                        "fund_weight": round(weight * 100, 2),
                        "lookthrough_weight": round((fund_value / total_assets) * weight * 100, 3),
                        "symbol": normalized_symbol,
                        "company_name": snapshot["company_name"],
                        "sector": snapshot["sector"],
                        "source": "mutual_fund",
                        "holdings_source": source,
                    }
                )
        return {"mutual_fund_exposure": positions}

    def decompose_etfs(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        statement_month = payload.get("statement_month") or None
        positions: list[dict[str, Any]] = []
        total_assets = self._portfolio_total_assets(payload)
        for etf in payload.get("etfs", []):
            etf_value = float(etf.get("market_value", 0))
            holdings, source = self.provider.get_etf_holdings(etf["instrument_name"], month=statement_month)
            if not holdings:
                positions.append(
                    self.provider.build_proxy_holding(
                        etf["instrument_name"],
                        "etf",
                        (etf_value / total_assets) * 100,
                        holdings_source=source,
                    )
                )
                continue
            for symbol, weight in holdings.items():
                normalized_symbol = self.provider.normalize_symbol(symbol)
                snapshot = self.provider.get_stock_snapshot(normalized_symbol)
                positions.append(
                    {
                        "instrument_name": etf["instrument_name"],
                        "fund_weight": round(weight * 100, 2),
                        "lookthrough_weight": round((etf_value / total_assets) * weight * 100, 3),
                        "symbol": normalized_symbol,
                        "company_name": snapshot["company_name"],
                        "sector": snapshot["sector"],
                        "source": "etf",
                        "holdings_source": source,
                    }
                )
        return {"etf_exposure": positions}

    def normalize_exposure(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        mf_positions = state.get("mutual_fund_exposure", [])
        etf_positions = state.get("etf_exposure", [])
        direct_positions = payload.get("direct_equities", [])
        total_assets = self._portfolio_total_assets(payload)
        aggregated: dict[str, dict[str, Any]] = {}
        for row in mf_positions + etf_positions:
            bucket = aggregated.setdefault(
                row["symbol"],
                {
                    "symbol": row["symbol"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "total_weight": 0.0,
                    "source_mix": defaultdict(float),
                    "attribution": [],
                },
            )
            bucket["total_weight"] += row["lookthrough_weight"]
            bucket["source_mix"][row["source"]] += row["lookthrough_weight"]
            bucket["attribution"].append(row)

        for row in direct_positions:
            raw_sym = row.get("symbol") or row.get("instrument_name") or ""
            if not raw_sym:
                continue
            symbol = self.provider.normalize_symbol(str(raw_sym))
            snapshot = self.provider.get_stock_snapshot(symbol)
            weight = float(row.get("market_value", 0)) / total_assets * 100
            bucket = aggregated.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "company_name": row.get("instrument_name") or snapshot["company_name"],
                    "sector": row.get("sector") or snapshot["sector"],
                    "total_weight": 0.0,
                    "source_mix": defaultdict(float),
                    "attribution": [],
                },
            )
            bucket["total_weight"] += weight
            bucket["source_mix"]["direct_equity"] += weight
            bucket["attribution"].append(
                {
                    "instrument_name": row.get("instrument_name") or symbol,
                    "symbol": symbol,
                    "lookthrough_weight": round(weight, 3),
                    "source": "direct_equity",
                }
            )

        normalized = [
            {
                "symbol": bucket["symbol"],
                "company_name": bucket["company_name"],
                "sector": bucket["sector"],
                "total_weight": round(bucket["total_weight"], 3),
                "source_mix": dict(bucket["source_mix"]),
                "attribution": bucket["attribution"],
            }
            for bucket in aggregated.values()
        ]
        normalized.sort(key=lambda row: row["total_weight"], reverse=True)
        self.repo.replace_normalized_exposure(normalized)
        return {"normalized_exposure": normalized}

    def compute_overlap(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = state.get("normalized_exposure") or self.repo.list_normalized_exposure()
        rows = build_overlap_score_rows(normalized)
        self.repo.replace_overlap_scores(rows)
        return {"overlap_scores": rows}

    # Sectors at or above this exposure level (as % of total portfolio) are already
    # well-covered via MF look-through and must NOT receive a BUY gap conviction.
    SECTOR_OVERWEIGHT_THRESHOLD = 6.0   # 6 %
    SECTOR_CAUTION_THRESHOLD    = 4.0   # 4 % — marginal add only, cap at NEUTRAL

    @staticmethod
    def _apply_overlap_cap(conviction: str, exposure: float, threshold: float, caution: float) -> tuple[str, str]:
        """Override conviction based on true sector exposure including MF look-through."""
        if exposure >= threshold:
            return "AVOID", f"EXCLUDED: already {exposure:.1f}% via MF/ETF holdings — above {threshold:.0f}% ceiling"
        if exposure >= caution:
            capped = conviction if conviction in {"NEUTRAL", "AVOID"} else "NEUTRAL"
            return capped, f"CAUTION: {exposure:.1f}% existing — marginal add only, conviction capped at NEUTRAL"
        return conviction, f"GAP: only {exposure:.1f}% existing — genuine opportunity"

    def identify_gaps(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = state.get("normalized_exposure") or self.repo.list_normalized_exposure()
        unified = self.repo.list_signals("unified")

        # STEP 1 — aggregate true sector exposure across ALL instruments
        # (MF look-through + ETF + direct equity combined).
        sector_weights: dict[str, float] = defaultdict(float)
        for row in normalized:
            sector_weights[row["sector"]] += row["total_weight"]

        target_weights = self.provider.get_sector_target_weights()
        gaps: list[dict[str, Any]] = []
        for signal in unified:
            sector = signal["sector"]
            target = target_weights.get(sector, 3.0)
            existing = round(sector_weights.get(sector, 0.0), 3)
            underweight_pct = round(max(0.0, target - existing), 3)

            # STEP 2 — apply hard sector exclusion before conviction scoring.
            conviction, overlap_note = self._apply_overlap_cap(
                signal["conviction"],
                existing,
                self.SECTOR_OVERWEIGHT_THRESHOLD,
                self.SECTOR_CAUTION_THRESHOLD,
            )

            # Skip sectors that are both overweight and have no positive signal.
            if underweight_pct <= 0 and existing >= 25:
                continue

            # STEP 3 — recompute score using overlap-adjusted conviction.
            base_score = round(clamp((underweight_pct / max(target, 1)) * 0.5 + signal["score"] * 0.5, 0.0, 1.0), 3)
            # Penalise score for sectors exceeding caution threshold.
            if existing >= self.SECTOR_OVERWEIGHT_THRESHOLD:
                score = 0.0
            elif existing >= self.SECTOR_CAUTION_THRESHOLD:
                score = round(base_score * 0.4, 3)
            else:
                score = base_score

            gaps.append(
                {
                    "sector": sector,
                    "underweight_pct": underweight_pct,
                    "conviction": conviction,
                    "score": score,
                    "reason": f"{overlap_note} | target {target:.1f}% | signal {signal['conviction']}",
                }
            )
        gaps.sort(key=lambda row: row["score"], reverse=True)
        self.repo.replace_gaps(gaps)
        return {
            "identified_gaps": gaps,
            "run_summary": {
                "portfolio_total_positions": len(normalized),
                "gap_count": len(gaps),
            },
        }
