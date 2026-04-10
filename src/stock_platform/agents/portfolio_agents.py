from __future__ import annotations

from collections import defaultdict
from typing import Any

from stock_platform.utils.rules import clamp


class PortfolioAgents:
    def __init__(self, repo: Any, provider: Any) -> None:
        self.repo = repo
        self.provider = provider

    def capture_user_portfolio(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        prefs = {
            "macro_thesis": payload.get("macro_thesis", ""),
            "investable_surplus": float(payload.get("investable_surplus", 0) or 0),
            "direct_equity_corpus": float(payload.get("direct_equity_corpus", 0) or 0),
        }
        self.repo.set_state("user_preferences", prefs)
        self.repo.replace_raw_holdings(
            "mutual_fund",
            [
                {
                    "instrument_name": row["instrument_name"],
                    "market_value": float(row.get("market_value", 0)),
                    "source": "user_input",
                }
                for row in payload.get("mutual_funds", [])
            ],
        )
        self.repo.replace_raw_holdings(
            "etf",
            [
                {
                    "instrument_name": row["instrument_name"],
                    "market_value": float(row.get("market_value", 0)),
                    "source": "user_input",
                }
                for row in payload.get("etfs", [])
            ],
        )
        self.repo.replace_raw_holdings(
            "direct_equity",
            [
                {
                    "instrument_name": row.get("instrument_name") or row.get("symbol"),
                    "symbol": row.get("symbol"),
                    "quantity": float(row.get("quantity", 0)),
                    "market_value": float(row.get("market_value", 0)),
                    "source": "user_input",
                }
                for row in payload.get("direct_equities", [])
            ],
        )
        return {"payload": payload}

    def parse_mutual_funds(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("payload", {})
        statement_month = payload.get("statement_month") or None
        positions: list[dict[str, Any]] = []
        total_value = sum(float(row.get("market_value", 0)) for row in payload.get("mutual_funds", [])) or 1.0
        for fund in payload.get("mutual_funds", []):
            fund_value = float(fund.get("market_value", 0))
            holdings, source = self.provider.get_fund_holdings(fund["instrument_name"], month=statement_month)
            if not holdings:
                positions.append(
                    self.provider.build_proxy_holding(
                        fund["instrument_name"],
                        "mutual_fund",
                        (fund_value / total_value) * 100,
                        holdings_source=source,
                    )
                )
                continue
            for symbol, weight in holdings.items():
                snapshot = self.provider.get_stock_snapshot(symbol)
                positions.append(
                    {
                        "instrument_name": fund["instrument_name"],
                        "fund_weight": round(weight * 100, 2),
                        "lookthrough_weight": round((fund_value / total_value) * weight * 100, 3),
                        "symbol": symbol,
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
        total_value = sum(float(row.get("market_value", 0)) for row in payload.get("etfs", [])) or 1.0
        for etf in payload.get("etfs", []):
            etf_value = float(etf.get("market_value", 0))
            holdings, source = self.provider.get_etf_holdings(etf["instrument_name"], month=statement_month)
            if not holdings:
                positions.append(
                    self.provider.build_proxy_holding(
                        etf["instrument_name"],
                        "etf",
                        (etf_value / total_value) * 100,
                        holdings_source=source,
                    )
                )
                continue
            for symbol, weight in holdings.items():
                snapshot = self.provider.get_stock_snapshot(symbol)
                positions.append(
                    {
                        "instrument_name": etf["instrument_name"],
                        "fund_weight": round(weight * 100, 2),
                        "lookthrough_weight": round((etf_value / total_value) * weight * 100, 3),
                        "symbol": symbol,
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
        total_assets = (
            sum(float(row.get("market_value", 0)) for row in payload.get("mutual_funds", []))
            + sum(float(row.get("market_value", 0)) for row in payload.get("etfs", []))
            + sum(float(row.get("market_value", 0)) for row in direct_positions)
        ) or 1.0
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
            symbol = self.provider.normalize_symbol(row["symbol"])
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
        rows: list[dict[str, Any]] = []
        for row in normalized:
            indirect_overlap = sum(
                item["lookthrough_weight"] for item in row["attribution"] if item["source"] != "direct_equity"
            )
            overlap_pct = round(indirect_overlap, 3)
            if overlap_pct > 3:
                band = "HARD_EXCLUDE"
            elif overlap_pct > 1:
                band = "FLAG"
            elif overlap_pct > 0.5:
                band = "YELLOW"
            else:
                band = "GREEN"
            rows.append(
                {
                    "symbol": row["symbol"],
                    "overlap_pct": overlap_pct,
                    "band": band,
                    "attribution": [item for item in row["attribution"] if item["source"] != "direct_equity"],
                }
            )
        self.repo.replace_overlap_scores(rows)
        return {"overlap_scores": rows}

    def identify_gaps(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized = state.get("normalized_exposure") or self.repo.list_normalized_exposure()
        unified = self.repo.list_signals("unified")
        sector_weights: dict[str, float] = defaultdict(float)
        for row in normalized:
            sector_weights[row["sector"]] += row["total_weight"]
        target_weights = self.provider.get_sector_target_weights()
        gaps: list[dict[str, Any]] = []
        for signal in unified:
            sector = signal["sector"]
            target = target_weights.get(sector, 3.0)
            existing = sector_weights.get(sector, 0.0)
            underweight_pct = round(max(0.0, target - existing), 3)
            score = round(clamp((underweight_pct / max(target, 1)) * 0.5 + signal["score"] * 0.5, 0.0, 1.0), 3)
            if underweight_pct <= 0 and existing >= 25:
                continue
            gaps.append(
                {
                    "sector": sector,
                    "underweight_pct": underweight_pct,
                    "conviction": signal["conviction"],
                    "score": score,
                    "reason": f"Current exposure {existing:.2f}% vs target {target:.1f}% with {signal['conviction']} signal",
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
