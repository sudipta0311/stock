from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any


class DemoDataProvider:
    """Deterministic local provider used to simulate the external systems in the UML."""

    def __init__(self, holdings_client: Any | None = None) -> None:
        self.holdings_client = holdings_client
        self.today = date.today()
        self.stock_master: dict[str, dict[str, Any]] = {
            "HDFCBANK": {
                "company_name": "HDFC Bank",
                "sector": "Private Banking",
                "price": 1680.0,
                "analyst_target": 1900.0,
                "us_revenue_pct": 2.0,
                "beta": 0.95,
                "avg_daily_value_cr": 32.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 16.2,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.4,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 7.0,
                "pe_trailing": 18.5,
                "pe_5yr_avg": 19.8,
                "sector_pe": 21.0,
                "pe_forward": 16.2,
            },
            "ICICIBANK": {
                "company_name": "ICICI Bank",
                "sector": "Private Banking",
                "price": 1235.0,
                "analyst_target": 1400.0,
                "us_revenue_pct": 1.5,
                "beta": 1.0,
                "avg_daily_value_cr": 28.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 17.1,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.2,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 9.0,
                "pe_trailing": 16.2,
                "pe_5yr_avg": 17.5,
                "sector_pe": 21.0,
                "pe_forward": 14.1,
            },
            "BEL": {
                "company_name": "Bharat Electronics",
                "sector": "Defence",
                "price": 315.0,
                "analyst_target": 370.0,
                "us_revenue_pct": 5.0,
                "beta": 1.15,
                "avg_daily_value_cr": 18.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 27.0,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.1,
                "promoter_trend": "rising",
                "de_ratio": 0.02,
                "drawdown_from_52w": 15.0,
                "pe_trailing": 42.0,
                "pe_5yr_avg": 30.5,
                "sector_pe": 46.0,
                "pe_forward": 36.0,
            },
            "HAL": {
                "company_name": "Hindustan Aeronautics",
                "sector": "Defence",
                "price": 4480.0,
                "analyst_target": 5200.0,
                "us_revenue_pct": 4.0,
                "beta": 1.05,
                "avg_daily_value_cr": 16.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 29.4,
                "fcf_positive_years": 5,
                "revenue_consistency": 7.8,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 11.0,
                "pe_trailing": 35.0,
                "pe_5yr_avg": 25.0,
                "sector_pe": 46.0,
                "pe_forward": 29.5,
            },
            "COCHINSHIP": {
                "company_name": "Cochin Shipyard",
                "sector": "Defence",
                "price": 1850.0,
                "analyst_target": 2200.0,
                "us_revenue_pct": 3.0,
                "beta": 1.18,
                "avg_daily_value_cr": 8.5,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 23.4,
                "fcf_positive_years": 4,
                "revenue_consistency": 7.4,
                "promoter_trend": "stable",
                "de_ratio": 0.04,
                "drawdown_from_52w": 24.0,
                "pe_trailing": 50.0,
                "pe_5yr_avg": 26.0,
                "sector_pe": 46.0,
                "pe_forward": 42.0,
            },
            "SOLARINDS": {
                "company_name": "Solar Industries",
                "sector": "Defence",
                "price": 9980.0,
                "analyst_target": 11500.0,
                "us_revenue_pct": 8.0,
                "beta": 1.25,
                "avg_daily_value_cr": 7.5,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 24.5,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.7,
                "promoter_trend": "rising",
                "de_ratio": 0.09,
                "drawdown_from_52w": 13.0,
                "pe_trailing": 55.0,
                "pe_5yr_avg": 40.0,
                "sector_pe": 46.0,
                "pe_forward": 48.0,
            },
            "DIVISLAB": {
                "company_name": "Divi's Laboratories",
                "sector": "CDMO",
                "price": 4140.0,
                "analyst_target": 4800.0,
                "us_revenue_pct": 52.0,
                "beta": 0.82,
                "avg_daily_value_cr": 11.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 24.8,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.8,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 18.0,
                "pe_trailing": 58.0,
                "pe_5yr_avg": 52.0,
                "sector_pe": 55.0,
                "pe_forward": 50.0,
            },
            "SUNPHARMA": {
                "company_name": "Sun Pharma",
                "sector": "Pharma Exports",
                "price": 1730.0,
                "analyst_target": 1970.0,
                "us_revenue_pct": 30.0,
                "beta": 0.78,
                "avg_daily_value_cr": 15.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 20.1,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.5,
                "promoter_trend": "stable",
                "de_ratio": 0.07,
                "drawdown_from_52w": 10.0,
                "pe_trailing": 38.0,
                "pe_5yr_avg": 35.0,
                "sector_pe": 38.0,
                "pe_forward": 33.5,
            },
            "ABBOTINDIA": {
                "company_name": "Abbott India",
                "sector": "Pharma Exports",
                "price": 28900.0,
                "analyst_target": 33000.0,
                "us_revenue_pct": 10.0,
                "beta": 0.6,
                "avg_daily_value_cr": 5.8,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 33.0,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.3,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 6.0,
                "pe_trailing": 52.0,
                "pe_5yr_avg": 48.0,
                "sector_pe": 38.0,
                "pe_forward": 46.0,
            },
            "KPITTECH": {
                "company_name": "KPIT Technologies",
                "sector": "IT Services",
                "price": 1485.0,
                "analyst_target": 1750.0,
                "us_revenue_pct": 40.0,
                "beta": 1.14,
                "avg_daily_value_cr": 6.9,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 32.0,
                "fcf_positive_years": 4,
                "revenue_consistency": 8.6,
                "promoter_trend": "stable",
                "de_ratio": 0.01,
                "drawdown_from_52w": 22.0,
                "pe_trailing": 65.0,
                "pe_5yr_avg": 46.0,
                "sector_pe": 60.0,
                "pe_forward": 54.0,
            },
            "TATAMOTORS": {
                "company_name": "Tata Motors",
                "sector": "Auto",
                "price": 1015.0,
                "analyst_target": 1150.0,
                "us_revenue_pct": 18.0,
                "beta": 1.44,
                "avg_daily_value_cr": 24.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 18.9,
                "fcf_positive_years": 4,
                "revenue_consistency": 7.7,
                "promoter_trend": "stable",
                "de_ratio": 0.52,
                "drawdown_from_52w": 17.0,
                "pe_trailing": 8.5,
                "pe_5yr_avg": 22.0,
                "sector_pe": 28.0,
                "pe_forward": 7.2,
            },
            "RELIANCE": {
                "company_name": "Reliance Industries",
                "sector": "Energy",
                "price": 3010.0,
                "analyst_target": 3400.0,
                "us_revenue_pct": 6.0,
                "beta": 0.96,
                "avg_daily_value_cr": 40.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 11.4,
                "fcf_positive_years": 3,
                "revenue_consistency": 8.0,
                "promoter_trend": "stable",
                "de_ratio": 0.44,
                "drawdown_from_52w": 12.0,
                "pe_trailing": 22.0,
                "pe_5yr_avg": 24.5,
                "sector_pe": 18.0,
                "pe_forward": 19.5,
            },
            "LT": {
                "company_name": "Larsen & Toubro",
                "sector": "Infrastructure",
                "price": 3880.0,
                "analyst_target": 4400.0,
                "us_revenue_pct": 4.0,
                "beta": 1.02,
                "avg_daily_value_cr": 21.0,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 16.8,
                "fcf_positive_years": 4,
                "revenue_consistency": 8.1,
                "promoter_trend": "stable",
                "de_ratio": 0.18,
                "drawdown_from_52w": 8.0,
                "pe_trailing": 32.0,
                "pe_5yr_avg": 28.0,
                "sector_pe": 30.0,
                "pe_forward": 27.5,
            },
            "DIXON": {
                "company_name": "Dixon Technologies",
                "sector": "Electronics Manufacturing",
                "price": 13940.0,
                "analyst_target": 16000.0,
                "us_revenue_pct": 12.0,
                "beta": 1.2,
                "avg_daily_value_cr": 7.2,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 24.2,
                "fcf_positive_years": 4,
                "revenue_consistency": 8.5,
                "promoter_trend": "rising",
                "de_ratio": 0.06,
                "drawdown_from_52w": 14.0,
                "pe_trailing": 130.0,
                "pe_5yr_avg": 80.0,
                "sector_pe": 100.0,
                "pe_forward": 98.0,
            },
            "TITAN": {
                "company_name": "Titan Company",
                "sector": "Consumption",
                "price": 3680.0,
                "analyst_target": 4300.0,
                "us_revenue_pct": 3.0,
                "beta": 0.92,
                "avg_daily_value_cr": 10.2,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 24.5,
                "fcf_positive_years": 5,
                "revenue_consistency": 8.0,
                "promoter_trend": "stable",
                "de_ratio": 0.04,
                "drawdown_from_52w": 19.0,
                "pe_trailing": 88.0,
                "pe_5yr_avg": 66.0,
                "sector_pe": 75.0,
                "pe_forward": 78.0,
            },
            "IRCTC": {
                "company_name": "IRCTC",
                "sector": "Tourism",
                "price": 960.0,
                "analyst_target": 1150.0,
                "us_revenue_pct": 0.0,
                "beta": 1.1,
                "avg_daily_value_cr": 7.9,
                "promoter_pledge_pct": 0.0,
                "sebi_flag": False,
                "roce_5y": 38.0,
                "fcf_positive_years": 5,
                "revenue_consistency": 7.1,
                "promoter_trend": "stable",
                "de_ratio": 0.0,
                "drawdown_from_52w": 28.0,
                "pe_trailing": 68.0,
                "pe_5yr_avg": 55.0,
                "sector_pe": 55.0,
                "pe_forward": 60.0,
            },
        }
        self.index_members = {
            "NIFTY50": [
                "HDFCBANK",
                "ICICIBANK",
                "BEL",
                "HAL",
                "SUNPHARMA",
                "RELIANCE",
                "LT",
                "TITAN",
                "TATAMOTORS",
                "DIXON",
                "DIVISLAB",
            ],
            "NIFTYNEXT50": [
                "COCHINSHIP",
                "SOLARINDS",
                "KPITTECH",
                "ABBOTINDIA",
                "IRCTC",
            ],
        }
        self.fund_holdings = {
            "Axis Bluechip Fund": {
                "HDFCBANK": 0.085,
                "ICICIBANK": 0.078,
                "LT": 0.046,
                "TITAN": 0.038,
                "RELIANCE": 0.052,
            },
            "Parag Parikh Flexi Cap": {
                "HDFCBANK": 0.06,
                "ICICIBANK": 0.035,
                "SUNPHARMA": 0.025,
                "RELIANCE": 0.05,
                "TATAMOTORS": 0.03,
            },
            "HDFC Defence Fund": {
                "BEL": 0.095,
                "HAL": 0.09,
                "COCHINSHIP": 0.055,
                "SOLARINDS": 0.025,
            },
        }
        self.etf_holdings = {
            "Nifty Bees": {
                "HDFCBANK": 0.12,
                "ICICIBANK": 0.09,
                "RELIANCE": 0.1,
                "LT": 0.035,
                "SUNPHARMA": 0.03,
            },
            "CPSE ETF": {
                "BEL": 0.08,
                "HAL": 0.06,
                "COCHINSHIP": 0.02,
            },
        }
        # Scores represent signal strength, not directionality — use low-positive
        # values (0.10–0.25) for headwind sectors instead of negative numbers so
        # that weighted aggregation doesn't drag the unified score below zero.
        self.geo_templates = [
            ("India-Pakistan tension", "Defence", 0.92, "STRONGLY_POSITIVE", "long"),
            ("India-Pakistan tension", "Tourism", 0.14, "CAUTIOUS", "short"),
            ("US-China decoupling", "Electronics Manufacturing", 0.76, "POSITIVE", "medium"),
            ("Middle East oil spike", "Energy", 0.18, "CAUTIOUS", "short"),
            ("Middle East oil spike", "Renewables", 0.55, "POSITIVE", "medium"),
            ("India-EU FTA progress", "Pharma Exports", 0.71, "POSITIVE", "medium"),
        ]
        # US tariff exposure by sector: score = headwind severity (higher = worse).
        # Scores are kept low-positive so aggregation doesn't drag unified scores negative.
        self.tariff_sector_exposure: dict[str, float] = {
            "IT Services": 0.72,
            "CDMO": 0.60,
            "Pharma Exports": 0.55,
            "Electronics Manufacturing": 0.45,
            "Auto": 0.35,
            "Consumption": 0.12,
            "Infrastructure": 0.08,
            "Defence": 0.06,
            "Private Banking": 0.04,
            "Energy": 0.10,
        }
        self.policy_templates = [
            ("RBI rate easing", "Private Banking", 0.68, "POSITIVE", "medium"),
            ("Budget capex push", "Infrastructure", 0.80, "POSITIVE", "medium"),
            ("PLI expansion", "Electronics Manufacturing", 0.82, "POSITIVE", "long"),
            ("SEBI tightening", "Broking", 0.16, "CAUTIOUS", "short"),
            ("State election win", "Consumption", 0.58, "POSITIVE", "short"),
        ]

    def _stable_value(self, key: str, low: float, high: float) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        basis = int(digest[:8], 16) / 0xFFFFFFFF
        return round(low + (high - low) * basis, 4)

    def _news_snippet(self, key: str) -> str:
        snippets = [
            "order pipeline remains strong",
            "margin outlook is improving",
            "management commentary remains measured",
            "investor sentiment is cautious but constructive",
            "recent execution trend supports the thesis",
        ]
        index = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(snippets)
        return snippets[index]

    def get_geopolitical_signals(self, macro_thesis: str = "") -> list[dict[str, Any]]:
        thesis_boost = 0.08 if macro_thesis else 0.0
        rows = []
        for signal_type, sector, score, conviction, horizon in self.geo_templates:
            final_score = round(score + thesis_boost if sector in macro_thesis else score, 3)
            rows.append(
                {
                    "signal_key": signal_type,
                    "sector": sector,
                    "conviction": conviction,
                    "score": final_score,
                    "source": "GEOAPI+NET",
                    "horizon": horizon,
                    "detail": f"{signal_type}: {sector} {conviction.lower().replace('_', ' ')}",
                    "as_of_date": self.today.isoformat(),
                    "payload": {"macro_override": sector in macro_thesis},
                }
            )
        return rows

    def get_tariff_signals(self) -> list[dict[str, Any]]:
        """Return US tariff headwind signals scored by sector revenue exposure to the US market."""
        rows = []
        for sector, exposure_score in self.tariff_sector_exposure.items():
            # Invert exposure: high US revenue exposure → low score (headwind).
            tariff_score = round(1.0 - exposure_score, 3)
            conviction = "CAUTIOUS" if exposure_score >= 0.5 else "NEUTRAL" if exposure_score >= 0.2 else "POSITIVE"
            rows.append(
                {
                    "signal_key": f"US_TARIFF_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": conviction,
                    "score": tariff_score,
                    "source": "TARIFF_MONITOR",
                    "horizon": "medium",
                    "detail": f"US tariff exposure {exposure_score * 100:.0f}% of revenue — {conviction.lower()} outlook",
                    "as_of_date": self.today.isoformat(),
                    "payload": {"us_revenue_exposure_pct": round(exposure_score * 100, 1)},
                }
            )
        return rows

    def get_policy_signals(self) -> list[dict[str, Any]]:
        rows = []
        for signal_type, sector, score, conviction, horizon in self.policy_templates:
            rows.append(
                {
                    "signal_key": signal_type,
                    "sector": sector,
                    "conviction": conviction,
                    "score": score,
                    "source": "PIB+RBI+SEBI",
                    "horizon": horizon,
                    "detail": f"{signal_type} supports {sector}",
                    "as_of_date": self.today.isoformat(),
                    "payload": {"policy_type": signal_type},
                }
            )
        return rows

    def get_flow_signals(self) -> list[dict[str, Any]]:
        return [
            {
                "signal_key": "FII_DII_FLOW",
                "sector": "Macro India",
                "conviction": "BUY",
                "score": 0.61,
                "source": "NSDL",
                "horizon": "short",
                "detail": "FII selling + DII buying indicates accumulation",
                "as_of_date": self.today.isoformat(),
                "payload": {"fii_net_30d_cr": -4820, "dii_net_30d_cr": 5235, "label": "ACCUMULATION"},
            }
        ]

    def get_contrarian_signals(self) -> list[dict[str, Any]]:
        rows = []
        for sector in {"Defence", "CDMO", "Tourism", "IT Services"}:
            raw = self._stable_value(sector, 0.20, 0.80)
            conviction = "BUY" if raw > 0.55 else "NEUTRAL" if raw > 0.35 else "CAUTION"
            rows.append(
                {
                    "signal_key": f"CONTRA_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": conviction,
                    "score": round(raw, 3),
                    "source": "NET+FIN",
                    "horizon": "entry",
                    "detail": f"Sentiment scan says {self._news_snippet(sector)}",
                    "as_of_date": self.today.isoformat(),
                    "payload": {"drawdown_from_52w": abs(int(raw * 30))},
                }
            )
        return rows

    def get_current_market_signal(self, sector: str) -> float:
        base = self._stable_value(f"market-{sector}", 0.25, 0.9)
        return round(base, 3)

    def get_index_members(self, index_name: str) -> list[dict[str, Any]]:
        members = self.index_members.get(index_name, self.index_members["NIFTY50"])
        return [self.get_stock_snapshot(symbol) for symbol in members]

    def get_stock_snapshot(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = self.normalize_symbol(symbol)
        master = self.stock_master.get(normalized_symbol)
        if master:
            return {"symbol": normalized_symbol, **master}
        return {
            "symbol": normalized_symbol,
            "company_name": normalized_symbol,
            "sector": self.infer_sector(normalized_symbol),
            "price": 100.0,
            "analyst_target": 112.0,
            "us_revenue_pct": 0.0,
            "beta": 1.0,
            "avg_daily_value_cr": 6.0,
            "promoter_pledge_pct": 0.0,
            "sebi_flag": False,
            "roce_5y": 14.0,
            "fcf_positive_years": 3,
            "revenue_consistency": 6.5,
            "promoter_trend": "stable",
            "de_ratio": 0.3,
            "drawdown_from_52w": 12.0,
            "pe_trailing": 25.0,
            "pe_5yr_avg": 25.0,
            "sector_pe": 25.0,
            "pe_forward": 22.0,
        }

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace(".NSE", "").replace(".BSE", "").replace(" ", "").upper()

    def infer_sector(self, instrument_name: str) -> str:
        name = instrument_name.lower()
        rules = [
            ("liquid", "Fixed Income"),
            ("short term", "Fixed Income"),
            ("sdl", "Fixed Income"),
            ("bond", "Fixed Income"),
            ("nasdaq", "US Equity"),
            ("us ", "US Equity"),
            ("u.s.", "US Equity"),
            ("global", "Global Equity"),
            ("technology", "Technology"),
            ("innovation", "Technology"),
            ("mid cap", "Mid Cap"),
            ("midcap", "Mid Cap"),
            ("small cap", "Small Cap"),
            ("large & mid", "Large & Mid Cap"),
            ("large cap", "Large Cap"),
            ("flexi cap", "Flexi Cap"),
            ("value", "Value"),
            ("nifty next 50", "Next 50 Index"),
            ("nifty index", "Large Cap Index"),
            ("mnc", "Consumption"),
            ("elss", "Tax Saver"),
            ("defence", "Defence"),
            ("bank", "Private Banking"),
            ("pharma", "Pharma Exports"),
        ]
        for needle, sector in rules:
            if needle in name:
                return sector
        return "Diversified Equity"

    def build_proxy_holding(
        self,
        instrument_name: str,
        source: str,
        lookthrough_weight: float,
        holdings_source: str = "proxy:scheme-level",
    ) -> dict[str, Any]:
        token = hashlib.md5(f"{source}:{instrument_name}".encode("utf-8")).hexdigest()[:8].upper()
        symbol = f"{source[:2].upper()}_{token}"
        return {
            "instrument_name": instrument_name,
            "fund_weight": 100.0,
            "lookthrough_weight": round(lookthrough_weight, 3),
            "symbol": symbol,
            "company_name": instrument_name,
            "sector": self.infer_sector(instrument_name),
            "source": source,
            "proxy": True,
            "holdings_source": holdings_source,
        }

    def get_stock_news(self, symbol: str) -> dict[str, Any]:
        info = self.get_stock_snapshot(symbol)
        return {
            "symbol": info["symbol"],
            "sentiment_score": self._stable_value(f"news-{info['symbol']}", -0.2, 0.85),
            "headline": f"{info['company_name']}: {self._news_snippet(info['symbol'])}",
        }

    def get_sector_news(self, sector: str) -> dict[str, Any]:
        return {
            "sector": sector,
            "signal_score": self.get_current_market_signal(sector),
            "summary": f"{sector} signals indicate {self._news_snippet(sector)}",
        }

    def get_financials(self, symbol: str) -> dict[str, Any]:
        info = self.get_stock_snapshot(symbol)
        return {
            "symbol": info["symbol"],
            "roce_5y": info["roce_5y"],
            "fcf_positive_years": info["fcf_positive_years"],
            "revenue_consistency": info["revenue_consistency"],
            "promoter_trend": info["promoter_trend"],
            "de_ratio": info["de_ratio"],
            "us_revenue_pct": info.get("us_revenue_pct", 0.0),
            "pe_trailing": info["pe_trailing"],
            "pe_5yr_avg": info["pe_5yr_avg"],
            "sector_pe": info["sector_pe"],
            "pe_forward": info["pe_forward"],
        }

    def get_risk_metrics(self, symbol: str) -> dict[str, Any]:
        info = self.get_stock_snapshot(symbol)
        return {
            "symbol": info["symbol"],
            "avg_daily_value_cr": info["avg_daily_value_cr"],
            "beta": info["beta"],
            "promoter_pledge_pct": info["promoter_pledge_pct"],
            "sebi_flag": info["sebi_flag"],
        }

    def get_price_context(self, symbol: str) -> dict[str, Any]:
        info = self.get_stock_snapshot(symbol)
        return {
            "symbol": info["symbol"],
            "price": info["price"],
            "analyst_target": info.get("analyst_target", round(info["price"] * 1.12, 2)),
            "drawdown_from_52w": info["drawdown_from_52w"],
            "price_change_1m": round(self._stable_value(f"1m-{info['symbol']}", -12.0, 16.0), 2),
            "price_change_6m": round(self._stable_value(f"6m-{info['symbol']}", -18.0, 28.0), 2),
        }

    def get_fund_holdings(self, fund_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        if fund_name in self.fund_holdings:
            return self.fund_holdings[fund_name], "demo:built_in"
        if self.holdings_client:
            live, source = self.holdings_client.resolve_with_meta(fund_name, month=month)
            if live:
                return live, source
        return {}, "unresolved"

    def get_etf_holdings(self, etf_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        if etf_name in self.etf_holdings:
            return self.etf_holdings[etf_name], "demo:built_in"
        if self.holdings_client:
            live, source = self.holdings_client.resolve_with_meta(etf_name, month=month)
            if live:
                return live, source
        return {}, "unresolved"

    def get_sector_target_weights(self) -> dict[str, float]:
        return {
            "Defence": 8.0,
            "Private Banking": 10.0,
            "CDMO": 5.0,
            "Pharma Exports": 7.0,
            "Electronics Manufacturing": 5.0,
            "Infrastructure": 6.0,
            "Consumption": 5.0,
            "IT Services": 4.0,
            "Tourism": 2.0,
        }

    def get_monitoring_price_series(self, symbol: str, entry_price: float | None = None) -> dict[str, Any]:
        snapshot = self.get_stock_snapshot(symbol)
        current = self.get_price_context(snapshot["symbol"])["price"]
        assumed_entry = entry_price or round(current / (1 - self._stable_value(f"entry-{snapshot['symbol']}", -0.18, 0.22)), 2)
        return {
            "symbol": snapshot["symbol"],
            "entry_price": round(assumed_entry, 2),
            "current_price": current,
            "drawdown_pct": round(((current - assumed_entry) / assumed_entry) * 100, 2),
            "as_of_date": self.today.isoformat(),
        }

    def recent_date(self, days_back: int) -> str:
        return (self.today - timedelta(days=days_back)).isoformat()

    def demo_portfolio_payload(self) -> dict[str, Any]:
        return {
            "macro_thesis": "Prefer Defence and Electronics Manufacturing, avoid Tourism for now",
            "investable_surplus": 500000,
            "direct_equity_corpus": 800000,
            "mutual_funds": [
                {"instrument_name": "Axis Bluechip Fund", "market_value": 650000},
                {"instrument_name": "Parag Parikh Flexi Cap", "market_value": 540000},
            ],
            "etfs": [
                {"instrument_name": "Nifty Bees", "market_value": 250000},
            ],
            "direct_equities": [
                {"instrument_name": "HDFC Bank", "symbol": "HDFCBANK", "quantity": 45, "market_value": 75600},
                {"instrument_name": "Titan Company", "symbol": "TITAN", "quantity": 20, "market_value": 73600},
            ],
        }
