from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.services.engine import PlatformEngine


def main() -> None:
    pdf_path = ROOT / "tests" / "NSDLe-CAS_109102284_FEB_2026.PDF"
    password = "AYFPS8467G"
    engine = PlatformEngine()

    payload = engine.parse_portfolio_pdf(pdf_path, password)
    ingestion = engine.ingest_portfolio(payload)
    buy = engine.run_buy_analysis(
        {
            "index_name": "NIFTY50",
            "horizon_months": 18,
            "risk_profile": "Balanced",
            "top_n": 3,
        }
    )
    monitoring = engine.run_monitoring()
    snapshot = engine.get_dashboard_snapshot()

    summary = {
        "parsed": {
            "mutual_funds": len(payload["mutual_funds"]),
            "etfs": len(payload["etfs"]),
            "direct_equities": len(payload["direct_equities"]),
            "direct_equity_corpus": payload["direct_equity_corpus"],
            "investable_surplus": payload["investable_surplus"],
        },
        "ingestion": {
            "normalized_exposure": len(ingestion["normalized_exposure"]),
            "overlap_scores": len(ingestion["overlap_scores"]),
            "identified_gaps": len(ingestion["identified_gaps"]),
            "holdings_sources": {
                row.get("holdings_source", "n/a"): 0 for row in (ingestion["mutual_fund_exposure"] + ingestion["etf_exposure"])
            },
        },
        "buy": buy["run_summary"],
        "monitoring": monitoring["run_summary"],
        "recommendations": [
            {
                "symbol": row["symbol"],
                "sector": row["sector"],
                "action": row["action"],
                "score": row["score"],
                "llm_used": row["payload"].get("llm_used"),
            }
            for row in snapshot["recommendations"]
        ],
        "monitor_actions": [
            {
                "symbol": row["symbol"],
                "action": row["action"],
                "severity": row["severity"],
                "llm_used": row["payload"].get("llm_used"),
            }
            for row in snapshot["monitoring_actions"][:10]
        ],
    }
    for row in ingestion["mutual_fund_exposure"] + ingestion["etf_exposure"]:
        key = row.get("holdings_source", "n/a")
        summary["ingestion"]["holdings_sources"][key] = summary["ingestion"]["holdings_sources"].get(key, 0) + 1
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
