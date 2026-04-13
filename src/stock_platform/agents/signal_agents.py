from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stock_platform.models import SignalRecord
from stock_platform.utils.rules import clamp, conviction_from_score
from stock_platform.utils.sector_config import SECTOR_GEO_OVERRIDES


class SignalAgents:
    def __init__(self, repo: Any, provider: Any) -> None:
        self.repo = repo
        self.provider = provider

    def collect_geopolitical_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        macro_thesis = state.get("macro_thesis") or self.repo.get_state("user_preferences", {}).get("macro_thesis", "")
        raw_geo = self.provider.get_geopolitical_signals(macro_thesis)
        tariff_signals = self.provider.get_tariff_signals()
        rows = [
            SignalRecord(
                family="geo",
                sector=row["sector"],
                conviction=row["conviction"],
                score=row["score"],
                source=row["source"],
                horizon=row["horizon"],
                detail=row["detail"],
                as_of_date=row["as_of_date"],
                signal_key=row.get("signal_key"),
                payload=row.get("payload", {}),
            )
            for row in raw_geo + tariff_signals
        ]
        self.repo.replace_signals("geo", rows)
        return {"geo_signals": [asdict(row) for row in rows]}

    def collect_policy_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        rows = [SignalRecord(family="policy", **row) for row in self.provider.get_policy_signals()]
        self.repo.replace_signals("policy", rows)
        return {"policy_signals": [asdict(row) for row in rows]}

    def collect_flow_sentiment(self, state: dict[str, Any]) -> dict[str, Any]:
        rows = [SignalRecord(family="flow", **row) for row in self.provider.get_flow_signals()]
        self.repo.replace_signals("flow", rows)
        return {"flow_signals": [asdict(row) for row in rows]}

    def detect_contrarian_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        normalized: list[SignalRecord] = []
        for row in self.provider.get_contrarian_signals():
            score = clamp((row["score"] + 0.3) / 1.2, 0.0, 1.0)
            conviction = "BUY" if row["conviction"] == "BUY" else "NEUTRAL" if row["conviction"] == "CAUTION" else "AVOID"
            normalized.append(
                SignalRecord(
                    family="contrarian",
                    sector=row["sector"],
                    conviction=conviction,
                    score=score,
                    source=row["source"],
                    horizon=row["horizon"],
                    detail=row["detail"],
                    as_of_date=row["as_of_date"],
                    signal_key=row.get("signal_key"),
                    payload=row.get("payload", {}),
                )
            )
        self.repo.replace_signals("contrarian", normalized)
        return {"contrarian_signals": [asdict(row) for row in normalized]}

    def aggregate_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        geo = state.get("geo_signals") or self.repo.list_signals("geo")
        policy = state.get("policy_signals") or self.repo.list_signals("policy")
        flow = state.get("flow_signals") or self.repo.list_signals("flow")
        contrarian = state.get("contrarian_signals") or self.repo.list_signals("contrarian")
        grouped: dict[str, dict[str, Any]] = {}

        def ingest(rows: list[dict[str, Any]], weight: float, family: str) -> None:
            for row in rows:
                bucket = grouped.setdefault(
                    row["sector"],
                    {"sector": row["sector"], "score": 0.0, "weight_used": 0.0, "sources": [], "source_weights": {}},
                )
                bucket["score"] += row["score"] * weight
                bucket["weight_used"] += weight
                bucket["sources"].append(family)
                bucket["source_weights"][family] = row["score"] * weight

        ingest(geo, 0.35, "geo")
        ingest(policy, 0.25, "policy")
        ingest(flow, 0.15, "flow")
        ingest(contrarian, 0.25, "contrarian")

        unified: list[SignalRecord] = []
        for sector, payload in grouped.items():
            # Normalise by the weight of signal families that actually contributed
            # to this sector so single-family sectors aren't penalised for missing
            # families (e.g. a sector with only a strong policy signal should score
            # near that signal's value, not be capped at 0.25).
            weight_used = payload["weight_used"]
            raw = payload["score"] / weight_used if weight_used > 0 else 0.0
            score = clamp(raw, 0.0, 1.0)
            primary_source = max(payload["source_weights"], key=payload["source_weights"].get)
            unified.append(
                SignalRecord(
                    family="unified",
                    sector=sector,
                    conviction=conviction_from_score(score),
                    score=round(score, 3),
                    source=primary_source,
                    horizon="blended",
                    detail=f"Unified signal across {', '.join(sorted(set(payload['sources'])))}",
                    as_of_date=self.provider.today.isoformat(),
                    signal_key=f"UNIFIED_{sector.upper().replace(' ', '_')}",
                    payload={
                        "source_weights": payload["source_weights"],
                        "confidence": round(min(0.95, 0.55 + len(payload["sources"]) * 0.08), 3),
                    },
                )
            )
        # Inject geo overrides for sectors absent from live signal feeds.
        # Only adds; never downgrades a sector that already has a live signal.
        existing_sectors = {row.sector for row in unified}
        for sector, override in SECTOR_GEO_OVERRIDES.items():
            if sector not in existing_sectors:
                unified.append(
                    SignalRecord(
                        family="unified",
                        sector=sector,
                        conviction=override["conviction"],
                        score=override["score"],
                        source=override["source"],
                        horizon="blended",
                        detail=override["reason"],
                        as_of_date=self.provider.today.isoformat(),
                        signal_key=f"UNIFIED_{sector.upper().replace(' ', '_')}",
                        payload={"override": True, "confidence": 0.80},
                    )
                )

        unified.sort(key=lambda row: row.score, reverse=True)
        self.repo.replace_signals("unified", unified)
        return {
            "unified_signals": [asdict(row) for row in unified],
            "run_summary": {
                "family_counts": {
                    "geo": len(geo),
                    "policy": len(policy),
                    "flow": len(flow),
                    "contrarian": len(contrarian),
                    "unified": len(unified),
                }
            },
        }
