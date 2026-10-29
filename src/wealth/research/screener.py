"""Deterministic sector-neutral factor scoring and weekly shortlist persistence.

For a hand-checkable three-security sector, suppose raw ROCE values are 0.10, 0.20,
and 0.30. Population mean is 0.20 and standard deviation is sqrt(0.02 / 3), so the
sector z-scores are -1.224744871..., 0, and +1.224744871.... The other factors use
the same population z-score rule and are clipped to [-3, 3] before group means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any
from uuid import UUID, uuid4

import yaml
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from wealth.db.models import FundamentalPIT, PriceEod, Security, UniverseMember
from wealth.db.session import make_engine, system_session
from wealth.research.dossier import Claim, Dossier, DossierSecurity, Evidence, Stage
from wealth.research.factors import fundamental_factors, price_factors

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RULES = _ROOT / "rules" / "screener.yaml"
_FACTOR_NAMES = (
    "roce",
    "roe",
    "accruals_ratio",
    "debt_to_equity",
    "earnings_yield",
    "fcf_yield",
    "momentum_12_1",
    "volatility_1y",
)
_QUALITY_INPUTS = ("roce", "roe", "accruals_ratio", "debt_to_equity")
_VALUE_INPUTS = ("earnings_yield", "fcf_yield")
_PRICE_ONLY_INPUTS = ("momentum_12_1", "volatility_1y")
_NEGATIVE_INPUTS = {"accruals_ratio", "debt_to_equity", "volatility_1y"}
_UNKNOWN_SECTOR = "Unclassified"
_WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")
_MAX_PRICE_AGE_DAYS = 20


@dataclass(frozen=True)
class ScreenerInput:
    security_id: str
    symbol: str
    market: str
    sector: str | None
    factors: Mapping[str, Decimal | None]
    median_3m_traded_value: Decimal | None = None
    in_sp500: bool = False
    price_only: bool = False


@dataclass(frozen=True)
class ScreenedSecurity:
    security_id: str
    symbol: str
    sector: str
    composite: Decimal
    decile: int
    groups: Mapping[str, Decimal]
    factor_coverage: str
    reasons: tuple[str, ...]
    z_scores: Mapping[str, Decimal]
    peer_inputs: Mapping[str, tuple[tuple[str, Decimal], ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "security_id": self.security_id,
            "symbol": self.symbol,
            "sector": self.sector,
            "composite": format(self.composite, "f"),
            "decile": self.decile,
            "groups": {name: format(value, "f") for name, value in self.groups.items()},
            "factor_coverage": self.factor_coverage,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ScreeningResult:
    shortlist: tuple[ScreenedSecurity, ...]
    excluded: Mapping[str, str]
    ranked: tuple[ScreenedSecurity, ...]


@dataclass(frozen=True)
class ScreenerRules:
    max_per_market: int
    liquidity_floor: Mapping[str, Decimal]
    price_only_markets: frozenset[str]

    @classmethod
    def load(cls, path: Path | None = None) -> ScreenerRules:
        raw: dict[str, Any] = yaml.safe_load((path or _DEFAULT_RULES).read_text(encoding="utf-8"))
        floors = raw.get("liquidity_floor", {})
        return cls(
            max_per_market=int(raw.get("max_per_market", 25)),
            liquidity_floor={key: Decimal(str(value)) for key, value in floors.items()},
            price_only_markets=frozenset(
                str(value).upper() for value in raw.get("price_only_markets", [])
            ),
        )


def _decimal_mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _peer_key(row: ScreenerInput) -> str:
    sector = (row.sector or "").strip()
    if not sector or sector.casefold() == _UNKNOWN_SECTOR.casefold():
        return f"unclassified:{row.security_id}"
    return f"sector:{sector.casefold()}"


def _z_scores(
    rows: Sequence[ScreenerInput],
) -> tuple[dict[str, dict[str, Decimal]], dict[str, dict[str, tuple[tuple[str, Decimal], ...]]]]:
    groups: dict[str, list[ScreenerInput]] = defaultdict(list)
    for row in rows:
        groups[_peer_key(row)].append(row)

    scores: dict[str, dict[str, Decimal]] = {row.security_id: {} for row in rows}
    peers: dict[str, dict[str, tuple[tuple[str, Decimal], ...]]] = {
        row.security_id: {} for row in rows
    }
    for sector_rows in groups.values():
        for factor_name in _FACTOR_NAMES:
            available = [
                (row, row.factors.get(factor_name))
                for row in sector_rows
                if row.factors.get(factor_name) is not None
            ]
            if not available:
                continue
            values = [value for _row, value in available if value is not None]
            mean = _decimal_mean(values)
            variance = _decimal_mean([(value - mean) ** 2 for value in values])
            deviation = variance.sqrt()
            normalized_values = tuple(
                sorted(
                    ((row.symbol, value) for row, value in available if value is not None),
                    key=lambda item: item[0],
                )
            )
            for row, raw_value in available:
                assert raw_value is not None
                directed = -raw_value if factor_name in _NEGATIVE_INPUTS else raw_value
                directed_mean = -mean if factor_name in _NEGATIVE_INPUTS else mean
                z_value = (
                    (directed - directed_mean) / deviation
                    if len(values) > 1 and deviation > 0
                    else Decimal(0)
                )
                scores[row.security_id][factor_name] = min(Decimal(3), max(Decimal(-3), z_value))
                peers[row.security_id][factor_name] = normalized_values
    return scores, peers


def _factor_groups(
    z_scores: Mapping[str, Decimal], *, price_only: bool, fundamentals_only: bool
) -> dict[str, Decimal]:
    if price_only:
        return {
            group: z_scores[name]
            for group, name in (
                ("momentum", "momentum_12_1"),
                ("low_vol", "volatility_1y"),
            )
            if name in z_scores
        }
    groups: dict[str, Decimal] = {}
    quality = [z_scores[name] for name in _QUALITY_INPUTS if name in z_scores]
    value = [z_scores[name] for name in _VALUE_INPUTS if name in z_scores]
    if quality:
        groups["quality"] = _decimal_mean(quality)
    if value:
        groups["value"] = _decimal_mean(value)
    if not fundamentals_only:
        if "momentum_12_1" in z_scores:
            groups["momentum"] = z_scores["momentum_12_1"]
        if "volatility_1y" in z_scores:
            groups["low_vol"] = z_scores["volatility_1y"]
    return groups


def _coverage(row: ScreenerInput, groups: Mapping[str, Decimal]) -> str | None:
    if row.market.upper() == "US":
        return "fundamentals_only" if {"quality", "value"} <= groups.keys() else None
    if row.price_only:
        return "price_only" if {"momentum", "low_vol"} <= groups.keys() else None
    return "full" if len(groups) >= 3 else None


def _decile(rank: int, population: int) -> int:
    return min(10, ((population - rank + 1) * 10 + population - 1) // population)


def screen_universe(
    rows: Sequence[ScreenerInput],
    *,
    market: str,
    max_per_market: int = 25,
    liquidity_floor: Decimal | None = None,
    price_only: bool = False,
) -> ScreeningResult:
    """Rank eligible securities by sector-adjusted group means, breaking ties by symbol."""
    normalized = [
        ScreenerInput(
            security_id=row.security_id,
            symbol=row.symbol,
            market=market.upper(),
            sector=row.sector or _UNKNOWN_SECTOR,
            factors=row.factors,
            median_3m_traded_value=row.median_3m_traded_value,
            in_sp500=row.in_sp500,
            price_only=market.upper() != "US" and (price_only or row.price_only),
        )
        for row in rows
    ]
    z_by_id, peer_by_id = _z_scores(normalized)
    eligible: list[ScreenedSecurity] = []
    excluded: dict[str, str] = {}

    for row in normalized:
        groups = _factor_groups(
            z_by_id[row.security_id],
            price_only=row.price_only,
            fundamentals_only=market.upper() == "US",
        )
        coverage = _coverage(row, groups)
        if coverage is None:
            excluded[row.security_id] = "insufficient_data"
            continue

        reasons: list[str] = []
        sector = (row.sector or _UNKNOWN_SECTOR).strip() or _UNKNOWN_SECTOR
        if sector.casefold() == _UNKNOWN_SECTOR.casefold():
            reasons.append("Unclassified sector; compared only with this security.")
        if market.upper() == "US" and row.in_sp500:
            reasons.append("Current S&P 500 membership satisfies the liquidity floor.")
        elif liquidity_floor is not None:
            traded_value = row.median_3m_traded_value
            if traded_value is None or traded_value < liquidity_floor:
                excluded[row.security_id] = "liquidity_below_floor"
                continue
            reasons.append("Three-month median traded value satisfies the liquidity floor.")
        composite = _decimal_mean(list(groups.values()))
        if coverage == "price_only":
            reasons.append("Price factors are available; fundamental inputs are not required.")
        elif coverage == "fundamentals_only":
            reasons.append("Quality and value groups use available company fundamentals.")
        eligible.append(
            ScreenedSecurity(
                security_id=row.security_id,
                symbol=row.symbol,
                sector=sector,
                composite=composite,
                decile=0,
                groups=groups,
                factor_coverage=coverage,
                reasons=tuple(reasons),
                z_scores=z_by_id[row.security_id],
                peer_inputs=peer_by_id[row.security_id],
            )
        )

    eligible.sort(key=lambda item: (-item.composite, item.symbol, item.security_id))
    ranked = tuple(
        ScreenedSecurity(
            security_id=item.security_id,
            symbol=item.symbol,
            sector=item.sector,
            composite=item.composite,
            decile=_decile(rank, len(eligible)),
            groups=item.groups,
            factor_coverage=item.factor_coverage,
            reasons=item.reasons,
            z_scores=item.z_scores,
            peer_inputs=item.peer_inputs,
        )
        for rank, item in enumerate(eligible, start=1)
    )
    shortlist = tuple(item for item in ranked if item.decile == 10)[: max(0, max_per_market)]
    return ScreeningResult(shortlist=shortlist, excluded=excluded, ranked=ranked)


def _median_traded_value(
    prices: Sequence[tuple[date, Decimal, int | None, Decimal | None]], as_of: date
) -> Decimal | None:
    cutoff = as_of - timedelta(days=90)
    values = [
        traded_value if traded_value is not None else close * volume
        for on, close, volume, traded_value in prices
        if cutoff <= on <= as_of and (traded_value is not None or volume is not None)
    ]
    return Decimal(str(median(values))) if values else None


def _factor_inputs(
    session: Session, security: Security, as_of: date
) -> tuple[ScreenerInput, list[str], dict[str, Any]]:
    cutoff = as_of - timedelta(days=400)
    price_rows = session.execute(
        select(PriceEod.as_of, PriceEod.close, PriceEod.volume, PriceEod.traded_value)
        .where(
            PriceEod.security_id == security.id,
            PriceEod.as_of >= cutoff,
            PriceEod.as_of <= as_of,
        )
        .order_by(PriceEod.as_of)
    ).all()
    facts = list(
        session.scalars(
            select(FundamentalPIT)
            .where(
                FundamentalPIT.security_id == security.id,
                FundamentalPIT.period_end <= as_of,
                FundamentalPIT.known_at <= datetime.combine(as_of, time.max, tzinfo=UTC),
            )
            .order_by(FundamentalPIT.period_end, FundamentalPIT.known_at)
        )
    )
    price = price_rows[-1].close if price_rows else None
    price_date = price_rows[-1].as_of if price_rows else None
    if price_date is not None and (as_of - price_date).days > _MAX_PRICE_AGE_DAYS:
        price = None
    fundamentals = fundamental_factors(facts, as_of=as_of, price=price)
    values: dict[str, Decimal | None] = {
        name: getattr(fundamentals, name) for name in _FACTOR_NAMES if hasattr(fundamentals, name)
    }
    if security.market == "IN":
        prices = [(row.as_of, row.close) for row in price_rows]
        derived = price_factors(prices, as_of)
        values["momentum_12_1"] = derived.momentum_12_1
        values["volatility_1y"] = derived.volatility_1y
    else:
        values["momentum_12_1"] = None
        values["volatility_1y"] = None

    source_urls = list(dict.fromkeys(row.source_url for row in facts))
    price_sources: list[str] = []
    if price_rows:
        price_sources = list(
            dict.fromkeys(
                session.scalars(
                    select(PriceEod.source_url).where(
                        PriceEod.security_id == security.id,
                        PriceEod.as_of == price_rows[-1].as_of,
                    )
                )
            )
        )
        source_urls.extend(price_sources)
    snapshot_inputs = {
        "as_of": as_of.isoformat(),
        "market": security.market,
        "security_id": str(security.id),
        "symbol": security.symbol,
        "sector": security.sector or _UNKNOWN_SECTOR,
        "industry": security.industry or _UNKNOWN_SECTOR,
        "raw_factors": {
            name: (format(value, "f") if value is not None else None)
            for name, value in values.items()
        },
        "price_as_of": price_date.isoformat() if price_date else None,
        "price_used_for_value": format(price, "f") if price is not None else None,
        "median_3m_traded_value": (
            format(_median_traded_value(price_rows, as_of), "f")
            if _median_traded_value(price_rows, as_of) is not None
            else None
        ),
        "fundamental_rows": [
            {
                "metric": row.metric,
                "period_end": row.period_end.isoformat(),
                "period_type": row.period_type,
                "value": format(row.value, "f"),
                "known_at": row.known_at.isoformat(),
                "source_url": row.source_url,
            }
            for row in facts
        ],
        "price_sources": price_sources,
        "source_urls": list(dict.fromkeys(source_urls)),
    }
    row = ScreenerInput(
        security_id=str(security.id),
        symbol=security.symbol or "",
        market=security.market,
        sector=security.sector,
        factors=values,
        median_3m_traded_value=_median_traded_value(price_rows, as_of),
    )
    return row, source_urls, snapshot_inputs


def _log_snapshot(engine: Engine, snapshot: dict[str, Any]) -> UUID:
    call_id = uuid4()
    raw = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    args = {key: snapshot[key] for key in ("as_of", "market", "security_id") if key in snapshot}
    raw_args = json.dumps(args, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with system_session(engine) as session:
        session.execute(
            text(
                "INSERT INTO tool_calls "
                "(tool_call_id, run_id, dossier_id, tool, args_json, result_json, "
                "result_sha256, source_url, retrieved_at, status, error_code) VALUES "
                "(:id, NULL, NULL, 'screener_snapshot', CAST(:args AS jsonb), "
                "CAST(:payload AS jsonb), :sha256, NULL, :retrieved_at, 'success', NULL)"
            ),
            {
                "id": call_id,
                "args": raw_args,
                "payload": raw,
                "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "retrieved_at": datetime.now(UTC),
            },
        )
    return call_id


def _save_pick(
    engine: Engine,
    pick: ScreenedSecurity,
    inputs: ScreenerInput,
    snapshot_inputs: dict[str, Any],
    *,
    week: str,
    as_of: date,
) -> None:
    exact_inputs = {
        **snapshot_inputs,
        "groups": {name: format(value, "f") for name, value in pick.groups.items()},
        "z_scores": {name: format(value, "f") for name, value in pick.z_scores.items()},
        "peer_inputs": {
            name: [[symbol, format(value, "f")] for symbol, value in peers]
            for name, peers in pick.peer_inputs.items()
        },
        "composite": format(pick.composite, "f"),
        "decile": pick.decile,
        "factor_coverage": pick.factor_coverage,
        "reasons": list(pick.reasons),
        "liquidity": {
            "median_3m_traded_value": (
                format(inputs.median_3m_traded_value, "f")
                if inputs.median_3m_traded_value is not None
                else None
            ),
            "in_sp500": inputs.in_sp500,
        },
    }
    call_id = _log_snapshot(engine, exact_inputs)
    group_summary = ", ".join(f"{name} {value:.4f}" for name, value in sorted(pick.groups.items()))
    claim_text = (
        f"Sector-adjusted factors: {group_summary}; composite {pick.composite:.4f}; "
        f"rank decile {pick.decile}."
    )
    dossier = Dossier(
        dossier_id=f"{week}/{inputs.market}/{inputs.symbol}",
        security=DossierSecurity(
            id=pick.security_id,
            market=inputs.market,
            type="EQUITY",
        ),
        stage=Stage.SCREENED,
        claims=[
            Claim(
                text=claim_text,
                kind="FACT",
                evidence=[Evidence(tool_call_id=str(call_id))],
            )
        ],
    )
    payload = json.dumps(dossier.model_dump(mode="json", by_alias=True), ensure_ascii=False)
    with system_session(engine) as session:
        session.execute(
            text(
                "INSERT INTO dossiers (id, week, security_id, stage, json) "
                "VALUES (:id, :week, :security_id, 'SCREENED', CAST(:payload AS jsonb)) "
                "ON CONFLICT (id) DO UPDATE SET security_id = EXCLUDED.security_id, "
                "stage = EXCLUDED.stage, json = EXCLUDED.json, updated_at = now() "
                "WHERE dossiers.stage IN ('QUEUED', 'SCREENED')"
            ),
            {
                "id": dossier.dossier_id,
                "week": week,
                "security_id": UUID(pick.security_id),
                "payload": payload,
            },
        )


def _load_market_rows(
    session: Session, as_of: date, rules: ScreenerRules
) -> tuple[dict[str, list[ScreenerInput]], dict[str, dict[str, Any]]]:
    members = session.execute(
        select(Security, UniverseMember.universe)
        .join(UniverseMember, UniverseMember.security_id == Security.id)
        .where(
            UniverseMember.removed_on.is_(None),
            Security.market.in_(("IN", "US")),
            Security.security_type == "equity",
            Security.symbol.is_not(None),
        )
        .order_by(Security.market, Security.symbol, UniverseMember.universe)
    ).all()
    securities_by_id: dict[UUID, Security] = {}
    universe_by_id: dict[UUID, set[str]] = defaultdict(set)
    for security, universe in members:
        securities_by_id[security.id] = security
        universe_by_id[security.id].add(universe)
    rows_by_market: dict[str, list[ScreenerInput]] = {"IN": [], "US": []}
    snapshots: dict[str, dict[str, Any]] = {}
    for security in securities_by_id.values():
        row, _source_urls, snapshot = _factor_inputs(session, security, as_of)
        row = ScreenerInput(
            security_id=row.security_id,
            symbol=row.symbol,
            market=row.market,
            sector=row.sector,
            factors=row.factors,
            median_3m_traded_value=row.median_3m_traded_value,
            in_sp500="sp500" in universe_by_id[security.id],
            price_only=security.market in rules.price_only_markets,
        )
        rows_by_market[security.market].append(row)
        snapshots[row.security_id] = snapshot
    return rows_by_market, snapshots


def run_screener(
    engine: Engine, week: str, *, rules: ScreenerRules | None = None
) -> dict[str, Any]:
    """Read dated shared inputs, rank each current universe, log evidence, and save dossiers."""
    parsed_week = _WEEK_PATTERN.fullmatch(week)
    if parsed_week is None:
        raise ValueError("week must use YYYY-Www")
    year, week_number = int(parsed_week.group(1)), int(parsed_week.group(2))
    as_of = date.fromisocalendar(year, week_number, 5)
    selected_rules = rules or ScreenerRules.load()
    with system_session(engine) as session:
        rows_by_market, snapshots = _load_market_rows(session, as_of, selected_rules)

    counts: dict[str, int] = {}
    saved: list[str] = []
    for market in ("IN", "US"):
        rows = rows_by_market[market]
        if not rows:
            counts[market] = 0
            continue
        result = screen_universe(
            rows,
            market=market,
            max_per_market=selected_rules.max_per_market,
            liquidity_floor=selected_rules.liquidity_floor.get(market),
        )
        by_id = {row.security_id: row for row in rows}
        for pick in result.shortlist:
            _save_pick(
                engine,
                pick,
                by_id[pick.security_id],
                snapshots[pick.security_id],
                week=week,
                as_of=as_of,
            )
            saved.append(pick.security_id)
        counts[market] = len(result.shortlist)
    return {"week": week, "as_of": as_of.isoformat(), "shortlisted": counts, "security_ids": saved}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank current stock universes without a model")
    parser.add_argument("--week", required=True, help="ISO week such as 2026-W44")
    args = parser.parse_args(argv)
    if not _WEEK_PATTERN.fullmatch(args.week):
        parser.error("--week must use YYYY-Www")
    try:
        date.fromisocalendar(int(args.week[:4]), int(args.week[-2:]), 5)
    except ValueError:
        parser.error("--week must be a valid ISO week")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    database_url = os.getenv("OWNER_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "unavailable", "reason": "database is not configured"}))
        return 2
    engine = make_engine(database_url)
    try:
        result = run_screener(engine, args.week)
    except Exception as error:
        print(json.dumps({"status": "unavailable", "reason": type(error).__name__}))
        return 1
    finally:
        engine.dispose()
    print(json.dumps({"status": "complete", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
