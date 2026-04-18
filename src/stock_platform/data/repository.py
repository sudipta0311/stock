from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from stock_platform.data.db import database_connection
from stock_platform.data.schema import initialize_schema
from stock_platform.models import MonitoringAction, RecommendationRecord, SignalRecord, utc_now_iso


class PlatformRepository:
    def __init__(
        self,
        db_path: Path,
        *,
        neon_database_url: str = "",
        # Legacy Turso params — accepted for backward compat, not used.
        turso_database_url: str = "",
        turso_auth_token: str = "",
        turso_sync_interval_seconds: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.neon_database_url = neon_database_url

    @contextmanager
    def connect(self) -> Iterator[Any]:
        with database_connection(
            self.db_path,
            neon_url=self.neon_database_url or None,
        ) as connection:
            yield connection

    def initialize(self) -> None:
        with self.connect() as connection:
            initialize_schema(connection)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        timestamp = utc_now_iso()
        payload = json.dumps(value)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO app_state(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, payload, timestamp),
            )

    def get_state(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM app_state WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return default or {}
        return json.loads(row["value_json"])

    def set_cache(self, cache_key: str, payload: Any, ttl_seconds: int | None = None) -> None:
        timestamp = utc_now_iso()
        expires_at = None
        if ttl_seconds is not None:
            expires_at = (
                datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries(cache_key, payload_json, updated_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at
                """,
                (cache_key, json.dumps(payload), timestamp, expires_at),
            )

    def get_cache(self, cache_key: str, default: Any = None, *, allow_expired: bool = False) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json, expires_at FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return default
        expires_at = row["expires_at"]
        if not allow_expired and expires_at:
            try:
                expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if expiry <= datetime.now(UTC):
                    return default
            except ValueError:
                return default
        return json.loads(row["payload_json"])

    def replace_raw_holdings(self, holding_type: str, rows: list[dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        with self.connect() as connection:
            connection.execute("DELETE FROM raw_holdings WHERE holding_type = ?", (holding_type,))
            connection.executemany(
                """
                INSERT INTO raw_holdings(
                    holding_type, instrument_name, symbol, quantity, market_value,
                    source, payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        holding_type,
                        row.get("instrument_name", ""),
                        row.get("symbol"),
                        row.get("quantity", 0),
                        row.get("market_value", 0),
                        row.get("source", holding_type),
                        json.dumps(row),
                        timestamp,
                    )
                    for row in rows
                ],
            )
        self.set_state("portfolio_meta", {"portfolio_last_updated": timestamp})

    def list_raw_holdings(self, holding_type: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM raw_holdings"
        params: tuple[Any, ...] = ()
        if holding_type:
            query += " WHERE holding_type = ?"
            params = (holding_type,)
        query += " ORDER BY instrument_name"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def replace_normalized_exposure(self, rows: list[dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        with self.connect() as connection:
            connection.execute("DELETE FROM normalized_exposure")
            connection.executemany(
                """
                INSERT INTO normalized_exposure(
                    symbol, company_name, sector, total_weight,
                    source_mix_json, attribution_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["symbol"],
                        row["company_name"],
                        row["sector"],
                        row["total_weight"],
                        json.dumps(row.get("source_mix", {})),
                        json.dumps(row.get("attribution", [])),
                        timestamp,
                    )
                    for row in rows
                ],
            )
        self.set_state("portfolio_meta", {"portfolio_last_updated": timestamp})

    def list_normalized_exposure(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM normalized_exposure ORDER BY total_weight DESC, symbol"
            ).fetchall()
        return [
            {
                **dict(row),
                "source_mix": json.loads(row["source_mix_json"]),
                "attribution": json.loads(row["attribution_json"]),
            }
            for row in rows
        ]

    def replace_overlap_scores(self, rows: list[dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        with self.connect() as connection:
            connection.execute("DELETE FROM overlap_scores")
            connection.executemany(
                """
                INSERT INTO overlap_scores(symbol, overlap_pct, band, attribution_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["symbol"],
                        row["overlap_pct"],
                        row["band"],
                        json.dumps(row.get("attribution", [])),
                        timestamp,
                    )
                    for row in rows
                ],
            )

    def list_overlap_scores(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM overlap_scores ORDER BY overlap_pct DESC, symbol"
            ).fetchall()
        return [
            {
                **dict(row),
                "attribution": json.loads(row["attribution_json"]),
            }
            for row in rows
        ]

    def replace_gaps(self, rows: list[dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        with self.connect() as connection:
            connection.execute("DELETE FROM identified_gaps")
            connection.executemany(
                """
                INSERT INTO identified_gaps(sector, underweight_pct, conviction, score, reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["sector"],
                        row["underweight_pct"],
                        row["conviction"],
                        row["score"],
                        row["reason"],
                        timestamp,
                    )
                    for row in rows
                ],
            )

    def list_gaps(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM identified_gaps ORDER BY score DESC, sector"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_direct_equity_holdings(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol, quantity, avg_buy_price, current_price, buy_date, source, updated_at
                FROM direct_equity
                ORDER BY symbol
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def portfolio_table_diagnostics(self) -> dict[str, Any]:
        with self.connect() as connection:
            dialect = getattr(connection, "dialect", "sqlite")
            if dialect == "postgresql":
                table_rows = connection.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """
                ).fetchall()
                table_names = [str(row["table_name"]) for row in table_rows]
            else:
                table_rows = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                ).fetchall()
                table_names = [str(row["name"]) for row in table_rows]

            counts: dict[str, int] = {}
            for table_name in ("direct_equity", "raw_holdings", "normalized_exposure", "overlap_scores"):
                if table_name not in table_names:
                    continue
                row = connection.execute(f"SELECT COUNT(*) AS row_count FROM {table_name}").fetchone()
                counts[table_name] = int(row["row_count"]) if row else 0

        return {
            "dialect": dialect,
            "tables": table_names,
            "counts": counts,
        }

    def replace_signals(self, family: str, rows: list[SignalRecord | dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        normalized_rows: list[SignalRecord] = []
        for row in rows:
            if isinstance(row, SignalRecord):
                normalized_rows.append(row)
            else:
                normalized_rows.append(SignalRecord(**row))
        with self.connect() as connection:
            connection.execute("DELETE FROM signals WHERE signal_family = ?", (family,))
            connection.executemany(
                """
                INSERT INTO signals(
                    signal_family, signal_key, sector, conviction, score,
                    source, horizon, detail, as_of_date, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.family,
                        item.signal_key,
                        item.sector,
                        item.conviction,
                        item.score,
                        item.source,
                        item.horizon,
                        item.detail,
                        item.as_of_date,
                        json.dumps(item.payload),
                        timestamp,
                    )
                    for item in normalized_rows
                ],
            )

    def list_signals(self, family: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM signals"
        params: tuple[Any, ...] = ()
        if family:
            query += " WHERE signal_family = ?"
            params = (family,)
        query += " ORDER BY score DESC, sector"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            (
                lambda base, payload: {
                    **base,
                    "payload": payload,
                    "overlap_pct": float(payload.get("overlap_pct", base.get("overlap_pct", 0.0)) or 0.0),
                }
            )(dict(row), json.loads(row["payload_json"]))
            for row in rows
        ]

    def save_recommendations(self, run_id: str, rows: list[RecommendationRecord | dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        normalized_rows: list[RecommendationRecord] = []
        for row in rows:
            if isinstance(row, RecommendationRecord):
                normalized_rows.append(row)
            else:
                normalized_rows.append(RecommendationRecord(**row))
        with self.connect() as connection:
            connection.execute("DELETE FROM recommendations WHERE run_id = ?", (run_id,))
            connection.executemany(
                """
                INSERT INTO recommendations(
                    run_id, symbol, company_name, sector, action, score,
                    confidence_band, rationale, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row.symbol,
                        row.company_name,
                        row.sector,
                        row.action,
                        row.score,
                        row.confidence_band,
                        row.rationale,
                        json.dumps(row.payload),
                        timestamp,
                    )
                    for row in normalized_rows
                ],
            )
        self.set_state("last_buy_run", {"run_id": run_id, "created_at": timestamp})

    def list_recommendations(self) -> list[dict[str, Any]]:
        run_meta = self.get_state("last_buy_run", {})
        run_id = run_meta.get("run_id")
        if not run_id:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM recommendations
                WHERE run_id = ?
                ORDER BY score DESC, symbol
                """,
                (run_id,),
            ).fetchall()
        return [
            (
                lambda base, payload: {
                    **base,
                    "payload": payload,
                    "overlap_pct": float(payload.get("overlap_pct", base.get("overlap_pct", 0.0)) or 0.0),
                }
            )(dict(row), json.loads(row["payload_json"]))
            for row in rows
        ]

    def clear_recommendations(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM recommendations")
        self.set_state("last_buy_run", {})

    def save_monitoring_actions(self, run_id: str, rows: list[MonitoringAction | dict[str, Any]]) -> None:
        timestamp = utc_now_iso()
        normalized_rows: list[MonitoringAction] = []
        for row in rows:
            if isinstance(row, MonitoringAction):
                normalized_rows.append(row)
            else:
                normalized_rows.append(MonitoringAction(**row))
        with self.connect() as connection:
            connection.execute("DELETE FROM monitoring_actions WHERE run_id = ?", (run_id,))
            connection.executemany(
                """
                INSERT INTO monitoring_actions(
                    run_id, symbol, action, severity, urgency, rationale, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row.symbol,
                        row.action,
                        row.severity,
                        row.urgency,
                        row.rationale,
                        json.dumps(row.payload),
                        timestamp,
                    )
                    for row in normalized_rows
                ],
            )
        self.set_state("last_monitor_run", {"run_id": run_id, "created_at": timestamp})

    def list_monitoring_actions(self) -> list[dict[str, Any]]:
        run_meta = self.get_state("last_monitor_run", {})
        run_id = run_meta.get("run_id")
        with self.connect() as connection:
            if not run_id:
                row = connection.execute(
                    "SELECT run_id FROM monitoring_actions ORDER BY created_at DESC, id DESC LIMIT 1"
                ).fetchone()
                if not row:
                    return []
                run_id = row["run_id"]
            rows = connection.execute(
                """
                SELECT
                    m.*,
                    COALESCE(o.overlap_pct, 0) AS overlap_pct
                FROM monitoring_actions m
                LEFT JOIN overlap_scores o
                    ON UPPER(TRIM(m.symbol)) = UPPER(TRIM(o.symbol))
                WHERE m.run_id = ?
                ORDER BY
                    CASE m.urgency
                        WHEN 'HIGH' THEN 1
                        WHEN 'MEDIUM' THEN 2
                        ELSE 3
                    END,
                    CASE m.severity
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'HIGH' THEN 2
                        WHEN 'MEDIUM' THEN 3
                        ELSE 4
                    END,
                    m.symbol
                """,
                (run_id,),
            ).fetchall()
        return [
            (
                lambda base, payload: {
                    **base,
                    "payload": payload,
                    "overlap_pct": float(payload.get("overlap_pct", base.get("overlap_pct", 0.0)) or 0.0),
                }
            )(dict(row), json.loads(row["payload_json"]))
            for row in rows
        ]

    def clear_monitoring_actions(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM monitoring_actions")
        self.set_state("last_monitor_run", {})

    def upsert_watchlist_stock(
        self, symbol: str, company_name: str, sector: str = "Unknown", note: str = ""
    ) -> None:
        timestamp = utc_now_iso()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO monitoring_watchlist(symbol, company_name, sector, note, added_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name=excluded.company_name,
                    sector=excluded.sector,
                    note=excluded.note,
                    added_at=excluded.added_at
                """,
                (symbol.upper(), company_name, sector, note, timestamp),
            )

    def remove_watchlist_stock(self, symbol: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM monitoring_watchlist WHERE symbol = ?",
                (symbol.upper(),),
            )

    def list_watchlist(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM monitoring_watchlist ORDER BY symbol"
            ).fetchall()
        return [dict(row) for row in rows]

    def load_portfolio_context(self) -> dict[str, Any]:
        portfolio_meta = self.get_state("portfolio_meta", {})
        return {
            "portfolio_meta": portfolio_meta,
            "raw_holdings": self.list_raw_holdings(),
            "normalized_exposure": self.list_normalized_exposure(),
            "overlap_scores": self.list_overlap_scores(),
            "identified_gaps": self.list_gaps(),
            "unified_signals": self.list_signals("unified"),
            "user_preferences": self.get_state("user_preferences", {}),
            "watchlist": self.list_watchlist(),
            "direct_equity_holdings": self.list_direct_equity_holdings(),
        }
