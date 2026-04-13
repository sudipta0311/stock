from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Neon (PostgreSQL) helpers
# ─────────────────────────────────────────────────────────────────────────────

def _neon_url(neon_url: str | None = None) -> str:
    return (neon_url if neon_url is not None else os.getenv("NEON_DATABASE_URL", "")).strip()


def neon_enabled(neon_url: str | None = None) -> bool:
    return bool(_neon_url(neon_url))


# ─────────────────────────────────────────────────────────────────────────────
# NeonWrapper — makes psycopg2 look like sqlite3 to all callers
#
# Differences resolved by the wrapper:
#   • ? placeholders  →  %s   (psycopg2 style)
#   • AUTOINCREMENT   →  SERIAL  (handled in schema.py per-dialect)
#   • row_factory     →  RealDictCursor always used; row_factory attr is a no-op
#   • fetchone/fetchall return _NeonRow objects that support both
#     row["key"]  and  dict(row)  — same interface as sqlite3.Row
# ─────────────────────────────────────────────────────────────────────────────

class _NeonRow:
    """Dict-like row compatible with sqlite3.Row."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data.values())

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    # Needed so dict(row) works the same as on sqlite3.Row
    def __len__(self) -> int:
        return len(self._data)


class _NeonCursor:
    """Wraps a psycopg2 cursor; fetchone/fetchall return _NeonRow objects."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    @property
    def description(self):
        return self._cur.description

    def fetchone(self) -> _NeonRow | None:
        row = self._cur.fetchone()
        return _NeonRow(dict(row)) if row is not None else None

    def fetchall(self) -> list[_NeonRow]:
        return [_NeonRow(dict(r)) for r in (self._cur.fetchall() or [])]


class NeonWrapper:
    """
    Wraps a psycopg2 connection to present the same interface as sqlite3.

    Key adaptations
    ───────────────
    • execute / executemany  : replace ? with %s before forwarding
    • row_factory attribute  : accepted but ignored (RealDictCursor used always)
    • dialect attribute      : 'postgresql' — lets schema.py generate PG DDL
    """

    dialect: str = "postgresql"
    row_factory: Any = None  # accept writes, do nothing

    def __init__(self, pg_conn: Any) -> None:
        self._conn = pg_conn

    @staticmethod
    def _pg(sql: str) -> str:
        """Convert SQLite ? placeholders to psycopg2 %s."""
        return sql.replace("?", "%s")

    def execute(self, sql: str, params: tuple | list = ()) -> _NeonCursor:
        from psycopg2.extras import RealDictCursor  # lazy import

        cur = self._conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(self._pg(sql), params if params else None)
        return _NeonCursor(cur)

    def executemany(self, sql: str, params_list: Any) -> None:
        cur = self._conn.cursor()
        cur.executemany(self._pg(sql), params_list)

    def cursor(self) -> Any:
        return self._conn.cursor()

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def rollback(self) -> None:
        self._conn.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Public connection API
# ─────────────────────────────────────────────────────────────────────────────

def connect_database(
    db_path: str | Path,
    *,
    neon_url: str | None = None,
    # Legacy Turso params — accepted for backward compat, silently ignored.
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> Any:
    """
    Return a database connection.

    Priority order
    ──────────────
    1. Neon (psycopg2) if NEON_DATABASE_URL is set or neon_url is given
    2. Local SQLite at db_path

    The returned object always presents the same interface:
      .execute(sql, params)  /  .executemany(sql, params_list)
      .commit()  /  .close()
      .row_factory (no-op on PG)
      .dialect   : 'postgresql' | 'sqlite'  (for schema.py)
    """
    url = _neon_url(neon_url)

    if url:
        try:
            import psycopg2  # type: ignore

            pg_conn = psycopg2.connect(url)
            pg_conn.autocommit = False
            print("DB: connected to Neon PostgreSQL")
            return NeonWrapper(pg_conn)
        except Exception as exc:
            print(
                f"WARNING: Neon connection failed ({exc}). "
                "Falling back to local SQLite."
            )

    # ── SQLite fallback ──────────────────────────────────────────────────────
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.dialect = "sqlite"  # type: ignore[attr-defined]
    print(f"DB: using local SQLite at {path}")
    return conn


def sync_database(connection: Any) -> None:
    """No-op for Neon; kept for API compatibility."""
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


@contextmanager
def database_connection(
    db_path: str | Path,
    *,
    neon_url: str | None = None,
    # Legacy Turso params — ignored.
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> Iterator[Any]:
    connection = connect_database(
        db_path,
        neon_url=neon_url,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    finally:
        connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Turso helpers — kept so existing imports don't break
# ─────────────────────────────────────────────────────────────────────────────

def turso_enabled(
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> bool:
    """Deprecated — always returns False. Use neon_enabled() instead."""
    return False
