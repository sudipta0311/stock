from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _turso_settings(
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> tuple[str, str, int | None]:
    url = (turso_url if turso_url is not None else os.getenv("TURSO_DATABASE_URL", "")).strip()
    token = (turso_token if turso_token is not None else os.getenv("TURSO_AUTH_TOKEN", "")).strip()
    sync_interval_raw = str(sync_interval) if sync_interval is not None else os.getenv("TURSO_SYNC_INTERVAL_SECONDS", "").strip()
    try:
        sync_interval = int(sync_interval_raw) if sync_interval_raw else None
    except ValueError:
        sync_interval = None
    return url, token, sync_interval


def turso_enabled(
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> bool:
    url, token, _ = _turso_settings(turso_url, turso_token, sync_interval)
    return bool(url and token)


def connect_database(
    db_path: str | Path,
    *,
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> Any:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    url, token, sync_interval = _turso_settings(turso_url, turso_token, sync_interval)
    if url and token:
        try:
            import libsql
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Turso is configured but the `libsql` package is not installed. "
                "Install `libsql` in this environment, or disable TURSO_DATABASE_URL/TURSO_AUTH_TOKEN "
                "to fall back to local SQLite."
            ) from exc

        kwargs: dict[str, Any] = {
            "sync_url": url,
            "auth_token": token,
        }
        if sync_interval is not None and sync_interval > 0:
            kwargs["sync_interval"] = sync_interval
        connection = libsql.connect(str(path), **kwargs)
    else:
        connection = sqlite3.connect(path)

    if hasattr(connection, "row_factory"):
        connection.row_factory = sqlite3.Row
    return connection


def sync_database(connection: Any) -> None:
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


@contextmanager
def database_connection(
    db_path: str | Path,
    *,
    turso_url: str | None = None,
    turso_token: str | None = None,
    sync_interval: int | None = None,
) -> Iterator[Any]:
    connection = connect_database(
        db_path,
        turso_url=turso_url,
        turso_token=turso_token,
        sync_interval=sync_interval,
    )
    try:
        sync_database(connection)
        yield connection
        connection.commit()
        sync_database(connection)
    finally:
        connection.close()
