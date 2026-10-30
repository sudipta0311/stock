"""Database engine and transaction-scoped session helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session


def normalize_database_url(url: str) -> str:
    """Return a synchronous psycopg URL for a PostgreSQL connection string."""
    parsed = make_url(url)
    if parsed.drivername in {"postgres", "postgresql"}:
        parsed = parsed.set(drivername="postgresql+psycopg")
    elif parsed.drivername != "postgresql+psycopg":
        raise ValueError("Database URL must use PostgreSQL with the psycopg driver")
    return parsed.render_as_string(hide_password=False)


def make_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine from an explicit URL or ``DATABASE_URL``."""
    database_url = url or os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Set DATABASE_URL before creating a database engine")
    return create_engine(normalize_database_url(database_url), pool_pre_ping=True)


def api_engine(url: str | None = None) -> Engine:
    """Create an API engine only for a non-owner login with safe RLS capabilities.

    Owner credentials belong only in ``OWNER_DATABASE_URL`` for migrations and scheduled jobs.
    """
    engine = make_engine(url)
    capabilities_query = text(
        """
        SELECT role.rolsuper AS is_superuser,
               role.rolbypassrls AS bypasses_rls,
               EXISTS (
                   SELECT 1
                   FROM pg_catalog.pg_class AS relation
                   JOIN pg_catalog.pg_namespace AS namespace
                     ON namespace.oid = relation.relnamespace
                   WHERE namespace.nspname = pg_catalog.current_schema()
                     AND relation.relkind IN ('r', 'p')
                     AND relation.relowner = role.oid
               ) AS owns_tables,
               pg_catalog.pg_has_role(current_user, 'wealth_app', 'MEMBER') AS app_member
        FROM pg_catalog.pg_roles AS role
        WHERE role.rolname = current_user
        """
    )
    try:
        with engine.connect() as connection:
            capabilities = connection.execute(capabilities_query).mappings().one_or_none()
    except Exception as error:
        engine.dispose()
        hint = _safe_connection_error_hint(error, url or os.getenv("DATABASE_URL"))
        raise ValueError(f"Unable to verify API database role capabilities ({hint})") from None

    if (
        capabilities is None
        or capabilities["is_superuser"]
        or capabilities["bypasses_rls"]
        or capabilities["owns_tables"]
        or not capabilities["app_member"]
    ):
        engine.dispose()
        raise ValueError(
            "API database login must be non-superuser, not BYPASSRLS, own no tables in its "
            "current schema, and be a member of wealth_app"
        )
    return engine


def _safe_connection_error_hint(error: Exception, database_url: str | None) -> str:
    original = getattr(error, "orig", error)
    diagnostic = getattr(original, "diag", None)
    message = getattr(diagnostic, "message_primary", None) or str(original)
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)

    message = re.sub(
        r"(?i)\b(?:postgres(?:ql)?(?:\+\w+)?|mysql(?:\+\w+)?|sqlite)://[^\s'\"]+",
        "[database URL removed]",
        message,
    )
    message = re.sub(
        r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*[^\s,;]+", "[credential removed]", message
    )
    message = re.sub(r"(?i)\bhost\s*[:=]\s*[^\s,;]+", "host=[host removed]", message)

    if database_url:
        try:
            parsed = make_url(database_url)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            for sensitive_value, replacement in (
                (parsed.password, "[credential removed]"),
                (parsed.host, "[host removed]"),
            ):
                if sensitive_value:
                    message = message.replace(sensitive_value, replacement)

    message = " ".join(message.split())[:240]
    exception_name = type(error).__name__
    state_hint = f", SQLSTATE {sqlstate}" if sqlstate else ""
    if message:
        return f"{exception_name}{state_hint}: {message}"
    return f"{exception_name}{state_hint}"


def resolve_user(
    engine: Engine,
    auth_subject: str,
    email: str,
    display_name: str | None,
) -> UUID:
    """Resolve or create the authenticated user through the restricted database function."""
    statement = text("SELECT app_resolve_user(:auth_subject, :email, :display_name)")
    with engine.begin() as connection:
        user_id = connection.scalar(
            statement,
            {
                "auth_subject": auth_subject,
                "email": email,
                "display_name": display_name,
            },
        )
    if user_id is None:
        raise RuntimeError("app_resolve_user returned no user id")
    return user_id if isinstance(user_id, UUID) else UUID(str(user_id))


def session_user(engine: Engine, session_id: UUID, auth_subject: str) -> UUID | None:
    """The user behind a live app session, or None; never creates a user."""
    statement = text("SELECT app_session_user(:session_id, :auth_subject)")
    with engine.begin() as connection:
        user_id = connection.scalar(
            statement, {"session_id": session_id, "auth_subject": auth_subject}
        )
    if user_id is None:
        return None
    return user_id if isinstance(user_id, UUID) else UUID(str(user_id))


@contextmanager
def user_session(
    engine: Engine, user_id: UUID, household_owner_id: UUID | None = None
) -> Iterator[Session]:
    """Yield a transaction scoped to a login and the household it selected."""
    selected_household = household_owner_id or user_id
    with Session(engine) as session, session.begin():
        session.execute(text("SET LOCAL ROLE wealth_app"))
        session.execute(
            text("SELECT set_config('app.user_id', :user_id, true)"),
            {"user_id": str(user_id)},
        )
        session.execute(
            text("SELECT set_config('app.household_owner', :owner_id, true)"),
            {"owner_id": str(selected_household)},
        )
        yield session


@contextmanager
def system_session(engine: Engine) -> Iterator[Session]:
    """Yield an owner-role transaction for migrations and scheduled jobs only.

    This session bypasses row-level security and can access data across all users. It must never be
    used by the API.
    """
    with Session(engine) as session, session.begin():
        yield session
