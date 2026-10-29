"""Add owner-written research runs, dossiers, and audit records."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

revision = "0016_research"
down_revision = "0015_highlight_dismissals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.create_table(
        "research_runs",
        Column("run_id", Uuid(as_uuid=True), primary_key=True),
        Column("week", String(8), nullable=False),
        Column("mode", String(24), nullable=False),
        Column("stage", String(32), nullable=False, server_default="QUEUED"),
        Column("status", String(24), nullable=False, server_default="queued"),
        Column("budget_usd", Numeric(12, 8), nullable=False),
        Column("spent_usd", Numeric(12, 8), nullable=False, server_default="0"),
        Column("started_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("finished_at", DateTime(timezone=True)),
        CheckConstraint("week ~ '^[0-9]{4}-W[0-9]{2}$'", name="ck_research_runs_week"),
        CheckConstraint(
            "status IN ('queued', 'running', 'waiting', 'completed', 'budget_exceeded', 'failed')",
            name="ck_research_runs_status",
        ),
        CheckConstraint("budget_usd >= 0 AND spent_usd >= 0", name="ck_research_runs_budget"),
    )
    op.create_index("ix_research_runs_week_mode", "research_runs", ["week", "mode"])

    op.create_table(
        "dossiers",
        Column("id", String(128), primary_key=True),
        Column("week", String(8), nullable=False),
        Column("security_id", Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="SET NULL")),
        Column("stage", String(32), nullable=False),
        Column("json", JSONB, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("week ~ '^[0-9]{4}-W[0-9]{2}$'", name="ck_dossiers_week"),
        CheckConstraint(
            "id ~ '^[0-9]{4}-W[0-9]{2}/(IN|US|DK|EU|GB)/[A-Za-z0-9._-]+$'",
            name="ck_dossiers_id",
        ),
        CheckConstraint(
            "stage IN ('QUEUED', 'SCREENED', 'ANALYSED', 'VALUED', 'RED_TEAM_DONE', "
            "'COMPLIANCE_DONE', 'APPROVED', 'VETOED', 'UNRESOLVED')",
            name="ck_dossiers_stage",
        ),
    )
    op.create_index("ix_dossiers_week_stage", "dossiers", ["week", "stage"])

    op.create_table(
        "llm_calls",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "run_id", Uuid(as_uuid=True), ForeignKey("research_runs.run_id", ondelete="CASCADE")
        ),
        Column("dossier_id", String(128), ForeignKey("dossiers.id", ondelete="SET NULL")),
        Column("agent", String(64), nullable=False),
        Column("tier", String(24), nullable=False),
        Column("deployment", String(100), nullable=False),
        Column("service_tier", String(24)),
        Column("prompt_sha256", String(64), nullable=False),
        Column("request_json", JSONB, nullable=False),
        Column("response_json", JSONB),
        Column("usage", JSONB),
        Column("cost_usd", Numeric(12, 8), nullable=False, server_default="0"),
        Column("latency_ms", Integer, nullable=False, server_default="0"),
        Column("status", String(24), nullable=False),
        Column("error_code", String(80)),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("prompt_sha256 ~ '^[0-9a-f]{64}$'", name="ck_llm_calls_prompt_sha256"),
        CheckConstraint("cost_usd >= 0 AND latency_ms >= 0", name="ck_llm_calls_cost_latency"),
    )
    op.create_index("ix_llm_calls_run_created", "llm_calls", ["run_id", "created_at"])

    op.create_table(
        "tool_calls",
        Column("tool_call_id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "run_id", Uuid(as_uuid=True), ForeignKey("research_runs.run_id", ondelete="CASCADE")
        ),
        Column("dossier_id", String(128), ForeignKey("dossiers.id", ondelete="SET NULL")),
        Column("tool", String(64), nullable=False),
        Column("args_json", JSONB, nullable=False),
        Column("result_json", JSONB),
        Column("result_sha256", String(64)),
        Column("source_url", String(2048)),
        Column("retrieved_at", DateTime(timezone=True), nullable=False),
        Column("status", String(24), nullable=False),
        Column("error_code", String(80)),
        CheckConstraint(
            "result_sha256 IS NULL OR result_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_tool_calls_result_sha256",
        ),
    )
    op.create_index("ix_tool_calls_run_retrieved", "tool_calls", ["run_id", "retrieved_at"])

    research_tables = ("research_runs", "llm_calls", "tool_calls")
    quoted_tables = ", ".join(f"{quoted_schema}.{table}" for table in research_tables)
    op.execute(f"REVOKE ALL ON TABLE {quoted_tables} FROM PUBLIC, wealth_app, wealth_api")
    op.execute(f"REVOKE ALL ON TABLE {quoted_schema}.dossiers FROM PUBLIC, wealth_app, wealth_api")
    op.execute(f"GRANT SELECT ON TABLE {quoted_schema}.dossiers TO wealth_app")


def downgrade() -> None:
    op.drop_index("ix_tool_calls_run_retrieved", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("ix_llm_calls_run_created", table_name="llm_calls")
    op.drop_table("llm_calls")
    op.drop_index("ix_dossiers_week_stage", table_name="dossiers")
    op.drop_table("dossiers")
    op.drop_index("ix_research_runs_week_mode", table_name="research_runs")
    op.drop_table("research_runs")
