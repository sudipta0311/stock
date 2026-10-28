"""Add alert audit, delivery reminders, and shared urgent event data."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_alerts_delivery"
down_revision = "0007_personalisation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reminders", Column("notify_on", Date(), nullable=True))
    op.add_column(
        "reminders",
        Column("payload", JSONB(), nullable=False, server_default=text("'{}'::jsonb")),
    )
    op.create_unique_constraint(
        "uq_reminders_delivery_dedupe", "reminders", ["user_id", "asset_id", "due_on", "kind"]
    )
    op.create_index("ix_reminders_notify_on", "reminders", ["user_id", "notify_on", "sent_at"])

    op.create_table(
        "alerts_log",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("kind", String(24), nullable=False),
        Column("week", String(8)),
        Column("dedupe_key", String(160), nullable=False),
        Column("subject", Text(), nullable=False),
        Column("provider_message_id", Text()),
        Column("body_sha256", String(64), nullable=False),
        Column("sent_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        UniqueConstraint("user_id", "id", name="uq_alerts_log_user_id_id"),
        UniqueConstraint("user_id", "dedupe_key", name="uq_alerts_log_user_dedupe"),
        CheckConstraint("kind IN ('weekly', 'reminder', 'urgent')", name="ck_alerts_log_kind"),
        CheckConstraint("body_sha256 ~ '^[0-9a-f]{64}$'", name="ck_alerts_log_body_sha256"),
    )
    op.create_index("ix_alerts_log_user_sent", "alerts_log", ["user_id", "sent_at"])
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE alerts_log TO wealth_app")
    op.execute("ALTER TABLE alerts_log ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY alerts_log_user_isolation ON alerts_log FOR ALL TO wealth_app "
        "USING (user_id = nullif(current_setting('app.user_id', true), '')::uuid) "
        "WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid)"
    )

    op.create_table(
        "sentinel_events",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("event_code", String(48), nullable=False),
        Column("source", Text(), nullable=False),
        Column("evidence_tool_call_id", Uuid(as_uuid=True)),
        Column("detected_at", DateTime(timezone=True), nullable=False),
        Column("method", String(8), nullable=False),
        Column("urgent", Boolean(), nullable=False, server_default=text("false")),
        UniqueConstraint("security_id", "event_code", "source", name="uq_sentinel_event_source"),
        CheckConstraint("method IN ('rule', 'llm')", name="ck_sentinel_events_method"),
    )
    op.create_index(
        "ix_sentinel_events_urgent_detected", "sentinel_events", ["urgent", "detected_at"]
    )
    op.execute("GRANT SELECT ON TABLE sentinel_events TO wealth_app")
    op.execute("ALTER TABLE sentinel_events DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_table("sentinel_events")
    op.drop_table("alerts_log")
    op.drop_index("ix_reminders_notify_on", table_name="reminders")
    op.drop_constraint("uq_reminders_delivery_dedupe", "reminders", type_="unique")
    op.drop_column("reminders", "payload")
    op.drop_column("reminders", "notify_on")
