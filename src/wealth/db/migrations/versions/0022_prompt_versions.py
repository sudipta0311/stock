"""Record the versioned prompt hash for every model call."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, String, text

revision = "0022_prompt_versions"
down_revision = "0021_screener_sentinel"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_calls", Column("prompt_version_sha256", String(64), nullable=True))
    op.execute(
        text(
            "UPDATE llm_calls SET prompt_version_sha256 = prompt_sha256 "
            "WHERE prompt_version_sha256 IS NULL"
        )
    )
    op.alter_column("llm_calls", "prompt_version_sha256", nullable=False)
    op.create_check_constraint(
        "ck_llm_calls_prompt_version_sha256",
        "llm_calls",
        "prompt_version_sha256 ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_llm_calls_prompt_version_sha256", "llm_calls", type_="check")
    op.drop_column("llm_calls", "prompt_version_sha256")
