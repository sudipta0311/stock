from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from dotenv import load_dotenv


def load_app_env(root_dir: Path | None = None) -> None:
    env_root = root_dir or Path(__file__).resolve().parents[2]
    load_dotenv(env_root / ".env", override=False)
    secrets_path = env_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as handle:
            secrets = tomllib.load(handle)
        for key, value in secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "platform.db")
    max_portfolio_age_days: int = 35
    max_direct_stocks: int = 4
    max_single_stock_pct: float = 30.0
    max_sector_pct: float = 30.0
    total_direct_equity_pct_cap: float = 25.0

    # ── Anthropic Claude ───────────────────────────────────────────────────
    # Fast tier  (Haiku 4.5)  : buy/monitoring rationale — high-volume, prompt-cached.
    # Reasoning tier (Sonnet 4.6) : industry, validation, thesis — complex reasoning.
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_fast_model: str = field(
        default_factory=lambda: os.getenv("LLM_FAST_MODEL", "claude-haiku-4-5-20251001")
    )
    llm_reasoning_model: str = field(
        default_factory=lambda: os.getenv("LLM_REASONING_MODEL", "claude-sonnet-4-6")
    )

    # ── OpenAI GPT ─────────────────────────────────────────────────────────
    # Fast tier      : gpt-5.4-mini  - lower-latency rationale generation.
    # Reasoning tier : gpt-5.4       - latest flagship model for complex analysis.
    #                  Override with OPENAI_REASONING_MODEL if you want a different tier.
    #                  reasoning tasks (qualitative_analysis, thesis_review).
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_fast_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_FAST_MODEL", "gpt-5.4-mini")
    )
    openai_reasoning_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_REASONING_MODEL", "gpt-5.4")
    )

    mf_api_base_url: str = field(default_factory=lambda: os.getenv("MF_API_BASE_URL", "https://mfdata.in/api/v1"))
    mf_holdings_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("MF_HOLDINGS_TIMEOUT_SECONDS", "20")))

    @property
    def anthropic_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def llm_enabled(self) -> bool:
        """True if at least one provider has a configured key."""
        return self.anthropic_enabled or self.openai_enabled


def ensure_data_dir(config: AppConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
