"""Typed access to the versioned model and price configuration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "rules" / "models.yaml"


@dataclass(frozen=True)
class TierConfig:
    deployment: str
    reasoning_effort: str | None = None
    max_input_tokens: int | None = None
    fallback_tier: str | None = None


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    output: Decimal
    cached_input: Decimal | None = None
    flex_factor: Decimal = Decimal("1")


@dataclass(frozen=True)
class FlexConfig:
    enabled_for_modes: tuple[str, ...]
    max_flex_retries: int
    fallback_to_default: bool


@dataclass(frozen=True)
class ModelConfig:
    tiers: dict[str, TierConfig]
    prices_usd_per_1m: dict[str, ModelPrice]
    flex: FlexConfig
    connect_timeout_s: float
    read_timeout_s: float
    flex_read_timeout_s: float = 900

    @classmethod
    def load(cls, path: Path | None = None) -> ModelConfig:
        config_path = path or _DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tiers = {
            name: TierConfig(
                deployment=str(values["deployment"]),
                reasoning_effort=(
                    str(values["reasoning_effort"])
                    if values.get("reasoning_effort") is not None
                    else None
                ),
                max_input_tokens=(
                    int(values["max_input_tokens"])
                    if values.get("max_input_tokens") is not None
                    else None
                ),
                fallback_tier=(
                    str(values["fallback_tier"])
                    if values.get("fallback_tier") is not None
                    else None
                ),
            )
            for name, values in raw["tiers"].items()
        }
        for name, tier in tiers.items():
            if tier.fallback_tier is not None and tier.fallback_tier not in tiers:
                raise ValueError(f"Tier {name} falls back to an unknown tier")
        prices = {
            name: ModelPrice(
                input=Decimal(str(values["input"])),
                output=Decimal(str(values["output"])),
                cached_input=(
                    Decimal(str(values["cached_input"]))
                    if values.get("cached_input") is not None
                    else None
                ),
                flex_factor=Decimal(str(values.get("flex_factor", "1"))),
            )
            for name, values in raw["prices_usd_per_1m"].items()
        }
        flex = raw["flex"]
        timeouts = raw["timeouts_s"]
        return cls(
            tiers=tiers,
            prices_usd_per_1m=prices,
            flex=FlexConfig(
                enabled_for_modes=tuple(str(value) for value in flex["enabled_for_modes"]),
                max_flex_retries=int(flex["max_flex_retries"]),
                fallback_to_default=bool(flex["fallback_to_default"]),
            ),
            connect_timeout_s=float(timeouts["connect"]),
            read_timeout_s=float(timeouts["read"]),
            flex_read_timeout_s=float(timeouts.get("flex_read", 900)),
        )
