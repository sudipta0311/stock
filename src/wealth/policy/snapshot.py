"""Aggregate valued assets into whole-wealth and investable policy views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from wealth.core.money import round_paise
from wealth.schemas.assets import (
    AnyAsset,
    GoldAsset,
    LoanAsset,
    default_liquidity,
    default_sleeves,
)
from wealth.schemas.common import LiquidityClass, Sleeve
from wealth.valuation.base import Valuation

PolicySleeve = Literal["equity", "debt", "gold", "cash"]
POLICY_SLEEVES: tuple[PolicySleeve, ...] = ("equity", "debt", "gold", "cash")
ValuedAsset = tuple[AnyAsset, Valuation]
MarketHolding = tuple[Sleeve | str, Decimal, LiquidityClass | str]


@dataclass(frozen=True, init=False)
class WealthSnapshot:
    """Immutable totals for assets, liabilities, liquidity, and policy sleeves.

    ``valued_assets`` contains ``(asset, valuation)`` pairs. Market holdings are
    three-tuples of ``(sleeve, value, liquidity)`` and are already distinct from
    the manual assets in the first argument.
    """

    valued_assets: tuple[ValuedAsset, ...]
    market_holdings: tuple[MarketHolding, ...]
    totals_by_sleeve: Mapping[Sleeve, Decimal]
    totals_by_liquidity: Mapping[LiquidityClass, Decimal]
    locked_by_sleeve: Mapping[Sleeve, Decimal]
    totals_by_sleeve_and_liquidity: Mapping[tuple[Sleeve, LiquidityClass], Decimal]
    investable_totals_by_sleeve: Mapping[PolicySleeve, Decimal]
    liabilities_total: Decimal
    all_assets_total: Decimal
    investable_total: Decimal

    def __init__(
        self,
        valued_assets: Iterable[ValuedAsset] = (),
        market_holdings: Iterable[MarketHolding] = (),
    ) -> None:
        object.__setattr__(self, "valued_assets", tuple(valued_assets))
        object.__setattr__(self, "market_holdings", tuple(market_holdings))
        sleeve_totals = {sleeve: Decimal("0.00") for sleeve in Sleeve}
        liquidity_totals = {liquidity: Decimal("0.00") for liquidity in LiquidityClass}
        locked_totals = {sleeve: Decimal("0.00") for sleeve in Sleeve}
        sleeve_liquidity: dict[tuple[Sleeve, LiquidityClass], Decimal] = {
            (sleeve, liquidity): Decimal("0.00")
            for sleeve in Sleeve
            for liquidity in LiquidityClass
        }
        investable_totals = {sleeve: Decimal("0.00") for sleeve in POLICY_SLEEVES}
        liabilities = Decimal("0.00")

        for asset, valuation in self.valued_assets:
            if isinstance(asset, LoanAsset):
                liabilities += asset.outstanding
                continue
            value = _money_value(valuation.value_today, "asset valuation")
            liquidity = default_liquidity(asset)
            split = _split_value(value, default_sleeves(asset))
            investable = _is_investable(asset)
            for sleeve, amount in split.items():
                sleeve_totals[sleeve] += amount
                liquidity_totals[liquidity] += amount
                sleeve_liquidity[(sleeve, liquidity)] += amount
                if liquidity is LiquidityClass.LOCKED:
                    locked_totals[sleeve] += amount
                policy_sleeve = _to_policy_sleeve(sleeve)
                if investable and policy_sleeve is not None:
                    investable_totals[policy_sleeve] += amount

        for raw_sleeve, raw_value, raw_liquidity in self.market_holdings:
            try:
                sleeve = raw_sleeve if isinstance(raw_sleeve, Sleeve) else Sleeve(raw_sleeve)
                liquidity = (
                    raw_liquidity
                    if isinstance(raw_liquidity, LiquidityClass)
                    else LiquidityClass(raw_liquidity)
                )
            except ValueError as error:
                raise ValueError(
                    "market holding uses an unknown sleeve or liquidity class"
                ) from error
            amount = _money_value(raw_value, "market holding value")
            sleeve_totals[sleeve] += amount
            liquidity_totals[liquidity] += amount
            sleeve_liquidity[(sleeve, liquidity)] += amount
            if liquidity is LiquidityClass.LOCKED:
                locked_totals[sleeve] += amount
            policy_sleeve = _to_policy_sleeve(sleeve)
            if policy_sleeve is not None:
                investable_totals[policy_sleeve] += amount

        object.__setattr__(self, "totals_by_sleeve", _freeze_money_map(sleeve_totals))
        object.__setattr__(self, "totals_by_liquidity", _freeze_money_map(liquidity_totals))
        object.__setattr__(self, "locked_by_sleeve", _freeze_money_map(locked_totals))
        object.__setattr__(
            self,
            "totals_by_sleeve_and_liquidity",
            _freeze_money_map(sleeve_liquidity),
        )
        object.__setattr__(
            self, "investable_totals_by_sleeve", _freeze_money_map(investable_totals)
        )
        object.__setattr__(self, "liabilities_total", round_paise(liabilities))
        object.__setattr__(
            self,
            "all_assets_total",
            round_paise(sum(sleeve_totals.values(), Decimal(0))),
        )
        object.__setattr__(
            self,
            "investable_total",
            round_paise(sum(investable_totals.values(), Decimal(0))),
        )

    @classmethod
    def build(
        cls,
        valued_assets: Iterable[ValuedAsset] = (),
        market_holdings: Iterable[MarketHolding] = (),
    ) -> WealthSnapshot:
        """Construct a snapshot from valued assets and market holdings."""
        return cls(valued_assets, market_holdings)

    @classmethod
    def from_assets(
        cls,
        valued_assets: Iterable[ValuedAsset] = (),
        market_holdings: Iterable[MarketHolding] = (),
    ) -> WealthSnapshot:
        """Construct a snapshot from valued manual assets and market holdings."""
        return cls(valued_assets, market_holdings)

    @property
    def net_worth(self) -> Decimal:
        """Return total assets less outstanding liabilities."""
        return round_paise(self.all_assets_total - self.liabilities_total)

    def amount_for(
        self,
        sleeve: Sleeve | str,
        liquidity: LiquidityClass | str | None = None,
    ) -> Decimal:
        """Return a sleeve total, optionally restricted to one liquidity class."""
        try:
            if isinstance(sleeve, str) and sleeve.lower() == "equity":
                matching_sleeves = (Sleeve.EQUITY_INDIA, Sleeve.EQUITY_GLOBAL)
            else:
                matching_sleeves = (sleeve if isinstance(sleeve, Sleeve) else Sleeve(sleeve),)
            selected_liquidity = (
                liquidity
                if isinstance(liquidity, LiquidityClass) or liquidity is None
                else LiquidityClass(liquidity)
            )
        except ValueError as error:
            raise ValueError("unknown sleeve or liquidity class") from error

        if selected_liquidity is None:
            return sum((self.totals_by_sleeve[item] for item in matching_sleeves), Decimal(0))
        return sum(
            (
                self.totals_by_sleeve_and_liquidity[(item, selected_liquidity)]
                for item in matching_sleeves
            ),
            Decimal(0),
        )


def _money_value(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a non-negative finite Decimal")
    return round_paise(value)


def _split_value(value: Decimal, sleeve_mix: Mapping[Sleeve, Decimal]) -> dict[Sleeve, Decimal]:
    if not sleeve_mix:
        return {}
    total_fraction = sum(sleeve_mix.values(), Decimal(0))
    if total_fraction != Decimal(1):
        raise ValueError("default sleeve fractions must sum to one")
    residual_sleeve = max(sleeve_mix, key=sleeve_mix.__getitem__)
    split: dict[Sleeve, Decimal] = {}
    allocated = Decimal(0)
    for sleeve, fraction in sleeve_mix.items():
        if sleeve == residual_sleeve:
            continue
        amount = round_paise(value * fraction)
        split[sleeve] = amount
        allocated += amount
    split[residual_sleeve] = value - allocated
    return split


def _to_policy_sleeve(sleeve: Sleeve) -> PolicySleeve | None:
    if sleeve in {Sleeve.EQUITY_INDIA, Sleeve.EQUITY_GLOBAL}:
        return "equity"
    if sleeve in {Sleeve.DEBT, Sleeve.GOLD, Sleeve.CASH}:
        return sleeve.value  # type: ignore[return-value]
    return None


def _is_investable(asset: AnyAsset) -> bool:
    if asset.asset_type in {"property", "term_insurance", "health_insurance", "loan"}:
        return False
    return not (
        isinstance(asset, GoldAsset) and asset.kind == "physical" and asset.purpose == "jewellery"
    )


def _freeze_money_map[Key](values: Mapping[Key, Decimal]) -> Mapping[Key, Decimal]:
    return MappingProxyType({key: round_paise(value) for key, value in values.items()})
