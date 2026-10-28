"""Deterministic Indian capital-gains tax estimates and tax-aware lot selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_FLOOR, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from wealth.core.dates import add_months
from wealth.core.money import round_paise

TaxClass = Literal[
    "listed_equity",
    "equity_mf",
    "foreign_equity",
    "intl_fof",
    "intl_etf_listed_in",
    "debt_mf_post_2023",
    "gold_etf",
    "sgb",
]
HoldingPeriod = Literal["ST", "LT"]
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "tax" / "india.yaml"
_EQUITY_CLASSES = frozenset({"listed_equity", "equity_mf"})
_ZERO = Decimal("0.00")


def _decimal(value: object, name: str, *, minimum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise ValueError(f"{name} must be a Decimal")
    if not value.is_finite() or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return value


@dataclass(frozen=True)
class Lot:
    """One acquisition lot; optional prices and maturity date are supplied by the caller."""

    lot_id: str
    tax_class: TaxClass
    quantity: Decimal
    cost_per_unit: Decimal
    acquired_on: date
    fmv_2018_01_31: Decimal | None = None
    market_price: Decimal | None = None
    maturity_date: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lot_id, str) or not self.lot_id.strip():
            raise ValueError("lot_id must be non-empty text")
        if self.tax_class not in _tax_rules()["tax_classes"]:
            raise ValueError(f"unsupported tax class: {self.tax_class}")
        if not isinstance(self.acquired_on, date):
            raise ValueError("acquired_on must be a date")
        if self.tax_class == "debt_mf_post_2023":
            debt_class_start = date.fromisoformat(
                str(_rule_for("debt_mf_post_2023")["effective_from"])
            )
            if self.acquired_on < debt_class_start:
                raise ValueError("debt_mf_post_2023 lots must be acquired on or after 2023-04-01")
        _decimal(self.quantity, "quantity", minimum=Decimal(0))
        if self.quantity == 0:
            raise ValueError("quantity must be positive")
        _decimal(self.cost_per_unit, "cost_per_unit", minimum=Decimal(0))
        for name, value in (
            ("fmv_2018_01_31", self.fmv_2018_01_31),
            ("market_price", self.market_price),
        ):
            if value is not None:
                _decimal(value, name, minimum=Decimal(0))
        if self.maturity_date is not None and not isinstance(self.maturity_date, date):
            raise ValueError("maturity_date must be a date")
        if self.maturity_date is not None and self.maturity_date < self.acquired_on:
            raise ValueError("maturity_date cannot be before acquired_on")


@dataclass(frozen=True)
class TaxContext:
    """Taxpayer rates and listed-equity gains already booked in the current FY."""

    slab_pct: Decimal = Decimal("30")
    surcharge_pct: Decimal = Decimal("0")
    fy_realised_equity_ltcg: Decimal = Decimal("0")
    fy_realised_equity_stcg: Decimal = Decimal("0")
    residency_status: Literal["RI", "NRI", "RNOR"] = "RI"
    us_person: bool = False

    def __post_init__(self) -> None:
        for name in ("slab_pct", "surcharge_pct"):
            value = _decimal(getattr(self, name), name, minimum=Decimal(0))
            if value > 100:
                raise ValueError(f"{name} must be between 0 and 100")
        for name in ("fy_realised_equity_ltcg", "fy_realised_equity_stcg"):
            _decimal(getattr(self, name), name, minimum=Decimal(0))


@dataclass(frozen=True)
class LotTax:
    """Tax outcome for one submitted lot sale, after loss offsets and exemption."""

    lot_id: str
    quantity: Decimal
    proceeds: Decimal
    cost_basis_per_unit: Decimal
    gain: Decimal
    holding_period: HoldingPeriod
    rate_pct: Decimal
    gain_after_loss_setoff: Decimal
    taxable_gain: Decimal
    tax_before_cess: Decimal
    note: str | None = None


@dataclass(frozen=True)
class TaxEstimate:
    """Paise-rounded tax totals and the sale-level detail used to derive them."""

    lines: tuple[LotTax, ...]
    gross_proceeds: Decimal
    tax_before_cess: Decimal
    cess: Decimal
    total_tax: Decimal
    exemption_used: Decimal
    net_proceeds: Decimal
    tds_estimate: Decimal | None = None
    notes: tuple[str, ...] = ()

    @property
    def per_lot(self) -> tuple[LotTax, ...]:
        """Alias for the line-by-line tax detail."""
        return self.lines


@dataclass(frozen=True)
class HarvestSuggestion:
    """Whole-unit eligible gains to realise before the Indian FY closes."""

    gains_booked: Decimal
    tax_saved_before_cess: Decimal
    lots: tuple[tuple[Lot, Decimal], ...]

    @property
    def tax_saved(self) -> Decimal:
        """Tax saved at 12.5%, before cess, for use of the exempt gain amount."""
        return self.tax_saved_before_cess


@lru_cache(maxsize=1)
def _tax_rules() -> dict[str, object]:
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        rules = yaml.safe_load(rules_file)
    if not isinstance(rules, dict) or not isinstance(rules.get("tax_classes"), dict):
        raise ValueError("Indian tax rules must contain a tax_classes mapping")
    entries: list[tuple[str, object]] = [
        (key, rules.get(key))
        for key in ("supported_sales_from", "cess", "equity_ltcg_exemption", "grandfathering")
    ]
    entries.extend(rules["tax_classes"].items())
    for key, entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"tax rule {key} must be a mapping")
        if not isinstance(entry.get("effective_from"), str) or not entry.get("source_url"):
            raise ValueError(f"tax rule {key} needs effective_from and source_url")
        if entry.get("verified") is not False:
            raise ValueError(f"tax rule {key} must remain marked unverified")
    expected_review = {"intl_fof", "intl_etf_listed_in", "gold_etf", "sgb"}
    for key, rule in rules["tax_classes"].items():
        if (key in expected_review) != (rule.get("needs_review") is True):
            raise ValueError(f"tax rule {key} has an invalid needs_review flag")
    return rules


def load_tax_rules() -> dict[str, object]:
    """Return a caller-owned copy of the checked-in tax rulebook."""
    return deepcopy(_tax_rules())


def _check_sale_date(value: date) -> date:
    checked_date = _date(value, "sale_date")
    support_rule = _tax_rules()["supported_sales_from"]
    assert isinstance(support_rule, dict)
    supported_from = date.fromisoformat(str(support_rule["effective_from"]))
    if checked_date < supported_from:
        raise ValueError("sales before 2024-07-23 are outside the supported tax rules")
    return checked_date


def _rule_for(tax_class: TaxClass) -> Mapping[str, object]:
    rules = _tax_rules()["tax_classes"]
    assert isinstance(rules, dict)
    try:
        rule = rules[tax_class]
    except KeyError as error:
        raise ValueError(f"unsupported tax class: {tax_class}") from error
    assert isinstance(rule, dict)
    return rule


def _date(value: object, name: str) -> date:
    if not isinstance(value, date):
        raise ValueError(f"{name} must be a date")
    return value


def _price(value: object) -> Decimal:
    return _decimal(value, "sale_price", minimum=Decimal(0))


def _rate_value(value: object, ctx: TaxContext) -> Decimal:
    if value == "slab":
        return ctx.slab_pct
    if not isinstance(value, str):
        raise ValueError("configured tax rates must be text percentages or slab")
    return Decimal(value)


def _long_term_months(lot: Lot) -> int | None:
    value = _rule_for(lot.tax_class).get("long_term_months")
    return None if value is None else int(value)


def is_long_term(lot: Lot, sale_date: date) -> bool:
    """Return true only after the configured holding period has fully elapsed."""
    checked_date = _date(sale_date, "sale_date")
    months = _long_term_months(lot)
    return months is not None and checked_date > add_months(lot.acquired_on, months)


def days_to_long_term(lot: Lot, today: date) -> int:
    """Count days until the first long-term sale date; assets without an LT period return zero."""
    checked_date = _date(today, "today")
    months = _long_term_months(lot)
    if months is None:
        return 0
    first_long_term_date = add_months(lot.acquired_on, months) + timedelta(days=1)
    return max(0, (first_long_term_date - checked_date).days)


def _cost_basis(lot: Lot, sale_price: Decimal) -> tuple[Decimal, str | None]:
    grandfathering = _tax_rules()["grandfathering"]
    assert isinstance(grandfathering, dict)
    acquired_before = date.fromisoformat(str(grandfathering["acquired_before"]))
    if lot.tax_class not in _EQUITY_CLASSES or lot.acquired_on >= acquired_before:
        return lot.cost_per_unit, None
    if lot.fmv_2018_01_31 is None:
        return lot.cost_per_unit, "grandfather_fmv_missing"
    return max(lot.cost_per_unit, min(lot.fmv_2018_01_31, sale_price)), None


def _holding_period(lot: Lot, sale_date: date) -> HoldingPeriod:
    return "LT" if is_long_term(lot, sale_date) else "ST"


def _configured_rate(lot: Lot, holding_period: HoldingPeriod, ctx: TaxContext) -> Decimal:
    rule = _rule_for(lot.tax_class)
    key = "long_term_rate_pct" if holding_period == "LT" else "short_term_rate_pct"
    return _rate_value(rule[key], ctx)


def _surcharge_pct(lot: Lot, rate_pct: Decimal, ctx: TaxContext) -> Decimal:
    rule = _rule_for(lot.tax_class)
    cap = rule.get("surcharge_cap_pct")
    if cap is not None and rate_pct in {Decimal("20"), Decimal("12.5")}:
        return min(ctx.surcharge_pct, Decimal(str(cap)))
    return ctx.surcharge_pct


@dataclass
class _Sale:
    lot: Lot
    quantity: Decimal
    sale_price: Decimal
    proceeds: Decimal
    basis_per_unit: Decimal
    gain: Decimal
    holding_period: HoldingPeriod
    rate_pct: Decimal
    gain_after_loss_setoff: Decimal
    taxable_gain: Decimal
    tax_before_cess: Decimal = _ZERO
    note: str | None = None
    maturity_exempt: bool = False


def _consume_losses(targets: list[_Sale], loss_amount: Decimal) -> Decimal:
    """Apply loss dollars to high-rate gains first, returning any unused loss."""
    candidates = sorted(
        (item for item in targets if item.gain_after_loss_setoff > 0),
        key=lambda item: (item.rate_pct, item.lot.lot_id),
        reverse=True,
    )
    remaining = loss_amount
    for item in candidates:
        applied = min(item.gain_after_loss_setoff, remaining)
        item.gain_after_loss_setoff = round_paise(item.gain_after_loss_setoff - applied)
        remaining = round_paise(remaining - applied)
        if remaining <= 0:
            break
    return remaining


def estimate_sale_tax(
    sales: list[tuple[Lot, Decimal, Decimal]], sale_date: date, ctx: TaxContext
) -> TaxEstimate:
    """Estimate tax after sale-set loss netting and the one shared equity LTCG exemption."""
    checked_date = _check_sale_date(sale_date)
    if not isinstance(ctx, TaxContext):
        raise ValueError("ctx must be a TaxContext")

    work: list[_Sale] = []
    sold_by_lot: dict[str, Decimal] = {}
    notes: list[str] = []
    gross_proceeds = Decimal(0)
    for sale in sales:
        if not isinstance(sale, tuple) or len(sale) != 3:
            raise ValueError("each sale must be a (Lot, quantity, sale_price) tuple")
        lot, quantity_value, price_value = sale
        if not isinstance(lot, Lot):
            raise ValueError("each sale must include a Lot")
        quantity = _decimal(quantity_value, "quantity", minimum=Decimal(0))
        price = _price(price_value)
        if quantity == 0:
            raise ValueError("sale quantity must be positive")
        sold_by_lot[lot.lot_id] = sold_by_lot.get(lot.lot_id, Decimal(0)) + quantity
        if sold_by_lot[lot.lot_id] > lot.quantity:
            raise ValueError(f"sale quantity exceeds available lot {lot.lot_id}")

        basis, note = _cost_basis(lot, price)
        if note is not None:
            notes.append(note)
        period = _holding_period(lot, checked_date)
        rate = _configured_rate(lot, period, ctx)
        proceeds = round_paise(quantity * price)
        gain = round_paise(quantity * (price - basis))
        rule = _rule_for(lot.tax_class)
        maturity_exempt = bool(
            lot.tax_class == "sgb"
            and rule.get("redemption_exempt_at_maturity")
            and lot.maturity_date is not None
            and checked_date == lot.maturity_date
        )
        gross_proceeds += proceeds
        work.append(
            _Sale(
                lot=lot,
                quantity=quantity,
                sale_price=price,
                proceeds=proceeds,
                basis_per_unit=basis,
                gain=gain,
                holding_period=period,
                rate_pct=rate,
                gain_after_loss_setoff=max(gain, Decimal(0)),
                taxable_gain=max(gain, Decimal(0)),
                note=note,
                maturity_exempt=maturity_exempt,
            )
        )

    short_losses = round_paise(
        sum(
            (
                -item.gain
                for item in work
                if item.gain < 0 and item.holding_period == "ST" and not item.maturity_exempt
            ),
            Decimal(0),
        )
    )
    long_losses = round_paise(
        sum(
            (
                -item.gain
                for item in work
                if item.gain < 0 and item.holding_period == "LT" and not item.maturity_exempt
            ),
            Decimal(0),
        )
    )
    short_gains = [
        item
        for item in work
        if item.gain > 0 and item.holding_period == "ST" and not item.maturity_exempt
    ]
    long_gains = [
        item
        for item in work
        if item.gain > 0 and item.holding_period == "LT" and not item.maturity_exempt
    ]
    short_losses = _consume_losses(short_gains, short_losses)
    if short_losses > 0:
        long_losses = round_paise(long_losses + short_losses)
    _consume_losses(long_gains, long_losses)

    equity_ltcg = [
        item
        for item in long_gains
        if item.lot.tax_class in _EQUITY_CLASSES and item.gain_after_loss_setoff > 0
    ]
    rules = _tax_rules()
    exemption_rule = rules.get("equity_ltcg_exemption")
    assert isinstance(exemption_rule, dict)
    exemption_limit = Decimal(str(exemption_rule["amount_inr"]))
    remaining_exemption = max(Decimal(0), exemption_limit - ctx.fy_realised_equity_ltcg)
    exemption_used = min(
        remaining_exemption,
        sum((item.gain_after_loss_setoff for item in equity_ltcg), Decimal(0)),
    )
    exemption_left = exemption_used
    for item in sorted(equity_ltcg, key=lambda entry: entry.lot.lot_id):
        applied = min(item.gain_after_loss_setoff, exemption_left)
        item.taxable_gain = round_paise(item.gain_after_loss_setoff - applied)
        exemption_left = round_paise(exemption_left - applied)

    for item in work:
        if item.maturity_exempt or item.gain < 0:
            item.taxable_gain = Decimal(0)
        elif item not in equity_ltcg:
            item.taxable_gain = item.gain_after_loss_setoff
        if item.gain >= 0 and not item.maturity_exempt:
            tax = round_paise(item.taxable_gain * item.rate_pct / Decimal(100))
            surcharge_rate = _surcharge_pct(item.lot, item.rate_pct, ctx)
            surcharge = round_paise(tax * surcharge_rate / Decimal(100))
            item.tax_before_cess = round_paise(tax + surcharge)

    tax_before_cess = round_paise(sum((item.tax_before_cess for item in work), Decimal(0)))
    cess_rule = rules.get("cess")
    assert isinstance(cess_rule, dict)
    cess = round_paise(tax_before_cess * Decimal(str(cess_rule["rate_pct"])) / Decimal(100))
    total_tax = round_paise(tax_before_cess + cess)
    lines = tuple(
        LotTax(
            lot_id=item.lot.lot_id,
            quantity=item.quantity,
            proceeds=item.proceeds,
            cost_basis_per_unit=item.basis_per_unit,
            gain=item.gain,
            holding_period=item.holding_period,
            rate_pct=item.rate_pct,
            gain_after_loss_setoff=round_paise(item.gain_after_loss_setoff),
            taxable_gain=round_paise(item.taxable_gain),
            tax_before_cess=item.tax_before_cess,
            note=item.note,
        )
        for item in work
    )
    tds_estimate: Decimal | None = None
    if ctx.residency_status == "NRI":
        listed_equity_tax = round_paise(
            sum(
                (
                    item.tax_before_cess
                    for item in work
                    if item.lot.tax_class in {"listed_equity", "equity_mf"}
                ),
                Decimal(0),
            )
        )
        tds_cess = round_paise(
            listed_equity_tax * Decimal(str(cess_rule["rate_pct"])) / Decimal(100)
        )
        tds_estimate = round_paise(listed_equity_tax + tds_cess)
        notes.extend(["tds_deducted_at_source", "residence_country_tax_may_apply"])
        if ctx.us_person:
            notes.append("us_tax_applies")
    return TaxEstimate(
        lines=lines,
        gross_proceeds=round_paise(gross_proceeds),
        tax_before_cess=tax_before_cess,
        cess=cess,
        total_tax=total_tax,
        exemption_used=round_paise(exemption_used),
        net_proceeds=round_paise(gross_proceeds - total_tax),
        tds_estimate=tds_estimate,
        notes=tuple(dict.fromkeys(notes)),
    )


def _lot_priority(lot: Lot, sale_price: Decimal, sale_date: date) -> tuple[int, Decimal, str]:
    basis, _ = _cost_basis(lot, sale_price)
    gain_per_unit = sale_price - basis
    if gain_per_unit < 0:
        return (0, gain_per_unit, lot.lot_id)
    if is_long_term(lot, sale_date):
        return (1, gain_per_unit, lot.lot_id)
    return (2, gain_per_unit, lot.lot_id)


def select_lots(
    lots: Sequence[Lot],
    quantity: Decimal,
    sale_price: Decimal,
    sale_date: date,
    ctx: TaxContext,
) -> list[tuple[Lot, Decimal]]:
    """Select losses first, then the lowest-gain LT lots, then the lowest-gain ST lots."""
    requested = _decimal(quantity, "quantity", minimum=Decimal(0))
    price = _price(sale_price)
    checked_date = _check_sale_date(sale_date)
    if not isinstance(ctx, TaxContext):
        raise ValueError("ctx must be a TaxContext")
    if requested <= 0:
        raise ValueError("quantity must be positive")
    if any(not isinstance(lot, Lot) for lot in lots):
        raise ValueError("lots must contain Lot values")
    if requested > sum((lot.quantity for lot in lots), Decimal(0)):
        raise ValueError("requested quantity exceeds available lots")
    selected: list[tuple[Lot, Decimal]] = []
    remaining = requested
    for lot in sorted(lots, key=lambda item: _lot_priority(item, price, checked_date)):
        take = min(lot.quantity, remaining)
        if take > 0:
            selected.append((lot, take))
            remaining -= take
        if remaining == 0:
            break
    return selected


def ltcg_harvest(
    lots: Sequence[Lot],
    price_by_lot: Mapping[str, Decimal],
    today: date,
    ctx: TaxContext,
) -> HarvestSuggestion | None:
    """Use remaining listed-equity LTCG exemption on low-gain whole units in Feb–Mar."""
    checked_date = _date(today, "today")
    if not (checked_date.month == 2 or checked_date.month == 3):
        return None
    support_rule = _tax_rules()["supported_sales_from"]
    assert isinstance(support_rule, dict)
    supported_from = date.fromisoformat(str(support_rule["effective_from"]))
    if checked_date < supported_from:
        raise ValueError("sales before 2024-07-23 are outside the supported tax rules")
    if not isinstance(ctx, TaxContext):
        raise ValueError("ctx must be a TaxContext")
    rules = _tax_rules()
    exemption = Decimal(str(rules["equity_ltcg_exemption"]["amount_inr"]))  # type: ignore[index]
    remaining_exemption = max(Decimal(0), exemption - ctx.fy_realised_equity_ltcg)
    if remaining_exemption <= 0:
        return None

    candidates: list[tuple[Decimal, str, Lot, Decimal]] = []
    for lot in lots:
        if lot.tax_class not in _EQUITY_CLASSES or not is_long_term(lot, checked_date):
            continue
        price_value = price_by_lot.get(lot.lot_id)
        if price_value is None:
            continue
        price = _price(price_value)
        basis, _ = _cost_basis(lot, price)
        gain_per_unit = price - basis
        whole_units = lot.quantity.to_integral_value(rounding=ROUND_FLOOR)
        if gain_per_unit > 0 and whole_units > 0:
            candidates.append((gain_per_unit, lot.lot_id, lot, price))

    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen: list[tuple[Lot, Decimal]] = []
    gains_booked = Decimal(0)
    for gain_per_unit, _, lot, _ in candidates:
        room = remaining_exemption - gains_booked
        if room <= 0:
            break
        units = min(
            lot.quantity.to_integral_value(rounding=ROUND_FLOOR),
            (room / gain_per_unit).to_integral_value(rounding=ROUND_FLOOR),
        )
        while units > 0 and round_paise(units * gain_per_unit) > room:
            units -= 1
        if units > 0:
            chosen.append((lot, units))
            gains_booked = round_paise(gains_booked + units * gain_per_unit)

    if not chosen:
        return None
    gains_booked = round_paise(gains_booked)
    return HarvestSuggestion(
        gains_booked=gains_booked,
        tax_saved_before_cess=round_paise(
            gains_booked
            * Decimal(str(_rule_for("listed_equity")["long_term_rate_pct"]))
            / Decimal(100)
        ),
        lots=tuple(chosen),
    )
