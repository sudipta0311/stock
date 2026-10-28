"""ECB euro reference rates cross-calculated to INR."""

from __future__ import annotations

import csv
import io
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from wealth.marketdata.sources.fred import ECB_SOURCE, FxObservation
from wealth.marketdata.sources.http import get

_ECB_SERIES_ROOT = "https://data-api.ecb.europa.eu/service/data/EXR"
ECB_INR_EUR_URL = f"{_ECB_SERIES_ROOT}/D.INR.EUR.SP00.A?lastNObservations=10&format=csvdata"
ECB_USD_EUR_URL = f"{_ECB_SERIES_ROOT}/D.USD.EUR.SP00.A?lastNObservations=10&format=csvdata"

# Checked these official ECB series keys on 2026-10-28:
# https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A?detail=serieskeysonly&format=csvdata
# The supported currencies below are the ECB-published reference series used by this adapter.
ECB_CURRENCIES = ("INR", "DKK", "USD", "SEK", "NOK", "GBP", "SGD", "CAD", "AUD")
ECB_CROSS_CURRENCIES = tuple(currency for currency in ECB_CURRENCIES if currency != "INR")
_FOUR_DP = Decimal("0.0001")


def _multi_series_url(currencies: Sequence[str]) -> str:
    currency_codes = "+".join(("INR", *currencies))
    return f"{_ECB_SERIES_ROOT}/D.{currency_codes}.EUR.SP00.A?lastNObservations=10&format=csvdata"


ECB_MULTI_CROSS_URL = _multi_series_url(ECB_CROSS_CURRENCIES)
ECB_EUR_CROSS_URLS = {
    currency: f"{_ECB_SERIES_ROOT}/D.{currency}.EUR.SP00.A?lastNObservations=10&format=csvdata"
    for currency in ECB_CROSS_CURRENCIES
}
ECB_SOURCE_URL = ECB_MULTI_CROSS_URL


@dataclass(frozen=True)
class FxFetchResult:
    observations: tuple[FxObservation, ...]
    errors: Mapping[str, str]


def _parse_euro_reference_csv(text: str, *, series_name: str) -> dict[date, Decimal]:
    if not text.strip():
        return {}
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    columns = {field.strip().upper(): field for field in reader.fieldnames or []}
    date_column = columns.get("TIME_PERIOD")
    value_column = columns.get("OBS_VALUE")
    if date_column is None or value_column is None:
        warnings.warn(
            f"ECB {series_name} CSV has an unrecognized header; rows were skipped",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    observations: dict[date, Decimal] = {}
    for line_number, row in enumerate(reader, start=2):
        try:
            date_text = (row.get(date_column) or "").strip()
            value_text = (row.get(value_column) or "").strip()
            if not date_text or not value_text or value_text == ".":
                continue
            as_of = date.fromisoformat(date_text)
            value = Decimal(value_text)
            if not value.is_finite() or value <= 0:
                raise ValueError("invalid reference rate")
            observations[as_of] = value
        except (InvalidOperation, ValueError):
            warnings.warn(
                f"ECB {series_name} row {line_number} was skipped", RuntimeWarning, stacklevel=2
            )
    return observations


def _parse_multi_euro_reference_csv(
    text: str,
    *,
    currencies: Sequence[str],
) -> tuple[dict[str, dict[date, Decimal]], dict[str, str]]:
    if not text.strip():
        return {}, {currency: "EmptyResponse" for currency in currencies}

    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    columns = {field.strip().upper(): field for field in reader.fieldnames or []}
    currency_column = columns.get("CURRENCY")
    date_column = columns.get("TIME_PERIOD")
    value_column = columns.get("OBS_VALUE")
    if currency_column is None or date_column is None or value_column is None:
        return {}, {currency: "InvalidCsvHeader" for currency in currencies}

    observations: dict[str, dict[date, Decimal]] = {}
    errors: dict[str, str] = {}
    expected = set(currencies)
    for line_number, row in enumerate(reader, start=2):
        currency = (row.get(currency_column) or "").strip().upper()
        if currency not in expected:
            continue
        try:
            date_text = (row.get(date_column) or "").strip()
            value_text = (row.get(value_column) or "").strip()
            if not date_text or not value_text or value_text == ".":
                continue
            as_of = date.fromisoformat(date_text)
            value = Decimal(value_text)
            if not value.is_finite() or value <= 0:
                raise ValueError("invalid reference rate")
            observations.setdefault(currency, {})[as_of] = value
        except (InvalidOperation, ValueError):
            errors[currency] = "MalformedObservation"
            warnings.warn(
                f"ECB {currency} row {line_number} was skipped", RuntimeWarning, stacklevel=2
            )
    return observations, errors


def parse_ecb_multi_fx_rates(
    text: str,
    *,
    currencies: Sequence[str] = ECB_CROSS_CURRENCIES,
    fetched_on: date | None = None,
    source_url: str = ECB_MULTI_CROSS_URL,
) -> FxFetchResult:
    """Parse one ECB multi-series CSV, retaining valid currencies when one is absent."""
    selected = tuple(dict.fromkeys(currencies))
    unsupported = set(selected) - set(ECB_CROSS_CURRENCIES)
    if unsupported:
        raise ValueError(f"unsupported ECB currency: {', '.join(sorted(unsupported))}")

    expected = ("INR", *selected)
    series, errors = _parse_multi_euro_reference_csv(text, currencies=expected)
    inr_per_eur = series.get("INR", {})
    if not inr_per_eur:
        errors.setdefault("INR", "NoObservations")
    effective_date = fetched_on or datetime.now(UTC).date()
    observations: list[FxObservation] = []

    for as_of, rate in sorted(inr_per_eur.items()):
        observations.append(
            FxObservation(
                pair="EUR/INR",
                rate=rate.quantize(_FOUR_DP, rounding=ROUND_HALF_UP),
                as_of=as_of,
                publication_lag_days=max(0, (effective_date - as_of).days),
                source=ECB_SOURCE,
                source_url=source_url,
            )
        )

    for currency in selected:
        currency_per_eur = series.get(currency, {})
        if not currency_per_eur:
            errors.setdefault(currency, "NoObservations")
            continue
        if not inr_per_eur:
            errors.setdefault(currency, "MissingBaseSeries")
            continue
        shared_dates = sorted(inr_per_eur.keys() & currency_per_eur.keys())
        if not shared_dates:
            errors.setdefault(currency, "NoOverlappingObservations")
            continue
        for as_of in shared_dates:
            rate = (inr_per_eur[as_of] / currency_per_eur[as_of]).quantize(
                _FOUR_DP, rounding=ROUND_HALF_UP
            )
            observations.append(
                FxObservation(
                    pair=f"{currency}/INR",
                    rate=rate,
                    as_of=as_of,
                    publication_lag_days=max(0, (effective_date - as_of).days),
                    source=ECB_SOURCE,
                    source_url=source_url,
                )
            )

    return FxFetchResult(tuple(observations), errors)


def parse_ecb_usd_inr(
    inr_per_eur_csv: str,
    usd_per_eur_csv: str,
    *,
    fetched_on: date | None = None,
) -> list[FxObservation]:
    """Compute USD/INR using the dates shared by the two ECB EUR reference series."""
    inr_per_eur = _parse_euro_reference_csv(inr_per_eur_csv, series_name="INR/EUR")
    usd_per_eur = _parse_euro_reference_csv(usd_per_eur_csv, series_name="USD/EUR")
    effective_date = fetched_on or datetime.now(UTC).date()
    observations: list[FxObservation] = []
    for as_of in sorted(inr_per_eur.keys() & usd_per_eur.keys()):
        rate = (inr_per_eur[as_of] / usd_per_eur[as_of]).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        observations.append(
            FxObservation(
                pair="USD/INR",
                rate=rate,
                as_of=as_of,
                publication_lag_days=max(0, (effective_date - as_of).days),
                source=ECB_SOURCE,
                source_url=f"{ECB_INR_EUR_URL}\n{ECB_USD_EUR_URL}",
            )
        )
    return observations


def parse_ecb_fx_rates(
    inr_per_eur_csv: str,
    euro_cross_csvs: Mapping[str, str],
    *,
    fetched_on: date | None = None,
) -> list[FxObservation]:
    """Parse individual ECB series; retained for callers with recorded single-series data."""
    inr_per_eur = _parse_euro_reference_csv(inr_per_eur_csv, series_name="INR/EUR")
    effective_date = fetched_on or datetime.now(UTC).date()
    observations: list[FxObservation] = []
    for currency in sorted(set(euro_cross_csvs) | {"EUR"}):
        if currency != "EUR" and currency not in ECB_CROSS_CURRENCIES:
            raise ValueError(f"unsupported ECB currency: {currency}")
        if currency == "EUR":
            currency_per_eur = {day: Decimal(1) for day in inr_per_eur}
            cross_url = ""
        else:
            currency_per_eur = _parse_euro_reference_csv(
                euro_cross_csvs[currency], series_name=f"{currency}/EUR"
            )
            cross_url = ECB_EUR_CROSS_URLS[currency]
        source_url = f"{ECB_INR_EUR_URL}\n{cross_url}" if cross_url else ECB_INR_EUR_URL
        for as_of in sorted(inr_per_eur.keys() & currency_per_eur.keys()):
            rate = (inr_per_eur[as_of] / currency_per_eur[as_of]).quantize(
                _FOUR_DP, rounding=ROUND_HALF_UP
            )
            observations.append(
                FxObservation(
                    pair=f"{currency}/INR",
                    rate=rate,
                    as_of=as_of,
                    publication_lag_days=max(0, (effective_date - as_of).days),
                    source=ECB_SOURCE,
                    source_url=source_url,
                )
            )
    return observations


def fetch_ecb_usd_inr(*, fetched_on: date | None = None) -> list[FxObservation]:
    """Fetch both ECB series and calculate the overlapping daily USD/INR rates."""
    inr_csv = get(ECB_INR_EUR_URL).text
    usd_csv = get(ECB_USD_EUR_URL).text
    return parse_ecb_usd_inr(inr_csv, usd_csv, fetched_on=fetched_on)


def _fetch_ecb_individual(currencies: Sequence[str], *, fetched_on: date | None) -> FxFetchResult:
    """Retry a failed multi-series request independently so good currencies survive."""
    effective_date = fetched_on or datetime.now(UTC).date()
    errors: dict[str, str] = {}
    observations: list[FxObservation] = []
    try:
        inr_csv = get(ECB_INR_EUR_URL).text
    except Exception as error:
        errors["INR"] = type(error).__name__
        for currency in currencies:
            errors[currency] = "MissingBaseSeries"
        return FxFetchResult((), errors)

    inr_per_eur = _parse_euro_reference_csv(inr_csv, series_name="INR/EUR")
    if not inr_per_eur:
        errors["INR"] = "NoObservations"
        for currency in currencies:
            errors[currency] = "MissingBaseSeries"
        return FxFetchResult((), errors)

    for as_of, rate in sorted(inr_per_eur.items()):
        observations.append(
            FxObservation(
                pair="EUR/INR",
                rate=rate.quantize(_FOUR_DP, rounding=ROUND_HALF_UP),
                as_of=as_of,
                publication_lag_days=max(0, (effective_date - as_of).days),
                source=ECB_SOURCE,
                source_url=ECB_INR_EUR_URL,
            )
        )

    for currency in currencies:
        url = ECB_EUR_CROSS_URLS[currency]
        try:
            csv_text = get(url).text
        except Exception as error:
            errors[currency] = type(error).__name__
            continue
        currency_per_eur = _parse_euro_reference_csv(csv_text, series_name=f"{currency}/EUR")
        if not currency_per_eur:
            errors[currency] = "NoObservations"
            continue
        shared_dates = sorted(inr_per_eur.keys() & currency_per_eur.keys())
        if not shared_dates:
            errors[currency] = "NoOverlappingObservations"
            continue
        for as_of in shared_dates:
            rate = (inr_per_eur[as_of] / currency_per_eur[as_of]).quantize(
                _FOUR_DP, rounding=ROUND_HALF_UP
            )
            observations.append(
                FxObservation(
                    pair=f"{currency}/INR",
                    rate=rate,
                    as_of=as_of,
                    publication_lag_days=max(0, (effective_date - as_of).days),
                    source=ECB_SOURCE,
                    source_url=f"{ECB_INR_EUR_URL}\n{url}",
                )
            )
    return FxFetchResult(tuple(observations), errors)


def fetch_ecb_fx_rates(
    currencies: Sequence[str] = ECB_CROSS_CURRENCIES,
    *,
    fetched_on: date | None = None,
) -> FxFetchResult:
    """Fetch one ECB multi-series CSV; isolate per-series retries if the batch fails."""
    selected = tuple(dict.fromkeys(currencies))
    unsupported = set(selected) - set(ECB_CROSS_CURRENCIES)
    if unsupported:
        raise ValueError(f"unsupported ECB currency: {', '.join(sorted(unsupported))}")
    source_url = _multi_series_url(selected)
    try:
        response_text = get(source_url).text
    except Exception:
        return _fetch_ecb_individual(selected, fetched_on=fetched_on)
    return parse_ecb_multi_fx_rates(
        response_text,
        currencies=selected,
        fetched_on=fetched_on,
        source_url=source_url,
    )
