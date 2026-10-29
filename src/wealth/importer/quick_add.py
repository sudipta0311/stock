"""Deterministic parser for common Money Map quick-add lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wealth.core.dates import parse_month_year
from wealth.core.money import parse_inr

_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DATE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?:({_MONTH})[ -](\d{{2,4}})|"
    rf"(\d{{4}}-\d{{1,2}})|(\d{{1,2}}/\d{{4}}))(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
_AMOUNT_PATTERN = re.compile(
    r"(?<![\w/])(?:₹|â‚¹|rs\.?|inr)?\s*"
    r"(?:\d{1,2}(?:,\d{2})*,\d{3}|\d+)(?:\.\d+)?"
    r"(?:\s*(?:thousand|lakhs?|lac|cr|crores?|[kKlL])\b)?(?![\w/%/.-])",
    re.IGNORECASE,
)
_FOREIGN_SAVINGS_AMOUNT_PATTERN = re.compile(
    r"(?i)(?<![\w/])(?:(?P<prefix>DKK|EUR|kr\.?|€)\s*)?"
    r"(?P<amount>\d{1,3}(?:[., ]\d{3})+|\d+)(?:([.,]\d{1,2}))?"
    r"\s*(?P<suffix>DKK|EUR|kr\.?|€)?(?!\w)"
)


class QuickAddNotUnderstood(ValueError):
    """Raised when no supported asset type can be identified."""


@dataclass(frozen=True)
class QuickAddResult:
    asset_type: str
    draft: dict[str, Any]
    missing: tuple[str, ...]
    confidence: str


@lru_cache(maxsize=1)
def _institutions() -> tuple[tuple[str, tuple[str, ...]], ...]:
    path = Path(__file__).resolve().parents[3] / "rules" / "institutions.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple((item["canonical"], tuple(item["aliases"])) for item in payload["institutions"])


def _normalized(text: str) -> str:
    return text.replace("â‚¹", "₹").replace("Rs.", "Rs ").strip()


def _date_match(text: str) -> tuple[re.Match[str] | None, str | None]:
    for match in _DATE_PATTERN.finditer(text):
        candidate = match.group(0).replace("-", " ", 1) if "-" in match.group(0) else match.group(0)
        try:
            month_year = parse_month_year(candidate)
        except ValueError:
            try:
                month_year = parse_month_year(match.group(0))
            except ValueError:
                continue
        return match, str(month_year)
    return None, None


def _percent(text: str) -> tuple[re.Match[str] | None, str | None]:
    match = _PERCENT_PATTERN.search(text)
    if match is None:
        return None, None
    try:
        amount = Decimal(match.group(1))
    except InvalidOperation:
        return match, None
    if not amount.is_finite():
        return match, None
    return match, format(amount.normalize(), "f")


def _amounts(
    text: str,
    *,
    excluded: tuple[re.Match[str] | None, ...] = (),
) -> list[tuple[re.Match[str], str]]:
    spans = [match.span() for match in excluded if match is not None]
    result: list[tuple[re.Match[str], str]] = []
    for match in _AMOUNT_PATTERN.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in spans):
            continue
        token = match.group(0).strip()
        if not re.search(r"\d", token):
            continue
        try:
            amount = parse_inr(token.replace("₹", "Rs "))
        except ValueError:
            continue
        result.append((match, format(amount, ".2f")))
    return result


def _foreign_savings_amount(text: str) -> tuple[re.Match[str], str, str] | None:
    match = next(
        (
            candidate
            for candidate in _FOREIGN_SAVINGS_AMOUNT_PATTERN.finditer(text)
            if candidate.group("prefix") or candidate.group("suffix")
        ),
        None,
    )
    if match is None:
        return None
    marker = (match.group("prefix") or match.group("suffix") or "").casefold().rstrip(".")
    currency = "DKK" if marker in {"dkk", "kr"} else "EUR"
    amount = match.group("amount")
    decimal_part = match.group(3)
    if decimal_part is not None:
        whole = amount
        fraction = decimal_part[1:]
        if "." in amount or "," in amount:
            whole = amount.replace(".", "").replace(",", "").replace(" ", "")
        normalized = f"{whole}.{fraction}"
    else:
        compact = amount.replace(" ", "")
        if "." in compact and "," in compact:
            decimal_separator = "." if compact.rfind(".") > compact.rfind(",") else ","
            grouping_separator = "," if decimal_separator == "." else "."
            normalized = compact.replace(grouping_separator, "")
            if decimal_separator == ",":
                normalized = normalized.replace(",", ".")
        elif "." in compact or "," in compact:
            separator = "." if "." in compact else ","
            groups = compact.split(separator)
            if all(len(group) == 3 for group in groups[1:]):
                normalized = "".join(groups)
            else:
                normalized = separator.join(groups[:-1]) + "." + groups[-1]
        else:
            normalized = compact
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    if not value.is_finite() or value < 0:
        return None
    return match, format(value.normalize(), "f"), currency


def _party(text: str, ignored: tuple[re.Match[str] | None, ...]) -> str | None:
    aliases: list[tuple[int, int, str]] = []
    for canonical, names in _institutions():
        for name in names:
            match = re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text, re.I)
            if match is not None and not any(
                ignored_match is not None
                and match.start() < ignored_match.end()
                and match.end() > ignored_match.start()
                for ignored_match in ignored
            ):
                aliases.append((len(match.group(0)), -match.start(), canonical))
    if aliases:
        return max(aliases)[2]

    remainder = text
    for match in sorted(
        (item for item in ignored if item is not None), key=lambda item: item.start(), reverse=True
    ):
        remainder = remainder[: match.start()] + " " + remainder[match.end() :]
    remainder = re.sub(
        r"(?i)\b(fd|fixed deposit|ppf|epf|nsc|scss|savings|account|gold|home loan|"
        r"loan|credit card|lic|term|premium|till|until|maturing|maturity|rate)\b",
        " ",
        remainder,
    )
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,.-%")
    return remainder or None


def _finish(asset_type: str, draft: dict[str, Any], required: tuple[str, ...]) -> QuickAddResult:
    missing = tuple(field for field in required if draft.get(field) in (None, ""))
    confidence = "high" if not missing else "medium"
    return QuickAddResult(asset_type, draft, missing, confidence)


def parse_quick_add(text: str) -> QuickAddResult:
    """Parse a supported Money Map shorthand line without external services."""
    if not isinstance(text, str) or not text.strip():
        raise QuickAddNotUnderstood("No supported asset type was found")
    source = _normalized(text)
    lower = source.casefold()
    date_match, date_value = _date_match(source)
    percent_match, rate_value = _percent(source)
    amount_items = _amounts(source, excluded=(date_match, percent_match))

    if re.search(r"\bcredit\s*card\b", lower) or re.search(r"\bhome\s+loan\b", lower):
        loan_kind = "credit_card" if "credit card" in lower else "home"
        return _finish(
            "loan",
            {
                "asset_type": "loan",
                "loan_kind": loan_kind,
                **({"outstanding": amount_items[0][1]} if amount_items else {}),
                **({"rate_pct": rate_value} if rate_value is not None else {}),
            },
            ("loan_kind", "outstanding", "rate_pct"),
        )

    if re.search(r"\b(?:fd|fixed\s+deposit)\b", lower):
        amount = amount_items[0][1] if amount_items else None
        bank = _party(
            source,
            (amount_items[0][0] if amount_items else None, percent_match, date_match),
        )
        return _finish(
            "fd",
            {
                "asset_type": "fd",
                **({"bank": bank} if bank is not None else {}),
                **({"principal": amount} if amount is not None else {}),
                **({"rate_pct": rate_value} if rate_value is not None else {}),
                **({"maturity": date_value} if date_value is not None else {}),
            },
            ("bank", "principal", "rate_pct", "maturity"),
        )

    if re.search(r"\bppf\b", lower):
        return _finish(
            "ppf",
            {
                "asset_type": "ppf",
                **({"balance": amount_items[0][1]} if amount_items else {}),
            },
            ("balance", "opened_fy_start_year"),
        )

    if re.search(r"\bepf\b", lower):
        return _finish(
            "epf",
            {
                "asset_type": "epf",
                **({"balance": amount_items[0][1]} if amount_items else {}),
            },
            ("balance",),
        )

    scheme = next((name for name in ("nsc", "scss") if re.search(rf"\b{name}\b", lower)), None)
    if scheme is not None:
        return _finish(
            "po_scheme",
            {
                "asset_type": "po_scheme",
                "scheme": scheme,
                **({"amount": amount_items[0][1]} if amount_items else {}),
                **({"purchase": date_value} if date_value is not None else {}),
            },
            ("scheme", "amount", "purchase"),
        )

    if re.search(r"\bgold\b", lower):
        grams_match = re.search(r"(?<!\w)(\d+(?:\.\d+)?)\s*g(?:ram|rams)?\b", lower)
        purity_match = re.search(r"(?<!\w)(18|22|24)\s*k\b", lower)
        draft: dict[str, Any] = {"asset_type": "gold", "kind": "physical"}
        if grams_match is not None:
            draft["grams"] = format(Decimal(grams_match.group(1)).normalize(), "f")
        if purity_match is not None:
            draft["purity"] = int(purity_match.group(1))
        return _finish("gold", draft, ("kind", "grams", "purity"))

    if re.search(r"\bsavings\b", lower):
        foreign_amount = _foreign_savings_amount(source)
        amount = amount_items[0] if amount_items else None
        ignored_amount = foreign_amount[0] if foreign_amount else amount[0] if amount else None
        bank = _party(source, (ignored_amount, percent_match, date_match))
        if foreign_amount is not None:
            draft = {
                "asset_type": "savings",
                "balance": foreign_amount[1],
                "currency": foreign_amount[2],
            }
            if bank is not None:
                draft["bank"] = bank
            return _finish("savings", draft, ("bank", "balance"))
        return _finish(
            "savings",
            {
                "asset_type": "savings",
                **({"bank": bank} if bank is not None else {}),
                **({"balance": amount[1]} if amount is not None else {}),
                "currency": "INR",
            },
            ("bank", "balance"),
        )

    if re.search(r"\blic\b", lower):
        amount = amount_items[0] if amount_items else None
        premium = next(
            (
                item
                for item in amount_items[1:]
                if re.search(r"\bpremium\s*$", source[: item[0].start()], re.I)
            ),
            None,
        )
        draft = {"asset_type": "traditional_policy", "insurer": "LIC"}
        if amount is not None:
            draft["sum_assured"] = amount[1]
        if premium is not None:
            draft["annual_premium"] = premium[1]
        return _finish(
            "traditional_policy",
            draft,
            (
                "insurer",
                "plan",
                "start_year",
                "term_years",
                "policy_kind",
                "sum_assured",
                "annual_premium",
            ),
        )

    if re.search(r"\bterm\b", lower):
        amount = amount_items[0] if amount_items else None
        insurer = _party(source, (amount[0] if amount else None, date_match, percent_match))
        year_match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", source)
        draft = {"asset_type": "term_insurance"}
        if insurer is not None:
            draft["insurer"] = insurer
        if amount is not None:
            draft["cover"] = amount[1]
        if year_match is not None:
            draft["ends_year"] = int(year_match.group(1))
        return _finish("term_insurance", draft, ("insurer", "cover", "ends_year"))

    raise QuickAddNotUnderstood("No supported asset type was found")
