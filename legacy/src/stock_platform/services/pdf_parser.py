from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ISIN_RE = re.compile(r"^[A-Z]{3}[A-Z0-9]{9}$")
NUMERIC_RE = re.compile(r"[0-9,]+\.\d+|[0-9]+")


def _to_float(value: str) -> float:
    return float(value.replace(",", ""))


def _clean_lines(text: str) -> list[str]:
    raw = [line.strip() for line in text.splitlines()]
    filtered = [line for line in raw if line]
    merged: list[str] = []
    for line in filtered:
        if merged and re.fullmatch(r"\d+", line) and re.search(r"[0-9,]+\.\d+$", merged[-1]):
            merged[-1] = merged[-1] + line
            continue
        merged.append(line)
    return merged


@dataclass(slots=True)
class ParsedPortfolio:
    direct_equities: list[dict[str, Any]]
    mutual_funds: list[dict[str, Any]]
    etfs: list[dict[str, Any]]
    total_portfolio_value: float
    statement_month: str = ""

    def to_payload(self) -> dict[str, Any]:
        direct_value = round(sum(item["market_value"] for item in self.direct_equities), 2)
        return {
            "macro_thesis": "",
            "investable_surplus": round(max(self.total_portfolio_value * 0.03, 100000.0), 2) if self.total_portfolio_value else 100000.0,
            "direct_equity_corpus": direct_value,
            "mutual_funds": self.mutual_funds,
            "etfs": self.etfs,
            "direct_equities": self.direct_equities,
            "statement_month": self.statement_month,
        }


class NSDLCASParser:
    def parse_file(self, pdf_path: str | Path, password: str) -> ParsedPortfolio:
        with open(pdf_path, "rb") as handle:
            return self.parse_bytes(handle.read(), password=password)

    def parse_bytes(self, pdf_bytes: bytes, password: str) -> ParsedPortfolio:
        reader = PdfReader(io.BytesIO(pdf_bytes), password=password)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        page_lines = [_clean_lines(text) for text in page_texts]
        total_portfolio_value = self._extract_total_portfolio_value(page_lines)
        direct_equities = self._extract_direct_equities(page_lines)
        etfs = self._extract_etfs(page_lines)
        mutual_funds = self._extract_mutual_funds(page_lines)
        return ParsedPortfolio(
            direct_equities=direct_equities,
            mutual_funds=mutual_funds,
            etfs=etfs,
            total_portfolio_value=total_portfolio_value,
            statement_month=self._extract_statement_month(page_lines),
        )

    def _extract_total_portfolio_value(self, page_lines: list[list[str]]) -> float:
        for lines in page_lines[:3]:
            for line in lines:
                match = re.search(r"YOUR CONSOLIDATED PORTFOLIO VALUE `\s*([0-9,]+\.\d+)", line)
                if match:
                    return _to_float(match.group(1))
        return 0.0

    def _extract_statement_month(self, page_lines: list[list[str]]) -> str:
        month_map = {
            "jan": "01",
            "feb": "02",
            "mar": "03",
            "apr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "aug": "08",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dec": "12",
        }
        for lines in page_lines[:3]:
            for line in lines:
                match = re.search(r"Statement for the period from \d{2}-([A-Za-z]{3})-(\d{4}) to \d{2}-[A-Za-z]{3}-\d{4}", line)
                if match:
                    month = month_map.get(match.group(1).lower())
                    if month:
                        return f"{match.group(2)}-{month}"
        return ""

    def _collect_section_lines(
        self,
        page_lines: list[list[str]],
        *,
        start_markers: tuple[str, ...],
        end_markers: tuple[str, ...],
    ) -> list[str]:
        section_lines: list[str] = []
        active = False
        for lines in page_lines:
            for line in lines:
                if line in start_markers:
                    active = True
                    continue
                if active and (line in end_markers or any(line.startswith(marker) for marker in end_markers)):
                    return section_lines
                if active:
                    section_lines.append(line)
        return section_lines

    def _extract_direct_equities(self, page_lines: list[list[str]]) -> list[dict[str, Any]]:
        section_lines = self._collect_section_lines(
            page_lines,
            start_markers=("Equity Shares",),
            end_markers=("Mutual Funds (M)", "Mutual Fund Folios (F)", "Notes:"),
        )
        records: list[dict[str, Any]] = []
        idx = 0
        while idx < len(section_lines):
            line = section_lines[idx]
            if not ISIN_RE.match(line):
                idx += 1
                continue
            isin = line
            if idx + 1 >= len(section_lines):
                break
            symbol = section_lines[idx + 1].replace(".NSE", "").replace(".BSE", "")
            idx += 2
            desc_parts: list[str] = []
            while idx < len(section_lines):
                current = section_lines[idx]
                if current.startswith("Sub Total"):
                    break
                combined = " ".join(desc_parts + [current]).strip()
                match = re.match(
                    r"^(?P<company>.+?)\s+(?P<face>[0-9,]+\.\d+)\s+(?P<shares>[0-9,]+)\s+(?P<price>[0-9,]+\.\d+)\s+(?P<value>[0-9,]+\.\d+)$",
                    combined,
                )
                if match:
                    records.append(
                        {
                            "instrument_name": match.group("company"),
                            "symbol": symbol,
                            "isin": isin,
                            "quantity": int(match.group("shares").replace(",", "")),
                            "market_value": _to_float(match.group("value")),
                        }
                    )
                    idx += 1
                    break
                desc_parts.append(current)
                idx += 1
        return records

    def _extract_etfs(self, page_lines: list[list[str]]) -> list[dict[str, Any]]:
        section_lines = self._collect_section_lines(
            page_lines,
            start_markers=("Mutual Funds (M)",),
            end_markers=("Sub Total", "Mutual Fund Folios (F)", "Notes:"),
        )
        records: list[dict[str, Any]] = []
        idx = 0
        while idx < len(section_lines):
            line = section_lines[idx]
            combined_match = re.match(
                r"^(?P<isin>[A-Z]{3}[A-Z0-9]{9})\s+(?P<name>.+?)\s+(?P<units>[0-9,]+)\s+(?P<nav>[0-9,]+\.\d+)\s+(?P<value>[0-9,]+\.\d+)$",
                line,
            )
            if combined_match:
                records.append(
                    {
                        "instrument_name": combined_match.group("name"),
                        "isin": combined_match.group("isin"),
                        "quantity": int(combined_match.group("units").replace(",", "")),
                        "market_value": _to_float(combined_match.group("value")),
                    }
                )
                idx += 1
                continue
            if ISIN_RE.match(line) and idx + 1 < len(section_lines):
                combined = section_lines[idx + 1]
                match = re.match(
                    r"^(?P<name>.+?)\s+(?P<units>[0-9,]+)\s+(?P<nav>[0-9,]+\.\d+)\s+(?P<value>[0-9,]+\.\d+)$",
                    combined,
                )
                if match:
                    records.append(
                        {
                            "instrument_name": match.group("name"),
                            "isin": line,
                            "quantity": int(match.group("units").replace(",", "")),
                            "market_value": _to_float(match.group("value")),
                        }
                    )
                idx += 2
            else:
                idx += 1
        return records

    def _extract_mutual_funds(self, page_lines: list[list[str]]) -> list[dict[str, Any]]:
        section_lines = self._collect_section_lines(
            page_lines,
            start_markers=("Mutual Fund Folios (F)",),
            end_markers=("Notes:", "Transactions for the period"),
        )

        records: list[dict[str, Any]] = []
        idx = 0
        while idx < len(section_lines):
            line = section_lines[idx]
            if not ISIN_RE.match(line):
                idx += 1
                continue
            isin = line
            idx += 1
            if idx >= len(section_lines):
                break
            ucc = section_lines[idx]
            idx += 1
            description_parts: list[str] = []
            while idx < len(section_lines):
                current = section_lines[idx]
                if ISIN_RE.match(current):
                    break
                numeric_tokens = NUMERIC_RE.findall(current)
                if numeric_tokens and len(numeric_tokens) >= 6 and current.split()[0].replace("/", "").replace("-", "").isalnum():
                    folio = current.split()[0]
                    current_value = self._extract_current_value(numeric_tokens)
                    description = " ".join(description_parts).replace("  ", " ").strip(" -")
                    if description:
                        records.append(
                            {
                                "instrument_name": description,
                                "folio_no": folio,
                                "isin": isin,
                                "ucc": ucc,
                                "market_value": current_value,
                            }
                        )
                    idx += 1
                    break
                description_parts.append(current)
                idx += 1
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in records:
            key = (row["isin"], row["folio_no"])
            deduped[key] = row
        return list(deduped.values())

    def _extract_current_value(self, numeric_tokens: list[str]) -> float:
        if len(numeric_tokens) >= 8:
            return _to_float(numeric_tokens[-3])
        if len(numeric_tokens) >= 7:
            return _to_float(numeric_tokens[-2])
        return _to_float(numeric_tokens[-1])
