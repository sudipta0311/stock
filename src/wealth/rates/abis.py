"""Import official Danish ABIS Excel lists into the curated fund rules."""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import yaml

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "eu_etfs.yaml"
_ABIS_URL = "https://skat.dk/media/fo5m24x5/september-2026-abis-liste-2021-2026.xlsx"
_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_DOC = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REL_PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def read_abis_isins(path: Path, year: int) -> set[str]:
    """Read ISINs from the year-named sheet's second column in a SKAT XLSX."""
    with zipfile.ZipFile(path) as workbook:
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{_REL_PACKAGE}Relationship")
        }
        sheet_target: str | None = None
        for sheet in workbook_root.findall(f".//{_MAIN}sheet"):
            if sheet.attrib.get("name") == str(year):
                sheet_target = targets.get(sheet.attrib.get(f"{_REL_DOC}id", ""))
                break
        if sheet_target is None:
            raise ValueError(f"The workbook has no {year} sheet")
        sheet_path = posixpath.normpath(
            sheet_target.lstrip("/")
            if sheet_target.startswith("/")
            else posixpath.join("xl", sheet_target)
        )
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = [
                "".join(text.text or "" for text in string.findall(f".//{_MAIN}t"))
                for string in strings_root.findall(f"{_MAIN}si")
            ]
        sheet_root = ET.fromstring(workbook.read(sheet_path))
        values: set[str] = set()
        for cell in sheet_root.findall(f".//{_MAIN}c"):
            if not re.fullmatch(r"B\d+", cell.attrib.get("r", "")):
                continue
            value_node = cell.find(f"{_MAIN}v")
            if value_node is None or value_node.text is None:
                continue
            value = value_node.text.strip().upper()
            if cell.attrib.get("t") == "s":
                try:
                    value = shared[int(value)].strip().upper()
                except (IndexError, ValueError):
                    continue
            if _ISIN.fullmatch(value):
                values.add(value)
        return values


def import_abis_file(
    path: Path,
    *,
    year: int = 2026,
    rules_path: Path = _RULES_PATH,
    source_url: str = _ABIS_URL,
) -> int:
    """Update curated ISIN statuses and preserve official list provenance."""
    document = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("instruments"), list):
        raise ValueError("eu_etfs.yaml must contain an instruments list")
    isins = read_abis_isins(path, year)
    changed = 0
    for instrument in document["instruments"]:
        if not isinstance(instrument, dict) or not isinstance(instrument.get("isin"), str):
            raise ValueError("Every curated instrument must have an ISIN")
        instrument["abis_listed"] = instrument["isin"].upper() in isins
        instrument["abis_list_year"] = year
        instrument["abis_source_url"] = source_url
        changed += 1
    rules_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import an official SKAT ABIS workbook")
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import", help="update the curated ETF ABIS statuses")
    importer.add_argument("file", type=Path)
    importer.add_argument("--year", type=int, default=2026)
    importer.add_argument("--source-url", default=_ABIS_URL)
    args = parser.parse_args(argv)
    try:
        count = import_abis_file(args.file, year=args.year, source_url=args.source_url)
    except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"ABIS import failed: {type(error).__name__}", file=sys.stderr)
        return 2
    print(f"Updated ABIS status for {count} curated instruments ({args.year}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
