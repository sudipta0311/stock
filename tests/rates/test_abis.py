from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

from wealth.rates.abis import import_abis_file, read_abis_isins

_SYNTHETIC_ISIN = "IE00ABC00001"


def _synthetic_workbook(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as workbook:
        workbook.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="2026" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
            "</Relationships>",
        )
        workbook.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<si><t>{_SYNTHETIC_ISIN}</t></si></sst>",
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="B1" t="inlineStr"><is><t>Code</t></is></c></row>'
            '<row r="2"><c r="B2" t="s"><v>0</v></c></row></sheetData></worksheet>',
        )


def test_read_official_list_sheet_from_synthetic_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "synthetic.xlsx"
    _synthetic_workbook(workbook)
    assert read_abis_isins(workbook, 2026) == {_SYNTHETIC_ISIN}


def test_import_updates_status_and_keeps_list_provenance(tmp_path: Path) -> None:
    workbook = tmp_path / "synthetic.xlsx"
    rules_path = tmp_path / "eu_etfs.yaml"
    _synthetic_workbook(workbook)
    rules_path.write_text(
        yaml.safe_dump(
            {"version": 1, "instruments": [{"isin": _SYNTHETIC_ISIN, "abis_listed": False}]}
        ),
        encoding="utf-8",
    )
    assert (
        import_abis_file(
            workbook, year=2026, rules_path=rules_path, source_url="https://skat.dk/synthetic.xlsx"
        )
        == 1
    )
    row = yaml.safe_load(rules_path.read_text(encoding="utf-8"))["instruments"][0]
    assert row["abis_listed"] is True
    assert row["abis_list_year"] == 2026
    assert row["abis_source_url"] == "https://skat.dk/synthetic.xlsx"
