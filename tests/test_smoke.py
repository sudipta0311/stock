from __future__ import annotations

import importlib
import re
from pathlib import Path

import wealth

SUBPACKAGES = (
    "core",
    "schemas",
    "valuation",
    "rates",
    "policy",
    "importer",
    "db",
    "api",
    "marketdata",
    "research",
    "alerts",
)
PAN_PATTERN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def test_version_and_subpackages_import() -> None:
    assert wealth.__version__ == "0.1.0"

    for subpackage in SUBPACKAGES:
        importlib.import_module(f"wealth.{subpackage}")


def test_source_and_tests_contain_no_pan_pattern() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    roots = (repository_root / "src", repository_root / "tests")
    match_count = 0

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            match_count += len(PAN_PATTERN.findall(content))

    assert match_count == 0
