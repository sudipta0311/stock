"""Export Pydantic models as checked-in JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from wealth.schemas.assets import ASSET_MODELS, AnyAsset
from wealth.schemas.profile import FinancialContext, InvestingPrefs, Profile


def schema_documents() -> dict[Path, bytes]:
    """Build all schema files in stable, reviewable JSON formatting."""
    output: dict[Path, bytes] = {}
    for asset_type, model in sorted(ASSET_MODELS.items()):
        output[Path("assets") / f"{asset_type}.json"] = _encode(model.model_json_schema())
    union_schema = TypeAdapter(AnyAsset).json_schema()
    union_schema.pop("discriminator", None)
    output[Path("assets") / "any_asset.json"] = _encode(union_schema)
    output[Path("profile.json")] = _encode(Profile.model_json_schema())
    output[Path("financial_context.json")] = _encode(FinancialContext.model_json_schema())
    output[Path("investing_prefs.json")] = _encode(InvestingPrefs.model_json_schema())
    return output


def _encode(document: dict[str, Any]) -> bytes:
    _remove_decimal_titles(document)
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _remove_decimal_titles(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("format") == "decimal":
            value.pop("title", None)
        for child in value.values():
            _remove_decimal_titles(child)
    elif isinstance(value, list):
        for child in value:
            _remove_decimal_titles(child)


def export_schemas(target: Path | None = None) -> list[Path]:
    """Write schemas to the repository schema directory (or a supplied path)."""
    schema_root = target or Path(__file__).resolve().parents[3] / "schemas"
    written: list[Path] = []
    for relative_path, content in schema_documents().items():
        destination = schema_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        written.append(destination)
    return written


def main() -> None:
    written = export_schemas()
    for path in written:
        print(path.relative_to(Path(__file__).resolve().parents[3]))


if __name__ == "__main__":
    main()
