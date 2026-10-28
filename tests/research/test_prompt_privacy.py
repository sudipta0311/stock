"""Research code may use market facts, but not household records, in prompts."""

from __future__ import annotations

import ast
from pathlib import Path

_PERSONAL_MODELS = {
    "Account",
    "HoldingLot",
    "HouseholdAccess",
    "HouseholdMember",
    "ImportJob",
    "ManualAsset",
    "Profile",
    "User",
}
_PERSONAL_FIELDS = {
    "account_number",
    "address",
    "date_of_birth",
    "display_name",
    "email",
    "financial_context",
    "household_id",
    "monthly_expenses",
    "pan",
    "phone",
    "salary",
    "user_id",
}
_MODEL_CALLS = {"complete", "generate", "invoke", "route", "send", "submit"}


def _constant_mapping_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            for key in child.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value.casefold())
    return keys


def test_research_model_calls_do_not_read_personal_records_or_prompt_fields() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "wealth" / "research"
    violations: list[str] = []

    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "wealth.db.models":
                imported_models = {item.name for item in node.names}
                if imported_models & _PERSONAL_MODELS:
                    violations.append(path.name)
            if isinstance(node, ast.Call):
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if function_name.casefold() in _MODEL_CALLS:
                    keys = _constant_mapping_keys(node)
                    keyword_keys = {
                        keyword.arg.casefold()
                        for keyword in node.keywords
                        if keyword.arg is not None
                    }
                    if (keys | keyword_keys) & _PERSONAL_FIELDS:
                        violations.append(path.name)

    assert violations == []
