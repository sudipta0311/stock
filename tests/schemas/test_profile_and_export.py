import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from wealth.schemas import FinancialContext
from wealth.schemas.export import schema_documents
from wealth.schemas.profile import (
    Goal,
    GoalType,
    HouseholdMember,
    Profile,
    Residency,
    ResidencyInput,
    TaxRegime,
    residency_from_stored,
)


def test_profile_and_goal_models_validate_expected_values() -> None:
    profile = Profile(tax_regime=TaxRegime.NEW, tax_slab_pct="20", city_tier=1)
    goal = Goal(
        name="Emergency fund",
        goal_type=GoalType.EMERGENCY,
        target_amount="5L",
        target_year=2030,
        currency="INR",
        priority=1,
    )
    member = HouseholdMember(
        id="00000000-0000-0000-0000-000000000001",
        display_name="Household member",
        relationship="self",
    )
    assert profile.timezone == "Asia/Kolkata"
    assert goal.target_amount == 500000
    assert member.display_name == "Household member"


def test_financial_context_defaults_and_schema_package_export() -> None:
    context = FinancialContext(monthly_expenses="50000", annual_income=None)

    assert context.dependants == 0
    assert context.employment_type == "salaried"
    assert context.income_stability == "medium"
    assert context.city_tier is None


def test_profile_models_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        Profile(tax_regime="new", city_tier=4)
    with pytest.raises(ValidationError):
        Goal(
            name="",
            goal_type="emergency",
            target_amount=1000,
            target_year=2030,
            currency="INR",
            priority=6,
        )


def test_residency_and_goal_currencies_validate_supported_values() -> None:
    profile = Profile(tax_regime=TaxRegime.NEW, residency={"status": "NRI", "country": "DK"})
    assert profile.residency.us_person is False
    assert Residency(status="NRI", country="DK").us_person is False
    assert "us_person" not in ResidencyInput(status="NRI", country="DK").model_dump()
    for currency in ("INR", "DKK", "EUR", "USD"):
        Goal(
            name=f"Synthetic {currency} goal",
            goal_type="education_abroad",
            target_amount="1000",
            target_year=2032,
            currency=currency,
            priority=1,
        )
    assert Residency(status="NRI").country == "OTHER"
    assert Residency(status="RNOR").country == "OTHER"
    with pytest.raises(ValidationError, match="planned_return_year"):
        Residency(status="NRI", country="DK", planned_return_year=2025)
    for country in ("US", "XX", "OTHER"):
        with pytest.raises(ValidationError):
            ResidencyInput(status="NRI", country=country)
    with pytest.raises(ValidationError):
        ResidencyInput(status="NRI", country="DK", us_person=True)
    with pytest.raises(ValidationError):
        ResidencyInput(status="RI", country="DK")
    legacy = residency_from_stored({"status": "NRI", "country": "US", "us_person": True})
    assert legacy.country == "OTHER"
    assert legacy.us_person is True
    with pytest.raises(ValidationError):
        Goal(
            name="Unsupported currency",
            goal_type="home",
            target_amount="1000",
            target_year=2032,
            currency="JPY",
            priority=1,
        )


def test_schema_files_match_fresh_deterministic_export() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    schema_root = repository_root / "schemas"
    expected = schema_documents()
    actual_files = {
        path.relative_to(schema_root): path.read_bytes()
        for path in schema_root.rglob("*.json")
        if path.name != "openapi.json"
    }
    assert actual_files == expected


def test_every_exported_schema_passes_draft_2020_12_meta_schema() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    for schema_path in (repository_root / "schemas").rglob("*.json"):
        Draft202012Validator.check_schema(json.loads(schema_path.read_text("utf-8")))


def test_money_schemas_use_canonical_decimal_strings() -> None:
    fd = json.loads(schema_documents()[Path("assets/fd.json")])
    assert fd["properties"]["principal"] == {
        "type": "string",
        "pattern": r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$",
        "format": "decimal",
        "x-unit": "INR",
    }
    assert fd["properties"]["rate_pct"] == {
        "type": "string",
        "pattern": r"^(0|[1-9][0-9]?)(\.[0-9]{1,4})?$",
        "format": "decimal",
        "x-minimum": "0",
        "x-maximum": "30",
    }
    union = json.loads(schema_documents()[Path("assets/any_asset.json")])
    assert "discriminator" not in union


def test_exported_schemas_use_only_standard_or_extension_keywords() -> None:
    standard_keywords = set(Draft202012Validator.VALIDATORS) | {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "title",
        "description",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        "format",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
    }
    schema_maps = {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
    schema_children = {
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "items",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
        "unevaluatedItems",
    }
    schema_lists = {"allOf", "anyOf", "oneOf", "prefixItems"}

    def check_schema_object(schema: object) -> None:
        if not isinstance(schema, dict):
            return
        assert all(key in standard_keywords or key.startswith("x-") for key in schema)
        for keyword in schema_maps:
            children = schema.get(keyword)
            if isinstance(children, dict):
                for child in children.values():
                    check_schema_object(child)
        for keyword in schema_children:
            check_schema_object(schema.get(keyword))
        for keyword in schema_lists:
            for child in schema.get(keyword, []):
                check_schema_object(child)

    for document in schema_documents().values():
        check_schema_object(json.loads(document))
