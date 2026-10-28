import json
import re
from decimal import Decimal
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from wealth.schemas.assets import (
    ASSET_MODELS,
    AnyAsset,
    default_liquidity,
    default_sleeves,
    parse_asset,
)
from wealth.schemas.common import LiquidityClass, Sleeve

ASSET_CASES: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "savings": ({"bank": "Test Bank", "balance": "₹5,00,000"}, {"bank": "Test Bank"}),
    "fd": (
        {"bank": "Test Bank", "principal": "5L", "rate_pct": "7.10", "maturity": "Mar 2027"},
        {"bank": "Test Bank", "principal": 5000, "maturity": "2027-03"},
    ),
    "rd": (
        {
            "institution": "Test Post Office",
            "monthly_amount": 1000,
            "rate_pct": "6.7",
            "maturity": "2027-03",
        },
        {"institution": "Test Post Office", "monthly_amount": 1000, "maturity": "2027-03"},
    ),
    "ppf": ({"balance": 1000, "opened_fy_start_year": 2020}, {"balance": 1000}),
    "epf": ({"balance": 1000, "balance_as_of": "2025-03"}, {}),
    "nps": ({"tier1_balance": 1000}, {}),
    "po_scheme": (
        {"scheme": "td", "amount": 1000, "purchase": "2025-03", "td_years": 3},
        {"scheme": "td", "amount": 1000, "purchase": "2025-03"},
    ),
    "bond": (
        {"kind": "gsec", "face_value": 1000, "maturity": "2030-03"},
        {"kind": "gsec", "face_value": 1000},
    ),
    "gold": (
        {"kind": "physical", "grams": "10", "purity": 22, "purpose": "investment"},
        {"kind": "physical", "grams": "10"},
    ),
    "term_insurance": (
        {"insurer": "Test Assurance", "cover": 5000000, "ends_year": 2040},
        {"insurer": "Test Assurance", "ends_year": 2040},
    ),
    "health_insurance": (
        {"insurer": "Test Assurance", "cover": 500000, "renewal_month": 4},
        {"insurer": "Test Assurance", "cover": 500000, "renewal_month": 13},
    ),
    "traditional_policy": (
        {
            "insurer": "Test Assurance",
            "plan": "Savings Plan",
            "policy_kind": "endowment",
            "sum_assured": 500000,
            "annual_premium": 25000,
            "start_year": 2020,
            "term_years": 20,
        },
        {"insurer": "Test Assurance", "policy_kind": "endowment"},
    ),
    "ulip": (
        {"insurer": "Test Assurance", "fund_value": 50000, "annual_premium": 12000},
        {"insurer": "Test Assurance", "annual_premium": 12000},
    ),
    "property": (
        {"property_kind": "residential", "estimated_value": 8000000},
        {"property_kind": "residential"},
    ),
    "loan": (
        {"loan_kind": "home", "outstanding": 1000000, "rate_pct": "8.5"},
        {"loan_kind": "home", "outstanding": 1000000},
    ),
}


@pytest.mark.parametrize("asset_type", list(ASSET_MODELS))
def test_each_asset_model_accepts_valid_and_rejects_invalid_data(asset_type: str) -> None:
    valid, invalid = ASSET_CASES[asset_type]
    asset_data = {"asset_type": asset_type, **valid}
    parsed = ASSET_MODELS[asset_type].model_validate(asset_data)
    assert parsed.asset_type == asset_type
    assert isinstance(parse_asset(asset_data), ASSET_MODELS[asset_type])

    with pytest.raises(ValidationError):
        ASSET_MODELS[asset_type].model_validate({"asset_type": asset_type, **invalid})


@pytest.mark.parametrize("asset_type", list(ASSET_MODELS))
def test_discriminated_union_parses_plain_dict_for_each_asset_type(asset_type: str) -> None:
    valid, _ = ASSET_CASES[asset_type]
    parsed: AnyAsset = parse_asset({"asset_type": asset_type, **valid})
    assert parsed.asset_type == asset_type
    assert "E" not in parsed.model_dump_json()


@pytest.mark.parametrize(
    "asset_type", [asset_type for asset_type in ASSET_MODELS if asset_type != "loan"]
)
def test_default_sleeves_sum_to_one_for_each_asset(asset_type: str) -> None:
    valid, _ = ASSET_CASES[asset_type]
    sleeves = default_sleeves(parse_asset({"asset_type": asset_type, **valid}))
    assert sum(sleeves.values(), Decimal(0)) == Decimal(1)


def test_loan_has_no_sleeves_and_cash_is_instant() -> None:
    assert default_sleeves(parse_asset({"asset_type": "loan", **ASSET_CASES["loan"][0]})) == {}
    savings = parse_asset({"asset_type": "savings", **ASSET_CASES["savings"][0]})
    assert default_sleeves(savings) == {Sleeve.CASH: Decimal(1)}
    assert default_liquidity(savings) is LiquidityClass.INSTANT


def test_nps_scheme_mix_is_validated_and_mapped_to_sleeves() -> None:
    nps = parse_asset(
        {
            "asset_type": "nps",
            "tier1_balance": 1000,
            "equity_pct": "50",
            "corporate_debt_pct": "20",
            "govt_debt_pct": "20",
            "alternate_pct": "10",
        }
    )
    assert default_sleeves(nps) == {
        Sleeve.EQUITY_INDIA: Decimal("0.5"),
        Sleeve.DEBT: Decimal("0.5"),
    }
    with pytest.raises(ValidationError):
        parse_asset(
            {
                "asset_type": "nps",
                "tier1_balance": 1000,
                "equity_pct": "50",
                "corporate_debt_pct": "20",
                "govt_debt_pct": "20",
                "alternate_pct": "5",
            }
        )


def test_savings_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        parse_asset({"asset_type": "savings", "bank": "Example", "balance": 2, "extra": True})


def test_month_year_fields_serialize_as_canonical_strings() -> None:
    fd = parse_asset(
        {
            "asset_type": "fd",
            "bank": "Test Bank",
            "principal": 1000,
            "rate_pct": "7.10",
            "maturity": "Mar 2027",
        }
    )
    assert fd.model_dump(mode="json")["maturity"] == "2027-03"


@pytest.mark.parametrize("value", ["5L", 500000, Decimal("5E+5")])
def test_rupees_normalize_all_input_forms_to_two_fixed_decimals(value: object) -> None:
    asset = parse_asset({"asset_type": "savings", "bank": "Test Bank", "balance": value})
    assert asset.balance == Decimal("500000.00")
    assert asset.balance.as_tuple().exponent == -2
    assert '"balance":"500000.00"' in asset.model_dump_json()


def test_rate_serialization_is_fixed_point_without_exponent() -> None:
    fd = parse_asset(
        {
            "asset_type": "fd",
            "bank": "Test Bank",
            "principal": "5L",
            "rate_pct": "7E+0",
            "maturity": "2027-03",
        }
    )
    assert '"principal":"500000.00"' in fd.model_dump_json()
    assert '"rate_pct":"7"' in fd.model_dump_json()


@pytest.mark.parametrize("asset_type", list(ASSET_MODELS))
def test_serialized_asset_examples_validate_against_exported_schemas(asset_type: str) -> None:
    valid, _ = ASSET_CASES[asset_type]
    asset = parse_asset({"asset_type": asset_type, **valid})
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas" / "assets" / f"{asset_type}.json").read_text("utf-8"))
    Draft202012Validator(schema).validate(asset.model_dump(mode="json"))


def test_exponent_and_negative_money_strings_fail_the_fd_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas" / "assets" / "fd.json").read_text("utf-8"))
    validator = Draft202012Validator(schema)
    example = parse_asset({"asset_type": "fd", **ASSET_CASES["fd"][0]}).model_dump(mode="json")
    for invalid_principal in ("5E+5", "-5.00"):
        invalid = {**example, "principal": invalid_principal}
        assert not validator.is_valid(invalid)


def test_loan_rate_schema_accepts_two_digit_rates_up_to_sixty() -> None:
    loan = parse_asset(
        {
            "asset_type": "loan",
            "loan_kind": "credit_card",
            "outstanding": "25000",
            "rate_pct": "36",
        }
    )
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas" / "assets" / "loan.json").read_text("utf-8"))
    rate_schema = schema["properties"]["rate_pct"]
    fd_schema = json.loads((root / "schemas" / "assets" / "fd.json").read_text("utf-8"))

    assert loan.rate_pct == Decimal("36")
    assert rate_schema["x-maximum"] == "60"
    assert re.fullmatch(rate_schema["pattern"], "36.5")
    assert not re.fullmatch(rate_schema["pattern"], "100")
    assert fd_schema["properties"]["rate_pct"]["x-maximum"] == "30"

    with pytest.raises(ValidationError):
        parse_asset(
            {
                "asset_type": "fd",
                "bank": "Demo Bank",
                "principal": "10000",
                "rate_pct": "36",
                "maturity": "2030-01",
            }
        )

    with pytest.raises(ValidationError):
        parse_asset(
            {
                "asset_type": "bond",
                "kind": "gsec",
                "face_value": "10000",
                "maturity": "2030-01",
                "coupon_pct": "36",
            }
        )
