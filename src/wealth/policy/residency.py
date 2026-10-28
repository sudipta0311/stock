"""Load residency rule data and expose conservative eligibility helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wealth.schemas.profile import Residency

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "residency.yaml"
_COPY_PATH = Path(__file__).resolve().parents[3] / "copy" / "en.yaml"
_STATUSES = ("RI", "NRI", "RNOR")
_ELIGIBILITY = {"allowed", "not_allowed", "conditional"}
_EXISTING = {"continue", "continue_no_extension", "must_close", "conditional"}

_DEBT_RULE_KEYS = {
    "sweep_fd": "bank_fd",
    "po_td_1y": "po_td",
    "po_td_2y": "po_td",
    "po_td_3y": "po_td",
    "po_td_5y": "po_td",
    "target_maturity_fund": "in_mf",
    "gsec_rbi_retail_direct": "rbi_retail_direct_gsec",
    "rbi_frsb": "rbi_frsb",
    "liquid_fund": "in_mf",
    "arbitrage_fund": "in_mf",
}


@lru_cache(maxsize=1)
def load_residency_rules() -> dict[str, Any]:
    """Load and validate the versioned residency rulebook."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        rules = yaml.safe_load(rules_file)
    if not isinstance(rules, dict) or rules.get("version") != 1:
        raise ValueError("unsupported residency rules")
    instruments = rules.get("instruments")
    if not isinstance(instruments, dict) or not instruments:
        raise ValueError("residency rules must define instruments")
    for key, rule in instruments.items():
        _validate_source_entry(rule, f"instruments.{key}")
        by_status = rule.get("by_status")
        if not isinstance(by_status, dict):
            raise ValueError(f"instruments.{key}.by_status must be a mapping")
        for status in _STATUSES:
            eligibility = by_status.get(status)
            if not isinstance(eligibility, dict):
                raise ValueError(f"instruments.{key} must define {status}")
            if eligibility.get("new") not in _ELIGIBILITY:
                raise ValueError(f"instruments.{key}.{status}.new is invalid")
            if eligibility.get("existing") not in _EXISTING:
                raise ValueError(f"instruments.{key}.{status}.existing is invalid")
        country_overrides = rule.get("country_overrides", {})
        if not isinstance(country_overrides, dict):
            raise ValueError(f"instruments.{key}.country_overrides must be a mapping")
        for country, overrides in country_overrides.items():
            if not isinstance(country, str) or not isinstance(overrides, dict):
                raise ValueError(f"instruments.{key}.country_overrides entries are invalid")
            for status, eligibility in overrides.items():
                if status not in _STATUSES or not isinstance(eligibility, dict):
                    raise ValueError(f"instruments.{key}.country_overrides entry is invalid")
                if eligibility.get("new") not in _ELIGIBILITY:
                    raise ValueError(f"instruments.{key}.{country}.{status}.new is invalid")
                if eligibility.get("existing") not in _EXISTING:
                    raise ValueError(f"instruments.{key}.{country}.{status}.existing is invalid")
    for section in ("tax_notes", "resident_only_rules", "us_person_rules", "notes"):
        entries = rules.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError(f"{section} must be a mapping")
        for key, entry in entries.items():
            _validate_source_entry(entry, f"{section}.{key}")
    country_rules = rules.get("country_rules", {})
    if not isinstance(country_rules, dict):
        raise ValueError("country_rules must be a mapping")
    for country, country_rule in country_rules.items():
        if not isinstance(country, str) or not isinstance(country_rule, dict):
            raise ValueError("country_rules entries must be mappings")
        if not isinstance(country_rule.get("india_core_vehicle"), str):
            raise ValueError(f"country_rules.{country}.india_core_vehicle must be text")
        if "local_currency" in country_rule and not isinstance(country_rule["local_currency"], str):
            raise ValueError(f"country_rules.{country}.local_currency must be text")
        for key in ("note_keys", "note_keys_with_indian_funds"):
            values = country_rule.get(key, [])
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError(f"country_rules.{country}.{key} must be a text list")
    denmark = rules.get("denmark")
    if not isinstance(denmark, dict):
        raise ValueError("denmark rules must be a mapping")
    _validate_source_entry(denmark, "denmark")
    for key, entry in denmark.items():
        if isinstance(entry, dict) and "source_url" in entry:
            _validate_source_entry(entry, f"denmark.{key}")
    return rules


def _validate_source_entry(entry: Any, name: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{name} must be a mapping")
    if not isinstance(entry.get("source_url"), str) or not entry["source_url"].startswith(
        "https://"
    ):
        raise ValueError(f"{name}.source_url must be an HTTPS URL")
    if entry.get("verified") is not False:
        raise ValueError(f"{name}.verified must remain false until reviewed")
    if entry.get("last_checked") != "2026-10-28":
        raise ValueError(f"{name}.last_checked must be refreshed for this release")
    if not isinstance(entry.get("needs_review"), bool):
        raise ValueError(f"{name}.needs_review must be boolean")


def eligibility_rule(instrument: str, residency: Residency) -> dict[str, Any] | None:
    """Return the rule for a status/country pair, or ``None`` when unlisted."""
    key = _DEBT_RULE_KEYS.get(instrument, instrument)
    rule = load_residency_rules()["instruments"].get(key)
    if rule is None:
        return None
    countries = rule.get("countries")
    if countries and residency.country not in countries:
        return {"new": "not_allowed", "existing": "conditional"}
    country_override = (
        rule.get("country_overrides", {}).get(residency.country, {}).get(residency.status)
    )
    if country_override is not None:
        return country_override
    return rule["by_status"][residency.status]


def eligible_for_new_money(instrument: str, residency: Residency) -> bool:
    """Allow only rules explicitly marked allowed for a new contribution."""
    rule = eligibility_rule(instrument, residency)
    if rule is None or rule["new"] != "allowed":
        return False
    key = _DEBT_RULE_KEYS.get(instrument, instrument)
    return not residency.us_person or key not in {"in_mf", "in_etf"}


def india_core_vehicle(residency: Residency) -> str:
    """Pick a domestic equity core from the country rulebook."""
    if residency.us_person:
        return "in_equity_direct_core"
    country_rules = load_residency_rules().get("country_rules", {})
    selected = country_rules.get(residency.country, {})
    return str(selected.get("india_core_vehicle", "in_core_index"))


def residency_note_codes(
    residency: Residency,
    *,
    has_indian_funds: bool = False,
    us_situs_exposure: bool = False,
) -> list[str]:
    """Return ordered user-facing note keys for the active residency profile."""
    rules = load_residency_rules()
    notes: list[str] = []
    if residency.country == "OTHER":
        notes.append("residency_country_unavailable")
    if residency.status == "RI":
        return notes
    notes.extend(
        [
            "nri_capital_gains_tds",
            "nre_interest_exempt_in_india",
            "fcnr_interest_exempt_in_india",
            "nro_interest_tds",
            "ppf_nri_eligibility",
            "small_savings_nri_eligibility",
            "direct_equity_nri_pis",
            "frsb_nri_eligibility",
            "resident_only_lrs_tcs",
            "resident_only_schedule_fa",
        ]
    )
    if residency.status == "RNOR":
        notes.append("rnor_foreign_income_window")
    country_rules = rules.get("country_rules", {}).get(residency.country, {})
    notes.extend(country_rules.get("note_keys", []))
    if has_indian_funds:
        notes.extend(country_rules.get("note_keys_with_indian_funds", []))
    eu_eea_countries = set(rules["denmark"]["priips"]["eu_eea_countries"])
    if residency.country in eu_eea_countries:
        notes.append("eu_priips_kid")
    if not residency.us_person and us_situs_exposure:
        notes.append("estate_tax_us_domiciled_assets")
    if residency.us_person:
        notes.extend(
            [
                "pfic_avoid",
                "fbar_fatca_reporting",
                "us_tax_applies",
            ]
        )
        if has_indian_funds:
            notes.append("pfic_holding_review")
    return list(dict.fromkeys(notes))


@lru_cache(maxsize=1)
def _copy() -> dict[str, Any]:
    with _COPY_PATH.open(encoding="utf-8") as copy_file:
        raw = yaml.safe_load(copy_file)
    if not isinstance(raw, dict):
        raise ValueError("copy/en.yaml must contain a mapping")
    return raw


def jurisdiction_disclaimer(residency: Residency) -> str:
    """Select the copy disclaimer without doing any jurisdiction tax calculation."""
    copy = _copy()
    disclaimers = copy["disclaimers"]
    if residency.status == "RI":
        return copy["email"]["disclaimer"]
    eu_eea_countries = set(load_residency_rules()["denmark"]["priips"]["eu_eea_countries"])
    if residency.country in eu_eea_countries:
        return disclaimers["eu_eea"]
    return disclaimers["international"]


def deployment_disclaimer(mode: str, residency: Residency) -> str:
    """Choose the deployment-level copy, retaining jurisdiction copy in public mode."""
    if mode == "personal":
        return str(_copy()["disclaimers"]["personal"])
    if mode == "public":
        return jurisdiction_disclaimer(residency)
    raise ValueError("deployment_mode must be personal or public")
