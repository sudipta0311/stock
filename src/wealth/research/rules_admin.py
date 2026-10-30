"""Human-only review CLI for applying watcher proposals to local rule YAML."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from wealth.db.models import RuleChangeProposal
from wealth.db.session import make_engine

_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"


def _pointer(document: dict[str, Any], pointer: str, *, create: bool) -> tuple[dict[str, Any], str]:
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/") if part]
    if not parts:
        raise ValueError("Proposal path must point to a YAML value")
    parent = document
    for part in parts[:-1]:
        value = parent.get(part)
        if value is None and create:
            value = {}
            parent[part] = value
        if not isinstance(value, dict):
            raise ValueError("Proposal path does not point through YAML mappings")
        parent = value
    return parent, parts[-1]


def apply_proposal_to_rules(proposal: RuleChangeProposal | dict[str, Any], rules_dir: Path) -> Path:
    """Apply a pending, value-bearing proposal to a rules file and mark its period verified."""
    values = (
        {
            "path": proposal.path,
            "kind": proposal.kind,
            "proposed_value": proposal.proposed_value,
            "evidence": proposal.evidence,
        }
        if isinstance(proposal, RuleChangeProposal)
        else proposal
    )
    if not isinstance(values.get("proposed_value"), dict):
        raise ValueError("This proposal has no code-verified value to approve")
    file_part, separator, pointer = str(values["path"]).partition("#")
    if not separator:
        raise ValueError("Proposal path must include a YAML pointer")
    relative = Path(file_part)
    if relative.parts and relative.parts[0] == "rules":
        relative = Path(*relative.parts[1:])
    root = rules_dir.resolve()
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError("Proposal path is outside the rules directory or missing")
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Target rule file must contain a YAML mapping")
    parent, key = _pointer(document, pointer, create=True)
    proposed = dict(values["proposed_value"])
    proposed["verified"] = True
    target_value = parent.get(key)
    if isinstance(target_value, list) or key.endswith("periods") or key == "watcher_proposals":
        entries = target_value if isinstance(target_value, list) else []
        if key == "watcher_proposals":
            entries.append(
                {
                    "kind": values["kind"],
                    "value": proposed,
                    "evidence": values.get("evidence", {}),
                    "verified": True,
                }
            )
        else:
            match_keys = ("from", "to") if "from" in proposed else ("fy",)
            existing = next(
                (
                    item
                    for item in entries
                    if isinstance(item, dict)
                    and all(item.get(name) == proposed.get(name) for name in match_keys)
                ),
                None,
            )
            if existing is None:
                entries.append(proposed)
            else:
                existing.update(proposed)
        parent[key] = entries
    else:
        if target_value is not None and not isinstance(target_value, dict):
            parent[key] = proposed.get("value", proposed)
        else:
            parent[key] = proposed
    target.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def _proposal_json(proposal: RuleChangeProposal) -> dict[str, Any]:
    return {
        "id": str(proposal.id),
        "kind": proposal.kind,
        "path": proposal.path,
        "current_value": proposal.current_value,
        "proposed_value": proposal.proposed_value,
        "evidence": proposal.evidence,
        "status": proposal.status,
        "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
        "decided_at": proposal.decided_at.isoformat() if proposal.decided_at else None,
    }


def list_proposals(engine: Engine) -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.scalars(
            select(RuleChangeProposal).order_by(
                RuleChangeProposal.created_at.desc(), RuleChangeProposal.id
            )
        )
        return [_proposal_json(row) for row in rows]


def get_proposal(engine: Engine, proposal_id: UUID) -> RuleChangeProposal | None:
    with Session(engine) as session:
        return session.get(RuleChangeProposal, proposal_id)


def decide_proposal(
    engine: Engine,
    proposal_id: UUID,
    *,
    decision: str,
    admin_name: str,
    rules_dir: Path | None = None,
) -> None:
    if decision not in {"approved", "rejected"}:
        raise ValueError("Decision must be approved or rejected")
    name = admin_name.strip()
    if not name or len(name) > 120 or "\n" in name or "\r" in name:
        raise ValueError("ADMIN_NAME must be set to a short display name")
    with Session(engine) as session:
        proposal = session.get(RuleChangeProposal, proposal_id)
        if proposal is None:
            raise ValueError("Proposal was not found")
        if proposal.status != "pending":
            raise ValueError("Only pending proposals can be decided")
        if decision == "approved":
            apply_proposal_to_rules(proposal, rules_dir or _RULES_DIR)
        proposal.status = decision
        proposal.decided_at = datetime.now(UTC)
        proposal.decided_by = name
        session.commit()


def _engine() -> Engine:
    database_url = os.getenv("OWNER_DATABASE_URL")
    if not database_url:
        raise RuntimeError("OWNER_DATABASE_URL is required")
    return make_engine(database_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wealth.research.rules_admin")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for name in ("show", "approve", "reject"):
        command = subparsers.add_parser(name)
        command.add_argument("id")
    args = parser.parse_args(argv)
    engine: Engine | None = None
    try:
        engine = _engine()
        if args.command == "list":
            print(json.dumps(list_proposals(engine), default=str, sort_keys=True))
            return 0
        try:
            proposal_id = UUID(args.id)
        except ValueError:
            print(json.dumps({"status": "not_found"}))
            return 3
        if args.command == "show":
            proposal = get_proposal(engine, proposal_id)
            print(
                json.dumps(_proposal_json(proposal), default=str, sort_keys=True)
                if proposal
                else json.dumps({"status": "not_found"})
            )
            return 0 if proposal else 3
        admin_name = os.getenv("ADMIN_NAME", "")
        decide_proposal(
            engine,
            proposal_id,
            decision="approved" if args.command == "approve" else "rejected",
            admin_name=admin_name,
        )
        print(
            json.dumps(
                {
                    "id": str(proposal_id),
                    "status": "approved" if args.command == "approve" else "rejected",
                }
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__}))
        return 3
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
