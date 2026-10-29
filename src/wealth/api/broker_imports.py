"""Broker upload staging helpers with process-only pending rows for column review."""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from wealth.core.ids import isin_check_digit_ok
from wealth.core.money import round_paise
from wealth.db.models import (
    Account,
    HoldingLot,
    ImportItem,
    ImportJob,
    ImportTemplate,
    RealisedTrade,
    Security,
)
from wealth.importer.brokers import (
    BrokerHolding,
    Layout,
    RealisedLot,
    Trade,
    detect_layout,
    fifo_lots,
    header_fingerprint,
    load_layouts,
    normalize_header,
    parse_holdings,
    parse_tradebook,
    read_table,
    suggested_mapping,
)

_PENDING_TTL_SECONDS = 20 * 60
_PENDING_LIMIT = 8
_pending_lock = threading.Lock()
_pending_rows: OrderedDict[UUID, tuple[float, list[str], list[dict[str, Any]]]] = OrderedDict()
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def parse_broker_upload(data: bytes, filename: str) -> tuple[list[str], list[dict[str, Any]]]:
    return read_table(data, filename)


def available_templates(session: Session, user_id: UUID) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(ImportTemplate)
        .where(ImportTemplate.user_id == user_id)
        .order_by(ImportTemplate.created_at)
    )
    return [
        {
            "name": row.name,
            "kind": row.kind,
            "header_fingerprint": row.header_fingerprint,
            "header_names": list(row.header_names),
            "columns": dict(row.columns),
        }
        for row in rows
    ]


def find_layout(headers: list[str], templates: list[dict[str, Any]]) -> Layout | None:
    return detect_layout(headers, load_layouts(), templates)


def masked_samples(headers: list[str], rows: list[dict[str, Any]]) -> list[list[str]]:
    samples: list[list[str]] = []
    for row in rows[:3]:
        values: list[str] = []
        for header in headers:
            raw = row.get(header)
            value = str(raw) if raw is not None else ""
            value = re.sub(r"[A-Za-z]", "x", value)
            value = re.sub(r"\d", "9", value)
            values.append(value)
        samples.append(values)
    return samples


def keep_pending_rows(job_id: UUID, headers: list[str], rows: list[dict[str, Any]]) -> None:
    now = time.monotonic()
    with _pending_lock:
        for old_id, (stored_at, _, _) in list(_pending_rows.items()):
            if now - stored_at > _PENDING_TTL_SECONDS:
                _pending_rows.pop(old_id, None)
        _pending_rows[job_id] = (now, headers, rows)
        _pending_rows.move_to_end(job_id)
        while len(_pending_rows) > _PENDING_LIMIT:
            _pending_rows.popitem(last=False)


def take_pending_rows(job_id: UUID) -> tuple[list[str], list[dict[str, Any]]] | None:
    now = time.monotonic()
    with _pending_lock:
        value = _pending_rows.pop(job_id, None)
    if value is None or now - value[0] > _PENDING_TTL_SECONDS:
        return None
    return value[1], value[2]


def peek_pending_rows(job_id: UUID) -> tuple[list[str], list[dict[str, Any]]] | None:
    now = time.monotonic()
    with _pending_lock:
        value = _pending_rows.get(job_id)
        if value is None:
            return None
        if now - value[0] > _PENDING_TTL_SECONDS:
            _pending_rows.pop(job_id, None)
            return None
        return value[1], value[2]


def discard_pending_rows(job_id: UUID) -> None:
    with _pending_lock:
        _pending_rows.pop(job_id, None)


def persist_template(
    session: Session,
    user_id: UUID,
    *,
    name: str,
    kind: str,
    headers: list[str],
    columns: dict[str, str],
) -> None:
    fingerprint = header_fingerprint(headers)
    header_names = sorted({normalize_header(header) for header in headers})
    statement = pg_insert(ImportTemplate).values(
        user_id=user_id,
        name=name[:80],
        kind=kind,
        header_fingerprint=fingerprint,
        header_names=header_names,
        columns=columns,
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[
                ImportTemplate.user_id,
                ImportTemplate.kind,
                ImportTemplate.header_fingerprint,
            ],
            set_={
                "name": statement.excluded.name,
                "header_names": statement.excluded.header_names,
                "columns": statement.excluded.columns,
            },
        )
    )


def _security_match(
    session: Session,
    *,
    isin: str | None,
    symbol: str | None,
    exchange: str | None,
) -> Security | None:
    if isin:
        return session.scalar(select(Security).where(Security.isin == isin.upper()))
    if not symbol or (exchange and exchange.upper() not in {"NSE", "N"}):
        return None
    return session.scalar(
        select(Security).where(
            Security.market == "IN",
            Security.nse_symbol == symbol.upper(),
        )
    )


def _holding_payload(holding: BrokerHolding) -> dict[str, Any]:
    value = holding.quantity * holding.last_price if holding.last_price is not None else None
    return {
        "isin": holding.isin,
        "symbol": holding.symbol,
        "name": holding.name or holding.symbol or holding.isin or "Unknown security",
        "quantity": format(holding.quantity, "f"),
        "avg_price": format(holding.avg_price, "f"),
        "last_price": format(holding.last_price, "f") if holding.last_price is not None else None,
        "statement_value": format(round_paise(value), "f") if value is not None else None,
        "exchange": holding.exchange,
    }


def _open_payload(lot: Any) -> dict[str, Any]:
    return {
        "quantity": format(lot.quantity, "f"),
        "cost_per_unit": format(lot.cost_per_unit, "f"),
        "acquired_on": lot.acquired_on.isoformat(),
        "trade_ids": list(lot.trade_ids),
    }


def _realised_payload(lot: RealisedLot) -> dict[str, Any]:
    return {
        "quantity": format(lot.quantity, "f"),
        "cost_per_unit": format(lot.cost_per_unit, "f"),
        "sale_price": format(lot.sale_price, "f"),
        "acquired_on": lot.acquired_on.isoformat(),
        "trade_date": lot.sold_on.isoformat(),
        "holding_period": lot.holding_period,
        "gain": format(round_paise(lot.gain), "f"),
        "trade_id": lot.sell_trade_id,
    }


def stage_broker_items(
    session: Session,
    user_id: UUID,
    job: ImportJob,
    headers: list[str],
    rows: list[dict[str, Any]],
    layout: Layout,
) -> dict[str, int]:
    """Parse mapped rows into the shared review table and return status counts."""
    items: list[ImportItem] = []
    counts: Counter[str] = Counter()
    job.doc_type = f"broker_{layout.kind}"
    job.status = "ready"
    report = dict(job.parse_report or {})
    report.update({"kind": layout.kind, "broker": layout.broker})

    if layout.kind == "holdings":
        for holding in parse_holdings(rows, layout):
            security = _security_match(
                session,
                isin=holding.isin,
                symbol=holding.symbol,
                exchange=holding.exchange,
            )
            status = "new"
            diff = None
            match_lot_id = None
            if security is None and not can_create_security({"isin": holding.isin}):
                status = "unknown"
            elif security is not None:
                lots = list(
                    session.scalars(
                        select(HoldingLot)
                        .where(
                            HoldingLot.user_id == user_id,
                            HoldingLot.member_id == job.member_id,
                            HoldingLot.security_id == security.id,
                            HoldingLot.source == "cas",
                        )
                        .order_by(HoldingLot.id)
                    )
                )
                quantity = sum((lot.quantity for lot in lots), Decimal(0))
                if lots and quantity == holding.quantity:
                    status = "enrich"
                    match_lot_id = lots[0].id
                elif lots:
                    status = "changed"
                    match_lot_id = lots[0].id
                    diff = {
                        "quantity": {
                            "old": format(quantity, "f"),
                            "new": format(holding.quantity, "f"),
                        }
                    }
            payload = _holding_payload(holding)
            payload["merge_status"] = status
            if status == "unknown":
                payload["reason"] = "symbol_not_found"
            counts[status] += 1
            items.append(
                ImportItem(
                    user_id=user_id,
                    job_id=job.id,
                    item_kind="equity",
                    payload=payload,
                    status=status,
                    match_lot_id=match_lot_id,
                    match_security_id=security.id if security else None,
                    diff=diff,
                )
            )
        report["warnings"] = []
    else:
        trades = parse_tradebook(rows, layout)
        trades_by_key: dict[str, list[Trade]] = defaultdict(list)
        for trade in trades:
            trades_by_key[trade.security_key].append(trade)
        warnings_seen: set[str] = set()
        for security_trades in trades_by_key.values():
            open_lots, realised, warnings = fifo_lots(security_trades)
            warnings_seen.update(warnings)
            first = security_trades[0]
            security = _security_match(
                session,
                isin=first.isin,
                symbol=first.symbol,
                exchange=first.exchange,
            )
            open_quantity = sum((lot.quantity for lot in open_lots), Decimal(0))
            old_lots = (
                list(
                    session.scalars(
                        select(HoldingLot).where(
                            HoldingLot.user_id == user_id,
                            HoldingLot.member_id == job.member_id,
                            HoldingLot.security_id == security.id,
                        )
                    )
                )
                if security is not None
                else []
            )
            old_quantity = sum((lot.quantity for lot in old_lots), Decimal(0))
            status = "replace_with_lots" if old_quantity == open_quantity else "quantity_mismatch"
            if security is None and not can_create_security({"isin": first.isin}):
                status = "unknown"
            diff = (
                None
                if status == "replace_with_lots"
                else {"current": format(old_quantity, "f"), "tradebook": format(open_quantity, "f")}
            )
            payload = {
                "isin": first.isin,
                "symbol": first.symbol,
                "name": first.symbol or first.isin or "Unknown security",
                "quantity": format(open_quantity, "f"),
                "exchange": first.exchange,
                "open_lots": [_open_payload(lot) for lot in open_lots],
                "realised": [_realised_payload(lot) for lot in realised],
                "warnings": warnings,
            }
            if status == "unknown":
                payload["reason"] = "symbol_not_found"
            payload["merge_status"] = status
            counts[status] += 1
            items.append(
                ImportItem(
                    user_id=user_id,
                    job_id=job.id,
                    item_kind="equity",
                    payload=payload,
                    status=status,
                    match_lot_id=None,
                    match_security_id=security.id if security else None,
                    diff=diff,
                )
            )
        report["warnings"] = sorted(warnings_seen)

    session.add_all(items)
    job.parse_report = report
    return dict(counts)


def _security_for_confirmation(
    session: Session, user_id: UUID, item: ImportItem
) -> Security | None:
    if item.match_security_id is not None:
        return session.scalar(select(Security).where(Security.id == item.match_security_id))
    payload = item.payload
    isin = str(payload.get("isin") or "").upper()
    if not can_create_security(payload) or not isin.startswith("IN"):
        return None
    raw_security_id = session.scalar(
        text(
            "SELECT app_upsert_security('IN', :isin, :symbol, :name, 'equity', "
            "'INR', 'equity_india', 'Direct equity')"
        ),
        {
            "isin": isin,
            "symbol": str(payload.get("symbol") or isin)[:32],
            "name": str(payload.get("name") or payload.get("symbol") or "Imported equity")[:160],
        },
    )
    try:
        security_id = UUID(str(raw_security_id))
    except (TypeError, ValueError):
        return None
    return session.scalar(select(Security).where(Security.id == security_id))


def confirm_broker_items(
    session: Session,
    user_id: UUID,
    job: ImportJob,
    items: list[ImportItem],
) -> dict[str, int]:
    """Apply a broker review while preserving manual lots and member ownership."""
    counts: Counter[str] = Counter()
    account = session.get(Account, job.account_id) if job.account_id is not None else None
    for item in items:
        if item.status in {"rejected", "unknown", "duplicate"}:
            continue
        payload = item.payload
        merge_status = str(payload.get("merge_status") or item.status)
        if job.doc_type == "broker_holdings":
            if item.status not in {"new", "changed", "enrich", "accepted"}:
                continue
            security = _security_for_confirmation(session, user_id, item)
            if security is None:
                continue
            if merge_status == "enrich":
                query = select(HoldingLot).where(
                    HoldingLot.user_id == user_id,
                    HoldingLot.member_id == job.member_id,
                    HoldingLot.security_id == security.id,
                    HoldingLot.source == "cas",
                )
                lots = list(session.scalars(query))
                for lot in lots:
                    lot.cost_per_unit = Decimal(str(payload["avg_price"]))
                    if job.account_id is not None and lot.account_id is None:
                        lot.account_id = job.account_id
                counts["enrich"] += 1
                continue

            sources = ("broker", "cas") if merge_status == "changed" else ("broker",)
            _delete_replaceable_lots(session, user_id, job, security.id, sources)
            session.add(
                HoldingLot(
                    user_id=user_id,
                    account_id=job.account_id,
                    member_id=job.member_id,
                    joint=account.joint if account is not None else False,
                    security_id=security.id,
                    quantity=Decimal(str(payload["quantity"])),
                    cost_per_unit=Decimal(str(payload["avg_price"])),
                    acquired_on=None,
                    source="broker",
                    import_job_id=job.id,
                    trade_ids=[],
                )
            )
            counts[merge_status if merge_status in {"new", "changed"} else "accepted"] += 1
            item.match_security_id = security.id
            continue

        if job.doc_type != "broker_tradebook":
            continue
        if merge_status == "quantity_mismatch" and item.status != "accepted":
            continue
        if item.status not in {"replace_with_lots", "quantity_mismatch", "accepted"}:
            continue
        security = _security_for_confirmation(session, user_id, item)
        if security is None:
            continue
        _delete_replaceable_lots(session, user_id, job, security.id, ("cas", "broker"))
        source_broker = str((job.parse_report or {}).get("broker", "mapped"))
        session.execute(
            delete(RealisedTrade).where(
                RealisedTrade.user_id == user_id,
                RealisedTrade.member_id == job.member_id,
                RealisedTrade.account_id.is_not_distinct_from(job.account_id),
                RealisedTrade.security_id == security.id,
                RealisedTrade.source_broker == source_broker,
            )
        )
        for lot in payload.get("open_lots", []):
            session.add(
                HoldingLot(
                    user_id=user_id,
                    account_id=job.account_id,
                    member_id=job.member_id,
                    joint=account.joint if account is not None else False,
                    security_id=security.id,
                    quantity=Decimal(str(lot["quantity"])),
                    cost_per_unit=Decimal(str(lot["cost_per_unit"])),
                    acquired_on=date.fromisoformat(str(lot["acquired_on"])),
                    source="broker",
                    import_job_id=job.id,
                    trade_ids=list(lot.get("trade_ids", [])),
                )
            )
        for realised in payload.get("realised", []):
            session.add(
                RealisedTrade(
                    user_id=user_id,
                    member_id=job.member_id,
                    account_id=job.account_id,
                    import_job_id=job.id,
                    security_id=security.id,
                    trade_date=date.fromisoformat(str(realised["trade_date"])),
                    acquired_on=date.fromisoformat(str(realised["acquired_on"])),
                    quantity=Decimal(str(realised["quantity"])),
                    cost_per_unit=Decimal(str(realised["cost_per_unit"])),
                    sale_price=Decimal(str(realised["sale_price"])),
                    gain=Decimal(str(realised["gain"])),
                    holding_period=str(realised["holding_period"]),
                    trade_id=realised.get("trade_id"),
                    source_broker=source_broker,
                )
            )
        counts["replace_with_lots"] += 1
        item.match_security_id = security.id
    return dict(counts)


def _delete_replaceable_lots(
    session: Session,
    user_id: UUID,
    job: ImportJob,
    security_id: UUID,
    sources: tuple[str, ...],
) -> None:
    lots = list(
        session.scalars(
            select(HoldingLot.id).where(
                HoldingLot.user_id == user_id,
                HoldingLot.member_id == job.member_id,
                HoldingLot.security_id == security_id,
                HoldingLot.source.in_(sources),
            )
        )
    )
    if lots:
        session.execute(
            update(ImportItem)
            .where(
                ImportItem.user_id == user_id,
                ImportItem.match_lot_id.in_(lots),
            )
            .values(match_lot_id=None)
        )
        session.execute(
            delete(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.id.in_(lots),
            )
        )


def template_layout(kind: str, columns: Mapping[str, str], *, broker: str = "mapped") -> Layout:
    return Layout(
        broker=broker,
        kind=kind,  # type: ignore[arg-type]
        fingerprint=frozenset(),
        columns=dict(columns),
        source_note="Saved by this household.",
    )


def valid_template_mapping(kind: str, columns: Mapping[str, str], headers: list[str]) -> bool:
    normalized_headers = {normalize_header(header) for header in headers}
    if kind == "holdings":
        required_groups = ({"isin", "symbol"}, {"quantity"}, {"avg_price"})
    elif kind == "tradebook":
        required_groups = (
            {"isin", "symbol"},
            {"trade_date"},
            {"trade_type"},
            {"quantity"},
            {"price"},
        )
    else:
        return False
    if any(not any(field in columns for field in group) for group in required_groups):
        return False
    return all(normalize_header(header) in normalized_headers for header in columns.values())


def mapping_suggestions(headers: list[str]) -> dict[str, str | None]:
    return suggested_mapping(headers)


def header_names(headers: list[str]) -> list[str]:
    return [normalize_header(header) for header in headers]


def can_create_security(payload: Mapping[str, Any]) -> bool:
    isin = payload.get("isin")
    return (
        isinstance(isin, str)
        and isin.upper().startswith("IN")
        and _ISIN.fullmatch(isin.upper()) is not None
        and isin_check_digit_ok(isin)
    )
