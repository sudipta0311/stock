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
    BrokerCashFlowRecord,
    FxRate,
    HoldingLot,
    ImportItem,
    ImportJob,
    ImportTemplate,
    RealisedTrade,
    Security,
)
from wealth.importer.brokers import (
    BrokerCashFlow,
    BrokerHolding,
    IBKRActivity,
    Layout,
    RealisedLot,
    Trade,
    detect_layout,
    fifo_lots,
    header_fingerprint,
    load_layouts,
    normalize_header,
    parse_cash_flows,
    parse_holdings,
    parse_tradebook,
    read_table,
    suggested_mapping,
)
from wealth.importer.classify import classify_broker_security

_PENDING_TTL_SECONDS = 20 * 60
_PENDING_LIMIT = 8
_pending_lock = threading.Lock()
_pending_rows: OrderedDict[UUID, tuple[float, list[str], list[dict[str, Any]]]] = OrderedDict()
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_MARKETS = {"IN", "US", "DK", "EU", "GB"}
_CURRENCIES = {"INR", "USD", "EUR", "DKK", "GBP", "SEK"}


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
            "columns": {
                key: value for key, value in row.columns.items() if not key.startswith("__")
            },
            "market_default": row.columns.get("__market_default", "IN"),
            "currency_default": row.columns.get("__currency_default", "INR"),
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
    market_default: str = "IN",
    currency_default: str = "INR",
) -> None:
    fingerprint = header_fingerprint(headers)
    header_names = sorted({normalize_header(header) for header in headers})
    statement = pg_insert(ImportTemplate).values(
        user_id=user_id,
        name=name[:80],
        kind=kind,
        header_fingerprint=fingerprint,
        header_names=header_names,
        columns={
            **columns,
            "__market_default": market_default,
            "__currency_default": currency_default,
        },
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
    market: str = "IN",
) -> Security | None:
    if isin:
        found = session.scalar(select(Security).where(Security.isin == isin.upper()))
        if found is not None:
            return found
    if not symbol or market not in _MARKETS:
        return None
    return session.scalar(
        select(Security).where(
            Security.market == market,
            Security.symbol == symbol.upper(),
        )
    )


def _holding_payload(holding: BrokerHolding) -> dict[str, Any]:
    value = holding.market_value
    if value is None and holding.last_price is not None:
        value = holding.quantity * holding.last_price
    sleeve, category = classify_broker_security(
        holding.market, holding.symbol or "", holding.name or "", holding.security_type
    )
    return {
        "isin": holding.isin,
        "symbol": holding.symbol,
        "name": holding.name or holding.symbol or holding.isin or "Unknown security",
        "quantity": format(holding.quantity, "f"),
        "avg_price": format(holding.avg_price, "f"),
        "last_price": format(holding.last_price, "f") if holding.last_price is not None else None,
        "statement_value": format(round_paise(value), "f") if value is not None else None,
        "exchange": holding.exchange,
        "market": holding.market,
        "currency": holding.currency,
        "security_type": holding.security_type,
        "sleeve": sleeve,
        "category": category,
    }


def _open_payload(lot: Any) -> dict[str, Any]:
    return {
        "quantity": format(lot.quantity, "f"),
        "cost_per_unit": format(lot.cost_per_unit, "f"),
        "acquired_on": lot.acquired_on.isoformat(),
        "trade_ids": list(lot.trade_ids),
        "market": lot.market,
        "currency": lot.currency,
        "security_type": lot.security_type,
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
        "market": lot.market,
        "currency": lot.currency,
        "commission": format(lot.commission, "f"),
        "security_type": lot.security_type,
    }


def _fx_rate_for_date(session: Session, currency: str, on_date: date) -> Decimal | None:
    """Use the nearest earlier reference rate; it approximates the SBI TT buying rate."""
    if currency == "INR":
        return Decimal(1)
    row = session.execute(
        select(FxRate.rate, FxRate.as_of)
        .where(FxRate.pair == f"{currency}/INR", FxRate.as_of <= on_date)
        .order_by(FxRate.as_of.desc())
        .limit(1)
    ).one_or_none()
    if row is None or (on_date - row.as_of).days > 7:
        return None
    return row.rate


def _cash_flow_payload(flow: BrokerCashFlow) -> dict[str, str | None]:
    return {
        "transaction_date": flow.transaction_date.isoformat(),
        "kind": flow.kind,
        "amount": format(flow.amount, "f"),
        "currency": flow.currency,
        "transaction_id": flow.transaction_id,
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
                market=holding.market,
            )
            status = "new"
            diff = None
            match_lot_id = None
            if security is None and not can_create_security(
                {
                    "isin": holding.isin,
                    "symbol": holding.symbol,
                    "market": holding.market,
                    "currency": holding.currency,
                }
            ):
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
            payload["broker_kind"] = "holdings"
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
        cash_flows = parse_cash_flows(rows, layout)
        report["cash_flows"] = [_cash_flow_payload(flow) for flow in cash_flows]
        report["cash_flow_count"] = len(cash_flows)
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
                market=first.market,
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
            if security is None and not can_create_security(
                {
                    "isin": first.isin,
                    "symbol": first.symbol,
                    "market": first.market,
                    "currency": first.currency,
                }
            ):
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
                "market": first.market,
                "currency": first.currency,
                "security_type": first.security_type,
                "sleeve": classify_broker_security(
                    first.market, first.symbol or "", first.symbol or "", first.security_type
                )[0],
                "category": classify_broker_security(
                    first.market, first.symbol or "", first.symbol or "", first.security_type
                )[1],
                "broker_kind": "tradebook",
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
    imported_wrapper = mapped_tax_wrapper(rows, layout)
    if imported_wrapper is not None:
        report["tax_wrapper"] = imported_wrapper
        report["tax_wrapper_question"] = False
    job.parse_report = report
    return dict(counts)


def stage_ibkr_activity(
    session: Session,
    user_id: UUID,
    job: ImportJob,
    activity: IBKRActivity,
) -> dict[str, int]:
    """Stage one IBKR activity file, preferring trade-derived lots when present."""
    trade_headers = [
        "ISIN",
        "Symbol",
        "Trade date",
        "Trade type",
        "Quantity",
        "Price",
        "Exchange",
        "Trade ID",
        "Market",
        "Currency",
        "Commission",
        "Amount",
        "Security type",
    ]
    trade_rows = [
        {
            "ISIN": trade.isin,
            "Symbol": trade.symbol,
            "Trade date": trade.trade_date.isoformat(),
            "Trade type": trade.trade_type,
            "Quantity": format(trade.quantity, "f"),
            "Price": format(trade.price, "f"),
            "Exchange": trade.exchange,
            "Trade ID": trade.trade_id,
            "Market": trade.market,
            "Currency": trade.currency,
            "Commission": format(trade.commission, "f"),
            "Amount": format(trade.proceeds, "f") if trade.proceeds is not None else None,
            "Security type": "ETFs" if trade.security_type == "etf" else "Stocks",
        }
        for trade in activity.trades
    ]
    counts: Counter[str] = Counter()
    traded_symbols: set[tuple[str, str]] = set()
    if trade_rows:
        trade_layout = Layout(
            broker="ibkr",
            kind="tradebook",
            fingerprint=frozenset(),
            columns={
                "isin": "ISIN",
                "symbol": "Symbol",
                "trade_date": "Trade date",
                "trade_type": "Trade type",
                "quantity": "Quantity",
                "price": "Price",
                "exchange": "Exchange",
                "trade_id": "Trade ID",
                "market": "Market",
                "currency": "Currency",
                "commission": "Commission",
                "amount": "Amount",
                "security_type": "Security type",
            },
            market_default="US",
            currency_default="USD",
        )
        counts.update(
            stage_broker_items(session, user_id, job, trade_headers, trade_rows, trade_layout)
        )
        staged = list(
            session.scalars(
                select(ImportItem).where(
                    ImportItem.user_id == user_id,
                    ImportItem.job_id == job.id,
                )
            )
        )
        for item in staged:
            payload = dict(item.payload)
            symbol_key = (str(payload.get("market") or "US"), str(payload.get("symbol") or ""))
            traded_symbols.add(symbol_key)
            position = next(
                (
                    value
                    for value in activity.positions
                    if (value.market, value.symbol or "") == symbol_key
                    or (value.isin and value.isin == payload.get("isin"))
                ),
                None,
            )
            if position is not None:
                payload["position_quantity"] = format(position.quantity, "f")
                payload["last_price"] = (
                    format(position.last_price, "f") if position.last_price is not None else None
                )
                statement_value = position.market_value
                if statement_value is None and position.last_price is not None:
                    statement_value = position.quantity * position.last_price
                payload["statement_value"] = (
                    format(round_paise(statement_value), "f")
                    if statement_value is not None
                    else None
                )
                item.payload = payload

    unmatched_positions = [
        holding
        for holding in activity.positions
        if (holding.market, holding.symbol or "") not in traded_symbols
    ]
    if unmatched_positions:
        position_headers = [
            "ISIN",
            "Symbol",
            "Name",
            "Quantity",
            "Average price",
            "Last price",
            "Exchange",
            "Market",
            "Currency",
            "Market value",
            "Security type",
        ]
        position_rows = [
            {
                "ISIN": holding.isin,
                "Symbol": holding.symbol,
                "Name": holding.name,
                "Quantity": format(holding.quantity, "f"),
                "Average price": format(holding.avg_price, "f"),
                "Last price": format(holding.last_price, "f")
                if holding.last_price is not None
                else None,
                "Exchange": holding.exchange,
                "Market": holding.market,
                "Currency": holding.currency,
                "Market value": format(holding.market_value, "f")
                if holding.market_value is not None
                else None,
                "Security type": "ETFs" if holding.security_type == "etf" else "Stocks",
            }
            for holding in unmatched_positions
        ]
        position_layout = Layout(
            broker="ibkr",
            kind="holdings",
            fingerprint=frozenset(),
            columns={
                "isin": "ISIN",
                "symbol": "Symbol",
                "name": "Name",
                "quantity": "Quantity",
                "avg_price": "Average price",
                "last_price": "Last price",
                "exchange": "Exchange",
                "market": "Market",
                "currency": "Currency",
                "market_value": "Market value",
                "security_type": "Security type",
            },
            market_default="US",
            currency_default="USD",
        )
        counts.update(
            stage_broker_items(
                session, user_id, job, position_headers, position_rows, position_layout
            )
        )

    job.doc_type = "broker_ibkr"
    job.status = "ready"
    job.counts = dict(counts)
    job.parse_report = {
        **(job.parse_report or {}),
        "broker": "ibkr",
        "warnings": sorted(
            set((job.parse_report or {}).get("warnings", [])) | set(activity.warnings)
        ),
    }
    return dict(counts)


def _security_for_confirmation(
    session: Session, user_id: UUID, item: ImportItem
) -> Security | None:
    if item.match_security_id is not None:
        return session.scalar(select(Security).where(Security.id == item.match_security_id))
    payload = item.payload
    isin = str(payload.get("isin") or "").upper()
    if not can_create_security(payload):
        return None
    market = str(payload.get("market") or ("IN" if isin.startswith("IN") else "US")).upper()
    currency = str(payload.get("currency") or ("INR" if market == "IN" else "USD")).upper()
    symbol = str(payload.get("symbol") or isin).upper()
    name = str(payload.get("name") or symbol or "Imported security")[:160]
    override = item.user_override or {}
    security_type = str(override.get("kind") or payload.get("security_type") or "equity")
    sleeve = str(
        override.get("sleeve")
        or payload.get("sleeve")
        or classify_broker_security(market, symbol, name, security_type)[0]
    )
    category = str(
        payload.get("category") or classify_broker_security(market, symbol, name, security_type)[1]
    )
    if market == "IN" and isin.startswith("IN"):
        raw_security_id = session.scalar(
            text(
                "SELECT app_upsert_security('IN', :isin, :symbol, :name, :kind, "
                ":currency, :sleeve, :category)"
            ),
            {
                "isin": isin,
                "symbol": symbol[:32],
                "name": name,
                "kind": security_type,
                "currency": currency,
                "sleeve": sleeve,
                "category": category,
            },
        )
    else:
        raw_security_id = session.scalar(
            text(
                "SELECT app_upsert_security_by_symbol(:market, :symbol, :exchange, :name, "
                ":kind, :currency, :isin, :sleeve, :category)"
            ),
            {
                "market": market,
                "symbol": symbol[:15],
                "exchange": str(payload.get("exchange") or "")[:32] or None,
                "name": name,
                "kind": security_type,
                "currency": currency,
                "isin": isin or None,
                "sleeve": sleeve,
                "category": category,
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
    source_broker = str((job.parse_report or {}).get("broker", "mapped"))
    for item in items:
        if item.status in {"rejected", "unknown", "duplicate"}:
            continue
        payload = item.payload
        merge_status = str(payload.get("merge_status") or item.status)
        broker_kind = str(
            payload.get("broker_kind")
            or ("holdings" if job.doc_type == "broker_holdings" else "tradebook")
        )
        if broker_kind == "holdings":
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
                    lot.currency = str(payload.get("currency") or security.currency)
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
                    statement_value=(
                        Decimal(str(payload["statement_value"]))
                        if payload.get("statement_value") is not None
                        else None
                    ),
                    statement_date=job.statement_date,
                    import_job_id=job.id,
                    trade_ids=[],
                    currency=str(payload.get("currency") or security.currency),
                    fx_missing=str(payload.get("currency") or security.currency) != "INR",
                )
            )
            counts[merge_status if merge_status in {"new", "changed"} else "accepted"] += 1
            item.match_security_id = security.id
            continue

        if broker_kind != "tradebook":
            continue
        if merge_status == "quantity_mismatch" and item.status != "accepted":
            continue
        if item.status not in {"replace_with_lots", "quantity_mismatch", "accepted"}:
            continue
        security = _security_for_confirmation(session, user_id, item)
        if security is None:
            continue
        _delete_replaceable_lots(session, user_id, job, security.id, ("cas", "broker"))
        session.execute(
            delete(RealisedTrade).where(
                RealisedTrade.user_id == user_id,
                RealisedTrade.member_id == job.member_id,
                RealisedTrade.account_id.is_not_distinct_from(job.account_id),
                RealisedTrade.security_id == security.id,
                RealisedTrade.source_broker == source_broker,
            )
        )
        for lot_index, lot in enumerate(payload.get("open_lots", [])):
            currency = str(lot.get("currency") or payload.get("currency") or security.currency)
            acquired_on = date.fromisoformat(str(lot["acquired_on"]))
            cost_per_unit = Decimal(str(lot["cost_per_unit"]))
            fx_at_acquisition = _fx_rate_for_date(session, currency, acquired_on)
            statement_value = payload.get("statement_value") if lot_index == 0 else None
            session.add(
                HoldingLot(
                    user_id=user_id,
                    account_id=job.account_id,
                    member_id=job.member_id,
                    joint=account.joint if account is not None else False,
                    security_id=security.id,
                    quantity=Decimal(str(lot["quantity"])),
                    cost_per_unit=cost_per_unit,
                    acquired_on=acquired_on,
                    source="broker",
                    statement_value=(
                        Decimal(str(statement_value)) if statement_value is not None else None
                    ),
                    statement_date=job.statement_date if statement_value is not None else None,
                    import_job_id=job.id,
                    trade_ids=list(lot.get("trade_ids", [])),
                    currency=currency,
                    fx_at_acquisition=fx_at_acquisition,
                    cost_per_unit_inr=(
                        cost_per_unit * fx_at_acquisition if fx_at_acquisition is not None else None
                    ),
                    fx_missing=fx_at_acquisition is None,
                )
            )
        for realised in payload.get("realised", []):
            currency = str(realised.get("currency") or payload.get("currency") or security.currency)
            acquired_on = date.fromisoformat(str(realised["acquired_on"]))
            trade_date = date.fromisoformat(str(realised["trade_date"]))
            cost_per_unit = Decimal(str(realised["cost_per_unit"]))
            sale_price = Decimal(str(realised["sale_price"]))
            quantity = Decimal(str(realised["quantity"]))
            commission = Decimal(str(realised.get("commission", "0")))
            fx_at_acquisition = _fx_rate_for_date(session, currency, acquired_on)
            fx_at_sale = _fx_rate_for_date(session, currency, trade_date)
            cost_per_unit_inr = (
                cost_per_unit * fx_at_acquisition if fx_at_acquisition is not None else None
            )
            sale_price_inr = sale_price * fx_at_sale if fx_at_sale is not None else None
            gain_inr = (
                round_paise(
                    (sale_price * fx_at_sale - cost_per_unit * fx_at_acquisition) * quantity
                    - commission * fx_at_sale
                )
                if fx_at_acquisition is not None and fx_at_sale is not None
                else None
            )
            session.add(
                RealisedTrade(
                    user_id=user_id,
                    member_id=job.member_id,
                    account_id=job.account_id,
                    import_job_id=job.id,
                    security_id=security.id,
                    trade_date=trade_date,
                    acquired_on=acquired_on,
                    quantity=quantity,
                    cost_per_unit=cost_per_unit,
                    sale_price=sale_price,
                    gain=Decimal(str(realised["gain"])),
                    holding_period=str(realised["holding_period"]),
                    trade_id=realised.get("trade_id"),
                    source_broker=source_broker,
                    currency=currency,
                    fx_at_acquisition=fx_at_acquisition,
                    fx_at_sale=fx_at_sale,
                    cost_per_unit_inr=cost_per_unit_inr,
                    sale_price_inr=sale_price_inr,
                    gain_inr=gain_inr,
                    fx_missing=fx_at_acquisition is None or fx_at_sale is None,
                )
            )
        counts["replace_with_lots"] += 1
        item.match_security_id = security.id

    cash_flows = (job.parse_report or {}).get("cash_flows", [])
    if job.member_id is not None:
        saved_flow_count = 0
        for flow in cash_flows:
            transaction_id = flow.get("transaction_id")
            if (
                transaction_id is not None
                and session.scalar(
                    select(BrokerCashFlowRecord.id).where(
                        BrokerCashFlowRecord.user_id == user_id,
                        BrokerCashFlowRecord.member_id == job.member_id,
                        BrokerCashFlowRecord.account_id.is_not_distinct_from(job.account_id),
                        BrokerCashFlowRecord.transaction_id == transaction_id,
                    )
                )
                is not None
            ):
                continue
            session.add(
                BrokerCashFlowRecord(
                    user_id=user_id,
                    member_id=job.member_id,
                    account_id=job.account_id,
                    import_job_id=job.id,
                    transaction_date=date.fromisoformat(str(flow["transaction_date"])),
                    kind=str(flow["kind"]),
                    amount=Decimal(str(flow["amount"])),
                    currency=str(flow["currency"]),
                    source_broker=source_broker,
                    transaction_id=transaction_id,
                )
            )
            saved_flow_count += 1
        if saved_flow_count:
            counts["cash_flows"] = saved_flow_count
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


def template_layout(
    kind: str,
    columns: Mapping[str, str],
    *,
    broker: str = "mapped",
    market_default: str = "US",
    currency_default: str = "USD",
) -> Layout:
    return Layout(
        broker=broker,
        kind=kind,  # type: ignore[arg-type]
        fingerprint=frozenset(),
        columns=dict(columns),
        source_note="Saved by this household.",
        market_default=market_default,
        currency_default=currency_default,
    )


def valid_template_mapping(kind: str, columns: Mapping[str, str], headers: list[str]) -> bool:
    normalized_headers = {normalize_header(header) for header in headers}
    if kind == "holdings":
        required_groups = ({"isin", "symbol"}, {"quantity"}, {"avg_price"})
    elif kind == "tradebook":
        has_securities = any(field in columns for field in {"isin", "symbol"}) and all(
            field in columns for field in ("trade_date", "trade_type", "quantity", "price")
        )
        has_cash_flows = all(field in columns for field in ("trade_date", "trade_type", "amount"))
        if not (has_securities or has_cash_flows):
            return False
        return all(normalize_header(header) in normalized_headers for header in columns.values())
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
    if isinstance(isin, str) and isin:
        normalized_isin = isin.upper()
        if _ISIN.fullmatch(normalized_isin) is None or not isin_check_digit_ok(normalized_isin):
            return False
        if normalized_isin.startswith("IN"):
            return str(payload.get("market") or "IN").upper() == "IN"
    market = str(payload.get("market") or "").upper()
    currency = str(payload.get("currency") or "").upper()
    symbol = str(payload.get("symbol") or "").upper()
    if market == "IN":
        return False
    return (
        market in {"US", "DK", "EU", "GB"}
        and currency in {"USD", "EUR", "DKK", "GBP", "SEK"}
        and re.fullmatch(r"[A-Z0-9.\-]{1,15}", symbol) is not None
    )


def infer_tax_wrapper(headers: list[str], rows: list[dict[str, Any]]) -> str | None:
    """Read only an explicitly labelled account-wrapper column; never infer from balances."""
    accepted = {"ask", "share savings", "aktiesparekonto", "ordinary", "regular", "pension"}
    for header in headers:
        if normalize_header(header) not in {
            "tax wrapper",
            "account type",
            "account product",
            "wrapper",
        }:
            continue
        for row in rows:
            value = str(row.get(header) or "").strip().casefold()
            if not value or not any(label in value for label in accepted):
                continue
            if "aktiesparekonto" in value or value == "ask" or "share savings" in value:
                return "ask"
            if "pension" in value:
                return "pension"
            if "ordinary" in value or "regular" in value:
                return "ordinary"
    return None


def mapped_tax_wrapper(rows: list[dict[str, Any]], layout: Layout) -> str | None:
    """Read account type only from a column the user mapped for this import."""
    mapped = layout.columns.get("tax_wrapper")
    if mapped is None:
        return None
    headers = [mapped] if isinstance(mapped, str) else list(mapped)
    return infer_tax_wrapper(headers, rows)


def is_danish_broker(broker: str, account: Account | None) -> bool:
    identity = f"{broker} {account.institution if account else ''}".casefold()
    return any(name in identity for name in ("nordnet", "saxo", "danske"))


def update_tax_wrapper_report(
    job: ImportJob,
    broker: str,
    account: Account | None,
    imported_wrapper: str | None = None,
) -> None:
    report = dict(job.parse_report or {})
    wrapper = imported_wrapper or (account.tax_wrapper if account is not None else None)
    report["tax_wrapper"] = wrapper
    report["tax_wrapper_question"] = (
        wrapper is None and account is not None and is_danish_broker(broker, account)
    )
    job.parse_report = report


def broker_defaults(broker: str) -> tuple[str, str]:
    if broker in {"nordnet", "saxo", "danske_bank"}:
        return "DK", "DKK"
    return "US", "USD"


def broker_for_account(broker: str, account: Account | None) -> str:
    if broker != "other" or account is None:
        return broker
    institution = account.institution.casefold()
    if "nordnet" in institution:
        return "nordnet"
    if "saxo" in institution:
        return "saxo"
    if "danske" in institution:
        return "danske_bank"
    return broker
