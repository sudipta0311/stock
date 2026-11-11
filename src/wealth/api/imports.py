"""Authenticated, in-memory CAS upload and review endpoints."""

from __future__ import annotations

import asyncio
import re
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import case, delete, func, select, text, update
from sqlalchemy.orm import Session
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from wealth.importer.cas import ParseError, parse_cas
from wealth.importer.classify import classify_holding
from wealth.importer.models import ParsedHolding, ParsedStatement

MAX_UPLOAD_MB = 10
PARSE_TIMEOUT_SECONDS = 20
_MAX_MULTIPART_FIELDS = 6
_MAX_FIELD_BYTES = 4096
_PARSE_SEMAPHORE = threading.BoundedSemaphore(1)
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_KINDS = ("equity", "etf", "mf", "bond", "sgb")


class ImportApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImportUploadResponse(ImportApiModel):
    job_id: UUID
    status: Literal["ready", "needs_mapping"]
    source: Literal["nsdl", "cdsl", "cams_kfin", "broker"]
    statement_date: date | None
    counts: dict[str, int]
    warnings: list[str]


class ImportOverride(ImportApiModel):
    kind: Literal["equity", "etf", "mf", "bond", "sgb"]
    sleeve: Literal[
        "equity_india",
        "equity_global",
        "debt",
        "gold",
        "cash",
        "real_estate",
        "protection",
    ]


class ImportItemPatch(ImportApiModel):
    status: Literal["accepted", "rejected"] | None = None
    override: ImportOverride | None = None
    security_id: UUID | None = None


class ImportItemResponse(ImportApiModel):
    id: UUID
    kind: str
    isin: str
    symbol: str | None = None
    name: str
    quantity: str
    statement_value: str | None
    account_ref: str | None
    market: str | None = None
    currency: str | None = None
    exchange: str | None = None
    amount: str | None = None
    security_type: str | None = None
    sleeve: str | None
    status: str
    diff: dict[str, Any] | None
    match: dict[str, UUID | None]
    reason: str | None = None
    action: str | None = None
    matched_by: str | None = None
    can_find_security: bool


class SecuritySearchResult(ImportApiModel):
    id: UUID
    symbol: str
    name: str
    security_type: Literal["equity", "etf"]


class ImportJobResponse(ImportApiModel):
    id: UUID
    doc_type: str
    status: Literal["ready", "needs_mapping", "confirmed", "failed", "discarded"]
    error_code: str | None
    source: Literal["nsdl", "cdsl", "cams_kfin", "broker"] | None
    statement_date: date | None
    account_id: UUID | None = None
    member_id: UUID | None = None
    counts: dict[str, int]
    created_at: datetime
    confirmed_at: datetime | None


class ImportDetailResponse(ImportJobResponse):
    items: list[ImportItemResponse]
    headers: list[str] | None = None
    samples: list[list[str]] = Field(default_factory=list)
    suggested_mapping: dict[str, str | None] = Field(default_factory=dict)
    mapping_kind: Literal["holdings", "tradebook"] | None = None
    auto_mapped: bool = False
    columns_used: dict[str, str] = Field(default_factory=dict)
    has_missing_cost: bool = False
    skipped_rows: dict[str, int] = Field(default_factory=dict)
    needs_market_default: bool = False
    needs_currency_default: bool = False
    broker: str | None = None
    detected_broker: str | None = None
    market_default: Literal["IN", "US", "DK", "EU", "GB"] = "US"
    currency_default: Literal["INR", "USD", "EUR", "DKK", "GBP", "SEK"] = "USD"
    tax_wrapper: Literal["ask", "ordinary", "pension"] | None = None
    tax_wrapper_question: bool = False
    cash_flow_count: int = 0
    currency_mismatch_values: int = 0


class ImportMappingRequest(ImportApiModel):
    kind: Literal["holdings", "tradebook"]
    columns: dict[str, str]
    save_as: str | None = None
    broker: Literal[
        "zerodha",
        "hdfc_securities",
        "groww",
        "icici_direct",
        "angel_one",
        "upstox",
        "other_indian_broker",
        "indmoney",
        "vested",
        "nordnet",
        "saxo",
        "danske_bank",
        "other_danish_broker",
        "ibkr",
        "other",
    ] = "other"
    market_default: Literal["IN", "US", "DK", "EU", "GB"] = "US"
    currency_default: Literal["INR", "USD", "EUR", "DKK", "GBP", "SEK"] = "USD"


class ImportConfirmationRequest(ImportApiModel):
    tax_wrapper: Literal["ask", "ordinary", "pension"] | None = None


class ImportConfirmationResponse(ImportApiModel):
    job_id: UUID
    status: Literal["confirmed"]
    counts: dict[str, int]
    net_worth: str


class _UploadTooLarge(Exception):
    pass


class _InMemoryMultipartParser(MultiPartParser):
    """Keep bounded uploads in RAM and reject a file as soon as its limit is crossed."""

    def __init__(self, headers: Any, stream: Any, max_file_bytes: int) -> None:
        super().__init__(
            headers,
            stream,
            max_files=1,
            max_fields=_MAX_MULTIPART_FIELDS,
            max_part_size=_MAX_FIELD_BYTES,
        )
        self.max_file_bytes = max_file_bytes
        self.spool_max_size = max_file_bytes + 1
        self._file_bytes_seen = 0
        self._file_prefix = bytearray()

    async def parse(self) -> FormData:
        try:
            return await super().parse()
        except _UploadTooLarge:
            for file_object in self._files_to_close_on_error:
                file_object.close()
            raise

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._file_bytes_seen = 0
        self._file_prefix.clear()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            size = end - start
            if self._file_bytes_seen + size > self.max_file_bytes:
                raise _UploadTooLarge
            if len(self._file_prefix) < 4:
                self._file_prefix.extend(data[start : min(end, start + 4 - len(self._file_prefix))])
            self._file_bytes_seen += size
        super().on_part_data(data, start, end)


def _problem(request: Request, status: int, code: str, detail: str) -> JSONResponse:
    title = {
        404: "Not Found",
        409: "Conflict",
        413: "Content Too Large",
        415: "Unsupported Media Type",
        422: "Unprocessable Content",
        504: "Gateway Timeout",
    }.get(status, "Request Failed")
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": request.url.path,
            "code": code,
        },
    )


def _parse_serialized(data: bytes, password: str | None, max_mb: int) -> ParsedStatement:
    with _PARSE_SEMAPHORE:
        return parse_cas(data, password, max_mb=max_mb)


def _parse_broker_serialized(data: bytes, filename: str) -> tuple[list[str], list[dict[str, Any]]]:
    from wealth.api.broker_imports import parse_broker_upload

    with _PARSE_SEMAPHORE:
        return parse_broker_upload(data, filename)


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _looks_like_text_csv(data: bytes, filename: str = "statement.csv") -> bool:
    from wealth.importer.brokers.parser import detect_broker_file

    return detect_broker_file(data, filename) == "text"


def _money_value(holding: ParsedHolding) -> Decimal | None:
    if holding.market_value is not None:
        value = holding.market_value
    elif holding.price is not None:
        value = holding.quantity * holding.price
    else:
        return None
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _holding_payload(
    holding: ParsedHolding,
    transactions: list[dict[str, Any]],
    *,
    amfi_category: str | None = None,
) -> dict[str, Any]:
    payload = holding.model_dump(mode="json")
    classification = classify_holding(holding, amfi_category)
    payload["sleeve"] = classification[0] if classification else None
    payload["category"] = classification[1] if classification else None
    statement_value = _money_value(holding)
    payload["statement_value"] = (
        _decimal_string(statement_value) if statement_value is not None else None
    )
    payload["transactions"] = transactions
    return payload


def _can_find_security(job: Any | None, item: Any) -> bool:
    payload = item.payload
    return (
        job is not None
        and job.source == "broker"
        and item.status == "unknown"
        and payload.get("reason") == "symbol_not_found"
        and payload.get("broker_kind") == "holdings"
        and str(payload.get("market") or "").upper() == "IN"
    )


def _get_item_response(item: Any, job: Any | None = None) -> dict[str, Any]:
    payload = item.payload
    override = item.user_override or {}
    statement_value = payload.get("statement_value")
    if statement_value is None and payload.get("last_price") is not None:
        try:
            value = Decimal(str(payload["quantity"])) * Decimal(str(payload["last_price"]))
            statement_value = _decimal_string(
                value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )
        except (InvalidOperation, KeyError, TypeError, ValueError):
            statement_value = None
    return {
        "id": item.id,
        "kind": (
            override.get("kind")
            or payload.get("security_type")
            or payload.get("kind")
            or item.item_kind
        ),
        "isin": payload.get("isin") or "",
        "symbol": payload.get("symbol"),
        "name": payload.get("name") or payload.get("symbol") or "Unknown security",
        "quantity": str(payload.get("quantity", "0")),
        "statement_value": statement_value,
        "account_ref": payload.get("account_ref"),
        "market": payload.get("market"),
        "currency": payload.get("currency"),
        "exchange": payload.get("exchange"),
        "amount": payload.get("amount"),
        "security_type": (
            override.get("kind") or payload.get("security_type") or payload.get("kind")
        ),
        "sleeve": override.get("sleeve", payload.get("sleeve")),
        "status": item.status,
        "diff": item.diff,
        "match": {"lot_id": item.match_lot_id, "security_id": item.match_security_id},
        "reason": payload.get("reason"),
        "action": payload.get("merge_status"),
        "matched_by": payload.get("matched_by"),
        "can_find_security": _can_find_security(job, item),
    }


def _record_failed_job(session: Session, user_id: UUID, doc_type: str, code: str) -> UUID:
    from wealth.db.models import ImportJob

    job = ImportJob(
        user_id=user_id,
        doc_type=doc_type,
        status="failed",
        error_code=code,
        counts={},
        file_deleted_at=datetime.now(UTC),
    )
    session.add(job)
    session.flush()
    return job.id


def _existing_match(
    session: Session, user_id: UUID, isin: str, account_ref: str | None
) -> tuple[Any | None, Any | None, Decimal]:
    from wealth.db.models import HoldingLot, Security

    security = session.scalar(select(Security).where(Security.isin == isin))
    if security is None:
        return None, None, Decimal(0)
    lots = list(
        session.scalars(
            select(HoldingLot)
            .where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == security.id,
                HoldingLot.source == "cas",
                HoldingLot.account_ref.is_not_distinct_from(account_ref),
            )
            .order_by(HoldingLot.id)
        )
    )
    return security, (lots[0] if lots else None), sum((lot.quantity for lot in lots), Decimal(0))


def _amfi_category(session: Session, isin: str) -> str | None:
    from wealth.db.models import Security

    security = session.scalar(select(Security).where(Security.isin == isin.upper()))
    if security is not None and security.security_type == "mf":
        return security.category
    return None


def _make_job_items(
    session: Session, user_id: UUID, job_id: UUID, parsed: ParsedStatement
) -> dict[str, int]:
    from wealth.db.models import ImportItem

    item_rows: list[ImportItem] = []
    counter: Counter[str] = Counter()
    transactions_by_key: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for transaction in parsed.transactions:
        transactions_by_key.setdefault((transaction.isin, transaction.account_ref), []).append(
            transaction.model_dump(mode="json")
        )

    for holding in parsed.holdings:
        security, match_lot, old_quantity = _existing_match(
            session, user_id, holding.isin, holding.account_ref
        )
        category = security.category if security and security.security_type == "mf" else None
        payload = _holding_payload(
            holding,
            transactions_by_key.get((holding.isin, holding.account_ref), []),
            amfi_category=category,
        )
        classified = classify_holding(holding, category)
        if classified is None:
            status = "unknown"
            diff = None
        elif match_lot is None:
            status = "new"
            diff = None
        elif old_quantity == holding.quantity:
            status = "duplicate"
            diff = None
        else:
            status = "changed"
            diff = {
                "quantity": {
                    "old": _decimal_string(old_quantity),
                    "new": _decimal_string(holding.quantity),
                }
            }
        counter[status] += 1
        item_rows.append(
            ImportItem(
                user_id=user_id,
                job_id=job_id,
                item_kind=holding.kind,
                payload=payload,
                status=status,
                match_lot_id=match_lot.id if match_lot else None,
                match_security_id=security.id if security else None,
                diff=diff,
            )
        )

    session.add_all(item_rows)
    return {key: counter[key] for key in ("new", "changed", "duplicate", "unknown")}


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    """Create tenant-scoped CAS upload, review, and confirmation routes."""
    from wealth.api.money_map import get_today
    from wealth.db.models import AuditLog, ImportItem, ImportJob, Security

    router = APIRouter()
    auth_dependency = Depends(authenticated_user)
    today_dependency = Depends(get_today)

    @router.get("/v1/securities/search", response_model=list[SecuritySearchResult])
    def search_securities(
        request: Request,
        q: str = Query(..., min_length=1, max_length=80),
        market: Literal["IN", "US", "DK", "EU", "GB"] = Query(...),
        user: Any = auth_dependency,
    ) -> Any:
        query = q.strip().casefold()
        if len(query) < 2:
            return _problem(request, 422, "invalid_search", "Enter at least two characters.")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = user.session.scalars(
            select(Security)
            .where(
                Security.market == market,
                Security.symbol.is_not(None),
                Security.security_type.in_(("equity", "etf")),
                (
                    func.lower(Security.symbol).like(f"{escaped}%", escape="\\")
                    | func.lower(Security.name).like(f"%{escaped}%", escape="\\")
                ),
            )
            .order_by(
                case((func.lower(Security.symbol) == query, 0), else_=1),
                func.lower(Security.name),
            )
            .limit(8)
        )
        return [
            {
                "id": security.id,
                "symbol": security.symbol,
                "name": security.name,
                "security_type": security.security_type,
            }
            for security in rows
        ]

    @router.post(
        "/v1/imports",
        status_code=201,
        response_model=ImportUploadResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "password": {"type": "string"},
                                "doc_type": {"type": "string", "maxLength": 32},
                                "account_id": {"type": "string", "format": "uuid"},
                                "member_id": {"type": "string", "format": "uuid"},
                                "broker": {"type": "string"},
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_import(
        request: Request,
        today: date = today_dependency,
        user: Any = auth_dependency,
    ) -> Any:
        max_mb = getattr(request.app.state.settings, "max_upload_mb", MAX_UPLOAD_MB)
        max_bytes = max_mb * 1024 * 1024
        form: FormData | None = None
        try:
            form = await _InMemoryMultipartParser(
                request.headers,
                request.stream(),
                max_file_bytes=max_bytes,
            ).parse()
        except _UploadTooLarge:
            return _problem(
                request,
                413,
                "too_large",
                f"The PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
            )
        except (MultiPartException, ValueError):
            return _problem(request, 422, "invalid_upload", "The multipart upload is invalid.")

        try:
            file = form.get("file") if form is not None else None
            password_value = form.get("password") if form is not None else None
            doc_type_value = form.get("doc_type") if form is not None else None
            account_id_value = form.get("account_id") if form is not None else None
            member_id_value = form.get("member_id") if form is not None else None
            broker_value = form.get("broker") if form is not None else None
            if not isinstance(file, UploadFile):
                return _problem(request, 422, "invalid_upload", "A statement file is required.")
            if password_value is not None and not isinstance(password_value, str):
                return _problem(request, 422, "invalid_upload", "The password field is invalid.")
            if doc_type_value is not None and not isinstance(doc_type_value, str):
                return _problem(request, 422, "invalid_upload", "The document type is invalid.")
            if account_id_value is not None and not isinstance(account_id_value, str):
                return _problem(request, 422, "invalid_upload", "The account selection is invalid.")
            if member_id_value is not None and not isinstance(member_id_value, str):
                return _problem(request, 422, "invalid_upload", "The member selection is invalid.")
            if broker_value is not None and not isinstance(broker_value, str):
                return _problem(request, 422, "invalid_upload", "The broker selection is invalid.")
            broker_name = (broker_value or "other").strip().casefold()
            from wealth.importer.brokers.parser import load_broker_config

            if broker_name not in load_broker_config():
                return _problem(request, 422, "invalid_broker", "Choose a supported broker.")
            doc_type = (doc_type_value or "cas")[:32]
            if doc_type_value is not None and len(doc_type_value) > 32:
                return _problem(request, 422, "invalid_upload", "The document type is too long.")
            data = await file.read()
        finally:
            if form is not None:
                await form.close()

        filename = file.filename or ""
        suffix = Path(filename).suffix.casefold()
        from wealth.importer.brokers.parser import detect_broker_file

        file_type = detect_broker_file(data, filename)
        is_pdf = file_type == "pdf"
        is_broker_file = file_type in {"text", "xlsx", "xls", "html_xls"}
        is_text_file = file_type == "text"
        if doc_type_value == "broker_statement" and suffix == ".pdf" and is_pdf:
            return _problem(
                request,
                415,
                "broker_pdf",
                "This looks like a PDF. Please download the Excel or CSV version from your broker.",
            )
        if not (is_pdf or is_broker_file):
            return _problem(
                request,
                415,
                "unsupported_format"
                if suffix in {".csv", ".tsv", ".xls", ".xlsx"}
                else "not_a_pdf",
                "Choose a PDF, CSV, TSV, text, XLS, or XLSX file that matches its file type.",
            )

        if is_broker_file:
            from wealth.api.broker_imports import (
                available_templates,
                broker_defaults,
                broker_for_account,
                find_layout,
                infer_tax_wrapper,
                keep_pending_rows,
                mapping_suggestions,
                masked_samples,
                stage_broker_items,
                stage_ibkr_activity,
                update_tax_wrapper_report,
            )
            from wealth.db.models import Account, HouseholdMember
            from wealth.db.repo import acting_household_member
            from wealth.importer.brokers import (
                is_activity_statement_csv,
                parse_activity_statement_csv,
            )
            from wealth.importer.brokers.parser import auto_map, mapping_default_requirements

            activity = None
            try:
                if is_text_file and is_activity_statement_csv(data):
                    activity = await asyncio.wait_for(
                        asyncio.to_thread(parse_activity_statement_csv, data),
                        timeout=PARSE_TIMEOUT_SECONDS,
                    )
                    broker_name = "ibkr"
                    headers, rows = [], []
                else:
                    headers, rows = await asyncio.wait_for(
                        asyncio.to_thread(_parse_broker_serialized, data, filename),
                        timeout=PARSE_TIMEOUT_SECONDS,
                    )
            except TimeoutError:
                _record_failed_job(
                    user.session, user.household_owner_id, "broker_statement", "parse_timeout"
                )
                return _problem(
                    request,
                    504,
                    "parse_timeout",
                    "The statement took too long to read. Please try a smaller file.",
                )
            except ValueError:
                return _problem(
                    request,
                    422,
                    "invalid_broker_file",
                    "The CSV or XLSX file could not be read.",
                )

            try:
                selected_member_id = UUID(member_id_value) if member_id_value else None
                selected_account_id = UUID(account_id_value) if account_id_value else None
            except ValueError:
                return _problem(request, 422, "invalid_selection", "Choose a valid account owner.")
            account = None
            if selected_account_id is not None:
                account = user.session.scalar(
                    select(Account).where(
                        Account.user_id == user.household_owner_id,
                        Account.id == selected_account_id,
                    )
                )
                if account is None:
                    return _problem(request, 404, "account_not_found", "The account was not found.")
                if selected_member_id is not None and selected_member_id != account.member_id:
                    return _problem(
                        request,
                        422,
                        "account_owner_mismatch",
                        "Choose the owner linked to this account.",
                    )
                selected_member_id = account.member_id
            elif selected_member_id is not None:
                member_exists = user.session.scalar(
                    select(HouseholdMember.id).where(
                        HouseholdMember.user_id == user.household_owner_id,
                        HouseholdMember.id == selected_member_id,
                    )
                )
                if member_exists is None:
                    return _problem(
                        request,
                        404,
                        "member_not_found",
                        "The selected household member was not found.",
                    )
            else:
                selected_member_id = acting_household_member(
                    user.session, user.user_id, user.household_owner_id
                ).id

            broker_name = broker_for_account(broker_name, account)
            market_default, currency_default = broker_defaults(broker_name)
            templates = available_templates(user.session, user.household_owner_id)
            layout = find_layout(headers, templates, broker_name) if activity is None else None
            detected_broker = None
            if layout is not None and layout.wins_over_picker:
                if layout.broker.casefold() != broker_name.casefold():
                    detected_broker = layout.broker
                broker_name = layout.broker
                market_default, currency_default = broker_defaults(broker_name)
            auto_mapped = False
            mapping_kind: str | None = None
            mapping: dict[str, str | None] = {}
            if layout is None and activity is None:
                automatic_layout, mapping_kind, mapping = auto_map(
                    headers,
                    rows,
                    broker=broker_name,
                    market_default=market_default,
                    currency_default=currency_default,
                )
                if automatic_layout is not None:
                    layout = automatic_layout
                    auto_mapped = True
            if layout is None:
                mapping = mapping or mapping_suggestions(headers, mapping_kind, rows)
                needs_market_default, needs_currency_default = mapping_default_requirements(
                    rows, mapping
                )
            elif auto_mapped:
                needs_market_default, needs_currency_default = mapping_default_requirements(
                    rows, mapping
                )
            else:
                needs_market_default = False
                needs_currency_default = False
            now = datetime.now(UTC)
            job = ImportJob(
                user_id=user.household_owner_id,
                doc_type="broker_statement",
                status="ready" if layout else "needs_mapping",
                source="broker",
                statement_date=today,
                member_id=selected_member_id,
                account_id=selected_account_id,
                counts={},
                parse_report={
                    "warnings": list(activity.warnings) if activity is not None else [],
                    "broker": broker_name,
                    **({"detected_broker": detected_broker} if detected_broker else {}),
                    "market_default": market_default,
                    "currency_default": currency_default,
                    "auto_mapped": auto_mapped,
                    "mapping_kind": layout.kind if auto_mapped and layout is not None else None,
                    "columns_used": {},
                    "needs_market_default": needs_market_default,
                    "needs_currency_default": needs_currency_default,
                },
                file_deleted_at=now,
            )
            user.session.add(job)
            user.session.flush()
            if activity is not None:
                counts = stage_ibkr_activity(user.session, user.household_owner_id, job, activity)
                user.session.flush()
                report = dict(job.parse_report or {})
                return ImportUploadResponse(
                    job_id=job.id,
                    status="ready",
                    source="broker",
                    statement_date=job.statement_date,
                    counts=counts,
                    warnings=list(report.get("warnings", [])),
                )
            if layout is None:
                mapping = mapping or mapping_suggestions(headers, mapping_kind, rows)
                job.parse_report = {
                    **(job.parse_report or {}),
                    "headers": headers,
                    "samples": masked_samples(headers, rows),
                    "suggested_mapping": mapping,
                    "mapping_kind": mapping_kind,
                    "warnings": [],
                }
                explicit_wrapper = infer_tax_wrapper(headers, rows)
                update_tax_wrapper_report(job, broker_name, account, explicit_wrapper)
                job.counts = {"new": 0, "changed": 0, "duplicate": 0, "unknown": 0}
                keep_pending_rows(job.id, headers, rows)
                user.session.flush()
                return ImportUploadResponse(
                    job_id=job.id,
                    status="needs_mapping",
                    source="broker",
                    statement_date=job.statement_date,
                    counts=job.counts,
                    warnings=[],
                )
            if auto_mapped:
                report = dict(job.parse_report or {})
                report.update(
                    {
                        "headers": headers,
                        "samples": masked_samples(headers, rows),
                        "suggested_mapping": mapping,
                        "mapping_kind": layout.kind,
                        "columns_used": {
                            field: header
                            for field, header in layout.columns.items()
                            if isinstance(header, str)
                        },
                        "needs_market_default": needs_market_default,
                        "needs_currency_default": needs_currency_default,
                    }
                )
                job.parse_report = report
                keep_pending_rows(job.id, headers, rows)
            counts = stage_broker_items(
                user.session, user.household_owner_id, job, headers, rows, layout
            )
            report = dict(job.parse_report or {})
            update_tax_wrapper_report(
                job,
                layout.broker,
                account,
                str(report["tax_wrapper"]) if report.get("tax_wrapper") else None,
            )
            job.counts = counts
            user.session.flush()
            report = dict(job.parse_report or {})
            return ImportUploadResponse(
                job_id=job.id,
                status="ready",
                source="broker",
                statement_date=job.statement_date,
                counts=counts,
                warnings=list(report.get("warnings", [])),
            )

        try:
            parsed = await asyncio.wait_for(
                asyncio.to_thread(_parse_serialized, data, password_value, max_mb),
                timeout=PARSE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _record_failed_job(user.session, user.household_owner_id, doc_type, "parse_timeout")
            return _problem(
                request,
                504,
                "parse_timeout",
                "The statement took too long to parse. Please try again.",
            )
        except ParseError as error:
            _record_failed_job(user.session, user.household_owner_id, doc_type, error.code)
            return _problem(request, 422, error.code, error.detail)

        job = ImportJob(
            user_id=user.household_owner_id,
            doc_type=doc_type,
            status="ready",
            source=parsed.source,
            statement_date=parsed.statement_date,
            counts={"new": 0, "changed": 0, "duplicate": 0, "unknown": 0},
            parse_report={"warnings": parsed.warnings},
            file_deleted_at=datetime.now(UTC),
        )
        user.session.add(job)
        user.session.flush()
        counts = _make_job_items(user.session, user.household_owner_id, job.id, parsed)
        job.counts = counts
        user.session.flush()
        return ImportUploadResponse(
            job_id=job.id,
            status="ready",
            source=parsed.source,
            statement_date=parsed.statement_date,
            counts=counts,
            warnings=parsed.warnings,
        )

    @router.get("/v1/imports", response_model=list[ImportJobResponse])
    def list_imports(user: Any = auth_dependency) -> list[dict[str, Any]]:
        jobs = user.session.scalars(
            select(ImportJob)
            .where(ImportJob.user_id == user.household_owner_id)
            .order_by(ImportJob.created_at.desc(), ImportJob.id.desc())
            .limit(20)
        )
        return [_job_response(job) for job in jobs]

    @router.get("/v1/imports/{job_id}", response_model=ImportDetailResponse)
    def get_import(job_id: UUID, request: Request, user: Any = auth_dependency) -> Any:
        job = user.session.scalar(
            select(ImportJob).where(
                ImportJob.user_id == user.household_owner_id,
                ImportJob.id == job_id,
            )
        )
        if job is None:
            return _problem(request, 404, "not_found", "The import was not found.")
        items = user.session.scalars(
            select(ImportItem)
            .where(ImportItem.user_id == user.household_owner_id, ImportItem.job_id == job.id)
            .order_by(ImportItem.id)
        )
        report = job.parse_report or {}
        return {
            **_job_response(job),
            "items": [_get_item_response(item, job) for item in items],
            "headers": report.get("headers"),
            "samples": report.get("samples", []),
            "suggested_mapping": report.get("suggested_mapping", {}),
            "mapping_kind": report.get("mapping_kind"),
            "auto_mapped": report.get("auto_mapped", False),
            "columns_used": report.get("columns_used", {}),
            "has_missing_cost": report.get("has_missing_cost", False),
            "skipped_rows": report.get("skipped_rows", {}),
            "needs_market_default": report.get("needs_market_default", False),
            "needs_currency_default": report.get("needs_currency_default", False),
            "broker": report.get("broker"),
            "detected_broker": report.get("detected_broker"),
            "market_default": report.get("market_default", "US"),
            "currency_default": report.get("currency_default", "USD"),
            "tax_wrapper": report.get("tax_wrapper"),
            "tax_wrapper_question": report.get("tax_wrapper_question", False),
            "cash_flow_count": report.get("cash_flow_count", 0),
            "currency_mismatch_values": report.get("currency_mismatch_values", 0),
        }

    @router.post("/v1/imports/{job_id}/mapping", response_model=ImportDetailResponse)
    def map_import_columns(
        job_id: UUID,
        payload: ImportMappingRequest,
        request: Request,
        user: Any = auth_dependency,
    ) -> Any:
        from wealth.api.broker_imports import (
            broker_defaults,
            broker_for_account,
            peek_pending_rows,
            persist_template,
            stage_broker_items,
            take_pending_rows,
            template_layout,
            update_tax_wrapper_report,
            valid_template_mapping,
        )

        job = user.session.scalar(
            select(ImportJob)
            .where(
                ImportJob.user_id == user.household_owner_id,
                ImportJob.id == job_id,
            )
            .with_for_update()
        )
        if job is None:
            return _problem(request, 404, "not_found", "The import was not found.")
        report = dict(job.parse_report or {})
        fixing_auto_mapping = job.status == "ready" and report.get("auto_mapped") is True
        if job.status != "needs_mapping" and not fixing_auto_mapping:
            return _problem(
                request, 409, "mapping_not_needed", "This import no longer needs a mapping."
            )
        pending = peek_pending_rows(job.id)
        if pending is None:
            return _problem(
                request,
                409,
                "mapping_expired",
                "This file is no longer available. Upload it again to continue.",
            )
        headers, rows = pending
        allowed = (
            {
                "isin",
                "symbol",
                "name",
                "quantity",
                "avg_price",
                "cost_value",
                "last_price",
                "exchange",
                "market",
                "currency",
                "market_value",
                "security_type",
                "tax_wrapper",
            }
            if payload.kind == "holdings"
            else {
                "isin",
                "symbol",
                "trade_date",
                "trade_type",
                "quantity",
                "price",
                "exchange",
                "market",
                "currency",
                "order_id",
                "trade_id",
                "amount",
                "commission",
                "security_type",
                "tax_wrapper",
            }
        )
        if (
            not payload.columns
            or set(payload.columns) - allowed
            or not valid_template_mapping(payload.kind, payload.columns, headers)
        ):
            return _problem(
                request, 422, "invalid_mapping", "Choose a valid column for each field."
            )
        if payload.save_as is not None and (
            not payload.save_as.strip() or len(payload.save_as) > 80
        ):
            return _problem(
                request, 422, "invalid_mapping_name", "Use a short name for this format."
            )

        if payload.save_as:
            persist_template(
                user.session,
                user.household_owner_id,
                name=payload.save_as.strip(),
                kind=payload.kind,
                headers=headers,
                columns=payload.columns,
                market_default=payload.market_default,
                currency_default=payload.currency_default,
            )
        from wealth.db.models import Account

        account = (
            user.session.scalar(
                select(Account).where(
                    Account.user_id == user.household_owner_id,
                    Account.id == job.account_id,
                )
            )
            if job.account_id is not None
            else None
        )
        selected_broker = payload.broker
        if selected_broker == "other":
            selected_broker = str(report.get("broker") or "other")
        selected_broker = broker_for_account(selected_broker, account)
        inferred_market, inferred_currency = broker_defaults(selected_broker)
        market_default = payload.market_default or str(
            report.get("market_default") or inferred_market
        )
        currency_default = payload.currency_default or str(
            report.get("currency_default") or inferred_currency
        )
        layout = template_layout(
            payload.kind,
            payload.columns,
            broker=payload.save_as.strip() if payload.save_as else selected_broker,
            market_default=market_default,
            currency_default=currency_default,
        )
        if fixing_auto_mapping:
            user.session.execute(
                delete(ImportItem).where(
                    ImportItem.user_id == user.household_owner_id,
                    ImportItem.job_id == job.id,
                )
            )
            report.pop("columns_used", None)
            report["auto_mapped"] = False
            report["mapping_kind"] = payload.kind
            job.parse_report = report
        job.counts = stage_broker_items(
            user.session, user.household_owner_id, job, headers, rows, layout
        )
        report = dict(job.parse_report or {})
        update_tax_wrapper_report(
            job,
            selected_broker,
            account,
            str(report["tax_wrapper"]) if report.get("tax_wrapper") else None,
        )
        report = dict(job.parse_report or {})
        job.parse_report = {
            **report,
            "broker": selected_broker,
            "market_default": market_default,
            "currency_default": currency_default,
            **({"template_name": payload.save_as.strip()} if payload.save_as else {}),
        }
        take_pending_rows(job.id)
        user.session.flush()
        items = user.session.scalars(
            select(ImportItem)
            .where(ImportItem.user_id == user.household_owner_id, ImportItem.job_id == job.id)
            .order_by(ImportItem.id)
        )
        return {
            **_job_response(job),
            "items": [_get_item_response(item, job) for item in items],
            "headers": (job.parse_report or {}).get("headers"),
            "samples": (job.parse_report or {}).get("samples", []),
            "suggested_mapping": (job.parse_report or {}).get("suggested_mapping", {}),
            "mapping_kind": (job.parse_report or {}).get("mapping_kind"),
            "auto_mapped": (job.parse_report or {}).get("auto_mapped", False),
            "columns_used": (job.parse_report or {}).get("columns_used", {}),
            "has_missing_cost": (job.parse_report or {}).get("has_missing_cost", False),
            "skipped_rows": (job.parse_report or {}).get("skipped_rows", {}),
            "needs_market_default": (job.parse_report or {}).get("needs_market_default", False),
            "needs_currency_default": (job.parse_report or {}).get("needs_currency_default", False),
            "broker": (job.parse_report or {}).get("broker"),
            "detected_broker": (job.parse_report or {}).get("detected_broker"),
            "market_default": (job.parse_report or {}).get("market_default", "US"),
            "currency_default": (job.parse_report or {}).get("currency_default", "USD"),
            "tax_wrapper": (job.parse_report or {}).get("tax_wrapper"),
            "tax_wrapper_question": (job.parse_report or {}).get("tax_wrapper_question", False),
            "cash_flow_count": (job.parse_report or {}).get("cash_flow_count", 0),
            "currency_mismatch_values": (job.parse_report or {}).get("currency_mismatch_values", 0),
        }

    @router.patch("/v1/imports/{job_id}/items/{item_id}", response_model=ImportItemResponse)
    async def patch_import_item(
        job_id: UUID,
        item_id: UUID,
        request: Request,
        user: Any = auth_dependency,
    ) -> Any:
        job = user.session.scalar(
            select(ImportJob).where(
                ImportJob.user_id == user.household_owner_id,
                ImportJob.id == job_id,
            )
        )
        item = user.session.scalar(
            select(ImportItem).where(
                ImportItem.user_id == user.household_owner_id,
                ImportItem.job_id == job_id,
                ImportItem.id == item_id,
            )
        )
        if job is None or item is None:
            return _problem(request, 404, "not_found", "The import item was not found.")
        if job.status != "ready":
            return _problem(request, 409, "import_not_ready", "The import is no longer editable.")
        try:
            raw = await request.json()
            patch = ImportItemPatch.model_validate(raw)
            if patch.status is None and patch.override is None and patch.security_id is None:
                raise ValueError("An item status or override is required.")
        except (ValueError, ValidationError):
            return _problem(request, 422, "invalid_item_update", "The item update is invalid.")

        if patch.security_id is not None:
            if patch.status is not None or patch.override is not None:
                return _problem(
                    request,
                    422,
                    "invalid_security_match",
                    "Choose a security for an unmatched broker row.",
                )
            if not _can_find_security(job, item):
                return _problem(
                    request,
                    422,
                    "invalid_security_match",
                    "Choose a security for an unmatched broker row.",
                )
            security = user.session.scalar(select(Security).where(Security.id == patch.security_id))
            row_market = str(item.payload.get("market") or "").upper()
            if security is None or security.market != row_market:
                return _problem(
                    request,
                    422,
                    "invalid_security_match",
                    "Choose a security from the same market as this row.",
                )
            from wealth.api.broker_imports import _holding_match_status

            report = job.parse_report or {}
            source_broker = str(report.get("detected_broker") or report.get("broker") or "other")
            status, diff, match_lot_id, reason = _holding_match_status(
                user.session,
                user.household_owner_id,
                job,
                item.payload,
                security,
                source_broker,
            )
            payload = dict(item.payload)
            payload.update(
                {
                    "symbol": security.symbol,
                    "isin": security.isin,
                    "matched_by": "owner",
                    "merge_status": status,
                }
            )
            if status == "unknown":
                payload["reason"] = reason
            else:
                payload.pop("reason", None)
            item.payload = payload
            item.status = status
            item.match_security_id = security.id
            item.match_lot_id = match_lot_id
            item.diff = diff
            user.session.flush()
            return _get_item_response(item, job)

        if patch.override is not None:
            if job.source == "broker":
                return _problem(
                    request,
                    422,
                    "invalid_override",
                    "This share could not be matched to a security.",
                )
            if classify_holding(
                _parsed_holding(item.payload),
                _amfi_category(user.session, item.payload.get("isin", "")),
            ) is not None or not _ISIN.fullmatch(item.payload.get("isin", "")):
                return _problem(
                    request,
                    422,
                    "invalid_override",
                    "Only an unknown holding with a valid ISIN can be classified.",
                )
            item.user_override = patch.override.model_dump(mode="json")
            if patch.status is None:
                item.status = "accepted"
        if patch.status is not None:
            item.status = patch.status
        user.session.flush()
        return _get_item_response(item, job)

    @router.post("/v1/imports/{job_id}/confirm", response_model=ImportConfirmationResponse)
    def confirm_import(
        job_id: UUID,
        request: Request,
        payload: ImportConfirmationRequest | None = None,
        today: date = today_dependency,
        user: Any = auth_dependency,
    ) -> Any:
        job = user.session.scalar(
            select(ImportJob)
            .where(ImportJob.user_id == user.household_owner_id, ImportJob.id == job_id)
            .with_for_update()
        )
        if job is None:
            return _problem(request, 404, "not_found", "The import was not found.")
        if job.status == "confirmed":
            return _problem(request, 409, "already_confirmed", "The import was already confirmed.")
        if job.status != "ready":
            return _problem(request, 409, "import_not_ready", "The import cannot be confirmed.")

        items = list(
            user.session.scalars(
                select(ImportItem)
                .where(ImportItem.user_id == user.household_owner_id, ImportItem.job_id == job.id)
                .order_by(ImportItem.id)
            )
        )
        if job.source == "broker":
            if any(item.status == "quantity_mismatch" for item in items):
                return _problem(
                    request,
                    409,
                    "quantity_choice_required",
                    "Choose which holdings to keep for every quantity difference.",
                )
            from wealth.api.broker_imports import (
                broker_label,
                confirm_broker_items,
                persist_template,
                take_pending_rows,
            )
            from wealth.db.models import Account

            report = dict(job.parse_report or {})
            wrapper = report.get("tax_wrapper")
            if wrapper is None and report.get("tax_wrapper_question"):
                wrapper = payload.tax_wrapper if payload and payload.tax_wrapper else "ordinary"
            if wrapper and job.account_id is not None:
                account = user.session.scalar(
                    select(Account).where(
                        Account.user_id == user.household_owner_id,
                        Account.id == job.account_id,
                    )
                )
                if account is not None:
                    account.tax_wrapper = wrapper
                    report["tax_wrapper"] = wrapper
                    report["tax_wrapper_question"] = False
                    job.parse_report = report

            applied = Counter(
                confirm_broker_items(user.session, user.household_owner_id, job, items)
            )
            from wealth.api.recorded_buys import match_recorded_buys_for_import
            from wealth.db.repo import acting_household_member

            importing_member_id = (
                job.member_id
                or acting_household_member(user.session, user.user_id, user.household_owner_id).id
            )
            confirmed_security_ids = {
                item.match_security_id
                for item in items
                if item.match_security_id is not None
                and item.status
                in {
                    "new",
                    "changed",
                    "accepted",
                    "enrich",
                    "replace_with_lots",
                    "quantity_mismatch",
                }
            }
            recorded_buys_matched = match_recorded_buys_for_import(
                user.session,
                user.household_owner_id,
                importing_member_id,
                job.source,
                job.statement_date,
                today,
                confirmed_security_ids,
            )
            applied["recorded_buys_matched"] = recorded_buys_matched
            if report.get("auto_mapped") is True:
                columns_used = report.get("columns_used")
                headers = report.get("headers")
                kind = report.get("mapping_kind")
                broker = str(report.get("broker") or "other")
                if (
                    isinstance(columns_used, dict)
                    and isinstance(headers, list)
                    and kind in {"holdings", "tradebook"}
                ):
                    persist_template(
                        user.session,
                        user.household_owner_id,
                        name=broker_label(broker),
                        kind=kind,
                        headers=[str(header) for header in headers],
                        columns={
                            str(field): str(header)
                            for field, header in columns_used.items()
                            if isinstance(header, str)
                        },
                        market_default=str(report.get("market_default") or "US"),
                        currency_default=str(report.get("currency_default") or "USD"),
                    )
            take_pending_rows(job.id)
            job.status = "confirmed"
            job.confirmed_at = datetime.now(UTC)
            user.session.add(
                AuditLog(
                    user_id=user.household_owner_id,
                    actor=str(user.user_id),
                    actor_user_id=user.user_id,
                    action="broker_import_confirmed",
                    detail={
                        **dict(applied),
                        "skipped": max(0, len(items) - sum(applied.values())),
                    },
                )
            )
            from wealth.api.money_map import refresh_networth_snapshot

            totals = refresh_networth_snapshot(user.session, user.household_owner_id, today)
            return {
                "job_id": job.id,
                "status": job.status,
                "counts": dict(applied),
                "net_worth": totals["net_worth"],
            }
        applied = Counter()
        from wealth.db.repo import acting_household_member

        member_id = (
            job.member_id
            or acting_household_member(user.session, user.user_id, user.household_owner_id).id
        )
        confirmed_security_ids: set[UUID] = set()
        for item in items:
            if item.status in {"rejected", "duplicate"}:
                continue
            payload = item.payload
            holding = _parsed_holding(payload)
            classification = classify_holding(holding, _amfi_category(user.session, holding.isin))
            override = item.user_override or {}
            if classification is None and not override:
                continue
            if item.status not in {"new", "changed", "accepted"}:
                continue

            kind = override.get("kind", holding.kind)
            sleeve = override.get("sleeve", classification[0] if classification else None)
            category = classification[1] if classification else f"{kind}:{sleeve}"
            isin = holding.isin.upper()
            market = "IN" if isin.startswith("IN") else "US"
            security_id = user.session.scalar(
                text(
                    "SELECT app_upsert_security(:market, :isin, :symbol, :name, :kind, "
                    ":currency, :sleeve, :category)"
                ),
                {
                    "market": market,
                    "isin": isin,
                    "symbol": isin,
                    "name": holding.name[:160],
                    "kind": kind,
                    "currency": "INR" if market == "IN" else "USD",
                    "sleeve": sleeve,
                    "category": category,
                },
            )
            security_id = security_id if isinstance(security_id, UUID) else UUID(str(security_id))
            _replace_cas_lots(
                user.session, user.household_owner_id, job, item, holding, security_id, member_id
            )
            if kind == "mf" and job.source == "cams_kfin":
                _persist_fund_transactions(
                    user.session,
                    user.household_owner_id,
                    member_id,
                    security_id,
                    payload,
                    holding.account_ref,
                )
            item.match_security_id = security_id
            confirmed_security_ids.add(security_id)
            applied[item.status] += 1

        from wealth.api.recorded_buys import match_recorded_buys_for_import

        recorded_buys_matched = match_recorded_buys_for_import(
            user.session,
            user.household_owner_id,
            member_id,
            job.source,
            job.statement_date,
            today,
            confirmed_security_ids,
        )
        applied["recorded_buys_matched"] = recorded_buys_matched
        job.counts = {**dict(job.counts or {}), "recorded_buys_matched": recorded_buys_matched}
        job.status = "confirmed"
        job.confirmed_at = datetime.now(UTC)
        user.session.add(
            AuditLog(
                user_id=user.household_owner_id,
                actor=str(user.user_id),
                actor_user_id=user.user_id,
                action="cas_import_confirmed",
                detail={
                    "new": applied["new"],
                    "changed": applied["changed"],
                    "accepted": applied["accepted"],
                    "recorded_buys_matched": recorded_buys_matched,
                    "skipped": len(items) - sum(applied.values()),
                },
            )
        )
        from wealth.api.money_map import refresh_networth_snapshot

        totals = refresh_networth_snapshot(user.session, user.household_owner_id, today)
        return {
            "job_id": job.id,
            "status": job.status,
            "counts": job.counts,
            "net_worth": totals["net_worth"],
        }

    @router.delete(
        "/v1/imports/{job_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def discard_import(job_id: UUID, request: Request, user: Any = auth_dependency) -> Any:
        job = user.session.scalar(
            select(ImportJob).where(
                ImportJob.user_id == user.household_owner_id,
                ImportJob.id == job_id,
            )
        )
        if job is None:
            return _problem(request, 404, "not_found", "The import was not found.")
        if job.status == "confirmed":
            return _problem(
                request,
                409,
                "already_confirmed",
                "A confirmed import cannot be discarded.",
            )
        if job.source == "broker":
            from wealth.api.broker_imports import discard_pending_rows

            discard_pending_rows(job.id)
        user.session.execute(
            delete(ImportItem).where(
                ImportItem.user_id == user.household_owner_id,
                ImportItem.job_id == job.id,
            )
        )
        job.status = "discarded"
        return Response(status_code=204)

    return router


def _parsed_holding(payload: dict[str, Any]) -> ParsedHolding:
    fields = ParsedHolding.model_fields
    return ParsedHolding.model_validate({key: payload[key] for key in fields if key in payload})


def _job_response(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "doc_type": job.doc_type,
        "status": job.status,
        "error_code": job.error_code,
        "source": job.source,
        "statement_date": job.statement_date,
        "account_id": job.account_id,
        "member_id": job.member_id,
        "counts": job.counts,
        "created_at": job.created_at,
        "confirmed_at": job.confirmed_at,
    }


def _transaction_lots(payload: dict[str, Any]) -> list[dict[str, Any]]:
    transactions = payload.get("transactions") or []
    lot_rows: list[dict[str, Any]] = []
    for index, transaction in sorted(
        enumerate(transactions), key=lambda pair: (pair[1]["date"], pair[0])
    ):
        if transaction.get("kind") in {"buy", "switch_in", "dividend_reinvest"}:
            units = Decimal(transaction["units"])
            amount = Decimal(transaction["amount"])
            if units > 0:
                lot_rows.append(
                    {
                        "units": units,
                        "cost_per_unit": (amount / units).quantize(
                            Decimal("0.0001"), rounding=ROUND_HALF_UP
                        ),
                        "acquired_on": date.fromisoformat(transaction["date"]),
                        "index": index,
                    }
                )
        elif transaction.get("kind") in {"sell", "switch_out"}:
            units_to_remove = Decimal(transaction["units"])
            for lot in lot_rows:
                removed = min(lot["units"], units_to_remove)
                lot["units"] -= removed
                units_to_remove -= removed
                if units_to_remove <= 0:
                    break
    if not lot_rows:
        return []

    target_quantity = Decimal(payload["quantity"])
    lot_rows = [lot for lot in lot_rows if lot["units"] > 0]
    current_quantity = sum((lot["units"] for lot in lot_rows), Decimal(0))
    if current_quantity > target_quantity:
        excess = current_quantity - target_quantity
        for lot in reversed(lot_rows):
            removed = min(lot["units"], excess)
            lot["units"] -= removed
            excess -= removed
            if excess <= 0:
                break
        lot_rows = [lot for lot in lot_rows if lot["units"] > 0]
    elif current_quantity < target_quantity:
        lot_rows.append(
            {
                "units": target_quantity - current_quantity,
                "cost_per_unit": None,
                "acquired_on": None,
                "index": len(transactions),
            }
        )
    return lot_rows


def _persist_fund_transactions(
    session: Session,
    user_id: UUID,
    member_id: UUID,
    security_id: UUID,
    payload: dict[str, Any],
    fallback_account_ref: str | None,
) -> None:
    from wealth.db.models import FundTransaction

    allowed_kinds = {"buy", "sell", "switch_in", "switch_out", "dividend_reinvest", "other"}
    seen: set[tuple[date, str, Decimal, Decimal, str | None]] = set()
    for transaction in payload.get("transactions") or []:
        try:
            transaction_date = date.fromisoformat(str(transaction["date"]))
            kind = str(transaction["kind"])
            amount = Decimal(str(transaction["amount"]))
            units = Decimal(str(transaction["units"]))
        except (InvalidOperation, KeyError, TypeError, ValueError):
            continue
        if (
            kind not in allowed_kinds
            or not amount.is_finite()
            or not units.is_finite()
            or amount < 0
            or units < 0
        ):
            continue
        account_ref = transaction.get("account_ref") or fallback_account_ref
        if account_ref is not None and not re.fullmatch(r"XXXX[A-Z0-9]{4}", str(account_ref)):
            account_ref = None
        signature = (transaction_date, kind, amount, units, account_ref)
        if signature in seen:
            continue
        seen.add(signature)
        exists = session.scalar(
            select(FundTransaction.id).where(
                FundTransaction.user_id == user_id,
                FundTransaction.member_id == member_id,
                FundTransaction.security_id == security_id,
                FundTransaction.transaction_date == transaction_date,
                FundTransaction.kind == kind,
                FundTransaction.amount == amount,
                FundTransaction.units == units,
                FundTransaction.account_ref.is_not_distinct_from(account_ref),
            )
        )
        if exists is None:
            session.add(
                FundTransaction(
                    user_id=user_id,
                    member_id=member_id,
                    security_id=security_id,
                    transaction_date=transaction_date,
                    kind=kind,
                    amount=amount,
                    units=units,
                    account_ref=account_ref,
                )
            )


def _replace_cas_lots(
    session: Session,
    user_id: UUID,
    job: Any,
    item: Any,
    holding: ParsedHolding,
    security_id: UUID,
    member_id: UUID,
) -> None:
    from wealth.db.models import HoldingLot, ImportItem

    matches = list(
        session.scalars(
            select(HoldingLot.id).where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == security_id,
                HoldingLot.source == "cas",
                HoldingLot.account_ref.is_not_distinct_from(holding.account_ref),
            )
        )
    )
    if matches:
        session.execute(
            update(ImportItem)
            .where(
                ImportItem.user_id == user_id,
                ImportItem.match_lot_id.in_(matches),
            )
            .values(match_lot_id=None)
        )
    session.execute(
        delete(HoldingLot).where(
            HoldingLot.user_id == user_id,
            HoldingLot.security_id == security_id,
            HoldingLot.source == "cas",
            HoldingLot.account_ref.is_not_distinct_from(holding.account_ref),
        )
    )

    lots = _transaction_lots(item.payload)
    if not lots:
        lots = [
            {
                "units": holding.quantity,
                "cost_per_unit": (
                    (holding.cost_value / holding.quantity).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                    if holding.cost_value is not None and holding.quantity > 0
                    else None
                ),
                "acquired_on": None,
                "index": 0,
            }
        ]

    total_value = (
        Decimal(item.payload["statement_value"])
        if item.payload.get("statement_value") is not None
        else None
    )
    remaining_value = total_value
    total_quantity = sum((lot["units"] for lot in lots), Decimal(0))
    for index, lot in enumerate(lots):
        if remaining_value is None or total_value is None:
            value = None
        elif index == len(lots) - 1:
            value = remaining_value
        else:
            value = (total_value * lot["units"] / total_quantity).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            remaining_value -= value
        session.add(
            HoldingLot(
                user_id=user_id,
                member_id=member_id,
                joint=False,
                security_id=security_id,
                quantity=lot["units"],
                cost_per_unit=lot["cost_per_unit"],
                acquired_on=lot["acquired_on"],
                source="cas",
                account_ref=holding.account_ref,
                statement_value=value,
                statement_date=job.statement_date,
                import_job_id=job.id,
            )
        )
