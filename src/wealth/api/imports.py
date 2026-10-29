"""Authenticated, in-memory CAS upload and review endpoints."""

from __future__ import annotations

import asyncio
import re
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete, select, text, update
from sqlalchemy.orm import Session
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from wealth.importer.cas import ParseError, parse_cas
from wealth.importer.classify import classify_holding
from wealth.importer.models import ParsedHolding, ParsedStatement

MAX_UPLOAD_MB = 10
PARSE_TIMEOUT_SECONDS = 20
_MAX_MULTIPART_FIELDS = 2
_MAX_FIELD_BYTES = 4096
_PARSE_SEMAPHORE = threading.BoundedSemaphore(1)
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_KINDS = ("equity", "etf", "mf", "bond", "sgb")


class ImportApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImportUploadResponse(ImportApiModel):
    job_id: UUID
    status: Literal["ready"]
    source: Literal["nsdl", "cdsl", "cams_kfin"]
    statement_date: date
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


class ImportItemResponse(ImportApiModel):
    id: UUID
    kind: str
    isin: str
    name: str
    quantity: str
    statement_value: str | None
    account_ref: str | None
    sleeve: str | None
    status: str
    diff: dict[str, Any] | None
    match: dict[str, UUID | None]


class ImportJobResponse(ImportApiModel):
    id: UUID
    doc_type: str
    status: Literal["ready", "confirmed", "failed", "discarded"]
    error_code: str | None
    source: Literal["nsdl", "cdsl", "cams_kfin"] | None
    statement_date: date | None
    counts: dict[str, int]
    created_at: datetime
    confirmed_at: datetime | None


class ImportDetailResponse(ImportJobResponse):
    items: list[ImportItemResponse]


class ImportConfirmationResponse(ImportApiModel):
    job_id: UUID
    status: Literal["confirmed"]
    counts: dict[str, int]
    net_worth: str


class _UploadTooLarge(Exception):
    pass


class _NotPdf(Exception):
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
        except (_NotPdf, _UploadTooLarge):
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
                if len(self._file_prefix) == 4 and self._file_prefix != b"%PDF":
                    raise _NotPdf
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


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


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


def _get_item_response(item: Any) -> dict[str, Any]:
    payload = item.payload
    override = item.user_override or {}
    return {
        "id": item.id,
        "kind": override.get("kind", item.item_kind),
        "isin": payload["isin"],
        "name": payload["name"],
        "quantity": payload["quantity"],
        "statement_value": payload.get("statement_value"),
        "account_ref": payload.get("account_ref"),
        "sleeve": override.get("sleeve", payload.get("sleeve")),
        "status": item.status,
        "diff": item.diff,
        "match": {"lot_id": item.match_lot_id, "security_id": item.match_security_id},
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
    from wealth.db.models import AuditLog, ImportItem, ImportJob

    router = APIRouter()
    auth_dependency = Depends(authenticated_user)
    today_dependency = Depends(get_today)

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
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_import(request: Request, user: Any = auth_dependency) -> Any:
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
        except _NotPdf:
            return _problem(request, 415, "not_a_pdf", "The uploaded file is not a PDF.")
        except (MultiPartException, ValueError):
            return _problem(request, 422, "invalid_upload", "The multipart upload is invalid.")

        try:
            file = form.get("file") if form is not None else None
            password_value = form.get("password") if form is not None else None
            doc_type_value = form.get("doc_type") if form is not None else None
            if not isinstance(file, UploadFile):
                return _problem(request, 422, "invalid_upload", "A PDF file is required.")
            if password_value is not None and not isinstance(password_value, str):
                return _problem(request, 422, "invalid_upload", "The password field is invalid.")
            if doc_type_value is not None and not isinstance(doc_type_value, str):
                return _problem(request, 422, "invalid_upload", "The document type is invalid.")
            doc_type = (doc_type_value or "cas")[:32]
            if doc_type_value is not None and len(doc_type_value) > 32:
                return _problem(request, 422, "invalid_upload", "The document type is too long.")
            data = await file.read()
        finally:
            if form is not None:
                await form.close()

        if len(data) < 4 or not data.startswith(b"%PDF"):
            return _problem(request, 415, "not_a_pdf", "The uploaded file is not a PDF.")

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
        return {**_job_response(job), "items": [_get_item_response(item) for item in items]}

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
            if patch.status is None and patch.override is None:
                raise ValueError("An item status or override is required.")
        except (ValueError, ValidationError):
            return _problem(request, 422, "invalid_item_update", "The item update is invalid.")

        if patch.override is not None:
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
        return _get_item_response(item)

    @router.post("/v1/imports/{job_id}/confirm", response_model=ImportConfirmationResponse)
    def confirm_import(
        job_id: UUID,
        request: Request,
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
        applied = Counter()
        from wealth.db.repo import acting_household_member

        member_id = acting_household_member(user.session, user.user_id, user.household_owner_id).id
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
            applied[item.status] += 1

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
