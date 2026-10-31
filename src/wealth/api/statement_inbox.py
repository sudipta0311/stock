"""Household statement upload keys and password-protected inbox processing."""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException

from wealth.api.auth_rate_limit import client_address
from wealth.api.imports import (
    MAX_UPLOAD_MB,
    PARSE_TIMEOUT_SECONDS,
    _InMemoryMultipartParser,
    _make_job_items,
    _parse_serialized,
    _UploadTooLarge,
)
from wealth.db.models import (
    ImportJob,
    StatementInboxItem,
    StatementInboxKey,
    StatementSavedPassword,
)
from wealth.db.session import user_session
from wealth.importer.cas import ParseError

_KEY_PATTERN = re.compile(r"^wsk_[A-Za-z0-9_-]{43}$")
_SENDER_TYPES = ("cams", "kfintech", "mfcentral", "nsdl", "cdsl", "other")
# Fund houses also email purchase notes and single-fund statements; only the
# consolidated account statement (CAS) can be imported.
_FULL_STATEMENT_SUBJECT = re.compile(
    r"consolidated\s+account\s+statement|\b(?:e-?)?cas\b", re.IGNORECASE
)
_SUBJECT_PREVIEW_CHARS = 140
_NOT_A_FULL_STATEMENT = "This file isn't a full account statement, so there is nothing to import."


class InboxApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StatementInboxKeyCreate(InboxApiModel):
    label: str = Field(min_length=1, max_length=80)


class StatementInboxKeyResponse(InboxApiModel):
    id: UUID
    label: str
    created_at: datetime
    last_used_at: datetime | None


class StatementInboxKeyCreated(StatementInboxKeyResponse):
    key: str


class StatementInboxItemResponse(InboxApiModel):
    id: UUID
    sender_type: Literal["cams", "kfintech", "mfcentral", "nsdl", "cdsl", "other"]
    received_at: datetime
    status: Literal["needs_password", "review", "unreadable"]
    error_code: str | None
    import_job_id: UUID | None
    subject: str = ""
    full_statement: bool = True
    saved_password: bool = False


class StatementInboxListResponse(InboxApiModel):
    items: list[StatementInboxItemResponse]
    cleanup_notice: str | None = None


class StatementInboxUploadResponse(InboxApiModel):
    status: Literal["received", "duplicate"]


class StatementInboxProcessRequest(InboxApiModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, str_strip_whitespace=True)

    password: str | None = Field(default=None, min_length=1, max_length=256)
    remember: bool = False
    # Statements that arrived before a password was saved can still use it.
    use_saved_password: bool = False


class StatementInboxProcessResponse(InboxApiModel):
    status: Literal["review"]
    job_id: UUID
    password_saved: bool
    notice: str | None = None


class _StatementProcessingFailure(Exception):
    def __init__(self, status: int, code: str, detail: str) -> None:
        self.status = status
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _problem(request: Request, status: int, code: str, detail: str) -> JSONResponse:
    title = {
        401: "Unauthorized",
        413: "Content Too Large",
        415: "Unsupported Media Type",
        422: "Unprocessable Content",
        429: "Too Many Requests",
        503: "Service Unavailable",
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


def _require_writer(user: Any) -> None:
    if user.role not in {"owner", "partner"}:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="read_only")


def _sender_type(sender: str, subject: str) -> str:
    value = f"{sender} {subject}".casefold()
    return next((name for name in _SENDER_TYPES[:-1] if name in value), "other")


def _parse_received_at(value: str) -> datetime | None:
    try:
        received_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=UTC)
    return received_at.astimezone(UTC)


def _aad(user_id: UUID, sender_type: str) -> bytes:
    return f"{user_id}:{sender_type}".encode("ascii")


def _remember_password(
    session: Session,
    user_id: UUID,
    sender_type: str,
    password: str,
    key: bytes,
) -> None:
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, password.encode("utf-8"), _aad(user_id, sender_type))
    saved = session.scalar(
        select(StatementSavedPassword).where(
            StatementSavedPassword.user_id == user_id,
            StatementSavedPassword.sender_type == sender_type,
        )
    )
    if saved is None:
        session.add(
            StatementSavedPassword(
                user_id=user_id,
                sender_type=sender_type,
                nonce=nonce,
                ciphertext=ciphertext,
                last_used_at=datetime.now(UTC),
            )
        )
    else:
        saved.nonce = nonce
        saved.ciphertext = ciphertext
        saved.last_used_at = datetime.now(UTC)


def _saved_password(saved: StatementSavedPassword, user_id: UUID, key: bytes) -> str | None:
    try:
        return (
            AESGCM(key)
            .decrypt(
                saved.nonce,
                saved.ciphertext,
                _aad(user_id, saved.sender_type),
            )
            .decode("utf-8")
        )
    except (InvalidTag, UnicodeDecodeError, ValueError):
        return None


async def _process_item(
    session: Session,
    item: StatementInboxItem,
    password: str,
    *,
    max_mb: int,
    password_key: bytes | None,
    remember: bool,
) -> StatementInboxProcessResponse:
    pdf_data = item.pdf_data
    if not pdf_data:
        raise _StatementProcessingFailure(
            422, "unreadable", "We couldn't read this file. Try uploading it in Money."
        )
    try:
        parsed = await asyncio.wait_for(
            asyncio.to_thread(_parse_serialized, pdf_data, password, max_mb),
            timeout=PARSE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise _StatementProcessingFailure(
            504, "parse_timeout", "This statement took too long to read. Please try again."
        ) from None
    except ParseError as error:
        if error.code == "wrong_password":
            item.error_code = "wrong_password"
            session.flush()
            raise _StatementProcessingFailure(
                422, "wrong_password", "That password didn't work. Please try again."
            ) from None
        # Keep the PDF (the 30-day cleanup still removes it): a reader fix can then
        # open it on the next try, and Gmail won't send the same statement again.
        if error.code == "unsupported_format" and item.sender_type != "cdsl":
            item.error_code = "not_a_full_statement"
            session.flush()
            raise _StatementProcessingFailure(
                422, "not_a_full_statement", _NOT_A_FULL_STATEMENT
            ) from None
        item.error_code = "unreadable"
        session.flush()
        raise _StatementProcessingFailure(
            422, "unreadable", "We couldn't read this file. Try uploading it in Money."
        ) from None

    now = datetime.now(UTC)
    job = ImportJob(
        user_id=item.user_id,
        doc_type="cas",
        status="ready",
        source=parsed.source,
        statement_date=parsed.statement_date,
        counts={"new": 0, "changed": 0, "duplicate": 0, "unknown": 0},
        parse_report={"warnings": parsed.warnings},
        file_deleted_at=now,
    )
    session.add(job)
    session.flush()
    job.counts = _make_job_items(session, item.user_id, job.id, parsed)
    item.status = "review"
    item.error_code = None
    item.import_job_id = job.id
    item.pdf_data = None
    password_saved = False
    notice = None
    if remember:
        if password_key is None:
            notice = "Saved passwords are unavailable. Enter this password each time."
        else:
            _remember_password(session, item.user_id, item.sender_type, password, password_key)
            password_saved = True
    session.flush()
    return StatementInboxProcessResponse(
        status="review", job_id=job.id, password_saved=password_saved, notice=notice
    )


def _decode_upload_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, separator, key = authorization.partition(" ")
    if not separator or scheme.casefold() != "statementkey" or not _KEY_PATTERN.fullmatch(key):
        return None
    return key


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    router = APIRouter()
    user_dependency = Depends(authenticated_user)

    @router.post(
        "/v1/statement-inbox/keys",
        status_code=201,
        response_model=StatementInboxKeyCreated,
    )
    def create_key(
        request: Request,
        body: StatementInboxKeyCreate,
        user: Any = user_dependency,
    ) -> StatementInboxKeyCreated:
        _require_writer(user)
        plain_key = f"wsk_{secrets.token_urlsafe(32)}"
        key = StatementInboxKey(
            user_id=user.household_owner_id,
            created_by=user.user_id,
            label=body.label,
            key_hash=hashlib.sha256(plain_key.encode("ascii")).hexdigest(),
        )
        user.session.add(key)
        user.session.flush()
        return StatementInboxKeyCreated(
            id=key.id,
            label=key.label,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            key=plain_key,
        )

    @router.get(
        "/v1/statement-inbox/keys",
        response_model=list[StatementInboxKeyResponse],
    )
    def list_keys(user: Any = user_dependency) -> list[StatementInboxKeyResponse]:
        rows = user.session.scalars(
            select(StatementInboxKey)
            .where(
                StatementInboxKey.user_id == user.household_owner_id,
                StatementInboxKey.revoked_at.is_(None),
            )
            .order_by(StatementInboxKey.created_at.desc())
        )
        return [
            StatementInboxKeyResponse(
                id=row.id,
                label=row.label,
                created_at=row.created_at,
                last_used_at=row.last_used_at,
            )
            for row in rows
        ]

    @router.delete("/v1/statement-inbox/keys/{key_id}", status_code=204)
    def revoke_key(key_id: UUID, user: Any = user_dependency) -> None:
        _require_writer(user)
        key = user.session.scalar(
            select(StatementInboxKey).where(
                StatementInboxKey.user_id == user.household_owner_id,
                StatementInboxKey.id == key_id,
                StatementInboxKey.revoked_at.is_(None),
            )
        )
        if key is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="The upload key was not found.")
        key.revoked_at = datetime.now(UTC)
        user.session.flush()

    @router.post(
        "/v1/statement-inbox/uploads",
        status_code=202,
        response_model=StatementInboxUploadResponse,
        responses={200: {"model": StatementInboxUploadResponse}},
        openapi_extra={
            "parameters": [
                {
                    "name": "Authorization",
                    "in": "header",
                    "required": True,
                    "schema": {
                        "type": "string",
                        "pattern": r"^StatementKey wsk_[A-Za-z0-9_-]{43}$",
                    },
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": [
                                "file",
                                "sender",
                                "subject",
                                "received_at",
                                "message_id",
                            ],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "sender": {"type": "string", "maxLength": 320},
                                "subject": {"type": "string", "maxLength": 512},
                                "received_at": {"type": "string", "format": "date-time"},
                                "message_id": {"type": "string", "maxLength": 255},
                            },
                        }
                    }
                },
            },
        },
    )
    async def upload_statement(request: Request) -> Any:
        limiter = getattr(request.app.state, "statement_upload_ip_limiter", None)
        retry_after = limiter.retry_after(client_address(request)) if limiter else None
        if retry_after is not None:
            return _problem(
                request,
                429,
                "rate_limited",
                "Please wait before sending another statement.",
            )
        engine = getattr(request.app.state, "db_engine", None)
        if engine is None:
            return _problem(
                request, 503, "service_unavailable", "Statement uploads are unavailable right now."
            )
        plain_key = _decode_upload_key(request)
        if plain_key is None:
            return _problem(request, 401, "invalid_key", "The upload key is invalid or revoked.")

        key_hash = hashlib.sha256(plain_key.encode("ascii")).hexdigest()
        try:
            with engine.connect() as connection:
                key_row = connection.execute(
                    text(
                        "SELECT key_id, user_id, created_by "
                        "FROM app_lookup_statement_inbox_key(:key_hash)"
                    ),
                    {"key_hash": key_hash},
                ).one_or_none()
        except SQLAlchemyError:
            return _problem(
                request, 503, "service_unavailable", "Statement uploads are unavailable right now."
            )
        if key_row is None:
            return _problem(request, 401, "invalid_key", "The upload key is invalid or revoked.")

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
                f"The PDF exceeds the {max_mb} MB upload limit.",
            )
        except (MultiPartException, ValueError):
            return _problem(request, 422, "invalid_upload", "The statement upload is invalid.")

        try:
            file = form.get("file") if form is not None else None
            sender_value = form.get("sender") if form is not None else None
            subject_value = form.get("subject") if form is not None else None
            received_value = form.get("received_at") if form is not None else None
            message_id_value = form.get("message_id") if form is not None else None
            upload_files = (
                [value for _name, value in form.multi_items() if isinstance(value, UploadFile)]
                if form is not None
                else []
            )
            if not isinstance(file, UploadFile) or len(upload_files) != 1:
                return _problem(request, 422, "invalid_upload", "One statement PDF is required.")
            fields = (sender_value, subject_value, received_value, message_id_value)
            if any(not isinstance(value, str) for value in fields):
                return _problem(
                    request, 422, "invalid_upload", "The statement details are invalid."
                )
            sender = sender_value.strip()
            subject = subject_value.strip()
            message_id = message_id_value.strip()
            received_at = _parse_received_at(received_value.strip())
            if (
                not sender
                or len(sender) > 320
                or len(subject) > 512
                or not message_id
                or len(message_id) > 255
                or received_at is None
                or "\x00" in sender + subject + message_id
            ):
                return _problem(
                    request, 422, "invalid_upload", "The statement details are invalid."
                )
            pdf_data = await file.read()
        finally:
            if form is not None:
                await form.close()

        if not pdf_data.startswith(b"%PDF"):
            return _problem(request, 415, "not_a_pdf", "Choose a PDF statement file.")

        item_id = uuid4()
        file_hash = hashlib.sha256(pdf_data).hexdigest()
        sender_type = _sender_type(sender, subject)
        try:
            with engine.begin() as connection:
                ingest = connection.execute(
                    text(
                        "SELECT status, item_id, user_id, created_by "
                        "FROM app_ingest_statement_inbox_item("
                        ":key_hash, :item_id, :sender_type, :sender, :subject, "
                        ":received_at, :message_id, :file_hash, :pdf_data)"
                    ),
                    {
                        "key_hash": key_hash,
                        "item_id": item_id,
                        "sender_type": sender_type,
                        "sender": sender,
                        "subject": subject,
                        "received_at": received_at,
                        "message_id": message_id,
                        "file_hash": file_hash,
                        "pdf_data": pdf_data,
                    },
                ).one()
        except SQLAlchemyError:
            return _problem(
                request, 503, "service_unavailable", "Statement uploads are unavailable right now."
            )

        if ingest.status == "invalid":
            return _problem(request, 401, "invalid_key", "The upload key is invalid or revoked.")
        if ingest.status == "rate_limited":
            return _problem(
                request,
                429,
                "rate_limited",
                "This upload key has reached today's limit.",
            )
        if ingest.status == "duplicate":
            return JSONResponse(status_code=200, content={"status": "duplicate"})

        password_key = request.app.state.settings.statement_password_key_value
        if password_key is not None:
            creator_id = UUID(str(ingest.created_by))
            household_id = UUID(str(ingest.user_id))
            with user_session(engine, creator_id, household_id) as session:
                item = session.scalar(
                    select(StatementInboxItem).where(
                        StatementInboxItem.user_id == ingest.user_id,
                        StatementInboxItem.id == ingest.item_id,
                    )
                )
                saved = session.scalar(
                    select(StatementSavedPassword).where(
                        StatementSavedPassword.user_id == ingest.user_id,
                        StatementSavedPassword.sender_type == sender_type,
                    )
                )
                remembered = _saved_password(saved, household_id, password_key) if saved else None
                if item is not None and remembered is not None:
                    try:
                        await _process_item(
                            session,
                            item,
                            remembered,
                            max_mb=max_mb,
                            password_key=password_key,
                            remember=False,
                        )
                        saved.last_used_at = datetime.now(UTC)
                    except _StatementProcessingFailure as failure:
                        item.error_code = failure.code
                        if failure.code in {"unreadable", "not_a_full_statement"}:
                            item.status = "unreadable"
        return JSONResponse(status_code=202, content={"status": "received"})

    @router.get("/v1/statement-inbox/items", response_model=StatementInboxListResponse)
    def list_items(user: Any = user_dependency) -> StatementInboxListResponse:
        now = datetime.now(UTC)
        expired = user.session.execute(
            delete(StatementInboxItem).where(
                StatementInboxItem.user_id == user.household_owner_id,
                StatementInboxItem.status.in_(("needs_password", "unreadable")),
                StatementInboxItem.received_at < now - timedelta(days=30),
            )
        )
        deleted_count = int(expired.rowcount or 0)
        rows = user.session.scalars(
            select(StatementInboxItem)
            .where(StatementInboxItem.user_id == user.household_owner_id)
            .order_by(StatementInboxItem.received_at.desc())
        )
        saved_senders = set(
            user.session.scalars(
                select(StatementSavedPassword.sender_type).where(
                    StatementSavedPassword.user_id == user.household_owner_id
                )
            )
        )
        items = [
            StatementInboxItemResponse(
                id=item.id,
                sender_type=item.sender_type,
                received_at=item.received_at,
                status=item.status,
                error_code=item.error_code,
                import_job_id=item.import_job_id,
                subject=item.subject[:_SUBJECT_PREVIEW_CHARS],
                full_statement=item.error_code != "not_a_full_statement"
                and bool(_FULL_STATEMENT_SUBJECT.search(item.subject)),
                saved_password=item.sender_type in saved_senders,
            )
            for item in rows
        ]
        return StatementInboxListResponse(
            items=items,
            cleanup_notice=(
                "Old statements without action were removed." if deleted_count else None
            ),
        )

    @router.post(
        "/v1/statement-inbox/items/{item_id}/process",
        response_model=StatementInboxProcessResponse,
    )
    async def process_item(
        item_id: UUID,
        request: Request,
        body: StatementInboxProcessRequest,
        user: Any = user_dependency,
    ) -> Any:
        _require_writer(user)
        item = user.session.scalar(
            select(StatementInboxItem).where(
                StatementInboxItem.user_id == user.household_owner_id,
                StatementInboxItem.id == item_id,
            )
        )
        if item is None:
            return _problem(request, 404, "not_found", "This statement could not be found.")
        if item.status == "review" and item.import_job_id is not None:
            return StatementInboxProcessResponse(
                status="review", job_id=item.import_job_id, password_saved=False
            )
        if item.status == "unreadable" and item.pdf_data is None:
            return _problem(
                request,
                422,
                "unreadable",
                "We couldn't read this file. Try uploading it in Money.",
            )
        password_key = request.app.state.settings.statement_password_key_value
        password = body.password
        saved: StatementSavedPassword | None = None
        if body.use_saved_password:
            saved = user.session.scalar(
                select(StatementSavedPassword).where(
                    StatementSavedPassword.user_id == user.household_owner_id,
                    StatementSavedPassword.sender_type == item.sender_type,
                )
            )
            password = (
                _saved_password(saved, item.user_id, password_key)
                if saved is not None and password_key is not None
                else None
            )
            if password is None:
                return _problem(
                    request,
                    422,
                    "no_saved_password",
                    "There's no saved password for these statements. Enter it instead.",
                )
        if not password:
            return _problem(request, 422, "password_required", "Enter the statement password.")
        try:
            response = await _process_item(
                user.session,
                item,
                password,
                max_mb=getattr(request.app.state.settings, "max_upload_mb", MAX_UPLOAD_MB),
                password_key=password_key,
                remember=body.remember and not body.use_saved_password,
            )
        except _StatementProcessingFailure as failure:
            return _problem(request, failure.status, failure.code, failure.detail)
        if saved is not None:
            saved.last_used_at = datetime.now(UTC)
        return response

    @router.delete("/v1/statement-inbox/items/{item_id}", status_code=204, response_model=None)
    def remove_item(item_id: UUID, request: Request, user: Any = user_dependency) -> Any:
        _require_writer(user)
        removed = user.session.execute(
            delete(StatementInboxItem).where(
                StatementInboxItem.user_id == user.household_owner_id,
                StatementInboxItem.id == item_id,
            )
        )
        if not removed.rowcount:
            return _problem(request, 404, "not_found", "This statement could not be found.")
        return None

    @router.delete("/v1/statement-inbox/passwords", status_code=204)
    def forget_passwords(user: Any = user_dependency) -> None:
        _require_writer(user)
        user.session.execute(
            delete(StatementSavedPassword).where(
                StatementSavedPassword.user_id == user.household_owner_id
            )
        )

    return router
