"""FastAPI service for authenticated, tenant-scoped wealth data."""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from wealth.api.money_map import MemberResponse
from wealth.schemas.common import RatePct, Rupees
from wealth.schemas.profile import FinancialContext, Residency, TaxRegime, residency_from_stored

if TYPE_CHECKING:
    from wealth.api.settings import Settings

_bearer_scheme = HTTPBearer(auto_error=False)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


EmploymentType = Literal["salaried", "self_employed", "business", "retired", "other"]
IncomeStability = Literal["high", "medium", "low"]
CityTier = Literal[1, 2, 3]


class FinancialContextFields(ApiModel):
    monthly_expenses: Rupees | None = None
    annual_income: Rupees | None = None
    dependants: int | None = Field(default=None, ge=0)
    employment_type: EmploymentType | None = None
    income_stability: IncomeStability | None = None
    city_tier: CityTier | None = None


class ProfileUpsert(ApiModel):
    tax_regime: TaxRegime
    tax_slab_pct: RatePct | None = None
    city_tier: CityTier | None = None
    timezone: str | None = Field(default=None, min_length=1, max_length=64)
    language: str | None = Field(default=None, min_length=1, max_length=16)
    monthly_expenses: Rupees | None = None
    annual_income: Rupees | None = None
    dependants: int | None = Field(default=None, ge=0)
    employment_type: EmploymentType | None = None
    income_stability: IncomeStability | None = None


class ProfileResponse(ApiModel):
    tax_regime: TaxRegime | None
    tax_slab_pct: RatePct | None
    city_tier: CityTier | None
    timezone: str
    language: str
    financial_context: FinancialContextFields | None
    risk_answers: dict[str, str] | None
    risk_profile: str | None
    residency: Residency


class MeResponse(ApiModel):
    id: UUID
    email: str
    display_name: str | None
    created_at: datetime
    profile: ProfileResponse | None
    residency: Residency
    checklist: list[str]
    household_owner_id: UUID
    role: Literal["owner", "partner", "viewer"]
    me_member_id: UUID
    members: list[MemberResponse] = Field(default_factory=list)


class RiskAnswersRequest(ApiModel):
    answers: dict[str, str]
    emergency_ok: bool = True


class RiskResultResponse(ApiModel):
    tolerance: Decimal
    capacity: Decimal
    final: str
    limited_by: Literal["tolerance", "capacity"] | None


class RiskOptionResponse(ApiModel):
    id: str
    text: str


class RiskQuestionResponse(ApiModel):
    id: str
    text: str
    options: list[RiskOptionResponse]


class HealthResponse(ApiModel):
    status: Literal["ok"]
    version: str
    db: Literal["ok", "error"] | None = None


@dataclass
class AuthenticatedUser:
    user_id: UUID
    session: Any
    household_owner_id: UUID | None = None
    role: Literal["owner", "partner", "viewer"] = "owner"
    deployment_mode: Literal["personal", "public"] = "personal"
    session_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.household_owner_id is None:
            self.household_owner_id = self.user_id


def _application_version() -> str:
    try:
        return version("wealth-copilot")
    except PackageNotFoundError:
        return "0.1.0"


def _origin_list_from_environment() -> list[str]:
    configured_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


def _profile_response(profile: Any) -> ProfileResponse:
    financial_context = (
        FinancialContextFields.model_validate(profile.financial_context)
        if profile.financial_context is not None
        else None
    )
    return ProfileResponse(
        tax_regime=profile.tax_regime,
        tax_slab_pct=profile.tax_slab_pct,
        city_tier=profile.city_tier,
        timezone=profile.timezone,
        language=profile.language,
        financial_context=financial_context,
        risk_answers=profile.risk_answers,
        risk_profile=profile.risk_profile,
        residency=residency_from_stored(profile.residency),
    )


def _financial_context_for_risk(profile: Any | None) -> FinancialContext:
    stored = dict(profile.financial_context or {}) if profile is not None else {}
    if stored.get("monthly_expenses") is None:
        stored["monthly_expenses"] = "0"
    if stored.get("dependants") is None:
        stored["dependants"] = 0
    if stored.get("employment_type") is None:
        stored["employment_type"] = "salaried"
    if stored.get("income_stability") is None:
        stored["income_stability"] = "medium"
    if stored.get("city_tier") is None and profile is not None:
        stored["city_tier"] = profile.city_tier
    return FinancialContext.model_validate(stored)


def _problem(
    request: Request,
    status: int,
    title: str,
    detail: str,
    headers: dict[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
) -> JSONResponse:
    content = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
    }
    if extensions:
        content.update(extensions)
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        headers=headers,
        content=content,
    )


def _validation_issues(error: RequestValidationError) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for item in error.errors():
        location = list(item.get("loc", ()))
        if location and location[0] == "body":
            location.pop(0)
        field = ".".join(str(part) for part in location) or "request"
        error_type = item.get("type")
        if error_type == "enum":
            expected = item.get("ctx", {}).get("expected")
            message = (
                f"Input should be {expected}" if expected else "Input must be an allowed option."
            )
        elif error_type == "missing":
            message = "Field required"
        elif error_type == "extra_forbidden":
            message = "Extra inputs are not permitted"
        elif field in {"monthly_expenses", "annual_income"}:
            message = "Input should be a valid INR amount"
        else:
            message = "Invalid value."
        issues.append({"field": field, "message": message})
    return issues


def create_app(settings: "Settings | None" = None) -> FastAPI:
    """Build the API and its request-scoped dependencies from validated settings."""
    configured = settings

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if configured is None:
            from wealth.api.settings import Settings

            runtime_settings = Settings()
        else:
            runtime_settings = configured
        from wealth.db.session import api_engine

        application.state.settings = runtime_settings
        database_url = runtime_settings.database_url_value
        engine = api_engine(database_url) if database_url else None
        application.state.db_engine = engine
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    application = FastAPI(
        title="Wealth Copilot API",
        version=_application_version(),
        lifespan=lifespan,
    )
    application.state.settings = configured
    application.state.jwks_cache = None
    application.state.db_engine = None
    from wealth.api.auth_rate_limit import AuthRateLimiter

    auth_limiter = AuthRateLimiter()
    application.state.auth_rate_limiter = auth_limiter
    application.state.google_exchange_limiter = auth_limiter
    application.add_middleware(
        CORSMiddleware,
        allow_origins=(
            configured.origin_list if configured is not None else _origin_list_from_environment()
        ),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Household"],
    )

    @application.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        titles = {
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            409: "Conflict",
            422: "Unprocessable Content",
            503: "Service Unavailable",
        }
        is_read_only = exc.status_code == 403 and exc.detail == "read_only"
        is_invalid_invite = exc.status_code == 400 and exc.detail == "invalid_invite"
        conflict_details = {
            "member_is_owner": "The owner’s own member cannot be assigned to an invite.",
            "member_already_linked": "This member already has an active login or invite.",
            "member_linked_to_login": "Revoke this member’s login access before deleting them.",
        }
        conflict_code = (
            exc.detail
            if exc.status_code == 409
            and isinstance(exc.detail, str)
            and exc.detail in conflict_details
            else None
        )
        detail = (
            "This household is read-only for your account."
            if is_read_only
            else conflict_details[conflict_code]
            if conflict_code is not None
            else exc.detail
            if isinstance(exc.detail, str)
            else "The request could not be completed."
        )
        extensions = (
            {"code": "read_only"}
            if is_read_only
            else {"code": "invalid_invite"}
            if is_invalid_invite
            else {"code": conflict_code}
            if conflict_code is not None
            else None
        )
        if request.url.path.startswith("/v1/imports"):
            extensions = {
                "code": "invalid_invite"
                if is_invalid_invite
                else "read_only"
                if is_read_only
                else conflict_code
                if conflict_code is not None
                else {
                    401: "unauthorized",
                    403: "forbidden",
                    404: "not_found",
                    409: "conflict",
                    422: "invalid_request",
                    503: "service_unavailable",
                }.get(exc.status_code, "request_failed")
            }
        return _problem(
            request,
            exc.status_code,
            titles.get(exc.status_code, "Request Failed"),
            detail,
            exc.headers,
            extensions=extensions,
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _problem(
            request,
            422,
            "Unprocessable Content",
            "The request did not match the expected format.",
            extensions={
                "errors": _validation_issues(_exc),
                **(
                    {"code": "invalid_request"}
                    if request.url.path.startswith("/v1/imports")
                    else {}
                ),
            },
        )

    @application.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
        return _problem(
            request,
            500,
            "Internal Server Error",
            "An unexpected error occurred.",
            extensions=(
                {"code": "internal_error"} if request.url.path.startswith("/v1/imports") else None
            ),
        )

    def authenticated_user(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
        household_header: Annotated[str | None, Header(alias="X-Household")] = None,
    ) -> Any:
        from wealth.api.auth import AuthenticationError, JwksCache, verify_token

        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if request.app.state.jwks_cache is None:
            request.app.state.jwks_cache = JwksCache()
        try:
            claims = verify_token(
                credentials.credentials,
                request.app.state.settings,
                request.app.state.jwks_cache,
            )
        except AuthenticationError as error:
            raise HTTPException(
                status_code=401,
                detail="The bearer token is invalid or expired.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

        allowed_emails = request.app.state.settings.allowed_email_list
        if allowed_emails and claims.email.casefold() not in allowed_emails:
            raise HTTPException(status_code=403, detail="This is a private family app.")

        engine = request.app.state.db_engine
        if engine is None:
            raise HTTPException(status_code=503, detail="The data service is unavailable.")
        from sqlalchemy import select, text
        from sqlalchemy.exc import SQLAlchemyError

        from wealth.db.models import AuthSession
        from wealth.db.session import resolve_user, user_session

        try:
            user_id = resolve_user(engine, claims.subject, claims.email, claims.name)
            if household_header is None:
                household_owner_id = user_id
            else:
                try:
                    household_owner_id = UUID(household_header)
                except ValueError as error:
                    raise HTTPException(
                        status_code=404, detail="The household was not found."
                    ) from error
            with user_session(engine, user_id, household_owner_id) as session:
                if claims.session_id is not None:
                    auth_session = session.scalar(
                        select(AuthSession).where(
                            AuthSession.id == claims.session_id,
                            AuthSession.user_id == user_id,
                        )
                    )
                    now = datetime.now(UTC)
                    if (
                        auth_session is None
                        or auth_session.revoked_at is not None
                        or auth_session.expires_at <= now
                        or auth_session.absolute_expires_at <= now
                    ):
                        raise HTTPException(
                            status_code=401,
                            detail="The sign-in session has expired.",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    if auth_session.last_used_at < now - timedelta(minutes=5):
                        auth_session.last_used_at = now
                role = session.scalar(
                    text("SELECT app_household_role(:owner_id)"),
                    {"owner_id": household_owner_id},
                )
                if role is None:
                    raise HTTPException(status_code=404, detail="The household was not found.")
                request.state.household_owner_id = household_owner_id
                request.state.household_role = role
                is_leave_request = (
                    request.method == "DELETE" and request.url.path == "/v1/households/leave"
                )
                if (
                    request.method not in {"GET", "HEAD", "OPTIONS"}
                    and role == "viewer"
                    and not is_leave_request
                ):
                    raise HTTPException(status_code=403, detail="read_only")
                yield AuthenticatedUser(
                    user_id=user_id,
                    session=session,
                    household_owner_id=household_owner_id,
                    role=role,
                    deployment_mode=request.app.state.settings.deployment_mode,
                    session_id=claims.session_id,
                )
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="The data service is unavailable."
            ) from error

    @application.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request, deep: Annotated[bool, Query()] = False) -> Any:
        from sqlalchemy import text

        result = {"status": "ok", "version": application.version}
        if not deep:
            return result
        engine = application.state.db_engine
        try:
            if engine is None:
                raise RuntimeError("Database is not configured")
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            return _problem(
                request,
                503,
                "Service Unavailable",
                "The data service is unavailable.",
                extensions={"db": "error", "version": application.version},
            )
        return {**result, "db": "ok"}

    @application.get("/v1/me", response_model=MeResponse)
    def get_me(user: Annotated[AuthenticatedUser, Depends(authenticated_user)]) -> MeResponse:
        from sqlalchemy import select

        from wealth.db.models import HouseholdMember, Profile, User
        from wealth.db.repo import acting_household_member

        member = user.session.get(User, user.user_id)
        if member is None:
            raise HTTPException(status_code=404, detail="The user record was not found.")
        profile = user.session.scalar(
            select(Profile).where(Profile.user_id == user.household_owner_id)
        )
        acting_member = acting_household_member(
            user.session,
            user.user_id,
            user.household_owner_id,
            display_name=member.display_name or "You",
        )
        members = list(
            user.session.scalars(
                select(HouseholdMember)
                .where(HouseholdMember.user_id == user.household_owner_id)
                .order_by(
                    HouseholdMember.is_self.desc(),
                    HouseholdMember.created_at,
                    HouseholdMember.id,
                )
            )
        )
        return MeResponse(
            id=member.id,
            email=member.email,
            display_name=member.display_name,
            created_at=member.created_at,
            profile=_profile_response(profile) if profile is not None else None,
            residency=residency_from_stored(acting_member.residency),
            checklist=list(profile.checklist) if profile is not None else [],
            household_owner_id=user.household_owner_id,
            role=user.role,
            me_member_id=acting_member.id,
            members=members,
        )

    @application.put("/v1/me/profile", response_model=ProfileResponse)
    def put_profile(
        payload: ProfileUpsert,
        user: Annotated[AuthenticatedUser, Depends(authenticated_user)],
    ) -> ProfileResponse:
        from sqlalchemy import select, text
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from wealth.db.models import Profile

        serialized = payload.model_dump(mode="json")
        financial_context = {
            key: serialized[key]
            for key in (
                "monthly_expenses",
                "annual_income",
                "dependants",
                "employment_type",
                "income_stability",
            )
        }
        financial_context["city_tier"] = serialized["city_tier"]
        values = {
            "id": uuid4(),
            "user_id": user.household_owner_id,
            "tax_regime": payload.tax_regime.value,
            "tax_slab_pct": payload.tax_slab_pct,
            "city_tier": payload.city_tier,
            "timezone": payload.timezone or "Asia/Kolkata",
            "language": payload.language or "en",
            "financial_context": financial_context,
        }
        statement = pg_insert(Profile).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[Profile.user_id],
            set_={
                "tax_regime": statement.excluded.tax_regime,
                "tax_slab_pct": statement.excluded.tax_slab_pct,
                "city_tier": statement.excluded.city_tier,
                "timezone": statement.excluded.timezone,
                "language": statement.excluded.language,
                "financial_context": statement.excluded.financial_context,
                "updated_at": text("now()"),
            },
        )
        user.session.execute(statement)
        profile = user.session.scalar(
            select(Profile).where(Profile.user_id == user.household_owner_id)
        )
        if profile is None:
            raise HTTPException(status_code=500, detail="The profile could not be stored.")
        from wealth.db.repo import acting_household_member

        acting_member = acting_household_member(user.session, user.user_id, user.household_owner_id)
        tax = dict(acting_member.tax or {})
        tax["tax_regime"] = payload.tax_regime.value
        tax["tax_slab_pct"] = (
            format(payload.tax_slab_pct, "f") if payload.tax_slab_pct is not None else None
        )
        for key in ("annual_income", "employment_type"):
            value = serialized.get(key)
            if value is not None:
                tax[key] = value
        if payload.annual_income is not None:
            tax.setdefault("annual_income_currency", "INR")
        if profile.investing and profile.investing.get("birth_year") is not None:
            tax["birth_year"] = profile.investing["birth_year"]
        acting_member.tax = {key: value for key, value in tax.items() if value is not None}
        return _profile_response(profile)

    @application.put("/v1/me/risk-answers", response_model=RiskResultResponse)
    def put_risk_answers(
        payload: RiskAnswersRequest,
        user: Annotated[AuthenticatedUser, Depends(authenticated_user)],
    ) -> RiskResultResponse:
        from sqlalchemy import select, text
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from wealth.db.models import Profile
        from wealth.policy.risk import risk_profile

        profile = user.session.scalar(
            select(Profile).where(Profile.user_id == user.household_owner_id)
        )
        try:
            result = risk_profile(
                _financial_context_for_risk(profile),
                horizon_years=10,
                emergency_ok=payload.emergency_ok,
                answers=payload.answers,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail="Answers must include a valid option for every risk question.",
            ) from error

        statement = pg_insert(Profile).values(
            id=uuid4(),
            user_id=user.household_owner_id,
            risk_answers=payload.answers,
            risk_profile=result.final,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Profile.user_id],
            set_={
                "risk_answers": statement.excluded.risk_answers,
                "risk_profile": statement.excluded.risk_profile,
                "updated_at": text("now()"),
            },
        )
        user.session.execute(statement)
        return RiskResultResponse(
            tolerance=result.tolerance,
            capacity=result.capacity,
            final=result.final,
            limited_by=result.limited_by,
        )

    @application.get("/v1/meta/risk-questions", response_model=list[RiskQuestionResponse])
    def get_risk_questions(
        _user: Annotated[AuthenticatedUser, Depends(authenticated_user)],
    ) -> list[dict[str, Any]]:
        from wealth.policy.risk import risk_questions

        return [
            {
                "id": question["id"],
                "text": question["text"],
                "options": [
                    {"id": option["id"], "text": option["text"]} for option in question["options"]
                ],
            }
            for question in risk_questions()
        ]

    @application.delete("/v1/me", status_code=204, response_class=Response)
    def delete_me(user: Annotated[AuthenticatedUser, Depends(authenticated_user)]) -> Response:
        from sqlalchemy import text

        user.session.execute(
            text("DELETE FROM auth_sessions WHERE user_id = :me"),
            {"me": user.user_id},
        )
        user.session.execute(
            text(
                "DELETE FROM highlight_dismissals "
                "WHERE login_user_id = :me AND household_owner_id = :owner"
            ),
            {"me": user.user_id, "owner": user.household_owner_id},
        )
        user.session.execute(
            text("DELETE FROM users WHERE id = :me"),
            {"me": user.user_id},
        )
        return Response(status_code=204)

    from wealth.api.money_map import build_router

    application.include_router(build_router(authenticated_user))
    from wealth.api.loans import build_router as build_loans_router

    application.include_router(build_loans_router(authenticated_user))
    from wealth.api.imports import build_router as build_imports_router

    application.include_router(build_imports_router(authenticated_user))
    from wealth.api.personalisation import build_router as build_personalisation_router

    application.include_router(build_personalisation_router(authenticated_user))
    from wealth.api.fund_check import build_router as build_fund_check_router

    application.include_router(build_fund_check_router(authenticated_user))
    from wealth.api.households import build_router as build_households_router

    application.include_router(build_households_router(authenticated_user))
    from wealth.api.alerts import build_router as build_alerts_router

    application.include_router(build_alerts_router(authenticated_user))
    from wealth.api.google_auth import build_router as build_google_auth_router

    application.include_router(build_google_auth_router())
    from wealth.api.auth_sessions import build_router as build_auth_sessions_router

    application.include_router(build_auth_sessions_router(authenticated_user))

    return application


app = create_app()
