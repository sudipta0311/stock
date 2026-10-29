"""Validated environment settings for the public API."""

from __future__ import annotations

import base64
import binascii
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    env: Literal["dev", "test", "production"] = Field("production", validation_alias="ENV")
    deployment_mode: Literal["personal", "public"] = Field(
        default="personal", validation_alias="DEPLOYMENT_MODE"
    )
    database_url: SecretStr | None = Field(
        default=None, validation_alias="DATABASE_URL", repr=False
    )
    auth_mode: Literal["oidc", "dev"] = Field("oidc", validation_alias="AUTH_MODE")
    auth_jwks_url: str | None = Field(default=None, validation_alias="AUTH_JWKS_URL")
    auth_issuer: str | None = Field(default=None, validation_alias="AUTH_ISSUER")
    auth_audience: str | None = Field(default=None, validation_alias="AUTH_AUDIENCE")
    google_client_id: str | None = Field(default=None, validation_alias="GOOGLE_CLIENT_ID")
    google_client_secret: SecretStr | None = Field(
        default=None, validation_alias="GOOGLE_CLIENT_SECRET", repr=False
    )
    allowed_emails: str = Field(default="", validation_alias="ALLOWED_EMAILS")
    dev_auth_secret: SecretStr | None = Field(
        default=None, validation_alias="DEV_AUTH_SECRET", repr=False
    )
    app_session_secret: SecretStr | None = Field(
        default=None, validation_alias="APP_SESSION_SECRET", repr=False
    )
    statement_password_key: SecretStr | None = Field(
        default=None, validation_alias="STATEMENT_PASSWORD_KEY", repr=False
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        validation_alias="ALLOWED_ORIGINS",
    )
    max_upload_mb: int = Field(default=10, gt=0, validation_alias="MAX_UPLOAD_MB")
    email_provider: Literal["brevo", "resend", "dry_run"] = Field(
        default="dry_run", validation_alias="EMAIL_PROVIDER"
    )
    email_from: str = Field(default="alerts@example.invalid", validation_alias="EMAIL_FROM")
    email_api_key: SecretStr | None = Field(
        default=None, validation_alias="EMAIL_API_KEY", repr=False
    )
    email_link_secret: SecretStr | None = Field(
        default=None, validation_alias="EMAIL_LINK_SECRET", repr=False
    )
    email_outbox_dir: str = Field(default="out/outbox", validation_alias="EMAIL_OUTBOX_DIR")
    email_public_base_url: str = Field(
        default="https://app.example", validation_alias="EMAIL_PUBLIC_BASE_URL"
    )
    email_manage_alerts_url: str = Field(
        default="https://app.example/settings/alerts", validation_alias="EMAIL_MANAGE_ALERTS_URL"
    )
    email_postal_address: str = Field(
        default="Postal address configured by sender", validation_alias="EMAIL_POSTAL_ADDRESS"
    )
    email_disclaimer: str = Field(
        default=(
            "Research and education only, not personalised investment advice; "
            "we're not SEBI-registered."
        ),
        validation_alias="EMAIL_DISCLAIMER",
    )
    vapid_public_key: SecretStr | None = Field(
        default=None, validation_alias="VAPID_PUBLIC_KEY", repr=False
    )
    vapid_private_key: SecretStr | None = Field(
        default=None, validation_alias="VAPID_PRIVATE_KEY", repr=False
    )
    vapid_subject: str = Field(
        default="mailto:alerts@example.invalid", validation_alias="VAPID_SUBJECT"
    )

    @model_validator(mode="after")
    def validate_security_settings(self) -> Settings:
        if self.auth_mode == "dev":
            if self.env not in {"dev", "test"}:
                raise ValueError("AUTH_MODE=dev is only allowed when ENV is dev or test")
            secret = self.dev_auth_secret_value
            if secret is None or len(secret) < 32:
                raise ValueError("DEV_AUTH_SECRET must be at least 32 characters in dev mode")
        if self.env == "production" and "*" in self.origin_list:
            raise ValueError("ALLOWED_ORIGINS must not contain '*' in production")
        if self.env == "production" and not self.allowed_email_list:
            raise ValueError("ALLOWED_EMAILS must be set in production")
        if self.auth_mode == "oidc" and not all(
            (self.auth_jwks_url, self.auth_issuer, self.auth_audience)
        ):
            raise ValueError("OIDC mode requires AUTH_JWKS_URL, AUTH_ISSUER, and AUTH_AUDIENCE")
        if self.env == "production" and self.auth_mode == "oidc":
            if not self.google_client_id or not self.google_client_secret_value:
                raise ValueError(
                    "Google sign-in requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET"
                )
            if self.google_client_id != self.auth_audience:
                raise ValueError("AUTH_AUDIENCE must match GOOGLE_CLIENT_ID")
        session_secret = self.app_session_secret_value
        if session_secret is not None and len(session_secret.encode("utf-8")) < 32:
            raise ValueError("APP_SESSION_SECRET must be at least 32 bytes")
        if self.env == "production" and session_secret is None:
            raise ValueError("APP_SESSION_SECRET must be set in production")
        return self

    @property
    def origin_list(self) -> list[str]:
        """Return the comma-separated origin setting as trimmed origins."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def allowed_email_list(self) -> list[str]:
        """Return configured sign-in addresses in case-insensitive comparison form."""
        return [
            email.strip().casefold() for email in self.allowed_emails.split(",") if email.strip()
        ]

    @property
    def database_url_value(self) -> str | None:
        """Return the configured database URL without including it in settings reprs."""
        return self.database_url.get_secret_value() if self.database_url is not None else None

    @property
    def dev_auth_secret_value(self) -> str | None:
        """Return the configured development signing secret."""
        return self.dev_auth_secret.get_secret_value() if self.dev_auth_secret is not None else None

    @property
    def app_session_secret_value(self) -> str | None:
        """Return the app-session signing key without exposing it in settings reprs."""
        return (
            self.app_session_secret.get_secret_value()
            if self.app_session_secret is not None
            else None
        )

    @property
    def statement_password_key_value(self) -> bytes | None:
        """Return a valid 32-byte AES key, or disable saved passwords."""
        if self.statement_password_key is None:
            return None
        try:
            key = base64.b64decode(
                self.statement_password_key.get_secret_value().encode("ascii"), validate=True
            )
        except (binascii.Error, UnicodeEncodeError, ValueError):
            return None
        return key if len(key) == 32 else None

    @property
    def google_client_secret_value(self) -> str | None:
        """Return the Google OAuth secret without exposing it in settings reprs."""
        return (
            self.google_client_secret.get_secret_value()
            if self.google_client_secret is not None
            else None
        )

    @property
    def email_api_key_value(self) -> str | None:
        """Return the configured email provider key."""
        return self.email_api_key.get_secret_value() if self.email_api_key is not None else None

    @property
    def email_link_secret_value(self) -> str | None:
        """Return the configured consent-link signing secret."""
        return (
            self.email_link_secret.get_secret_value()
            if self.email_link_secret is not None
            else None
        )

    @property
    def vapid_public_key_value(self) -> str | None:
        """Return the configured VAPID public key."""
        return (
            self.vapid_public_key.get_secret_value() if self.vapid_public_key is not None else None
        )

    @property
    def vapid_private_key_value(self) -> str | None:
        """Return the configured VAPID private key."""
        return (
            self.vapid_private_key.get_secret_value()
            if self.vapid_private_key is not None
            else None
        )
