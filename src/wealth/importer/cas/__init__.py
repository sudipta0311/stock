"""In-memory CAS PDF extraction and issuer-specific pure text parsers."""

from __future__ import annotations

import io
import os
from typing import Literal

from wealth.importer.cas.cams_kfin import parse_cams_kfin
from wealth.importer.cas.cdsl import parse_cdsl
from wealth.importer.cas.common import ParseError, clean_lines
from wealth.importer.cas.nsdl import parse_nsdl
from wealth.importer.models import ParsedStatement

CasType = Literal["nsdl", "cdsl", "cams_kfin", "unknown"]
_CDSL_BETA_DETAIL = (
    "CDSL CAS support is coming soon. Upload your NSDL CAS or broker statement instead."
)


def pdf_text(data: bytes, password: str | None) -> list[str]:
    """Decrypt and extract ordered text lines without retaining the PDF."""
    if not isinstance(data, bytes):
        raise ParseError("not_a_pdf")

    from pypdf import PdfReader

    reader: PdfReader | None = None
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt(password or "")
            except Exception:
                raise ParseError("wrong_password") from None
            if not decrypted:
                raise ParseError("wrong_password")

        extracted: list[str] = []
        for page in reader.pages:
            extracted.extend((page.extract_text() or "").splitlines())
        return clean_lines(extracted)
    except ParseError:
        raise
    except Exception:
        raise ParseError("not_a_pdf") from None


def detect_cas_type(lines: list[str]) -> CasType:
    """Identify a CAS issuer from its header markers."""
    header = "\n".join(line.upper() for line in lines[:40])
    if (
        "CENTRAL DEPOSITORY SERVICES (INDIA) LIMITED" in header
        or "CDSL CONSOLIDATED ACCOUNT STATEMENT" in header
        or "CDSL CAS" in header
    ):
        return "cdsl"
    if (
        "NSDL CONSOLIDATED ACCOUNT STATEMENT" in header
        or "ABOUT NSDL" in header
        or ("NSDL" in header and "DEMAT ACCOUNT" in header)
    ):
        return "nsdl"
    if (
        "CAMSCASWS" in header
        or "KFINCASWS" in header
        or "CAMS CONSOLIDATED ACCOUNT STATEMENT" in header
        or "KFINTECH CONSOLIDATED ACCOUNT STATEMENT" in header
        or ("CAMS" in header and "CONSOLIDATED ACCOUNT STATEMENT" in header)
        or ("KFINTECH" in header and "CONSOLIDATED ACCOUNT STATEMENT" in header)
    ):
        return "cams_kfin"
    return "unknown"


def parse_cas(data: bytes, password: str | None, *, max_mb: int = 10) -> ParsedStatement:
    """Extract and parse a password-protected CAS entirely in memory."""
    if max_mb <= 0:
        raise ValueError("max_mb must be positive")
    if not isinstance(data, bytes):
        raise ParseError("not_a_pdf")
    if len(data) > max_mb * 1024 * 1024:
        raise ParseError("too_large")

    lines = pdf_text(data, password)
    cas_type = detect_cas_type(lines)
    if cas_type == "unknown":
        raise ParseError("unsupported_format")
    if cas_type == "cdsl":
        if not _cdsl_beta_enabled():
            raise ParseError("unsupported_format", detail=_CDSL_BETA_DETAIL)
        parsed = parse_cdsl(lines)
        if "cdsl_beta_verify_values" not in parsed.warnings:
            parsed.warnings.append("cdsl_beta_verify_values")
        return parsed
    parser = {
        "nsdl": parse_nsdl,
        "cams_kfin": parse_cams_kfin,
    }[cas_type]
    return parser(lines)


def _cdsl_beta_enabled() -> bool:
    value = os.getenv("CDSL_PARSER_BETA", "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


__all__ = [
    "CasType",
    "ParseError",
    "detect_cas_type",
    "parse_cams_kfin",
    "parse_cas",
    "parse_cdsl",
    "parse_nsdl",
    "pdf_text",
]
