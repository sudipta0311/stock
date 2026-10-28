"""Build small encrypted PDFs for parser tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader, PdfWriter


def build_encrypted_pdf(lines: list[str], password: str) -> bytes:
    """Render fixture lines, encrypt the PDF, and return it in memory."""
    pdf = FPDF()
    pdf.set_margins(left=8, top=8, right=8)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    for line in lines:
        pdf.multi_cell(w=194, h=5, text=line)

    rendered = pdf.output()
    reader = PdfReader(BytesIO(bytes(rendered)))
    writer = PdfWriter()
    writer.append(reader)
    writer.encrypt(user_password=password)
    encrypted = BytesIO()
    writer.write(encrypted)
    return encrypted.getvalue()


def write_encrypted_pdf(lines: list[str], password: str, path: Path) -> Path:
    """Write a generated PDF to a caller-owned temporary path."""
    path.write_bytes(build_encrypted_pdf(lines, password))
    return path
