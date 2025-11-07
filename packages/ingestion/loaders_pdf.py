# packages/ingestion/loaders_pdf.py
"""
PDF loader — small, explicit, with sensible fallbacks.

What this module provides
-------------------------
- load_pdf_text(path: str) -> str
    Extracts readable text from a PDF using:
      1) pypdf (preferred; fast, pure-Python)
      2) PyPDF2 (older package name; compatible)
      3) pdfminer.six (slower, but better at some layouts)
    Returns a **single cleaned string** (page breaks as "\n\n").

Design goals
------------
- Keep the code tiny and tutorial-clear.
- Avoid over-engineering: tables/figures are handled elsewhere.
- Be resilient: try multiple libraries; never crash on minor parsing issues.
- Normalize whitespace and fix common OCR issues (soft hyphens, weird spaces).

Dependencies (optional, best-effort)
------------------------------------
- pypdf          (`pip install pypdf`)
- PyPDF2         (legacy name; many envs still have it)
- pdfminer.six   (`pip install pdfminer.six`)

If none are available, we raise a RuntimeError with a clear message.

Usage
-----
from packages.ingestion.loaders_pdf import load_pdf_text
text = load_pdf_text("data/samples/10k_2023.pdf")
print(text[:500])
"""

from __future__ import annotations

import os
from typing import Callable, List


# ---------------------------
# Utilities
# ---------------------------

def _clean(s: str) -> str:
    """
    Normalize common artifacts:
    - soft hyphens (U+00AD) → removed
    - NBSP (U+00A0) → space
    - collapse 3+ newlines to 2
    - collapse long runs of spaces
    """
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\u00a0", " ")  # nbsp
    # Normalize newlines
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    # Trim trailing spaces per line
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


# ---------------------------
# Backends
# ---------------------------

def _backend_pypdf(path: str) -> str:
    from pypdf import PdfReader  # type: ignore
    reader = PdfReader(path)
    chunks: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


def _backend_pypdf2(path: str) -> str:
    from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(path)
    chunks: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


def _backend_pdfminer(path: str) -> str:
    # pdfminer.six: richer layout parsing, slower; but a good fallback
    from pdfminer.high_level import extract_text  # type: ignore
    txt = extract_text(path) or ""
    return txt


# ---------------------------
# Public API
# ---------------------------

def load_pdf_text(path: str) -> str:
    """
    Extract text from a PDF with layered fallbacks.

    Raises:
        FileNotFoundError if path does not exist.
        RuntimeError if no PDF backends are available.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try backends in order
    errors: List[str] = []

    for name, loader in (
        ("pypdf", _backend_pypdf),
        ("PyPDF2", _backend_pypdf2),
        ("pdfminer.six", _backend_pdfminer),
    ):
        try:
            # Check import availability lazily
            if name == "pypdf":
                import importlib
                if importlib.util.find_spec("pypdf") is None:
                    raise ImportError("pypdf not installed")
            elif name == "PyPDF2":
                import importlib
                if importlib.util.find_spec("PyPDF2") is None:
                    raise ImportError("PyPDF2 not installed")
            else:  # pdfminer.six
                import importlib
                if importlib.util.find_spec("pdfminer") is None:
                    raise ImportError("pdfminer.six not installed")

            text = loader(path)
            return _clean(text)

        except Exception as e:
            errors.append(f"{name}: {e}")

    # If we get here, no backend succeeded
    msg = "PDF parsing failed — install one of: pypdf, PyPDF2, or pdfminer.six. Errors: " + "; ".join(errors)
    raise RuntimeError(msg)
