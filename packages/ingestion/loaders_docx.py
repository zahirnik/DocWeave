# packages/ingestion/loaders_docx.py
"""
DOCX loader — tiny, resilient, with a pure-stdlib fallback.

What this module provides
-------------------------
- load_docx_text(path: str) -> str
    Extracts readable text from a .docx file using:
      1) python-docx (preferred; preserves paragraph breaks & simple tables)
      2) stdlib zipfile + xml.etree (fallback when python-docx isn't installed)

Design goals
------------
- Keep it **tutorial-clear** and dependency-light.
- Return a single normalized string suitable for chunking/embedding.
- Handle simple tables by concatenating cell texts separated by tabs.
- Normalize whitespace and fix common artifacts (NBSP, soft hyphens).

Dependencies (optional)
-----------------------
- python-docx   (`pip install python-docx`)
  If missing, we use a lightweight XML fallback that reads document.xml.

Usage
-----
from packages.ingestion.loaders_docx import load_docx_text
text = load_docx_text("data/samples/report.docx")
print(text[:500])
"""

from __future__ import annotations

import os
import re
import zipfile
from typing import List, Optional


# ---------------------------
# Normalization helpers
# ---------------------------

def _clean(s: str) -> str:
    """
    Normalize common artifacts:
    - soft hyphens (U+00AD) → removed
    - NBSP (U+00A0) → space
    - collapse 3+ newlines to 2
    - strip trailing spaces per line
    """
    s = s.replace("\u00ad", "")   # soft hyphen
    s = s.replace("\u00a0", " ")  # nbsp
    # Collapse >2 blank lines
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    # Strip trailing spaces
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


# ---------------------------
# Backend 1: python-docx
# ---------------------------

def _backend_python_docx(path: str) -> str:
    """
    Use python-docx to read paragraphs and simple tables.
    """
    import docx  # type: ignore

    doc = docx.Document(path)
    parts: List[str] = []

    # Paragraphs
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if txt:
            parts.append(txt)

    # Tables (very common in reports/filings)
    for t in getattr(doc, "tables", []):
        for row in t.rows:
            cells = []
            for cell in row.cells:
                # cell.text already flattens inner paragraphs
                c = (cell.text or "").strip()
                cells.append(c)
            if any(cells):
                parts.append("\t".join(cells))

    # Join with double newlines between blocks
    return _clean("\n\n".join(parts))


# ---------------------------
# Backend 2: stdlib XML fallback
# ---------------------------

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _xml_text(el) -> str:
    """
    Extract concatenated text from an XML element (w:t nodes).
    """
    from xml.etree import ElementTree as ET  # stdlib
    texts: List[str] = []
    for node in el.iter():
        if node.tag.endswith("}t") or node.tag == f"{_W_NS}t":  # w:t
            if node.text:
                texts.append(node.text)
    return "".join(texts)


def _xml_run_to_text(run) -> str:
    """
    Extract text from a w:r (run). Honors soft line breaks (w:br) as '\n'.
    """
    from xml.etree import ElementTree as ET  # stdlib
    buf: List[str] = []
    for node in list(run):
        tag = node.tag
        if tag.endswith("}t") or tag == f"{_W_NS}t":
            buf.append(node.text or "")
        elif tag.endswith("}br") or tag == f"{_W_NS}br":
            buf.append("\n")
    return "".join(buf)


def _backend_xml(path: str) -> str:
    """
    Pure-stdlib fallback that parses document.xml to collect paragraphs and table cells.
    """
    from xml.etree import ElementTree as ET  # stdlib

    parts: List[str] = []
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()

    # Paragraphs: w:p → runs w:r → text w:t
    for p in root.iter(f"{_W_NS}p"):
        runs = [r for r in p.iter(f"{_W_NS}r")]
        if runs:
            txt = "".join(_xml_run_to_text(r) for r in runs).strip()
        else:
            # as a fallback, extract all w:t directly under p
            txt = _xml_text(p).strip()
        if txt:
            parts.append(txt)

    # Tables: w:tbl → w:tr → w:tc → w:p/w:r/w:t  (join cells by tabs)
    for tbl in root.iter(f"{_W_NS}tbl"):
        for tr in tbl.iter(f"{_W_NS}tr"):
            row_cells: List[str] = []
            for tc in tr.iter(f"{_W_NS}tc"):
                # Get text inside the cell
                cell_txt_parts: List[str] = []
                for p in tc.iter(f"{_W_NS}p"):
                    runs = [r for r in p.iter(f"{_W_NS}r")]
                    if runs:
                        cell_txt_parts.append("".join(_xml_run_to_text(r) for r in runs))
                    else:
                        cell_txt_parts.append(_xml_text(p))
                cell_txt = "\n".join(t.strip() for t in cell_txt_parts if t and t.strip())
                row_cells.append(cell_txt)
            if any(c.strip() for c in row_cells):
                parts.append("\t".join(c.strip() for c in row_cells))

    return _clean("\n\n".join(parts))


# ---------------------------
# Public API
# ---------------------------

def load_docx_text(path: str) -> str:
    """
    Extract text from a .docx file with layered fallbacks.

    Raises:
        FileNotFoundError if path does not exist.
        RuntimeError if the file is not a valid .docx (zip) archive or unreadable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try python-docx first (if installed)
    try:
        import importlib
        if importlib.util.find_spec("docx") is not None:
            return _backend_python_docx(path)
    except Exception:
        # Fall through to XML fallback
        pass

    # Fallback: parse document.xml from the zip (works for most simple docs)
    try:
        return _backend_xml(path)
    except KeyError as e:
        # document.xml missing or invalid docx
        raise RuntimeError(f"Invalid .docx (missing {e})") from e
    except zipfile.BadZipFile as e:
        raise RuntimeError("Not a valid .docx (zip) file") from e
    except Exception as e:
        raise RuntimeError(f"DOCX parsing failed: {e}") from e
