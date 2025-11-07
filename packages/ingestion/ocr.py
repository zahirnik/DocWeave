# packages/ingestion/ocr.py
"""
OCR utilities — optional, safe-by-default, tutorial-clear.

What this module provides
-------------------------
- ocr_image_bytes(data: bytes, lang: str = "eng", *, raise_on_error: bool = False) -> str
    Best-effort OCR for a single image (PNG/JPEG/TIFF). Returns a cleaned text string.
    If Tesseract/PIL are not installed or OCR fails, returns "" (unless `raise_on_error=True`).

- is_ocr_available() -> bool
    Quick probe to tell if local OCR is likely to work.

Design goals
------------
- Keep OCR *optional*: when not available, ingestion still proceeds (images just yield no text).
- Prefer Tesseract via `pytesseract` (local, free). You can swap to a cloud OCR later.
- Be explicit and tiny; no background services required.

Environment knobs
-----------------
OCR_PROVIDER=tesseract|none         (default: tesseract if libs present, else none)
OCR_LANG=eng                         (Tesseract language pack code, e.g., "eng", "eng+deu")
TESSERACT_CMD=/usr/bin/tesseract     (optional explicit path)
TESSDATA_PREFIX=...                  (optional language data directory)

Install
-------
pip install pillow pytesseract
# and ensure the OS package 'tesseract-ocr' is installed with the language packs you need.

Usage
-----
from packages.ingestion.ocr import ocr_image_bytes, is_ocr_available

if is_ocr_available():
    text = ocr_image_bytes(open("scan.jpg","rb").read(), lang="eng")
"""

from __future__ import annotations

import io
import os
from typing import Optional


def _provider() -> str:
    env = (os.getenv("OCR_PROVIDER") or "").strip().lower()
    if env in {"tesseract", "none"}:
        return env
    # Auto-detect: tesseract if modules are available, else none
    try:
        import PIL  # noqa: F401
        import pytesseract  # noqa: F401
        return "tesseract"
    except Exception:
        return "none"


def is_ocr_available() -> bool:
    """Return True if local OCR is likely to work (PIL + pytesseract importable)."""
    if _provider() != "tesseract":
        return False
    try:
        import PIL  # noqa: F401
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False


def _ocr_tesseract_image(data: bytes, lang: str) -> str:
    """
    Run Tesseract OCR via pytesseract on a single image.
    - Uses page segmentation mode 6 (assumes a block of text).
    - Respects TESSERACT_CMD/TESSDATA_PREFIX if set in env.
    """
    # Optional: allow callers to set an explicit binary path
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = cmd  # type: ignore[attr-defined]
        except Exception:
            pass

    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore

    # Open image from bytes (Pillow auto-detects format)
    img = Image.open(io.BytesIO(data))

    # For scanned pages, higher DPI improves accuracy; Pillow images have no DPI by default.
    # Tesseract ignores DPI from the PIL object here; we can upscale modestly if desired.
    # Keep it simple: do not rescale by default; you can tweak externally if needed.

    # psm 6: Assume a single uniform block of text
    config = "--psm 6"
    text = pytesseract.image_to_string(img, lang=lang or "eng", config=config) or ""
    return _post_clean(text)


def _post_clean(s: str) -> str:
    """
    Light normalization for OCR output:
    - Normalize Windows newlines to '\n'
    - Trim trailing spaces on lines
    - Collapse 3+ blank lines to 2
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip()


def ocr_image_bytes(data: bytes, lang: Optional[str] = None, *, raise_on_error: bool = False) -> str:
    """
    Best-effort OCR for image bytes. Returns "" if OCR isn't available or fails.

    Args:
      data           : image bytes (PNG/JPEG/TIFF, etc.)
      lang           : language code(s) for Tesseract (default from OCR_LANG or "eng")
      raise_on_error : if True, raise the underlying exception on failure

    Returns:
      Extracted text (possibly empty).
    """
    provider = _provider()
    if provider != "tesseract":
        return ""

    try:
        l = (lang or os.getenv("OCR_LANG") or "eng").strip() or "eng"
        return _ocr_tesseract_image(data, l)
    except Exception as e:
        if raise_on_error:
            raise
        # Fail-quietly by default; callers can proceed without OCR text.
        return ""
